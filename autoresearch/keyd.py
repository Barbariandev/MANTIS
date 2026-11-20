# MIT License
# Copyright (c) 2024 MANTIS
"""loop-keyd: privilege-separated Unix-socket jail for miner inference keys.

``validator.py`` and ``SealedRunner`` never hold a miner's plaintext
``cpk_...`` (Chutes) or ``gm_live_...`` (SayGM) key in RAM.  Instead:

1. ``loop-keyd`` runs in its own isolated OS process (standalone daemon
   under an unprivileged ``loop-vault`` user, or an isolated ``python -I``
   subprocess spawned at startup), with core dumps disabled
   (``RLIMIT_CORE = 0``) and ``PR_SET_DUMPABLE(0)`` on Linux so non-root
   processes cannot ``ptrace`` or read ``/proc/<pid>/mem``.
2. Only ``loop-keyd`` holds the rotating per-round ``loop_seal`` X25519
   seeds (``sealbox.Keyring``).  It never holds the validator's ``sr25519``
   hotkey or ``owner_hpke_x25519.pem`` (``W_owner``).
3. ``loop-keyd`` exposes a tiny local ``AF_UNIX`` (``0600``) JSON-line
   protocol supporting only:
     * ``rotate``  — rotate round keys & shred expired round seeds,
                     returning public keys only for UID 0 to sign;
     * ``vet``     — decrypt ``key_sealed`` in-process, enforce pre-flight
                     quarantine (refuse OpenRouter keys, refuse Chutes
                     admin keys, refuse SayGM keys > $25), discard key,
                     and return ``{ok, provider}`` only;
     * ``models``  — list normalized model catalog for a sealed submission;
     * ``chat``    — enforce ``budget_credits`` independently inside
                     ``keyd`` and dispatch strictly to the hardcoded
                     Chutes / SayGM completion endpoints;
     * ``release`` — clear in-memory spend tracking when a run ends.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import resource
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "autoresearch"

from . import sealbox
from .providers import (SAYGM_BASE_URL, SAYGM_MAX_KEY_BALANCE_USD,
                        InferenceClient, KeyQuarantineError, ProviderError,
                        quarantine_key)
from .submission import open_submission_key, parse_submission, redact

logger = logging.getLogger(__name__)

MAX_SOCKET_LINE = 2_000_000
_PR_SET_DUMPABLE = 4


def harden_process() -> Dict[str, bool]:
    """Disable core dumps and mark the process non-dumpable on Linux."""
    status = {"rlimit_core_zero": False, "non_dumpable": False}
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        status["rlimit_core_zero"] = True
    except (ValueError, OSError):
        pass
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) == 0:
                status["non_dumpable"] = True
        except OSError:
            pass
    return status


class _ThreadedUnixStreamServer(socketserver.ThreadingMixIn,
                                socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class KeydState:
    """In-memory state of the ``loop-keyd`` daemon."""

    def __init__(self, keyring: sealbox.Keyring,
                 saygm_base_url: str = SAYGM_BASE_URL,
                 max_saygm_balance_usd: float = SAYGM_MAX_KEY_BALANCE_USD,
                 http: Any = None):
        self.keyring = keyring
        self.saygm_base_url = saygm_base_url
        self.max_saygm_balance_usd = max_saygm_balance_usd
        self.http = http
        self._lock = threading.Lock()
        self._spent: Dict[str, float] = {}     # run_id -> cumulative USD

    def _open_key(self, hotkey: str, sub_raw: Any) -> str:
        sub = parse_submission(hotkey, sub_raw)
        key = open_submission_key(self.keyring, hotkey, sub)
        if not key:
            raise ValueError("submission carries no key_sealed")
        return key

    def handle(self, req: Any) -> Dict[str, Any]:
        if not isinstance(req, dict):
            return {"ok": False, "kind": "bad_request",
                    "error": "request must be a JSON object"}
        op = req.get("op")
        try:
            if op == "ping":
                return {"ok": True, "keys": self.keyring.public_keys()}
            if op == "rotate":
                now = req.get("now")
                state = self.keyring.rotate(
                    float(now) if isinstance(now, (int, float)) else None)
                return {"ok": True, "state": state}
            if op == "vet":
                return self._op_vet(req)
            if op == "models":
                return self._op_models(req)
            if op == "chat":
                return self._op_chat(req)
            if op == "release":
                rid = str(req.get("run_id") or "")
                with self._lock:
                    self._spent.pop(rid, None)
                return {"ok": True}
            return {"ok": False, "kind": "bad_request",
                    "error": f"unknown op {op!r}"}
        except KeyQuarantineError as e:
            return {"ok": False, "kind": "quarantine",
                    "error": redact(str(e))[:300]}
        except ProviderError as e:
            return {"ok": False, "kind": "provider",
                    "error": redact(str(e))[:300]}
        except ValueError as e:
            return {"ok": False, "kind": "seal",
                    "error": redact(str(e))[:300]}
        except Exception as e:  # noqa: BLE001
            logger.warning("keyd op %s failed: %s", op, type(e).__name__)
            return {"ok": False, "kind": "internal",
                    "error": type(e).__name__}

    def _op_vet(self, req: dict) -> dict:
        hotkey = str(req.get("hotkey") or "")
        key = self._open_key(hotkey, req.get("sub"))
        try:
            if req.get("skip_remote"):
                from .providers import detect_provider
                return {"ok": True, "provider": detect_provider(key)}
            res = quarantine_key(
                key,
                saygm_base_url=self.saygm_base_url,
                max_saygm_balance_usd=self.max_saygm_balance_usd,
                http=self.http,
            )
            return {"ok": True, "provider": res["provider"]}
        finally:
            del key

    def _op_models(self, req: dict) -> dict:
        hotkey = str(req.get("hotkey") or "")
        key = self._open_key(hotkey, req.get("sub"))
        try:
            client = InferenceClient(
                api_key=key,
                saygm_base_url=self.saygm_base_url,
                http=self.http,
            )
            return {"ok": True, "models": client.list_models()}
        finally:
            del key

    def _op_chat(self, req: dict) -> dict:
        run_id = str(req.get("run_id") or "")
        hotkey = str(req.get("hotkey") or "")
        budget = float(req.get("budget_credits") or 0.0)
        prior_floor = float(req.get("prior_spent") or 0.0)
        body = req.get("body")
        if not run_id or budget <= 0 or not isinstance(body, dict):
            raise ValueError("run_id, positive budget_credits and body "
                             "required")
        with self._lock:
            current = max(self._spent.get(run_id, 0.0), prior_floor)
            self._spent[run_id] = current
            if current >= budget:
                return {"ok": False, "kind": "budget",
                        "error": "run budget exhausted inside keyd"}

        key = self._open_key(hotkey, req.get("sub"))
        try:
            client = InferenceClient(
                api_key=key,
                saygm_base_url=self.saygm_base_url,
                http=self.http,
            )
            resp, cost = client.chat(body)
            prov = client.last_provider
        finally:
            del key

        with self._lock:
            self._spent[run_id] = self._spent.get(run_id, 0.0) + float(cost)
        return {"ok": True, "response": resp, "cost": float(cost),
                "provider": prov}


class KeydServer:
    """In-process or standalone Unix-socket server wrapping ``KeydState``."""

    def __init__(self, socket_path: str, state: KeydState,
                 socket_mode: int = 0o600):
        self.socket_path = os.path.abspath(socket_path)
        self.state = state
        sock_dir = os.path.dirname(self.socket_path)
        os.makedirs(sock_dir, mode=0o700, exist_ok=True)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        outer = self

        class _Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                while True:
                    line = self.rfile.readline(MAX_SOCKET_LINE + 1)
                    if not line:
                        return
                    if len(line) > MAX_SOCKET_LINE:
                        resp = {"ok": False, "kind": "bad_request",
                                "error": "request too large"}
                    else:
                        try:
                            req = json.loads(line.decode("utf-8"))
                        except ValueError:
                            resp = {"ok": False, "kind": "bad_request",
                                    "error": "invalid JSON"}
                        else:
                            resp = outer.state.handle(req)
                    out = (json.dumps(resp, separators=(",", ":"))
                           + "\n").encode("utf-8")
                    try:
                        self.wfile.write(out)
                        self.wfile.flush()
                    except OSError:
                        return

        old_umask = os.umask(0o177)
        try:
            self._srv = _ThreadedUnixStreamServer(self.socket_path, _Handler)
        finally:
            os.umask(old_umask)
        try:
            os.chmod(self.socket_path, int(socket_mode))
        except OSError:
            pass
        self._thread: Optional[threading.Thread] = None

    def start_background(self) -> "KeydServer":
        self._thread = threading.Thread(
            target=self._srv.serve_forever, daemon=True, name="loop-keyd")
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self._srv.serve_forever()

    def close(self) -> None:
        try:
            self._srv.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._srv.server_close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except OSError:
            pass


# ------------------------------------------------------------ client side

class KeydClient:
    """Validator-side client to ``loop-keyd`` over ``AF_UNIX``.  Holds zero
    private seeds and receives zero plaintext inference keys."""

    def __init__(self, socket_path: str, timeout: float = 210.0):
        self.socket_path = os.path.abspath(socket_path)
        self.timeout = timeout

    def _rpc(self, payload: dict,
             timeout: Optional[float] = None) -> Dict[str, Any]:
        t = self.timeout if timeout is None else timeout
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(t)
            s.connect(self.socket_path)
            wire = (json.dumps(payload, separators=(",", ":"))
                    + "\n").encode("utf-8")
            s.sendall(wire)
            buf = bytearray()
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                if b"\n" in chunk:
                    break
        line = bytes(buf).split(b"\n", 1)[0]
        if not line:
            raise ProviderError("empty response from loop-keyd")
        return json.loads(line.decode("utf-8"))

    def ping(self, timeout: float = 5.0) -> Dict[str, str]:
        res = self._rpc({"op": "ping"}, timeout=timeout)
        if not res.get("ok"):
            raise ProviderError(res.get("error", "keyd ping failed"))
        return dict(res.get("keys") or {})

    def rotate(self, now: Optional[float] = None) -> Dict[str, Any]:
        res = self._rpc({"op": "rotate", "now": now}, timeout=10.0)
        if not res.get("ok"):
            raise ProviderError(res.get("error", "keyd rotate failed"))
        return dict(res["state"])

    def vet(self, hotkey: str, sub: dict,
            skip_remote: bool = False) -> str:
        """Validate & quarantine-check ``sub['key_sealed']`` inside ``keyd``.
        Returns the detected provider name (``'chutes'`` or ``'saygm'``), or
        raises ``KeyQuarantineError`` / ``ValueError`` / ``ProviderError``."""
        res = self._rpc(
            {"op": "vet", "hotkey": hotkey, "sub": sub,
             "skip_remote": bool(skip_remote)},
            timeout=30.0,
        )
        if res.get("ok"):
            return str(res.get("provider") or "inference")
        kind = res.get("kind")
        msg = str(res.get("error") or "sealed key rejected")
        if kind == "quarantine":
            raise KeyQuarantineError(msg)
        if kind == "seal":
            raise ValueError(msg)
        raise ProviderError(msg)

    def release_run(self, run_id: str) -> None:
        try:
            self._rpc({"op": "release", "run_id": run_id}, timeout=5.0)
        except Exception:  # noqa: BLE001
            pass

    def proxy_for_run(self, run_id: str, hotkey: str, sub: dict,
                      budget_credits: float,
                      prior_spent: float = 0.0) -> "KeydInferenceProxy":
        return KeydInferenceProxy(
            self, run_id=run_id, hotkey=hotkey, sub=dict(sub),
            budget_credits=float(budget_credits),
            prior_spent=float(prior_spent))


class KeydInferenceProxy:
    """Drop-in ``InferenceClient`` substitute held by ``Broker`` during a
    BYOK run: carries only the sealed submission dict, never the plaintext
    inference key."""

    def __init__(self, client: KeydClient, run_id: str, hotkey: str,
                 sub: dict, budget_credits: float,
                 prior_spent: float = 0.0):
        self.client = client
        self.run_id = run_id
        self.hotkey = hotkey
        self.sub = sub
        self.budget_credits = budget_credits
        self.prior_spent = prior_spent
        self.last_provider = "inference"

    def available(self) -> bool:
        return bool(self.sub.get("key_sealed"))

    def list_models(self) -> List[dict]:
        res = self.client._rpc(
            {"op": "models", "hotkey": self.hotkey, "sub": self.sub},
            timeout=45.0,
        )
        if not res.get("ok"):
            raise ProviderError(str(res.get("error") or "models failed"))
        return list(res.get("models") or [])

    def chat(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        res = self.client._rpc({
            "op": "chat",
            "run_id": self.run_id,
            "hotkey": self.hotkey,
            "sub": self.sub,
            "budget_credits": self.budget_credits,
            "prior_spent": self.prior_spent,
            "body": body,
        })
        if not res.get("ok"):
            raise ProviderError(str(res.get("error") or "chat failed"))
        self.last_provider = str(res.get("provider") or "inference")
        return dict(res["response"]), float(res["cost"])

    def release(self) -> None:
        self.client.release_run(self.run_id)


# ------------------------------------------------ isolated child launcher

class KeydProcess:
    """Manages an isolated ``python -I -m autoresearch.keyd`` subprocess
    when no external ``loop-keyd`` service is already listening on
    ``socket_path``."""

    def __init__(self, proc: Optional[subprocess.Popen],
                 client: KeydClient, temp_dir: Optional[str] = None):
        self.proc = proc
        self.client = client
        self._temp_dir = temp_dir

    @classmethod
    def connect_or_spawn(cls, keyring_dir: str,
                         socket_path: str = "",
                         retention_days: int = 5) -> "KeydProcess":
        temp_dir = None
        if socket_path:
            client = KeydClient(socket_path)
            try:
                client.ping(timeout=2.0)
                logger.info("loop-keyd: attached to running daemon at %s",
                            socket_path)
                return cls(None, client, None)
            except Exception:  # noqa: BLE001
                pass
            sock_dir = os.path.dirname(os.path.abspath(socket_path))
            existed = os.path.isdir(sock_dir)
            try:
                os.makedirs(sock_dir, mode=0o700, exist_ok=True)
                if not existed:
                    os.chmod(sock_dir, 0o700)
                if not os.access(sock_dir, os.W_OK):
                    socket_path = ""
            except OSError:
                socket_path = ""

        if not socket_path:
            # AF_UNIX path length limit is ~104 bytes on Linux/macOS
            temp_dir = tempfile.mkdtemp(prefix="loop-keyd-", dir="/tmp")
            os.chmod(temp_dir, 0o700)
            socket_path = os.path.join(temp_dir, "keyd.sock")

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
        # Clean environment: no inherited secrets, no URL/cap overrides.
        env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONPATH": repo_root,
        }
        cmd = [
            sys.executable, "-I", "-B", os.path.abspath(__file__),
            "--socket", socket_path,
            "--keyring-dir", keyring_dir,
            "--retention-days", str(int(retention_days)),
        ]
        proc = subprocess.Popen(
            cmd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        client = KeydClient(socket_path)
        deadline = time.monotonic() + 10.0
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                err = (proc.stderr.read().decode("utf-8", "replace")
                       if proc.stderr else "")
                raise RuntimeError(
                    f"loop-keyd exited immediately ({proc.returncode}): "
                    f"{redact(err)[:300]}")
            try:
                client.ping(timeout=1.0)
                return cls(proc, client, temp_dir)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.1)
        proc.kill()
        raise RuntimeError(f"loop-keyd did not answer on {socket_path}: "
                           f"{last_err}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    self.proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    harden = harden_process()
    ap = argparse.ArgumentParser(
        description="loop-keyd: privilege-separated inference key jail")
    ap.add_argument("--socket", required=True,
                    help="AF_UNIX socket path")
    ap.add_argument("--socket-mode", default="0600",
                    help="octal mode for AF_UNIX socket (e.g. 0600 or 0660)")
    ap.add_argument("--keyring-dir", required=True,
                    help="directory holding rotating seal-YYYY-MM-DD.hex keys")
    ap.add_argument("--retention-days", type=int, default=5)
    args = ap.parse_args(argv)

    mode_int = int(str(args.socket_mode), 8)
    if (mode_int & 0o007) != 0:
        raise ValueError("socket-mode must never grant world access")
    keyring = sealbox.Keyring(
        args.keyring_dir,
        retention_days=args.retention_days,
    )
    keyring.rotate()
    state = KeydState(keyring)
    srv = KeydServer(args.socket, state, socket_mode=mode_int)
    logger.info("loop-keyd listening on %s (hardening=%s)",
                args.socket, harden)

    def _stop(_sig, _frame):
        threading.Thread(target=srv.close, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        srv.serve_forever()
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] loop-keyd - %(message)s",
    )
    sys.exit(main())
