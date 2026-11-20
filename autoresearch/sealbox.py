# MIT License
# Copyright (c) 2024 MANTIS
"""Seal a small secret to an X25519 public key, with rotating round keys.

The miner's Chutes (``cpk_...``) or SayGM (``gm_live_...``) inference key
rides inside the run submission.  TLS already covers the wire, but TLS
ends wherever it is terminated — a proxy, a load balancer, a request
log.  A sealed box ends only inside the key-jail daemon (``keyd.py``):
ephemeral X25519 -> HKDF-SHA256 -> ChaCha20-Poly1305, with the
submission's identity (``loop-submission:{hotkey}:{sha256}``) as
associated data, so the box opens for exactly one submission.  Captured
and replayed under another hotkey or sha256, it does not open; and
because each (hotkey, sha256) runs at most once, it cannot be replayed
under its own either.

    blob = seal(seal_pubkey_hex, api_key, f"loop-submission:{hotkey}:{sha}")
    ...
    api_key = open_sealed(seal_seed_hex, blob, f"loop-submission:{hotkey}:{sha}")

Forward secrecy & W_owner separation
------------------------------------
Outer MANTIS v2 trade payloads wrap to ``OWNER_HPKE_PUBLIC_KEY_HEX``
(``W_owner``).  LOOP inference keys instead seal to a dedicated rotating
``Keyring`` (one X25519 keypair per 24h UTC round, published signed by
UID 0 as ``seal_keys.json`` on ``LOOP_PUBLIC_URL``).  Expired round
private keys are overwritten and unlinked once they pass the resume
window, so compromising a host in the future can never decrypt past
submissions.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

INFO = b"mantis-loop-sealbox-v1"
RECEIPT_INFO = b"mantis-loop-receipt-hmac-v1"
MAX_PLAINTEXT = 512
SEAL_KEYS_VERSION = 1
_ROUND_KEY_FILE = re.compile(r"^seal-(\d{4}-\d{2}-\d{2})\.hex$")


def _raw(key) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw,
                            serialization.PublicFormat.Raw)


def _derive(shared: bytes, eph_pub: bytes, recip_pub: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=INFO + eph_pub + recip_pub).derive(shared)


def keypair(seed_hex: str) -> tuple:
    """``(private_key, public_key_hex)`` from a 32-byte hex seed."""
    seed = bytes.fromhex(seed_hex)
    if len(seed) != 32:
        raise ValueError("the seed is 32 bytes of hex")
    priv = X25519PrivateKey.from_private_bytes(seed)
    return priv, _raw(priv.public_key()).hex()


def generate_seed() -> str:
    """Generate a fresh 32-byte X25519 private seed as 64 lowercase hex chars."""
    return os.urandom(32).hex()


def seed_from_file(path: str) -> str:
    """The hex seed of the X25519 private key stored at ``path``, as PEM
    (PKCS#8) or as 64 hex characters."""
    with open(path, "rb") as f:
        data = f.read()
    text = data.strip()
    if len(text) == 64:
        try:
            bytes.fromhex(text.decode("ascii"))
            return text.decode("ascii").lower()
        except (UnicodeDecodeError, ValueError):
            pass
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: not an unencrypted PEM private key "
                         f"({e})") from None
    if not isinstance(key, X25519PrivateKey):
        raise ValueError(f"{path} does not hold an X25519 private key")
    return key.private_bytes(serialization.Encoding.Raw,
                             serialization.PrivateFormat.Raw,
                             serialization.NoEncryption()).hex()


def receipt_secret(seed_hex: str) -> str:
    """The receipt HMAC key, derived from the seed: the one key file is
    all the auditor keeps, and the secret cannot drift across restarts."""
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=RECEIPT_INFO).derive(bytes.fromhex(seed_hex)).hex()


def seal(recipient_pubkey_hex: str, plaintext: str, context: str) -> str:
    """Encrypt ``plaintext`` so only the holder of the recipient's
    private key can read it, bound to ``context``.  Returns hex:
    ephemeral public key (32) || nonce (12) || ciphertext."""
    data = plaintext.encode("utf-8")
    if not 0 < len(data) <= MAX_PLAINTEXT:
        raise ValueError(f"plaintext: 1..{MAX_PLAINTEXT} bytes")
    recip = X25519PublicKey.from_public_bytes(
        bytes.fromhex(recipient_pubkey_hex))
    eph = X25519PrivateKey.generate()
    eph_pub = _raw(eph.public_key())
    key = _derive(eph.exchange(recip), eph_pub, _raw(recip))
    nonce = os.urandom(12)
    ct = ChaCha20Poly1305(key).encrypt(nonce, data,
                                       context.encode("utf-8"))
    return (eph_pub + nonce + ct).hex()


def open_sealed(seed_hex: str, blob_hex: str, context: str) -> str:
    """Decrypt a sealed box made for this seed's public key and this
    exact ``context``.  Raises ``ValueError`` on anything else."""
    try:
        blob = bytes.fromhex(blob_hex)
    except (TypeError, ValueError):
        raise ValueError("not a sealed box")
    if not 44 < len(blob) <= 44 + MAX_PLAINTEXT + 16:
        raise ValueError("not a sealed box")
    priv, _pub = keypair(seed_hex)
    eph_pub, nonce, ct = blob[:32], blob[32:44], blob[44:]
    eph = X25519PublicKey.from_public_bytes(eph_pub)
    key = _derive(priv.exchange(eph), eph_pub, _raw(priv.public_key()))
    try:
        data = ChaCha20Poly1305(key).decrypt(nonce, ct,
                                             context.encode("utf-8"))
    except Exception:  # noqa: BLE001  (InvalidTag and friends)
        raise ValueError("the sealed box does not open for this "
                         "submission")
    return data.decode("utf-8")


# ------------------------------------------------------- rotating keyring

def shred_file(path: str) -> bool:
    """Overwrite ``path`` in place with random bytes + zeros, fsync, and
    unlink so an expired round's X25519 seed does not linger on disk."""
    if not os.path.exists(path):
        return False
    try:
        size = max(os.path.getsize(path), 64)
        with open(path, "r+b") as f:
            for fill in (os.urandom(size), b"\x00" * size):
                f.seek(0)
                f.write(fill)
                f.flush()
                os.fsync(f.fileno())
    except OSError:
        pass
    try:
        os.unlink(path)
        return True
    except OSError:
        return False


def _utc_date_str(ts: Optional[float] = None, offset_days: int = 0) -> str:
    base = time.time() if ts is None else float(ts)
    dt = _dt.datetime.fromtimestamp(
        base, _dt.timezone.utc) + _dt.timedelta(days=offset_days)
    return dt.strftime("%Y-%m-%d")


class Keyring:
    """Per-round rotating X25519 seal keys, decoupled from ``W_owner``.

    Each UTC date ``YYYY-MM-DD`` gets its own 32-byte seed file
    ``seal-YYYY-MM-DD.hex`` (mode ``0600``) inside ``directory``
    (mode ``0700``).  ``rotate(now)`` ensures seeds exist for today's
    round and tomorrow's intake window, and shreds any seed older than
    ``retention_days`` (default 5 days: covers the 24h intake window +
    the 72h pause/resume window, then permanently destroys the key).
    """

    DEFAULT_RETENTION_DAYS = 5

    def __init__(self, directory: str,
                 retention_days: int = DEFAULT_RETENTION_DAYS,
                 fallback_seed: str = ""):
        self.directory = os.path.abspath(os.path.expanduser(directory))
        self.retention_days = max(2, int(retention_days))
        self.fallback_seed = (fallback_seed or "").strip()
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass

    def _path(self, round_id: str) -> str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", round_id):
            raise ValueError(f"invalid round_id {round_id!r}")
        return os.path.join(self.directory, f"seal-{round_id}.hex")

    def ensure_round(self, round_id: str) -> Tuple[str, str]:
        """Return ``(round_id, pubkey_hex)``, creating the seed atomically
        with mode ``0600`` if absent."""
        path = self._path(round_id)
        if os.path.exists(path):
            seed = seed_from_file(path)
            return round_id, keypair(seed)[1]
        seed = generate_seed()
        tmp = f"{path}.tmp.{os.getpid()}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, (seed + "\n").encode("ascii"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return round_id, keypair(seed)[1]

    def rotate(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Ensure today's and tomorrow's round keys exist, and shred
        round keys older than ``retention_days``.  Returns summary metadata
        (public keys only, never private seeds)."""
        today = _utc_date_str(now, 0)
        tomorrow = _utc_date_str(now, 1)
        _, today_pub = self.ensure_round(today)
        _, next_pub = self.ensure_round(tomorrow)
        cutoff = _utc_date_str(now, -self.retention_days)
        shredded: List[str] = []
        for name in sorted(os.listdir(self.directory)):
            m = _ROUND_KEY_FILE.match(name)
            if not m:
                continue
            rid = m.group(1)
            if rid < cutoff:
                if shred_file(os.path.join(self.directory, name)):
                    shredded.append(rid)
        return {
            "current": {"key_id": today, "pubkey": today_pub},
            "next": {"key_id": tomorrow, "pubkey": next_pub},
            "active": self.public_keys(),
            "shredded": shredded,
        }

    def public_keys(self) -> Dict[str, str]:
        """``{round_id: pubkey_hex}`` for every active seed on disk,
        newest round first."""
        out: Dict[str, str] = {}
        if not os.path.isdir(self.directory):
            return out
        for name in sorted(os.listdir(self.directory), reverse=True):
            m = _ROUND_KEY_FILE.match(name)
            if not m:
                continue
            rid = m.group(1)
            try:
                seed = seed_from_file(os.path.join(self.directory, name))
                out[rid] = keypair(seed)[1]
            except ValueError:
                continue
        return out

    def current_pubkey(self, now: Optional[float] = None) -> Tuple[str, str]:
        """``(key_id, pubkey_hex)`` miners should seal new submissions to:
        tomorrow's UTC round key (since submissions read today run at the
        next 0000 UTC boundary)."""
        state = self.rotate(now)
        nxt = state["next"]
        return nxt["key_id"], nxt["pubkey"]

    def open(self, blob_hex: str, context: str,
             key_id: Optional[str] = None) -> str:
        """Open ``blob_hex`` against ``key_id`` if named, else try active
        round keys newest-first (plus ``fallback_seed`` when configured)."""
        candidates: List[str] = []
        if key_id and re.match(r"^\d{4}-\d{2}-\d{2}$", str(key_id)):
            p = self._path(str(key_id))
            if os.path.exists(p):
                candidates.append(seed_from_file(p))
        for name in sorted(os.listdir(self.directory), reverse=True):
            m = _ROUND_KEY_FILE.match(name)
            if not m:
                continue
            p = os.path.join(self.directory, name)
            try:
                s = seed_from_file(p)
                if s not in candidates:
                    candidates.append(s)
            except ValueError:
                continue
        if self.fallback_seed and self.fallback_seed not in candidates:
            candidates.append(self.fallback_seed)
        if not candidates:
            raise ValueError("no active seal keys in keyring")
        last_err: Optional[Exception] = None
        for seed in candidates:
            try:
                return open_sealed(seed, blob_hex, context)
            except ValueError as e:
                last_err = e
                if str(e) == "not a sealed box":
                    raise
        raise last_err or ValueError("the sealed box does not open for this "
                                     "submission")


def seal_keys_message(auditor_hotkey: str, ts: float,
                      keys: Dict[str, str], recommended_id: str) -> str:
    return json.dumps({
        "loop_seal_keys": SEAL_KEYS_VERSION,
        "auditor_hotkey": auditor_hotkey,
        "ts": ts,
        "recommended_key_id": recommended_id,
        "keys": keys,
    }, separators=(",", ":"), sort_keys=True)


def export_seal_keys(keyring: Keyring, auditor_hotkey: str,
                     sign: Callable[[str], str],
                     now: Optional[float] = None) -> dict:
    """Export the active rotating public keys, signed sr25519 by UID 0,
    for publication at ``<LOOP_PUBLIC_URL>/seal_keys.json``."""
    state = keyring.rotate(now)
    rec_id = state["next"]["key_id"]
    keys = keyring.public_keys()
    ts = round(time.time() if now is None else float(now), 3)
    msg = seal_keys_message(auditor_hotkey, ts, keys, rec_id)
    return {
        "version": SEAL_KEYS_VERSION,
        "auditor_hotkey": auditor_hotkey,
        "ts": ts,
        "recommended_key_id": rec_id,
        "recommended_pubkey": keys[rec_id],
        "keys": keys,
        "signature": sign(msg),
    }


def verify_seal_keys(payload: Any, auditor_hotkey: str = "",
                     verify: Optional[Callable[[str, str, str], bool]] = None,
                     max_age_seconds: Optional[float] = None
                     ) -> Tuple[str, str]:
    """Verify a published ``seal_keys.json`` payload (sr25519 signature,
    hotkey match, and optional freshness window) and return
    ``(recommended_key_id, recommended_pubkey_hex)``."""
    if not isinstance(payload, dict) \
            or payload.get("version") != SEAL_KEYS_VERSION:
        raise ValueError("not a loop seal_keys manifest")
    hk = payload.get("auditor_hotkey")
    if not isinstance(hk, str) or not hk:
        raise ValueError("missing auditor_hotkey in seal_keys")
    if auditor_hotkey and hk != auditor_hotkey:
        raise ValueError("seal_keys signed by unexpected hotkey")
    ts = payload.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        raise ValueError("bad seal_keys timestamp")
    if max_age_seconds is not None:
        if abs(time.time() - float(ts)) > float(max_age_seconds):
            raise ValueError("stale seal_keys manifest")
    rec_id = payload.get("recommended_key_id")
    keys = payload.get("keys")
    if not isinstance(rec_id, str) or not isinstance(keys, dict) \
            or rec_id not in keys:
        raise ValueError("missing recommended key in seal_keys")
    for k, pub in keys.items():
        if not (isinstance(k, str) and isinstance(pub, str)
                and len(pub) == 64
                and all(c in "0123456789abcdef" for c in pub)):
            raise ValueError("malformed pubkey in seal_keys")
    if payload.get("recommended_pubkey") not in (None, keys[rec_id]):
        raise ValueError("recommended_pubkey mismatch in seal_keys")
    sig = payload.get("signature")
    if verify is None:
        from .gateway import _verify_sr25519
        verify = _verify_sr25519
    msg = seal_keys_message(hk, float(ts), keys, rec_id)
    if not isinstance(sig, str) or not verify(hk, msg, sig):
        raise ValueError("invalid sr25519 signature on seal_keys")
    return rec_id, keys[rec_id]
