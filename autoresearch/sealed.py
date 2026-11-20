# MIT License
# Copyright (c) 2024 MANTIS
"""Sealed runs: how a miner's pipeline is executed.

A miner's entry on LOOP is a *pipeline* — a Python program that
researches a trading model — plus the credits to run it.  The pipeline
is published, and the gateway runs it sealed:

* in a sandbox with **no network and no host filesystem**, as an
  unprivileged user (``Sandbox``), on gVisor's kernel rather than the
  host's where the host has it;
* on a fixed share of the machine — its own cores, a memory cap — and
  a wall clock the same for every run in a round (``slots``);
* with **one way out**: a line-delimited JSON channel to a ``Broker``
  on the gateway side, which makes the model calls, runs the
  experiments on the protocol's own harness and data, and **writes the
  trace itself**.

So nothing in a run is the miner's word.  The trace is the gateway's
log of what the published code did; the metrics are the harness's; the
dossier is assembled here, not submitted.  A miner cannot hand-pick a
model and dress it as research, because the only thing that can add a
candidate, score it or promote it is the code everyone can read.

What a pipeline can still do is know things it did not learn in the
run — constants in its source, hints in its prompts.  That is why the
source is public, the validator's judge reads it, and the promoted
model is also scored on bars the pipeline never saw (``holdout``).

Local use, exactly as the gateway does it:

    python -m autoresearch.sealed my_pipeline.py --args '{"iterations": 4}'
"""

from __future__ import annotations

import collections
import contextlib
import functools
import json
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import harness
from . import slots as run_slots
from .credits import CreditLedger
from .providers import ComputeJob, InferenceClient
from .schema import (Dossier, FeatureSpec, UsageReceipt, canonical_json,
                     events_digest, new_id, receipts_digest, sha256_hex)
from .submission import validate_inference_key_format

logger = logging.getLogger(__name__)

PROTOCOL = "mantis-loop/2"
STEPS = ("propose", "critique", "other")
VERDICTS = ("promote", "refine", "prune")

MAX_SOURCE_CHARS = 200_000      # a pipeline's source
MAX_ARGS_CHARS = 4_000          # its arguments, as JSON
MAX_LINE_CHARS = 1_000_000      # one request from the pipeline
MAX_CANDIDATES = 500
MAX_STDERR_CHARS = 8_000        # how much of its stderr is kept (in RAM only)
MAX_REQUESTS_PER_RUN = 25_000   # hard cap on total stdio RPCs per run segment
MAX_STATIC_READS = 16           # cap on op_data / op_vocabulary / op_args spam
MAX_CHECKPOINTS_PER_RUN = 2_000 # cap on SQLite checkpoint writes per segment
_RESERVED_EVENTS = 8            # room kept for the events that close a run
# A model call is text in, text out.  Anything that would let a model
# fetch something for the pipeline — web plugins, ":online" models,
# image or file URLs in a message — is a way out of the sandbox, and a
# way to read the bars that were held back.
_LLM_PARAMS = ("temperature", "top_p", "top_k", "max_tokens",
               "max_completion_tokens", "seed", "stop", "reasoning",
               "response_format", "frequency_penalty", "presence_penalty")
_LLM_ROLES = ("system", "user", "assistant")


class SubmitError(ValueError):
    """A run could not be opened; the message is for the miner."""


_BUSY = ("this hotkey already has a run in progress (a paused run holds "
         "the slot: resume or cancel it)")


# ----------------------------------------------------------------- sandbox

# Runs as the first process of fresh mount/pid/net/ipc/uts namespaces.
# Builds a root that holds nothing but /usr (read-only), the pipeline's
# directory (read-only) and a small tmpfs, pivots into it, and drops
# every privilege before Python starts.
_SANDBOX_SH = r'''
set -e
R="$1"; W="$2"; Z="$3"; shift 3
mount --make-rprivate /
mount -t tmpfs -o size="$Z",nosuid,nodev,mode=755 tmpfs "$R"
mkdir -p "$R/usr" "$R/work" "$R/tmp" "$R/proc" "$R/dev" "$R/old"
chmod 1777 "$R/tmp"
mount --bind /usr "$R/usr"; mount -o remount,bind,ro,nosuid,nodev "$R/usr"
for l in lib lib64 bin sbin; do
  if [ -L "/$l" ]; then ln -s "usr/$l" "$R/$l"; fi
done
mount --bind "$W" "$R/work"; mount -o remount,bind,ro,nosuid,nodev "$R/work"
mount -t proc -o nosuid,nodev,noexec proc "$R/proc"
for d in null urandom; do
  touch "$R/dev/$d"; mount --bind "/dev/$d" "$R/dev/$d"
done
cd "$R"; pivot_root . old; cd /; umount -l /old; rmdir /old
cd /work
exec setpriv "$@"
'''

# The first thing that runs inside: keep the channel to the broker for
# the SDK, point the pipeline's own stdout at stderr so a stray print()
# cannot speak on it, then run the pipeline.
_MAIN_PY = r'''
import os, runpy, sys
sys.path.insert(0, os.getcwd())
import loop_sdk
loop_sdk._attach(os.fdopen(os.dup(0), "r", encoding="utf-8"),
                 os.fdopen(os.dup(1), "w", encoding="utf-8"))
os.dup2(2, 1)
_null = os.open(os.devnull, os.O_RDONLY); os.dup2(_null, 0); os.close(_null)
try:
    runpy.run_path("pipeline.py", run_name="__main__")
except loop_sdk.BudgetExhausted:
    print("budget exhausted; ending the run", file=sys.stderr)
except loop_sdk.Paused:
    print("gateway is pausing this run; exiting", file=sys.stderr)
'''

_PATH = "/usr/bin:/usr/sbin:/bin:/sbin"
_ENV = {"PATH": _PATH, "HOME": "/tmp"}    # all a sealed pipeline is given
SEALED = ("gvisor", "namespaces")     # what may run somebody else's code
_RUNSC_STATE = "/run/loop-runsc"

_RLIMITS = (("RLIMIT_AS", 2 * 1024 ** 3), ("RLIMIT_FSIZE", 16 * 1024 ** 2),
            ("RLIMIT_NPROC", 256), ("RLIMIT_CORE", 0))


def _limits(nproc: bool = True) -> None:
    # RLIMIT_NPROC counts every sandbox's processes together (they all
    # run as nobody); a run with a cgroup has its own pids cap instead
    for name, cap in _RLIMITS:
        if nproc or name != "RLIMIT_NPROC":
            resource.setrlimit(getattr(resource, name), (cap, cap))


def _trusted_executable(path: str) -> bool:
    """True when ``path`` is an executable regular file and (when running
    as root) owned by root and not group- or world-writable."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        return False
    if os.geteuid() == 0 and (st.st_uid != 0 or (st.st_mode & 0o022) != 0):
        return False
    return True


def runsc_binary() -> str:
    """gVisor's runtime from hardcoded root-owned paths (never env-overridable)."""
    try:
        from config import LOOP_RUNSC_PATHS
    except ImportError:
        LOOP_RUNSC_PATHS = (
            "/usr/local/bin/runsc",
            "/usr/bin/runsc",
            "/opt/gvisor/runsc",
        )
    for candidate in LOOP_RUNSC_PATHS:
        if _trusted_executable(candidate):
            return candidate
    found = shutil.which("runsc") or ""
    return found if found and _trusted_executable(found) else ""


def _gvisor_spec(python: str, workdir: str, scratch: str,
                 envelope: Optional["run_slots.Envelope"]) -> dict:
    """The OCI spec of a gVisor sandbox: the same view as the namespace
    sandbox — /usr and the pipeline's directory read-only, a small
    tmpfs, nobody, no capabilities, no network."""
    spec: Dict[str, Any] = {
        "ociVersion": "1.0.2",
        "process": {
            "terminal": False,
            "user": {"uid": 65534, "gid": 65534},
            "args": [python, "-I", "-B", "_main.py"],
            "env": [f"{k}={v}" for k, v in _ENV.items()],
            "cwd": "/work",
            "capabilities": {k: [] for k in (
                "bounding", "effective", "inheritable", "permitted",
                "ambient")},
            "rlimits": [{"type": name, "hard": cap, "soft": cap}
                        for name, cap in _RLIMITS],
            "noNewPrivileges": True},
        "root": {"path": "rootfs", "readonly": True},
        "hostname": "loop",
        "mounts": [
            {"destination": "/proc", "type": "proc", "source": "proc"},
            {"destination": "/usr", "type": "bind", "source": "/usr",
             "options": ["rbind", "ro", "nosuid", "nodev"]},
            {"destination": "/work", "type": "bind", "source": workdir,
             "options": ["rbind", "ro", "nosuid", "nodev"]},
            {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs",
             "options": ["nosuid", "nodev", "mode=1777",
                         f"size={scratch}"]}],
        "linux": {"namespaces": [{"type": t} for t in (
            "pid", "network", "ipc", "uts", "mount")]}}
    if envelope is not None:
        # the limits are the run's cgroup's; repeated here so the
        # pipeline sees its own cores and memory, not the host's
        spec["linux"]["cgroupsPath"] = envelope.leaf_rel("sandbox")
        spec["linux"]["resources"] = {
            "cpu": {"cpus": envelope.cpuset},
            "memory": {"limit": envelope.shape.memory_mb << 20},
            "pids": {"limit": envelope.shape.pids}}
    return spec


class Sandbox:
    """Starts a pipeline with stdin/stdout as its only connection.

    ``isolation``, strongest first:

    * ``"gvisor"`` — gVisor (``runsc``): the pipeline's system calls go
      to a kernel of gVisor's own, in user space, so a kernel bug the
      pipeline finds is not the host's.  Needs root and runsc.
    * ``"namespaces"`` — Linux namespaces on the host's kernel
      (util-linux; root or unprivileged user namespaces).
    * ``"none"`` — a plain subprocess: fine for developing a pipeline
      of your own, never for running somebody else's.

    Both sealed kinds show a pipeline the same world: no network, a
    root holding only /usr and its own directory (both read-only) and a
    small tmpfs, uid nobody, no capabilities.  ``detect()`` picks the
    strongest the host has.
    """

    _probes: Dict[str, str] = {}     # kind -> "" (it works) or why not
    _probe_lock = threading.Lock()

    def __init__(self, isolation: Optional[str] = None):
        self.isolation = isolation or self.detect()

    @classmethod
    def detect(cls) -> str:
        return next((k for k in SEALED if cls.works(k)), "none")

    @classmethod
    def works(cls, kind: str) -> bool:
        """Whether this host can seal a run as ``kind`` (tried once)."""
        return not cls.why_not(kind)

    @classmethod
    def why_not(cls, kind: str) -> str:
        """"" when this host can seal a run as ``kind``, else why not."""
        with cls._probe_lock:
            if kind not in cls._probes:
                cls._probes[kind] = cls._self_test(kind)
            return cls._probes[kind]

    @classmethod
    def _self_test(cls, kind: str) -> str:
        if kind not in SEALED:
            return f"no sandbox called {kind!r}"
        if not sys.platform.startswith("linux"):
            return "not Linux"
        if kind == "gvisor" and not runsc_binary():
            return "runsc not found (install gVisor, or set LOOP_RUNSC)"
        if kind == "gvisor" and os.geteuid() != 0:
            return "gVisor runs pipelines only as root"
        missing = [t for t in ("unshare", "setpriv", "pivot_root", "mount")
                   if kind == "namespaces" and not shutil.which(t, path=_PATH)]
        if missing:
            return f"util-linux tools missing: {' '.join(missing)}"
        work = tempfile.mkdtemp(prefix="loop-probe-")
        proc = None
        try:
            with open(os.path.join(work, "_main.py"), "w") as f:
                f.write("import os\n"
                        "assert not os.path.exists('/etc')\n"
                        "print('sealed')\n")
            os.chmod(work, 0o755)
            with tempfile.TemporaryFile() as err:
                proc = cls(isolation=kind).start(work, stderr=err)
                out, _ = proc.communicate(timeout=60)
                if proc.returncode == 0 and out.strip() == "sealed":
                    return ""
                err.seek(0)
                said = err.read().decode("utf-8", "replace").split("\n")
                said = [line.strip() for line in said if line.strip()]
                return (f"a test sandbox failed (exit {proc.returncode})"
                        + (f": {said[-1][:300]}" if said else ""))
        except Exception as e:  # noqa: BLE001
            return f"a test sandbox failed: {e}"
        finally:
            if proc is not None:
                cls.stop(proc)
            shutil.rmtree(work, ignore_errors=True)

    def start(self, workdir: str, stderr: Any = subprocess.DEVNULL,
              envelope: Optional["run_slots.Envelope"] = None
              ) -> subprocess.Popen:
        """Run ``workdir/_main.py``.  ``workdir`` must be world-readable.
        ``envelope``: the run's cgroup, which the pipeline starts in —
        on the run's cores, under its caps."""
        python = os.path.realpath(sys.executable)
        scratch = f"{envelope.shape.scratch_mb if envelope else 64}m"
        common = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      stderr=stderr, text=True, encoding="utf-8",
                      errors="replace", start_new_session=True)
        if self.isolation == "gvisor":
            return self._start_gvisor(python, workdir, scratch, envelope,
                                      common)
        join = envelope.join("sandbox") if envelope else []
        common["preexec_fn"] = functools.partial(
            _limits, nproc=envelope is None)
        if self.isolation != "namespaces":
            return subprocess.Popen(
                join + [python, "-I", "-B", "_main.py"], cwd=workdir,
                env={"PATH": _PATH}, **common)
        root = tempfile.mkdtemp(prefix="loop-root-")
        drop = ["--inh-caps=-all", "--bounding-set=-all", "--no-new-privs"]
        ns = ["unshare", "--net", "--ipc", "--uts", "--pid", "--fork",
              "--mount", "--kill-child"]
        if os.geteuid() == 0:
            drop = ["--reuid=65534", "--regid=65534", "--clear-groups"] + drop
        else:
            ns += ["--user", "--map-root-user"]
        proc = subprocess.Popen(
            join + ns + ["sh", "-c", _SANDBOX_SH, "sh", root, workdir,
                         scratch] + drop + [python, "-I", "-B", "_main.py"],
            env=dict(_ENV), **common)
        proc.loop_root = root       # removed by whoever reaps the process
        return proc

    @staticmethod
    def _start_gvisor(python: str, workdir: str, scratch: str,
                      envelope: Optional["run_slots.Envelope"],
                      common: dict) -> subprocess.Popen:
        runsc = runsc_binary()
        if not runsc:
            raise RuntimeError("gVisor sandbox, but no runsc (install "
                               "gVisor or set LOOP_RUNSC)")
        bundle = tempfile.mkdtemp(prefix="loop-gvisor-")
        try:
            rootfs = os.path.join(bundle, "rootfs")
            for d in ("usr", "work", "tmp", "proc", "dev"):
                os.makedirs(os.path.join(rootfs, d))
            for link in ("lib", "lib64", "bin", "sbin"):
                if os.path.islink("/" + link):
                    os.symlink("usr/" + link, os.path.join(rootfs, link))
            with open(os.path.join(bundle, "config.json"), "w") as f:
                json.dump(_gvisor_spec(python, workdir, scratch, envelope),
                          f)
            cid = os.path.basename(bundle)
            proc = subprocess.Popen(
                [runsc, f"--root={_RUNSC_STATE}", "--network=none", "run",
                 "--bundle", bundle, cid], env={"PATH": _PATH}, **common)
        except Exception:
            shutil.rmtree(bundle, ignore_errors=True)
            raise
        proc.loop_gvisor = (runsc, cid)
        proc.loop_root = bundle
        return proc

    @staticmethod
    def stop(proc: subprocess.Popen) -> None:
        """Kill the pipeline and everything it started."""
        gvisor = getattr(proc, "loop_gvisor", None)

        def runsc(*args: str) -> None:
            try:
                subprocess.run([gvisor[0], f"--root={_RUNSC_STATE}", *args],
                               capture_output=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if gvisor:
            runsc("kill", "--all", gvisor[1], "KILL")
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        if gvisor:
            runsc("delete", "--force", gvisor[1])
        root = getattr(proc, "loop_root", None)
        if root:
            shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ broker

class _Refused(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _text(x: Any, limit: int) -> str:
    return (x if isinstance(x, str) else "")[:limit]


@dataclass
class _Candidate:
    cid: str
    spec: dict
    name: str
    rationale: str
    hyp_seq: int
    res_seq: Optional[int] = None
    metrics: Optional[dict] = None


@dataclass
class SealedSettings:
    secret: str
    holdout_bars: int = 480             # held back from every pipeline
    max_run_seconds: int = 43200          # 12h: the real bottleneck
    max_experiment_seconds: int = 120
    pipeline_rate_per_sec: float = 0.0001
    llm_fee: float = 0.0
    max_concurrent_runs: int = 8
    allow_unsealed: bool = False        # run without a sandbox (dev only)
    # how long a pausing pipeline gets to checkpoint and exit
    pause_grace_seconds: float = 20.0
    # how long a paused run stays resumable before it fails
    resume_window_seconds: float = 259200.0     # 72h

MAX_CHECKPOINT_CHARS = 32_000   # a pipeline's saved state


class Broker:
    """The gateway side of one run: answers the pipeline's requests and
    writes down what happened.  Every event in a run's trace is emitted
    here, from what the gateway itself did or checked."""

    def __init__(self, ledger: CreditLedger, settings: SealedSettings,
                 inference: InferenceClient,
                 compute: Callable[[str, str, int], ComputeJob],
                 run: dict, dataset: dict):
        self.ledger, self.st = ledger, settings
        self.orc, self.compute = inference, compute
        self.run_id, self.hotkey = run["run_id"], run["hotkey"]
        self.run = run
        self.ds = dataset
        # what the pipeline may see: every series cut at the same moment
        self.research = harness.research_window(dataset,
                                                settings.holdout_bars)
        self._models: Optional[List[dict]] = None
        self.cands: Dict[str, _Candidate] = {}
        self.calls: Dict[str, str] = {}         # llm receipt id -> step
        self._sourced: set = set()   # propose receipts already attributed
        self.promoted: Optional[str] = None
        self.prose = {"narrative": "", "hyperparameter_rationale": ""}
        self._seq = 0
        self._requests = 0
        self._static_reads = 0
        self._checkpoints = 0
        self.flood_aborted: Optional[str] = None
        # set while the gateway pauses the run: every request but
        # checkpoint is then answered "paused" (loop_sdk raises Paused)
        self.pausing = threading.Event()
        self.deadline: Optional[float] = None   # monotonic, while running

    def rebuild(self) -> None:
        """Reload this run's state from its own record — events and
        receipts the ledger already holds — so a resumed run continues
        the same tree: candidate ids stay valid, sequence numbers
        continue, the promoted model stands."""
        events = self.ledger.events_for_run(self.run_id)
        for e in events:
            self._seq = max(self._seq, int(e["seq"]) + 1)
            kind, data = e["kind"], e.get("data") or {}
            if kind == "hypothesis.proposed":
                feature = dict(data.get("feature") or {})
                spec = {"feature": {k: feature[k] for k in
                                    ("transform", "lookback_hours",
                                     "series") if k in feature},
                        "hyperparams": dict(data.get("hyperparams") or {})}
                cid = data.get("candidate_id")
                if isinstance(cid, str):
                    self.cands[cid] = _Candidate(
                        cid, spec, feature.get("name", ""),
                        data.get("rationale", ""), int(e["seq"]))
                rid = data.get("receipt_id")
                if isinstance(rid, str):
                    self._sourced.add(rid)
            elif kind == "experiment.result":
                cand = self.cands.get(data.get("candidate_id"))
                if cand is not None:
                    cand.metrics = data.get("metrics")
                    cand.res_seq = int(e["seq"])
            elif kind == "model.promoted":
                cid = data.get("candidate_id")
                if isinstance(cid, str) and cid in self.cands:
                    self.promoted = cid
        for r in self.ledger.receipts(run_id=self.run_id):
            if r.get("kind") != "llm":
                continue
            meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            attribution = meta.get("attribution") or {}
            step = attribution.get("step")
            if step in STEPS:
                self.calls[r["receipt_id"]] = step

    # ---------------------------------------------------------- recording

    def emit(self, kind: str, data: dict,
             parent_seq: Optional[int] = None) -> int:
        seq = self._seq
        stored = self.ledger.append_events(self.run_id, [{
            "seq": seq, "kind": kind, "ts": time.time(),
            "parent_seq": parent_seq, "data": data}])
        if stored != 1:
            raise RuntimeError(f"could not record {kind} #{seq}")
        self._seq += 1
        return seq

    def _room(self) -> None:
        if self._seq >= self.ledger.MAX_EVENTS_PER_RUN - _RESERVED_EVENTS:
            raise _Refused("rejected", "this run's trace is full")

    def _spendable(self) -> None:
        if not self.ledger.can_spend(self.run_id):
            raise _Refused("budget", "the run's budget is spent")

    def receipt(self, kind: str, provider: str, item: str, credits: float,
                request_hash: str, meta: dict, force: bool = False
                ) -> UsageReceipt:
        # Quantised to microcredits before signing: that is what the
        # ledger stores, and the signature must survive the round trip.
        rec = UsageReceipt(
            receipt_id=new_id("rcpt"), hotkey=self.hotkey,
            run_id=self.run_id, kind=kind, provider=provider, item=item,
            credits=round(float(credits) * 1e6) / 1e6, ts=time.time(),
            request_hash=request_hash, meta=meta).sign(self.st.secret)
        self.ledger.debit(rec, force=force)
        return rec

    def _label(self, step: str, cid: str = "") -> dict:
        return {"protocol": PROTOCOL, "step": step, "candidate_id": cid}

    def _cand(self, cid: Any, evaluated: bool = False) -> _Candidate:
        cand = self.cands.get(cid) if isinstance(cid, str) else None
        if cand is None:
            raise _Refused("rejected", "no such candidate")
        if evaluated and cand.metrics is None:
            raise _Refused("rejected", "run its experiment first")
        return cand

    def _source(self, receipt_id: Any, step: str) -> Optional[str]:
        if receipt_id is None:
            return None
        if self.calls.get(receipt_id) != step:
            raise _Refused("rejected", f"source is not a {step!r} model "
                                       "call of this run")
        # one proposing exchange explains one candidate: a single
        # innocuous call cannot front a parade of carried-in specs,
        # and the promoted candidate's published exchange is the call
        # that actually proposed it
        if step == "propose":
            if receipt_id in self._sourced:
                raise _Refused("rejected", "this model call already "
                               "sourced a candidate; propose from a "
                               "fresh call")
            self._sourced.add(receipt_id)
        return receipt_id

    # ---------------------------------------------------------- requests

    def handle(self, msg: Any) -> dict:
        """One request from the pipeline -> one reply."""
        self._requests += 1
        if self._requests > MAX_REQUESTS_PER_RUN:
            self.flood_aborted = (
                f"pipeline exceeded {MAX_REQUESTS_PER_RUN} broker requests")
            return {"id": None, "ok": False, "kind": "rejected",
                    "error": self.flood_aborted}
        mid = msg.get("id") if isinstance(msg, dict) else None
        op = msg.get("op") if isinstance(msg, dict) else None
        if self.pausing.is_set() and op != "checkpoint":
            return {"id": mid, "ok": False, "kind": "paused",
                    "error": "the gateway is pausing this run: "
                             "checkpoint your state and exit; the run "
                             "resumes later"}
        fn = getattr(self, f"op_{op}", None) if isinstance(op, str) else None
        if fn is None:
            return {"id": mid, "ok": False, "kind": "rejected",
                    "error": "unknown request"}
        try:
            return {"id": mid, "ok": True, "result": fn(msg)}
        except _Refused as e:
            return {"id": mid, "ok": False, "kind": e.kind, "error": str(e)}

    def _guard_static_read(self) -> None:
        self._static_reads += 1
        if self._static_reads > MAX_STATIC_READS:
            raise _Refused("rejected", "static read called too many times; "
                           "cache loop.data() / loop.vocabulary() locally")

    def op_args(self, _msg: dict) -> dict:
        self._guard_static_read()
        return self.run["args"]

    def op_data(self, _msg: dict) -> dict:
        self._guard_static_read()
        return self.research

    def op_vocabulary(self, _msg: dict) -> dict:
        self._guard_static_read()
        return harness.vocabulary(self.research)

    def op_models(self, _msg: dict) -> List[dict]:
        """Every model the gateway can call, with its pricing — so a
        pipeline can choose (and budget) its own models at run time."""
        if self._models is None:
            try:
                raw = self.orc.list_models()
            except Exception as e:  # noqa: BLE001
                raise _Refused("provider", str(e)[:300])
            models = []
            for m in raw:
                if not isinstance(m, dict) or not isinstance(m.get("id"),
                                                             str):
                    continue
                pricing = m.get("pricing") or {}
                models.append({
                    "id": m["id"],
                    "context_length": m.get("context_length"),
                    "prompt_usd_per_token": pricing.get("prompt"),
                    "completion_usd_per_token": pricing.get("completion")})
            self._models = sorted(models, key=lambda m: m["id"])
        return self._models

    def op_budget(self, _msg: dict) -> dict:
        info = self.ledger.run_info(self.run_id)
        out = {"budget": info.budget_credits, "spent": info.spent_credits,
               "remaining": max(info.budget_credits - info.spent_credits,
                                0.0)}
        if self.deadline is not None:
            out["seconds_left"] = round(
                max(self.deadline - time.monotonic(), 0.0), 1)
        return out

    def op_llm(self, msg: dict) -> dict:
        step, cid = msg.get("step"), msg.get("candidate") or ""
        params = msg.get("params") or {}
        if step not in STEPS:
            raise _Refused("rejected", f"step must be one of "
                                       f"{', '.join(STEPS)}")
        if cid:
            self._cand(cid)
        model, messages = msg.get("model"), msg.get("messages")
        if (not isinstance(model, str) or not model or ":online" in model
                or not isinstance(params, dict)
                or any(k not in _LLM_PARAMS for k in params)):
            raise _Refused("rejected", "llm(model, messages, **params): "
                           "an offline model, and params among "
                           + ", ".join(_LLM_PARAMS))
        if (not isinstance(messages, list) or not messages or not all(
                isinstance(m, dict) and set(m) == {"role", "content"}
                and m["role"] in _LLM_ROLES
                and isinstance(m["content"], str) for m in messages)):
            raise _Refused("rejected", "messages: a list of {role, "
                           "content} with text content")
        self._room()
        self._spendable()
        body = {**params, "model": msg["model"], "messages": msg["messages"]}
        request = canonical_json(body)
        try:
            resp, cost = self.orc.chat(body)
        except Exception as e:  # noqa: BLE001
            raise _Refused("provider", str(e)[:300])
        # The whole exchange is published: what a call was for is then a
        # matter of record, not of the label the pipeline gave it.
        self.ledger.put_blob(request)
        provider_name = getattr(self.orc, "last_provider", "") or "inference"
        rec = self.receipt(
            "llm", provider_name, msg["model"],
            cost * (1.0 + self.st.llm_fee), sha256_hex(request),
            {"usage": resp.get("usage", {}), "or_id": resp.get("id", ""),
             "attribution": self._label(step, cid),
             "response_sha256": self.ledger.put_blob(canonical_json(resp))})
        self.calls[rec.receipt_id] = step
        self.emit("llm.call", {"step": step, "candidate_id": cid,
                               "model": msg["model"],
                               "receipt_id": rec.receipt_id,
                               "credits": rec.credits},
                  self.cands[cid].hyp_seq if cid else None)
        try:
            choice = resp["choices"][0]
            text = choice["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            choice, text = {}, None
        return {"text": text if isinstance(text, str) else None,
                "receipt_id": rec.receipt_id, "credits": rec.credits,
                "model": msg["model"],
                "finish_reason": choice.get("finish_reason")}

    def op_propose(self, msg: dict) -> str:
        if len(self.cands) >= MAX_CANDIDATES:
            raise _Refused("rejected", "too many candidates in one run")
        self._room()
        spec, problems = harness.clean_spec(
            msg.get("spec"), list(self.ds["series"]))
        if problems:
            raise _Refused("rejected", "; ".join(problems))
        # the series is resolved here so the record never depends on a
        # default that could drift
        spec["feature"].setdefault("series", self.ds["target"])
        refines = msg.get("refines")
        parent = self._cand(refines) if refines is not None else None
        source = self._source(msg.get("source"), "propose")
        cid = new_id("cand")
        name = _text(msg.get("name"), 64) or (
            f"{spec['feature']['transform']}_"
            f"{spec['feature']['lookback_hours']}h")
        rationale = _text(msg.get("rationale"), 600)
        seq = self.emit("hypothesis.proposed", {
            "candidate_id": cid,
            "feature": {**spec["feature"], "name": name,
                        "rationale": rationale},
            "hyperparams": spec["hyperparams"], "rationale": rationale,
            "refines": refines, "receipt_id": source},
            parent.hyp_seq if parent else None)
        self.cands[cid] = _Candidate(cid, spec, name, rationale, seq)
        return cid

    def op_experiment(self, msg: dict) -> dict:
        cand = self._cand(msg.get("candidate"))
        if cand.metrics is not None:
            return cand.metrics
        self._room()
        self._spendable()
        stdin = json.dumps({"op": "evaluate", "dataset": self.research,
                            "spec": cand.spec})
        job = self.compute(harness.HARNESS_SOURCE, stdin,
                           self.st.max_experiment_seconds)
        line = (job.output_tail or "").strip().split("\n", 1)[0].strip()
        try:
            metrics = json.loads(line) if job.status == "done" else None
        except ValueError:
            metrics = None
        if not isinstance(metrics, dict):
            metrics = {"error": f"experiment {job.status}"}
        rec = self.receipt(
            "compute", job.provider, job.sku, job.cost_credits,
            sha256_hex(stdin),
            {"seconds": job.seconds, "status": job.status,
             "attribution": self._label("experiment", cand.cid),
             "code_sha256": harness.HARNESS_SHA256,
             "dataset_sha256": self.run["dataset_sha256"],
             "result_sha256": sha256_hex(line)})
        cand.metrics = metrics
        cand.res_seq = self.emit("experiment.result", {
            "candidate_id": cand.cid, "metrics": metrics,
            "receipt_id": rec.receipt_id}, cand.hyp_seq)
        return metrics

    def op_critique(self, msg: dict) -> None:
        cand = self._cand(msg.get("candidate"), evaluated=True)
        verdict = msg.get("verdict")
        if verdict not in VERDICTS:
            raise _Refused("rejected", "verdict must be one of "
                                       + ", ".join(VERDICTS))
        self._room()
        self.emit("critique", {
            "candidate_id": cand.cid, "verdict": verdict,
            "notes": _text(msg.get("notes"), 600),
            "receipt_id": self._source(msg.get("source"), "critique")},
            cand.res_seq)

    def op_promote(self, msg: dict) -> None:
        cand = self._cand(msg.get("candidate"), evaluated=True)
        if cand.metrics.get("error"):
            raise _Refused("rejected", "its experiment failed")
        self._room()
        self.emit("model.promoted", {"candidate_id": cand.cid,
                                     "metrics": cand.metrics}, cand.res_seq)
        self.promoted = cand.cid

    def op_prune(self, msg: dict) -> None:
        cand = self._cand(msg.get("candidate"))
        self._room()
        self.emit("branch.pruned", {"candidate_id": cand.cid,
                                    "reason": _text(msg.get("reason"), 300)},
                  cand.res_seq if cand.res_seq is not None else cand.hyp_seq)

    def op_finish(self, msg: dict) -> None:
        self.prose = {
            "narrative": _text(msg.get("narrative"), 4_000),
            "hyperparameter_rationale": _text(
                msg.get("hyperparameter_rationale"), 1_000)}

    def op_checkpoint(self, msg: dict) -> None:
        """The pipeline's own memory across a pause.  Not an event (it
        is operational state, not research evidence) and not a way in:
        only the sealed pipeline itself can write it, so nothing the
        run could not already know can enter through it."""
        self._checkpoints += 1
        if self._checkpoints > MAX_CHECKPOINTS_PER_RUN:
            raise _Refused("rejected", "too many checkpoints in one run")
        try:
            state = json.dumps(msg.get("state"), sort_keys=True)
        except (TypeError, ValueError):
            raise _Refused("rejected", "state must be JSON-serialisable")
        if len(state) > MAX_CHECKPOINT_CHARS:
            raise _Refused("rejected", "state too large: at most "
                           f"{MAX_CHECKPOINT_CHARS} characters as JSON")
        self.ledger.put_checkpoint(self.run_id, state)

    def op_restore(self, _msg: dict) -> Any:
        state = self.ledger.get_checkpoint(self.run_id)
        return json.loads(state) if state is not None else None

    # ------------------------------------------------------------ closing

    def close(self) -> Dossier:
        """What the run amounts to.  The promoted model is scored once on
        the bars the pipeline never saw, its trade at the last bar is
        computed here, and the dossier is assembled from the record."""
        cand = self.cands.get(self.promoted) if self.promoted else None
        features: List[FeatureSpec] = []
        hp: Dict[str, Any] = {}
        validation: Dict[str, Any] = {"method": "walk_forward"}
        trade: Dict[str, Any] = {}
        if cand is not None:
            held = harness.holdout(self.ds, cand.spec,
                                   self.st.holdout_bars)
            self.emit("holdout.result", {"candidate_id": cand.cid,
                                         "metrics": held}, cand.res_seq)
            trade = harness.trade(
                cand.spec, harness.signal(self.ds, cand.spec),
                int(time.time()) % (2 ** 24 - 1) or 1)
            self.emit("trade.intent", {"candidate_id": cand.cid, **trade},
                      cand.res_seq)
            m = cand.metrics or {}
            features = [FeatureSpec(
                name=cand.name,
                inputs=[cand.spec["feature"]["series"]],
                lookback_hours=float(cand.spec["feature"]["lookback_hours"]),
                transform=cand.spec["feature"]["transform"],
                rationale=cand.rationale)]
            hp = dict(cand.spec["hyperparams"])
            validation.update({
                "n_folds": int(m.get("n_folds", 0) or 0),
                "embargo_hours": float(m.get("embargo_hours", 0) or 0),
                "metrics": {k: v for k, v in m.items()
                            if k not in ("n_folds", "embargo_hours")},
                "holdout": held})
        receipts = self.ledger.receipts(run_id=self.run_id)
        return Dossier(
            version=2, hotkey=self.hotkey, created_ts=time.time(),
            model_name=cand.name if cand else "none",
            model_code_hash=harness.HARNESS_SHA256, seed=0,
            features=features, hyperparameters=hp,
            hyperparameter_rationale=self.prose["hyperparameter_rationale"],
            validation=validation,
            regimes_traded=[int(hp["regime"])] if hp else [],
            run_ids=[self.run_id],
            events_digest=events_digest(
                self.ledger.events_for_run(self.run_id)),
            receipts_digest=receipts_digest(
                [r["receipt_id"] for r in receipts]),
            narrative=self.prose["narrative"],
            pipeline_sha256=self.run["pipeline_sha256"],
            pipeline_args=self.run["args"],
            dataset_sha256=self.run["dataset_sha256"], trade=trade)


# ------------------------------------------------------------------ runner

class SealedRunner:
    """Opens, executes and closes sealed runs.

    ``dataset`` returns the canonical market dataset a new run should
    research (closed bars only; see ``harness.load_dataset``).  Every
    run records the hash of what it was given, so a validator replays
    on the same bars.
    """

    def __init__(self, ledger: CreditLedger, settings: SealedSettings,
                 inference: InferenceClient,
                 compute: Callable[[str, str, int], ComputeJob],
                 dataset: Callable[[], List[float]],
                 sandbox: Optional[Sandbox] = None,
                 slots: Optional["run_slots.Slots"] = None):
        self.ledger, self.st = ledger, settings
        self.orc, self.compute, self.dataset = inference, compute, dataset
        self.sandbox = sandbox or Sandbox()
        # with slots, every run gets its own cores and caps, and only as
        # many run at once as there are slots; the rest wait in line
        self.slots = slots
        self._pool = ThreadPoolExecutor(
            slots.count if slots is not None else settings.max_concurrent_runs,
            thread_name_prefix="sealed-run")
        self._submitting = threading.Lock()
        # Production runs pass KeydInferenceProxy builders via _proxies
        # so plaintext inference keys NEVER enter this process's memory.
        # _keys exists only for direct dev-gateway / local CLI runs.
        self._keys: Dict[str, str] = {}
        self._proxies: Dict[str, Any] = {}
        # run_id -> {"broker", "proc", "done": Event} while executing
        self._active: Dict[str, dict] = {}
        self._active_lock = threading.Lock()
        self._draining = False
        with open(os.path.join(os.path.dirname(__file__),
                               "loop_sdk.py")) as f:
            self._sdk = f.read()
        self._recover()

    def _recover(self) -> None:
        """What a restart owes the runs a previous process held.

        Interrupted runs are *parked*, not killed: their trace, budget
        and checkpoint are all in the ledger, so they continue from
        where they were.  A credit-funded run resumes by itself; a
        BYOK run waits for its miner to bring the key back (it only
        ever lived in the old process's memory), for
        ``resume_window_seconds`` before it fails."""
        for run in (self.ledger.sealed_runs(status="running")
                    + self.ledger.sealed_runs(status="queued")):
            if self.slots is not None:
                self.slots.reap(run["run_id"])
            self._park(run["run_id"], "gateway restarted",
                       float(run.get("elapsed") or 0.0))
        now = time.time()
        for run in self.ledger.sealed_runs(status="paused"):
            if now - float(run.get("paused_ts") or now) \
                    > self.st.resume_window_seconds:
                self._end(run["run_id"], "failed",
                          "paused and never resumed",
                          float(run.get("elapsed") or 0.0))
            elif run.get("funding") != "byok":
                self.ledger.update_sealed_run(run["run_id"],
                                              status="queued")
                self._pool.submit(self._guarded, run["run_id"])

    # ------------------------------------------------------------- submit

    def submit(self, hotkey: str, source: Any, args: Any,
               budget_credits: Any, nonce: str = "",
               api_key: Optional[str] = None,
               max_seconds: Optional[float] = None,
               dataset_text: Optional[str] = None,
               proxy_factory: Optional[Callable[[str], Any]] = None) -> str:
        """Open a run for ``source`` and queue it.  Returns its id.

        ``proxy_factory(run_id)``: production path — binds a
        ``KeydInferenceProxy`` to ``run_id`` so the miner's key stays
        inside ``loop-keyd`` and never enters this process's memory.
        ``api_key``: local/dev BYOK key (Chutes ``cpk_...`` or SayGM
        ``gm_live_...``; OpenRouter keys are refused)."""
        has_byok = (api_key is not None) or (proxy_factory is not None)
        self.check(hotkey, source, args, budget_credits, api_key=api_key,
                   has_proxy=(proxy_factory is not None))
        if max_seconds is not None:
            try:
                max_seconds = float(max_seconds)
            except (TypeError, ValueError):
                raise SubmitError("max_seconds must be a number")
            if not max_seconds > 0:
                raise SubmitError("max_seconds must be positive")
        args = {} if args is None else args
        budget = float(budget_credits)
        if has_byok:
            # the miner's key pays the models; the ledger entry only
            # meters and caps, so the gateway covers it
            self.ledger.grant(hotkey, budget,
                              memo="miner-provided inference")
        try:
            dataset = harness.load_dataset(
                dataset_text if dataset_text is not None
                else harness.dataset_text(self.dataset()))
            harness.research_window(dataset, self.st.holdout_bars)
        except Exception as e:  # noqa: BLE001
            raise SubmitError(f"no dataset to research: {e}")
        with self._submitting:
            if self._in_progress(hotkey):       # two submits can race
                raise SubmitError(_BUSY)
            try:
                info = self.ledger.open_sealed_run(
                    hotkey, budget, nonce or new_id("nonce"),
                    self.ledger.put_blob(source),
                    self.ledger.put_blob(harness.dataset_text(dataset)),
                    args,
                    funding="byok" if has_byok else "credits",
                    max_seconds=max_seconds)
            except Exception as e:  # noqa: BLE001  (no balance, bad budget…)
                raise SubmitError(str(e))
        if proxy_factory is not None:
            self._proxies[info.run_id] = proxy_factory(info.run_id)
        elif api_key is not None:
            self._keys[info.run_id] = validate_inference_key_format(api_key)
        return info.run_id

    def _in_progress(self, hotkey: str) -> bool:
        return any(r["status"] in ("queued", "running", "paused")
                   for r in self.ledger.sealed_runs(hotkey=hotkey))

    def check(self, hotkey: str, source: Any, args: Any,
              budget_credits: Any,
              api_key: Optional[str] = None,
              has_proxy: bool = False) -> None:
        """Raise ``SubmitError`` if ``submit`` would turn this run away.
        Opens nothing and charges nothing."""
        if self.sandbox.isolation not in SEALED and (
                not self.st.allow_unsealed):
            raise SubmitError("this gateway cannot seal a run (no "
                              "sandbox on this host)")
        if not isinstance(source, str) or not source.strip() or (
                len(source) > MAX_SOURCE_CHARS):
            raise SubmitError("source: the pipeline's Python source, at "
                              f"most {MAX_SOURCE_CHARS} characters")
        try:
            compile(source, "pipeline.py", "exec")
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            raise SubmitError(f"source does not compile: {e}")
        args = {} if args is None else args
        if not isinstance(args, dict) or (
                len(json.dumps(args)) > MAX_ARGS_CHARS):
            raise SubmitError("args: a JSON object, at most "
                              f"{MAX_ARGS_CHARS} characters")
        try:
            budget = float(budget_credits)
        except (TypeError, ValueError):
            raise SubmitError("budget_credits must be a number")
        if api_key is not None or has_proxy:
            if api_key is not None:
                try:
                    validate_inference_key_format(api_key)
                except ValueError as e:
                    raise SubmitError(str(e)) from None
            if not budget > 0:
                raise SubmitError("budget_credits must be positive (it "
                                  "caps what your key may spend; set it "
                                  "as high as you like — the wall clock "
                                  "is the bottleneck)")
        elif not budget > 0:
            raise SubmitError("budget_credits must be positive")
        elif budget > self.ledger.balance(hotkey):
            # the run's own time is charged when it ends, whatever is
            # left by then: it must be covered before the run starts
            raise SubmitError("no SayGM key: seal a SayGM (gm_live_...) "
                              "key into the submission "
                              "(build_loop_field(api_key=...)); without "
                              "one, budget_credits must fit the hotkey's "
                              "owner-granted balance")
        if self._in_progress(hotkey):
            raise SubmitError(_BUSY)

    def start(self, run_id: str) -> None:
        """Execute a submitted run in the background."""
        self._pool.submit(self._guarded, run_id)

    def resume(self, run_id: str, api_key: Optional[str] = None,
               proxy_factory: Optional[Callable[[str], Any]] = None
               ) -> None:
        """Continue a paused run (same pipeline, same trace, same
        budget; the wall clock picks up where it stopped).  A BYOK run
        needs its key/proxy again."""
        run = self.ledger.sealed_run(run_id)
        if run is None:
            raise SubmitError("unknown run")
        if run["status"] != "paused":
            raise SubmitError(f"run is {run['status']}, not paused")
        if time.time() - float(run.get("paused_ts") or 0) \
                > self.st.resume_window_seconds:
            self._end(run_id, "failed", "paused and never resumed",
                      float(run.get("elapsed") or 0.0))
            raise SubmitError("the resume window has passed; this run "
                              "has now failed — submit a new one")
        if run.get("funding") == "byok":
            if proxy_factory is not None:
                self._proxies[run_id] = proxy_factory(run_id)
            else:
                if api_key is None:
                    raise SubmitError("api_key: this run pays with your "
                                      "SayGM key; send it again to resume")
                try:
                    self._keys[run_id] = validate_inference_key_format(
                        api_key)
                except ValueError as e:
                    raise SubmitError(str(e)) from None
        # queued, not paused, from the moment the resume is accepted —
        # a crash before it runs parks it right back
        self.ledger.update_sealed_run(run_id, status="queued")
        self._pool.submit(self._guarded, run_id)

    def cancel(self, run_id: str) -> None:
        """A miner abandons its paused run, freeing the hotkey's slot."""
        run = self.ledger.sealed_run(run_id)
        if run is None:
            raise SubmitError("unknown run")
        if run["status"] != "paused":
            raise SubmitError(f"run is {run['status']}, not paused")
        self._end(run_id, "failed", "cancelled by its miner",
                  float(run.get("elapsed") or 0.0))

    def _guarded(self, run_id: str) -> None:
        try:
            self.execute(run_id)
        except Exception:  # noqa: BLE001
            logger.exception("sealed run %s crashed", run_id)

    def pause_all(self, grace: Optional[float] = None) -> List[str]:
        """Stop the gateway without losing anybody's run.  Every active
        pipeline is told to pause (its next call raises ``Paused``;
        ``checkpoint`` still works), given ``grace`` seconds to save and
        exit, then stopped; every interrupted run is parked as paused.
        New executions are refused.  Returns the paused run ids."""
        grace = self.st.pause_grace_seconds if grace is None else grace
        self._draining = True
        with self._active_lock:
            active = dict(self._active)
        for a in active.values():
            a["broker"].pausing.set()
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(
                a["proc"].poll() is None for a in active.values()):
            time.sleep(0.2)
        for a in active.values():
            if a["proc"].poll() is None:
                Sandbox.stop(a["proc"])
        # the run threads write the paused records; wait for them
        for a in active.values():
            a["done"].wait(timeout=grace + 30)
        return sorted(active)

    # ------------------------------------------------------------ execute

    def execute(self, run_id: str) -> dict:
        """Run a queued or paused pipeline until it ends or pauses
        (blocking).  Returns the run's record."""
        run = self.ledger.sealed_run(run_id)
        if run is None or run["status"] not in ("queued", "paused"):
            raise ValueError(f"run {run_id} is not waiting to run")
        prior = float(run.get("elapsed") or 0.0)
        if self._draining:
            self._park(run_id, "gateway is shutting down", prior)
            return self.ledger.sealed_run(run_id)
        try:
            self._execute(run)
        except Exception:
            self._end(run_id, "failed", "gateway error", prior)
            raise
        return self.ledger.sealed_run(run_id)

    def _execute(self, run: dict) -> None:
        run_id = run["run_id"]
        prior = float(run.get("elapsed") or 0.0)
        source = self.ledger.get_blob(run["pipeline_sha256"])
        dataset = harness.load_dataset(
            self.ledger.get_blob(run["dataset_sha256"]))
        info = self.ledger.run_info(run_id)
        proxy = self._proxies.pop(run_id, None)
        key = self._keys.pop(run_id, None)
        if run.get("funding") == "byok" and proxy is None and key is None:
            # cannot happen through the public paths (submit and resume
            # both set the proxy or key); refuse rather than bill the gateway
            self._park(run_id, "no inference key in memory", prior)
            return
        if proxy is not None:
            orc = proxy
        elif key is not None:
            orc = (self.orc if getattr(self.orc, "_is_stub", False)
                   else InferenceClient(api_key=key))
        else:
            orc = self.orc
        del key
        broker = Broker(self.ledger, self.st, orc, self.compute,
                        run, dataset)
        # Running time is charged per segment, so the whole run is
        # capped by what the budget could pay for; a resumed run gets
        # what the clock has left, not a fresh clock.
        limit = min(float(run.get("max_seconds") or self.st.max_run_seconds),
                    info.budget_credits / max(self.st.pipeline_rate_per_sec,
                                              1e-12)) - prior
        if limit <= 0:
            self._end(run_id, "failed", "ran out of time", prior)
            return
        self.ledger.update_sealed_run(
            run_id, status="running", isolation=self.sandbox.isolation)
        # The trace itself says whether this is a resume: a run parked
        # before it ever started has no events and starts fresh.
        broker.rebuild()
        share = ({"cpus": self.slots.shape.cpus,
                  "memory_mb": self.slots.shape.memory_mb,
                  "scratch_mb": self.slots.shape.scratch_mb}
                 if self.slots is not None else {})
        if broker._seq > 0:
            broker.emit("run.resumed", {
                "elapsed_seconds": round(prior, 1),
                "remaining_seconds": round(limit, 1),
                "isolation": self.sandbox.isolation, **share})
        else:
            research = broker.research["series"]
            broker.emit("run.started", {
                "budget_credits": info.budget_credits,
                "pipeline_sha256": run["pipeline_sha256"],
                "dataset_sha256": run["dataset_sha256"],
                "harness_sha256": harness.HARNESS_SHA256,
                "args": run["args"],
                "target": dataset["target"],
                "bars": len(research[dataset["target"]]["closes"]),
                "series": {name: len(s["closes"])
                           for name, s in sorted(research.items())},
                "holdout_bars": self.st.holdout_bars,
                "seconds": round(limit, 1),
                "isolation": self.sandbox.isolation, **share})

        work = tempfile.mkdtemp(prefix="loop-run-")
        os.chmod(work, 0o755)
        for name, text in (("pipeline.py", source), ("loop_sdk.py", self._sdk),
                           ("_main.py", _MAIN_PY)):
            with open(os.path.join(work, name), "w", encoding="utf-8") as f:
                f.write(text)
        done = threading.Event()
        held = (self.slots.held(run_id) if self.slots is not None
                else contextlib.nullcontext(None))
        err_ring: "collections.deque[bytes]" = collections.deque()
        err_bytes = 0

        def _drain_stderr(stream: Any) -> None:
            nonlocal err_bytes
            if stream is None:
                return
            buf = getattr(stream, "buffer", stream)
            try:
                while True:
                    chunk = buf.read(4096)
                    if not chunk:
                        break
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8", "replace")
                    if len(chunk) >= MAX_STDERR_CHARS:
                        err_ring.clear()
                        err_ring.append(chunk[-MAX_STDERR_CHARS:])
                        err_bytes = MAX_STDERR_CHARS
                    else:
                        err_ring.append(chunk)
                        err_bytes += len(chunk)
                        while err_bytes > MAX_STDERR_CHARS * 2 and len(err_ring) > 1:
                            err_bytes -= len(err_ring.popleft())
            except Exception:  # noqa: BLE001
                pass

        with held as envelope:
            t0 = time.monotonic()
            proc = self.sandbox.start(
                work, stderr=subprocess.PIPE, envelope=envelope)
            err_thread = threading.Thread(
                target=_drain_stderr, args=(proc.stderr,),
                daemon=True, name=f"stderr-{run_id[:8]}")
            err_thread.start()
            with self._active_lock:
                self._active[run_id] = {"broker": broker, "proc": proc,
                                        "done": done}
            timed_out = threading.Event()

            def expire() -> None:
                timed_out.set()
                Sandbox.stop(proc)
            broker.deadline = t0 + limit
            timer = threading.Timer(limit, expire)
            timer.start()
            try:
                with run_slots.compute_in(envelope):
                    reason = self._serve(proc, broker)
                if not reason and not timed_out.is_set() \
                        and not broker.pausing.is_set():
                    try:
                        # the channel closed: let the pipeline exit on
                        # its own so its exit status is its own
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        reason = ("pipeline closed its channel and kept "
                                  "running")
            finally:
                timer.cancel()
                Sandbox.stop(proc)
                err_thread.join(timeout=2.0)
                shutil.rmtree(work, ignore_errors=True)
                with self._active_lock:
                    self._active.pop(run_id, None)
                if proxy is not None and hasattr(proxy, "release") \
                        and not broker.pausing.is_set():
                    proxy.release()
            seconds = time.monotonic() - t0
            used = envelope.usage() if envelope is not None else {}
            raw_tail = b"".join(err_ring)[-MAX_STDERR_CHARS:]
            log = raw_tail.decode("utf-8", "replace")
        elapsed = prior + seconds
        pausing = broker.pausing.is_set()
        if timed_out.is_set():
            reason = "ran out of time"
        elif used.get("oom_kills") and not pausing:
            reason = (f"ran out of memory (its share is "
                      f"{self.slots.shape.memory_mb / 1024:g} GB)")
        elif not pausing and not reason and proc.returncode != 0:
            reason = f"pipeline exited with status {proc.returncode}"

        broker.receipt(
            "compute", "sealed", f"pipeline-{self.sandbox.isolation}",
            seconds * self.st.pipeline_rate_per_sec,
            sha256_hex(run["pipeline_sha256"]),
            {"seconds": seconds,
             "status": ("paused" if pausing and not timed_out.is_set()
                        else "failed" if reason else "done"),
             "attribution": broker._label("pipeline")}, force=True)
        if pausing and not timed_out.is_set():
            # parked, not ended: the trace, budget and checkpoint all
            # live in the ledger; only the process (and a BYOK key)
            # are gone.  No dossier until the run really finishes.
            broker.emit("run.paused",
                        {"seconds": round(seconds, 1),
                         "elapsed_seconds": round(elapsed, 1)})
            self._park(run_id, "gateway paused the run", elapsed, log=log)
            done.set()
            return
        dossier = None
        if not reason:
            dossier = broker.close()
            broker.emit("dossier.submitted",
                        {"dossier_hash": dossier.dossier_hash()})
        broker.emit("run.ended", {"status": "failed" if reason else "done",
                                  "reason": reason,
                                  "seconds": round(elapsed, 1), **used})
        self._end(run_id, "failed" if reason else "done", reason, elapsed,
                  log=log, dossier=dossier, used=used)
        done.set()

    def _serve(self, proc: subprocess.Popen, broker: Broker) -> str:
        """Answer the pipeline until it exits.  Returns why the run must
        be thrown away, or "" if it ended in order."""
        while True:
            line = proc.stdout.readline(MAX_LINE_CHARS + 1)
            if not line:
                return ""
            if len(line) > MAX_LINE_CHARS or not line.endswith("\n"):
                return "pipeline sent an oversized or broken request"
            try:
                msg = json.loads(line)
            except ValueError:
                return "pipeline sent something that is not a request"
            reply = broker.handle(msg)
            if broker.flood_aborted:
                return broker.flood_aborted
            try:
                proc.stdin.write(json.dumps(reply) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return ""

    def _park(self, run_id: str, reason: str, elapsed: float,
              log: str = "") -> None:
        """Pause, not end: the run keeps its slot, its budget stays
        open, and the clock remembers what was used."""
        self.ledger.update_sealed_run(
            run_id, status="paused", paused_ts=time.time(),
            elapsed=round(elapsed, 3),
            detail={"reason": reason, "seconds": round(elapsed, 1),
                    "stderr_tail": log, "resumable": True})

    def _end(self, run_id: str, status: str, reason: str, seconds: float,
             log: str = "", dossier: Optional[Dossier] = None,
             used: Optional[Dict[str, float]] = None) -> None:
        fields: Dict[str, Any] = {
            "status": status, "ended_ts": time.time(),
            "elapsed": round(seconds, 3), "paused_ts": None,
            "detail": {"reason": reason, "seconds": round(seconds, 1),
                       "stderr_tail": log, **(used or {})}}
        if dossier is not None:
            fields["dossier"] = dossier.to_dict()
        self.ledger.update_sealed_run(run_id, **fields)
        self.ledger.close_run(run_id)


# -------------------------------------------------------------- local runs

@dataclass
class LocalRun:
    """A pipeline run on this machine, as the gateway would run it."""
    run: dict
    dossier: Optional[Dossier]
    events: List[dict]
    receipts: List[dict]
    ledger: CreditLedger = field(repr=False, default=None)

    @property
    def payload_vector(self) -> List[float]:
        return harness.payload_vector(self.dossier.trade
                                      if self.dossier else {})


def run_local(source: str, dataset: Any,
              inference: InferenceClient, args: Optional[dict] = None,
              hotkey: str = "local", budget_credits: float = 5.0,
              db_path: Optional[str] = None, secret: str = "local",
              sandbox: Optional[Sandbox] = None,
              **settings: Any) -> LocalRun:
    """Run a pipeline here, end to end: same broker, same harness, same
    sandbox (when this host has one) as a gateway.  For developing a
    pipeline — a gateway does not accept the result.  ``dataset`` is a
    dataset dict, or a plain list of hourly closes."""
    from .providers import LocalComputeProvider
    if isinstance(dataset, list):
        dataset = harness.single_series(dataset)
    tmp = None if db_path else tempfile.mkdtemp(prefix="loop-local-")
    try:
        ledger = CreditLedger(db_path or os.path.join(tmp, "ledger.db"))
        ledger.grant(hotkey, budget_credits)
        local = LocalComputeProvider()
        runner = SealedRunner(
            ledger, SealedSettings(secret=secret, allow_unsealed=True,
                                   **settings),
            inference,
            lambda src, stdin, secs: local.run(src, secs, stdin),
            lambda: dataset, sandbox)
        run_id = runner.submit(hotkey, source, args, budget_credits)
        run = runner.execute(run_id)
        return LocalRun(
            run=run,
            dossier=Dossier.from_dict(run["dossier"]) if run["dossier"]
            else None,
            events=ledger.events_for_run(run_id),
            receipts=ledger.receipts(run_id=run_id), ledger=ledger)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    from .loop import StubInference, synthetic_dataset
    p = argparse.ArgumentParser(
        description="Run a pipeline locally, as a gateway would.")
    p.add_argument("pipeline", help="path to the pipeline's source")
    p.add_argument("--args", default="{}", help="JSON arguments")
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--dataset", help="JSON dataset file (a dict of "
                                     "series, or a plain list of hourly "
                                     "closes; default: synthetic)")
    p.add_argument("--live", action="store_true",
                   help="call real models with SAYGM_API_KEY "
                        "(default: the offline stub, free)")
    a = p.parse_args(argv)
    with open(a.pipeline) as f:
        source = f.read()
    if a.dataset:
        with open(a.dataset) as f:
            dataset = json.load(f)
    else:
        dataset = synthetic_dataset(4000)
    out = run_local(source, dataset,
                    InferenceClient() if a.live else StubInference(),
                    json.loads(a.args), budget_credits=a.budget)
    for e in out.events:
        print(f"{e['seq']:>4} {e['kind']:<20} "
              f"{json.dumps(e['data'], sort_keys=True)[:140]}")
    print(json.dumps({"run": {k: out.run[k] for k in
                              ("status", "isolation", "detail")},
                      "dossier": out.dossier.to_dict() if out.dossier
                      else None}, indent=2))
    return 0 if out.run["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
