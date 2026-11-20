# MIT License
# Copyright (c) 2024 MANTIS
"""The LOOP 24-hour round system: submissions in by midnight, evaluated
at midnight, published when done.

A new evaluation round opens at 0000 UTC.  The auditor validator (UID
0) reads every payload it fetches with the owner key — the payload's
owner wrap, no waiting for the timelock — and keeps the LOOP submission
in its private inbox, stamped with when it was first read
(``RoundRunner.take_payloads``).  Each hotkey's newest submission first
read in the preceding 24 hours is evaluated then, in this process:

    collect -> screen -> run sealed -> audit -> publish -> dashboard

Every entry in a round runs on the same terms (``slots.py``).  At most
``slots`` run at once, each on cores of its own under the same memory
cap, and the rest wait in line.  Every run's wall clock is the round's
run window (20 hours) split into waves, ``min(12 h, 20 h / ceil(entries
/ slots))``, so even an entry from every UID ends inside the day; only
entries that pass the screen and could start count.  All of them
research the one dataset snapshot taken when the round opens.
Screening and audits run a few at a time; a run is audited as soon as
it ends.

The dashboard shows a round's code and research only once the round
has closed, so nobody copies a pipeline before it has been judged; from
then on it is public, and fair game for the rounds after.  The datalog
still decrypts every payload a week after upload
(``ledger.py::loop_submissions``); those matured copies are a backstop
for anything the inbox missed, and never run a submission twice.

There is no submission server, no gateway, no port.  Submissions arrive
through the chain (the miner's committed R2 payload), evidence is read
straight off the runner's ledger (``audit.local_evidence``), the verdict
map goes out through the already-built signed feed + on-chain pointer,
and the dashboard is a rendered file.

The screen
----------
Before any miner code executes, a screening judge — a strong open-weights
model (``LOOP_SCREEN_MODELS``, default Kimi K3.1 Pro, falling back to
Qwen3.8 Max Prime) — reads the source together with the args it will
receive and answers one yes/no question: *does this code try to
escape or subvert its sandbox?*  "Yes" means the code never runs and the
verdict is on the dashboard with the judge's reason.  The screen is a
tripwire in front of the sandbox, not a quality bar: bad
research passes the screen and fails the audit.  If every screen model
is unreachable the submission does NOT run — fail closed — and is
recorded ``unscreened`` (it can run in a later round by resubmitting).

Every step of a round is journaled twice: structured JSONL under
``LOOP_ROUND_JOURNAL_DIR`` (one file per round, append-only) and the
ordinary process log.  Nothing key-shaped is ever written to either
(see ``submission.redact``).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from . import harness
from . import slots as run_slots
from .audit import AuditorPanel, local_evidence
from .credits import CreditLedger
from .schema import Dossier
from .sealed import SealedRunner, SubmitError
from .submission import (loop_submission_of, open_payload,
                         open_submission_key, parse_submission, redact)

logger = logging.getLogger(__name__)

ROUND_SECONDS = 86_400
SCREEN_WORKERS = 8              # screen calls in flight as a round opens
AUDIT_WORKERS = 4               # audits in flight as runs end

try:
    from config import LOOP_SCREEN_MODELS as _CFG_SCREEN_MODELS
    SCREEN_MODELS = list(_CFG_SCREEN_MODELS)
except ImportError:
    SCREEN_MODELS = [
        "kimi-k3.1-pro-tee",
        "qwen3.5-397b-a17b-tee",
        "qwen3.8-27b",
    ]

SCREEN_SYSTEM = """\
You are the sandbox screen for the MANTIS LOOP challenge.  You read one
miner-submitted Python pipeline before it is executed inside a sealed
sandbox (no network, no host files; its only I/O is a JSON
protocol on stdin/stdout to a research broker, via the provided
loop_sdk module).  The source comes first, then the JSON args the
pipeline receives from loop.args().  Judge them as one program: any
code the args carry that the pipeline runs, imports or evaluates is
part of the pipeline.

Answer one question only: does this code attempt to ESCAPE OR SUBVERT
its execution environment?  That means things like: breaking out of or
probing the sandbox; opening sockets or reaching any network or host
resource beyond the stdio protocol; reading or writing files outside
its working directory; spawning processes to the same end; tampering
with, exhausting, or crashing the host or the broker; smuggling host
information out through the protocol; or interfering with the machinery
that runs, measures, or audits it.  Running code you cannot read
(encoded, compressed or encrypted, then executed) counts as "yes":
research has no reason to hide its code.

It is NOT your question whether the research is good, honest, or
compliant with the challenge rules — a separate audit judges that.
Plain research code that merely wastes its budget, hardcodes answers,
or writes bad Python is a "no".

Reply with strict JSON, nothing else:
{"escape": "yes" or "no", "reason": "<one or two sentences>"}
"""


class ScreenUnavailable(Exception):
    """No screen model could be reached.  The code does not run."""


def screen_text(sub: dict) -> str:
    """What the screen reads: the source, then the args the pipeline
    gets from ``loop.args()`` — code can ride in either.  Not redacted:
    the screen must see exactly what will run."""
    return (sub["source"] + "\n\n# ---- args: the JSON this pipeline "
            "receives from loop.args() ----\n"
            + json.dumps(sub.get("args") or {}, indent=2, sort_keys=True))


def screen_source(inference, source: str,
                  models: Optional[List[str]] = None) -> Dict[str, Any]:
    """One yes/no from the first screen model that answers.

    Returns ``{"escape": bool, "reason": str, "model": str}``.  A reply
    that cannot be parsed as a verdict counts as that model failing —
    never as a pass."""
    errors: List[str] = []
    for model in models or SCREEN_MODELS:
        try:
            reply, _cost = inference.chat({
                "model": model,
                "messages": [
                    {"role": "system", "content": SCREEN_SYSTEM},
                    {"role": "user", "content": source}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            })
            text = reply["choices"][0]["message"]["content"]
            start, end = text.find("{"), text.rfind("}")
            verdict = json.loads(text[start:end + 1])
            answer = str(verdict.get("escape", "")).strip().lower()
            if answer not in ("yes", "no"):
                raise ValueError(f"no yes/no in reply: {text[:120]!r}")
            return {"escape": answer == "yes",
                    "reason": str(verdict.get("reason", ""))[:500],
                    "model": model}
        except Exception as e:  # noqa: BLE001
            errors.append(f"{model}: {e}")
    raise ScreenUnavailable("; ".join(errors)[:800])


# ------------------------------------------------------------ round math

def round_id_at(ts: Optional[float] = None) -> str:
    """The round whose evaluation began at the most recent 0000 UTC:
    named by that boundary's date."""
    ts = time.time() if ts is None else float(ts)
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime(
        "%Y-%m-%d")

def round_window(round_id: str) -> tuple:
    """``(start_ts, end_ts)``: submissions first read in the 24 hours
    before the round's 0000 UTC boundary belong to it."""
    end = _dt.datetime.strptime(round_id, "%Y-%m-%d").replace(
        tzinfo=_dt.timezone.utc).timestamp()
    return end - ROUND_SECONDS, end


def matured_submissions(datalog_db: str, start_ts: float,
                        end_ts: float) -> List[dict]:
    """Submissions the datalog decrypted inside the window — through
    the timelock, a week after upload.  Plain SQLite read: the round
    runner never imports the datalog."""
    if not os.path.exists(datalog_db):
        return []
    conn = sqlite3.connect(f"file:{datalog_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT hotkey, sha256, submission, ts FROM loop_submissions "
            "WHERE ts >= ? AND ts < ? ORDER BY ts, sha256",
            (start_ts, end_ts)).fetchall()
    except sqlite3.OperationalError:       # pre-rounds datalog
        return []
    finally:
        conn.close()
    return [{"hotkey": hk, "sha256": sha, "body": body, "ts": ts}
            for hk, sha, body, ts in rows]


# ----------------------------------------------------------- the journal

class Journal:
    """Append-only JSONL, one file per round, mirrored to the process
    log.  Everything a miner needs to understand what happened to its
    entry, and everything an operator needs to replay a round."""

    def __init__(self, journal_dir: str, round_id: str):
        os.makedirs(journal_dir, exist_ok=True)
        self.path = os.path.join(journal_dir, f"round-{round_id}.jsonl")
        self.round_id = round_id
        self._lock = threading.Lock()

    def log(self, event: str, **fields: Any) -> None:
        rec = {"ts": round(time.time(), 3), "round": self.round_id,
               "event": event, **fields}
        line = redact(json.dumps(rec, sort_keys=True))
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info("round %s %s %s", self.round_id, event,
                    redact(" ".join(f"{k}={v}" for k, v in
                                    sorted(fields.items())))[:400])


# --------------------------------------------------------- round runner

# A round entry moves through exactly one of these ends:
#   refused     - never ran: malformed, unfundable, or slot blocked
#   blocked     - the screen said yes: sandbox-escape attempt, never ran
#   unscreened  - no screen model answered: never ran (fail closed)
#   failed      - ran and broke (or was cancelled by the clock)
#   audited     - ran to a dossier and holds a verdict multiplier
TERMINAL = {"refused", "blocked", "unscreened", "failed", "audited"}


def dashboard_dir() -> str:
    """The public site's directory (``config.LOOP_DASHBOARD_DIR``): the round
    pages and the signed verdict files are published from it."""
    import config as mantis_config
    return mantis_config.LOOP_DASHBOARD_DIR


def check_enc_seed(enc_seed: str, published_pubkey: str) -> None:
    """``ValueError`` unless the seed opens what miners seal.  Their keys
    are sealed to the published X25519 key: under any other seed every
    BYOK submission is refused, and nothing says so until the round."""
    if not published_pubkey:
        return
    if not enc_seed:
        raise ValueError("no private key, but miners seal their keys to "
                         "the published owner key")
    from . import sealbox
    if sealbox.keypair(enc_seed)[1] != published_pubkey.strip().lower():
        raise ValueError("this is not the private half of the owner key "
                         "miners seal to (OWNER_HPKE_PUBLIC_KEY_HEX): every "
                         "BYOK submission would be refused")


def owner_seed(path: str) -> str:
    """The owner's X25519 seed, read from its key file."""
    from . import sealbox
    try:
        return sealbox.seed_from_file(path)
    except FileNotFoundError:
        raise ValueError(f"the owner key is not at {path}: copy "
                         f"owner_hpke_x25519.pem there") from None


def _write_seal_keys_file(site_dir: str, auditor_hotkey: str,
                          sign: Callable[[str], str],
                          keyd_state: Dict[str, Any]) -> str:
    """Write ``<site_dir>/seal_keys.json`` signed by UID 0 from the
    public keys returned by ``loop-keyd`` (UID 0 never touches the
    rotating private seeds)."""
    from . import sealbox
    keys = dict(keyd_state.get("active") or {})
    rec_id = str((keyd_state.get("next") or {}).get("key_id") or "")
    if not rec_id or rec_id not in keys:
        raise ValueError("keyd returned no recommended next-round key")
    ts = round(time.time(), 3)
    msg = sealbox.seal_keys_message(auditor_hotkey, ts, keys, rec_id)
    payload = {
        "version": sealbox.SEAL_KEYS_VERSION,
        "auditor_hotkey": auditor_hotkey,
        "ts": ts,
        "recommended_key_id": rec_id,
        "recommended_pubkey": keys[rec_id],
        "keys": keys,
        "signature": sign(msg),
    }
    os.makedirs(site_dir, exist_ok=True)
    dest = os.path.join(site_dir, "seal_keys.json")
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
    os.replace(tmp, dest)
    return dest


def from_env(datalog_db: str,
             publish: Optional[Callable[[AuditorPanel], None]] = None,
             auditor_hotkey: str = "",
             sign: Optional[Callable[[str], str]] = None
             ) -> "RoundRunner":
    """The auditor's round runner (configured strictly from ``config.py``;
    only ``SAYGM_API_KEY`` and optional ``RUNPOD_API_KEY`` secrets come
    from environment variables).  Production enforces unconditionally:
      * ``OWNER_HPKE_PRIVATE_KEY_PATH`` verified against
        ``OWNER_HPKE_PUBLIC_KEY_HEX`` (opens outer v2 ``W_owner`` wraps
        and derives the receipt HMAC key),
      * ``SAYGM_API_KEY`` (auditor's SayGM inference for the escape screen
        and audit judge),
      * ``gvisor`` sandbox + root ``cgroup v2`` dedicated per-run slots
        (no env downgrade possible),
      * ``allow_unsealed = False`` on both runner and audit panel,
      * ``loop-keyd`` out-of-process privilege separation on
        ``LOOP_KEYD_SOCKET`` over ``LOOP_SEAL_KEYRING_DIR`` (no ``W_owner``
        fallback)."""
    import config as mantis_config
    from . import sealbox
    from .data import MarketData
    from .gateway import GatewaySettings
    from .keyd import KeydProcess
    from .providers import (InferenceClient, LocalComputeProvider,
                            RunPodProvider, best_available)
    from .sealed import Sandbox, SealedSettings

    seed = owner_seed(mantis_config.OWNER_HPKE_PRIVATE_KEY_PATH)
    check_enc_seed(seed, mantis_config.OWNER_HPKE_PUBLIC_KEY_HEX)
    st = GatewaySettings(
        secret=sealbox.receipt_secret(seed),
        enc_seed=seed,
        dev_mode=False,
    )
    orc = InferenceClient()
    if not orc.available():
        raise ValueError(
            "SAYGM_API_KEY is unset: the escape screen and the audit "
            "judge run on SayGM (SN28)")
    kind = str(mantis_config.LOOP_SANDBOX).strip().lower()
    if kind != "gvisor":
        raise ValueError(
            f"production LOOP rounds require LOOP_SANDBOX='gvisor' "
            f"(got {kind!r}); unsealed or host-kernel downgrades are forbidden")
    if getattr(mantis_config, "LOOP_ALLOW_UNSEALED", False):
        raise ValueError("production LOOP rounds forbid LOOP_ALLOW_UNSEALED")
    why = Sandbox.why_not("gvisor")
    if why:
        raise ValueError(
            f"gVisor cannot run pipelines here ({why}): install runsc "
            "(see autoresearch/README.md, The auditor's host)")
    why = run_slots.cgroups_ready()
    if why:
        raise ValueError(f"runs cannot get a share of their own ({why}): "
                         "the auditor runs as root on cgroup v2")

    keyd_proc = KeydProcess.connect_or_spawn(
        keyring_dir=mantis_config.LOOP_SEAL_KEYRING_DIR,
        socket_path=mantis_config.LOOP_KEYD_SOCKET,
        retention_days=int(mantis_config.LOOP_SEAL_RETENTION_DAYS),
    )

    slots = run_slots.Slots.from_config()
    logger.info("LOOP runs: %d at a time, each on %s, sandbox %s, "
                "auditor inference %s",
                slots.count, slots.shape.describe(), kind, orc.provider_name)
    ledger = CreditLedger(mantis_config.LOOP_LEDGER_DB)
    runner = SealedRunner(
        ledger,
        SealedSettings(
            secret=st.secret, holdout_bars=st.holdout_bars,
            max_run_seconds=st.max_run_seconds, llm_fee=st.llm_fee,
            pipeline_rate_per_sec=st.local_rate_per_sec,
            max_concurrent_runs=slots.count,
            pause_grace_seconds=st.pause_grace_seconds,
            resume_window_seconds=st.resume_window_seconds,
            allow_unsealed=False),       # production: sealed or refused
        orc,
        best_available(RunPodProvider(), LocalComputeProvider(
            rate_per_sec=st.local_rate_per_sec)),
        MarketData(), sandbox=Sandbox("gvisor"), slots=slots)
    panel = AuditorPanel(
        cache_db=mantis_config.LOOP_AUDIT_DB,
        inference=orc, judges=mantis_config.LOOP_AUDIT_JUDGES,
        gateway_secret=st.secret,
        allow_unsealed=False)
    site_dir = dashboard_dir()

    def refresh_seal_manifest() -> None:
        state = keyd_proc.client.rotate()
        if auditor_hotkey and sign is not None:
            _write_seal_keys_file(site_dir, auditor_hotkey, sign, state)

    refresh_seal_manifest()

    def render(led: CreditLedger, _rid: str) -> None:
        from .round_site import render_rounds_site
        refresh_seal_manifest()
        render_rounds_site(led, datalog_db, site_dir, panel)

    return RoundRunner(
        datalog_db, ledger, runner, panel, orc, enc_seed=st.enc_seed,
        keyd=keyd_proc.client, keyd_proc=keyd_proc,
        require_keyd=True,
        journal_dir=mantis_config.LOOP_ROUND_JOURNAL_DIR,
        publish=publish, render=render)


from_config = from_env


class RoundRunner:
    """Evaluates one 24h round at a time on the auditor validator.

    ``publish`` is called with the panel after every round so the
    verdict map leaves the building (signed feed + chain pointer);
    ``render`` is called with (ledger, round_id) to redraw the
    dashboard.  Both optional, both injected, so this class stays a
    pure orchestrator."""

    def __init__(self, datalog_db: str, ledger: CreditLedger,
                 runner: SealedRunner, panel: AuditorPanel,
                 inference, enc_seed: str = "",
                 keyd: Optional[Any] = None,
                 keyd_proc: Optional[Any] = None,
                 keyring: Optional[Any] = None,
                 require_keyd: bool = False,
                 screen: Optional[Callable[[str], dict]] = None,
                 journal_dir: str = "logs/loop",
                 publish: Optional[Callable[[AuditorPanel], None]] = None,
                 render: Optional[Callable[[CreditLedger, str], None]] = None,
                 run_timeout: Optional[float] = None,
                 window_seconds: Optional[float] = None):
        if require_keyd and keyd is None:
            raise ValueError("production RoundRunner requires loop-keyd")
        self.datalog_db, self.ledger, self.runner = datalog_db, ledger, runner
        self.panel, self.orc = panel, inference
        self.enc_seed = enc_seed
        self.keyd = keyd
        self._keyd_proc = keyd_proc
        self.keyring = keyring
        self.require_keyd = require_keyd
        self.screen = screen or (
            lambda text: screen_source(self.orc, text))
        self.journal_dir = journal_dir
        self.publish, self.render = publish, render
        # the window every run of a round must end in, and how long the
        # round waits on its runs before it gives up on them (screening
        # and audits take their time too)
        self.window = (run_slots.window_seconds() if window_seconds is None
                       else float(window_seconds))
        self.run_timeout = (self.window + 4 * 3600.0 if run_timeout is None
                            else float(run_timeout))
        self.poll_seconds = 2.0
        self._busy = threading.Lock()
        self._uploads: Dict[str, str] = {}  # hotkey -> last payload read

    def close(self) -> None:
        """Stop any managed ``loop-keyd`` child process on shutdown."""
        if self._keyd_proc is not None:
            self._keyd_proc.stop()

    # ------------------------------------------------------------- tick

    def tick(self, now: Optional[float] = None) -> Optional[str]:
        """Run the current round if it has not finished.  Returns the
        round id when this call did the work (blocking), else None.
        Call it from a background thread; it is single-flight."""
        rid = round_id_at(now)
        done = self.ledger.round(rid)
        if done is not None and done["finished_ts"] is not None:
            return None
        if not self._busy.acquire(blocking=False):
            return None
        try:
            self.run_round(rid)
            return rid
        finally:
            self._busy.release()

    def tick_forever(self, poll_seconds: float = 60.0) -> None:
        """The auditor's loop: evaluate each round as midnight passes.
        Run this in a daemon thread next to the validator."""
        while True:
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                logger.exception("round tick failed")
            time.sleep(poll_seconds)

    # ----------------------------------------------------------- intake

    def take_payloads(self, payloads: Dict[str, Any],
                      now: Optional[float] = None) -> int:
        """Take the LOOP submissions out of the payloads the validator
        just fetched (hotkey -> v2 payload), opened with the owner key.
        A submission is stamped when first read, and the stamp picks its
        round: the one opening at the next 0000 UTC.  Only the
        submission is kept, in this runner's private ledger — never the
        public datalog.  Returns how many were new."""
        if not self.enc_seed:
            return 0
        new = 0
        for hk, payload in payloads.items():
            try:
                upload = str(payload["C"]["nonce"])  # fresh per encryption
            except (TypeError, KeyError, IndexError):
                continue
            if self._uploads.get(hk) == upload:
                continue
            self._uploads[hk] = upload
            found = loop_submission_of(open_payload(payload, hk,
                                                    self.enc_seed))
            if found is None or not self.ledger.inbox_add(hk, *found,
                                                          seen_ts=now):
                continue
            new += 1
            seen = time.time() if now is None else now
            Journal(self.journal_dir,
                    round_id_at(seen + ROUND_SECONDS)).log(
                "entry.received", hotkey=hk, sha256=found[0][:16])
        return new

    # ------------------------------------------------------------ rounds

    def run_round(self, rid: str) -> dict:
        self.ledger.open_round(rid)
        j = Journal(self.journal_dir, rid)
        start_ts, end_ts = round_window(rid)
        for item in matured_submissions(self.datalog_db, start_ts, end_ts):
            self.ledger.inbox_add(item["hotkey"], item["sha256"],
                                  item["body"], seen_ts=item["ts"])
        subs = self.ledger.inbox_window(start_ts, end_ts)
        prior = {e["hotkey"]: e for e in self.ledger.round_entries(rid)}
        j.log("round.opened", submissions=len(subs),
              window=[start_ts, end_ts],
              resuming=sum(1 for e in prior.values()
                           if e["status"] not in TERMINAL))
        deadline = time.monotonic() + self.run_timeout

        running: Dict[str, dict] = {}       # hotkey -> entry with run_id
        fresh: List[str] = []
        for hk in sorted(subs):
            entry = prior.get(hk)
            if entry and entry["status"] in TERMINAL:
                continue                     # a restart owes it nothing
            if entry and entry["status"] == "running" and entry["run_id"]:
                revived = self._revive(j, hk, subs[hk], entry["run_id"])
                if revived:
                    running[hk] = {"run_id": entry["run_id"],
                                   "carry": entry["detail"] or {}}
                continue
            fresh.append(hk)
        vetted: List[tuple] = []
        if fresh:
            # screened a few at a time, before the round's terms are
            # set: what is turned away does not shorten anyone's clock
            with ThreadPoolExecutor(min(SCREEN_WORKERS, len(fresh)),
                                    thread_name_prefix="loop-screen") as ex:
                checked = list(ex.map(
                    lambda hk: self._vet(j, rid, hk, subs[hk]), fresh))
            vetted = [(hk, v) for hk, v in zip(fresh, checked) if v]
        plan = (self._plan(j, rid, len(vetted) + len(running))
                if vetted or running else {})
        dataset = (self.ledger.get_blob(plan["dataset_sha256"])
                   if plan.get("dataset_sha256") else None)
        for hk, v in vetted:
            started = self._start_entry(j, rid, hk, subs[hk], v, plan,
                                        dataset)
            if started is not None:
                running[hk] = started
        del vetted                          # the keys went with the runs
        self._render(j, rid)                 # the day's entries, running

        self._finish_all(j, rid, subs, running, deadline)

        counts: Dict[str, int] = {}
        for e in self.ledger.round_entries(rid):
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        self.ledger.finish_round(rid, {"submissions": len(subs),
                                       "status_counts": counts})
        j.log("round.finished", **counts)
        self._render(j, rid)             # first: publishing may wait out
        if self.publish is not None:     # chain retries
            try:
                self.publish(self.panel)
                j.log("verdicts.published")
            except Exception as e:  # noqa: BLE001
                j.log("verdicts.publish_failed", error=str(e)[:300])
        return self.ledger.round(rid)

    def _plan(self, j: Journal, rid: str, entries: int) -> dict:
        """The round's terms, fixed when it first opens and kept across
        restarts: how many run at once, every run's wall clock and
        share, and the one dataset snapshot they all research."""
        plan = (self.ledger.round(rid) or {}).get("plan") or {}
        if plan:
            return plan
        st, slots = self.runner.st, self.runner.slots
        count = slots.count if slots is not None else st.max_concurrent_runs
        plan = {"entries": entries, "slots": count,
                "run_seconds": run_slots.run_seconds(
                    entries, count, self.window, st.max_run_seconds),
                "window_seconds": int(self.window),
                "isolation": self.runner.sandbox.isolation}
        if slots is not None:
            plan.update(cpus=slots.shape.cpus,
                        memory_mb=slots.shape.memory_mb,
                        scratch_mb=slots.shape.scratch_mb)
        try:
            plan["dataset_sha256"] = self.ledger.put_blob(
                harness.dataset_text(self.runner.dataset()))
        except Exception as e:  # noqa: BLE001  (each run fetches its own)
            j.log("round.dataset_failed", error=str(e)[:300])
        self.ledger.set_round_plan(rid, plan)
        j.log("round.plan", **plan)
        return plan

    def _render(self, j: Journal, rid: str) -> None:
        if self.render is None:
            return
        try:
            self.render(self.ledger, rid)
            j.log("dashboard.rendered")
        except Exception as e:  # noqa: BLE001
            j.log("dashboard.render_failed", error=str(e)[:300])

    # ----------------------------------------------------------- entries

    def _record(self, j: Journal, rid: str, hk: str, sha: str, status: str,
                detail: Optional[dict] = None,
                run_id: Optional[str] = None) -> None:
        self.ledger.record_round_entry(rid, hk, sha, status, detail,
                                       run_id)
        if status in TERMINAL:
            # Wipe key_sealed from loop_inbox as soon as an entry ends so
            # no sealed inference key blob persists past its round.
            self.ledger.scrub_inbox_key(hk, sha)
        j.log(f"entry.{status}", hotkey=hk, sha256=sha[:16],
              run_id=run_id, **(detail or {}))

    def _is_stub_mode(self) -> bool:
        runner_orc = getattr(self.runner, "orc", None)
        return bool(getattr(self.orc, "_is_stub", False)
                    or getattr(runner_orc, "_is_stub", False))

    def _vet(self, j: Journal, rid: str, hk: str,
             item: dict) -> Optional[dict]:
        """Parse, screen and check one submission without running it.
        Returns what launching it takes, or None when it was turned away
        (and recorded so)."""
        sha = item["sha256"]
        try:
            sub = parse_submission(hk, json.loads(item["body"]))
        except (ValueError, TypeError) as e:
            self._record(j, rid, hk, sha, "refused",
                         {"reason": str(e)[:300]})
            return None

        try:
            verdict = self.screen(screen_text(sub))
        except ScreenUnavailable as e:
            self._record(j, rid, hk, sha, "unscreened",
                         {"reason": str(e)[:300]})
            return None
        if verdict.get("escape"):
            self._record(j, rid, hk, sha, "blocked",
                         {"reason": verdict.get("reason", ""),
                          "screen_model": verdict.get("model", "")})
            return None
        j.log("entry.screened", hotkey=hk, sha256=sha[:16],
              screen_model=verdict.get("model", ""))

        has_sealed = sub.get("key_sealed") is not None
        key: Optional[str] = None
        proxy_factory: Optional[Callable[[str], Any]] = None

        if has_sealed and self.keyd is not None:
            from .providers import KeyQuarantineError, ProviderError
            try:
                self.keyd.vet(hk, sub, skip_remote=self._is_stub_mode())
            except (KeyQuarantineError, ValueError, ProviderError) as e:
                self._record(j, rid, hk, sha, "refused",
                             {"reason": f"sealed key: {e}"[:300]})
                return None
            budget = float(sub["budget_credits"])
            proxy_factory = (
                lambda run_id, _hk=hk, _sub=dict(sub), _b=budget:
                self.keyd.proxy_for_run(run_id, _hk, _sub, _b)
            )
        elif has_sealed:
            if self.require_keyd:
                self._record(j, rid, hk, sha, "refused",
                             {"reason": "sealed key: loop-keyd unavailable"})
                return None
            from .providers import (KeyQuarantineError, ProviderError,
                                    quarantine_key)
            opener = self.keyring if self.keyring is not None else self.enc_seed
            try:
                key = open_submission_key(opener, hk, sub)
                if key and not self._is_stub_mode():
                    quarantine_key(key)
            except (KeyQuarantineError, ValueError, ProviderError) as e:
                self._record(j, rid, hk, sha, "refused",
                             {"reason": f"sealed key: {e}"[:300]})
                return None

        try:
            self.runner.check(hk, sub["source"], sub["args"],
                              sub["budget_credits"], api_key=key,
                              has_proxy=(proxy_factory is not None))
        except SubmitError as e:
            self._record(j, rid, hk, sha, "refused",
                         {"reason": str(e)[:300]})
            return None
        return {"sub": sub, "key": key, "proxy_factory": proxy_factory,
                "verdict": verdict}

    def _start_entry(self, j: Journal, rid: str, hk: str, item: dict,
                     vetted: dict, plan: Optional[dict] = None,
                     dataset: Optional[str] = None) -> Optional[dict]:
        """Launch one vetted submission on the round's terms.  Returns
        the running entry, or None when it was turned away."""
        sha, sub, verdict = item["sha256"], vetted["sub"], vetted["verdict"]
        key = vetted.pop("key", None)
        proxy_factory = vetted.pop("proxy_factory", None)
        try:
            run_id = self.runner.submit(
                hk, sub["source"], sub["args"], sub["budget_credits"],
                nonce=f"round-{sha[:24]}", api_key=key,
                proxy_factory=proxy_factory,
                max_seconds=(plan or {}).get("run_seconds"),
                dataset_text=dataset)
        except SubmitError as e:
            self._record(j, rid, hk, sha, "refused",
                         {"reason": str(e)[:300]})
            return None
        carry = {"funding": ("byok" if (key or proxy_factory)
                             else "credits"),
                 "screen_model": verdict.get("model", ""),
                 "screen_reason": str(verdict.get("reason", ""))[:300]}
        self._record(j, rid, hk, sha, "running", carry, run_id)
        self.runner.start(run_id)
        return {"run_id": run_id, "carry": carry}

    def _revive(self, j: Journal, hk: str, item: dict,
                run_id: str) -> bool:
        """A restarted auditor picks its round back up.  Credit runs
        resumed themselves; a parked BYOK run re-attaches through keyd
        (or opens fresh from the un-scrubbed paused inbox submission)."""
        run = self.ledger.sealed_run(run_id)
        if run is None:
            return False
        if run["status"] != "paused":
            return True                      # queued/running/ended: wait
        try:
            sub = parse_submission(hk, json.loads(item["body"]))
            if self.keyd is not None and sub.get("key_sealed") is not None:
                self.keyd.vet(hk, sub, skip_remote=self._is_stub_mode())
                info = self.ledger.run_info(run_id)
                self.runner.resume(
                    run_id,
                    proxy_factory=lambda rid, _hk=hk, _sub=dict(sub),
                    _b=info.budget_credits, _s=info.spent_credits:
                    self.keyd.proxy_for_run(rid, _hk, _sub, _b,
                                            prior_spent=_s))
            else:
                if self.require_keyd and sub.get("key_sealed") is not None:
                    raise ValueError("sealed key: loop-keyd unavailable")
                opener = (self.keyring if self.keyring is not None
                          else self.enc_seed)
                key = open_submission_key(opener, hk, sub)
                self.runner.resume(run_id, api_key=key)
            j.log("entry.revived", hotkey=hk, run_id=run_id)
            return True
        except Exception as e:  # noqa: BLE001
            j.log("entry.revive_failed", hotkey=hk, run_id=run_id,
                  error=str(e)[:300])
            return False

    def _finish_all(self, j: Journal, rid: str, subs: Dict[str, dict],
                    running: Dict[str, dict], deadline: float) -> None:
        """Wait out the round's runs and audit each one as it ends, a few
        at a time — in the order they end, so no audit waits on a run
        queued behind it."""
        pending = dict(running)
        settling = []
        with ThreadPoolExecutor(AUDIT_WORKERS,
                                thread_name_prefix="loop-audit") as ex:
            while pending:
                late = time.monotonic() > deadline
                for hk in sorted(pending):
                    info = pending[hk]
                    run = self.ledger.sealed_run(info["run_id"])
                    if (run is not None and not late
                            and run["status"] not in ("done", "failed")):
                        continue
                    del pending[hk]
                    settling.append(ex.submit(
                        self._settle, j, rid, hk, subs[hk]["sha256"],
                        info["run_id"], run, info["carry"]))
                if pending:
                    time.sleep(self.poll_seconds)
        for f in settling:          # an entry left unsettled: the round
            f.result()              # stays open and the next tick retries

    def _settle(self, j: Journal, rid: str, hk: str, sha: str, run_id: str,
                run: Optional[dict], carry: Optional[dict] = None
                ) -> Optional[Any]:
        """Record how one run ended, auditing what it produced.  ``carry``
        (funding, screen verdict) stays on the entry's final record."""
        carry = dict(carry or {})
        if run is None:
            self._record(j, rid, hk, sha, "failed",
                         {**carry, "reason": "run vanished"}, run_id)
            return None
        if run["status"] not in ("done", "failed"):
            self._record(j, rid, hk, sha, "failed",
                         {**carry, "reason": "round timed out waiting"},
                         run_id)
            return None

        if run["status"] == "failed":
            self._record(j, rid, hk, sha, "failed",
                         {**carry, "reason": ((run.get("detail") or {}).get(
                             "reason") or "")[:300]}, run_id)
            return None

        try:
            dossier = Dossier.from_dict(run["dossier"])
            report = self.panel.audit(dossier,
                                      local_evidence(self.ledger, dossier))
        except Exception as e:  # noqa: BLE001
            self._record(j, rid, hk, sha, "failed",
                         {**carry, "reason": f"audit error: {e}"[:300]},
                         run_id)
            return None
        try:
            spent = self.ledger.run_info(run_id).spent_credits
        except Exception:  # noqa: BLE001
            spent = None
        self._record(j, rid, hk, sha, "audited",
                     {**carry, "multiplier": report.multiplier,
                      "verdict_source": report.source,
                      "violations": [v[:200] for v in
                                     (report.violations or [])][:8],
                      "spent_credits": spent},
                     run_id)
        return report
