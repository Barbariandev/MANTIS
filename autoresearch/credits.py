# MIT License
# Copyright (c) 2024 MANTIS
"""Credit ledger for the LOOP autoresearch challenge.

Metering flow:

    run budgets (miner-set: a spend cap on the miner's own SayGM key)
        -> metered spends (SayGM inference / RunPod or local compute)
        -> HMAC-signed usage receipts (audit evidence + visuals)

1 credit = 1 USD of metered spend.  Miners pay for their runs with their
own SayGM (``gm_live_...``) key, so credits meter and cap a run rather
than fund it; the only balance that funds anything is what the owner
grants by hand (``python -m autoresearch.credits grant``).
On-chain deposits are retired.

Balances are stored as integer microcredits to avoid float drift.  The
ledger is append-only on grants and spends (grants live in the
``deposits`` table); the only mutation is the derived account balance
and per-run spent counters, both updated inside the same transaction as
the append.
"""

from __future__ import annotations

import functools
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .schema import UsageReceipt, new_id

logger = logging.getLogger(__name__)

MICRO = 1_000_000  # microcredits per credit


def _to_micro(credits: float) -> int:
    return int(round(float(credits) * MICRO))


def _from_micro(micro: int) -> float:
    return micro / MICRO


_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    hotkey      TEXT PRIMARY KEY,
    evm_address TEXT,
    balance     INTEGER NOT NULL DEFAULT 0,      -- microcredits
    updated_ts  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS deposits (
    tx_hash  TEXT PRIMARY KEY,
    hotkey   TEXT NOT NULL,
    raw_amount TEXT NOT NULL,                    -- wei, as string
    credits  INTEGER NOT NULL,                   -- microcredits granted
    block    INTEGER,
    ts       REAL NOT NULL,
    memo     TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id   TEXT PRIMARY KEY,
    hotkey   TEXT NOT NULL,
    token    TEXT NOT NULL,                      -- bearer token for the loop
    budget   INTEGER NOT NULL,                   -- microcredits reserved
    spent    INTEGER NOT NULL DEFAULT 0,
    status   TEXT NOT NULL DEFAULT 'open',       -- open | exhausted | closed
    created_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spends (
    receipt_id TEXT PRIMARY KEY,
    hotkey   TEXT NOT NULL,
    run_id   TEXT NOT NULL,
    kind     TEXT NOT NULL,                      -- llm | compute
    provider TEXT NOT NULL,
    item     TEXT NOT NULL,
    credits  INTEGER NOT NULL,                   -- microcredits
    ts       REAL NOT NULL,
    request_hash TEXT NOT NULL,
    meta     TEXT,
    sig      TEXT
);
CREATE TABLE IF NOT EXISTS events (
    run_id   TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    kind     TEXT NOT NULL,
    ts       REAL NOT NULL,
    parent_seq INTEGER,
    data     TEXT,
    PRIMARY KEY (run_id, seq)
);
-- Content-addressed public store: pipeline sources, datasets and the
-- request/response of every model call.
CREATE TABLE IF NOT EXISTS blobs (
    sha256   TEXT PRIMARY KEY,
    body     TEXT NOT NULL,
    ts       REAL NOT NULL
);
-- What the gateway attests about a sealed run (see sealed.py).
CREATE TABLE IF NOT EXISTS sealed_runs (
    run_id   TEXT PRIMARY KEY,
    hotkey   TEXT NOT NULL,
    nonce    TEXT NOT NULL,
    pipeline_sha256 TEXT NOT NULL,
    dataset_sha256  TEXT NOT NULL,
    args     TEXT NOT NULL,                      -- JSON
    status   TEXT NOT NULL,              -- queued|running|paused|done|failed
    isolation TEXT NOT NULL DEFAULT '',
    detail   TEXT NOT NULL DEFAULT '{}',         -- JSON: how it ended
    dossier  TEXT,                               -- JSON, once done
    created_ts REAL NOT NULL,
    ended_ts REAL,
    funding  TEXT NOT NULL DEFAULT 'credits',    -- credits|byok
    elapsed  REAL NOT NULL DEFAULT 0,            -- run seconds consumed
    paused_ts REAL,
    max_seconds REAL,                    -- its wall clock (NULL: default)
    UNIQUE (hotkey, nonce)
);
-- A pipeline's own saved state: within-run memory that survives a
-- pause (written only by the sealed pipeline itself, one slot per run).
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id   TEXT PRIMARY KEY,
    state    TEXT NOT NULL,                      -- JSON
    ts       REAL NOT NULL
);
-- Nonces burned outside run submission (resume/cancel requests), so a
-- signed request can never be replayed.
CREATE TABLE IF NOT EXISTS used_nonces (
    hotkey   TEXT NOT NULL,
    nonce    TEXT NOT NULL,
    ts       REAL NOT NULL,
    PRIMARY KEY (hotkey, nonce)
);
-- The 24h round system (autoresearch/rounds.py): one row per round and
-- one per (round, hotkey) entry, so a restarted auditor knows exactly
-- what was evaluated and what is still owed.
CREATE TABLE IF NOT EXISTS rounds (
    round_id TEXT PRIMARY KEY,                   -- "2026-11-13"
    started_ts  REAL NOT NULL,
    finished_ts REAL,
    detail   TEXT NOT NULL DEFAULT '{}',         -- JSON summary
    plan     TEXT NOT NULL DEFAULT '{}'          -- JSON: every run's terms
);
CREATE TABLE IF NOT EXISTS round_entries (
    round_id TEXT NOT NULL,
    hotkey   TEXT NOT NULL,
    sha256   TEXT NOT NULL,                      -- submission name
    status   TEXT NOT NULL,                      -- see rounds.py
    detail   TEXT NOT NULL DEFAULT '{}',         -- JSON (reasons etc.)
    run_id   TEXT,                               -- once it ran
    ts       REAL NOT NULL,
    PRIMARY KEY (round_id, hotkey)
);
-- Submissions the auditor took out of miners' payloads with the owner
-- key on arrival: private until each one's round closes and the
-- dashboard shows its code.  seen_ts (first read) picks the round.
CREATE TABLE IF NOT EXISTS loop_inbox (
    hotkey     TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    seen_ts    REAL NOT NULL,
    submission TEXT NOT NULL,                    -- JSON, key still sealed
    PRIMARY KEY (hotkey, sha256)
);
CREATE INDEX IF NOT EXISTS idx_loop_inbox_seen ON loop_inbox(seen_ts);
CREATE INDEX IF NOT EXISTS idx_sealed_hotkey ON sealed_runs(hotkey, created_ts);
CREATE INDEX IF NOT EXISTS idx_spends_hotkey ON spends(hotkey, ts);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
"""


class BudgetExceeded(Exception):
    pass


class UnknownRun(Exception):
    pass


@dataclass
class RunInfo:
    run_id: str
    hotkey: str
    token: str
    budget_credits: float
    spent_credits: float
    status: str


def _locked(method):
    """Run a ledger method holding the ledger lock."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class CreditLedger:
    """SQLite-backed credit accounting.  Thread-safe via one lock —
    gateway traffic is metered API calls, not a hot path.

    The lock covers reads as well as writes.  All threads share one
    connection, and two statements running on it at once can hand a
    reader another query's row: a valid run token read back as "bad",
    a funded run as "exhausted".
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # columns added after launch (CREATE IF NOT EXISTS cannot):
        for table, ddls in (
                ("sealed_runs", ("funding TEXT NOT NULL DEFAULT 'credits'",
                                 "elapsed REAL NOT NULL DEFAULT 0",
                                 "paused_ts REAL", "max_seconds REAL")),
                ("rounds", ("plan TEXT NOT NULL DEFAULT '{}'",))):
            have = {r[1] for r in self._conn.execute(
                f"PRAGMA table_info({table})")}
            for ddl in ddls:
                if ddl.split()[0] not in have:
                    with self._conn:
                        self._conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {ddl}")
        self._lock = threading.RLock()

    # ------------------------------------------------------------ accounts

    def register(self, hotkey: str, evm_address: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO accounts (hotkey, evm_address, balance, updated_ts) "
                "VALUES (?, ?, 0, ?) "
                "ON CONFLICT(hotkey) DO UPDATE SET "
                "evm_address=excluded.evm_address, updated_ts=excluded.updated_ts",
                (hotkey, evm_address.lower(), time.time()),
            )

    @_locked
    def balance(self, hotkey: str) -> float:
        row = self._conn.execute(
            "SELECT balance FROM accounts WHERE hotkey=?", (hotkey,)
        ).fetchone()
        return _from_micro(row[0]) if row else 0.0

    @_locked
    def funding_address(self, hotkey: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT evm_address FROM accounts WHERE hotkey=?", (hotkey,)
        ).fetchone()
        return row[0] if row else None

    @_locked
    def address_to_hotkey(self) -> Dict[str, str]:
        return {addr: hk for hk, addr in self._conn.execute(
            "SELECT hotkey, evm_address FROM accounts WHERE evm_address IS NOT NULL"
        )}

    # ------------------------------------------------------------ credit in

    def credit_deposit(self, tx_hash: str, hotkey: str, raw_amount: str,
                       credits: float, block: Optional[int] = None,
                       memo: str = "") -> bool:
        """Idempotent by tx_hash; returns True when newly credited."""
        micro = _to_micro(credits)
        if micro <= 0:
            return False
        with self._lock, self._conn:
            try:
                self._conn.execute(
                    "INSERT INTO deposits (tx_hash, hotkey, raw_amount, credits, block, ts, memo) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (tx_hash, hotkey, str(raw_amount), micro, block,
                     time.time(), memo),
                )
            except sqlite3.IntegrityError:
                return False
            self._conn.execute(
                "INSERT INTO accounts (hotkey, balance, updated_ts) VALUES (?, ?, ?) "
                "ON CONFLICT(hotkey) DO UPDATE SET "
                "balance = balance + excluded.balance, updated_ts=excluded.updated_ts",
                (hotkey, micro, time.time()),
            )
        logger.info("deposit credited: %s -> %s (%.4f credits)",
                    tx_hash[:12], hotkey[:12], credits)
        return True

    def grant(self, hotkey: str, credits: float, memo: str = "owner_grant") -> None:
        """Owner path (testing / promotional credit), stored as a
        synthetic deposit row."""
        self.credit_deposit(new_id("grant"), hotkey, "0", credits, memo=memo)

    # ------------------------------------------------------------ runs

    def open_run(self, hotkey: str, budget_credits: float) -> RunInfo:
        """Miner-set budget.  Reserves nothing upfront; the budget is a
        hard per-run spend cap and each debit also checks the live
        account balance, so an account can safely run several loops."""
        budget = _to_micro(budget_credits)
        if budget <= 0:
            raise ValueError("budget must be positive")
        run_id, token = new_id("run"), new_id("tok")
        with self._lock, self._conn:
            bal = self._conn.execute(
                "SELECT balance FROM accounts WHERE hotkey=?", (hotkey,)
            ).fetchone()
            if not bal or bal[0] <= 0:
                raise BudgetExceeded("no credit balance: bring your own "
                                     "SayGM key")
            self._conn.execute(
                "INSERT INTO runs (run_id, hotkey, token, budget, created_ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, hotkey, token, budget, time.time()),
            )
        return RunInfo(run_id, hotkey, token,
                       _from_micro(budget), 0.0, "open")

    @_locked
    def run_info(self, run_id: str) -> RunInfo:
        row = self._conn.execute(
            "SELECT run_id, hotkey, token, budget, spent, status FROM runs "
            "WHERE run_id=?", (run_id,)
        ).fetchone()
        if not row:
            raise UnknownRun(run_id)
        return RunInfo(row[0], row[1], row[2], _from_micro(row[3]),
                       _from_micro(row[4]), row[5])

    @_locked
    def check_token(self, run_id: str, token: str) -> RunInfo:
        info = self.run_info(run_id)
        if info.token != token:
            raise PermissionError("bad run token")
        return info

    def close_run(self, run_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET status='closed' WHERE run_id=?", (run_id,))

    # ------------------------------------------------------------ spends

    def debit(self, receipt: UsageReceipt, force: bool = False) -> None:
        """Atomically record a spend and decrement balance + run budget.

        ``force`` books the spend even on a run that is already exhausted
        or closed: for costs that are only known once the work is over
        (a pipeline's running time).

        A metered cost can only be known *after* the provider call (actual
        token usage / runtime), so the debit may overshoot the remaining
        budget by at most one call; the run is then marked exhausted and
        further spends are refused.  Pre-call gating happens in the
        gateway (refuses calls on exhausted runs / empty balances).
        """
        micro = _to_micro(receipt.credits)
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT hotkey, budget, spent, status FROM runs WHERE run_id=?",
                (receipt.run_id,),
            ).fetchone()
            if not row:
                raise UnknownRun(receipt.run_id)
            hotkey, budget, spent, status = row
            if status != "open" and not force:
                raise BudgetExceeded(f"run {receipt.run_id} is {status}")
            self._conn.execute(
                "INSERT INTO spends (receipt_id, hotkey, run_id, kind, provider, "
                "item, credits, ts, request_hash, meta, sig) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (receipt.receipt_id, hotkey, receipt.run_id, receipt.kind,
                 receipt.provider, receipt.item, micro, receipt.ts,
                 receipt.request_hash,
                 __import__("json").dumps(receipt.meta), receipt.sig),
            )
            new_spent = spent + micro
            exhausted = new_spent >= budget
            self._conn.execute(
                "UPDATE runs SET spent=?, status=? WHERE run_id=?",
                (new_spent, status if status != "open" else
                 ("exhausted" if exhausted else "open"), receipt.run_id),
            )
            self._conn.execute(
                "UPDATE accounts SET balance = MAX(balance - ?, 0), updated_ts=? "
                "WHERE hotkey=?",
                (micro, time.time(), hotkey),
            )

    @_locked
    def can_spend(self, run_id: str) -> bool:
        try:
            info = self.run_info(run_id)
        except UnknownRun:
            return False
        return (info.status == "open"
                and self.balance(info.hotkey) > 0.0
                and info.spent_credits < info.budget_credits)

    @_locked
    def receipts(self, hotkey: Optional[str] = None,
                 run_id: Optional[str] = None) -> List[dict]:
        import json as _json
        q = ("SELECT receipt_id, hotkey, run_id, kind, provider, item, "
             "credits, ts, request_hash, meta, sig FROM spends")
        cond, args = [], []
        if hotkey:
            cond.append("hotkey=?"); args.append(hotkey)
        if run_id:
            cond.append("run_id=?"); args.append(run_id)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY ts"
        out = []
        for r in self._conn.execute(q, args):
            out.append({
                "receipt_id": r[0], "hotkey": r[1], "run_id": r[2],
                "kind": r[3], "provider": r[4], "item": r[5],
                "credits": _from_micro(r[6]), "ts": r[7],
                "request_hash": r[8], "meta": _json.loads(r[9] or "{}"),
                "sig": r[10],
            })
        return out

    # ------------------------------------------------------------ events

    MAX_EVENTS_PER_RUN = 2000
    MAX_EVENT_BYTES = 8192

    def append_events(self, run_id: str, events: List[dict]) -> int:
        """Store loop telemetry (free; size- and count-capped).  Returns
        the number of events actually stored."""
        import json as _json
        stored = 0
        with self._lock, self._conn:
            (n,) = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)
            ).fetchone()
            for ev in events:
                if n + stored >= self.MAX_EVENTS_PER_RUN:
                    break
                data = _json.dumps(ev.get("data") or {})
                if len(data) > self.MAX_EVENT_BYTES:
                    continue
                try:
                    self._conn.execute(
                        "INSERT INTO events (run_id, seq, kind, ts, parent_seq, data) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, int(ev["seq"]), str(ev["kind"]),
                         float(ev.get("ts", time.time())),
                         ev.get("parent_seq"), data),
                    )
                    stored += 1
                except (sqlite3.IntegrityError, KeyError, ValueError, TypeError):
                    continue
        return stored

    @_locked
    def events_for_run(self, run_id: str) -> List[dict]:
        import json as _json
        return [{
            "run_id": run_id, "seq": r[0], "kind": r[1], "ts": r[2],
            "parent_seq": r[3], "data": _json.loads(r[4] or "{}"),
        } for r in self._conn.execute(
            "SELECT seq, kind, ts, parent_seq, data FROM events "
            "WHERE run_id=? ORDER BY seq", (run_id,))]

    # -------------------------------------------------------------- blobs

    def put_blob(self, body: str) -> str:
        """Store ``body`` under its sha256 and return the hash."""
        import hashlib
        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO blobs (sha256, body, ts) "
                "VALUES (?, ?, ?)", (sha, body, time.time()))
        return sha

    @_locked
    def get_blob(self, sha256: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT body FROM blobs WHERE sha256=?", (sha256,)).fetchone()
        return row[0] if row else None

    # -------------------------------------------------------- sealed runs

    _SEALED_COLS = ("run_id", "hotkey", "pipeline_sha256", "dataset_sha256",
                    "args", "status", "isolation", "detail", "dossier",
                    "created_ts", "ended_ts", "funding", "elapsed",
                    "paused_ts", "max_seconds")

    def open_sealed_run(self, hotkey: str, budget_credits: float,
                        nonce: str, pipeline_sha256: str,
                        dataset_sha256: str, args: dict,
                        funding: str = "credits",
                        max_seconds: Optional[float] = None) -> RunInfo:
        """Open a run and its attestation record together.  Raises
        ``ValueError`` if this hotkey already used ``nonce``."""
        import json as _json
        with self._lock:
            if self._nonce_used(hotkey, nonce):
                raise ValueError("nonce already used")
            info = self.open_run(hotkey, budget_credits)
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO sealed_runs (run_id, hotkey, nonce, "
                        "pipeline_sha256, dataset_sha256, args, status, "
                        "created_ts, funding, max_seconds) "
                        "VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
                        (info.run_id, hotkey, nonce, pipeline_sha256,
                         dataset_sha256, _json.dumps(args, sort_keys=True),
                         time.time(), funding, max_seconds))
            except sqlite3.IntegrityError:
                with self._conn:
                    self._conn.execute(
                        "DELETE FROM runs WHERE run_id=?", (info.run_id,))
                raise ValueError("nonce already used")
        return info

    def _nonce_used(self, hotkey: str, nonce: str) -> bool:
        return bool(self._conn.execute(
            "SELECT 1 FROM sealed_runs WHERE hotkey=? AND nonce=? "
            "UNION SELECT 1 FROM used_nonces WHERE hotkey=? AND nonce=?",
            (hotkey, nonce, hotkey, nonce)).fetchone())

    def claim_nonce(self, hotkey: str, nonce: str) -> None:
        """Burn ``nonce`` for ``hotkey`` (resume/cancel requests).
        Raises ``ValueError`` if it was ever used — by a submission or
        by an earlier request — so no signed request can be replayed."""
        with self._lock:
            if self._nonce_used(hotkey, nonce):
                raise ValueError("nonce already used")
            with self._conn:
                self._conn.execute(
                    "INSERT INTO used_nonces (hotkey, nonce, ts) "
                    "VALUES (?, ?, ?)", (hotkey, nonce, time.time()))

    def put_checkpoint(self, run_id: str, state: str) -> None:
        """The pipeline's own saved state; one slot, overwritten."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints (run_id, state, ts) "
                "VALUES (?, ?, ?)", (run_id, state, time.time()))

    @_locked
    def get_checkpoint(self, run_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT state FROM checkpoints WHERE run_id=?",
            (run_id,)).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------- rounds

    def open_round(self, round_id: str) -> bool:
        """Claim a round for evaluation.  False if it was ever opened —
        a restarted auditor continues it instead of starting over."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO rounds (round_id, started_ts) "
                "VALUES (?, ?)", (round_id, time.time()))
            return cur.rowcount == 1

    def finish_round(self, round_id: str, detail: dict) -> None:
        import json as _json
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE rounds SET finished_ts=?, detail=? "
                "WHERE round_id=?",
                (time.time(), _json.dumps(detail, sort_keys=True),
                 round_id))

    def set_round_plan(self, round_id: str, plan: dict) -> None:
        """A round's terms, fixed when it opens (``rounds.py``)."""
        import json as _json
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE rounds SET plan=? WHERE round_id=?",
                (_json.dumps(plan, sort_keys=True), round_id))

    @_locked
    def round(self, round_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT round_id, started_ts, finished_ts, detail, plan "
            "FROM rounds WHERE round_id=?", (round_id,)).fetchone()
        return self._round_row(row) if row else None

    @_locked
    def rounds(self, limit: int = 60) -> List[dict]:
        return [self._round_row(r) for r in self._conn.execute(
            "SELECT round_id, started_ts, finished_ts, detail, plan "
            "FROM rounds ORDER BY round_id DESC LIMIT ?", (int(limit),))]

    @staticmethod
    def _round_row(row) -> dict:
        import json as _json
        return {"round_id": row[0], "started_ts": row[1],
                "finished_ts": row[2],
                "detail": _json.loads(row[3] or "{}"),
                "plan": _json.loads(row[4] or "{}")}

    def record_round_entry(self, round_id: str, hotkey: str, sha256: str,
                           status: str, detail: Optional[dict] = None,
                           run_id: Optional[str] = None) -> None:
        import json as _json
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO round_entries "
                "(round_id, hotkey, sha256, status, detail, run_id, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(round_id, hotkey) DO UPDATE SET "
                "status=excluded.status, detail=excluded.detail, "
                "run_id=COALESCE(excluded.run_id, run_id), "
                "ts=excluded.ts",
                (round_id, hotkey, sha256, status,
                 _json.dumps(detail or {}, sort_keys=True), run_id,
                 time.time()))

    @_locked
    def round_entries(self, round_id: Optional[str] = None) -> List[dict]:
        import json as _json
        where = "WHERE round_id=?" if round_id else ""
        rows = self._conn.execute(
            "SELECT round_id, hotkey, sha256, status, detail, run_id, ts "
            f"FROM round_entries {where} ORDER BY round_id DESC, hotkey",
            (round_id,) if round_id else ()).fetchall()
        return [{"round_id": r[0], "hotkey": r[1], "sha256": r[2],
                 "status": r[3], "detail": _json.loads(r[4] or "{}"),
                 "run_id": r[5], "ts": r[6]} for r in rows]

    def inbox_add(self, hotkey: str, sha256: str, submission: str,
                  seen_ts: Optional[float] = None) -> bool:
        """Keep a submission the first time it is read; True if new.
        Stamped under the lock: a round reading its window after the
        boundary sees everything stamped before it."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO loop_inbox "
                "(hotkey, sha256, seen_ts, submission) VALUES (?, ?, ?, ?)",
                (hotkey, sha256,
                 time.time() if seen_ts is None else float(seen_ts),
                 submission))
            return cur.rowcount == 1

    @_locked
    def inbox_window(self, start_ts: float, end_ts: float) -> Dict[str, dict]:
        """hotkey -> its newest submission first read inside the window."""
        out: Dict[str, dict] = {}
        for hk, sha, body, ts in self._conn.execute(
                "SELECT hotkey, sha256, submission, seen_ts FROM loop_inbox "
                "WHERE seen_ts >= ? AND seen_ts < ? ORDER BY seen_ts, sha256",
                (start_ts, end_ts)):
            out[hk] = {"hotkey": hk, "sha256": sha, "body": body, "ts": ts}
        return out

    @_locked
    def inbox_submission(self, hotkey: str, sha256: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT submission FROM loop_inbox WHERE hotkey=? AND sha256=?",
            (hotkey, sha256)).fetchone()
        return row[0] if row else None

    def scrub_inbox_key(self, hotkey: str, sha256: str) -> bool:
        """Strip ``key_sealed`` / ``key_id`` from ``loop_inbox`` once a
        submission has reached a terminal state so no sealed key blob
        persists on disk after its run ends."""
        from .submission import strip_sealed_key_fields
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT submission FROM loop_inbox "
                "WHERE hotkey=? AND sha256=?", (hotkey, sha256)).fetchone()
            if not row or not row[0]:
                return False
            cleaned = strip_sealed_key_fields(row[0])
            if cleaned == row[0]:
                return False
            self._conn.execute(
                "UPDATE loop_inbox SET submission=? "
                "WHERE hotkey=? AND sha256=?",
                (cleaned, hotkey, sha256))
            return True

    def update_sealed_run(self, run_id: str, **fields) -> None:
        import json as _json
        allowed = {"status", "isolation", "detail", "dossier", "ended_ts",
                   "elapsed", "paused_ts"}
        if not fields or set(fields) - allowed:
            raise ValueError(f"cannot update {sorted(fields)}")
        for key in ("detail", "dossier"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = _json.dumps(fields[key], sort_keys=True)
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE sealed_runs SET {sets} WHERE run_id=?",
                (*fields.values(), run_id))

    def _sealed_row(self, row) -> dict:
        import json as _json
        d = dict(zip(self._SEALED_COLS, row))
        for key in ("args", "detail", "dossier"):
            d[key] = _json.loads(d[key]) if d[key] else None
        return d

    @_locked
    def sealed_run(self, run_id: str) -> Optional[dict]:
        row = self._conn.execute(
            f"SELECT {', '.join(self._SEALED_COLS)} FROM sealed_runs "
            "WHERE run_id=?", (run_id,)).fetchone()
        return self._sealed_row(row) if row else None

    @_locked
    def sealed_runs(self, hotkey: Optional[str] = None,
                    status: Optional[str] = None) -> List[dict]:
        q = f"SELECT {', '.join(self._SEALED_COLS)} FROM sealed_runs"
        cond, args = [], []
        if hotkey:
            cond.append("hotkey=?"); args.append(hotkey)
        if status:
            cond.append("status=?"); args.append(status)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        return [self._sealed_row(r) for r in
                self._conn.execute(q + " ORDER BY created_ts", args)]

    # ------------------------------------------------------------ network

    @_locked
    def network_summary(self) -> dict:
        """Aggregates that power the public progress visuals."""
        c = self._conn
        (n_accounts,) = c.execute("SELECT COUNT(*) FROM accounts").fetchone()
        (n_runs,) = c.execute("SELECT COUNT(*) FROM runs").fetchone()
        (spent,) = c.execute("SELECT COALESCE(SUM(credits),0) FROM spends").fetchone()
        (deposited,) = c.execute("SELECT COALESCE(SUM(credits),0) FROM deposits").fetchone()
        by_kind = dict(c.execute(
            "SELECT kind, COUNT(*) FROM events GROUP BY kind"))
        by_provider = {k: _from_micro(v) for k, v in c.execute(
            "SELECT provider, COALESCE(SUM(credits),0) FROM spends GROUP BY provider")}
        top_models = [
            {"item": r[0], "credits": _from_micro(r[1]), "calls": r[2]}
            for r in c.execute(
                "SELECT item, COALESCE(SUM(credits),0), COUNT(*) FROM spends "
                "WHERE kind='llm' GROUP BY item ORDER BY 2 DESC LIMIT 20")
        ]
        return {
            "accounts": n_accounts,
            "runs": n_runs,
            "credits_deposited": _from_micro(deposited),
            "credits_spent": _from_micro(spent),
            "events_by_kind": by_kind,
            "spend_by_provider": by_provider,
            "top_models": top_models,
        }


def _cli():
    import argparse
    import config as mantis_config

    p = argparse.ArgumentParser(description="LOOP credit ledger admin")
    p.add_argument("--db", default=os.path.join(
        mantis_config.STORAGE_DIR, "loop_credits.db"))
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grant"); g.add_argument("hotkey"); g.add_argument("credits", type=float)
    b = sub.add_parser("balance"); b.add_argument("hotkey")
    sub.add_parser("summary")
    args = p.parse_args()

    ledger = CreditLedger(args.db)
    if args.cmd == "grant":
        ledger.grant(args.hotkey, args.credits)
        print(f"granted {args.credits} credits -> balance "
              f"{ledger.balance(args.hotkey):.4f}")
    elif args.cmd == "balance":
        print(f"{ledger.balance(args.hotkey):.6f}")
    elif args.cmd == "summary":
        import json
        print(json.dumps(ledger.network_summary(), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()
