from __future__ import annotations
"""
MIT License

Copyright (c) 2024 MANTIS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import asyncio, json, logging, os, hashlib, sqlite3, time
import requests
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np, aiohttp, bittensor as bt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from timelock import Timelock
import config
from range_breakout import RangeBreakoutTracker

logger = logging.getLogger(__name__)
SAMPLE_EVERY = config.SAMPLE_EVERY

DRAND_SIGNATURE_RETRIES = 3
DRAND_SIGNATURE_RETRY_DELAY = 1.0
# A matured payload whose drand round cannot be fetched is RETAINED and
# retried on later passes rather than consumed as zeros: a transient
# beacon/network outage must not destroy submissions.  The grace window
# bounds the retry so a payload carrying a garbage round number cannot
# pin raw_payloads forever (one extra week past maturity, in blocks).
DRAND_RETAIN_GRACE_BLOCKS = 50_400

# Storage dim for MULTIBREAKOUT: 2 floats per asset, flattened across all BREAKOUT_ASSETS.
# config.MULTI_BREAKOUT_CHALLENGE["dim"] stays 2 (the per-asset dimension) but storage
# needs the full vector so per-asset predictions survive the pack/unpack round-trip.
_MB_STORAGE_DIM = 2 * len(config.BREAKOUT_ASSETS)
def _get_storage_dim(ticker: str) -> int:
    spec = config.CHALLENGE_MAP.get(ticker)
    if not spec:
        return config.ASSET_EMBEDDING_DIMS.get(ticker, 0)
    assets = spec.get("assets")
    if assets:
        return spec["dim"] * len(assets)
    return spec["dim"]


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS blocks (
    idx INTEGER PRIMARY KEY,
    block INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS challenge_meta (
    ticker TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    blocks_ahead INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS challenge_data (
    ticker TEXT NOT NULL,
    sidx INTEGER NOT NULL,
    price REAL,
    price_data TEXT,
    hotkeys TEXT,
    embeddings BLOB,
    PRIMARY KEY (ticker, sidx)
);
CREATE TABLE IF NOT EXISTS raw_payloads (
    ts INTEGER NOT NULL,
    hotkey TEXT NOT NULL,
    payload BLOB,
    PRIMARY KEY (ts, hotkey)
);
CREATE TABLE IF NOT EXISTS drand_cache (
    round INTEGER PRIMARY KEY,
    signature BLOB
);
CREATE TABLE IF NOT EXISTS breakout_state (
    asset TEXT PRIMARY KEY,
    state_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_blocks_block ON blocks(block);
"""


def _ensure_price_data_col(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(challenge_data)")}
    if "price_data" not in cols:
        conn.execute("ALTER TABLE challenge_data ADD COLUMN price_data TEXT")
        conn.commit()


def _emb_dtype(ticker: str):
    # FLOW carries an integer trade_id and horizons up to 336 in its
    # embedding; float16 represents integers exactly only to 2048, so
    # ids above that collide (2049->2048) or overflow.  FLOW is stored
    # float32 (exact integers to 2**24); every other challenge, whose
    # values live in [-1,1], stays float16.
    return np.float32 if ticker == "FLOW" else np.float16


def _pack_embeddings(emb: Dict[str, np.ndarray],
                     dtype=np.float16) -> bytes:
    if not emb:
        return b""
    hk_list = sorted(emb.keys())
    vecs = np.array([np.asarray(emb[hk], dtype=dtype) for hk in hk_list],
                    dtype=dtype)
    return json.dumps(hk_list).encode() + b"\x00" + vecs.tobytes()


def _unpack_embeddings(blob: bytes, dim: int) -> Dict[str, np.ndarray]:
    if not blob:
        return {}
    sep = blob.index(b"\x00")
    hk_list = json.loads(blob[:sep].decode())
    raw = blob[sep + 1:]
    n = len(hk_list)
    if n == 0 or dim == 0:
        return {}
    # self-describing by byte length: float16 (2B/val) or float32 (4B),
    # so the stored width need not be threaded through every reader
    if len(raw) == n * dim * 2:
        dtype = np.float16
    elif len(raw) == n * dim * 4:
        dtype = np.float32
    else:
        return {}
    vecs = np.frombuffer(raw, dtype=dtype).reshape(n, dim)
    return {hk: vecs[i].copy() for i, hk in enumerate(hk_list)}


def _sqlite_ok(path: str) -> bool:
    """True iff the file opens as SQLite and passes a quick integrity
    check.  Used for publish snapshots only; downloads are deliberately
    NOT gated on this (the live prod archive carries known transient
    corruption in the payloads region that quick_check would flag)."""
    try:
        conn = sqlite3.connect(path)
        try:
            row = conn.execute("PRAGMA quick_check").fetchone()
            return bool(row) and row[0] == "ok"
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def ensure_datalog(path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = config.DATALOG_ARCHIVE_URL
    r = requests.get(url, timeout=1500, stream=True)
    if r.status_code == 200:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, path)
        return path
    raise SystemExit(f"Failed to download datalog from {url}")


# --------------------------------------------------------------------------
# FLOW storage split.  FLOW challenge_data lives in its own SQLite
# file next to the main datalog, published to the same bucket as a much
# smaller object (the legacy datalog has grown too large to keep
# re-shipping for one challenge).  The file is ATTACHed to every
# connection as schema `inv`, so all existing SQL keeps working with the
# table name routed through _cd_table().  Shared state (blocks,
# raw_payloads, drand_cache) stays in the main DB: a raw payload is one
# blob covering every challenge, so it cannot be split per ticker.

_FLOW_DB_TICKERS = frozenset({"FLOW"})
_FLOW_DB_FILENAME = "flow_datalog.db"

_FLOW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS challenge_data (
    ticker TEXT NOT NULL,
    sidx INTEGER NOT NULL,
    price REAL,
    price_data TEXT,
    hotkeys TEXT,
    embeddings BLOB,
    PRIMARY KEY (ticker, sidx)
);
"""


def _cd_table(ticker: str) -> str:
    """challenge_data table for a ticker: the attached FLOW DB or main."""
    return ("inv.challenge_data" if ticker in _FLOW_DB_TICKERS
            else "challenge_data")


def flow_db_path(db_path: str) -> str:
    """The FLOW DB sits next to the main datalog file."""
    return os.path.join(os.path.dirname(db_path) or ".", _FLOW_DB_FILENAME)


def ensure_flow_datalog(path: str) -> str:
    """Download the FLOW archive if the local file is missing.

    Non-fatal on failure: the challenge is young and a fresh, empty
    local file is a valid starting point (unlike the main datalog,
    where history is load-bearing for the other tickers).
    """
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = getattr(config, "FLOW_DATALOG_ARCHIVE_URL", "")
    if url:
        try:
            r = requests.get(url, timeout=1500, stream=True)
            if r.status_code == 200:
                tmp = path + ".tmp"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, path)
                return path
            logger.warning("FLOW datalog download failed (HTTP %s); "
                           "starting empty", r.status_code)
        except Exception as e:  # noqa: BLE001
            logger.warning("FLOW datalog download failed (%s); "
                           "starting empty", e)
    return path


def _attach_flow(conn, db_path: str) -> str:
    """Create the FLOW DB if needed and ATTACH it as `inv`.

    A corrupt side file is quarantined (renamed aside) and replaced
    with a fresh, empty one instead of crashing the validator: the
    FLOW DB is young and re-downloadable, unlike the main datalog
    where history is load-bearing and corruption must fail loudly."""
    inv_path = flow_db_path(db_path)

    def _prepare() -> None:
        side = sqlite3.connect(inv_path)
        try:
            side.execute("PRAGMA journal_mode=WAL")
            side.executescript(_FLOW_SCHEMA_SQL)
            side.commit()
        finally:
            side.close()

    try:
        _prepare()
    except sqlite3.DatabaseError:
        quarantine = f"{inv_path}.corrupt-{int(time.time())}"
        os.replace(inv_path, quarantine)
        logger.warning("FLOW datalog corrupt; quarantined to %s and "
                       "recreated empty", quarantine)
        _prepare()
    conn.execute("ATTACH DATABASE ? AS inv", (inv_path,))
    return inv_path


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def _hkdf_key_nonce(shared_secret: bytes, info: bytes = b"mantis-owner-wrap", key_len: int = 32, nonce_len: int = 12):
    out = HKDF(algorithm=hashes.SHA256(), length=key_len + nonce_len, salt=None, info=info).derive(shared_secret)
    return out[:key_len], out[key_len:]


def _binding(hk: str, rnd: int, owner_pk: bytes, pke: bytes) -> bytes:
    return _sha256(hk.encode("utf-8"), b":", str(rnd).encode("ascii"), b":", owner_pk, b":", pke)


def _derive_pke(ske_raw: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(ske_raw).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _convert_tlock_ct(ct_bytes: bytes) -> bytes:
    """372-byte ark-serialize 0.4 TLECiphertext -> 356-byte 0.5 form."""
    import struct
    if len(ct_bytes) != 372:
        return ct_bytes
    v_len = struct.unpack_from('<Q', ct_bytes, 96)[0]
    w_len = struct.unpack_from('<Q', ct_bytes, 96 + 8 + 32)[0]
    if v_len != 32 or w_len != 32:
        return ct_bytes
    u = ct_bytes[0:96]
    v = ct_bytes[104:136]
    w = ct_bytes[144:176]
    rest = ct_bytes[176:]
    return u + v + w + rest


def _expand_tlock_ct(ct_bytes: bytes) -> bytes:
    """356-byte ark-serialize 0.5 TLECiphertext -> 372-byte 0.4 form."""
    import struct
    if len(ct_bytes) != 356:
        return ct_bytes
    u = ct_bytes[0:96]
    v = ct_bytes[96:128]
    w = ct_bytes[128:160]
    rest = ct_bytes[160:]
    return u + struct.pack('<Q', 32) + v + struct.pack('<Q', 32) + w + rest


def _tld_skeK(tlock: Timelock, ct_bytes: bytes, sig: bytes):
    """Unlock W_time against whichever wasm this process loaded.

    Later timelock wants the 356-byte form and *panics* (not a Python
    exception) on a raw 372-byte blob.  Older wasm wants the 372-byte
    form and raises on 356.  Always try 356 first so a new wasm never
    sees the 372; fall back to 372 (native or expanded) so an old
    install can still unlock live 356-byte miner payloads.
    """
    form_356 = _convert_tlock_ct(ct_bytes)
    form_372 = _expand_tlock_ct(ct_bytes)
    last = None
    seen: set[bytes] = set()
    for ct in (form_356, form_372):
        if ct in seen:
            continue
        seen.add(ct)
        try:
            return tlock.tld(ct, sig)
        except Exception as e:
            last = e
    if last is not None:
        raise last
    return None


def _decrypt_v2_payload(payload: dict, sig: bytes | None, tlock: Timelock) -> bytes | None:
    try:
        if not sig:
            return None
        configured_owner_pk_hex = getattr(config, "OWNER_HPKE_PUBLIC_KEY_HEX", "").strip()
        if not configured_owner_pk_hex:
            return None
        payload_owner_pk_hex = payload.get("owner_pk")
        if isinstance(payload_owner_pk_hex, str) and payload_owner_pk_hex.lower() != configured_owner_pk_hex.lower():
            return None
        owner_pk = bytes.fromhex(configured_owner_pk_hex)
        pke = bytes.fromhex(payload["W_owner"]["pke"])
        binding = _binding(payload["hk"], int(payload["round"]), owner_pk, pke)
        if binding != bytes.fromhex(payload["binding"]):
            return None
        skeK_raw = _tld_skeK(
            tlock, bytes.fromhex(payload["W_time"]["ct"]), sig)
        if isinstance(skeK_raw, str):
            try:
                skeK = bytes.fromhex(skeK_raw)
            except ValueError:
                skeK = skeK_raw.encode("utf-8")
        else:
            skeK = bytes(skeK_raw)
            if len(skeK) == 128:
                try:
                    skeK = bytes.fromhex(skeK.decode("ascii"))
                except (UnicodeDecodeError, ValueError):
                    pass
        if len(skeK) != 64:
            return None
        ske, key = skeK[:32], skeK[32:]
        if _derive_pke(ske) != pke:
            return None
        shared = X25519PrivateKey.from_private_bytes(ske).exchange(X25519PublicKey.from_public_bytes(owner_pk))
        k1, _ = _hkdf_key_nonce(shared, info=b"mantis-owner-wrap")
        nonce = bytes.fromhex(payload["W_owner"]["nonce"])
        wrapped = ChaCha20Poly1305(k1).decrypt(nonce, bytes.fromhex(payload["W_owner"]["ct"]), binding)
        if wrapped != key:
            return None
        return ChaCha20Poly1305(key).decrypt(
            bytes.fromhex(payload["C"]["nonce"]),
            bytes.fromhex(payload["C"]["ct"]),
            binding,
        )
    except Exception:
        return None


@dataclass
class ChallengeData:
    dim: int
    blocks_ahead: int = 0
    sidx: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    def set_price(self, sidx: int, price: float):
        d = self.sidx.setdefault(sidx, {"hotkeys": [], "price": None, "emb": {}})
        d["price"] = float(price)
    def set_emb(self, sidx: int, hk: str, vec: List[float]):
        d = self.sidx.setdefault(sidx, {"hotkeys": [], "price": None, "emb": {}})
        d["emb"][hk] = np.array(vec, dtype=np.float16)
        if hk not in d["hotkeys"]:
            d["hotkeys"].append(hk)


class DataLog:
    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)

        _ensure_price_data_col(self._conn)
        self._flow_db_path = _attach_flow(self._conn, db_path)

        # one-time migration: any FLOW rows written to the main DB
        # before the storage split move to the attached DB (idempotent)
        n_stray = self._conn.execute(
            "SELECT COUNT(*) FROM challenge_data WHERE ticker='FLOW'"
        ).fetchone()[0]
        if n_stray:
            self._conn.execute(
                "INSERT OR REPLACE INTO inv.challenge_data "
                "(ticker, sidx, price, price_data, hotkeys, embeddings) "
                "SELECT ticker, sidx, price, price_data, hotkeys, embeddings "
                "FROM challenge_data WHERE ticker='FLOW'")
            self._conn.execute(
                "DELETE FROM challenge_data WHERE ticker='FLOW'")
            self._conn.commit()
            logger.info("Migrated %d FLOW rows to %s (storage split)",
                        n_stray, self._flow_db_path)

        for spec in config.CHALLENGES:
            self._conn.execute(
                "INSERT OR REPLACE INTO challenge_meta (ticker, dim, blocks_ahead) VALUES (?, ?, ?)",
                (spec["ticker"], spec["dim"], spec.get("blocks_ahead", 0)),
            )

        # TRADE-MIX deprecation (FLOW launch, 2026-08-10): the challenge
        # is off the roster and its historical data is scrapped, not
        # archived.  Runs on every open; a no-op once purged.
        for _dep in ("TRADEMIX",):
            n_purged = self._conn.execute(
                "DELETE FROM challenge_data WHERE ticker=?", (_dep,)
            ).rowcount
            self._conn.execute(
                "DELETE FROM challenge_meta WHERE ticker=?", (_dep,)
            )
            if n_purged:
                logger.info("Purged %d %s rows (challenge deprecated)",
                            n_purged, _dep)
        self._conn.commit()

        self.tlock = Timelock(config.DRAND_PUBLIC_KEY)
        self._lock = asyncio.Lock()

        self._drand_cache: Dict[int, bytes] = {}
        self._DRAND_MEM_CAP = 10_000
        self._block_count: int = self._conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]

        self._breakout_trackers: Dict[str, RangeBreakoutTracker] = {}
        self._init_breakout_trackers()
        self._load_breakout_state()

        try:
            self._drand_db_count: int = self._conn.execute("SELECT COUNT(*) FROM drand_cache").fetchone()[0]
        except sqlite3.DatabaseError as e:
            logger.warning(f"drand_cache table seems corrupted: {e}")
            self._drand_db_count = 0

        logger.info(
            "Opened live SQLite datalog: %s (%d blocks, drand_in_db=%d)",
            db_path, self._block_count, self._drand_db_count,
        )

    def _init_breakout_trackers(self):
        mb = config.CHALLENGE_MAP.get("MULTIBREAKOUT")
        if not mb:
            return
        for asset in mb.get("assets", []):
            if asset not in self._breakout_trackers:
                self._breakout_trackers[asset] = RangeBreakoutTracker(
                    ticker=asset,
                    range_lookback_blocks=mb.get("range_lookback_blocks", 7200),
                    barrier_pct=mb.get("barrier_pct", 10.0),
                    min_range_pct=mb.get("min_range_pct", 1.0),
                )

    def _load_breakout_state(self):
        tables = {r[0] for r in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "breakout_state" not in tables:
            return
        mb = config.CHALLENGE_MAP.get("MULTIBREAKOUT")
        if not mb:
            return
        for asset, state_json in self._conn.execute("SELECT asset, state_json FROM breakout_state"):
            if asset in self._breakout_trackers:
                state = json.loads(state_json)
                restored = RangeBreakoutTracker.from_dict(state)
                restored.range_lookback_blocks = mb.get("range_lookback_blocks", restored.range_lookback_blocks)
                restored.barrier_pct = mb.get("barrier_pct", restored.barrier_pct)
                restored.min_range_pct = mb.get("min_range_pct", restored.min_range_pct)
                restored.max_pending_blocks = mb.get("max_pending_blocks", restored.max_pending_blocks)
                self._breakout_trackers[asset] = restored

    @property
    def block_count(self) -> int:
        return self._block_count

    @staticmethod
    def load(path: str) -> "DataLog":
        if os.path.exists(path):
            return DataLog(path)
        db_path = path if path.endswith(".db") else os.path.splitext(path)[0] + ".db"
        return DataLog(db_path)

    async def append_step(self, block: int, prices: Dict[str, float], payloads: Dict[str, bytes], mg: "bt.Metagraph"):
        async with self._lock:
            c = self._conn.cursor()
            idx = self._block_count
            c.execute("INSERT INTO blocks (idx, block) VALUES (?, ?)", (idx, block))
            self._block_count += 1

            sidx = block // SAMPLE_EVERY
            for spec in config.CHALLENGES:
                ticker = spec["ticker"]
                if ticker == "MULTIXSEC":
                    pd_map = {}
                    for a in config.BREAKOUT_ASSETS:
                        pv = prices.get(a)
                        if isinstance(pv, (int, float)) and pv > 0:
                            pd_map[a] = float(pv)
                    if pd_map:
                        c.execute(
                            "INSERT INTO challenge_data (ticker, sidx, price_data, hotkeys, embeddings) "
                            "VALUES (?, ?, ?, '[]', X'') "
                            "ON CONFLICT(ticker, sidx) DO UPDATE SET price_data=excluded.price_data",
                            (ticker, sidx, json.dumps(pd_map)),
                        )
                    continue
                if ticker in ("TRADEMIX", "FLOW"):
                    # Prices are recorded at BLOCK ARRIVAL, before payloads
                    # decrypt.  For FLOW this is what fixes the entry
                    # price E: miners commit brackets as fractions under
                    # timelock, and E is the validator's own price at the
                    # submission row — never a miner-reported value.
                    tm_assets = spec.get("assets") or []
                    pd_map = {}
                    for a in tm_assets:
                        pv = prices.get(a)
                        if isinstance(pv, (int, float)) and pv > 0:
                            pd_map[a] = float(pv)
                            if ticker != "FLOW":
                                continue
                            # Wick channels for spec-1.3 resolution: the
                            # price service may publish per-asset 1-minute
                            # consensus wicks ("BTC_HIGH" = min of venue
                            # highs, "BTC_LOW" = max of venue lows).  Stored
                            # when present and sane; a row without them
                            # resolves close-only (release section 6.14).
                            hi = prices.get(f"{a}_HIGH")
                            lo = prices.get(f"{a}_LOW")
                            if (isinstance(hi, (int, float))
                                    and isinstance(lo, (int, float))
                                    and 0 < float(lo) <= float(hi)):
                                pd_map[f"{a}_HIGH"] = float(hi)
                                pd_map[f"{a}_LOW"] = float(lo)
                    if pd_map:
                        c.execute(
                            f"INSERT INTO {_cd_table(ticker)} (ticker, sidx, price_data, hotkeys, embeddings) "
                            "VALUES (?, ?, ?, '[]', X'') "
                            "ON CONFLICT(ticker, sidx) DO UPDATE SET price_data=excluded.price_data",
                            (ticker, sidx, json.dumps(pd_map)),
                        )
                    continue
                if ticker == "FUNDINGXSEC":
                    funding_rates = prices.get("_funding_rates", {})
                    if not isinstance(funding_rates, dict):
                        funding_rates = {}
                    fd_map = {}
                    for a in config.FUNDING_ASSETS:
                        fr = funding_rates.get(a)
                        if isinstance(fr, (int, float)):
                            fd_map[a] = float(fr)
                    if fd_map:
                        c.execute(
                            "INSERT INTO challenge_data (ticker, sidx, price_data, hotkeys, embeddings) "
                            "VALUES (?, ?, ?, '[]', X'') "
                            "ON CONFLICT(ticker, sidx) DO UPDATE SET price_data=excluded.price_data",
                            (ticker, sidx, json.dumps(fd_map)),
                        )
                    continue
                p = prices.get(ticker)
                if p is not None:
                    c.execute(
                        "INSERT INTO challenge_data (ticker, sidx, price, hotkeys, embeddings) "
                        "VALUES (?, ?, ?, '[]', X'') "
                        "ON CONFLICT(ticker, sidx) DO UPDATE SET price=excluded.price",
                        (ticker, sidx, float(p)),
                    )

            for hk in mg.hotkeys:
                ct = payloads.get(hk)
                raw = json.dumps(ct).encode() if ct else b"{}"
                c.execute(
                    "INSERT OR REPLACE INTO raw_payloads (ts, hotkey, payload) VALUES (?, ?, ?)",
                    (idx, hk, raw),
                )

            self._conn.commit()
            self._update_breakout_trackers(sidx, block, prices)

    def _flush_breakout_state(self):
        if not self._breakout_trackers:
            return
        c = self._conn.cursor()
        c.execute("DELETE FROM breakout_state")
        for asset, tracker in self._breakout_trackers.items():
            c.execute(
                "INSERT INTO breakout_state (asset, state_json) VALUES (?, ?)",
                (asset, json.dumps(tracker.to_dict())),
            )
        self._conn.commit()

    def _update_breakout_trackers(self, sidx: int, block: int, prices: Dict[str, float]):
        if not self._breakout_trackers:
            return

        for asset, tracker in self._breakout_trackers.items():
            p = prices.get(asset)
            if not p or p <= 0:
                for spec in config.CHALLENGES:
                    if spec.get("price_key") == asset and spec["ticker"] in prices:
                        p = prices[spec["ticker"]]
                        break
            if not p or p <= 0:
                continue
            tracker.update_price(sidx, p)
            tracker.check_trigger(sidx, block, p, {})
            tracker.check_resolutions(block, p)
        self._flush_breakout_state()

    def _backfill_breakout_embeddings(self):
        """Attach correct embeddings to any pending or completed breakout
        that still has empty embeddings, by reading from challenge_data."""
        if not self._breakout_trackers:
            return
        dim = _MB_STORAGE_DIM
        asset_indices = {asset: i for i, asset in enumerate(config.BREAKOUT_ASSETS)}
        emb_cache: Dict[int, Dict[str, np.ndarray] | None] = {}

        def _get_emb(sidx: int) -> Dict[str, np.ndarray]:
            if sidx in emb_cache:
                return emb_cache[sidx] or {}
            row = self._conn.execute(
                "SELECT embeddings FROM challenge_data "
                "WHERE ticker='MULTIBREAKOUT' AND sidx=?",
                (sidx,),
            ).fetchone()
            result = _unpack_embeddings(row[0], dim) if row and row[0] else {}
            emb_cache[sidx] = result or None
            return result

        def _fill(sample, asset: str, start: int):
            if sample.embeddings:
                return
            full_emb = _get_emb(sample.trigger_sidx)
            if not full_emb:
                return
            for hk, full_vec in full_emb.items():
                if full_vec.shape[0] >= start + 2:
                    sample.embeddings[hk] = full_vec[start:start + 2]
            if sample.embeddings:
                logger.info(
                    "[%s] Backfilled %d miner embeddings for breakout at sidx=%d",
                    asset, len(sample.embeddings), sample.trigger_sidx,
                )

        for asset, tracker in self._breakout_trackers.items():
            aidx = asset_indices.get(asset)
            if aidx is None:
                continue
            start = aidx * 2
            for pending in (tracker.pending_high, tracker.pending_low):
                if pending is not None:
                    _fill(pending, asset, start)
            for completed in tracker.completed:
                _fill(completed, asset, start)

    async def _get_drand_signature(self, round_num: int, session: aiohttp.ClientSession | None = None) -> bytes | None:
        cached = self._drand_cache.get(round_num)
        if cached:
            return cached
        try:
            row = self._conn.execute(
                "SELECT signature FROM drand_cache WHERE round=?", (round_num,)
            ).fetchone()
            if row and row[0]:
                self._drand_cache[round_num] = row[0]
                return row[0]
        except sqlite3.DatabaseError as e:
            logger.warning(f"Failed to read from drand_cache: {e}")

        url = f"{config.DRAND_API}/beacons/{config.DRAND_BEACON_ID}/rounds/{round_num}"
        sig = None
        try:
            if session is None:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            sig = bytes.fromhex((await resp.json())["signature"])
            else:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        sig = bytes.fromhex((await resp.json())["signature"])
        except Exception:
            pass
        if not sig:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    sig_hex = resp.json().get("signature", "")
                    if sig_hex:
                        sig = bytes.fromhex(sig_hex)
            except Exception:
                return None
        if sig:
            self._drand_cache[round_num] = sig
            if len(self._drand_cache) > self._DRAND_MEM_CAP:
                to_drop = sorted(self._drand_cache)[:len(self._drand_cache) - self._DRAND_MEM_CAP // 2]
                for k in to_drop:
                    del self._drand_cache[k]
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO drand_cache (round, signature) VALUES (?, ?)",
                    (round_num, sig),
                )
                self._conn.commit()
            except sqlite3.DatabaseError as e:
                logger.warning(f"Failed to write to drand_cache: {e}")
        return sig

    def _zero_vecs(self):
        return {
            c["ticker"]: [0.0] * _get_storage_dim(c["ticker"])
            for c in config.CHALLENGES
        }

    def _validate_submission(self, sub: Any) -> Dict[str, List[float]]:
        def _sanitize_lbfgs_vec(vec: List[float]) -> List[float]:
            if not any((float(v) != 0.0) for v in vec):
                return [0.0] * 17
            arr = np.asarray(vec, dtype=float).copy()
            if arr.shape != (17,):
                return [0.0] * 17
            p = np.clip(arr[0:5], 1e-6, 1.0 - 1e-6)
            s = float(np.sum(p))
            if s <= 0:
                p[:] = 1.0 / 5.0
            else:
                p = p / s
            q = arr[5:17]
            q = np.clip(q, 1e-6, 1.0 - 1e-6)
            out = np.concatenate([p, q]).astype(float)
            return out.tolist()

        def _sanitize_flow_vec(vec: List[float]) -> List[float]:
            # FLOW fields per regime are [d, f, sl, tp1, tp2, h, trade_id]:
            # horizons (up to 336) and the integer trade_id legitimately
            # exceed 1, so the generic [-1,1] gate must not apply here.
            # Keep finite values verbatim (decode_trades in flow.py does
            # the real per-field validation); nan/inf collapse to 0.
            out: List[float] = []
            for v in vec:
                if isinstance(v, (int, float)) and np.isfinite(v):
                    out.append(float(v))
                else:
                    out.append(0.0)
            return out

        def _flatten_multibreakout_dict(d: dict) -> List[float]:
            flat: List[float] = []
            for asset in config.BREAKOUT_ASSETS:
                pair = d.get(asset)
                if (isinstance(pair, list) and len(pair) == 2
                        and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in pair)):
                    flat.extend([float(pair[0]), float(pair[1])])
                else:
                    flat.extend([0.0, 0.0])
            return flat

        def _flatten_xsec_dict(d: dict) -> List[float]:
            flat: List[float] = []
            for asset in config.BREAKOUT_ASSETS:
                val = d.get(asset)
                if isinstance(val, (int, float)) and -1 <= val <= 1:
                    flat.append(float(val))
                elif isinstance(val, list) and len(val) == 1 and isinstance(val[0], (int, float)):
                    flat.append(float(np.clip(val[0], -1, 1)))
                else:
                    flat.append(0.0)
            return flat

        def _flatten_funding_xsec_dict(d: dict) -> List[float]:
            flat: List[float] = []
            for asset in config.FUNDING_ASSETS:
                val = d.get(asset)
                if isinstance(val, (int, float)) and -1 <= val <= 1:
                    flat.append(float(val))
                elif isinstance(val, list) and len(val) == 1 and isinstance(val[0], (int, float)):
                    flat.append(float(np.clip(val[0], -1, 1)))
                else:
                    flat.append(0.0)
            return flat

        def _flatten_trademix_dict(d: dict) -> List[float]:
            flat: List[float] = []
            for asset in config.TRADE_MIX_ASSETS:
                val = d.get(asset)
                if isinstance(val, (int, float)) and -1 <= val <= 1:
                    flat.append(float(val))
                elif isinstance(val, list) and len(val) == 1 and isinstance(val[0], (int, float)):
                    flat.append(float(np.clip(val[0], -1, 1)))
                else:
                    flat.append(0.0)
            return flat

        if isinstance(sub, list) and len(sub) == len(config.CHALLENGES):
            out = {}
            for vec, c in zip(sub, config.CHALLENGES):
                ticker = c["ticker"]
                dim = _get_storage_dim(ticker)
                spec = config.CHALLENGE_MAP.get(ticker)
                if ticker == "MULTIBREAKOUT" and isinstance(vec, dict):
                    out[ticker] = _flatten_multibreakout_dict(vec)
                    continue
                if ticker == "MULTIXSEC" and isinstance(vec, dict):
                    out[ticker] = _flatten_xsec_dict(vec)
                    continue
                if ticker == "FUNDINGXSEC" and isinstance(vec, dict):
                    out[ticker] = _flatten_funding_xsec_dict(vec)
                    continue
                if ticker == "TRADEMIX" and isinstance(vec, dict):
                    out[ticker] = _flatten_trademix_dict(vec)
                    continue
                if isinstance(vec, list) and len(vec) == dim:
                    if spec and spec.get("loss_func") == "lbfgs":
                        out[ticker] = _sanitize_lbfgs_vec(vec) if dim == 17 else [0.0] * dim
                    elif spec and spec.get("loss_func") == "flow":
                        out[ticker] = _sanitize_flow_vec(vec)
                    else:
                        ok = all(isinstance(v, (int, float)) and -1 <= v <= 1 for v in vec)
                        out[ticker] = [float(v) for v in vec] if ok else [0.0] * dim
                else:
                    out[ticker] = [0.0] * dim
            return out

        if isinstance(sub, dict):
            out = self._zero_vecs()
            for key, vec in sub.items():
                if key == "hotkey":
                    continue
                ticker = key if key in config.CHALLENGE_MAP else config.CHALLENGE_NAME_TO_TICKER.get(key)
                if not ticker:
                    continue
                if ticker == "MULTIBREAKOUT" and isinstance(vec, dict):
                    out[ticker] = _flatten_multibreakout_dict(vec)
                    continue
                if ticker == "MULTIXSEC" and isinstance(vec, dict):
                    out[ticker] = _flatten_xsec_dict(vec)
                    continue
                if ticker == "FUNDINGXSEC" and isinstance(vec, dict):
                    out[ticker] = _flatten_funding_xsec_dict(vec)
                    continue
                if ticker == "TRADEMIX" and isinstance(vec, dict):
                    out[ticker] = _flatten_trademix_dict(vec)
                    continue
                dim = _get_storage_dim(ticker)
                if not isinstance(vec, list) or len(vec) != dim:
                    continue
                spec = config.CHALLENGE_MAP.get(ticker)
                if spec and spec.get("loss_func") == "lbfgs" and dim == 17:
                    out[ticker] = _sanitize_lbfgs_vec(vec)
                elif spec and spec.get("loss_func") == "flow":
                    out[ticker] = _sanitize_flow_vec(vec)
                else:
                    if not all(isinstance(v, (int, float)) and -1 <= v <= 1 for v in vec):
                        continue
                    out[ticker] = [float(v) for v in vec]
            return out
        return self._zero_vecs()

    async def process_pending_payloads(self):
        async with self._lock:
            last_row = self._conn.execute(
                "SELECT block FROM blocks ORDER BY idx DESC LIMIT 1"
            ).fetchone()
            if not last_row:
                return
            current_block = last_row[0]

            mature_rows = self._conn.execute(
                "SELECT rp.ts, rp.hotkey, rp.payload, b.block "
                "FROM raw_payloads rp "
                "JOIN blocks b ON rp.ts = b.idx "
                "WHERE ? - b.block >= ?",
                (current_block, int(config.PAYLOAD_MATURITY_BLOCKS)),
            ).fetchall()

        if not mature_rows:
            return

        rounds = defaultdict(list)
        mature = set()
        ts_to_block: Dict[int, int] = {}
        stats = {
            "payloads": 0, "decrypt_failures": 0,
            "signature_fetch_attempts": 0, "signature_fetch_failures": 0,
            "v2": 0, "v2_fail": 0, "unsupported": 0,
        }

        for ts, hk, raw, block in mature_rows:
            ts = int(ts)
            mature.add((ts, hk))
            ts_to_block[ts] = block
            try:
                data = json.loads(raw.decode()) if raw else {}
            except Exception:
                data = {}
            version = 2 if isinstance(data, dict) and data.get("v") == 2 else 0
            if version == 2:
                stats["v2"] += 1
                try:
                    rnd_key = int(data.get("round", 0))
                except (TypeError, ValueError):
                    rnd_key = 0
                rounds[rnd_key].append((ts, hk, data, version))
            else:
                stats["unsupported"] += 1

        if not mature:
            return

        dec = {}
        retained: set = set()
        retain_before = int(config.PAYLOAD_MATURITY_BLOCKS) + DRAND_RETAIN_GRACE_BLOCKS

        async def _work(rnd, items, sess: aiohttp.ClientSession):
            sig = None
            if rnd > 0:
                stats["signature_fetch_attempts"] += 1
                attempts = 0
                while attempts < DRAND_SIGNATURE_RETRIES and not sig:
                    sig = await self._get_drand_signature(rnd, sess)
                    if sig:
                        break
                    attempts += 1
                    if attempts < DRAND_SIGNATURE_RETRIES:
                        await asyncio.sleep(DRAND_SIGNATURE_RETRY_DELAY)
                if not sig and items:
                    logger.warning("Failed to fetch Drand signature for round %s after %d attempts", rnd, DRAND_SIGNATURE_RETRIES)
                    stats["signature_fetch_failures"] += 1
            for ts, hk, data, version in items:
                vecs = self._zero_vecs()
                if not sig:
                    # valid round, transient fetch failure: keep the
                    # payload pending and retry next pass (within grace)
                    age = current_block - ts_to_block.get(ts, current_block)
                    if rnd > 0 and age < retain_before:
                        retained.add((ts, hk))
                        continue
                    dec.setdefault(ts, {})[hk] = vecs
                    continue
                if version == 2:
                    stats["payloads"] += 1
                    pt_bytes = _decrypt_v2_payload(data, sig, self.tlock)
                    if not pt_bytes:
                        stats["decrypt_failures"] += 1
                        stats["v2_fail"] += 1
                    else:
                        try:
                            obj = json.loads(pt_bytes.decode("utf-8"))
                            if isinstance(obj, dict) and obj.get("hotkey") == hk:
                                vecs = self._validate_submission(obj)
                        except Exception:
                            stats["decrypt_failures"] += 1
                            stats["v2_fail"] += 1
                dec.setdefault(ts, {})[hk] = vecs

        ROUND_BATCH = 16
        round_items = list(rounds.items())
        async with aiohttp.ClientSession() as sess:
            for i in range(0, len(round_items), ROUND_BATCH):
                batch = round_items[i:i + ROUND_BATCH]
                await asyncio.gather(*(_work(r, items, sess) for r, items in batch))
                await asyncio.sleep(0.1)

        if retained:
            mature -= retained
            logger.warning(
                "Retained %d matured payloads pending drand signatures "
                "(will retry next pass)", len(retained))
        if not mature:
            return

        emb_updates: Dict[tuple, Dict[str, np.ndarray]] = defaultdict(dict)
        for ts, by_hk in dec.items():
            block = ts_to_block.get(ts)
            if block is None or block % SAMPLE_EVERY:
                continue
            sidx = block // SAMPLE_EVERY
            for hk, vecs in by_hk.items():
                for ticker, vec in vecs.items():
                    if any(v != 0.0 for v in vec):
                        emb_updates[(ticker, sidx)][hk] = np.array(
                            vec, dtype=_emb_dtype(ticker))

        async with self._lock:
            c = self._conn.cursor()
            for (ticker, sidx), new_embs in emb_updates.items():
                dim = _get_storage_dim(ticker)
                row = c.execute(
                    f"SELECT hotkeys, embeddings FROM {_cd_table(ticker)} WHERE ticker=? AND sidx=?",
                    (ticker, sidx),
                ).fetchone()
                if row:
                    existing_hks = json.loads(row[0]) if row[0] else []
                    existing_emb = _unpack_embeddings(row[1], dim) if row[1] else {}
                else:
                    existing_hks = []
                    existing_emb = {}

                existing_emb.update(new_embs)
                for hk in new_embs:
                    if hk not in existing_hks:
                        existing_hks.append(hk)

                hks_json = json.dumps(existing_hks)
                emb_blob = _pack_embeddings(existing_emb, _emb_dtype(ticker))
                c.execute(
                    f"INSERT INTO {_cd_table(ticker)} (ticker, sidx, price, hotkeys, embeddings) "
                    "VALUES (?, ?, NULL, ?, ?) "
                    "ON CONFLICT(ticker, sidx) DO UPDATE SET hotkeys=excluded.hotkeys, embeddings=excluded.embeddings",
                    (ticker, sidx, hks_json, emb_blob),
                )

            self._backfill_breakout_embeddings()
            self._flush_breakout_state()

            c.executemany(
                "DELETE FROM raw_payloads WHERE ts=? AND hotkey=?",
                list(mature),
            )
            self._conn.commit()

        total_payloads = stats["payloads"]
        if total_payloads > 0:
            pct = 100.0 * stats["decrypt_failures"] / total_payloads
            logger.info(
                "Payload decryption failures: %s/%s (%.2f%%)",
                stats["decrypt_failures"], total_payloads, pct,
            )
        version_total = stats["v2"] + stats["unsupported"]
        if version_total:
            v2_pct = 100.0 * stats["v2"] / version_total
            unsupported_pct = 100.0 * stats["unsupported"] / version_total
            v2_fail_pct = (100.0 * stats["v2_fail"] / stats["v2"]) if stats["v2"] else 0.0
            logger.info(
                "Payload mix (matured): V2 %d/%d (%.1f%%), unsupported %d/%d (%.1f%%); V2 failures %d/%d (%.1f%%)",
                stats["v2"], version_total, v2_pct,
                stats["unsupported"], version_total, unsupported_pct,
                stats["v2_fail"], stats["v2"], v2_fail_pct,
            )
        fetch_attempts = stats["signature_fetch_attempts"]
        if fetch_attempts > 0:
            pct_sig = 100.0 * stats["signature_fetch_failures"] / fetch_attempts
            logger.info(
                "Drand signature fetch failures: %s/%s rounds (%.2f%%)",
                stats["signature_fetch_failures"], fetch_attempts, pct_sig,
            )

    async def save(self, path: str):
        t0 = time.monotonic()
        async with self._lock:
            self._flush_breakout_state()
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            self._conn.execute("PRAGMA inv.wal_checkpoint(PASSIVE)")
        elapsed = time.monotonic() - t0
        logger.info(
            "Full save: breakout state (%d trackers) + WAL checkpoint [%.1fs]",
            len(self._breakout_trackers), elapsed,
        )

    async def snapshot_for_publish(self, dest_dir: str) -> Dict[str, str]:
        """Write a consistent, point-in-time copy of the FLOW DB.

        Copying a live WAL database (file + -wal + -shm) is a torn read:
        the copy can be internally inconsistent.  `VACUUM INTO` runs a
        single read transaction, so the snapshot is a valid standalone
        database regardless of concurrent writes.  Upload THIS file to
        the bucket, never the live one.  Returns {object_name: path}.

        FLOW only: the main datalog keeps its existing save/publish
        flow untouched.
        """
        os.makedirs(dest_dir, exist_ok=True)
        t0 = time.monotonic()
        dest = os.path.join(dest_dir, _FLOW_DB_FILENAME)
        tmp = dest + ".tmp"
        async with self._lock:
            if os.path.exists(tmp):
                os.remove(tmp)
            self._conn.execute("VACUUM inv INTO ?", (tmp,))
            os.replace(tmp, dest)
        logger.info("Publish snapshot: %s [%.1fs]",
                    dest, time.monotonic() - t0)
        return {_FLOW_DB_FILENAME: dest}

    @staticmethod
    def iter_challenge_training_data(
        db_path: str,
        max_block_number: int | None = None,
        active_hotkeys: set[str] | None = None,
    ):
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _ensure_price_data_col(conn)
        _attach_flow(conn, db_path)

        for spec in config.CHALLENGES:
            ticker = spec["ticker"]
            dim = int(spec["dim"])
            blocks_ahead = int(spec.get("blocks_ahead", 0))
            loss_func = spec.get("loss_func")

            if loss_func in ("lbfgs", "hitfirst"):
                payload = DataLog._build_lbfgs_from_db(
                    conn, ticker, dim, blocks_ahead, max_block_number,
                    active_hotkeys=active_hotkeys,
                )
                if payload:
                    yield ticker, payload
                continue

            if loss_func == "range_breakout_multi":
                completed = DataLog._load_breakout_from_db(
                    conn, max_block_number, active_hotkeys=active_hotkeys,
                )
                if completed:
                    yield ticker, {"completed_samples": completed}
                continue

            if loss_func == "xsec_rank":
                payload = DataLog._build_xsec_from_db(
                    conn, dim, blocks_ahead, max_block_number,
                    active_hotkeys=active_hotkeys,
                )
                if payload:
                    yield ticker, payload
                continue

            if loss_func == "funding_xsec":
                payload = DataLog._build_funding_xsec_from_db(
                    conn, dim, blocks_ahead, max_block_number,
                    active_hotkeys=active_hotkeys,
                )
                if payload:
                    yield ticker, payload
                continue

            if loss_func in ("trade_mix", "flow"):
                # FLOW reuses the trade-mix payload shape: per-hotkey
                # embeddings (dim x assets) + per-row prices + sidx.
                payload = DataLog._build_trade_mix_from_db(
                    conn, ticker, blocks_ahead, max_block_number,
                    active_hotkeys=active_hotkeys, spec=spec,
                )
                if payload:
                    yield ticker, payload
                continue

            payload = DataLog._build_binary_from_db(
                conn, ticker, dim, blocks_ahead, max_block_number,
                active_hotkeys=active_hotkeys,
            )
            if payload:
                yield ticker, payload

        conn.close()

    @staticmethod
    def _collect_hotkeys(conn, ticker, active_hotkeys=None):
        hks: set[str] = set()
        for (hks_json,) in conn.execute(
            f"SELECT hotkeys FROM {_cd_table(ticker)} WHERE ticker = ? AND hotkeys != '[]'",
            (ticker,),
        ):
            for hk in json.loads(hks_json):
                hks.add(hk)
        if active_hotkeys is not None:
            hks &= active_hotkeys
        if not hks:
            return None, None
        sorted_hks = sorted(hks)
        return sorted_hks, {hk: i for i, hk in enumerate(sorted_hks)}

    @staticmethod
    def _build_lbfgs_from_db(conn, ticker, dim, blocks_ahead, max_block_number, *, active_hotkeys=None):
        c = conn.cursor()
        spec = config.CHALLENGE_MAP.get(ticker)
        if not spec or spec.get("loss_func") not in ("lbfgs", "hitfirst"):
            return None

        all_hks_sorted, hk2idx = DataLog._collect_hotkeys(c, ticker, active_hotkeys)
        if all_hks_sorted is None:
            return None
        D = dim

        rows: list[np.ndarray] = []
        prices: list[float] = []
        sidx_list: list[int] = []

        for sidx, price, emb_blob in c.execute(
            "SELECT sidx, price, embeddings FROM challenge_data "
            "WHERE ticker = ? ORDER BY sidx",
            (ticker,),
        ):
            block = int(sidx) * SAMPLE_EVERY
            if max_block_number and block > max_block_number:
                break
            if price is None:
                continue
            pf = float(price)
            if not np.isfinite(pf) or pf <= 0:
                continue

            emb = _unpack_embeddings(emb_blob, D) if emb_blob else {}
            row = np.zeros((len(all_hks_sorted), D), dtype=np.float32)
            for hk, vec in emb.items():
                idx = hk2idx.get(hk)
                if idx is not None:
                    arr = np.asarray(vec, dtype=np.float32)
                    if arr.shape == (D,):
                        row[idx] = arr
                    elif arr.size == D:
                        row[idx] = arr.reshape(D)
            rows.append(row.reshape(-1))
            prices.append(pf)
            sidx_list.append(int(sidx))
            del emb

        if not rows:
            return None
        return {
            "hist": (np.stack(rows, axis=0), hk2idx),
            "price": np.asarray(prices, dtype=np.float64),
            "sidx": np.asarray(sidx_list, dtype=np.int64),
            "blocks_ahead": blocks_ahead,
        }

    @staticmethod
    def _build_binary_from_db(conn, ticker, dim, blocks_ahead, max_block_number, *, active_hotkeys=None):
        c = conn.cursor()
        ahead = blocks_ahead // SAMPLE_EVERY

        prices_by_sidx: dict[int, float] = {}
        for sidx, price in c.execute(
            "SELECT sidx, price FROM challenge_data WHERE ticker = ?", (ticker,),
        ):
            if price is not None:
                prices_by_sidx[int(sidx)] = float(price)

        all_hks_sorted, hk2idx = DataLog._collect_hotkeys(c, ticker, active_hotkeys)
        if all_hks_sorted is None:
            return None

        X_list: list[np.ndarray] = []
        y_list: list[float] = []
        prev_price = None
        unchanged_streak = 0
        max_unchanged = int(getattr(config, "MAX_UNCHANGED_TIMESTEPS", 0) or 0)

        for sidx, price, emb_blob in c.execute(
            "SELECT sidx, price, embeddings FROM challenge_data "
            "WHERE ticker = ? ORDER BY sidx",
            (ticker,),
        ):
            sidx = int(sidx)
            block = sidx * SAMPLE_EVERY
            if max_block_number and block > max_block_number:
                break

            price_now = float(price) if price is not None else None
            price_fut = prices_by_sidx.get(sidx + ahead)

            if price_now is not None:
                if prev_price is None or price_now != prev_price:
                    prev_price = price_now
                    unchanged_streak = 0
                else:
                    unchanged_streak += 1

            if max_unchanged > 0 and unchanged_streak > max_unchanged:
                continue

            if price_now is None or price_fut is None:
                continue
            if price_now <= 0.0 or price_fut <= 0.0:
                continue

            emb = _unpack_embeddings(emb_blob, dim) if emb_blob else {}
            if not emb:
                continue

            mat = np.zeros((len(all_hks_sorted), dim), dtype=np.float16)
            any_nonzero = False
            for hk, vec in emb.items():
                arr = np.asarray(vec, dtype=np.float16)
                if not any_nonzero and (arr != 0).any():
                    any_nonzero = True
                idx = hk2idx.get(hk)
                if idx is not None:
                    mat[idx] = arr
            del emb

            if not any_nonzero:
                continue

            X_list.append(mat.flatten())
            y_list.append((price_fut - price_now) / price_now if price_now else 0.0)

        if not X_list:
            return None
        return (
            (np.array(X_list, dtype=np.float16), hk2idx),
            np.array(y_list, dtype=np.float32),
        )

    @staticmethod
    def _build_xsec_from_db(conn, dim, blocks_ahead, max_block_number, *, active_hotkeys=None):
        ticker = "MULTIXSEC"
        storage_dim = dim * len(config.BREAKOUT_ASSETS)
        c = conn.cursor()

        all_hks_sorted, hk2idx = DataLog._collect_hotkeys(c, ticker, active_hotkeys)
        if all_hks_sorted is None:
            return None

        rows: list[np.ndarray] = []
        prices_list: list[list[float]] = []

        for sidx, price_data, emb_blob in c.execute(
            "SELECT sidx, price_data, embeddings FROM challenge_data "
            "WHERE ticker = ? ORDER BY sidx",
            (ticker,),
        ):
            block = int(sidx) * SAMPLE_EVERY
            if max_block_number and block > max_block_number:
                break

            if not price_data:
                continue
            pd_dict = json.loads(price_data)
            price_vec = [float(pd_dict.get(a, 0.0)) for a in config.BREAKOUT_ASSETS]
            if not any(p > 0 for p in price_vec):
                continue

            emb = _unpack_embeddings(emb_blob, storage_dim) if emb_blob else {}
            row = np.zeros((len(all_hks_sorted), storage_dim), dtype=np.float32)
            for hk, vec in emb.items():
                idx = hk2idx.get(hk)
                if idx is not None:
                    arr = np.asarray(vec, dtype=np.float32)
                    if arr.size == storage_dim:
                        row[idx] = arr.reshape(storage_dim)
            rows.append(row.reshape(-1))
            prices_list.append(price_vec)
            del emb

        if not rows:
            return None
        return {
            "hist": (np.stack(rows, axis=0), hk2idx),
            "prices_multi": np.array(prices_list, dtype=np.float64),
            "blocks_ahead": blocks_ahead,
        }

    @staticmethod
    def _build_funding_xsec_from_db(conn, dim, blocks_ahead, max_block_number, *, active_hotkeys=None):
        """Build training data for the FUNDING-XSEC challenge.

        Returns dict with 'hist', 'funding_rates', 'sidx_arr', and
        'blocks_ahead'.  The sidx array is included so the scoring module
        can pair rows by *actual* sidx distance rather than row-position,
        avoiding label misalignment from gaps in the data.
        """
        ticker = "FUNDINGXSEC"
        n_assets = len(config.FUNDING_ASSETS)
        storage_dim = dim * n_assets
        c = conn.cursor()

        all_hks_sorted, hk2idx = DataLog._collect_hotkeys(c, ticker, active_hotkeys)
        if all_hks_sorted is None:
            return None

        rows: list[np.ndarray] = []
        funding_list: list[list[float]] = []
        sidx_list: list[int] = []

        # No stale-rate filtering here.  Funding rates settle every 8h, so
        # ~480 consecutive rows will have identical rates by design.  Keeping
        # all rows is necessary so the walk-forward has enough data for
        # feature selection and meta-model fitting (CHUNK_T=4000, LAG=60).
        # The 480 rows within a settlement window share the same label but
        # have different miner embeddings, which is exactly what the
        # meta-model needs to learn from.

        for sidx, price_data, emb_blob in c.execute(
            "SELECT sidx, price_data, embeddings FROM challenge_data "
            "WHERE ticker = ? ORDER BY sidx",
            (ticker,),
        ):
            block = int(sidx) * SAMPLE_EVERY
            if max_block_number and block > max_block_number:
                break

            if not price_data:
                continue
            fd_dict = json.loads(price_data)
            funding_vec = [float(fd_dict.get(a, np.nan)) for a in config.FUNDING_ASSETS]
            if all(np.isnan(v) for v in funding_vec):
                continue

            emb = _unpack_embeddings(emb_blob, storage_dim) if emb_blob else {}
            row = np.zeros((len(all_hks_sorted), storage_dim), dtype=np.float32)
            for hk, vec in emb.items():
                idx = hk2idx.get(hk)
                if idx is not None:
                    arr = np.asarray(vec, dtype=np.float32)
                    if arr.size == storage_dim:
                        row[idx] = arr.reshape(storage_dim)
            rows.append(row.reshape(-1))
            funding_list.append(funding_vec)
            sidx_list.append(int(sidx))
            del emb

        if not rows:
            return None
        return {
            "hist": (np.stack(rows, axis=0), hk2idx),
            "funding_rates": np.array(funding_list, dtype=np.float64),
            "sidx_arr": np.array(sidx_list, dtype=np.int64),
            "blocks_ahead": blocks_ahead,
        }

    @staticmethod
    def _build_trade_mix_from_db(conn, ticker, blocks_ahead, max_block_number, *, active_hotkeys=None, spec=None,
                                 emb_overlay=None):
        """Build training data for the TRADE-MIX / FLOW challenges.

        Returns dict with 'hist' (positions matrix), 'prices_multi'
        (T x n_assets price array), 'sidx_arr', and 'blocks_ahead'.
        Storage dim = len(assets) (one signed scalar position per asset).

        `emb_overlay` ({sidx: {hotkey: vec}}, default None) fills
        embeddings the DB does not have yet — the owner-side live
        feed (team settlement tooling, kept outside this repo) passes
        payloads it decrypted before the public tlock matured.  DB
        values win on overlap (the matured plaintext is the same
        bytes), rows the DB has no price for are skipped exactly as
        without the overlay, and None keeps this builder
        byte-identical to the public path.  This repo itself only
        ever decrypts the public way.

        FLOW wick channels: when any stored row carries the price
        service's per-asset wick keys ("BTC_HIGH"/"BTC_LOW"),
        'prices_multi' is emitted as [close, high, low] columns for
        spec-1.3 wick resolution (flow.py).  Rows without wicks fall
        back per row (channel = close, exactly close-only for that row),
        and each channel is clamped to contain its own close, so a
        skewed publisher print can never place the close outside the
        row's own range.  With no wick data anywhere the shape stays
        T x 1 and resolution is close-only end to end.
        """
        if spec is None:
            spec = config.CHALLENGE_MAP.get(ticker)
        if not spec:
            return None
        assets = spec.get("assets") or []
        n_assets = len(assets)
        if n_assets == 0:
            return None
        storage_dim = int(spec.get("dim", 1)) * n_assets
        wick_asset = assets[0] if (ticker in _FLOW_DB_TICKERS
                                   and n_assets == 1) else None
        c = conn.cursor()

        all_hks_sorted, hk2idx = DataLog._collect_hotkeys(c, ticker, active_hotkeys)
        if emb_overlay:
            # union: a miner whose first payload has not matured yet
            # exists only in the overlay, and must still get a column
            overlay_hks = {hk for by_hk in emb_overlay.values() for hk in by_hk}
            if active_hotkeys is not None:
                overlay_hks &= active_hotkeys
            merged = sorted(set(all_hks_sorted or []) | overlay_hks)
            if merged:
                all_hks_sorted = merged
                hk2idx = {hk: i for i, hk in enumerate(merged)}
        if all_hks_sorted is None:
            return None

        rows: list[np.ndarray] = []
        prices_list: list[list[float]] = []
        wick_list: list[tuple] = []      # (hi | None, lo | None) per row
        any_wicks = False
        sidx_list: list[int] = []

        for sidx, price_data, emb_blob in c.execute(
            f"SELECT sidx, price_data, embeddings FROM {_cd_table(ticker)} "
            "WHERE ticker = ? ORDER BY sidx",
            (ticker,),
        ):
            block = int(sidx) * SAMPLE_EVERY
            if max_block_number and block > max_block_number:
                break

            if not price_data:
                continue
            pd_dict = json.loads(price_data)
            price_vec = [float(pd_dict.get(a, 0.0)) for a in assets]
            if not any(p > 0 for p in price_vec):
                continue

            if wick_asset is not None:
                hi = pd_dict.get(f"{wick_asset}_HIGH")
                lo = pd_dict.get(f"{wick_asset}_LOW")
                if (isinstance(hi, (int, float))
                        and isinstance(lo, (int, float))
                        and 0 < float(lo) <= float(hi)):
                    wick_list.append((float(hi), float(lo)))
                    any_wicks = True
                else:
                    wick_list.append((None, None))

            emb = _unpack_embeddings(emb_blob, storage_dim) if emb_blob else {}
            row = np.zeros((len(all_hks_sorted), storage_dim), dtype=np.float32)
            if emb_overlay:
                for hk, vec in (emb_overlay.get(int(sidx)) or {}).items():
                    idx = hk2idx.get(hk)
                    if idx is not None and hk not in emb:
                        arr = np.asarray(vec, dtype=np.float32)
                        if arr.size == storage_dim:
                            row[idx] = arr.reshape(storage_dim)
            for hk, vec in emb.items():
                idx = hk2idx.get(hk)
                if idx is not None:
                    arr = np.asarray(vec, dtype=np.float32)
                    if arr.size == storage_dim:
                        row[idx] = arr.reshape(storage_dim)
            rows.append(row.reshape(-1))
            prices_list.append(price_vec)
            sidx_list.append(int(sidx))
            del emb

        if not rows:
            return None

        prices_multi = np.array(prices_list, dtype=np.float64)
        if wick_asset is not None and any_wicks:
            close = prices_multi[:, 0]
            hi_col = np.array([h if h is not None else c_
                               for (h, _), c_ in zip(wick_list, close)])
            lo_col = np.array([l if l is not None else c_
                               for (_, l), c_ in zip(wick_list, close)])
            hi_col = np.maximum(hi_col, close)   # channel contains its close
            lo_col = np.minimum(lo_col, close)
            prices_multi = np.column_stack([close, hi_col, lo_col])

        return {
            "hist": (np.stack(rows, axis=0), hk2idx),
            "prices_multi": prices_multi,
            "sidx_arr": np.array(sidx_list, dtype=np.int64),
            "blocks_ahead": blocks_ahead,
        }

    @staticmethod
    def _load_breakout_from_db(conn, max_block_number, *, active_hotkeys=None):
        from collections import deque
        from range_breakout import CompletedBreakoutSample

        mb = config.CHALLENGE_MAP.get("MULTIBREAKOUT")
        if not mb:
            return []

        assets = mb["assets"]
        n_assets = len(assets)
        asset_indices = {asset: i for i, asset in enumerate(assets)}
        storage_dim = 2 * n_assets

        lookback_sidxs = mb.get("range_lookback_blocks", 28800) // SAMPLE_EVERY
        barrier_frac = mb.get("barrier_pct", 25.0) / 100.0
        min_range_frac = mb.get("min_range_pct", 1.0) / 100.0
        max_pending_blocks = 43200

        c = conn.cursor()

        # ---- Phase 1: pre-MULTIXSEC samples from breakout_state ----
        row = c.execute(
            "SELECT MIN(sidx) FROM challenge_data "
            "WHERE ticker='MULTIXSEC' AND price_data IS NOT NULL"
        ).fetchone()
        xsec_start = int(row[0]) if row and row[0] else None

        tables = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}

        pre_xsec: list = []
        if "breakout_state" in tables and xsec_start is not None:
            for asset, state_json in c.execute(
                "SELECT asset, state_json FROM breakout_state"
            ):
                aidx = asset_indices.get(asset)
                if aidx is None:
                    continue
                estart = aidx * 2
                for cd in json.loads(state_json).get("completed", []):
                    if cd["trigger_sidx"] >= xsec_start:
                        continue
                    if max_block_number and cd["resolution_block"] > max_block_number:
                        continue
                    emb_raw = cd.get("embeddings", {})
                    per_asset = {}
                    for hk, v in emb_raw.items():
                        arr = np.array(v, dtype=np.float16)
                        if arr.shape[0] >= estart + 2:
                            per_asset[hk] = arr[estart:estart + 2]
                        elif arr.shape == (2,):
                            per_asset[hk] = arr
                    if active_hotkeys is not None:
                        per_asset = {k: v for k, v in per_asset.items()
                                     if k in active_hotkeys}
                    pre_xsec.append(CompletedBreakoutSample(
                        trigger_sidx=cd["trigger_sidx"],
                        trigger_block=cd["trigger_block"],
                        resolution_block=cd["resolution_block"],
                        direction=cd["direction"],
                        label=cd["label"],
                        embeddings=per_asset,
                    ))

        if xsec_start is None:
            logger.info("No MULTIXSEC data; returning %d pre-xsec samples", len(pre_xsec))
            return pre_xsec

        # ---- Phase 2: load MULTIXSEC prices into numpy ----
        sidxs_list, price_rows = [], []
        for sidx, price_data in c.execute(
            "SELECT sidx, price_data FROM challenge_data "
            "WHERE ticker='MULTIXSEC' AND price_data IS NOT NULL ORDER BY sidx",
        ):
            sidx = int(sidx)
            if max_block_number and sidx * SAMPLE_EVERY > max_block_number:
                break
            pd = json.loads(price_data)
            sidxs_list.append(sidx)
            price_rows.append([float(pd.get(a, 0.0)) for a in assets])

        if not sidxs_list:
            return pre_xsec

        sidx_arr = np.array(sidxs_list, dtype=np.int64)
        price_mat = np.array(price_rows, dtype=np.float64)
        T = len(sidxs_list)

        # ---- Phase 3: batch-load MULTIBREAKOUT embedding blobs ----
        emb_blobs: dict = {}
        for sidx, blob in c.execute(
            "SELECT sidx, embeddings FROM challenge_data "
            "WHERE ticker='MULTIBREAKOUT' AND sidx>=? "
            "AND embeddings IS NOT NULL",
            (xsec_start,),
        ):
            emb_blobs[int(sidx)] = blob

        emb_decoded: dict = {}

        def _slice_emb(trigger_sidx: int, ai: int) -> dict:
            if trigger_sidx not in emb_decoded:
                blob = emb_blobs.get(trigger_sidx)
                emb_decoded[trigger_sidx] = (
                    _unpack_embeddings(blob, storage_dim) if blob else {}
                )
            full = emb_decoded[trigger_sidx]
            if not full:
                return {}
            s = ai * 2
            out = {}
            for hk, vec in full.items():
                if vec.shape[0] >= s + 2:
                    out[hk] = vec[s:s + 2]
            if active_hotkeys is not None:
                out = {k: v for k, v in out.items() if k in active_hotkeys}
            return out

        # ---- Phase 4: vectorised replay per asset ----
        post_xsec: list = []

        for ai in range(n_assets):
            prices = price_mat[:, ai]

            # O(n) rolling min/max via monotone deques (trailing window by sidx)
            rng_lo = np.full(T, np.nan)
            rng_hi = np.full(T, np.nan)
            rng_cnt = np.zeros(T, dtype=np.int32)
            lo_q: deque = deque()
            hi_q: deque = deque()
            valid_in_window = 0
            win_head = 0

            for t in range(T):
                cur_sidx = int(sidx_arr[t])
                win_lo = cur_sidx - lookback_sidxs

                while win_head < t and sidx_arr[win_head] < win_lo:
                    if prices[win_head] > 0:
                        valid_in_window -= 1
                    win_head += 1
                while lo_q and lo_q[0][1] < win_head:
                    lo_q.popleft()
                while hi_q and hi_q[0][1] < win_head:
                    hi_q.popleft()

                if lo_q:
                    rng_lo[t] = lo_q[0][0]
                if hi_q:
                    rng_hi[t] = hi_q[0][0]
                rng_cnt[t] = valid_in_window

                if prices[t] > 0:
                    pv = prices[t]
                    while lo_q and lo_q[-1][0] >= pv:
                        lo_q.pop()
                    lo_q.append((pv, t))
                    while hi_q and hi_q[-1][0] <= pv:
                        hi_q.pop()
                    hi_q.append((pv, t))
                    valid_in_window += 1

            pending_hi = None
            pending_lo = None
            half_lookback = lookback_sidxs // 2

            for t in range(T):
                p = prices[t]
                if p <= 0:
                    continue
                cur_sidx = int(sidx_arr[t])
                cur_block = cur_sidx * SAMPLE_EVERY

                if pending_hi is not None:
                    tsidx, tblk, d, cont_b, rev_b = pending_hi
                    if cur_block - tblk > max_pending_blocks:
                        pending_hi = None
                    elif p >= cont_b:
                        post_xsec.append(CompletedBreakoutSample(
                            trigger_sidx=tsidx, trigger_block=tblk,
                            resolution_block=cur_block, direction=d,
                            label=1, embeddings=_slice_emb(tsidx, ai),
                        ))
                        pending_hi = None
                    elif p <= rev_b:
                        post_xsec.append(CompletedBreakoutSample(
                            trigger_sidx=tsidx, trigger_block=tblk,
                            resolution_block=cur_block, direction=d,
                            label=0, embeddings=_slice_emb(tsidx, ai),
                        ))
                        pending_hi = None

                if pending_lo is not None:
                    tsidx, tblk, d, cont_b, rev_b = pending_lo
                    if cur_block - tblk > max_pending_blocks:
                        pending_lo = None
                    elif p <= cont_b:
                        post_xsec.append(CompletedBreakoutSample(
                            trigger_sidx=tsidx, trigger_block=tblk,
                            resolution_block=cur_block, direction=d,
                            label=1, embeddings=_slice_emb(tsidx, ai),
                        ))
                        pending_lo = None
                    elif p >= rev_b:
                        post_xsec.append(CompletedBreakoutSample(
                            trigger_sidx=tsidx, trigger_block=tblk,
                            resolution_block=cur_block, direction=d,
                            label=0, embeddings=_slice_emb(tsidx, ai),
                        ))
                        pending_lo = None

                if rng_cnt[t] < half_lookback:
                    continue
                lo, hi = rng_lo[t], rng_hi[t]
                if np.isnan(lo) or np.isnan(hi):
                    continue
                rw = hi - lo
                if rw < p * min_range_frac:
                    continue
                bd = rw * barrier_frac

                if p > hi and pending_hi is None:
                    pending_hi = (cur_sidx, cur_block, 1, p + bd, p - bd)
                if p < lo and pending_lo is None:
                    pending_lo = (cur_sidx, cur_block, -1, p - bd, p + bd)

        logger.info(
            "Breakout: %d pre-xsec + %d recomputed from %d price rows",
            len(pre_xsec), len(post_xsec), T,
        )
        return pre_xsec + post_xsec

    @staticmethod
    def get_hotkey_first_blocks_from_db(db_path: str, sample_every: int) -> dict[str, int]:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _attach_flow(conn, db_path)
        hotkey_first_block: dict[str, int] = {}
        for sidx, hks_json in conn.execute(
            "SELECT sidx, hotkeys FROM challenge_data "
            "WHERE hotkeys != '[]' "
            "UNION ALL "
            "SELECT sidx, hotkeys FROM inv.challenge_data "
            "WHERE hotkeys != '[]' ORDER BY sidx ASC",
        ):
            block = int(sidx) * sample_every
            for hk in json.loads(hks_json):
                if hk not in hotkey_first_block:
                    hotkey_first_block[hk] = block
        conn.close()
        return hotkey_first_block

