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


def _pack_embeddings(emb: Dict[str, np.ndarray]) -> bytes:
    if not emb:
        return b""
    hk_list = sorted(emb.keys())
    vecs = np.array([np.asarray(emb[hk], dtype=np.float16) for hk in hk_list], dtype=np.float16)
    return json.dumps(hk_list).encode() + b"\x00" + vecs.tobytes()


def _unpack_embeddings(blob: bytes, dim: int) -> Dict[str, np.ndarray]:
    if not blob:
        return {}
    sep = blob.index(b"\x00")
    hk_list = json.loads(blob[:sep].decode())
    raw = blob[sep + 1:]
    n = len(hk_list)
    if n == 0 or len(raw) != n * dim * 2:
        return {}
    vecs = np.frombuffer(raw, dtype=np.float16).reshape(n, dim)
    return {hk: vecs[i].copy() for i, hk in enumerate(hk_list)}


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
    """Convert old ark-serialize 0.4.x TLECiphertext format (372 bytes) to 0.5.x (356 bytes).

    In ark-serialize 0.4, the IBECiphertext's [u8; 32] fields (v, w) were
    serialized with u64 length prefixes. In 0.5, fixed-size arrays are
    written directly without length prefixes. This strips the two 8-byte
    prefixes so the current timelock_wasm_wrapper can deserialize them.
    """
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
        ct_bytes = _convert_tlock_ct(bytes.fromhex(payload["W_time"]["ct"]))
        skeK_raw = tlock.tld(ct_bytes, sig)
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

        for spec in config.CHALLENGES:
            self._conn.execute(
                "INSERT OR REPLACE INTO challenge_meta (ticker, dim, blocks_ahead) VALUES (?, ?, ?)",
                (spec["ticker"], spec["dim"], spec.get("blocks_ahead", 0)),
            )
        self._conn.commit()

        self.tlock = Timelock(config.DRAND_PUBLIC_KEY)
        self._lock = asyncio.Lock()

        self._drand_cache: Dict[int, bytes] = {}
        self._DRAND_MEM_CAP = 10_000
        self._block_count: int = self._conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]

        self._breakout_trackers: Dict[str, RangeBreakoutTracker] = {}
        self._init_breakout_trackers()
        self._load_breakout_state()

        self._drand_db_count: int = self._conn.execute("SELECT COUNT(*) FROM drand_cache").fetchone()[0]

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
        row = self._conn.execute(
            "SELECT signature FROM drand_cache WHERE round=?", (round_num,)
        ).fetchone()
        if row and row[0]:
            self._drand_cache[round_num] = row[0]
            return row[0]

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
            self._conn.execute(
                "INSERT OR REPLACE INTO drand_cache (round, signature) VALUES (?, ?)",
                (round_num, sig),
            )
            self._conn.commit()
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
                if isinstance(vec, list) and len(vec) == dim:
                    if spec and spec.get("loss_func") == "lbfgs":
                        out[ticker] = _sanitize_lbfgs_vec(vec) if dim == 17 else [0.0] * dim
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
                dim = _get_storage_dim(ticker)
                if not isinstance(vec, list) or len(vec) != dim:
                    continue
                spec = config.CHALLENGE_MAP.get(ticker)
                if spec and spec.get("loss_func") == "lbfgs" and dim == 17:
                    out[ticker] = _sanitize_lbfgs_vec(vec)
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
                "WHERE ? - b.block >= 300",
                (current_block,),
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

        emb_updates: Dict[tuple, Dict[str, np.ndarray]] = defaultdict(dict)
        for ts, by_hk in dec.items():
            block = ts_to_block.get(ts)
            if block is None or block % SAMPLE_EVERY:
                continue
            sidx = block // SAMPLE_EVERY
            for hk, vecs in by_hk.items():
                for ticker, vec in vecs.items():
                    if any(v != 0.0 for v in vec):
                        emb_updates[(ticker, sidx)][hk] = np.array(vec, dtype=np.float16)

        async with self._lock:
            c = self._conn.cursor()
            for (ticker, sidx), new_embs in emb_updates.items():
                dim = _get_storage_dim(ticker)
                row = c.execute(
                    "SELECT hotkeys, embeddings FROM challenge_data WHERE ticker=? AND sidx=?",
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
                emb_blob = _pack_embeddings(existing_emb)
                c.execute(
                    "INSERT INTO challenge_data (ticker, sidx, price, hotkeys, embeddings) "
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
        elapsed = time.monotonic() - t0
        logger.info(
            "Full save: breakout state (%d trackers) + WAL checkpoint [%.1fs]",
            len(self._breakout_trackers), elapsed,
        )

    @staticmethod
    def iter_challenge_training_data(
        db_path: str,
        max_block_number: int | None = None,
        active_hotkeys: set[str] | None = None,
    ):
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _ensure_price_data_col(conn)

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
            "SELECT hotkeys FROM challenge_data WHERE ticker = ? AND hotkeys != '[]'",
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
        hotkey_first_block: dict[str, int] = {}
        for sidx, hks_json in conn.execute(
            "SELECT sidx, hotkeys FROM challenge_data "
            "WHERE hotkeys != '[]' ORDER BY sidx ASC",
        ):
            block = int(sidx) * sample_every
            for hk in json.loads(hks_json):
                if hk not in hotkey_first_block:
                    hotkey_first_block[hk] = block
        conn.close()
        return hotkey_first_block

