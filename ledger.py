from __future__ import annotations
import asyncio, copy, json, logging, os, pickle, gzip
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np, aiohttp, bittensor as bt
from timelock import Timelock
import config

logger = logging.getLogger(__name__)
SAMPLE_EVERY = config.SAMPLE_EVERY

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

DRAND_SIGNATURE_RETRIES = 3
DRAND_SIGNATURE_RETRY_DELAY = 1.0

@dataclass
class ChallengeData:
    dim: int
    blocks_ahead: int
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
    def __init__(self):
        self.blocks: List[int] = []
        self.challenges = {c["ticker"]: ChallengeData(c["dim"], c["blocks_ahead"]) for c in config.CHALLENGES}
        self.raw_payloads: Dict[int, Dict[str, bytes]] = {}
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
        self._lock = asyncio.Lock()
        # Cache Drand round -> signature bytes so multiple payloads don't re-fetch over the network.
        self._drand_cache: Dict[int, bytes] = {}

    async def append_step(self, block: int, prices: Dict[str, float], payloads: Dict[str, bytes], mg: bt.metagraph):
        async with self._lock:
            self.blocks.append(block)
            ts = len(self.blocks) - 1
            self.raw_payloads[ts] = {}
            sidx = block // SAMPLE_EVERY
            for t, ch in self.challenges.items():
                p = prices.get(t)
                if p is not None:
                    ch.set_price(sidx, p)
            for hk in mg.hotkeys:
                ct = payloads.get(hk)
                self.raw_payloads[ts][hk] = json.dumps(ct).encode() if ct else b"{}"

    async def _get_drand_signature(self, round_num: int, session: aiohttp.ClientSession | None = None) -> bytes | None:
        cached = self._drand_cache.get(round_num)
        if cached:
            return cached
        try:
            if session is None:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/rounds/{round_num}", timeout=10) as resp:
                        if resp.status == 200:
                            sig = bytes.fromhex((await resp.json())["signature"])
                            if sig:
                                self._drand_cache[round_num] = sig
                            return sig
            else:
                async with session.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/rounds/{round_num}", timeout=10) as resp:
                    if resp.status == 200:
                        sig = bytes.fromhex((await resp.json())["signature"])
                        if sig:
                            self._drand_cache[round_num] = sig
                        return sig
        except Exception:
            return None
        return None

    def _zero_vecs(self):
        return {c["ticker"]: [0.0] * c["dim"] for c in config.CHALLENGES}

    def _validate_submission(self, sub: Any) -> Dict[str, List[float]]:
        if not isinstance(sub, list) or len(sub) != len(config.CHALLENGES):
            return self._zero_vecs()
        out = {}
        for vec, c in zip(sub, config.CHALLENGES):
            d = c["dim"]
            if isinstance(vec, list) and len(vec) == d and all(isinstance(v, (int, float)) and -1 <= v <= 1 for v in vec):
                out[c["ticker"]] = vec
            else:
                out[c["ticker"]] = [0.0] * d
        return out

    async def process_pending_payloads(self):
        async with self._lock:
            payloads = copy.deepcopy(self.raw_payloads)
            blocks = list(self.blocks)
        if not payloads:
            return
        current_block = blocks[-1]
        rounds = defaultdict(list)
        mature = set()
        for ts, by_hk in payloads.items():
            if current_block - blocks[ts] >= 300:
                for hk, raw in by_hk.items():
                    mature.add((ts, hk))
                    try:
                        d = json.loads(raw.decode()) if raw else {}
                        rounds[d.get("round", 0)].append((ts, hk, d.get("ciphertext", "")))
                    except Exception:
                        rounds[0].append((ts, hk, ""))
        if not mature:
            return
        dec = {}
        stats = {
            "payloads": 0,
            "decrypt_failures": 0,
            "signature_fetch_attempts": 0,
            "signature_fetch_failures": 0,
        }
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
            for ts, hk, ct_hex in items:
                vecs = self._zero_vecs()
                if sig and ct_hex:
                    stats["payloads"] += 1
                    try:
                        pt = self.tlock.tld(bytes.fromhex(ct_hex), sig).decode()
                        emb_str, hk_in = pt.rsplit(":::", 1)
                        if hk_in == hk:
                            vecs = self._validate_submission(json.loads(emb_str))
                    except Exception:
                        stats["decrypt_failures"] += 1
                        pass
                dec.setdefault(ts, {})[hk] = vecs

        ROUND_BATCH = 16
        round_items = list(rounds.items())
        async with aiohttp.ClientSession() as sess:
            for i in range(0, len(round_items), ROUND_BATCH):
                batch = round_items[i:i + ROUND_BATCH]
                await asyncio.gather(*(_work(r, items, sess) for r, items in batch))
                await asyncio.sleep(0.1)
        async with self._lock:
            for ts, by_hk in dec.items():
                block = blocks[ts]
                if block % SAMPLE_EVERY: continue
                sidx = block // SAMPLE_EVERY
                for hk, vecs in by_hk.items():
                    for t, vec in vecs.items():
                        if any(v != 0.0 for v in vec):
                            self.challenges[t].set_emb(sidx, hk, vec)
            for ts, hk in mature:
                self.raw_payloads.get(ts, {}).pop(hk, None)
                if ts in self.raw_payloads and not self.raw_payloads[ts]:
                    del self.raw_payloads[ts]

        total_payloads = stats["payloads"]
        if total_payloads > 0:
            pct = 100.0 * stats["decrypt_failures"] / total_payloads
            logger.info(
                "Payload decryption failures: %s/%s (%.2f%%)",
                stats["decrypt_failures"],
                total_payloads,
                pct,
            )
        fetch_attempts = stats["signature_fetch_attempts"]
        if fetch_attempts > 0:
            pct_sig = 100.0 * stats["signature_fetch_failures"] / fetch_attempts
            logger.info(
                "Drand signature fetch failures: %s/%s rounds (%.2f%%)",
                stats["signature_fetch_failures"],
                fetch_attempts,
                pct_sig,
            )

    def prune_hotkeys(self, active: List[str]):
        active_set = set(active)
        for ch in self.challenges.values():
            for d in ch.sidx.values():
                d["emb"] = {hk: v for hk, v in d["emb"].items() if hk in active_set}
                d["hotkeys"] = [hk for hk in d["hotkeys"] if hk in active_set]

    def get_training_data_sync(self, max_block_number: int | None = None) -> dict:
        res = {}
        for t, ch in self.challenges.items():
            ahead = ch.blocks_ahead // SAMPLE_EVERY
            all_hks = sorted({hk for d in ch.sidx.values() for hk in d["emb"].keys()})
            if not all_hks: continue
            hk2idx = {hk: i for i, hk in enumerate(all_hks)}
            X, y = [], []
            # Track consecutive unchanged price streak for this asset
            prev_price = None
            unchanged_streak = 0
            max_unchanged = int(getattr(config, "MAX_UNCHANGED_TIMESTEPS", 0) or 0)
            for sidx, data in sorted(ch.sidx.items()):
                block = sidx * SAMPLE_EVERY
                if max_block_number and block > max_block_number: break
                future = ch.sidx.get(sidx + ahead)
                price_now = data.get("price")
                price_fut = future.get("price") if future else None

                # Update unchanged streak only when current price is available
                if price_now is not None:
                    if prev_price is None or price_now != prev_price:
                        prev_price = price_now
                        unchanged_streak = 0
                    else:
                        unchanged_streak += 1

                # Drop if price unchanged beyond threshold for this asset
                if max_unchanged > 0 and unchanged_streak > max_unchanged:
                    continue

                # Require both current and future prices and they must be > 0.0
                if (not future) or (price_now is None) or (price_fut is None):
                    continue
                try:
                    if float(price_now) <= 0.0 or float(price_fut) <= 0.0:
                        continue
                except Exception:
                    continue
                # Drop rows where embeddings dict is empty or all embeddings are zeros
                if not data["emb"]:
                    continue

                mat = np.zeros((len(all_hks), ch.dim), dtype=np.float16)
                any_nonzero = False
                for hk, vec in data["emb"].items():
                    arr = np.asarray(vec, dtype=np.float16)
                    if not any_nonzero and (arr != 0).any():
                        any_nonzero = True
                    mat[hk2idx[hk]] = arr
                if not any_nonzero:
                    continue
                X.append(mat.flatten())
                p0, p1 = price_now, price_fut
                y.append((p1 - p0) / p0 if p0 else 0.0)
            if X:
                res[t] = ((np.array(X, dtype=np.float16), hk2idx), np.array(y, dtype=np.float32))
        return res

    async def save(self, path: str):
        async with self._lock:
            with (gzip.open if path.endswith('.gz') else open)(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "DataLog":
        if not os.path.exists(path):
            return DataLog()
        try:
            with (gzip.open if path.endswith('.gz') else open)(path, 'rb') as f:
                obj = pickle.load(f)
                obj._lock = asyncio.Lock()
                if not hasattr(obj, "_drand_cache"):
                    obj._drand_cache = {}
                return obj
        except Exception:
            return DataLog()


