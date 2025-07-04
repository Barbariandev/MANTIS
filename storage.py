import gzip
import json
import logging
import os
import pickle
import ast
from typing import Any, Dict, List
import requests
import asyncio
import time
import secrets
import aiohttp
import copy

from timelock import Timelock

import bittensor as bt
import torch

import config

logger = logging.getLogger(__name__)

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

ZERO_VEC = [0.0] * config.FEATURE_LENGTH

def _precompute_encrypted_zero():
    try:
        tlock = Timelock(DRAND_PUBLIC_KEY)
        round_num = 1
        vector_str = str(ZERO_VEC)
        salt = secrets.token_bytes(32)
        
        ciphertext_hex = tlock.tle(round_num, vector_str, salt).hex()
        
        payload_dict = {"round": round_num, "ciphertext": ciphertext_hex}
        return json.dumps(payload_dict).encode("utf-8")
    except Exception as e:
        logger.error(f"Failed to pre-compute encrypted zero vector, using fallback: {e}")
        return b'{"round": 1, "ciphertext": "error"}'

ENCRYPTED_ZERO_PAYLOAD = _precompute_encrypted_zero()


class DataLog:
    """
    A unified, append-only log for all historical data in the subnet.

    This class manages the complete state of miner data, including block numbers,
    BTC prices, raw encrypted payloads, and a cache for decrypted plaintext data.
    It is designed to be the single source of truth.

    The log is persisted to a single file, making it a self-contained and portable data store.
    """

    def __init__(self):
        self.blocks: List[int] = []
        self.btc_prices: List[float] = []
        self.plaintext_cache: List[Dict[int, List[float]]] = []
        self.raw_payloads: Dict[int, Dict[int, bytes]] = {}

        self._lock = asyncio.Lock()

        self.tlock = Timelock(DRAND_PUBLIC_KEY)
        self._drand_info: Dict[str, Any] = {}
        self._drand_info_last_update: float = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_lock' in state:
            del state['_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = asyncio.Lock()

    async def _get_drand_info(self) -> Dict[str, Any]:
        if not self._drand_info or time.time() - self._drand_info_last_update > 3600:
            try:
                url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        response.raise_for_status()
                        self._drand_info = await response.json()
                self._drand_info_last_update = time.time()
                logger.info("Updated Drand beacon info.")
            except Exception as e:
                logger.error(f"Failed to get Drand info: {e}")
                return {}
        return self._drand_info

    async def _get_drand_signature(self, round_num: int) -> bytes | None:
        await asyncio.sleep(2)
        try:
            url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/rounds/{round_num}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"✅ Signature fetched for round {round_num}")
                        return bytes.fromhex(data["signature"])
                    else:
                        logger.warning(
                            f"-> Failed to fetch signature for round {round_num}, "
                            f"status: {response.status}"
                        )
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"-> Timeout fetching signature for round {round_num}")
            return None
        except Exception as e:
            logger.error(f"-> Error fetching signature for round {round_num}: {e}")
            return None

    async def append_step(
        self, block: int, btc_price: float, payloads: Dict[int, bytes]
    ) -> None:
        async with self._lock:
            self.blocks.append(block)
            self.btc_prices.append(btc_price)
            self.plaintext_cache.append({})

            current_timestep = len(self.blocks) - 1
            self.raw_payloads[current_timestep] = {}

            all_known_uids = self._get_all_uids_unsafe()
            for uid in all_known_uids:
                if uid not in payloads:
                    payloads[uid] = ENCRYPTED_ZERO_PAYLOAD

            for uid, payload in payloads.items():
                self.raw_payloads[current_timestep][uid] = payload

                if not self._is_known_uid_unsafe(uid):
                    self._backfill_new_uid_unsafe(uid)

    def _get_all_uids_unsafe(self) -> List[int]:
        uids = set()
        for step_cache in self.plaintext_cache:
            uids.update(step_cache.keys())
        for step_payloads in self.raw_payloads.values():
            uids.update(step_payloads.keys())
        return sorted(list(uids))

    def get_all_uids_sync(self) -> List[int]:
        return self._get_all_uids_unsafe()

    def is_known_uid(self, uid: int) -> bool:
        if not self.plaintext_cache:
            return False
        return uid in self.plaintext_cache[0]

    def _is_known_uid_unsafe(self, uid: int) -> bool:
        if not self.plaintext_cache:
            return False
        return uid in self.plaintext_cache[0]

    def _backfill_new_uid_unsafe(self, uid: int) -> None:
        if len(self.plaintext_cache) <= 1:
            return

        logger.info(f"New miner detected (UID: {uid}). Backfilling history.")
        for step_cache in self.plaintext_cache[:-1]:
            if uid not in step_cache:
                step_cache[uid] = ZERO_VEC

    async def process_pending_payloads(self) -> None:
        async with self._lock:
            payloads_to_process = copy.deepcopy(self.raw_payloads)
            current_block = self.blocks[-1] if self.blocks else 0
            block_map = {ts: self.blocks[ts] for ts in payloads_to_process}

        if not payloads_to_process:
            return

        rounds_to_process: Dict[int, List[Dict]] = {}
        timesteps_to_discard = []

        for ts, payloads_at_step in payloads_to_process.items():
            block_age = current_block - block_map[ts]

            if block_age > 600:
                logger.warning(f"Discarding stale raw payloads at timestep {ts} (age: {block_age} blocks)")
                timesteps_to_discard.append(ts)
                continue
            
            if not (300 <= block_age):
                continue

            for uid, payload_bytes in payloads_at_step.items():
                try:
                    if isinstance(payload_bytes, dict):
                        p = payload_bytes
                    else:
                        p = json.loads(payload_bytes)

                    round_num = p["round"]
                    if round_num not in rounds_to_process:
                        rounds_to_process[round_num] = []
                    rounds_to_process[round_num].append(
                        {"ts": ts, "uid": uid, "ct_hex": p["ciphertext"]}
                    )
                except Exception:
                    if "malformed" not in rounds_to_process:
                        rounds_to_process["malformed"] = []
                    rounds_to_process["malformed"].append({"ts": ts, "uid": uid})
        
        sem = asyncio.Semaphore(16)
        decrypted_results = {}
        processed_keys = []

        async def _fetch_and_decrypt(round_num, items):
            nonlocal processed_keys
            if round_num == "malformed":
                for item in items:
                    decrypted_results.setdefault(item["ts"], {})[item["uid"]] = ZERO_VEC
                    processed_keys.append((item['ts'], item['uid']))
                return

            async with sem:
                sig = await self._get_drand_signature(round_num)
                if not sig:
                    return

                logger.info(f"Decrypting batch of {len(items)} payloads for Drand round {round_num}")
                for item in items:
                    ts, uid, ct_hex = item["ts"], item["uid"], item["ct_hex"]
                    try:
                        pt_bytes = self.tlock.tld(bytes.fromhex(ct_hex), sig)
                        vector = ast.literal_eval(pt_bytes.decode())
                        result = vector if self._validate_vector(vector) else ZERO_VEC
                    except Exception as e:
                        logger.warning(f"tlock decryption failed for UID {uid} at ts {ts}: {e}")
                        result = ZERO_VEC
                    
                    decrypted_results.setdefault(ts, {})[uid] = result
                    processed_keys.append((ts, uid))

        await asyncio.gather(*[_fetch_and_decrypt(r, i) for r, i in rounds_to_process.items()])

        if decrypted_results or processed_keys or timesteps_to_discard:
            async with self._lock:
                for ts, uid_vectors in decrypted_results.items():
                    if ts < len(self.plaintext_cache):
                        self.plaintext_cache[ts].update(uid_vectors)

                for ts, uid in processed_keys:
                    if ts in self.raw_payloads and uid in self.raw_payloads[ts]:
                        del self.raw_payloads[ts][uid]
                        if not self.raw_payloads[ts]:
                            del self.raw_payloads[ts]

                for ts in timesteps_to_discard:
                    if ts in self.raw_payloads:
                        del self.raw_payloads[ts]

    def get_training_data(self, max_block_number: int | None = None) -> tuple[dict[int, list], list[float]] | None:
        if not self.plaintext_cache:
            logger.warning("Not enough data to create a training set (no plaintext cache).")
            return None

        if max_block_number is not None:
            end_idx = next((i for i, b in enumerate(self.blocks) if b >= max_block_number), len(self.blocks))
            if end_idx == 0:
                logger.warning(f"No data available before block {max_block_number}.")
                return None
            
            blocks = self.blocks[:end_idx]
            btc_prices = self.btc_prices[:end_idx]
            plaintext_cache = self.plaintext_cache[:end_idx]
        else:
            blocks = self.blocks
            btc_prices = self.btc_prices
            plaintext_cache = self.plaintext_cache

        TIMESTEPS_TO_SKIP = 1000
        if len(blocks) <= TIMESTEPS_TO_SKIP:
            logger.warning(
                f"Not enough data for training after filtering. "
                f"Have {len(blocks)} timesteps, but require more than "
                f"{TIMESTEPS_TO_SKIP} to skip the initial period."
            )
            return None
        
        blocks = blocks[TIMESTEPS_TO_SKIP:]
        btc_prices = btc_prices[TIMESTEPS_TO_SKIP:]
        plaintext_cache = plaintext_cache[TIMESTEPS_TO_SKIP:]

        if len(blocks) < config.LAG * 2 + 1:
            logger.warning("Not enough data to create a training set after filtering and skipping initial timesteps.")
            return None

        T = len(blocks)
        all_uids = self._get_all_uids_unsafe()
        history_dict = {uid: [] for uid in all_uids}
        btc_returns = []

        for t in range(T - config.LAG):
            p_initial = btc_prices[t]
            p_final = btc_prices[t + config.LAG]
            if p_initial > 0:
                btc_returns.append((p_final - p_initial) / p_initial)
            else:
                btc_returns.append(0.0)

        effective_T = len(btc_returns)
        for t in range(effective_T):
            for uid in all_uids:
                vector = plaintext_cache[t].get(uid, ZERO_VEC)
                history_dict[uid].append(vector)

        return history_dict, btc_returns

    @staticmethod
    def _validate_vector(vector: Any) -> bool:
        if not isinstance(vector, list) or len(vector) != config.FEATURE_LENGTH:
            return False
        return all(isinstance(v, (int, float)) and -1.0 <= v <= 1.0 for v in vector)

    async def save(self, path: str) -> None:
        async with self._lock:
            datalog_copy = copy.deepcopy(self)

        def _save_job():
            try:
                tmp_path = path + ".tmp"
                with gzip.open(tmp_path, "wb") as f:
                    pickle.dump(datalog_copy, f)
                os.replace(tmp_path, path)
                logger.info(f"Saved DataLog to {path}")
            except Exception as e:
                logger.error(f"Error in background save thread: {e}", exc_info=True)

        await asyncio.to_thread(_save_job)

    @staticmethod
    def load(path: str) -> "DataLog":
        logger.info(f"Fetching latest datalog from R2 bucket: {config.DATALOG_ARCHIVE_URL}")
        try:
            url = config.DATALOG_ARCHIVE_URL
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            
            tmp_path = path + ".tmp"
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            os.replace(tmp_path, path)
            logger.info(f"Downloaded and saved archive to {path}")

            with gzip.open(path, "rb") as f:
                log = pickle.load(f)
            logger.info(f"Loaded DataLog from {path}")
            return log

        except Exception as e:
            logger.error(f"Failed to download or load archive from {config.DATALOG_ARCHIVE_URL}: {e}")
            logger.warning("Starting with a new, empty DataLog.")
            return DataLog() 
