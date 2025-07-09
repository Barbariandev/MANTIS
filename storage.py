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
import numpy as np

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

ZERO_MULTI_ASSET_VEC = [
    [0.0] * config.ASSET_EMBEDDING_DIMS[asset] for asset in config.ASSETS
]

def _precompute_encrypted_zero():
    try:
        tlock = Timelock(DRAND_PUBLIC_KEY)
        round_num = 1
        vector_str = str(ZERO_MULTI_ASSET_VEC)
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
    multi-asset prices, raw encrypted payloads, and a cache for decrypted plaintext data.
    It is designed to be the single source of truth.

    The log is persisted to a single file, making it a self-contained and portable data store.
    """

    def __init__(self):
        self.blocks: List[int] = []
        self.asset_prices: List[Dict[str, float]] = []
        self.plaintext_cache: List[Dict[int, Dict[str, List[float]]]] = []
        self.raw_payloads: Dict[int, Dict[int, dict]] = {}
        self.uid_owner: Dict[int, str] = {}

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

    async def sync_miners(self, uid_to_hotkey: Dict[int, str]) -> None:
        """
        Synchronizes miner UIDs with their hotkeys, wiping the history for any
        UID that has changed ownership. This should be called on each metagraph sync.
        """
        async with self._lock:
            for uid, hotkey in uid_to_hotkey.items():
                prev_hotkey = self.uid_owner.get(uid)

                if prev_hotkey is None:
                    self.uid_owner[uid] = hotkey
                elif prev_hotkey != hotkey:
                    logger.warning(
                        f"Ownership change for UID {uid}: '{prev_hotkey[:8]}...' -> '{hotkey[:8]}...'. "
                        f"Wiping history for this UID."
                    )
                    self.uid_owner[uid] = hotkey
                    self._backfill_new_uid_unsafe(uid)

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
        self, block: int, asset_prices: Dict[str, float], payloads: Dict[int, dict]
    ) -> None:
        async with self._lock:
            self.blocks.append(block)
            self.asset_prices.append(asset_prices)
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

    def _create_zero_embeddings(self) -> Dict[str, List[float]]:
        return {
            asset: [0.0] * config.ASSET_EMBEDDING_DIMS[asset] 
            for asset in config.ASSETS
        }

    def _parse_and_validate_submission(self, submission: Any) -> Dict[str, List[float]]:
        """
        Parse and validate miner submission for the multi-asset format.
        
        New format: List[List[float]] (one list per asset in order)
        """
        try:
            # Check if this is the new format (list of lists)
            if (isinstance(submission, list) and 
                len(submission) == len(config.ASSETS) and
                all(isinstance(asset_vec, list) for asset_vec in submission)):
                
                result = {}
                for i, asset in enumerate(config.ASSETS):
                    asset_vec = submission[i]
                    if self._validate_single_asset_vector(asset_vec, asset):
                        result[asset] = asset_vec
                    else:
                        logger.warning(f"Invalid embedding for asset {asset}")
                        result[asset] = [0.0] * config.ASSET_EMBEDDING_DIMS[asset]
                
                return result
            
            # Invalid format
            logger.warning(f"Invalid submission format: {type(submission)}")
            return self._create_zero_embeddings()
            
        except Exception as e:
            logger.warning(f"Error parsing submission: {e}")
            return self._create_zero_embeddings()

    def _validate_single_asset_vector(self, vector: Any, asset: str) -> bool:
        expected_dim = config.ASSET_EMBEDDING_DIMS[asset]
        
        if not isinstance(vector, list) or len(vector) != expected_dim:
            return False
        
        if not all(isinstance(v, (int, float)) for v in vector):
            return False
            
        if not all(-1.0 <= v <= 1.0 for v in vector):
            return False
            
        return True

    def _backfill_new_uid_unsafe(self, uid: int) -> None:
        if len(self.plaintext_cache) <= 1:
            return

        logger.info(f"New miner detected (UID: {uid}). Backfilling history.")
        for step_cache in self.plaintext_cache[:-1]:
            if uid not in step_cache:
                step_cache[uid] = self._create_zero_embeddings()

    async def process_pending_payloads(self, uid_to_hotkey: Dict[int, str]) -> None:
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

            for uid, payload_dict in payloads_at_step.items():
                try:
                    round_num = payload_dict["round"]
                    if round_num not in rounds_to_process:
                        rounds_to_process[round_num] = []
                    rounds_to_process[round_num].append(
                        {"ts": ts, "uid": uid, "ct_hex": payload_dict["ciphertext"]}
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
                    decrypted_results.setdefault(item["ts"], {})[item["uid"]] = self._create_zero_embeddings()
                    processed_keys.append((item['ts'], item['uid']))
                return

            async with sem:
                sig = await self._get_drand_signature(round_num)
                if not sig:
                    return

                logger.info(f"Decrypting batch of {len(items)} payloads for Drand round {round_num}")
                for item in items:
                    ts, uid, ct_hex = item["ts"], item["uid"], item["ct_hex"]
                    is_valid = False
                    try:
                        pt_bytes = self.tlock.tld(bytes.fromhex(ct_hex), sig)
                        
                        DECRYPTED_PAYLOAD_LIMIT_BYTES = 32 * 1024 # 32KB limit
                        if len(pt_bytes) > DECRYPTED_PAYLOAD_LIMIT_BYTES:
                            raise ValueError(f"Decrypted payload size {len(pt_bytes)} exceeds limit")

                        full_plaintext = pt_bytes.decode('utf-8')
                        
                        delimiter = ":::"
                        parts = full_plaintext.rsplit(delimiter, 1)
                        if len(parts) != 2:
                            raise ValueError("Payload missing hotkey delimiter.")

                        embeddings_str, payload_hotkey = parts
                        
                        expected_hotkey = uid_to_hotkey.get(uid)
                        if not expected_hotkey or payload_hotkey != expected_hotkey:
                            raise ValueError(f"Hotkey mismatch for UID {uid}. Expected {expected_hotkey[:8]}, got {payload_hotkey[:8]}")

                        submission = ast.literal_eval(embeddings_str)
                        result = self._parse_and_validate_submission(submission)
                        is_valid = True
                    except Exception as e:
                        logger.warning(f"tlock decryption failed for UID {uid} at ts {ts}: {e}")
                        result = self._create_zero_embeddings()
                    
                    decrypted_results.setdefault(ts, {})[uid] = result
                    processed_keys.append((ts, uid))
                return is_valid

        tasks = [_fetch_and_decrypt(r, i) for r, i in rounds_to_process.items()]
        results = await asyncio.gather(*tasks)

        total_payloads = sum(len(i) for i in rounds_to_process.values())
        valid_payloads = sum(res for res in results if isinstance(res, bool) and res) # Filter out Nones
        
        if total_payloads > 0:
            valid_percentage = (valid_payloads / total_payloads) * 100
            logger.info(f"Decryption round complete. Valid payloads: {valid_payloads}/{total_payloads} ({valid_percentage:.1f}%)")

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

    def get_training_data(self, max_block_number: int | None = None) -> Dict[str, tuple[dict[int, list], list[float]]] | None:
        """
        Get training data for all assets with price change filtering.
        
        Returns a dictionary mapping asset names to (history_dict, returns) tuples.
        """
        if not self.plaintext_cache:
            logger.warning("Not enough data to create a training set (no plaintext cache).")
            return None

        if max_block_number is not None:
            end_idx = next((i for i, b in enumerate(self.blocks) if b >= max_block_number), len(self.blocks))
            if end_idx == 0:
                logger.warning(f"No data available before block {max_block_number}.")
                return None
            
            blocks = self.blocks[:end_idx]
            asset_prices = self.asset_prices[:end_idx] if self.asset_prices else []
            plaintext_cache = self.plaintext_cache[:end_idx]
        else:
            blocks = self.blocks
            asset_prices = self.asset_prices if self.asset_prices else []
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
        asset_prices = asset_prices[TIMESTEPS_TO_SKIP:] if asset_prices else []
        plaintext_cache = plaintext_cache[TIMESTEPS_TO_SKIP:]

        if len(blocks) < config.LAG * 2 + 1:
            logger.warning("Not enough data to create a training set after filtering and skipping initial timesteps.")
            return None

        if not asset_prices:
            logger.warning("No asset price data available.")
            return None

        T = len(blocks)
        all_uids = self._get_all_uids_unsafe()
        
        result = {}
        
        for asset in config.ASSETS:
            logger.info(f"Processing training data for {asset}")
            
            price_series = []
            for t in range(T):
                if t < len(asset_prices) and asset in asset_prices[t]:
                    price_series.append(asset_prices[t][asset])
                else:
                    price_series = None
                    break
            
            if price_series is None:
                logger.warning(f"Missing price data for {asset}, skipping")
                continue
            
            price_series = self._filter_unchanged_prices(price_series, config.MAX_UNCHANGED_TIMESTEPS)
            
            asset_returns = []
            valid_embedding_indices = []
            
            for t in range(len(price_series) - config.LAG):
                p_initial = price_series[t]
                p_final = price_series[t + config.LAG]

                if not np.isnan(p_initial) and not np.isnan(p_final):
                    if p_initial > 0:
                        asset_returns.append((p_final - p_initial) / p_initial)
                        valid_embedding_indices.append(t)

            if not asset_returns:
                logger.warning(f"No valid returns calculated for {asset} after filtering.")
                continue
            
            history_dict = {uid: [] for uid in all_uids}
            
            for t_idx in valid_embedding_indices:
                for uid in all_uids:
                    vector = [0.0] * config.ASSET_EMBEDDING_DIMS[asset]
                    if t_idx < len(plaintext_cache):
                        cache_entry = plaintext_cache[t_idx].get(uid, {})
                        if isinstance(cache_entry, dict) and asset in cache_entry:
                            vector = cache_entry[asset]
                    history_dict[uid].append(vector)

            result[asset] = (history_dict, asset_returns)
            logger.info(f"Created training data for {asset}: {len(asset_returns)} samples")
        
        if not result:
            logger.warning("No training data generated for any asset")
            return None
            
        return result

    def _filter_unchanged_prices(self, prices: List[float], max_unchanged: int) -> np.ndarray:
        """
        Filter out periods where the price is stagnant for more than `max_unchanged`
        timesteps by replacing them with np.nan.
        
        Returns a NumPy array of prices with stagnant periods nulled out.
        """
        if not prices:
            return np.array([])
        
        price_array = np.array(prices, dtype=np.float64)
        is_stagnant = np.zeros_like(price_array, dtype=bool)
        unchanged_count = 0
        last_price = None
        
        for i, price in enumerate(price_array):
            if last_price is None or price != last_price:
                unchanged_count = 0
            else:
                unchanged_count += 1
            
            if unchanged_count > max_unchanged:
                is_stagnant[i] = True
            
            last_price = price
            
        price_array[is_stagnant] = np.nan
        return price_array

    async def save(self, path: str) -> None:
        async with self._lock:
            datalog_copy = copy.deepcopy(self)

        try:
            await asyncio.to_thread(_save_datalog, datalog_copy, path)
            logger.info(f"✅ Datalog saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save datalog to {path}: {e}")

    @staticmethod
    def load(path: str) -> "DataLog":
        logger.info(f"Fetching latest datalog from R2 bucket: {config.DATALOG_ARCHIVE_URL}")
        try:
            url = config.DATALOG_ARCHIVE_URL
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            
            tmp_path = path + ".tmp.pkl.gz"
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            os.replace(tmp_path, path)
            logger.info(f"Downloaded and saved archive to {path}")

            with gzip.open(path, "rb") as f:
                log = pickle.load(f)
            logger.info(f"Loaded DataLog from {path}")
            return log

        except (requests.exceptions.RequestException, pickle.UnpicklingError, gzip.BadGzipFile) as e:
            logger.error(f"Failed to download or load archive from {config.DATALOG_ARCHIVE_URL}: {e}")
            logger.warning("Starting with a new, empty DataLog.")
            return DataLog()
        except Exception as e:
             logger.error(f"An unexpected error occurred during datalog load: {e}", exc_info=True)
             logger.warning("Starting with a new, empty DataLog due to unexpected error.")
             return DataLog() 

def _save_datalog(datalog: DataLog, path: str) -> None:
    """Synchronous function to save the datalog to a file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        temp_path = f"{path}.tmp.{os.getpid()}"
        
        with gzip.open(temp_path, "wb") as f:
            pickle.dump(datalog, f)
            
        os.rename(temp_path, path)
        
    except Exception as e:
        logger.error(f"Error in _save_datalog: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise 
