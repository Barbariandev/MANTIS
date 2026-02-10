# MIT License
# Copyright (c) 2024 MANTIS

from __future__ import annotations

import argparse
import ctypes
import gc
import logging
import os

os.environ.setdefault("MALLOC_ARENA_MAX", "4")
import threading
import time
import asyncio
import copy
import json

import bittensor as bt
import torch
import aiohttp
from dotenv import load_dotenv
import numpy as np
import pickle

import config
from cycle import get_miner_payloads
from model import multi_salience as sal_fn
from ledger import DataLog, ensure_datalog

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "main.log"), mode="a"),
    ],
)

weights_logger = logging.getLogger("weights")
weights_logger.setLevel(logging.DEBUG)
weights_logger.addHandler(
    logging.FileHandler(os.path.join(LOG_DIR, "weights.log"), mode="a")
)
per_challenge_logger = logging.getLogger("per_challenge_weights")
per_challenge_logger.setLevel(logging.DEBUG)
per_challenge_logger.addHandler(
    logging.FileHandler(os.path.join(LOG_DIR, "per_challenge_weights.log"), mode="a")
)

for noisy in ("websockets", "aiohttp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

load_dotenv()

os.makedirs(config.STORAGE_DIR, exist_ok=True)
DATALOG_PATH = os.path.join(config.STORAGE_DIR, "mantis_datalog.db")
WEIGHTS_PATH = os.path.join(config.STORAGE_DIR, "saved_weights.pkl")
SAVE_INTERVAL = 480


def save_weights(weights_tensor: torch.Tensor, uids: list[int], block: int):
    weights_data = {
        "weights": weights_tensor,
        "uids": uids,
        "block": block,
    }
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump(weights_data, f)
    logging.info(f"Saved weights calculated at block {block} to {WEIGHTS_PATH}")


def load_weights() -> dict | None:
    if not os.path.exists(WEIGHTS_PATH):
        return None
    with open(WEIGHTS_PATH, "rb") as f:
        return pickle.load(f)


async def get_asset_prices(session: aiohttp.ClientSession) -> dict[str, float] | None:
    specs = list(config.CHALLENGES)
    base = {spec["ticker"]: 0.0 for spec in specs}
    try:
        async with session.get(config.PRICE_DATA_URL) as resp:
            resp.raise_for_status()
            text = await resp.text()
            data = json.loads(text)
            fetched = data.get("prices", {}) or {}
            out = dict(base)
            for spec in specs:
                ticker = spec["ticker"]
                price_key = spec.get("price_key", ticker)
                v = fetched.get(price_key)
                try:
                    fv = float(v)
                    if 0 < fv < float('inf'):
                        out[ticker] = fv
                except Exception:
                    pass
            for asset in getattr(config, "BREAKOUT_ASSETS", []):
                if asset not in out:
                    v = fetched.get(asset)
                    if isinstance(v, (int, float)) and 0 < v < float('inf'):
                        out[asset] = float(v)
            logging.info(f"Fetched prices, filled zeros where missing: {out}")
            return out
    except Exception as e:
        logging.error(f"Failed to fetch prices from {config.PRICE_DATA_URL}: {e}")
        return base

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wallet.name", required=True)
    p.add_argument("--wallet.hotkey", required=True)
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=config.NETUID)
    p.add_argument(
        "--do_save",
        action="store_true",
        default=False,
        help="Whether to save the datalog periodically."
    )
    p.add_argument(
        "--save-every-seconds",
        type=int,
        default=SAVE_INTERVAL * 12,
        help="How often to save the datalog, in seconds (default: SAVE_INTERVAL blocks * 12s).",
    )
    p.add_argument(
        "--skip-set-weights",
        action="store_true",
        default=False,
        help="Skip weight setting on-chain (useful for unregistered keys).",
    )
    args = p.parse_args()

    while True:
        try:
            sub = bt.Subtensor(network=args.network)
            wallet = bt.Wallet(name=getattr(args, "wallet.name"), hotkey=getattr(args, "wallet.hotkey"))
            mg = bt.Metagraph(netuid=args.netuid, network=args.network, sync=True)
            break
        except Exception as e:
            logging.exception("Subtensor connect failed")
            time.sleep(30)
            continue

    if not os.path.exists(DATALOG_PATH):
        try:
            logging.info(f"Attempting to download initial datalog from {config.DATALOG_ARCHIVE_URL} → {DATALOG_PATH}")
            ensure_datalog(DATALOG_PATH)
            logging.info("Downloaded datalog to local storage.")
        except Exception:
            logging.info("Remote datalog unavailable; starting with a new, empty ledger.")
    logging.info(f"Loading datalog from {DATALOG_PATH}...")
    datalog = DataLog.load(DATALOG_PATH)
        
    stop_event = asyncio.Event()

    try:
        asyncio.run(run_main_loop(args, sub, wallet, mg, datalog, stop_event))
    except KeyboardInterrupt:
        logging.info("Exit signal received. Shutting down.")
    finally:
        stop_event.set()
        if args.do_save:
            try:
                logging.info("Final save on shutdown...")
                asyncio.run(datalog.save(DATALOG_PATH))
            except Exception:
                logging.exception("Final save failed.")
        logging.info("Shutdown complete.")


async def decrypt_loop(datalog: DataLog, stop_event: asyncio.Event):
    logging.info("Decryption loop started (Drand signatures cached by default).")
    while not stop_event.is_set():
        try:
            await datalog.process_pending_payloads()
        except asyncio.CancelledError:
            break
        except Exception:
            logging.exception("An error occurred in the decryption loop.")
        await asyncio.sleep(5)
    logging.info("Decryption loop stopped.")


async def save_loop(datalog: DataLog, do_save: bool, save_every_seconds: int, stop_event: asyncio.Event):
    if not do_save:
        logging.info("DO_SAVE is False, skipping periodic saves.")
        return
    logging.info("Save loop started.")
    try:
        logging.info("Initiating initial datalog save...")
        await datalog.save(DATALOG_PATH)
    except Exception:
        logging.exception("Initial save failed.")
    while not stop_event.is_set():
        try:
            await asyncio.sleep(save_every_seconds)
            logging.info("Initiating periodic datalog save...")
            await datalog.save(DATALOG_PATH)
        except asyncio.CancelledError:
            break
        except Exception:
            logging.exception("An error occurred in the save loop.")
    logging.info("Save loop stopped.")


subtensor_lock = threading.Lock()


async def get_current_block_with_retry(sub: bt.Subtensor, lock: threading.Lock, timeout: int = 10) -> int:
    retry_delay = 5
    while True:
        try:
            def get_block():
                with lock:
                    return sub.get_current_block()

            current_block = await asyncio.wait_for(
                asyncio.to_thread(get_block),
                timeout=float(timeout)
            )
            return current_block
        except asyncio.TimeoutError:
            logging.warning(
                f"Getting current block timed out after {timeout}s. "
                f"Retrying in {retry_delay}s..."
            )
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while getting current block: {e}. "
                f"Retrying in {retry_delay}s..."
            )
        await asyncio.sleep(retry_delay)


async def run_main_loop(
    args: argparse.Namespace,
    sub: bt.Subtensor,
    wallet: bt.Wallet,
    mg: bt.Metagraph,
    datalog: DataLog,
    stop_event: asyncio.Event,
):
    last_block = await get_current_block_with_retry(sub, subtensor_lock)
    weight_thread: threading.Thread | None = None

    tasks = [
        asyncio.create_task(decrypt_loop(datalog, stop_event)),
        asyncio.create_task(save_loop(datalog, args.do_save, args.save_every_seconds, stop_event)),
    ]

    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                current_block = await get_current_block_with_retry(sub, subtensor_lock)
                
                if current_block == last_block:
                    await asyncio.sleep(1)
                    continue
                last_block = current_block

                if current_block % config.SAMPLE_EVERY != 0:
                    continue
                logging.info(f"Sampled block {current_block}")

                if current_block % 100 == 0:
                    with subtensor_lock:
                        mg.sync(subtensor=sub)
                    logging.info("Metagraph synced.")
                    async with datalog._lock:
                        datalog.prune_hotkeys(mg.hotkeys)

                asset_prices = await get_asset_prices(session)
                if not asset_prices:
                    logging.error("Failed to fetch prices for required assets.")
                    continue
                
                payloads = await get_miner_payloads(netuid=args.netuid, mg=mg)
                logging.info(f"Retrieved {len(payloads)} payloads from miners. Hotkey sample: {list(payloads.keys())[:3]}")
                await datalog.append_step(current_block, asset_prices, payloads, mg)

                if (
                    current_block % config.WEIGHT_CALC_INTERVAL == 0
                    and (weight_thread is None or not weight_thread.is_alive())
                    and datalog.block_count >= config.LAG * 2 + 1
                ):
                    def calc_worker(db_path, block_snapshot, metagraph, cli_args):
                        _t0 = time.monotonic()
                        max_block = block_snapshot - config.TASK_INTERVAL

                        weights_logger.info(f"=== Starting weight calculation | block {block_snapshot} ===")
                        weights_logger.info("Streaming training data from SQLite (one challenge at a time)...")

                        training_iter = DataLog.iter_challenge_training_data(db_path, max_block_number=max_block)

                        _t1 = time.monotonic()
                        general_sal_hk, per_challenge = sal_fn(training_iter, return_breakdown=True)
                        weights_logger.info(
                            f"Salience took {time.monotonic()-_t1:.1f}s, "
                            f"hotkeys={len(general_sal_hk)}, "
                            f"challenges={list(per_challenge.keys())}"
                        )
                        if per_challenge:
                            total_w = float(
                                sum(float(config.CHALLENGE_MAP.get(t, {}).get("weight", 1.0)) for t in per_challenge.keys())
                            )
                            per_challenge_logger.info(
                                json.dumps(
                                    {
                                        "block": int(block_snapshot),
                                        "total_challenge_weight": total_w,
                                        "per_challenge": {
                                            t: {
                                                "challenge_weight": float(config.CHALLENGE_MAP.get(t, {}).get("weight", 1.0)),
                                                "salience": s,
                                                "weighted": {
                                                    hk: float(v) * float(config.CHALLENGE_MAP.get(t, {}).get("weight", 1.0))
                                                    for hk, v in s.items()
                                                },
                                            }
                                            for t, s in per_challenge.items()
                                        },
                                    },
                                    sort_keys=True,
                                )
                            )
                        if not general_sal_hk:
                            weights_logger.info("Salience computation returned empty.")

                        hk2uid = {hk: uid for uid, hk in zip(metagraph.uids.tolist(), metagraph.hotkeys)}
                        general_sal = {hk2uid.get(hk, -1): s for hk, s in general_sal_hk.items() if hk in hk2uid}
                        sal = {uid: score for uid, score in general_sal.items() if uid != -1}

                        if not sal:
                            weights_logger.warning("Salience is empty. Cannot calculate weights - insufficient training data.")
                            return

                        uids = metagraph.uids.tolist()

                        young_threshold = 36000
                        hotkey_first_block = DataLog.get_hotkey_first_blocks_from_db(
                            db_path, int(config.SAMPLE_EVERY),
                        )

                        young_uids = set()
                        for hk, first_block in hotkey_first_block.items():
                            age_blocks = block_snapshot - int(first_block)
                            if age_blocks < young_threshold:
                                uid = hk2uid.get(hk)
                                if uid is not None:
                                    young_uids.add(uid)
                        weights_logger.info(f"Found {len(young_uids)} young UIDs by hotkey-first-nonzero (<{young_threshold} blocks).")

                        mature_uid_scores = {uid: score for uid, score in sal.items() if uid not in young_uids}
                        
                        total_mature_score = sum(mature_uid_scores.values())
                        
                        fixed_weight_per_young_uid = 0.0001
                        
                        active_young_uids = {uid for uid in young_uids if uid in uids}
                        
                        weight_for_mature_uids = 1.0 - (len(active_young_uids) * fixed_weight_per_young_uid)
                        weight_for_mature_uids = max(0, weight_for_mature_uids)

                        final_weights = {}

                        if total_mature_score > 0 and weight_for_mature_uids > 0:
                            for uid, score in mature_uid_scores.items():
                                if uid in uids:
                                    final_weights[uid] = (score / total_mature_score) * weight_for_mature_uids
                        
                        for uid in active_young_uids:
                            final_weights[uid] = fixed_weight_per_young_uid

                        if not final_weights:
                            weights_logger.warning("No weights to set after processing.")
                            return

                        total_weight = sum(final_weights.values())
                        if total_weight <= 0:
                            weights_logger.warning("Total calculated weight is zero or negative, skipping set.")
                            return
                        
                        normalized_weights = {uid: w / total_weight for uid, w in final_weights.items()}

                        burn_pct = float(getattr(config, "BURN_PCT", 0.0))
                        if burn_pct > 0:
                            for uid in normalized_weights:
                                normalized_weights[uid] *= (1.0 - burn_pct)
                            normalized_weights[0] = normalized_weights.get(0, 0.0) + burn_pct
                            weights_logger.info(f"Reserved {burn_pct*100:.0f}% for UID 0 (burn)")

                        w = torch.tensor([normalized_weights.get(uid, 0.0) for uid in uids], dtype=torch.float32)
                        
                        non_zero_count = (w > 0).sum().item()
                        if non_zero_count > 0:
                            unique_values = torch.unique(w[w > 0])
                            if len(unique_values) == 1:
                                weights_logger.warning(f"Detected degenerate uniform weights ({unique_values[0]:.6f}). Skipping set to allow averaging with other challenges.")
                                return
                            if non_zero_count > 1:
                                mean_val = w[w > 0].mean()
                                std_val = w[w > 0].std()
                                if mean_val > 0 and (std_val / mean_val) < 0.001:
                                    weights_logger.warning(f"Detected near-uniform weights (CV < 0.1%). Skipping set.")
                                    return

                        if w.sum() > 0:
                            final_w = w / w.sum()
                        else:
                            weights_logger.warning("Zero-sum weights tensor, skipping set.")
                            return
                        
                        weights_to_log = {uid: f"{weight:.8f}" for uid, weight in normalized_weights.items() if uid in uids and weight > 0}
                        weights_logger.info(f"Normalized weights for block {block_snapshot}: {json.dumps(weights_to_log)}")
                        weights_logger.info(f"Final tensor sum: {final_w.sum().item()}")
                        
                        save_weights(final_w, uids, block_snapshot)
                        weights_logger.info(f"Weights calculated and saved at block {block_snapshot} (max={final_w.max():.4f})")

                        del general_sal_hk, per_challenge, sal
                        del final_w, normalized_weights, w
                        gc.collect()
                        try:
                            ctypes.CDLL("libc.so.6").malloc_trim(0)
                        except OSError:
                            pass
                        weights_logger.info("Post-calc memory cleanup done.")

                    weight_thread = threading.Thread(
                        target=calc_worker,
                        args=(DATALOG_PATH, current_block, copy.deepcopy(mg), copy.deepcopy(args)),
                        daemon=True,
                    )
                    weight_thread.start()
                
                if current_block % config.WEIGHT_SET_INTERVAL == 0:
                    if args.skip_set_weights:
                        logging.info(f"Block {current_block}: --skip-set-weights active, not setting weights on-chain.")
                    else:
                        weights_data = load_weights()
                        if weights_data is None:
                            logging.info(f"No saved weights found at block {current_block}, skipping weight setting.")
                        else:
                            calc_block = weights_data.get("block", "unknown")
                            final_w = weights_data["weights"]
                            saved_uids = weights_data["uids"]
                            
                            weights_logger.info(f"Setting weights from saved array (calculated at block {calc_block})")
                            
                            if list(saved_uids) != mg.uids.tolist():
                                weights_logger.warning("UID mismatch between saved weights and current metagraph, skipping.")
                            else:
                                sub.set_weights(
                                    netuid=args.netuid,
                                    wallet=wallet,
                                    uids=mg.uids,
                                    weights=final_w,
                                    wait_for_inclusion=False,
                                )
                                weights_logger.info(f"Weights set at block {current_block} (from block {calc_block}, max={final_w.max():.4f})")

            except KeyboardInterrupt:
                stop_event.set()
            except Exception:
                logging.error("Error in main loop", exc_info=True)
                await asyncio.sleep(10)
    
    logging.info("Main loop finished. Cleaning up background tasks.")
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    main()





