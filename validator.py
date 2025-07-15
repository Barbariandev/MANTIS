# MIT License
#
# Copyright (c) 2024 MANTIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import asyncio
import copy
import json

import bittensor as bt
import torch
import aiohttp
from dotenv import load_dotenv

import config
from cycle import get_miner_payloads
from model import salience as sal_fn
from storage import DataLog

LOG_DIR = os.path.expanduser("~/new_system_mantis")
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

for noisy in ("websockets", "aiohttp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

load_dotenv()

os.makedirs(config.STORAGE_DIR, exist_ok=True)
DATALOG_PATH = os.path.join(config.STORAGE_DIR, "mantis_datalog.pkl.gz")
SAVE_INTERVAL = 120


async def _fetch_price_source(session, url, parse_json=True):
    async with session.get(url, timeout=5) as resp:
        resp.raise_for_status()
        if parse_json:
            return await resp.json()
        else:
            return await resp.text()

async def _get_price_from_sources(session, source_list):
    for name, url, parser in source_list:
        try:
            parse_json = not url.endswith("e=csv")
            data = await _fetch_price_source(session, url, parse_json=parse_json)
            price = parser(data)
            if price is not None:
                return price
        except Exception:
            continue
    return None


async def get_asset_prices(session: aiohttp.ClientSession) -> dict[str, float] | None:
    try:
        async with session.get(config.PRICE_DATA_URL) as resp:
            resp.raise_for_status()
            text = await resp.text()
            data = json.loads(text)
            prices = data.get("prices", {})
            logging.info(f"Fetched prices for {len(prices)} assets: {prices}")
            return prices
    except Exception as e:
        logging.error(f"Failed to fetch prices from {config.PRICE_DATA_URL}: {e}")
        return {}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wallet.name", required=True)
    p.add_argument("--wallet.hotkey", required=True)
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=config.NETUID)
    p.add_argument(
        "--no_download_datalog", 
        action="store_true", 
        help="Start with a fresh datalog instead of downloading from the archive."
    )
    p.add_argument(
        "--do_save",
        action="store_true",
        default=False,
        help="Whether to save the datalog periodically."
    )
    args = p.parse_args()

    while True:
        try:
            sub = bt.subtensor(network=args.network)
            wallet = bt.wallet(name=getattr(args, "wallet.name"), hotkey=getattr(args, "wallet.hotkey"))
            mg = bt.metagraph(netuid=args.netuid, network=args.network, sync=True)
            break
        except Exception as e:
            logging.exception("Subtensor connect failed")
            time.sleep(30)
            continue

    if args.no_download_datalog:
        logging.info("`--no_download_datalog` flag set. Starting with a new, empty DataLog.")
        datalog = DataLog()
    else:
        datalog = DataLog.load(DATALOG_PATH)
        
    stop_event = asyncio.Event()

    try:
        asyncio.run(run_main_loop(args, sub, wallet, mg, datalog, stop_event))
    except KeyboardInterrupt:
        logging.info("Exit signal received. Shutting down.")
    finally:
        stop_event.set()
        logging.info("Shutdown complete.")


async def decrypt_loop(datalog: DataLog, mg: bt.metagraph, stop_event: asyncio.Event):
    logging.info("Decryption loop started.")
    while not stop_event.is_set():
        try:
            uid_to_hotkey = dict(zip(mg.uids.tolist(), mg.hotkeys))
            await datalog.process_pending_payloads(uid_to_hotkey=uid_to_hotkey)
        except asyncio.CancelledError:
            break
        except Exception:
            logging.exception("An error occurred in the decryption loop.")
        await asyncio.sleep(5)
    logging.info("Decryption loop stopped.")


async def save_loop(datalog: DataLog, do_save: bool, stop_event: asyncio.Event):
    if not do_save:
        logging.info("DO_SAVE is False, skipping periodic saves.")
        return
    logging.info("Save loop started.")
    save_interval_seconds = SAVE_INTERVAL * 12
    while not stop_event.is_set():
        try:
            await asyncio.sleep(save_interval_seconds)
            logging.info("Initiating periodic datalog save...")
            await datalog.save(DATALOG_PATH)
        except asyncio.CancelledError:
            break
        except Exception:
            logging.exception("An error occurred in the save loop.")
    logging.info("⏹️ Save loop stopped.")


subtensor_lock = threading.Lock()

async def run_main_loop(
    args: argparse.Namespace,
    sub: bt.subtensor,
    wallet: bt.wallet,
    mg: bt.metagraph,
    datalog: DataLog,
    stop_event: asyncio.Event,
):
    last_block = sub.get_current_block()
    weight_thread: threading.Thread | None = None

    tasks = [
        asyncio.create_task(decrypt_loop(datalog, mg, stop_event)),
        asyncio.create_task(save_loop(datalog, args.do_save, stop_event)),
    ]

    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                with subtensor_lock:
                    current_block = sub.get_current_block()
                
                if current_block == last_block:
                    await asyncio.sleep(1)
                    continue
                last_block = current_block

                if current_block % config.SAMPLE_STEP != 0:
                    continue
                logging.info(f"Sampled block {current_block}")

                if current_block % 100 == 0:
                    with subtensor_lock:
                        mg.sync(subtensor=sub)
                    logging.info("Metagraph synced.")
                    await datalog.sync_miners(dict(zip(mg.uids.tolist(), mg.hotkeys)))

                asset_prices = await get_asset_prices(session)
                if not asset_prices:
                    logging.error("Failed to fetch prices for required assets.")
                    continue

                payloads = await get_miner_payloads(netuid=args.netuid, mg=mg)
                await datalog.append_step(current_block, asset_prices, payloads)

                if (
                    current_block % config.TASK_INTERVAL == 0
                    and (weight_thread is None or not weight_thread.is_alive())
                    and len(datalog.blocks) >= config.LAG * 2 + 1
                ):
                    def worker(training_data, block_snapshot, metagraph, cli_args):
                        if not training_data:
                            weights_logger.warning("Not enough data for salience.")
                            return
                        
                        weights_logger.info(f"=== Starting multi-asset salience | block {block_snapshot} ===")
                        sal = sal_fn(training_data)

                        if not sal:
                            weights_logger.info("Salience computation returned empty.")
                            return
                        
                        w = torch.tensor([sal.get(uid, 0.0) for uid in metagraph.uids.tolist()], dtype=torch.float32)
                        if w.sum() <= 0:
                            weights_logger.warning("Zero-sum weights, skipping set.")
                            return
                        
                        try:
                            thread_sub = bt.subtensor(network=cli_args.network)
                            thread_wallet = bt.wallet(
                                name=getattr(cli_args, "wallet.name"), 
                                hotkey=getattr(cli_args, "wallet.hotkey")
                            )
                            thread_sub.set_weights(
                                netuid=cli_args.netuid, wallet=thread_wallet,
                                uids=metagraph.uids, weights=w / w.sum(),
                                wait_for_inclusion=False,
                            )
                            weights_logger.info(f"Weights set at block {block_snapshot} (max={w.max():.4f})")
                        except Exception as e:
                            weights_logger.error(f"Failed to set weights: {e}", exc_info=True)

                    max_block_for_training = current_block - config.TASK_INTERVAL
                    async with datalog._lock:
                        training_data_copy = copy.deepcopy(datalog.get_training_data(max_block_number=max_block_for_training))

                    weight_thread = threading.Thread(
                        target=worker,
                        args=(training_data_copy, current_block, copy.deepcopy(mg), copy.deepcopy(args)),
                        daemon=True,
                    )
                    weight_thread.start()

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
