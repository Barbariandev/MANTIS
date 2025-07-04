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
import queue
import asyncio
import copy

import bittensor as bt
import torch
import requests
from dotenv import load_dotenv
import aiohttp

import config
from cycle import get_miner_payloads
from model import salience as sal_fn
from storage import DataLog

LOG_DIR = os.path.expanduser("~/new_system_mantis")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
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

DATALOG_PATH = os.path.expanduser("~/mantis_datalog.pkl.gz")
PROCESS_INTERVAL = 10
SAVE_INTERVAL = 100


async def get_btc_price(session: aiohttp.ClientSession) -> float | None:
    sources = {
        "Binance": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        "Coinbase": "https://api.coinbase.com/v2/prices/BTC-USDT/spot",
        "Kraken": "https://api.kraken.com/0/public/Ticker?pair=XBTUSDT",
    }

    async def _fetch(name, url):
        try:
            async with session.get(url, timeout=5) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if name == "Binance":
                    return float(data["price"])
                if name == "Coinbase":
                    return float(data["data"]["amount"])
                if name == "Kraken":
                    pair = list(data["result"].keys())[0]
                    return float(data["result"][pair]["c"][0])
        except Exception as e:
            logging.debug(f"Failed to fetch from {name}: {e}")
            return None
    
    tasks = [asyncio.create_task(_fetch(name, url)) for name, url in sources.items()]
    
    for task in asyncio.as_completed(tasks):
        price = await task
        if price is not None:
            for t in tasks:
                if not t.done():
                    t.cancel()
            logging.info(f"Fetched BTC price: {price}")
            return price

    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wallet.name", required=True)
    p.add_argument("--wallet.hotkey", required=True)
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=config.NETUID)
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

    datalog = DataLog.load(DATALOG_PATH)
    stop_event = asyncio.Event()

    try:
        asyncio.run(run_main_loop(args, sub, wallet, mg, datalog, stop_event))
    except KeyboardInterrupt:
        logging.info("Exit signal received. Shutting down.")
    except Exception as e:
        logging.error(f"An unexpected error forced the main loop to exit: {e}", exc_info=True)
    finally:
        stop_event.set()
        logging.info("Shutdown complete.")


async def decrypt_loop(datalog: DataLog, stop_event: asyncio.Event):
    logging.info("Decryption loop started.")
    while not stop_event.is_set():
        try:
            await datalog.process_pending_payloads()
        except asyncio.CancelledError:
            logging.info("Decrypt loop cancelled.")
            break
        except Exception:
            logging.exception("An error occurred in the decryption loop.")
        await asyncio.sleep(5)
    logging.info("Decryption loop stopped.")


async def save_loop(datalog: DataLog, stop_event: asyncio.Event):
    logging.info("Save loop started.")
    save_interval_seconds = SAVE_INTERVAL * 12
    while not stop_event.is_set():
        try:
            await asyncio.sleep(save_interval_seconds)
            logging.info("Initiating periodic datalog save...")
            await datalog.save(DATALOG_PATH)
        except asyncio.CancelledError:
            logging.info("Save loop cancelled.")
            break
        except Exception:
            logging.exception("An error occurred in the save loop.")
    logging.info("⏹️ Save loop stopped.")


# Create a global lock for subtensor access
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

    decrypt_task = asyncio.create_task(decrypt_loop(datalog, stop_event))
    save_task = asyncio.create_task(save_loop(datalog, stop_event))
    background_tasks = [decrypt_task, save_task]

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

                price = await get_btc_price(session)
                if price is None:
                    logging.error("Failed to fetch BTC price from all sources.")
                    continue

                payloads = await get_miner_payloads(netuid=args.netuid, mg=mg)
                await datalog.append_step(current_block, price, payloads)

                if (
                    current_block % config.TASK_INTERVAL == 0
                    and (weight_thread is None or not weight_thread.is_alive())
                    and len(datalog.blocks) >= config.LAG * 2 + 1
                ):

                    def worker(
                        training_data_snapshot: tuple | None,
                        block_snapshot: int,
                        metagraph: bt.metagraph,
                        network: str,
                        netuid: int,
                        wallet_name: str,
                        wallet_hotkey: str,
                    ):
                        if not training_data_snapshot:
                            weights_logger.warning("Not enough data to compute salience.")
                            return
                        
                        weights_logger.info(
                            f"=== Weight computation start | block {block_snapshot} ==="
                        )

                        history, btc_returns = training_data_snapshot
                        if not history:
                            weights_logger.warning("Training data was empty.")
                            return

                        sal = sal_fn(history, btc_returns)
                        if not sal:
                            weights_logger.info("Salience unavailable.")
                            return

                        uids_to_set = metagraph.uids.tolist()
                        w = torch.tensor(
                            [sal.get(uid, 0.0) for uid in uids_to_set],
                            dtype=torch.float32,
                        )
                        if w.sum() <= 0:
                            weights_logger.warning("Zero-sum weights, skipping.")
                            return

                        w_norm = w / w.sum()
                        
                        # Create a new subtensor instance for this thread
                        try:
                            thread_sub = bt.subtensor(network=network)
                            thread_wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
                            
                            thread_sub.set_weights(
                                netuid=netuid,
                                wallet=thread_wallet,
                                uids=uids_to_set,
                                weights=w_norm,
                                wait_for_inclusion=False,
                            )
                            weights_logger.info(
                                f"Weights set at block {block_snapshot} (max={w_norm.max():.4f})"
                            )
                        except Exception as e:
                            weights_logger.error(f"Failed to set weights: {e}", exc_info=True)

                    max_block_for_training = current_block - config.TASK_INTERVAL
                    async with datalog._lock:
                        training_data = datalog.get_training_data(
                            max_block_number=max_block_for_training
                        )
                        training_data_copy = copy.deepcopy(training_data)

                    weight_thread = threading.Thread(
                        target=worker,
                        args=(
                            training_data_copy, 
                            current_block, 
                            mg,
                            args.network,
                            args.netuid,
                            getattr(args, "wallet.name"),
                            getattr(args, "wallet.hotkey"),
                        ),
                        daemon=True,
                    )
                    weight_thread.start()

            except KeyboardInterrupt:
                logging.info("Keyboard interrupt in main loop. Signaling shutdown.")
                stop_event.set()
            except Exception as e:
                logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    logging.info("Main loop finished. Waiting for background tasks to stop...")
    for task in background_tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*background_tasks, return_exceptions=True)


if __name__ == "__main__":
    main()
