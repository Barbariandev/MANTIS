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

import asyncio, bittensor as bt, requests, config, comms, logging, os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

NETWORK = "finney"
sub = bt.subtensor(network=NETWORK)

MAX_PAYLOAD_BYTES = 25 * 1024 * 1024

async def get_miner_payloads(
    netuid: int = 123, mg: bt.metagraph = None
) -> dict[int, dict]:
    if mg is None:
        mg = bt.metagraph(netuid=netuid, network=NETWORK, sync=True)
    
    commits = sub.get_all_commitments(netuid)
    uid2hot = dict(zip(mg.uids.tolist(), mg.hotkeys))
    payloads = {}

    async def _fetch_one(uid: int):
        hotkey = uid2hot.get(uid)
        object_url = commits.get(hotkey) if hotkey else None
        if not object_url:
            return

        try:
            parsed_url = urlparse(object_url)
            path = parsed_url.path

            if path.endswith('/'):
                logger.warning(f"UID {uid} commit URL must not be a directory: {object_url}")
                return
            
            path_parts = path.lstrip('/').split('/')
            if len(path_parts) != 1:
                logger.warning(f"UID {uid} commit URL must only contain the hotkey as the path: {object_url}")
                return

            object_name = path_parts[0]
            if object_name.lower() != (hotkey or "").lower():
                logger.warning(
                    f"UID {uid} commit URL filename '{object_name}' does not match hotkey"
                )
                return
                
        except Exception as e:
            logger.warning(f"UID {uid} commit URL validation failed for {object_url}: {e}")
            return

        try:
            payload_raw = await comms.download(object_url, max_size_bytes=MAX_PAYLOAD_BYTES)
            if payload_raw:
                payloads[uid] = payload_raw
        except Exception as e:
            logger.warning(f"Download failed for UID {uid} at {object_url}: {e}")

    await asyncio.gather(*(
        _fetch_one(int(u)) for u in mg.uids
    ), return_exceptions=True)

    return payloads


