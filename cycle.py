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
import re

logger = logging.getLogger(__name__)

NETWORK = "finney"
sub = bt.subtensor(network=NETWORK)

MAX_PAYLOAD_BYTES = 25 * 1024 * 1024

R2_URL_RE = re.compile(
    r"^https://pub-[a-f0-9]{32}\.r2\.dev/([^/]+)$",
    re.IGNORECASE
)

def _r2_url_is_valid(url: str, expected_hotkey: str) -> bool:
    """
    Return True iff `url` is exactly of the form

        https://pub-19ef1d497c954cd79216cf8f3b43cd01.r2.dev/<hotkey>

    • scheme must be HTTPS
    • host must begin with 'pub-' and end with '.r2.dev'
    • bucket ID must be 32 hex chars
    • exactly ONE path segment (the miner's hotkey)
    """
    m = R2_URL_RE.match(url)
    return bool(m and m.group(1).lower() == expected_hotkey.lower())

async def get_miner_payloads(
    netuid: int = 123, mg: bt.metagraph = None
) -> dict[int, bytes]:
    """
    Fetches the current block's payloads from all active miners.

    This function is now stateless. It queries the subtensor for miner commitments
    and downloads the corresponding payloads, returning a simple dictionary
    mapping UID to the raw payload bytes. All state management is handled by the
    DataLog.

    Returns:
        A dictionary mapping miner UIDs to their payload bytes.
    """
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

        if not _r2_url_is_valid(object_url, hotkey):
            logger.warning("Rejecting malformed R2 URL: %s", object_url)
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


