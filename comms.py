from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)


def _is_v2_payload(d: Any) -> bool:
    need = {"v", "round", "hk", "owner_pk", "C", "W_owner", "W_time", "binding", "alg"}
    return isinstance(d, dict) and d.get("v") == 2 and isinstance(d.get("round"), int) and need.issubset(d.keys())


async def _object_size(url: str, session: aiohttp.ClientSession, timeout: int = 10) -> int | None:
    req_timeout = aiohttp.ClientTimeout(total=timeout)
    async with session.head(url, timeout=req_timeout, allow_redirects=False,
                            headers={"Accept-Encoding": "identity"}) as r:
        if r.status == 200:
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit():
                return int(cl)
    async with session.get(url, timeout=req_timeout, allow_redirects=False,
                           headers={"Accept-Encoding": "identity", "Range": "bytes=0-0"}) as r:
        if r.status in (200, 206):
            cr = r.headers.get("Content-Range")
            if cr:
                m = re.match(r"bytes \d+-\d+/(\d+)", cr)
                if m:
                    return int(m.group(1))
    return None


async def download(url: str, max_size_bytes: int | None = None) -> dict:
    """
    Download a v2 JSON payload from a URL (used by cycle.py).

    Raises on:
      - HTTP errors or unexpected redirects
      - size limit exceeded (checked both pre-flight and mid-stream)
      - invalid JSON / invalid v2 payload schema
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL")

    cap = int(max_size_bytes) if (max_size_bytes is not None and int(max_size_bytes) > 0) else None
    get_timeout = aiohttp.ClientTimeout(total=45, connect=10, sock_read=20)
    async with aiohttp.ClientSession() as s:
        if cap is not None:
            sz = await _object_size(url, s)
            if sz is not None and sz > cap:
                raise ValueError(f"Object size {sz} exceeds limit {cap}")
        async with s.get(url, timeout=get_timeout, allow_redirects=False) as r:
            if 300 <= r.status < 400:
                raise ValueError(f"Redirects are not permitted for miner payloads ({r.status})")
            r.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            async for chunk in r.content.iter_chunked(65536):
                total += len(chunk)
                if cap is not None and total > cap:
                    raise ValueError(f"Streamed payload exceeded limit {cap} bytes")
                chunks.append(chunk)
            body = b"".join(chunks)

    data = json.loads(body.decode("utf-8"))
    if not _is_v2_payload(data):
        logger.warning("Invalid payload from %s (not v2)", url)
        raise ValueError("Payload must be a v2 JSON object.")
    return data


