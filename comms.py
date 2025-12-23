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
    async with session.head(url, timeout=timeout, headers={"Accept-Encoding": "identity"}) as r:
        if r.status == 200:
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit():
                return int(cl)
    async with session.get(url, timeout=timeout, headers={"Accept-Encoding": "identity", "Range": "bytes=0-0"}) as r:
        if r.status in (200, 206):
            cr = r.headers.get("Content-Range")
            if cr:
                m = re.match(r"bytes \\d+-\\d+/(\\d+)", cr)
                if m:
                    return int(m.group(1))
    return None


async def download(url: str, max_size_bytes: int | None = None) -> dict:
    """
    Download a v2 JSON payload from a URL (used by cycle.py).

    Raises on:
      - HTTP errors
      - size limit exceeded (if max_size_bytes is set)
      - invalid JSON / invalid v2 payload schema
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL")

    async with aiohttp.ClientSession() as s:
        if max_size_bytes is not None and int(max_size_bytes) > 0:
            sz = await _object_size(url, s)
            if sz is not None and sz > int(max_size_bytes):
                raise ValueError(f"Object size {sz} exceeds limit {int(max_size_bytes)}")
        async with s.get(url, timeout=600) as r:
            r.raise_for_status()
            body = await r.read()

    data = json.loads(body.decode("utf-8"))
    if not _is_v2_payload(data):
        logger.warning("Invalid payload from %s (not v2)", url)
        raise ValueError("Payload must be a v2 JSON object.")
    return data


