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
from dotenv import load_dotenv
import boto3
import logging
import os
import json
import asyncio
import botocore
from typing import Dict, List
try:
    from aiobotocore.session import get_session
except ImportError:
    get_session = None
import aiohttp, email.utils
import base64
from pathlib import Path
from botocore.client import Config
import re
from urllib.parse import urlparse

import config

logger = logging.getLogger(__name__)

_ENV_CACHE: Dict[str, str | None] = {}


def _cached_env(key: str) -> str | None:
    if key not in _ENV_CACHE:
        try:
            _ENV_CACHE[key] = os.getenv(key)
        except Exception:
            _ENV_CACHE[key] = None
    return _ENV_CACHE[key]


def bucket() -> str | None:
    return _cached_env("R2_BUCKET_ID")


def load_r2_account_id() -> str | None:
    return _cached_env("R2_ACCOUNT_ID")


def load_r2_endpoint_url() -> str | None:
    account_id = load_r2_account_id()
    return f"https://{account_id}.r2.cloudflarestorage.com" if account_id else None


def load_r2_write_access_key_id() -> str | None:
    return _cached_env("R2_WRITE_ACCESS_KEY_ID")


def load_r2_write_secret_access_key() -> str | None:
    return _cached_env("R2_WRITE_SECRET_ACCESS_KEY")

CLIENT_CONFIG = botocore.config.Config(max_pool_connections=256)
session = get_session() if get_session else None


def _r2_client_kwargs():
    endpoint = load_r2_endpoint_url()
    access_key = load_r2_write_access_key_id()
    secret_key = load_r2_write_secret_access_key()
    if not endpoint or not access_key or not secret_key:
        return None
    return dict(
        endpoint_url=endpoint,
        region_name="enam",
        config=CLIENT_CONFIG,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _r2_client():
    kwargs = _r2_client_kwargs()
    if not (session and kwargs):
        return None
    return session.create_client("s3", **kwargs)


def get_local_path(bucket: str, filename: str) -> str:
    return os.path.join(config.STORAGE_DIR, "local_cache", bucket, filename)


async def exists_locally(bucket: str, filename: str) -> bool:
    return os.path.exists(get_local_path(bucket, filename))


async def delete_locally(bucket: str, filename: str) -> None:
    if os.path.exists(path := get_local_path(bucket, filename)):
        try:
            await asyncio.to_thread(os.remove, path)
        except Exception:
            pass

async def load(bucket: str, filename: str) -> dict | None:
    path = Path(get_local_path(bucket, filename))
    try:
        return await asyncio.to_thread(lambda p: json.loads(p.read_text()), path)
    except Exception:
        return None


async def _local_path_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc
    safe_host = re.sub(r'[^a-zA-Z0-9_.-]+', '_', host)
    path_filename = os.path.basename(parsed.path)
    if not path_filename or '..' in path_filename or '/' in path_filename:
         raise ValueError(f"Invalid path component in URL: {path_filename}")
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]+', '_', path_filename)
    return os.path.join(config.STORAGE_DIR, "url_cache", safe_host, safe_filename)

async def _object_size(url: str, session: aiohttp.ClientSession, timeout: int = 10) -> int | None:
    try:
        async with session.head(url, timeout=timeout, headers={"Accept-Encoding": "identity"}) as r:
            if r.status == 200:
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit():
                    return int(cl)
        async with session.get(url, timeout=timeout, headers={"Accept-Encoding": "identity", "Range": "bytes=0-0"}) as r:
            if r.status in (200, 206):
                cr = r.headers.get("Content-Range")
                if cr:
                    m = re.match(r"bytes \d+-\d+/(\d+)", cr)
                    if m:
                        return int(m.group(1))
    except Exception:
        return None
    return None

def _is_v2_payload(d: dict) -> bool:
    need = {"v", "round", "hk", "owner_pk", "C", "W_owner", "W_time", "binding", "alg"}
    return (
        isinstance(d, dict)
        and d.get("v") == 2
        and isinstance(d.get("round"), int)
        and need.issubset(d.keys())
    )


async def download(url: str, max_size_bytes: int | None = None):
    path = await _local_path_from_url(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiohttp.ClientSession() as s:
        if max_size_bytes is not None and max_size_bytes > 0:
            try:
                sz = await _object_size(url, s)
                if sz is not None and sz > max_size_bytes:
                    raise ValueError(f"Object size {sz} exceeds limit {max_size_bytes}")
            except Exception as e:
                logger.warning("Size check failed for %s: %s", url, e)
                raise
        async with s.get(url, timeout=600) as r:
            r.raise_for_status()
            body = await r.read()
    try:
        data = json.loads(body.decode("utf-8"))
        if not _is_v2_payload(data):
            raise ValueError("Payload must be a v2 JSON object.")
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        logger.warning(f"Invalid payload from {url}: {e}")
        raise ValueError("Invalid payload format") from e
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
    return data




async def exists(bucket: str, filename: str) -> bool:
    client_ctx = _r2_client()
    if not client_ctx:
        return False
    try:
        async with client_ctx as s3_client:
            await s3_client.head_object(Bucket=bucket, Key=filename)
            return True
    except botocore.exceptions.ClientError:
        return False
    except Exception:
        return False
    
    
async def list(bucket: str, prefix: str) -> List[str]:
    client_ctx = _r2_client()
    if not client_ctx:
        return []
    keys: List[str] = []
    try:
        async with client_ctx as s3_client:
            paginator = s3_client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                keys.extend(obj.get("Key", "") for obj in page.get("Contents", []))
    except Exception:
        pass
    return keys

async def timestamp(resource: str, filename: str | None = None):
    if filename is not None:
        client_ctx = _r2_client()
        if not client_ctx:
            return None
        try:
            async with client_ctx as s3_client:
                resp = await s3_client.head_object(Bucket=resource, Key=filename)
                return resp.get("LastModified")
        except Exception:
            return None
    try:
        async with aiohttp.ClientSession() as s:
            async with s.head(resource, timeout=10) as r:
                if r.status == 200:
                    lm = r.headers.get("Last-Modified")
                    return email.utils.parsedate_to_datetime(lm) if lm else None
    except Exception:
        return None



def _sanitize_b64(obj):
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, dict):
        return {k: _sanitize_b64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_b64(v) for v in obj]
    return obj


def upload(bucket: str, object_key: str, file_path: str | Path) -> None:

    load_dotenv()

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_WRITE_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_WRITE_SECRET_ACCESS_KEY"]

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    logger.info("Uploading %s to bucket %s as %s", file_path, bucket, object_key)

    s3.upload_file(str(file_path), bucket, object_key)
    logger.info("✅ Uploaded %s → s3://%s/%s", file_path, bucket, object_key)









