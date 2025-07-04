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
from typing import List, Dict
from aiobotocore.session import get_session
import aiohttp, email.utils
import base64
from pathlib import Path
from botocore.client import Config
import re


# Module logger
logger = logging.getLogger(__name__)

R2_BUCKET_ID = None
def bucket():
    global R2_BUCKET_ID
    if R2_BUCKET_ID is not None:
        return R2_BUCKET_ID
    try:
        R2_BUCKET_ID = os.getenv("R2_BUCKET_ID")
        if not R2_BUCKET_ID:
            return None
        return R2_BUCKET_ID
    except Exception as e:
        return None

R2_ACCOUNT_ID = None
def load_r2_account_id():
    global R2_ACCOUNT_ID
    if R2_ACCOUNT_ID is not None:
        return R2_ACCOUNT_ID
    try:
        R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
        if not R2_ACCOUNT_ID:
            return None
        return R2_ACCOUNT_ID
    except Exception as e:
        return None

R2_ENDPOINT_URL = None
def load_r2_endpoint_url():
    global R2_ENDPOINT_URL
    account_id = load_r2_account_id()
    if not account_id:
        return None
    try:
        R2_ENDPOINT_URL = f"https://{account_id}.r2.cloudflarestorage.com"
        return R2_ENDPOINT_URL
    except Exception as e:
        return None

R2_WRITE_ACCESS_KEY_ID = None
def load_r2_write_access_key_id():
    global R2_WRITE_ACCESS_KEY_ID
    if R2_WRITE_ACCESS_KEY_ID is not None:
        return R2_WRITE_ACCESS_KEY_ID
    try:
        R2_WRITE_ACCESS_KEY_ID = os.getenv("R2_WRITE_ACCESS_KEY_ID")
        if not R2_WRITE_ACCESS_KEY_ID:
            return None
        return R2_WRITE_ACCESS_KEY_ID
    except Exception as e:
        return None

R2_WRITE_SECRET_ACCESS_KEY = None
def load_r2_write_secret_access_key():
    global R2_WRITE_SECRET_ACCESS_KEY
    if R2_WRITE_SECRET_ACCESS_KEY is not None:
        return R2_WRITE_SECRET_ACCESS_KEY
    try:
        R2_WRITE_SECRET_ACCESS_KEY = os.getenv("R2_WRITE_SECRET_ACCESS_KEY")
        if not R2_WRITE_SECRET_ACCESS_KEY:
            return None
        return R2_WRITE_SECRET_ACCESS_KEY
    except Exception as e:
        return None

CLIENT_CONFIG = botocore.config.Config(max_pool_connections=256)
session = get_session()
    
def get_local_path(bucket: str, filename: str ) -> str:
    return os.path.join(os.path.expanduser("~/storage"), bucket, filename)

async def exists_locally( bucket:str, filename: str ) -> bool:
    local_path = get_local_path( bucket, filename )
    return os.path.exists(local_path)

async def delete_locally( bucket:str, filename: str ) -> None:
    local_path = get_local_path( bucket, filename )
    if os.path.exists(local_path):
        try:
            await asyncio.to_thread(os.remove, local_path)
        except Exception as e:
            pass

async def load( bucket:str, filename: str) -> dict:
    local_path = get_local_path(bucket, filename)
    try:
        def load_json(path):
            with open(path, "r") as f:
                return json.load(f)
        data = await asyncio.to_thread(load_json, local_path)
        return data
    except Exception as e:
        return None


async def _local_path_from_url(url: str) -> str:
    """Map https://host/path -> ~/storage/host/path"""
    name = url.split("://", 1)[1]
    return os.path.join(os.path.expanduser("~/storage"), name)

async def _object_size(url: str, session: aiohttp.ClientSession, timeout: int = 10) -> int | None:
    """Return object size in bytes using HEAD or Range request, or None."""
    try:
        # HEAD first – many servers include Content-Length here.
        async with session.head(url, timeout=timeout, headers={"Accept-Encoding": "identity"}) as r:
            if r.status == 200:
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit():
                    return int(cl)
        # Fallback: Range 0-0 GET to retrieve Content-Range header.
        async with session.get(url, timeout=timeout, headers={
            "Accept-Encoding": "identity",
            "Range": "bytes=0-0",
        }) as r:
            if r.status in (200, 206):
                cr = r.headers.get("Content-Range")
                if cr:
                    m = re.match(r"bytes \d+-\d+/(\d+)", cr)
                    if m:
                        return int(m.group(1))
    except Exception:
        # Any failure → size unknown
        return None
    return None

async def download(url: str, max_size_bytes: int | None = None):
    """Download object at URL and cache to ~/storage.

    Parameters
    ----------
    url : str
        HTTP(S) location to download.
    max_size_bytes : int | None
        Optional hard ceiling in bytes. If provided, the function first
        issues a lightweight HEAD/Range request to obtain the object size and
        **skips the download** (raising ``ValueError``) when that size exceeds
        the limit. This prevents accidental OOM on gigantic or malicious
        payloads.
    """

    path = await _local_path_from_url(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    async with aiohttp.ClientSession() as s:
        # ------------------------------------------------------------------
        # Optional size pre-check – avoids downloading huge payloads.
        # ------------------------------------------------------------------
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
            
            # Always read the raw body first for maximum reliability.
            body = await r.read()
            try:
                # Attempt to decode as text and then parse as JSON. This is
                # the most common case for our payloads.
                data = json.loads(body.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If it's not valid JSON or not valid UTF-8, it's some other
                # binary format. Return the raw bytes.
                data = body

    # The local caching logic handles both dicts (from JSON) and bytes.
    try:
        if isinstance(data, (dict, list)):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, bytes):
            with open(path, "wb") as f:
                f.write(data)
    except Exception:
        pass # Fail silently on cache write errors.
    return data




async def exists( bucket:str, filename: str ) -> bool:
    try:
        endpoint_url = load_r2_endpoint_url()
        access_key = load_r2_write_access_key_id()
        secret_key = load_r2_write_secret_access_key()
        if not endpoint_url or not access_key or not secret_key:
            return False

        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            await s3_client.head_object(Bucket=bucket, Key=filename)
            return True
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ["404", "NoSuchKey", "NotFound"]:
            return False
        else:
            return False
    except Exception as e:
        return False
    
    
async def timestamp( bucket:str, filename: str ):
    try:
        endpoint_url = load_r2_endpoint_url()
        access_key = load_r2_write_access_key_id()
        secret_key = load_r2_write_secret_access_key()
        if not endpoint_url or not access_key or not secret_key:
            return None

        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            response = await s3_client.head_object(Bucket=bucket, Key=filename)
            last_modified = response.get("LastModified")
            if last_modified is not None:
                return last_modified
            else:
                return None
    except Exception as e:
        return None

async def list( bucket:str, prefix: str ) -> List[str]:
    endpoint_url = load_r2_endpoint_url()
    access_key = load_r2_write_access_key_id()
    secret_key = load_r2_write_secret_access_key()

    if not endpoint_url or not access_key or not secret_key:
        return []

    matching_keys = []    
    try:
        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            paginator = s3_client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj.get("Key", "")
                        matching_keys.append(key)
    except Exception as e:
        pass

    return matching_keys


async def timestamp(url: str):
    """Return Last-Modified datetime of a public HTTP object or None."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.head(url, timeout=10) as r:
                if r.status == 200:
                    lm = r.headers.get("Last-Modified")
                    if lm:
                        return email.utils.parsedate_to_datetime(lm)
    except Exception:
        pass
    return None



def _sanitize_b64(obj):
    """Recursively replace any bytes/bytearray with Base64-encoded ASCII str."""
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, dict):
        return {k: _sanitize_b64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_b64(v) for v in obj]
    return obj


def upload(bucket: str, object_key: str, file_path: str | Path) -> None:
    """
    Upload a local file to Cloudflare R2. If an object with the same key already
    exists it will be replaced.

    Parameters
    ----------
    bucket      Name of the R2 bucket.
    object_key  Key (path/filename) inside the bucket.
    file_path   Path to the local file to upload.

    Environment variables (loaded via `.env` or shell):
    ---------------------------------
    R2_ACCOUNT_ID
    R2_WRITE_ACCESS_KEY_ID
    R2_WRITE_SECRET_ACCESS_KEY
    """

    # Populate env vars from a .env file if present.
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
        config=Config(signature_version="s3v4"),  # R2 uses SigV4
    )

    logger.info("Uploading %s to bucket %s as %s", file_path, bucket, object_key)

    s3.upload_file(str(file_path), bucket, object_key)
    logger.info("✅ Uploaded %s → s3://%s/%s", file_path, bucket, object_key)
