from __future__ import annotations
"""
MIT License

Copyright (c) 2024 MANTIS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import json
import os
import random
import secrets
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import config
from timelock import Timelock


def generate_multi_asset_embeddings() -> List[List[float]]:
    return [[random.uniform(-1, 1) for _ in range(c["dim"])] for c in config.CHALLENGES]


def _target_round(lock_seconds: int) -> int:
    info = requests.get(f"{config.DRAND_API}/beacons/{config.DRAND_BEACON_ID}/info", timeout=10).json()
    future_time = time.time() + lock_seconds
    return int((future_time - info["genesis_time"]) // info["period"])


def _hkdf_key_nonce(shared_secret: bytes, info: bytes, key_len: int = 32, nonce_len: int = 12):
    out = HKDF(algorithm=hashes.SHA256(), length=key_len + nonce_len, salt=None, info=info).derive(shared_secret)
    return out[:key_len], out[key_len:]


def _binding(hk: str, rnd: int, owner_pk: bytes, pke: bytes) -> bytes:
    h = hashes.Hash(hashes.SHA256())
    h.update(hk.encode("utf-8"))
    h.update(b":")
    h.update(str(rnd).encode("ascii"))
    h.update(b":")
    h.update(owner_pk)
    h.update(b":")
    h.update(pke)
    return h.finalize()


def _derive_pke(ske_raw: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(ske_raw).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _prepare_v2_plaintext(hotkey: str, payload_text: Optional[str], embeddings: List[List[float]]) -> Dict[str, Any]:
    if payload_text:
        text = payload_text.strip()
        if ":::" in text:
            body, hk = text.rsplit(":::", 1)
            if hk != hotkey:
                raise ValueError("legacy payload hotkey mismatch")
            data = json.loads(body.replace("'", '"'))
        else:
            data = json.loads(text)
        if isinstance(data, list):
            mapping: Dict[str, Any] = {}
            for vec, spec in zip(data, config.CHALLENGES):
                mapping[spec["ticker"]] = vec
            data = mapping
        if not isinstance(data, dict):
            raise ValueError("v2 payload must decode to an object or list")
        obj = dict(data)
    else:
        obj = {c["ticker"]: vec for vec, c in zip(embeddings, config.CHALLENGES)}
    obj["hotkey"] = hotkey
    return obj


def generate_v2(hotkey: str, lock_seconds: int, owner_pk_hex: str, payload_text: Optional[str], embeddings: List[List[float]]):
    if not owner_pk_hex:
        raise ValueError("OWNER_HPKE_PUBLIC_KEY_HEX is required for v2 payloads")
    owner_pk = bytes.fromhex(owner_pk_hex)
    obj = _prepare_v2_plaintext(hotkey, payload_text, embeddings)
    pt = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    rnd = _target_round(lock_seconds)
    ske = os.urandom(32)
    key = os.urandom(32)
    pke = _derive_pke(ske)
    binding = _binding(hotkey, rnd, owner_pk, pke)
    c_nonce = os.urandom(12)
    c_ct = ChaCha20Poly1305(key).encrypt(c_nonce, pt, binding)
    shared = X25519PrivateKey.from_private_bytes(ske).exchange(X25519PublicKey.from_public_bytes(owner_pk))
    wrap_key, _ = _hkdf_key_nonce(shared, info=b"mantis-owner-wrap")
    wrap_nonce = os.urandom(12)
    w_owner_ct = ChaCha20Poly1305(wrap_key).encrypt(wrap_nonce, key, binding)
    tlock = Timelock(config.DRAND_PUBLIC_KEY)
    combined_hex = (ske + key).hex()
    w_time_ct = tlock.tle(rnd, combined_hex, secrets.token_bytes(32))
    return {
        "v": 2,
        "round": rnd,
        "hk": hotkey,
        "owner_pk": owner_pk_hex,
        "C": {"nonce": c_nonce.hex(), "ct": c_ct.hex()},
        "W_owner": {"pke": pke.hex(), "nonce": wrap_nonce.hex(), "ct": w_owner_ct.hex()},
        "W_time": {"ct": w_time_ct.hex()},
        "binding": binding.hex(),
        "alg": config.ALG_LABEL_V2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MANTIS v2 payloads.")
    parser.add_argument("--hotkey", required=True)
    parser.add_argument("--lock-seconds", type=int, default=config.TLOCK_DEFAULT_LOCK_SECONDS)
    parser.add_argument("--owner-pk-hex", default=config.OWNER_HPKE_PUBLIC_KEY_HEX)
    parser.add_argument("--payload", help="Optional plaintext override. For v2 expects JSON.")
    parser.add_argument("--payload-file", help="Path to payload JSON/text override.")
    parser.add_argument("--out", help="Write payload to file instead of stdout.")
    args = parser.parse_args()

    payload_text = args.payload
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as fh:
            payload_text = fh.read()

    embeddings = generate_multi_asset_embeddings()

    try:
        payload = generate_v2(args.hotkey, args.lock_seconds, args.owner_pk_hex, payload_text, embeddings)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"Encrypted payload saved to: {args.out}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
