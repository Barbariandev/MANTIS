### Payload Migration Guide (V1 → V2)

This guide explains how to migrate miner payloads from V1 to V2, how to wrap your own embeddings (without using the provided script), and answers common questions.

---

### TL;DR

- **V2 keeps time-locking (Drand) and adds an owner wrap** so the validator can verify the payload was prepared for the subnet owner without revealing keys before maturity.
- **You do not need to recommit your URL** if it already points to Cloudflare R2 and the object key equals your hotkey. Just start uploading V2 JSON to the same object.
- If you submit only some assets, the rest are treated as **all-zeros**, which do not contribute to your salience for those assets.
- The validator currently accepts both **V1 and V2**; V2 is recommended.
- The validator **enforces the owner public key from config**. The `owner_pk` field in the payload is optional; if present it must match the configured key or it will be rejected.

---

### What changes from V1 to V2?

- **V1** payload is a JSON object with `{"round", "ciphertext"}`. The plaintext is the string representation of a Python list of embeddings followed by `:::hotkey`.
- **V2** payload is a JSON object containing:
  - `v: 2`, `round: <int>`, `hk: <hotkey>`, `owner_pk: <hex>` (optional), `alg: "x25519-hkdf-sha256+chacha20poly1305+drand-tlock"` (informational)
  - `C` (AEAD-encrypted plaintext embeddings JSON)
  - `W_owner` (owner-wrap of the symmetric key using X25519+HKDF+ChaCha20-Poly1305)
  - `W_owner.nonce` is a random 12-byte nonce included in the payload; the validator uses the provided nonce (it does not derive it)
  - `W_time` (Drand time-lock of the sender key + symmetric key)
  - `binding` (SHA-256 over `(hk, round, owner_pk, pke)` used as AEAD associated data)
  - Note: The validator uses the **configured** owner public key to verify/decrypt; any mismatch with the `owner_pk` field in the payload is rejected.

The V2 plaintext is a JSON object that must include your `"hotkey"` and embeddings for each asset. Keys can be tickers (recommended) or challenge names.

---

### V2 payload schema

```json
{
  "v": 2,
  "round": 1234567,
  "hk": "<SS58 hotkey>",
  "owner_pk": "<hex-encoded owner HPKE public key>",
  "C": { "nonce": "<hex>", "ct": "<hex>" },
  "W_owner": { "pke": "<hex>", "nonce": "<hex>", "ct": "<hex>" },
  "W_time": { "ct": "<hex>" },
  "binding": "<hex>",
  "alg": "x25519-hkdf-sha256+chacha20poly1305+drand-tlock"
}
```



Plaintext inside `C.ct` (UTF-8 JSON) must look like:

```json
{
  "BTC": [/* 100 floats in [-1,1] */],
  "ETH": [/* 2 floats */],
  "EURUSD": [/* 2 floats */],
  "GBPUSD": [/* 2 floats */],
  "CADUSD": [/* 2 floats */],
  "NZDUSD": [/* 2 floats */],
  "CHFUSD": [/* 2 floats */],
  "XAUUSD": [/* 2 floats */],
  "XAGUSD": [/* 2 floats */],
  "hotkey": "<SS58 hotkey>"
}
```

Asset dimensions (current):
- **BTC**: 100
- **ETH, EURUSD, GBPUSD, CADUSD, NZDUSD, CHFUSD, XAUUSD, XAGUSD**: 2

Values must be numeric and within **[-1.0, 1.0]**.

---

### How to build a V2 payload (Python, minimal example)

Below is a direct recipe mirroring what the validator expects. You can plug in your own embedding generator; just ensure dims/ranges are correct.

```python
import json, os, time, secrets, requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from timelock import Timelock

# Drand
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

# Subnet owner HPKE public key (hex) — also published in config
OWNER_HPKE_PUBLIC_KEY_HEX = "fbfe185ded7a4e6865effceb23cbac32894170587674e751ac237a06f72b3067"

ALG_LABEL_V2 = "x25519-hkdf-sha256+chacha20poly1305+drand-tlock"

def drand_target_round(lock_seconds: int) -> int:
    info = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10).json()
    future_time = time.time() + int(lock_seconds)
    return int((future_time - info["genesis_time"]) // info["period"])

def hkdf_key(shared_secret: bytes, info: bytes = b"mantis-owner-wrap"):
    # Derive only the wrap key. The wrap nonce is random per payload and included in `W_owner.nonce`.
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared_secret)

def binding(hk: str, rnd: int, owner_pk: bytes, pke: bytes) -> bytes:
    h = hashes.Hash(hashes.SHA256())
    h.update(hk.encode("utf-8")); h.update(b":"); h.update(str(rnd).encode("ascii")); h.update(b":")
    h.update(owner_pk); h.update(b":"); h.update(pke)
    return h.finalize()

def derive_pke(ske_raw: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(ske_raw).public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )

def build_v2_payload(hotkey: str, embeddings_by_ticker: dict[str, list[float]], lock_seconds: int = 30) -> dict:
    # 1) Plaintext JSON (must include your hotkey)
    obj = dict(embeddings_by_ticker)
    obj["hotkey"] = hotkey
    pt = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # 2) Drand round
    rnd = drand_target_round(lock_seconds)

    # 3) Generate sender key (ske), ephemeral public key (pke), and payload symmetric key
    ske = os.urandom(32)
    key = os.urandom(32)
    pke = derive_pke(ske)

    # 4) Build binding and encrypt payload C
    owner_pk = bytes.fromhex(OWNER_HPKE_PUBLIC_KEY_HEX)
    ad = binding(hotkey, rnd, owner_pk, pke)
    c_nonce = os.urandom(12)
    c_ct = ChaCha20Poly1305(key).encrypt(c_nonce, pt, ad)

    # 5) Owner-wrap the symmetric key using ECDH(ske, owner_pk) → HKDF → AEAD
    shared = X25519PrivateKey.from_private_bytes(ske).exchange(X25519PublicKey.from_public_bytes(owner_pk))
    wrap_key = hkdf_key(shared)
    wrap_nonce = os.urandom(12)
    w_owner_ct = ChaCha20Poly1305(wrap_key).encrypt(wrap_nonce, key, ad)

    # 6) Time-lock the concatenated ske||key for round rnd
    tle = Timelock(DRAND_PUBLIC_KEY)
    combined_hex = (ske + key).hex()
    w_time_ct = tle.tle(rnd, combined_hex, secrets.token_bytes(32))

    # 7) Assemble V2 JSON
    return {
        "v": 2,
        "round": rnd,
        "hk": hotkey,
        "owner_pk": OWNER_HPKE_PUBLIC_KEY_HEX,
        "C": {"nonce": c_nonce.hex(), "ct": c_ct.hex()},
        "W_owner": {"pke": pke.hex(), "nonce": wrap_nonce.hex(), "ct": w_owner_ct.hex()},
        "W_time": {"ct": w_time_ct.hex()},
        "binding": ad.hex(),
        "alg": ALG_LABEL_V2,
    }
```

Notes:
- The validator verifies `binding`, that `pke` matches `ske`, that the owner-wrap decrypts to the same symmetric key, and then decrypts `C`.
- The validator uses the **configured owner public key** at verification time; if the payload includes `owner_pk`, it must match that key (otherwise omit the field).
- `W_owner.nonce` is random per payload and included in the JSON; the validator uses the provided nonce and does not derive it from ECDH/HKDF.
- The `alg` field is informational and may be ignored by the validator.
- The V2 plaintext must include `"hotkey"` equal to the committing miner hotkey, or it will be rejected.
- You may provide a list (in exact challenge order) instead of a dict, but a dict keyed by tickers is safer.

---

### Hosting and commits (URLs)

- The validator currently accepts only **Cloudflare R2** URLs (hosts ending with `.r2.dev` or `.r2.cloudflarestorage.com`).
- The object key (final path component) must be **exactly your hotkey**. Do not include directories or extra path segments.
- Maximum payload size enforced by the validator: **25 MB**.
- You usually **commit once**. Keep using the same URL and overwrite the object content as you produce new payloads.

Example commit (one time):

```python
import bittensor as bt

wallet = bt.wallet(name="your_wallet_name", hotkey="your_hotkey_name")
subtensor = bt.subtensor(network="mainnet")  # or "finney" if you are on testnet

public_url = f"https://<bucket-id>.r2.dev/{wallet.hotkey.ss58_address}"
subtensor.commit(wallet=wallet, netuid=123, data=public_url)
```

If you change where you host (e.g., you move to a different bucket/domain), update the on-chain commitment once to the new R2 URL.

---

### FAQ

- **Must I recommit the URL?**
  - **No** if your current commitment already points to Cloudflare R2 and the object key equals your hotkey. Just start uploading V2 JSON to the same object. **Yes** if you need to switch hosts (e.g., from a non-R2 host) or change the path.

- **What happens if I only submit embeddings for 1 or 2 assets?**
  - Missing assets are treated as **all-zeros** for that step. The validator only stores non-zero embeddings. In training, rows with empty/all-zero embeddings for a challenge are dropped. Your salience is then computed on whichever challenges you provide non-zero data for. Submitting all assets gives you exposure to all challenge weights.

- **Do I need to include my hotkey in the plaintext?**
  - **Yes.** In V2 the decrypted JSON must contain `"hotkey"` equal to your on-chain hotkey. In V1 it was appended to the plaintext string; in V2 it’s a JSON field.

- **Is V1 still accepted?**
  - Yes. The validator detects `{"round","ciphertext"}` as V1 and will decrypt it once the round’s signature is available. V2 is recommended and will become the standard going forward.

- **Which owner public key is used for V2?**
  - The **subnet-configured owner key** (published in config). The validator ignores arbitrary values and requires any `owner_pk` present in the payload to match this key.

- **How far in the future should I lock?**
  - For development: ~3530 seconds
  
- **What if my values fall outside [-1, 1] or dimensions are wrong?**
  - The validator will treat invalid vectors as zeros for that asset and step. Ensure correct dims and clamp your outputs to [-1, 1].

- **Does filename/extension matter?**
  - The object name must be **exactly your hotkey** (no extension required). Any mismatch is rejected.

- **Can I use directories or nested paths?**
  - No. The commit URL must have exactly one path component (the hotkey).

---

### Migration steps

1) Keep your existing R2 object URL (or commit a new one if needed) whose key equals your hotkey.
2) Switch your generator to produce the V2 JSON structure above, with your embeddings encoded as JSON and `"hotkey"` included.
3) Re-encrypt every time with a new Drand round in the future, then overwrite the same R2 object.
4) Verify your JSON is under 25 MB and values are in [-1, 1] with correct dimensions.

That’s it—you’re on V2.

