# MANTIS Mining Guide

A quick reference for setting up your MANTIS miner. This guide details how to generate multi-asset embeddings, encrypt them securely with your hotkey, and submit them to the network.

## 1. Prerequisites

- **Python Environment:** Python 3.10 or newer.
- **Registered Hotkey:** Your hotkey must be registered on the subnet. Without this, you cannot commit your data URL.
- **Cloudflare R2 bucket**: the validator only accepts commit URLs hosted on R2 (`*.r2.dev` or `*.r2.cloudflarestorage.com`), and the object key must be exactly your hotkey.

## 2. Setup

Install the necessary Python packages for encryption and API requests (and optional upload helpers if you want).

```bash
pip install timelock requests cryptography
```

It is also recommended to use a tool like `boto3` and `python-dotenv` if you want to programmatically upload to R2.

## 3. The Mining Process: Step-by-Step

The core mining loop involves creating data, encrypting it for a future time, uploading it to your public URL, and ensuring the network knows where to find it.

Your required challenge list is defined in `config.py` (`CHALLENGES`). At the time of writing, it includes:

- **Binary** (dim=2): `ETH`, `CADUSD`, `NZDUSD`, `CHFUSD`, `XAGUSD`
- **HITFIRST** (dim=3): `ETHHITFIRST`
- **LBFGS** (dim=17): `ETHLBFGS`, `BTCLBFGS`

### Step 1: Build Your Multi-Asset Embeddings

You must submit embeddings for all configured challenges. Each challenge has a required embedding dimension defined in the network's configuration.

**Important (ranges / semantics):**

- **Binary challenges (dim=2, `loss_func="binary"`)**
  - Two features per challenge.
  - Must be in **[-1, 1]** (validator drops invalid values to zeros).
  - These are treated as features for a classifier, not necessarily probabilities.
- **HITFIRST (`ETHHITFIRST`, dim=3, `loss_func="hitfirst"`)**
  - Submit a 3-way probability vector in **(0, 1)** that sums to 1.
  - Interpretable as: `[P(up first), P(down first), P(neither)]`.
- **LBFGS (`ETHLBFGS`, `BTCLBFGS`, dim=17, `loss_func="lbfgs"`)**
  - `p[0:5]`: 5-bucket probabilities (in (0,1), sum to 1)
  - `q[5:17]`: 12 probabilities in (0,1) used for Q-path scoring
  - Index map:
    - `[0:5]`  => `p[0..4]`
    - `[5:8]`  => Q bucket 0, thresholds [0.5σ, 1.0σ, 2.0σ]
    - `[8:11]` => Q bucket 1, thresholds [0.5σ, 1.0σ, 2.0σ]
    - `[11:14]` => Q bucket 3, thresholds [0.5σ, 1.0σ, 2.0σ]
    - `[14:17]` => Q bucket 4, thresholds [0.5σ, 1.0σ, 2.0σ]

```python
import numpy as np
from config import CHALLENGES

# Generate embeddings for each challenge (replace with your model outputs)
multi_asset_embedding = [
    np.random.uniform(-1, 1, size=c["dim"]).tolist()
    for c in CHALLENGES
]
```

### Step 2: Timelock-Encrypt Your Payload (V2 Only)

The validator accepts only V2 JSON payloads. You can call the helper script or embed the logic directly in your miner.

**CLI helper (recommended)**

```bash
python generate_and_encrypt.py --hotkey "$MY_HOTKEY" --lock-seconds 30 --out "$MY_HOTKEY"
```

The script uses the owner public key and Drand parameters from `config.py`, targets a round roughly 30 seconds ahead, and writes a JSON payload whose filename matches your hotkey.

**Inline Python example**

```python
import json
from generate_and_encrypt import generate_v2, generate_multi_asset_embeddings
from config import OWNER_HPKE_PUBLIC_KEY_HEX

embeddings = generate_multi_asset_embeddings()
payload = generate_v2(
    hotkey=my_hotkey,
    lock_seconds=30,
    owner_pk_hex=OWNER_HPKE_PUBLIC_KEY_HEX,
    payload_text=None,
    embeddings=embeddings,
)

with open(my_hotkey, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
```

The resulting JSON contains fields such as `v`, `round`, `hk`, `owner_pk`, `C`, `W_owner`, `W_time`, `binding`, and `alg`. Do not modify or strip these fields; the validator verifies them when decrypting the payload.

### Step 3: Upload to Your Public URL
Upload the generated payload file to your public hosting solution (e.g., R2, personal server). The file must be publicly accessible via a direct download link.

**Important**:

- The validator expects the **filename in the commit URL** to match your hotkey (case-insensitive).
- The commit URL must be **Cloudflare R2** (`*.r2.dev` or `*.r2.cloudflarestorage.com`).
- The URL path must be **exactly one segment**: `/<hotkey>` (no directories).

### Step 4: Commit the URL to the Subnet
Finally, you must commit the public URL of your payload file to the subtensor. **You only need to do this once**, unless your URL changes. After the initial commit, you just need to update the file at that URL (Steps 1-3).

```python
import bittensor as bt

# Configure your wallet and the subtensor
wallet = bt.wallet(name="your_wallet_name", hotkey="your_hotkey_name")
subtensor = bt.subtensor(network="finney")

# The public URL where the validator can download your payload file.
# The final path component MUST match your hotkey.
public_url = f"https://your-public-url.com/{my_hotkey}" 

# Commit the URL on-chain
subtensor.commit(wallet=wallet, netuid=123, data=public_url)  # Use the correct netuid
```

## 4. Summary Flow

**Once:**
1.  Set up your public hosting (e.g., R2 bucket, server) and get its base URL.
2.  Run the `subtensor.commit()` script (Step 4) to register your full payload URL on the network.

**Frequently (e.g., every minute):**
1.  Generate new multi-asset embeddings (Step 1).
2.  Encrypt and write the V2 payload (Step 2).
3.  Upload the new file to your public URL, overwriting the old one (Step 3).

## 5. Scoring and Rewards

The network trains a predictive model for each asset and calculates your salience (importance) across all of them. Your final reward is based on your total predictive contribution to the system.

- **Asset Filtering**: The system automatically filters out periods where asset prices haven't changed for a configured number of timesteps (e.g., during market closures), ensuring you are not penalized for stale data feeds.
- **Zero Submissions**: If you submit only zeros for an asset, your contribution for that asset will be 0. Providing valuable embeddings for all assets is the best way to maximize your rewards.

You are now ready to mine with multi-asset support!
