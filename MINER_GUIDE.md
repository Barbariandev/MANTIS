# MANTIS Mining Guide

A quick reference for setting up your MANTIS miner. This guide details how to generate multi-asset embeddings, encrypt them securely with your hotkey, and submit them to the network.

## 1. Prerequisites

- **Python Environment:** Python 3.8 or newer.
- **Registered Hotkey:** Your hotkey must be registered on the subnet. Without this, you cannot commit your data URL.
- **R2 Bucket** to contain the file containing your payloads.

## 2. Setup

Install the necessary Python packages for encryption and API requests.

```bash
pip install timelock requests
```

It is also recommended to use a tool like `boto3` and `python-dotenv` if you are using an R2 bucket for hosting.

## 3. The Mining Process: Step-by-Step

The core mining loop involves creating data, encrypting it for a future time, uploading it to your public URL, and ensuring the network knows where to find it.

For the LBFGS challenge (17-dim per bar, p/Q format), see `lbfgs_guide.md` for the exact embedding layout and scoring rules (50/50 classifier/Q within challenge).

### Step 1: Build Your Multi-Asset Embeddings

You must submit embeddings for all configured challenges. Each challenge has a required embedding dimension defined in the network's configuration.
For LBFGS challenges specifically, one embedding per bar per hotkey must be 17 floats in the order documented in `lbfgs_guide.md`.

All values in your embeddings must be between -1.0 and 1.0. The task for all assets is a binary prediction of the price change over the next 1 hour.

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

**Important**: The validator expects the filename in the commit URL to match your hotkey. For example, if your hotkey is `5D...`, a valid commit URL would be `https://myserver.com/5D...`.

### Step 5: Commit the URL to the Subnet
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
subtensor.commit(wallet=wallet, netuid=123, data=public_url) # Use the correct netuid
```

## 4. Summary Flow

**Once:**
1.  Set up your public hosting (e.g., R2 bucket, server) and get its base URL.
2.  Run the `subtensor.commit()` script (Step 5) to register your full payload URL on the network.

**Frequently (e.g., every minute):**
1.  Generate new multi-asset embeddings (Step 1).
2.  Encrypt and write the V2 payload (Step 2).
3.  Upload the new file to your public URL, overwriting the old one (Step 3).

## 5. Scoring and Rewards

The network trains a predictive model for each asset and calculates your salience (importance) across all of them. Your final reward is based on your total predictive contribution to the system.

- **Asset Filtering**: The system automatically filters out periods where asset prices haven't changed for a configured number of timesteps (e.g., during market closures), ensuring you are not penalized for stale data feeds.
- **Zero Submissions**: If you submit only zeros for an asset, your contribution for that asset will be 0. Providing valuable embeddings for all assets is the best way to maximize your rewards.

You are now ready to mine with multi-asset support!

