# MANTIS Mining Guide

A quick reference for setting up your MANTIS miner. This guide details how to generate multi-asset embeddings, encrypt them securely with your hotkey, and submit them to the network.

## 1. Prerequisites

- **Python Environment:** Python 3.8 or newer.
- **Registered Hotkey:** Your hotkey must be registered on the subnet. Without this, you cannot commit your data URL.
- **Publicly Accessible URL:** You need a stable URL (e.g., from a Cloudflare R2 bucket, a personal server, or a gist) where you can host your payload file. The validator will download your submission from this URL.

## 2. Setup

Install the necessary Python packages for encryption and API requests.

```bash
pip install timelock requests
```

It is also recommended to use a tool like `boto3` and `python-dotenv` if you are using an R2 bucket for hosting.

## 3. The Mining Process: Step-by-Step

The core mining loop involves creating data, encrypting it for a future time, uploading it to your public URL, and ensuring the network knows where to find it.

### Step 1: Build Your Multi-Asset Embeddings

You must submit embeddings for all configured challenges. Each challenge has a required embedding dimension defined in the network's configuration.

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

### Step 2: Timelock-Encrypt Your Payload

To ensure security and prove ownership, you must bundle your hotkey with your embeddings before encryption. The system uses a `:::` delimiter to separate the data from the signature.

```python
import json
import time
import secrets
import requests
from timelock import Timelock

# Your Bittensor hotkey
my_hotkey = "5D..." # <-- REPLACE WITH YOUR HOTKEY

# Drand beacon configuration (do not change)
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

# Fetch beacon info to calculate a future round
info = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10).json()
future_time = time.time() + 30  # Target a round ~30 seconds in the future
target_round = int((future_time - info["genesis_time"]) // info["period"])

# Create the plaintext by joining embeddings and the hotkey
plaintext = f"{str(multi_asset_embedding)}:::{my_hotkey}"

# Encrypt the plaintext for the target round
tlock = Timelock(DRAND_PUBLIC_KEY)
salt = secrets.token_bytes(32)
ciphertext_hex = tlock.tle(target_round, plaintext, salt).hex()
```

### Step 3: Create and Save the Payload File
The payload is a JSON object containing the `round` and `ciphertext`. It's recommended to name the file after your hotkey for easy management.

```python
# The filename can be anything, but using the hotkey is good practice.
filename = my_hotkey 
payload = {
    "round": target_round,
    "ciphertext": ciphertext_hex,
}

with open(filename, "w") as f:
    json.dump(payload, f)
```

### Step 4: Upload to Your Public URL
Upload the generated payload file to your public hosting solution (e.g., R2, personal server). The file must be publicly accessible via a direct download link.

**Important**: The validator expects the filename in the commit URL to match your hotkey. For example, if your hotkey is `5D...`, a valid commit URL would be `https://myserver.com/5D...`.

### Step 5: Commit the URL to the Subnet
Finally, you must commit the public URL of your payload file to the subtensor. **You only need to do this once**, unless your URL changes. After the initial commit, you just need to update the file at that URL (Steps 1-4).

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
2.  Encrypt them with your hotkey for a future round (Step 2).
3.  Save the payload file (Step 3).
4.  Upload the new file to your public URL, overwriting the old one (Step 4).

## 5. Scoring and Rewards

The network trains a predictive model for each asset and calculates your salience (importance) across all of them. Your final reward is based on your total predictive contribution to the system.

- **Asset Filtering**: The system automatically filters out periods where asset prices haven't changed for a configured number of timesteps (e.g., during market closures), ensuring you are not penalized for stale data feeds.
- **Zero Submissions**: If you submit only zeros for an asset, your contribution for that asset will be 0. Providing valuable embeddings for all assets is the best way to maximize your rewards.

You are now ready to mine with multi-asset support!

