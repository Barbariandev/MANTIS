# MANTIS Mining Guide

A quick reference for setting up your MANTIS miner. This guide details how to generate embeddings, encrypt them using `tlock`, and submit them to the network.

## 1. Prerequisites

- **Python Environment:** Python 3.8 or newer.
- **Registered Hotkey:** Your hotkey must be registered on the netuid 123. Without this, you cannot commit your data URL.
- **Cloudflare R2 Bucket:**
    - You need an R2 bucket for hosting your payload.
    - The bucket must be configured for public access.
    - **Crucially, the filename name must exactly match your miner's hotkey.**

## 2. Setup

### A. Install Required Libraries
Install the necessary Python packages for encryption, API requests, and uploading to R2.
```bash
pip install timelock requests boto3 python-dotenv
```


```env
# .env file
R2_ACCOUNT_ID="your_r2_account_id"
R2_WRITE_ACCESS_KEY_ID="your_r2_access_key_id"
R2_WRITE_SECRET_ACCESS_KEY="your_r2_secret_access_key"
```

## 3. The Mining Process: Step-by-Step

The core mining loop involves creating data, encrypting it for a future time, uploading it, and ensuring the network knows where to find it.

### Step 1: Build Your Embedding
First, generate your predictive embedding. This must be a list of 100 floating-point numbers, where each value is between -1.0 and 1.0. 
A recommended first step would be to build a model that attempts to predict BTCUSD 1 hour returns.

```python
import numpy as np
# Replace this with the real output from your predictive model
embedding = np.random.uniform(-1, 1, size=100).tolist()
```

### Step 2: Timelock-Encrypt the Embedding
We use `tlock`, a decentralized timelock system powered by the Drand network, to encrypt the embedding. This process involves targeting a future Drand "round" which acts as the key to unlock the data.

```python
import json
import time
import secrets
import requests
from timelock import Timelock

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

# Encrypt the embedding for the target round
tlock = Timelock(DRAND_PUBLIC_KEY)
ciphertext_hex = tlock.tle(target_round, str(embedding), secrets.token_bytes(32)).hex()
```

### Step 3: Create and Save the Payload File
The payload is a JSON object containing the `round` and `ciphertext`. The filename **must** be your hotkey.

```python
hotkey = "5DhoYw2EyGGqcXt3Cgnpcaf2VRCJcJYJntpwyQryphTgmYWs" # <-- REPLACE WITH YOUR HOTKEY
payload = {
    "round": target_round,
    "ciphertext": ciphertext_hex,
}

with open(hotkey, "w") as f:
    json.dump(payload, f)
```

### Step 4: Upload to Your R2 Bucket
Upload the generated file to your R2 bucket. Remember, the **filename must be your hotkey**. The object key (the name of the file inside the bucket) must also be your hotkey.

```python
import os
import boto3
from dotenv import load_dotenv

load_dotenv() # Load credentials from .env file

def upload_to_r2(bucket_name, object_key, file_path):
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ['R2_WRITE_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_WRITE_SECRET_ACCESS_KEY'],
        region_name="auto",
    )
    s3.upload_file(file_path, bucket_name, object_key)
    print(f"Successfully uploaded {file_path} to R2 bucket {bucket_name}.")

# Usage:
upload_to_r2(bucket_name=hotkey, object_key=hotkey, file_path=hotkey)
```

### Step 5: Commit the URL to the Subnet
Finally, you must commit the public URL of your R2 object to the subtensor. **You only need to do this once**, unless your URL changes. After the initial commit, you only need to update the file in your R2 bucket (Steps 1-4).

```python
import bittensor as bt

# Configure your wallet and the subtensor
wallet = bt.wallet(name="your_wallet_name", hotkey="your_hotkey_name")
subtensor = bt.subtensor(network="finney")

# The public URL of your object in R2
# NOTE: The public URL format may vary slightly based on your R2 setup.
# Ensure your bucket is public and the URL is correct.
r2_public_url = f"https://pub-your_public_hash.r2.dev/{hotkey}" 

# Commit the URL on-chain
subtensor.commit(wallet=wallet, netuid=123, data=r2_public_url) # Use the correct netuid
```

## 4. Summary Flow

**Once:**
1.  Set up your R2 bucket and make it public.
2.  Run the `subtensor.commit()` script (Step 5) to register your URL on the network.

**Frequently (e.g., every minute):**
1.  Generate a new embedding (Step 1).
2.  Encrypt it for a future round (Step 2).
3.  Save the payload file (Step 3).
4.  Upload the new file to R2, overwriting the old one (Step 4).

You are now mining!
