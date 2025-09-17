import json
import random
import secrets
import time
from typing import List

import requests

from timelock import Timelock

from config import CHALLENGES

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)
LOCK_TIME_SECONDS = 30

def generate_multi_asset_embeddings() -> List[List[float]]:
    return [[random.uniform(-1, 1) for _ in range(c["dim"])] for c in CHALLENGES]

def generate_and_encrypt(hotkey: str, filename: str | None = None):
    if filename is None:
        filename = hotkey

    embeddings = generate_multi_asset_embeddings()
    
    try:
        info = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10).json()
        future_time = time.time() + LOCK_TIME_SECONDS
        round_num = int((future_time - info["genesis_time"]) // info["period"])
    except Exception as e:
        print(f"Error fetching Drand info: {e}")
        return None

    try:
        tlock = Timelock(DRAND_PUBLIC_KEY)
        plaintext = f"{str(embeddings)}:::{hotkey}"
        salt = secrets.token_bytes(32)
        
        ciphertext_hex = tlock.tle(round_num, plaintext, salt).hex()
    except Exception as e:
        print(f"Error during encryption: {e}")
        return None

    payload_dict = {"round": round_num, "ciphertext": ciphertext_hex}
    
    if filename:
        try:
            with open(filename, "w") as f:
                json.dump(payload_dict, f, indent=2)
            print(f"Encrypted payload saved to: {filename}")
        except Exception as e:
            print(f"Error saving to file {filename}: {e}")

    return payload_dict

if __name__ == "__main__":
    example_hotkey = "5..."
    generate_and_encrypt(hotkey=example_hotkey, filename=f"{example_hotkey}") 

