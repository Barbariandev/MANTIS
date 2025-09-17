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


import os

DATALOG_ARCHIVE_URL = "https://pub-879ad825983e43529792665f4f510cd6.r2.dev/mantis_datalog.pkl"

PRICE_DATA_URL = "https://pub-ba8c1b8edb8046edaccecbd26b5ca7f8.r2.dev/latest_prices.json"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(PROJECT_ROOT, ".storage")

NETUID = 123

NUM_UIDS = 256

CHALLENGES = [
    {
        "name": "BTC-1H-BINARY",
        "ticker": "BTC",
        "dim": 100,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
    {
        "name": "ETH-1H-BINARY",
        "ticker": "ETH",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
    {
        "name": "EURUSD-1H-BINARY",
        "ticker": "EURUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
    {
        "name": "GBPUSD-1H-BINARY",
        "ticker": "GBPUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
    {
        "name": "CADUSD-1H-BINARY",
        "ticker": "CADUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
    {
        "name": "NZDUSD-1H-BINARY",
        "ticker": "NZDUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
    {
        "name": "CHFUSD-1H-BINARY",
        "ticker": "CHFUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
    {
        "name": "XAUUSD-1H-BINARY",
        "ticker": "XAUUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
    {
        "name": "XAGUSD-1H-BINARY",
        "ticker": "XAGUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
]
CHALLENGE_MAP = {c["ticker"]: c for c in CHALLENGES}

ASSET_EMBEDDING_DIMS = {c["ticker"]: c["dim"] for c in CHALLENGES}

MAX_UNCHANGED_TIMESTEPS = 15

HIDDEN_SIZE = 32
LEARNING_RATE = 1e-3

SEED = 42

SAMPLE_EVERY = 5

LAG = 60

TASK_INTERVAL = 500 








