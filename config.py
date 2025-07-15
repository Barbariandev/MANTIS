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

DATALOG_ARCHIVE_URL = "https://pub-879ad825983e43529792665f4f510cd6.r2.dev/mantis_datalog.pkl.gz"

PRICE_DATA_URL = "https://pub-ba8c1b8edb8046edaccecbd26b5ca7f8.r2.dev/latest_prices.json"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(PROJECT_ROOT, ".storage")

NETUID = 123

NUM_UIDS = 256

ASSETS = ["BTC", "ETH", "EURUSD", "GBPUSD", "CADUSD", "NZDUSD", "CHFUSD", "XAUUSD", "XAGUSD"]

ASSET_EMBEDDING_DIMS = {
    "BTC": 100,
    "ETH": 2,
    "EURUSD": 2,
    "GBPUSD": 2,
    "CADUSD": 2,
    "NZDUSD": 2,
    "CHFUSD": 2,
    "XAUUSD": 2,
    "XAGUSD": 2,
}

MAX_UNCHANGED_TIMESTEPS = 15

HIDDEN_SIZE = 32
LEARNING_RATE = 1e-3

SEED = 42

SAMPLE_STEP = 5

LAG = 60

TASK_INTERVAL = 500 
