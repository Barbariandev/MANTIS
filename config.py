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


import os

# ---------------------------------------------------------------------------
# Cross-hardware reproducibility env pinning.
#
# Every numerical module in MANTIS imports `config`, so this block runs before
# numpy / sklearn / torch are loaded by any consumer.  BLAS libraries cache
# their thread-count decision at load time, so these env vars MUST be set
# before the first numpy import — placing them here is the only place that
# guarantees that ordering across every entry point (validator,
# offline scoring scripts, etc.).
#
# Single-threaded BLAS makes parallel reductions associative-stable, which is
# the dominant requirement for two validators on different hardware to agree
# on weight vectors and therefore drive vtrust toward 1.0.
# ---------------------------------------------------------------------------
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("BLIS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("MKL_DYNAMIC", "FALSE"),
    ("OMP_DYNAMIC", "FALSE"),
    ("PYTHONHASHSEED", "0"),
    ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
):
    os.environ.setdefault(_k, _v)
# CUDA is hidden by default — the scoring path is pure numpy/sklearn, so
# disabling GPU at the env layer guarantees that GPU and CPU validators take
# exactly the same numerical path.  Set MANTIS_ALLOW_CUDA=1 to opt in.
if os.environ.get("MANTIS_ALLOW_CUDA", "").lower() not in ("1", "true", "yes"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

DATALOG_ARCHIVE_URL = "https://pub-879ad825983e43529792665f4f510cd6.r2.dev/datalog.db"

# FLOW lives in its own SQLite file, published to the same bucket.
# The legacy datalog has grown too large to keep re-shipping for one
# challenge; the split keeps the FLOW archive small (its own rows
# only) while the old tickers stay where they are.  The validator
# attaches it alongside the main DB (see ledger.py); a missing remote
# archive is non-fatal (fresh challenge, empty local file).
FLOW_DATALOG_ARCHIVE_URL = (
    "https://pub-879ad825983e43529792665f4f510cd6.r2.dev/flow_datalog.db"
)

PRICE_DATA_URL = "https://pub-ba8c1b8edb8046edaccecbd26b5ca7f8.r2.dev/latest_prices.json"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(PROJECT_ROOT, ".storage")

NETUID = 123

NUM_UIDS = 256

CHALLENGES = [
    {
        "name": "ETH-1H-BINARY",
        "ticker": "ETH",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
    {
        "name": "ETH-HITFIRST-100M",
        "ticker": "ETHHITFIRST",
        "price_key": "ETH",
        "dim": 3,
        "blocks_ahead": 500,
        "loss_func": "hitfirst",
        "weight": 1.25,
    },
    {
        "name": "ETH-LBFGS",
        "ticker": "ETHLBFGS",
        "price_key": "ETH",
        "dim": 17,
        "blocks_ahead": 300,
        "loss_func": "lbfgs",
        "weight": 3.5,
    },
    {
        "name": "BTC-LBFGS-6H",
        "ticker": "BTCLBFGS",
        "price_key": "BTC",
        "dim": 17,
        "blocks_ahead": 1800,
        "loss_func": "lbfgs",
        "weight": 2.875,
    },
    {
        "name": "CADUSD-1H-BINARY",
        "ticker": "CADUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 0.5,
    },
    {
        "name": "NZDUSD-1H-BINARY",
        "ticker": "NZDUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 0.5,
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
        "name": "XAGUSD-1H-BINARY",
        "ticker": "XAGUSD",
        "dim": 2,
        "blocks_ahead": 300,
        "loss_func": "binary",
        "weight": 1,
    },
]

BREAKOUT_ASSETS = [
    "BTC", "ETH", "XRP", "SOL", "TRX", "DOGE", "ADA", "BCH", "XMR",
    "LINK", "LEO", "HYPE", "XLM", "ZEC", "SUI", "LTC", "AVAX", "HBAR", "SHIB",
    "TON", "CRO", "DOT", "UNI", "MNT", "BGB", "TAO", "AAVE", "PEPE",
    "NEAR", "ICP", "ETC", "ONDO", "SKY",
]

FUNDING_ASSETS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "SUI",
    "NEAR", "AAVE", "UNI", "LTC", "HBAR", "PEPE", "TRX", "SHIB", "TAO", "ONDO",
]

MULTI_BREAKOUT_CHALLENGE = {
    "name": "MULTI-BREAKOUT",
    "ticker": "MULTIBREAKOUT",
    "assets": BREAKOUT_ASSETS,
    "dim": 2,
    "loss_func": "range_breakout_multi",
    "range_lookback_blocks": 28800,
    "barrier_pct": 25.0,
    "min_range_pct": 1.0,
    "weight": 5.0,
    "gate_top_pct": 0.10,
}

CHALLENGES.append(MULTI_BREAKOUT_CHALLENGE)

XSEC_RANK_CHALLENGE = {
    "name": "XSEC-RANK",
    "ticker": "MULTIXSEC",
    "assets": BREAKOUT_ASSETS,
    "dim": 1,
    "blocks_ahead": 1200,
    "loss_func": "xsec_rank",
    "weight": 3.0,
}

CHALLENGES.append(XSEC_RANK_CHALLENGE)

FUNDING_XSEC_CHALLENGE = {
    "name": "FUNDING-XSEC",
    "ticker": "FUNDINGXSEC",
    "assets": FUNDING_ASSETS,
    "dim": 1,
    "blocks_ahead": 2400,  
    "loss_func": "funding_xsec",
    "weight": 4.0,
}

CHALLENGES.append(FUNDING_XSEC_CHALLENGE)

# TRADE-MIX is DEPRECATED as of the FLOW launch (2026-08-17): removed
# from the active roster, superseded by FLOW, and its historical
# challenge_data is purged on datalog open (see ledger.py).  The assets
# constant stays because the datalog price rows still carry them.
TRADE_MIX_ASSETS = ["BTC", "ETH", "TAO", "SOL"]

# FLOW: capital-at-risk BTC trades with the evidence-gated payment layer.
# Spec + simulation evidence: the FLOW release paper
# (FLOW_RELEASE.pdf, on the MANTIS site).  Parameters below are the
# launch table (section 1.8); scoring implementation is flow.py; the
# collateral-pool client is flow_collateral.py (bet-basis score
# weighting reads the live pool below; set FLOW_COLLATERAL_ADDRESS=off
# to force f-only).  There is no entry bond: posted bets plus the
# significance gate carry deterrence (release section 6).
#
# Wick feed: when latest_prices.json carries "BTC_HIGH"/"BTC_LOW"
# (per-minute consensus wicks: min of venue highs, max of venue lows),
# the validator stores them per row and resolution runs on the wick
# channels (release sections 1.3, 2.4).  Rows without them resolve
# close-only, a priced degradation (section 6.14), so the publisher
# must ship the keys from challenge start.
#
# Timeline (announced 2026-07-31):
#   2026-08-17  challenge live: validators collect and score payloads,
#               gate evidence accrues, emission weight 0 (nothing paid).
#   2026-09-14  after four weeks of payload collection, emissions turn on
#               at FLOW_TARGET_FRACTION of the pool.  That flip is a
#               deliberate one-line PR: set "weight" below to
#               _FLOW_WEIGHT_AT_TURNON.  It is NOT automatic.
FLOW_TARGET_FRACTION = 0.25
_FLOW_WEIGHT_AT_TURNON = (
    FLOW_TARGET_FRACTION * sum(c["weight"] for c in CHALLENGES) /
    (1.0 - FLOW_TARGET_FRACTION)
)

FLOW_CHALLENGE = {
    "name": "FLOW-BTC",
    "ticker": "FLOW",
    "assets": ["BTC"],
    "dim": 28,                    # 4 regimes x [d, f, sl, tp1, tp2, h, trade_id]
    "blocks_ahead": 0,            # resolution is horizon-driven, not fixed-lag
    "loss_func": "flow",
    "weight": float(os.environ.get("MANTIS_FLOW_WEIGHT", "0.0")),
    # --- launch parameters (release paper section 1.8)
    "delta": 0.30,
    "lam": 0.50,
    "gamma": 0.25,
    "tau": 1.00,
    "ewma_alpha": 0.05,
    "kappa_hours": 672.0,
    "probation_hours": 168.0,
    "gate_block_h": 168.0,
    "gate_min_n": 4,
    "gate_z_in": 1.25,
    "gate_z_out": 0.50,
    "dust": 0.02,
}

CHALLENGES.append(FLOW_CHALLENGE)

# Live FlowCollateralPool on Bittensor EVM mainnet (chain id 964).
# Deployed 2026-08-16 in block 8859422; week 1 opens at periodZero.
# Env overrides; FLOW_COLLATERAL_ADDRESS=off disables collateral reads.
FLOW_COLLATERAL_ADDRESS = os.environ.get(
    "FLOW_COLLATERAL_ADDRESS",
    "0xD9c805202b16671A2901307fBC9A8750E2453427",
).strip()
FLOW_COLLATERAL_RPC = os.environ.get(
    "FLOW_COLLATERAL_RPC",
    "https://lite.chain.opentensor.ai",
)
FLOW_COLLATERAL_PERIOD_ZERO = 1787004000  # 2026-08-17 22:00:00 UTC
FLOW_COLLATERAL_NETUID = 123
FLOW_COLLATERAL_OWNER = "0x5186318Ba00Ca115d92C37D2b646eA3867C2c754"
FLOW_COLLATERAL_MIRROR_COLDKEY = (
    "0x379d4712e9902a9ca2ba1b827f4cb2c48ec03b510b107bf700c92592d9df067a"
)
FLOW_COLLATERAL_DEPLOY_BLOCK = 8859422

CHALLENGE_MAP = {c["ticker"]: c for c in CHALLENGES}
CHALLENGE_NAME_TO_TICKER = {c["name"]: c["ticker"] for c in CHALLENGES}
ASSET_EMBEDDING_DIMS = {c["ticker"]: c["dim"] for c in CHALLENGES}

# Lowered from 0.45 alongside the TRADE-MIX v2 launch: the v2 mechanism
# burns the unearned share of its 10% pool, so the effective total burn
# stays roughly where it was.
BURN_PCT = 0.35

MAX_DAYS = 60

MAX_UNCHANGED_TIMESTEPS = 15

HIDDEN_SIZE = 32
LEARNING_RATE = 1e-3

SEED = 42

SAMPLE_EVERY = 5

LAG = 60

TASK_INTERVAL = 500

WEIGHT_CALC_INTERVAL = 1000
WEIGHT_SET_INTERVAL = 360

OWNER_HPKE_PUBLIC_KEY_HEX="fbfe185ded7a4e6865effceb23cbac32894170587674e751ac237a06f72b3067"
TLOCK_DEFAULT_LOCK_SECONDS = int(os.getenv("TLOCK_DEFAULT_LOCK_SECONDS", "30"))
TLOCK_PROD_SUGGESTED_LOCK_SECONDS = int(os.getenv("TLOCK_PROD_SUGGESTED_LOCK_SECONDS", "604800"))

# Blocks a raw payload must age before the validator attempts decryption.
# One week (matching the suggested tlock horizon above): a payload becomes
# publicly readable one week after its entry, which covers regimes A-C
# entirely; a regime-D trade at the top of its band (up to 336h) can still
# be live when its opening payload decrypts.  The owner-wrapped key path
# (W_owner) is unaffected — copy-trade execution reads payloads
# immediately.  Prices are recorded at submission time and embeddings
# backfill onto those rows after decryption, so entry pricing is
# unaffected by the delay.
PAYLOAD_MATURITY_BLOCKS = int(os.getenv("PAYLOAD_MATURITY_BLOCKS", "50400"))
ALG_LABEL_V2 = "x25519-hkdf-sha256+chacha20poly1305+drand-tlock"
SUPPORTED_PAYLOAD_VERSIONS = {1, 2}

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)



