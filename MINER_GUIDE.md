# Miner Guide

## 1. Setup

```bash
pip install requests cryptography boto3 python-dotenv maturin
# Do not `pip install timelock` from PyPI — that is the old 372-byte
# stack and will not match live subnet payloads (356-byte W_time).
git clone https://github.com/ideal-lab5/timelock.git
cd timelock && git checkout ccccca019409c89f31fd687352db8060bfb4aae6
pip install ./py
cd wasm && maturin build --features python --release
pip install ../target/wheels/timelock_wasm_wrapper-0.3.0-*.whl
```

Requirements:
- Python 3.10+ (Rust + maturin to build the timelock wasm wrapper)
- Registered hotkey on subnet 123
- Cloudflare R2 bucket (commit URLs must be `*.r2.dev` or `*.r2.cloudflarestorage.com`, object key = your hotkey)

---

## 2. Submission Loop

**Once:** commit your R2 URL on-chain via `subtensor.commit()`.

**Every ~60s:** generate embeddings for all challenges, encrypt as V2 payload, upload to R2 (overwriting previous file).

```python
import json
from generate_and_encrypt import generate_v2
from config import CHALLENGES, OWNER_HPKE_PUBLIC_KEY_HEX

embeddings = build_all_embeddings()  # see below

payload = generate_v2(
    hotkey=my_hotkey,
    lock_seconds=30,
    owner_pk_hex=OWNER_HPKE_PUBLIC_KEY_HEX,
    embeddings=embeddings,
)

with open(my_hotkey, "w") as f:
    json.dump(payload, f)
# upload to R2
```

---

## 3. Challenge Specifications

### 3.1 Binary (dim=2)

Tickers: `ETH`, `CADUSD`, `NZDUSD`, `CHFUSD`, `XAGUSD`

Horizon: 300 blocks (1h). Two features in $[-1, 1]$. These are inputs to a logistic regression classifier, not probabilities. Scoring: feature selection (per-miner L2 logistic, AUC on held-out half, top-50 selected), then ElasticNet (L1 ratio 0.5) meta-model on walk-forward OOS base-model predictions. Importance = $|\beta_j|$.

```python
embeddings["ETH"] = [0.3, -0.1]       # your model output
embeddings["CADUSD"] = [-0.5, 0.2]
# ... etc for all 5
```

### 3.2 HITFIRST (dim=3)

Ticker: `ETHHITFIRST` (price_key: `ETH`)

Horizon: 500 blocks. Three-way probability vector in $(0, 1)$ summing to 1: $[P(\text{up first}),\; P(\text{down first}),\; P(\text{neither})]$.

Barriers are set at $\pm 1\sigma$ of recent returns. Scoring: two independent L2 logistic regressions on logit-transformed miner probabilities (one for up-barrier-hit, one for down). Single fit on all valid samples (no walk-forward). Importance = $|\beta_j^{\text{up}}| + |\beta_j^{\text{down}}|$.

```python
embeddings["ETHHITFIRST"] = [0.4, 0.35, 0.25]
```

### 3.3 LBFGS (dim=17)

Tickers: `ETHLBFGS` (1h), `BTCLBFGS` (6h)

Two scoring paths combined 75/25:

**Classifier path (75%)** — `p[0:5]`: 5-bucket probability distribution over volatility regimes (boundaries at $\pm 1\sigma$, $\pm 2\sigma$). Must be in $(0, 1)$, sum to 1. Scoring: per-class L2 logistic regressions on argmax predictions, walk-forward segmented. Importance = $\sum_c \beta_{j,c}^2$. Uniqueness penalty suppresses miners with >85% argmax overlap with higher-ranked peers.

**Q-path (25%)** — `q[5:17]`: 12 exceedance probabilities. For tail buckets 0, 1, 3, 4 (not the center bucket 2), predict $P(|\text{return}| > k\sigma)$ at thresholds $k \in \{0.5, 1.0, 2.0\}$. Scoring: 12 independent binary L2 logistic models on logit-transformed probabilities. Importance = averaged $|\beta_j|$ across sub-models.

```
Index   Meaning
[0:5]   p[0..4] — regime probabilities
[5:8]   Q bucket 0, thresholds [0.5σ, 1.0σ, 2.0σ]
[8:11]  Q bucket 1, thresholds [0.5σ, 1.0σ, 2.0σ]
[11:14] Q bucket 3, thresholds [0.5σ, 1.0σ, 2.0σ]
[14:17] Q bucket 4, thresholds [0.5σ, 1.0σ, 2.0σ]
```

```python
import numpy as np

p = np.array([0.05, 0.15, 0.60, 0.15, 0.05])  # regime probs
q = np.random.uniform(0.01, 0.99, 12).tolist()   # exceedance probs

embeddings["ETHLBFGS"] = p.tolist() + q
embeddings["BTCLBFGS"] = p.tolist() + q
```

### 3.4 MULTI-BREAKOUT (dim=2 per asset, 33 assets)

Ticker: `MULTIBREAKOUT`

A state machine tracks rolling 4-day price ranges per asset. When price breaches a barrier (25% of range width), a breakout event triggers. Predict whether it continues or reverses.

**Parameters:**

| Parameter | Value |
|---|---|
| `range_lookback_blocks` | 28800 (4 days) |
| `barrier_pct` | 25% of range |
| `min_range_pct` | 1% (skip tight ranges) |

**Submission:** dict keyed by asset. Each value is $[P_{\text{continuation}},\; P_{\text{reversal}}]$ in $(0, 1)$.

```python
from config import BREAKOUT_ASSETS

embeddings["MULTIBREAKOUT"] = {
    asset: [float(np.clip(your_model(asset), 0.01, 0.99)),
            float(np.clip(1 - your_model(asset), 0.01, 0.99))]
    for asset in BREAKOUT_ASSETS
}
```

**Assets (33):**

```python
BREAKOUT_ASSETS = [
    "BTC", "ETH", "XRP", "SOL", "TRX", "DOGE", "ADA", "BCH", "XMR",
    "LINK", "LEO", "HYPE", "XLM", "ZEC", "SUI", "LTC", "AVAX", "HBAR", "SHIB",
    "TON", "CRO", "DOT", "UNI", "MNT", "BGB", "TAO", "AAVE", "PEPE",
    "NEAR", "ICP", "ETC", "ONDO", "SKY",
]
```

**Scoring:** two-stage. (1) AUC gate — per-miner AUC on $P_{\text{continuation}}$ vs realized label, requiring AUC > 0.5, ≥ 2 temporal episodes, and prediction std > 0.03. (2) L2 logistic regression on z-scored predictions from qualifying miners with episode-balanced sample weighting (each temporal episode gets equal total weight). Importance = $|\beta_j|$.

**Breakouts are rare.** ~1-5 per asset per day. Submissions only matter at the instant a breakout triggers. Continuous submission is mandatory.

### 3.5 XSEC-RANK (dim=1 per asset, 33 assets)

Ticker: `MULTIXSEC`

Horizon: 1200 blocks (4h). Predict which assets will have above-median forward returns relative to the cross-section.

**Submission:** dict keyed by asset. Each value is a single score in $[-1, 1]$.

**Label construction:** for each (timestep, asset) pair:

$$
y_{t,a} = \mathbf{1}\!\Big[r_{t \to t+h}^{(a)} > \text{median}_a\big(r_{t \to t+h}\big)\Big]
$$

All 33 assets are pooled into a single binary classification (33x sample multiplier). Walk-forward meta-model with AUC-scaled coefficients.

```python
from config import BREAKOUT_ASSETS

embeddings["MULTIXSEC"] = {
    asset: float(np.clip(your_score(asset), -1, 1))
    for asset in BREAKOUT_ASSETS
}
```

### 3.6 FUNDING-XSEC (dim=1 per asset, 20 assets)

Ticker: `FUNDINGXSEC`

Horizon: 2400 blocks (8h). Predict which assets' perpetual funding rates will change more than the cross-sectional median over the next settlement window.

**Label construction:**

$$
\Delta f_a = f_{t+h}^{(a)} - f_t^{(a)}
$$

$$
y_{t,a} = \mathbf{1}\!\Big[\Delta f_a > \text{median}_a(\Delta f)\Big]
$$

Using changes rather than levels destroys the high autocorrelation in funding rate levels ($\phi \approx 0.97$) and isolates asset-specific deviations. The cross-sectional median subtraction removes the market-wide funding factor (beta). Base rate is exactly 50% by construction.

**Submission:** dict keyed by asset. Each value is a single score in $[-1, 1]$. Positive = expect above-median funding change. Magnitude matters (used as logistic regression feature). Missing assets default to 0.0 (neutral).

```python
from config import FUNDING_ASSETS

embeddings["FUNDINGXSEC"] = {
    asset: float(np.clip(your_model(asset), -1, 1))
    for asset in FUNDING_ASSETS
}
```

**Assets (20):**

```python
FUNDING_ASSETS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "SUI",
    "NEAR", "AAVE", "UNI", "LTC", "HBAR", "PEPE", "TRX", "SHIB", "TAO", "ONDO",
]
```

**Scoring:** identical structure to XSEC-RANK. All 20 assets pooled (20x sample multiplier). Walk-forward meta-model with embargo $= \max(\text{LAG}, \text{ahead})$. Stale miners (temporal std < $10^{-4}$ per asset column) are zeroed before pooling.

**Useful features:**
- Current funding rate levels (mean reversion: extreme rates tend to normalize)
- Open interest changes and long/short ratio shifts
- Recent price momentum relative to peers
- Liquidation volume and order book skew
- Cross-asset lead-lag (BTC funding often leads alts by 1-2 settlement periods)

### 3.7 FLOW (dim=28)

Ticker: `FLOW`. Live 2026-08-17; emissions turn on 2026-09-14 at a
25% pool allocation. Full specification, math, and simulation
evidence: the FLOW release paper (`FLOW_RELEASE.pdf`, published on
the MANTIS site).

Capital-at-risk BTC bracket trades instead of predictions. 28 floats,
seven per regime, in regime order A, B, C, D:

```python
# per regime: [d, f, sl, tp1, tp2, h, trade_id]
#   d         -1.0 short, +1.0 long, 0.0 no trade
#   f         Kelly fraction in [0.01, 0.25] (max bet: a quarter of collateral)
#   sl        stop distance as a fraction of entry (0 < sl <= 0.5)
#   tp1, tp2  target distances as fractions of entry (tp1 < tp2 <= 0.5)
#   h         horizon in hours, within the regime's bounds
#   trade_id  miner-chosen positive integer; bump it to open a new trade
```

Regime horizon bounds: A = 1–24h, B = 24–48h, C = 72–168h,
D = 168–336h. Entry prices are recorded by the validator from its own
feed at the observed open; nothing in the payload can set a price.
Semantics are constant-emission: repeat your current intent every
payload and bump `trade_id` to open. A dropped payload delays an open
by at most one sampling interval and never cancels a live trade. Opens
are rate-limited to one per hour per regime; implied win probability
$p = (fb + 1)/(b + 1)$ must lie in $[0.01, 0.99]$ or the submission is
rejected.

**Payment is evidence-gated, not score-proportional.** Realized
per-trade R sums into 168h blocks; with ≥ 4 blocks a t-statistic is
computed, a key clears at $t \geq 1.25$ (on the better of its full
history and its last four blocks) and stays cleared while
$t \geq 0.5$. Cleared keys split 98% of the challenge pool pro rata;
uncleared keys with positive score split a 2% dust tier; the remainder
burns. New keys earn nothing for their first 168h (probation). There
is no entry deposit. Once the collateral pool is configured, emission
weight and settlement money are both priced by the bet you posted
(see the owner-settlement section of `README.md`), and every 7 days
the alpha lost on losing trades is redistributed to winners; there is
no slash mechanic on posted alpha.

**THE MODEL: you post your own bets, by hand, on chain — we can only
resolve them.** Every trade the money layer will ever touch exists
because you personally put it there: `postTrade(hotkey, tradeKey,
collateral)` reserves the bet's worst case (Kelly f × your collateral)
behind the trade id, before the payload that carries it uploads. We
have no way to open, resize, or extend a bet — our entire write
surface over your capital is resolving your bets (each debit
hard-bounded by the bet you posted), settling the weekly pool
(zero-sum, enforced), and repairing rounding dust. What you never
post can never lose — or win — anything.

**Opening the position is hotkey-authorized.** Each registered hotkey
has one slot. Whoever opens it becomes the depositor (the only key
that can later top up or withdraw) and fixes the refund coldkey that
receives exits. If opening were unsigned, anyone could fund 1 alpha
against every registered hotkey and lock those slots — the squat.
So the only way to create a position is `addCollateralSigned`: your
hotkey signs a digest binding chain, contract, hotkey, the funder
(`msg.sender`, who becomes the depositor), the refund coldkey, the
amount, and a strictly increasing nonce. Without that signature there
is no position. Top-ups after that are depositor-only
(`addCollateral`) and do not need the hotkey again. `flow_post.py
fund` does both.

```bash
# Get alpha under your EVM key's mirror coldkey first
# (btcli evm stake, or transfer_stake to the mirror ss58 of the H160).
export FLOW_RPC=https://lite.chain.opentensor.ai
export FLOW_CONTRACT=0xD9c805202b16671A2901307fBC9A8750E2453427
export FLOW_EVM_KEY=0x...           # funder; becomes the depositor
python3 flow_post.py fund --hotkey 5F... --alpha 10 \
    --wallet mywallet --wallet-hotkey myhotkey
    # optional: --refund-coldkey 5G...  (defaults to the EVM key's mirror)
python3 flow_post.py book --hotkey 5F...
# later top-up from the same depositor: no --wallet needed
python3 flow_post.py fund --hotkey 5F... --alpha 5
```

A hotkey-signed `evict` force-exits a *flat* position (no open bets,
no parked pool rao) and frees the slot. The payout can go only to the
**recorded** refund coldkey — it cannot redirect a rao, whoever signs
or relays. Two uses: you lost the depositor EVM key and want your
alpha back on your own coldkey so you can re-open with a fresh key;
or a position exists against your wishes and you want the slot back
(the funder just gets their own alpha). Open reserves expire on-chain
in at most 384h, so an evict is never blocked for long.

```bash
python3 flow_post.py evict --hotkey 5F... \
    --wallet mywallet --wallet-hotkey myhotkey
```

A stolen hotkey still cannot steal capital: the refund was fixed
under your signature at open, top-ups and withdrawals stay
depositor-only, and the worst a compromised hotkey can do is
force-exit *your money to your own coldkey*.

Your hotkey IS an sr25519 public key, and it is the key your position
is booked to, so you post the native way: sign the bet with the same
hotkey you mine with, and the contract verifies the signature through
the chain's sr25519 precompile (`postTradeSigned`). Any EVM account
can relay the transaction — it only pays gas and has no authority:
the signed digest binds chain, contract, hotkey, trade id, size and a
strictly increasing nonce, so a relayer can neither alter a bet nor
replay one. (The depositor EVM key that funded the position can also
post directly as a fallback.) Bets are immutable once posted — no
resize, no cancel. Direction and levels are not on chain: they live
in the encrypted payload on your R2 object and stay under the public
timelock, so you already know them while everyone else waits. Each
reserve expires on-chain by its
regime's horizon ceiling plus 48h, with `sweepExpired` permissionless,
so nothing can lock your margin. Post the bet BEFORE (or as) you
upload the payload with the matching trade id: settlement only
credits bets that were on chain when the trade opened — a bet posted
after the fact is voided both ways, since late posting would be a
free look at the tape. A bet reveals size only — direction and levels
stay under the timelock. `flow_post.py` is the supported miner surface: `fund` (open the
slot with your hotkey, or top up), `book`, `post` with `--wallet`
(signs each bet with your local hotkey), `status`, `audit` (replays
the contract's event log and proves every debit stayed within the
bet you posted), and `evict` (flat-only recovery). Wire `post` into
your submission loop so the bet and the payload go out together;
whatever UI you want on top is yours to build (and yours to trust
with your credentials).

**The clock: what happens to your bet, and when.** Settlement is run
by the team (the contract owner) as a separate process — its code is
not in this repo, but everything it may do is bounded on chain and
recomputable by you. The team decrypts payloads at arrival through
the envelope's owner leg (`W_owner`, §2.2 of the release doc), so
resolution is live; the public timelock opens the same plaintexts to
everyone a week later. Expect:

| When | What you see on chain |
| --- | --- |
| you open the position (`flow_post.py fund`) | `CollateralAdded`: your hotkey signed the funder and refund coldkey; nobody else can occupy this slot |
| you post the bet (before the trade opens; ≤ 15 min pipeline grace) | `TradePosted`: the reserve is held, the bet is immutable; everything above your reserves stays instantly withdrawable |
| your trade resolves (SL / TP1+horizon / TP2 / horizon) | `TradeClosed`, typically within minutes to a few hours: a loss debits `max(−R,0) × bet` — never more than the bet, never more than ¼ of your book per week — and a win or flat closes at loss 0, freeing the reserve |
| each week end + 24 h straggler lag | `Settled`: the week's collected losses pay that week's net winners pro rata, over-collections are refunded in the same batch, exactly zero-sum |
| one week after each payload | the drand timelock opens it publicly: you (or anyone) can recompute every close and settle from the panel, the tape, and the bets, and `flow_post.py audit` proves no debit exceeded a bet |
| regime ceiling + 48 h with no close | owner-outage escape hatch: the reserve self-releases and `sweepExpired` is permissionless — no team failure can lock your margin |

A bet posted after its trade opened is voided both ways (the
free-look guard is symmetric — it costs the late poster the win too).
Money and emissions share one basis: your posted bet. A trade with no
bet scores for evidence but carries no money and earns no emission;
idle balance behind unposted trades earns nothing; deposits or
withdrawals after a trade opened change nothing about it.

---

## 4. Full Embedding Assembly

```python
import numpy as np
from config import CHALLENGES, BREAKOUT_ASSETS, FUNDING_ASSETS

embeddings = {}

for spec in CHALLENGES:
    ticker = spec["ticker"]

    if ticker == "MULTIBREAKOUT":
        embeddings[ticker] = {a: [0.5, 0.5] for a in BREAKOUT_ASSETS}

    elif ticker == "MULTIXSEC":
        embeddings[ticker] = {a: 0.0 for a in BREAKOUT_ASSETS}

    elif ticker == "FUNDINGXSEC":
        embeddings[ticker] = {a: 0.0 for a in FUNDING_ASSETS}

    else:
        # Flat vectors, including FLOW (28 floats, see 3.7;
        # all-zeros = no open trades).
        embeddings[ticker] = np.zeros(spec["dim"]).tolist()

# Replace all zeros above with your actual model outputs.
```

---

## 5. Scoring Details

### Weight allocation

Per-challenge salience is normalized to sum to 1, then weighted:

| Challenge | Weight | Share of total |
|---|---|---|
| MULTI-BREAKOUT | 5.0 | ~21% |
| FUNDING-XSEC | 4.0 | ~17% |
| ETHLBFGS | 3.5 | ~15% |
| XSEC-RANK | 3.0 | ~13% |
| BTCLBFGS | 2.875 | ~12% |
| ETHHITFIRST | 1.25 | ~5% |
| Binary (ETH, CHFUSD, XAGUSD at 1.0; CADUSD, NZDUSD at 0.5) | 4.0 total | ~17% total |
| FLOW | 0.0 until 2026-09-14, then 7.875 | 0% → 25% |

> **TRADE-MIX is deprecated** as of the FLOW launch (2026-08-17): removed from the active roster, superseded by FLOW, historical challenge data purged from validator datalogs. Submissions to it are ignored.
>
> **FLOW emissions**: zero until 2026-09-14, then 25% of the pool. Within the challenge, everything the significance gate withholds is **burned**, not redistributed: uncleared miners split a 2% dust tier at most. Gate evidence accrues from your first submission and is never reset, so miners submitting from August 17th arrive at emission turn-on with four weeks of evidence banked.
>
> **FLOW probation**: a new hotkey earns nothing for its first 168h of history; the gate statistic needs at least 4 completed 168h blocks before it is defined at all.

### Scoring by challenge type

Not all challenges use the same scoring structure. Summary:

| Challenge | Scoring method | Importance metric |
|---|---|---|
| Binary | Walk-forward ElasticNet (L1/L2) meta-model on OOS base-model predictions | $\|\beta_j\|$ |
| LBFGS | 75% classifier path (per-class L2 logreg, $\beta_j^2$, uniqueness penalty) + 25% Q-path (12 sub-models, averaged $\|\beta_j\|$) | blended |
| HITFIRST | Two single-fit L2 logistic regressions (up-hit, down-hit) — no walk-forward | $\|\beta_j^{\text{up}}\| + \|\beta_j^{\text{down}}\|$ |
| MULTI-BREAKOUT | AUC gate → L2 logreg on z-scored predictions, episode-balanced weighting | $\|\beta_j\|$ |
| XSEC-RANK | Walk-forward L2 meta-model, AUC-scaled coefficients, recency-weighted segments | $\|\beta_j\| \cdot \text{AUC\_scale}$ |
| FUNDING-XSEC | Same as XSEC-RANK + stale filter ($\text{std} < 10^{-4}$) + extended embargo | $\|\beta_j\| \cdot \text{AUC\_scale}$ |
| FLOW | Bracket trades resolved against multi-venue klines (stop-first ties) → path-penalized, tail-amplified, Kelly-weighted R → per-regime EWMA → significance gate on 168h block sums (clear $t \geq 1.25$ on the max of full-history and rolling last-4 statistics, latch $t \geq 0.5$) → cleared keys split the pool pro rata, uncleared split 2% dust, remainder burned | block-sum $t$-statistic (full history or last 4 blocks, whichever is stronger); unearned pool burned |

For challenges with walk-forward segments, recency weighting applies: $w_i = \gamma^{n - 1 - i}$ where $\gamma = 0.5^{1/\text{HALFLIFE}}$.

### What gets you zero weight

- Submitting constant values (temporal std < $10^{-4}$)
- Submitting all zeros
- Random noise (AUC ≈ 0.5 → coefficient pushed to zero by L1/L2)
- Copying a top miner (L2 splits coefficient mass among clones)

---

## 6. Validation Checklist

```python
from config import CHALLENGES, BREAKOUT_ASSETS, FUNDING_ASSETS

def validate_embeddings(emb: dict) -> list[str]:
    errors = []
    for spec in CHALLENGES:
        tk = spec["ticker"]
        if tk not in emb:
            errors.append(f"Missing {tk}")
            continue
        val = emb[tk]

        if tk == "MULTIBREAKOUT":
            if not isinstance(val, dict):
                errors.append(f"{tk}: expected dict"); continue
            for a in BREAKOUT_ASSETS:
                v = val.get(a)
                if not isinstance(v, list) or len(v) != 2:
                    errors.append(f"{tk}.{a}: need [p_cont, p_rev]")
                elif not all(0 < x < 1 for x in v):
                    errors.append(f"{tk}.{a}: values must be in (0,1)")

        elif tk == "MULTIXSEC":
            if not isinstance(val, dict):
                errors.append(f"{tk}: expected dict"); continue
            for a in BREAKOUT_ASSETS:
                v = val.get(a, None)
                if not isinstance(v, (int, float)) or not (-1 <= v <= 1):
                    errors.append(f"{tk}.{a}: need float in [-1,1]")

        elif tk == "FUNDINGXSEC":
            if not isinstance(val, dict):
                errors.append(f"{tk}: expected dict"); continue
            for a in FUNDING_ASSETS:
                v = val.get(a, None)
                if v is not None and (not isinstance(v, (int, float)) or not (-1 <= v <= 1)):
                    errors.append(f"{tk}.{a}: need float in [-1,1]")

        else:
            if not isinstance(val, list) or len(val) != spec["dim"]:
                errors.append(f"{tk}: expected list of length {spec['dim']}")

    return errors
```

---

## 7. Model Iteration Tool

The [MANTIS Model Iteration Tool](https://github.com/BarbarianDev/mantis_model_iteration_tool) is an open-source framework for developing and backtesting MANTIS mining strategies. It runs autonomous agents that implement `Featurizer` and `Predictor` classes, evaluates them with causal data access and walk-forward backtesting, and tracks iterations in a web dashboard.

**Quick start:**

```bash
git clone https://github.com/BarbarianDev/mantis_model_iteration_tool.git
cd mantis_model_iteration_tool
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m mantis_model_iteration_tool.gui   # http://127.0.0.1:8420
```

**Supported challenges:**

| Challenge | Metric |
|---|---|
| `ETH-1H-BINARY` | AUC |
| `ETH-HITFIRST-100M` | Log loss |
| `ETH-LBFGS` / `BTC-LBFGS-6H` | Balanced accuracy |
| `MULTI-BREAKOUT` | AUC |
| `XSEC-RANK` / `FUNDING-XSEC` | Spearman |

**SDK example:**

```python
from mantis_model_iteration_tool import Featurizer, Predictor, evaluate
import numpy as np

class MyFeaturizer(Featurizer):
    warmup = 200
    compute_interval = 1

    def compute(self, view):
        prices = view.prices("ETH")
        returns = np.diff(np.log(prices[-100:]))
        return {"momentum": np.array([returns.mean()]),
                "volatility": np.array([returns.std()])}

class MyPredictor(Predictor):
    def predict(self, features):
        p_up = 0.5 + 0.5 * np.tanh(features["momentum"][0] * 500)
        return np.array([p_up, 1.0 - p_up])

result = evaluate("ETH-1H-BINARY", MyFeaturizer(), MyPredictor(), days_back=60)
```

The tool enforces causal data access (no future leakage), containerized execution with resource limits, and structured iteration tracking. See the [repository](https://github.com/BarbarianDev/mantis_model_iteration_tool) for full documentation.

---

## 8. Common Mistakes

- **Wrong ticker key**: use `"MULTIBREAKOUT"` not `"MULTI-BREAKOUT"`, `"FUNDINGXSEC"` not `"FUNDING-XSEC"`.
- **LBFGS probabilities don't sum to 1**: `p[0:5]` must form a valid distribution.
- **HITFIRST probabilities outside (0,1)**: hard zeros or ones cause log-loss issues.
- **Breakout values at boundary**: use `(0, 1)` not `[0, 1]`.
- **Stale submissions**: if your embedding doesn't change across timestamps, the stale filter zeroes your contribution.
- **Missing challenges**: omitting a challenge means zero weight for that fraction of emissions.
