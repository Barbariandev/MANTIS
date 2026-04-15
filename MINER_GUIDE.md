# Miner Guide

## 1. Setup

```bash
pip install timelock requests cryptography boto3 python-dotenv
```

Requirements:
- Python 3.10+
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

Horizon: 300 blocks (1h). Two features in \([-1, 1]\). These are inputs to a logistic regression classifier, not probabilities. Scoring: feature selection (per-miner L2 logistic, AUC on held-out half, top-50 selected), then ElasticNet (L1 ratio 0.5) meta-model on walk-forward OOS base-model predictions. Importance = \(|\beta_j|\).

```python
embeddings["ETH"] = [0.3, -0.1]       # your model output
embeddings["CADUSD"] = [-0.5, 0.2]
# ... etc for all 5
```

### 3.2 HITFIRST (dim=3)

Ticker: `ETHHITFIRST` (price\_key: `ETH`)

Horizon: 500 blocks. Three-way probability vector in \((0, 1)\) summing to 1: \([P(\text{up first}),\; P(\text{down first}),\; P(\text{neither})]\).

Barriers are set at \(\pm 1\sigma\) of recent returns. Scoring: two independent L2 logistic regressions on logit-transformed miner probabilities (one for up-barrier-hit, one for down). Single fit on all valid samples (no walk-forward). Importance = \(|\beta_j^{\text{up}}| + |\beta_j^{\text{down}}|\).

```python
embeddings["ETHHITFIRST"] = [0.4, 0.35, 0.25]
```

### 3.3 LBFGS (dim=17)

Tickers: `ETHLBFGS` (1h), `BTCLBFGS` (6h)

Two scoring paths combined 75/25:

**Classifier path (75%)** — `p[0:5]`: 5-bucket probability distribution over volatility regimes (boundaries at \(\pm 1\sigma\), \(\pm 2\sigma\)). Must be in \((0, 1)\), sum to 1. Scoring: per-class L2 logistic regressions on argmax predictions, walk-forward segmented. Importance = \(\sum_c \beta_{j,c}^2\). Uniqueness penalty suppresses miners with >85% argmax overlap with higher-ranked peers.

**Q-path (25%)** — `q[5:17]`: 12 exceedance probabilities. For tail buckets 0, 1, 3, 4 (not the center bucket 2), predict \(P(|\text{return}| > k\sigma)\) at thresholds \(k \in \{0.5, 1.0, 2.0\}\). Scoring: 12 independent binary L2 logistic models on logit-transformed probabilities. Importance = averaged \(|\beta_j|\) across sub-models.

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

**Submission:** dict keyed by asset. Each value is \([P_{\text{continuation}},\; P_{\text{reversal}}]\) in \((0, 1)\).

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

**Scoring:** two-stage. (1) AUC gate — per-miner AUC on \(P_{\text{continuation}}\) vs realized label, requiring AUC > 0.5, ≥ 2 temporal episodes, and prediction std > 0.03. (2) L2 logistic regression on z-scored predictions from qualifying miners with episode-balanced sample weighting (each temporal episode gets equal total weight). Importance = \(|\beta_j|\).

**Breakouts are rare.** ~1-5 per asset per day. Submissions only matter at the instant a breakout triggers. Continuous submission is mandatory.

### 3.5 XSEC-RANK (dim=1 per asset, 33 assets)

Ticker: `MULTIXSEC`

Horizon: 1200 blocks (4h). Predict which assets will have above-median forward returns relative to the cross-section.

**Submission:** dict keyed by asset. Each value is a single score in \([-1, 1]\).

**Label construction:** for each (timestep, asset) pair:

\[
y_{t,a} = \mathbf{1}\!\Big[r_{t \to t+h}^{(a)} > \text{median}_a\big(r_{t \to t+h}\big)\Big]
\]

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

\[
\Delta f_a = f_{t+h}^{(a)} - f_t^{(a)}
\]
\[
y_{t,a} = \mathbf{1}\!\Big[\Delta f_a > \text{median}_a(\Delta f)\Big]
\]

Using changes rather than levels destroys the high autocorrelation in funding rate levels (\(\phi \approx 0.97\)) and isolates asset-specific deviations. The cross-sectional median subtraction removes the market-wide funding factor (beta). Base rate is exactly 50% by construction.

**Submission:** dict keyed by asset. Each value is a single score in \([-1, 1]\). Positive = expect above-median funding change. Magnitude matters (used as logistic regression feature). Missing assets default to 0.0 (neutral).

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

**Scoring:** identical structure to XSEC-RANK. All 20 assets pooled (20x sample multiplier). Walk-forward meta-model with embargo \(= \max(\text{LAG}, \text{ahead})\). Stale miners (temporal std < \(10^{-4}\) per asset column) are zeroed before pooling.

**Useful features:**
- Current funding rate levels (mean reversion: extreme rates tend to normalize)
- Open interest changes and long/short ratio shifts
- Recent price momentum relative to peers
- Liquidation volume and order book skew
- Cross-asset lead-lag (BTC funding often leads alts by 1-2 settlement periods)

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
        embeddings[ticker] = np.zeros(spec["dim"]).tolist()

# Replace all zeros above with your actual model outputs.
```

---

## 5. Scoring Details

### Weight allocation

Per-challenge salience is normalized to sum to 1, then weighted:

| Challenge | Weight | Share of total |
|---|---|---|
| MULTI-BREAKOUT | 5.0 | ~22% |
| FUNDING-XSEC | 4.0 | ~17% |
| ETHLBFGS | 3.5 | ~15% |
| XSEC-RANK | 3.0 | ~13% |
| BTCLBFGS | 2.875 | ~12% |
| ETHHITFIRST | 2.5 | ~11% |
| Binary (5x) | 1.0 each | ~22% total |

### Scoring by challenge type

Not all challenges use the same scoring structure. Summary:

| Challenge | Scoring method | Importance metric |
|---|---|---|
| Binary | Walk-forward ElasticNet (L1/L2) meta-model on OOS base-model predictions | \(\|\beta_j\|\) |
| LBFGS | 75% classifier path (per-class L2 logreg, \(\beta_j^2\), uniqueness penalty) + 25% Q-path (12 sub-models, averaged \(\|\beta_j\|\)) | blended |
| HITFIRST | Two single-fit L2 logistic regressions (up-hit, down-hit) — no walk-forward | \(\|\beta_j^{\text{up}}\| + \|\beta_j^{\text{down}}\|\) |
| MULTI-BREAKOUT | AUC gate → L2 logreg on z-scored predictions, episode-balanced weighting | \(\|\beta_j\|\) |
| XSEC-RANK | Walk-forward L2 meta-model, AUC-scaled coefficients, recency-weighted segments | \(\|\beta_j\| \cdot \text{AUC\_scale}\) |
| FUNDING-XSEC | Same as XSEC-RANK + stale filter (\(\text{std} < 10^{-4}\)) + extended embargo | \(\|\beta_j\| \cdot \text{AUC\_scale}\) |

For challenges with walk-forward segments, recency weighting applies: \(w_i = \gamma^{n - 1 - i}\) where \(\gamma = 0.5^{1/\text{HALFLIFE}}\).

### What gets you zero weight

- Submitting constant values (temporal std < \(10^{-4}\))
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

## 7. Common Mistakes

- **Wrong ticker key**: use `"MULTIBREAKOUT"` not `"MULTI-BREAKOUT"`, `"FUNDINGXSEC"` not `"FUNDING-XSEC"`.
- **LBFGS probabilities don't sum to 1**: `p[0:5]` must form a valid distribution.
- **HITFIRST probabilities outside (0,1)**: hard zeros or ones cause log-loss issues.
- **Breakout values at boundary**: use `(0, 1)` not `[0, 1]`.
- **Stale submissions**: if your embedding doesn't change across timestamps, the stale filter zeroes your contribution.
- **Missing challenges**: omitting a challenge means zero weight for that fraction of emissions.
