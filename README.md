# MANTIS

Bittensor Subnet 123 — Multi-challenge signal aggregation network.

---

## Architecture

Validators sample miner payloads every `SAMPLE_EVERY` blocks, decrypt them after a timelock maturation window, and store (embedding, price) pairs in a SQLite database. Periodically, walk-forward scoring computes per-hotkey salience for each challenge, aggregates across challenges by weight, applies EMA smoothing, and sets on-chain weights.

```mermaid
graph TD
    subgraph validator["validator.py"]
        A["Sample block"] --> B["cycle.get_miner_payloads()"]
        B --> C["ledger.append_step()"]
        A --> D["Periodic weight calc"]
        D --> E["ledger.iter_challenge_training_data()"]
        E --> F["model.multi_salience()"]
        F --> G["EMA smooth + set_weights()"]
    end

    subgraph ledger["ledger.py (SQLite)"]
        H["challenge_data"]
        I["raw_payloads"]
        J["drand_cache"]
    end

    subgraph external["External"]
        K["Miners (R2)"]
        L["Drand beacon"]
        M["Subtensor"]
        N["price_service.py → R2"]
    end

    C --> H
    C --> I
    E -- reads --> H
    B -- downloads --> K
    I -- decrypt via --> L
    G -- writes --> M
    B -- reads commits --> M
    N -- publishes --> H
```

---

## Challenges

All challenges are defined in `config.py` under `CHALLENGES`. Each specifies a `ticker`, `dim`, `blocks_ahead` (forward horizon in blocks at 12s/block), `loss_func` (scoring dispatch key), and `weight` (relative importance in final aggregation).

| Challenge | Ticker | dim | Horizon | loss\_func | Weight | Description |
|---|---|---|---|---|---|---|
| ETH-1H-BINARY | `ETH` | 2 | 300 (1h) | `binary` | 1.0 | Binary direction prediction |
| CADUSD-1H-BINARY | `CADUSD` | 2 | 300 | `binary` | 1.0 | " |
| NZDUSD-1H-BINARY | `NZDUSD` | 2 | 300 | `binary` | 1.0 | " |
| CHFUSD-1H-BINARY | `CHFUSD` | 2 | 300 | `binary` | 1.0 | " |
| XAGUSD-1H-BINARY | `XAGUSD` | 2 | 300 | `binary` | 1.0 | " |
| ETH-HITFIRST | `ETHHITFIRST` | 3 | 500 | `hitfirst` | 2.5 | Barrier-hit direction |
| ETH-LBFGS | `ETHLBFGS` | 17 | 300 (1h) | `lbfgs` | 3.5 | Volatility regime + quantile paths |
| BTC-LBFGS-6H | `BTCLBFGS` | 17 | 1800 (6h) | `lbfgs` | 2.875 | " |
| MULTI-BREAKOUT | `MULTIBREAKOUT` | 2/asset | event | `range_breakout_multi` | 5.0 | Range breakout continuation/reversal (33 assets) |
| XSEC-RANK | `MULTIXSEC` | 1/asset | 1200 (4h) | `xsec_rank` | 3.0 | Cross-sectional return ranking (33 assets) |
| FUNDING-XSEC | `FUNDINGXSEC` | 1/asset | 2400 (8h) | `funding_xsec` | 4.0 | Cross-sectional funding rate ranking (20 assets) |

---

## Scoring

### Per-challenge scoring

Each `loss_func` has its own scoring path. All use L2 logistic regression and coefficient-based importance, but the structure differs.

**Binary** (`binary`) — Walk-forward with ElasticNet meta-model. Feature selection: per-miner L2 logistic on first half, AUC on second half, select top-\(K\) (default 50). Meta-model: ElasticNet logistic (L1 ratio 0.5) on OOS base-model predictions across walk-forward segments. Importance = \(|\beta_j|\). Segments weighted by recency.

**LBFGS** (`lbfgs`) — Two independent scoring paths blended 75/25:
- *Classifier path* (`compute_linear_salience`): per-class L2 logistic regressions on 5-bucket argmax predictions. Importance = \(\beta_j^2\) summed across classes. Vectorized balanced accuracy evaluation. Uniqueness penalty suppresses miners with >85% argmax overlap with higher-ranked peers.
- *Q-path* (`compute_q_path_salience`): 12 independent binary L2 logistic models (one per tail-bucket / sigma-threshold combination). Importance = averaged \(|\beta_j|\) across sub-models.

Both paths are individually top-\(K\) renormalized with exponential rank decay before blending.

**HITFIRST** (`hitfirst`) — Two L2 logistic regressions on logit-transformed miner probabilities: one for up-barrier-hit (\(y=1\) if price hits +\(\sigma\) first), one for down-barrier-hit. Importance = \(|\beta_j^{\text{up}}| + |\beta_j^{\text{down}}|\). No walk-forward — single fit on all valid samples.

**MULTI-BREAKOUT** (`range_breakout_multi`) — Operates on completed breakout events (not time series). Two-stage: (1) Empirical AUC gate — per-miner AUC on \(P_{\text{continuation}}\) vs realized label, requiring AUC > 0.5 and ≥ 2 temporal episodes. (2) L2 logistic on z-scored miner predictions with episode-balanced sample weighting (each temporal episode gets equal total weight regardless of event count). Importance = \(|\beta_j|\).

**XSEC-RANK** (`xsec_rank`) — Cross-sectional binary reformulation: label = 1 if asset's forward return exceeds the cross-sectional median. All assets pooled (\(N_{\text{assets}} \times\) sample multiplier). Walk-forward meta-model: feature selection by per-miner univariate AUC, top-\(K\) (default 20) selected, L2 logistic meta-model. Importance per segment:

\[
w_j = |\beta_j| \cdot \max\!\Big(\frac{\text{AUC}_{\text{meta}} - 0.5}{0.5},\; 0\Big)
\]

Segments aggregated with exponential recency weighting.

**FUNDING-XSEC** (`funding_xsec`) — Same structure as XSEC-RANK but on funding rate changes instead of price returns. Embargo = \(\max(\text{LAG}, \text{ahead})\) with explicit `train_cutoff = val_start - ahead` to prevent label leakage from forward-looking labels. Stale miners (temporal std < \(10^{-4}\) per asset column) zeroed before pooling.

### Sybil resistance

L2 regularization splits coefficient mass among correlated miners. If \(n\) clones submit identical predictions, each receives \(\approx w/n\) weight. L1 (in binary challenges) or the uniqueness penalty (in LBFGS) drives zero-information or duplicate miners to zero.

### Weight aggregation

Per-challenge salience vectors are normalized to sum to 1, multiplied by challenge weight, and averaged:

\[
s_j = \frac{1}{\sum_c w_c} \sum_c w_c \cdot \hat{s}_{j,c}
\]

EMA smoothing (\(\alpha = 0.15\)) is applied across weight-setting intervals to reduce block-to-block variance. Degenerate distributions (near-uniform or zero-sum) are rejected.

---

## Encryption

Dual-path encryption ensures no party can observe predictions before maturation:

1. **Owner path** — X25519 ECDH + ChaCha20-Poly1305 AEAD. The owner can decrypt immediately for trading.
2. **Timelock path** — Drand IBE (BLS12-381). After the specified Drand round, validators decrypt via the published beacon signature.

A SHA-256 binding hash over (hotkey, round, owner\_pk, ephemeral\_pk) is used as AAD, preventing replay, relay, and substitution attacks.

---

## Modules

| File | Role |
|---|---|
| `config.py` | Challenge definitions, network constants, encryption params |
| `validator.py` | Block sampling, payload collection, decryption scheduling, weight setting |
| `ledger.py` | SQLite storage, submission validation, training data iteration, Drand cache |
| `model.py` | `multi_salience()` — dispatches to per-challenge scoring, aggregates |
| `cycle.py` | Miner payload download, commit URL validation |
| `funding_xsec.py` | FUNDING-XSEC scoring: forward pairing, label construction, walk-forward |
| `xsec_rank.py` | XSEC-RANK scoring |
| `range_breakout.py` | MULTI-BREAKOUT state machine + scoring |
| `bucket_forecast.py` | LBFGS classifier + Q-path salience |
| `hitfirst.py` | HITFIRST barrier-hit scoring |
| `price_service.py` | Fetches spot prices (Polygon) + funding rates (OKX/HL/CoinGlass), uploads to R2 |

---

## Storage

SQLite with WAL mode. Tables:

- **`challenge_data`** — `(ticker, sidx)` → `price` or `price_data` (JSON for multi-asset), `hotkeys` (JSON list), `embeddings` (binary float16 blob)
- **`challenge_meta`** — `(ticker)` → `dim`, `blocks_ahead`
- **`block_index`** — Sequential index → block number mapping
- **`raw_payloads`** — Encrypted ciphertexts held until maturation
- **`drand_cache`** — Cached beacon signatures
- **`breakout_state`** — Serialized range tracker state

Training data is streamed via generator iteration, not loaded into memory.

---

## Key Parameters

| Parameter | Value | Location |
|---|---|---|
| `SAMPLE_EVERY` | 5 blocks (60s) | `config.py` |
| `LAG` | 60 samples | `config.py` |
| `TASK_INTERVAL` | 500 blocks | `config.py` |
| `WEIGHT_CALC_INTERVAL` | 1000 blocks | `config.py` |
| `WEIGHT_SET_INTERVAL` | 360 blocks | `config.py` |
| `BURN_PCT` | 0.30 (UID 0) | `config.py` |
| `MAX_DAYS` | 60 | `config.py` |
| `EMA alpha` | 0.15 | `validator.py` |
| `TOP_K` (feature selection) | 20 | `funding_xsec.py`, `xsec_rank.py` |

---

## Payload Format

V2 JSON only. Required fields: `v`, `round`, `hk`, `owner_pk`, `C`, `W_owner`, `W_time`, `binding`, `alg`.

Commit constraints:
- Host: Cloudflare R2 (`*.r2.dev` or `*.r2.cloudflarestorage.com`)
- Object key: exactly your hotkey (no path segments)
- Size: ≤ 25 MB

---

## Dependencies

Declared in `pyproject.toml`. Core: `bittensor`, `torch`, `scikit-learn`, `numpy`, `requests`, `aiohttp`, `tqdm`, `boto3`.

See `MINER_GUIDE.md` for submission details per challenge type.

---

## License

MIT License (c) 2024 MANTIS
