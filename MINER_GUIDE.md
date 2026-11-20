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

### 3.8 LOOP (dim=28)

Ticker: `LOOP`. The autoresearch challenge: your edge is the
**pipeline** you design — a program that spends inference and
experiments to find a trading model. You publish its source and the
auditor — the subnet owner's validator, UID 0 — runs it. Its share is
5% of the pool, raised as the
field proves out. **Start with
`LOOP.md`** (submitting, incentives, launch parameters); the full spec
is `autoresearch/README.md`.

**Rounds.** LOOP runs on a **24-hour round**: a new evaluation round
opens at **0000 UTC** and covers every submission the auditor received
in the 24 hours before it. There is no submission server — your
submission rides the same R2 payload you already upload, under the
`LOOP` field, and the auditor, which holds the owner key every payload
is wrapped to, reads it as soon as it fetches your payload (about once
a minute). Upload by midnight and it runs that night; leave a few
minutes' margin. At midnight
the auditor validator takes each hotkey's newest submission, screens
its source and args with a strong open-weights model for **sandbox-escape
attempts** (escape = never runs, verdict on the dashboard; executing
code you hide — encoded, compressed or encrypted — counts), runs the
survivors **sealed** — in a gVisor sandbox with no network, no host
files, a fixed share of cores and memory, and one channel to the
research broker — audits the results (see below), publishes the signed
verdict map, and redraws the public dashboard
(<https://pub-879ad825983e43529792665f4f510cd6.r2.dev/loop/index.html>).
While a round runs the dashboard lists its entries and their status
only; when the round closes it shows **the full source of every
evaluated submission**, its research and its verdict, and from then on
they are fair game for later rounds: LOOP pays for published research.
Until then your code is private to the auditor and the models it
screens and judges with — provided your payload's timelock outlasts
the round. Encrypt it with the week-long lock
(`TLOCK_PROD_SUGGESTED_LOCK_SECONDS`), not the 30 seconds of the §2
example, or anyone can read your code off R2 as soon as the lock opens.

**What you send** (`autoresearch.submission.build_loop_field`):
pipeline source, JSON arguments, a budget, and your SayGM (`gm_live_...`)
key sealed to LOOP's rotating round key:

```python
from autoresearch.submission import build_loop_field

payload_obj["LOOP"] = build_loop_field(
    MY_HOTKEY,
    open("my_pipeline.py").read(),
    args={"iterations": 12},
    budget_credits=4.0,
    api_key=MY_SAYGM_KEY,                  # dedicated <= $100 SayGM burner key
    trade=current_trade_vec)               # your hourly 28-float trade
# ...encrypt payload_obj under the week-long lock
# (lock_seconds=TLOCK_PROD_SUGGESTED_LOCK_SECONDS) and upload to R2.
```

One submission runs **once**: it is named by the sha256 of
(source, args, budget), and each (hotkey, sha) is evaluated in the
round after the auditor first read it. Re-running identical code earns
nothing anyway (pay follows *progress*, below) — change something to
submit again. With your own key the budget is your own spend cap —
there is no ceiling; the **wall clock** is the bottleneck. Every run
in a round gets the same one, and the same share of the machine (a
core, 4 GB of memory, 64 MB of scratch). The clock is 12 hours when
few enter, shorter when many do, so that the whole round fits in 20
hours (at 20 runs at a time, 200 entries that pass the screen get 2
hours each); the
round's page states it, and `loop.budget()["seconds_left"]` tells your
pipeline how much is left. Build pipelines that checkpoint their
judgment and call `loop.finish()` before the deadline. Results land
on the dashboard: your entry's status
(`refused` / `blocked` / `unscreened` / `failed` / `audited`) and, by
the time the round closes, the reason in plain text, the verdict
multiplier, the full trace and the dossier.

**Inside a pipeline** (`import loop_sdk as loop`): `loop.data()` (the
dataset: hourly BTC — the target, last 480 bars held back — plus ~10
years of daily BTC, EURUSD, USDJPY, GBPUSD, AUDUSD, USDCHF, ETH and
gold, all cut at the same moment), `loop.models()` (every SayGM SN28
model with pricing), `loop.llm(model, messages, step=...)` (any model,
any mix of models, text in / text out), `loop.propose(spec)`,
`loop.experiment(candidate)`, `loop.critique(...)`, `loop.promote(...)`,
`loop.prune(...)`, `loop.finish(narrative, hyperparameter_rationale)`,
and `loop.checkpoint(state)` / `loop.restore()` — your pipeline's own
memory across a pause (one JSON slot only you can write).
A model is a **spec** — one feature from the published vocabulary
(transform, lookback 8–720 h, and the series it reads: a cross-market
feature on forex or gold is a first-class move) plus bracket
hyperparameters — and the protocol's harness is the only thing that
scores it. You cannot report a metric,
write a trace event, or write a dossier: the runner produces all of
them. Start from `autoresearch/pipelines/reference.py`; develop locally
for free with `python -m autoresearch.sealed my_pipeline.py`.

**Pauses.** When the auditor is stopped mid-run (operator maintenance,
a validator restart) your run is **parked, not killed**: any SDK call
raises `loop.Paused`, you get a grace period in which
`loop.checkpoint()` still works, and the run resumes on the next start
— same trace, same candidate ids, same budget, and the run's clock
picks up where it stopped. Call `loop.restore()` first and checkpoint
every iteration (the reference pipeline shows the shape) and a pause
costs you at most one iteration. Resumption is automatic for everyone:
your sealed key is re-opened from the submission still on record, so
the plaintext never needs to be stored and you never need to act.

**The dossier** is assembled by the runner and published for you: the
verdict that pays is the one on your latest audited round entry. **No
audited dossier, no pay.**

**The payload** is the FLOW wire format exactly (section 3.7): 28
floats, `[d, f, sl, tp1, tp2, h, trade_id]` per regime, scored by the
same bracket resolution/EWMA/significance machinery (f-only — no
collateral pool for LOOP). It should be your promoted model's trade
(`dossier["trade"]`, or `harness.payload_vector`).

```python
payload_obj["LOOP"] = {
    "trade": harness.payload_vector(dossier["trade"]),   # 28 floats
    "submit": {...},     # only when you are submitting a new pipeline
}
# a bare 28-float list still parses when you have nothing to submit
```

**The audit** multiplies your FLOW-criteria score by 1 or 0:

1. *Attested* — your dossier must be byte-for-byte the runner's, for a
   sealed run of your hotkey that ran to its end.
2. *Searched* — at least 3 experiments and a promoted model.
3. *Replayed* — the validator re-runs the harness on the run's dataset
   and must get your dossier's numbers.
4. *Judged* — a frontier model (Astra) reads your pipeline's source,
   the trace, the exchanges behind the promoted model and your run
   history against the ruleset (`autoresearch/audit.py::RULESET`) and
   answers one question: was anything smuggled in?  Not smuggled = 1,
   smuggled = 0.

Failing 1–3 is multiplier 0. What fails the judge, and nothing else
does: **smuggling** — a carried-in answer (hard-coded candidates,
prompts that steer to one, a search space of one, rerunning elsewhere
until something passed), a long-horizon conviction dressed as a
short-horizon model, or text in your source, prompts or narrative that
addresses the auditor. Honest and naive **passes**: the audit does not
judge whether your research is good — the market does, because pay
follows **progress**: your model's trades accumulate a FLOW EWMA score
as they resolve, and you earn by pushing the best audited score the
challenge has ever seen *higher* — a credit equal to your improvement,
halving every 14 days, gone after eight weeks. Matching the leader pays
nothing; dossiers are public, so study the leader's research and beat
it. When nobody has improved anything in eight weeks, the whole LOOP
slice burns. The
ruleset is a v0 and says so in its own text — and it is iterated, not
debated: gaming it is expected, and each new way that emerges gets an
amendment and a version bump, fast. Every validator applies the same
verdict: the auditor — always the validator holding UID 0 — signs its
verdict map and commits a pointer to it on chain with its hotkey; every
validator discovers it there, verifies it and pays on it, so there is
no judge lottery across validators. The harness still prints quality
receipts for your own
loop to use: `always_long_avg_r` for drift, `unfiltered_avg_r` for
naked factor exposure, and the quantile receipt — `quantile_r` (mean
forward return by feature band around the 1st/10th/50th/90th/99th
percentiles), `monotone` (±1 when the chain is strictly ordered) and
`monotone_robust` (the direction survives the lookback pulled ±25%).
A robust monotone response curve is the highest-confidence promote
signal there is; models that are secretly beta pass the gate and then
lose to the market, so prune them yourself.

**Inference is yours to provide (SayGM SN28 only — no OpenRouter)**:
put your own **SayGM** (`gm_live_...`) key in the submission.  OpenRouter
(`sk-or-...`) and non-SayGM keys are refused.  `build_loop_field` fetches
UID 0's signed rotating round seal key from `<LOOP_PUBLIC_URL>/seal_keys.json`
(separate from `W_owner`; old round private keys are shredded after the
resume window) and seals your key with `loop-submission:{hotkey}:{sha256}`
as associated data.  `key_sealed` is stripped before matured payloads
enter `datalog.db` and scrubbed from `loop_inbox` when your entry
finishes; decryption happens exclusively inside the isolated `loop-keyd`
process (`validator.py` never holds your plaintext key).

> **Enforced burner ceiling (checked before any code executes):**
> Fund a dedicated TAO/Alpha burner account on [saygm.com](https://saygm.com)
> with **<= $100** prepaid credit (`loop-keyd` checks `/v1/credits` and
> **refuses** keys above `$100` or at `$0`).  SayGM stops cold with HTTP
> `402` before dispatch if credit runs out, with zero auto-topup.

**Metering**: a run's budget covers its model calls (SayGM settled USD
cost), its experiments and its own running time; one run per hotkey at a
time. A submission without a key runs only on credits the owner granted
the hotkey; there is nothing to buy.

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
| LOOP | 1.073 | 5% |

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
