# MANTIS: a Distributed Signal Marketplace

MANTIS runs on Bittensor netuid 123 and incentivises predictions across multiple asset classes including crypto, forex, and commodities. Miners submit one embedding per challenge; validators decrypt the payloads after a timelock maturation window and measure each hotkey's contribution to forecasting returns.

## Submission and commit–reveal
- Miners upload a timelocked V2 payload to Cloudflare R2 and commit the URL on-chain. The payload contains embeddings keyed by challenge ticker and the miner's hotkey.
- Validators download the ciphertext every `SAMPLE_EVERY` blocks and record the current prices for every ticker.
- When the time lock matures (300+ blocks), the validator retrieves a Drand signature, decrypts the payload and checks that the embedded hotkey matches the commit.

## DataLog layout
The validator stores all state in a SQLite database with WAL mode:

- **`challenge_data`** — `(ticker, sidx)` → `price` or `price_data` (JSON for multi-asset), `hotkeys` (JSON list), `embeddings` (binary float16 blob)
- **`challenge_meta`** — `(ticker)` → `dim`, `blocks_ahead`
- **`blocks`** — Sequential index → block number mapping
- **`raw_payloads`** — Encrypted ciphertexts held until maturation
- **`drand_cache`** — Cached beacon signatures
- **`breakout_state`** — Serialized range tracker state

Prices and embeddings are keyed by sample index so that each hotkey vector aligns with its future price move. Training data is streamed via generator iteration, not loaded into memory.

## Salience and weight setting
Every `WEIGHT_CALC_INTERVAL` blocks the validator streams training data from SQLite one challenge at a time. For each challenge it computes per-hotkey salience using L2 logistic regression and coefficient-based importance. Scores are normalised within each challenge, weighted by challenge weight, averaged across challenges, and turned into on-chain weights. EMA smoothing (alpha=0.15) is applied across weight-setting intervals to reduce block-to-block variance. UIDs that have only recently produced their first embedding receive a small fixed allocation before salience is applied.

## Security considerations
- **Time-lock encryption** (Drand IBE + X25519 owner wrap) prevents late submissions.
- **Hotkey verification** stops miners from spoofing another miner's identity.
- **Binding hash** (SHA-256 over hotkey, round, owner_pk, ephemeral_pk) prevents replay, relay, and substitution attacks.
- **Input validation** clamps embeddings to the expected shape and value range; malformed inputs become zero vectors.
- **L2 regularization** splits coefficient mass among correlated miners, making sybil cloning unprofitable.
