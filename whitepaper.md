# MANTIS: a challenge-based signal market

MANTIS runs on Bittensor netuid 123 and incentivises predictions of one‑hour price moves for several assets such as BTC, ETH and major forex pairs. Miners submit one embedding per challenge; validators decrypt the payloads after a 300‑block delay and measure each hotkey's contribution to forecasting returns.

## Submission and commit–reveal
- Each block miners upload a timelocked payload to Cloudflare R2 and commit the URL on‑chain. The payload contains a list of embeddings ordered to match `config.CHALLENGES` and the miner's hotkey.
- Validators download the ciphertext and record the current prices for every ticker.
- When the time lock matures (300 blocks), the validator retrieves a Drand signature, decrypts the payload and checks that the embedded hotkey matches the commit.

## DataLog layout
The validator stores all state in a `DataLog` object:
```python
class ChallengeData:
    dim: int
    blocks_ahead: int
    sidx: Dict[int, Dict[str, Any]]  # sample index -> {price, emb{hotkey: vec}}

class DataLog:
    blocks: List[int]
    challenges: Dict[str, ChallengeData]
    raw_payloads: Dict[int, Dict[str, bytes]]
```
Prices and embeddings are keyed by sample index so that each hotkey vector aligns with its future price move.

## Salience and weight setting
Every `TASK_INTERVAL` blocks the validator builds a training set from the `DataLog`. For each challenge it trains an XGBoost model and computes permutation importance for every hotkey. Scores are normalised within each challenge, averaged across challenges and turned into on‑chain weights. UIDs that have only recently produced their first embedding receive a small fixed allocation before salience is applied.

## Security considerations
- **Time‑lock encryption** prevents late submissions.
- **Hotkey verification** stops miners from spoofing another miner's identity.
- **Input validation** clamps embeddings to the expected shape and value range; malformed inputs become zero vectors.
- **Inactive hotkey pruning** removes dead participants from historical data.

## Outlook
The challenge list is explicit and can be expanded without altering the core logic. As more history accumulates, alternative models and loss functions can be explored while the salience framework remains unchanged.

