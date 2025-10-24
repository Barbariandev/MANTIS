## LBFGS Mining Guide

This guide focuses on the LBFGS challenge (17-dim per bar). For general mining setup, payload encryption, and commit flow, see `MINER_GUIDE.md`.

- Embedding dim is 17
- Validators aggregate all miners into a matrix and compute two saliences: a p-only 5-bucket classifier salience and a Q-only path salience.
- Within the LBFGS challenge, final per-hotkey score = 50% classifier + 50% Q; then weighted by the challenge weight in `config.py` and normalised across challenges.

Submit 17 probabilities in this exact layout:

- p[0..4] (5 numbers): 5-bucket class probabilities for the endpoint return at horizon H
  - 0: z ≤ -2σ
  - 1: -2σ < z < -1σ
  - 2: -1σ ≤ z ≤ 1σ
  - 3: 1σ < z < 2σ
  - 4: z ≥ 2σ

To provide an example: [0.02, 0.14, 0.68, 0.14, 0.02]

- Q(c) for c ∈ {0,1,3,4} (3 numbers each): per-threshold opposite-move probabilities conditioned on bucket c being the realized class at t+H
  - Thresholds are exactly [0.5σ, 1.0σ, 2.0σ] (σ computed over 60 minutes in system time)
  - c=0 (−2σ bucket): opposite direction is UP
  - c=1 (−1σ bucket): opposite direction is UP
  - c=3 (+1σ bucket): opposite direction is DOWN
  - c=4 (+2σ bucket): opposite direction is DOWN

Index map (D=17):

- [0:5]    => p[0], p[1], p[2], p[3], p[4]
- [5:8]    => Q(c=0) thresholds [0.5σ, 1.0σ, 2.0σ]
- [8:11]   => Q(c=1) thresholds [0.5σ, 1.0σ, 2.0σ]
- [11:14]  => Q(c=3) thresholds [0.5σ, 1.0σ, 2.0σ]
- [14:17]  => Q(c=4) thresholds [0.5σ, 1.0σ, 2.0σ]

Requirements per submission

- p must be a valid probability vector: p[k] ∈ (0,1), Σ p[k] = 1.0 (within 1e-6).
- All Q entries must be in (0,1). They are interpreted as probabilities.

### Timing and horizon

- The system decides the horizon H in blocks (e.g., 60 minutes => 300 blocks). We align H to samples as H_steps = round(blocks_ahead / sample_every).
- Opposite-move labels for Q use the price path within [t, t+H]. Baseline is the bar just before t (i.e., t-1).

### How you’re scored

We compute two out-of-sample saliences in a walk-forward loop (daily steps, with an embargo). Your final salience is the average of the two.

1) Classifier salience (p-only)

- We train a global 5-bucket classifier using only miners’ p[0..4]. Experts are pooled via a learned weighting and small param set; no Q features are used here.
- Evaluation uses rolling out-of-sample windows with a time-embargo to prevent leakage and an exponential time-decay (half-life).
- Your contribution c over an eval window is higher when your p’s are informative against the pooled baseline across classes. Contributions are masked before your activation.
- Class-imbalance handling: the classifier uses class-weighted loss across the 5 buckets.
- Your classifier salience is your aggregated contribution across eval windows, normalized across all miners.

2) Q salience (opposite-move, gated)

- Q uses only the Q(c) slices for c ∈ {0,1,3,4}. We gate samples by the realized bucket class at horizon. For negative buckets (0,1), opposite is UP; for positive buckets (3,4), opposite is DOWN.
- For each direction (pos, neg) and thresholds [0.5σ, 1.0σ, 2.0σ], we fit a convex logistic stacking model on the simplex across miners. Training and eval are walk-forward and time-weighted.
- Your Q contribution is the marginal loss increase if your hotkey were removed and the remaining weights renormalized, summed over directions.
- Class-imbalance handling: per-threshold positive/negative weighting in the logistic loss to avoid trivial always-negative solutions.
- Your Q salience is your aggregated marginal contribution across eval windows, normalized across miners.

Final salience

- S_final(hotkey) = 0.5 * S_classifier + 0.5 * S_Q
- We renormalize across all miners to sum to 1.0.

### Data volume and when you start getting credit

- Minimum data: We need roughly 5 days of samples at the current cadence to produce stable salience. Until then the LBFGS challenge will not effect incentives.
- Activation: Your salience only starts accruing after your first non-zero 17-vector. All-zero rows before that are ignored for you.
- Update cadence: Saliences are recomputed every 1000 blocks (walk-forward step = samples_per_day) with a short embargo (≥ max(60 bars, H_steps)).


Constraints

- No NaNs/inf; clamp to (1e-6, 1-1e-6) if needed.

### “Opposite-direction move” definition (for Q labels)

Let H be the horizon in samples, σ_60(t) the rolling 60-minute sigma at time t based on 1-step log returns.

- If the realized class at t+H is positive (c ∈ {3,4}): opposite is DOWN. A threshold hit at level m ∈ {0.5, 1.0, 2.0} occurs if the minimum over [t, t+H] of logP[u]−logP[t−1] ≤ −m·σ_60(t).
- If the realized class at t+H is negative (c ∈ {0,1}): opposite is UP. A threshold hit at level m occurs if the maximum over [t, t+H] of logP[u]−logP[t−1] ≥ m·σ_60(t).

You predict the probability of those opposite-move hits, conditional on c being the realized class.


### Best practices

- Calibrate p to be well-calibrated probabilities; don’t be overconfident.
- Ensure Q predictions meaningfully track opposite excursions; exploit all three thresholds.
- Don’t game for the majority class. We apply class weighting in both tracks.

### Quick self-check snippet (Python)

```python
import numpy as np

def validate_embedding(e):
    e = np.asarray(e, float)
    assert e.shape == (17,)
    p = e[0:5]
    assert np.all(p > 0) and np.all(p < 1)
    assert abs(p.sum() - 1.0) < 1e-6
    q = e[5:]
    assert np.all(q > 0) and np.all(q < 1)
    return True
```

### Validator-side sanitation and edge cases

- Validators clamp LBFGS probabilities to (1e-6, 1-1e-6) and renormalize `p[0..4]` to sum to 1.
- All-zero vectors are treated as “not active yet” for your hotkey until the first non-zero row.
- If there is insufficient history for stable LBFGS evaluation, the LBFGS challenge is skipped (no equal-weight split). Other challenges are averaged without it.

### FAQ

- What if I only have p? You can submit valid p and set all Q to 0.5 until you have Q; Q-only salience will be low until then.
- What if I submit all zeros? You won’t be scored until your first non-zero embedding (activation).



