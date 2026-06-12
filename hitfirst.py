from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import EPS, MIN_REQUIRED_SAMPLES, logit, rolling_std_fast

logger = logging.getLogger(__name__)

__all__ = ("compute_hitfirst_salience",)

# Leave-one-block-out replicates for the L2 importance fits.  The single
# fit over ~340 collinear miner columns sits in a flat loss subspace whose
# minimum moves O(1) in normalised L1 under small row perturbations (the
# per-validator datalog skew).  Averaging |coef| across n deterministic
# leave-block-out replicates estimates a smooth functional of the data
# distribution instead of one arbitrary point in the flat subspace.
# Deterministic (no RNG): block boundaries are proportional to T, so two
# validators with slightly different row counts produce nearly identical
# replicate sets.  Set to 0 or 1 to recover the single-fit behaviour.
HITFIRST_LBO_BLOCKS: int = 4


def _lbo_l2_importance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_blocks: int = HITFIRST_LBO_BLOCKS,
    seed: int = 42,
) -> np.ndarray | None:
    """Leave-one-block-out aggregated |coef| of an L2 logistic fit.

    Fits ``n_blocks`` replicates, each excluding one contiguous timeline
    block (preserving autocorrelation structure within the kept rows),
    and returns the mean absolute coefficient vector.  Falls back to a
    single full fit when ``n_blocks <= 1`` or replicates degenerate.
    """

    def _fit(Xf: np.ndarray, yf: np.ndarray) -> np.ndarray | None:
        if len(np.unique(yf)) < 2:
            return None
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500,
            tol=1e-4,
            random_state=seed,
        )
        clf.fit(Xf, yf)
        return np.abs(np.asarray(clf.coef_, dtype=float).ravel())

    T = int(X.shape[0])
    if int(n_blocks) <= 1 or T < 2 * int(n_blocks):
        return _fit(X, y)

    bounds = np.linspace(0, T, int(n_blocks) + 1).astype(int)
    reps: list[np.ndarray] = []
    for b in range(int(n_blocks)):
        keep = np.ones(T, dtype=bool)
        keep[bounds[b]:bounds[b + 1]] = False
        coef = _fit(X[keep], y[keep])
        if coef is not None:
            reps.append(coef)
    if not reps:
        return _fit(X, y)
    return np.mean(np.stack(reps, axis=0), axis=0)


def compute_hitfirst_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    blocks_ahead: int,
    sample_every: int,
    min_days: float = 5.0,
    half_life_days: float = 5.0,
) -> Dict[str, float]:
    X_flat, hk2idx = hist
    price = np.asarray(price_data, dtype=float)
    if price.ndim != 1 or not isinstance(hk2idx, dict):
        return {}

    H = len(hk2idx)
    if H == 0:
        return {}
    HD = int(X_flat.shape[1])
    if HD % H != 0:
        return {}
    D = HD // H
    if D != 3:
        return {}

    samples_per_day = int((24 * 60 * 60) // (12 * max(1, sample_every)))
    required = int(max(MIN_REQUIRED_SAMPLES, np.ceil(samples_per_day * float(min_days))))
    if price.size < required or X_flat.shape[0] < required:
        logger.info("Hit-first salience requires %d samples; only %d available.", required, price.size)
        return {}

    horizon_steps = max(1, int(round(blocks_ahead / max(1, sample_every))))
    len_r = max(0, int(price.shape[0]) - horizon_steps)
    if len_r <= 0:
        return {}

    r_h = np.log(price[horizon_steps:] + EPS) - np.log(price[:-horizon_steps] + EPS)
    vol_window = int(max(required, MIN_REQUIRED_SAMPLES))
    if r_h.size < vol_window:
        return {}
    sig_raw = rolling_std_fast(r_h, vol_window)
    sig = np.full(len_r, np.nan)
    if sig_raw.size > 0:
        sig[vol_window - 1 :] = sig_raw

    log_price = np.log(price + EPS)
    labels = np.full(len_r, -1, dtype=int)
    for t in range(len_r):
        sigma_t = float(sig[t])
        if not np.isfinite(sigma_t) or sigma_t <= 0.0:
            continue
        base_lp = float(log_price[t])
        seg = log_price[t + 1 : t + 1 + horizon_steps] - base_lp
        idx_up = np.where(seg >= sigma_t)[0]
        idx_dn = np.where(seg <= -sigma_t)[0]
        has_up = idx_up.size > 0
        has_dn = idx_dn.size > 0
        if not has_up and not has_dn:
            labels[t] = 2
            continue
        i_up = int(idx_up[0]) if has_up else horizon_steps + 1
        i_dn = int(idx_dn[0]) if has_dn else horizon_steps + 1
        labels[t] = 0 if i_up < i_dn else 1 if i_dn < i_up else 2

    valid_mask = labels >= 0
    if int(np.sum(valid_mask)) < required:
        return {}

    X_valid = np.asarray(X_flat[:len_r][valid_mask], dtype=float).reshape(-1, H, D)

    submitted = np.any(X_valid != 0.0, axis=2)

    probs = np.clip(X_valid, EPS, 1.0 - EPS)
    sums = probs.sum(axis=2, keepdims=True)
    probs /= np.where(sums <= 0.0, 1.0, sums)

    up_scores = logit(probs[:, :, 0])
    dn_scores = logit(probs[:, :, 1])
    up_scores = np.where(submitted, up_scores, 0.0)
    dn_scores = np.where(submitted, dn_scores, 0.0)
    up_scores = np.nan_to_num(up_scores, nan=0.0, posinf=0.0, neginf=0.0)
    dn_scores = np.nan_to_num(dn_scores, nan=0.0, posinf=0.0, neginf=0.0)

    y_up = (labels[valid_mask] == 0).astype(int)
    y_dn = (labels[valid_mask] == 1).astype(int)

    importance = np.zeros(H, dtype=float)

    if np.unique(y_up).size == 2 and up_scores.shape[0] > 0:
        coef_up = _lbo_l2_importance(up_scores, y_up)
        if coef_up is not None:
            importance += coef_up

    if np.unique(y_dn).size == 2 and dn_scores.shape[0] > 0:
        coef_dn = _lbo_l2_importance(dn_scores, y_dn)
        if coef_dn is not None:
            importance += coef_dn

    total = float(np.sum(importance))
    if total <= 0.0:
        return {}

    out = {}
    for hk, idx in hk2idx.items():
        i = int(idx)
        if 0 <= i < H:
            out[hk] = float(importance[i] / total)
    return out



