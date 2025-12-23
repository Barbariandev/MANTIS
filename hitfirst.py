from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import EPS, MIN_REQUIRED_SAMPLES, logit, rolling_std_fast

logger = logging.getLogger(__name__)

__all__ = ("compute_hitfirst_salience",)


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
    probs = np.clip(X_valid, EPS, 1.0 - EPS)
    sums = probs.sum(axis=2, keepdims=True)
    probs /= np.where(sums <= 0.0, 1.0, sums)

    up_scores = logit(probs[:, :, 0])
    dn_scores = logit(probs[:, :, 1])
    up_scores = np.nan_to_num(up_scores, nan=0.0, posinf=0.0, neginf=0.0)
    dn_scores = np.nan_to_num(dn_scores, nan=0.0, posinf=0.0, neginf=0.0)

    y_up = (labels[valid_mask] == 0).astype(int)
    y_dn = (labels[valid_mask] == 1).astype(int)

    n_samples = int(up_scores.shape[0])
    split = int(max(1, np.floor(0.7 * n_samples)))

    X_up_train, y_up_train = up_scores[:split], y_up[:split]
    X_up_test = up_scores[split:]

    X_dn_train, y_dn_train = dn_scores[:split], y_dn[:split]
    X_dn_test = dn_scores[split:]

    importance = np.zeros(H, dtype=float)

    if np.unique(y_up_train).size == 2 and X_up_train.shape[0] > 0:
        clf_up = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500,
            tol=1e-4,
        )
        clf_up.fit(X_up_train, y_up_train)
        coef_up = np.asarray(clf_up.coef_, dtype=float).ravel()
        scale_up = np.mean(np.abs(X_up_test), axis=0) if X_up_test.shape[0] > 0 else np.ones(H, dtype=float)
        importance += np.abs(coef_up) * scale_up

    if np.unique(y_dn_train).size == 2 and X_dn_train.shape[0] > 0:
        clf_dn = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500,
            tol=1e-4,
        )
        clf_dn.fit(X_dn_train, y_dn_train)
        coef_dn = np.asarray(clf_dn.coef_, dtype=float).ravel()
        scale_dn = np.mean(np.abs(X_dn_test), axis=0) if X_dn_test.shape[0] > 0 else np.ones(H, dtype=float)
        importance += np.abs(coef_dn) * scale_dn

    total = float(np.sum(importance))
    if total <= 0.0:
        return {}

    out = {}
    for hk, idx in hk2idx.items():
        i = int(idx)
        if 0 <= i < H:
            out[hk] = float(importance[i] / total)
    return out



