from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import (
    EPS,
    MIN_REQUIRED_SAMPLES,
    logit,
    make_bins_from_price,
    recent_mass_weights,
    rolling_std_fast,
)

logger = logging.getLogger(__name__)

__all__ = (
    "compute_lbfgs_salience",
    "compute_q_path_salience",
)

RECENT_SAMPLES = 14_000
RECENT_MASS = 0.5
TOP_K = 25
_WF_CHUNK = 6000
_MAX_TRAIN = 3 * _WF_CHUNK
_META_K = 100
_RECENCY_GAMMA = 0.5 ** 0.1


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    per_c = []
    for c in range(K):
        mask = y_true == c
        if mask.sum() > 0:
            per_c.append(float((y_pred[mask] == c).sum()) / float(mask.sum()))
    return float(np.mean(per_c)) if per_c else 0.0


def _vectorized_balanced_accuracy(preds: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """Balanced accuracy for every miner at once.

    preds: (T, H) int argmax predictions
    y:     (T,) int true labels
    Returns (H,) balanced accuracy per miner.
    """
    mbal = np.zeros(preds.shape[1], dtype=np.float64)
    for c in range(K):
        mask_c = y == c
        nc = mask_c.sum()
        if nc > 0:
            mbal += (preds[mask_c] == c).astype(np.float64).mean(axis=0)
    mbal /= K
    return mbal


def _uniqueness_penalty(preds: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Penalise miners whose argmax predictions heavily overlap with
    higher-ranked miners in *order*.

    For exact sybils (100% overlap), penalty -> 0.
    """
    n = len(order)
    pen = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        mi = int(order[i])
        best_overlap = 0.0
        for j in range(i):
            mj = int(order[j])
            ov = float(np.mean(preds[:, mi] == preds[:, mj]))
            if ov > best_overlap:
                best_overlap = ov
        if best_overlap > 0.85:
            pen[i] = max(0.0, 1.0 - best_overlap)
    return pen


def compute_linear_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    *,
    blocks_ahead: int,
    sample_every: int,
    max_epochs: int = 80,
    device: str = "cpu",
) -> Dict[str, float]:
    """Walk-forward sybil-resistant meta-model salience.

    Combines two signals per walk-forward segment:
      - Individual balanced accuracy lift on OOS data (prediction quality).
      - Per-class binary LogReg coef**2 (sybil resistance via L2 splitting).

    importance_j = individual_lift_j * meta_weight_j

    For n sybil clones:
      - individual_lift is identical per clone.
      - meta_weight shrinks by ~1/n^2 (L2 splits coef, then squared).
      - Group total shrinks by ~1/n. Cloning is unprofitable.

    Post-hoc uniqueness penalty catches remaining overlaps.
    """
    X_flat, hk2idx = hist
    price_arr = np.asarray(price_data, dtype=float)
    if not hk2idx or price_arr.ndim != 1:
        return {}

    required = int(MIN_REQUIRED_SAMPLES)
    if price_arr.size < required or X_flat.shape[0] < required:
        return {}

    H = len(hk2idx)
    if X_flat.ndim != 2:
        return {}
    HD = int(X_flat.shape[1])
    if H <= 0 or HD <= 0 or (HD % H) != 0:
        return {}
    D = HD // H
    if D != 17:
        return {}

    horizon_steps = max(1, int(round(blocks_ahead / max(1, sample_every))))
    vol_window = max(required // 2, 1000)
    y_all, valid_idx = make_bins_from_price(
        price_arr, horizon_steps=horizon_steps, vol_window=vol_window
    )
    if valid_idx.size < required:
        return {}

    X_valid = np.nan_to_num(X_flat[valid_idx], nan=0.0)
    y = y_all
    N = X_valid.shape[0]
    K = 5
    random_bal = 1.0 / K

    X_3d = X_valid.reshape(N, H, D)
    raw5 = X_3d[:, :, :5]
    bp = np.clip(raw5.copy(), 1e-6, None)
    bp /= bp.sum(axis=2, keepdims=True)
    bp_argmax = bp.argmax(axis=2)

    active_frac = np.mean(np.any(raw5 > 0.01, axis=2), axis=0)
    active = np.where(active_frac > 0.05)[0]
    n_active = int(active.size)
    if n_active < 2:
        return {}

    warmup = 2 * _WF_CHUNK
    segments: list[tuple[int, int]] = []
    f = warmup
    while f < N:
        ve = min(f + _WF_CHUNK, N)
        if ve - f < 200:
            break
        segments.append((f, ve))
        f = ve

    if not segments:
        return {}

    total_imp = np.zeros(n_active)
    total_w = 0.0

    for si, (vs, ve) in enumerate(segments):
        y_val = y[vs:ve]
        if np.unique(y_val).size < 2:
            continue

        ts = max(0, vs - _MAX_TRAIN)
        tlen = vs - ts
        y_fit = y[ts:vs]

        preds_val = bp_argmax[vs:ve, active]
        indiv_ba_oos = _vectorized_balanced_accuracy(preds_val, y_val, K)
        indiv_lift = np.maximum(indiv_ba_oos - random_bal, 0.0)

        if indiv_lift.max() <= 0:
            continue

        preds_train = bp_argmax[ts:vs, active]
        indiv_ba_train = _vectorized_balanced_accuracy(preds_train, y_fit, K)
        meta_k = min(_META_K, n_active)
        selected = np.argsort(-indiv_ba_train)[:meta_k]
        sel_miners = active[selected]

        sw = recent_mass_weights(
            np.arange(tlen, dtype=float),
            recent_samples=RECENT_SAMPLES,
            recent_mass=RECENT_MASS,
        )

        meta_imp_sel = np.zeros(meta_k)
        for c in range(K):
            y_fit_c = (y_fit == c).astype(int)
            if np.unique(y_fit_c).size < 2:
                continue
            feat_fit = bp[ts:vs, sel_miners, c]
            clf = LogisticRegression(
                penalty="l2",
                C=0.1,
                class_weight="balanced",
                solver="liblinear",
                max_iter=100,
                random_state=42,
            )
            clf.fit(feat_fit, y_fit_c, sample_weight=sw)
            meta_imp_sel += clf.coef_.ravel() ** 2

        meta_imp = np.zeros(n_active)
        meta_imp[selected] = meta_imp_sel
        max_meta = meta_imp.max()
        if max_meta > 0:
            meta_weight = meta_imp / max_meta
        else:
            meta_weight = np.ones(n_active)

        seg_imp = indiv_lift * meta_weight

        if seg_imp.sum() <= 0:
            continue

        imp_norm = seg_imp / seg_imp.sum()
        vote_scores = np.zeros((ve - vs, K))
        for c in range(K):
            vote_scores[:, c] = ((preds_val == c) * imp_norm[None, :]).sum(axis=1)
        seg_preds = vote_scores.argmax(axis=1)
        seg_ba = _balanced_accuracy(y_val, seg_preds, K)
        if seg_ba <= random_bal:
            continue

        w = _RECENCY_GAMMA ** (len(segments) - 1 - si)
        total_imp += seg_imp * w
        total_w += w

    if total_w <= 0:
        return {}

    imp_full = np.zeros(H)
    imp_full[active] = total_imp / total_w
    if imp_full.sum() <= 0:
        return {}

    preds_tail = bp_argmax[-_WF_CHUNK:]
    nz = np.where(imp_full > 0)[0]
    if nz.size > 1:
        order = nz[np.argsort(-imp_full[nz])]
        pen = _uniqueness_penalty(preds_tail, order)
        for i, mi in enumerate(order):
            imp_full[mi] *= pen[i]

    if imp_full.sum() <= 0:
        return {}

    order = np.argsort(-imp_full)[:TOP_K]
    pruned = np.zeros_like(imp_full)
    pruned[order] = imp_full[order]
    s = pruned.sum()
    if s <= 0:
        return {}
    pruned /= s

    inv_map = {v: k for k, v in hk2idx.items()}
    return {inv_map[i]: float(pruned[i]) for i in range(H) if pruned[i] > 0 and i in inv_map}


def compute_lbfgs_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    blocks_ahead: int,
    sample_every: int,
    lbfgs_cfg: Optional[object] = None,
    min_days: float = 5.0,
    half_life_days: float = 5.0,
    use_class_weights: bool = True,
) -> Dict[str, float]:
    return compute_linear_salience(
        hist,
        price_data,
        blocks_ahead=blocks_ahead,
        sample_every=sample_every,
    )


def compute_q_path_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    blocks_ahead: int,
    sample_every: int,
    min_days: float = 5.0,
    half_life_days: float = 5.0,
    sigma_minutes: int = 60,
    gating_classes: Iterable[int] = (0, 1, 3, 4),
) -> Dict[str, float]:
    """
    Q-path salience: 12 independent balanced binary LogReg models
      c in {0,1,3,4} x threshold in {0.5sig, 1.0sig, 2.0sig}

    Each model uses miner quantile logits as features to predict threshold hits.
    Salience is derived from absolute coefficient magnitudes, averaged across
    all 12 sub-models and top-K pruned.
    """
    X_flat, hk2idx = hist
    price = np.asarray(price_data, dtype=float)
    if price.ndim != 1:
        return {}
    if not isinstance(hk2idx, dict) or not hk2idx:
        return {}

    required = int(MIN_REQUIRED_SAMPLES)
    if price.size < required or X_flat.shape[0] < required:
        return {}

    H = int(len(hk2idx))
    if X_flat.ndim != 2:
        return {}
    HD = int(X_flat.shape[1])
    if H <= 0 or HD <= 0 or (HD % H) != 0:
        return {}
    D = int(HD // H)
    if D != 17:
        return {}

    horizon_steps = max(1, int(round(blocks_ahead / max(1, sample_every))))
    T = int(price.shape[0])
    len_r = T - int(horizon_steps)
    if len_r <= 1:
        return {}

    vol_window = max(required // 2, 1000)
    y_all, valid_idx = make_bins_from_price(price, horizon_steps=horizon_steps, vol_window=vol_window)
    y_r = np.full(len_r, -1, dtype=int)
    if valid_idx.size > 0:
        y_r[valid_idx] = y_all

    r_h = np.log(price[horizon_steps:] + EPS) - np.log(price[:-horizon_steps] + EPS)
    vol_window_q = int(max(MIN_REQUIRED_SAMPLES, 10))
    sig_raw = rolling_std_fast(r_h, vol_window_q)
    sigma_h = np.full(len_r, np.nan)
    if sig_raw.size > 0:
        sigma_h[vol_window_q - 1 :] = sig_raw

    logp = np.log(price + EPS)
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(logp, int(horizon_steps) + 1)
    max_lp = win.max(axis=1)
    min_lp = win.min(axis=1)

    t0 = np.arange(1, len_r, dtype=int)
    base_lp = logp[t0 - 1]
    up = max_lp[t0] - base_lp
    dn = min_lp[t0] - base_lp
    sig = sigma_h[t0]
    y_bucket = y_r[t0]

    valid_common = np.isfinite(sig) & (sig > 0.0) & (y_bucket >= 0)
    if not np.any(valid_common):
        return {}

    thr_mult = np.array([0.5, 1.0, 2.0], dtype=float)
    hit_up = up[:, None] >= (thr_mult[None, :] * sig[:, None])
    hit_dn = dn[:, None] <= -(thr_mult[None, :] * sig[:, None])

    Q_START = {0: 5, 1: 8, 3: 11, 4: 14}

    Xr = np.nan_to_num(np.asarray(X_flat[:len_r], dtype=float), nan=0.0).reshape(len_r, H, 17)

    per_model_weights: list[np.ndarray] = []

    for c in (0, 1, 3, 4):
        start = Q_START.get(int(c))
        if start is None:
            continue
        mask_c = valid_common & (y_bucket == int(c))
        if not np.any(mask_c):
            continue
        t_sel = t0[mask_c]

        sw = recent_mass_weights(t_sel.astype(float), recent_samples=RECENT_SAMPLES, recent_mass=RECENT_MASS)

        hits = hit_dn[mask_c] if int(c) in (3, 4) else hit_up[mask_c]

        for j in range(3):
            y_hit = hits[:, j].astype(np.float32)
            if y_hit.size < 100 or np.unique(y_hit).size < 2:
                continue

            q_raw = Xr[t_sel, :, start + j]
            q = np.asarray(q_raw, dtype=float)
            q[~np.isfinite(q)] = 0.5
            q[q == 0.0] = 0.5
            q = np.clip(q, EPS, 1.0 - EPS)
            x_logits = logit(q)

            clf = LogisticRegression(
                penalty="l2",
                C=0.5,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=500,
                random_state=42,
            )
            clf.fit(x_logits, (y_hit > 0.5).astype(int), sample_weight=sw)
            coef = np.abs(clf.coef_.ravel())
            cs = coef.sum()
            if cs > 0:
                per_model_weights.append(coef / cs)

    if not per_model_weights:
        return {}

    w_avg = np.mean(np.stack(per_model_weights, axis=0), axis=0)

    if w_avg.shape[0] > TOP_K:
        order = np.argsort(-w_avg)
        keep = order[:TOP_K]
        kept = w_avg[keep]
        s = float(np.sum(kept))
        pruned = np.zeros_like(w_avg)
        if s > 0.0:
            pruned[keep] = kept / s
        w_avg = pruned

    inv_map = {idx: hk for hk, idx in hk2idx.items()}
    sal: Dict[str, float] = {}
    for i in range(H):
        hk = inv_map.get(i)
        if hk is not None and float(w_avg[i]) > 0.0:
            sal[hk] = float(w_avg[i])
    return sal
