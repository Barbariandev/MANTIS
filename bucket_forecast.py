from __future__ import annotations

import logging
import weakref
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from cluster import equalize_within_cluster
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
# Walk-forward segment length in samples (~8.33 days at 60s cadence).
_WF_CHUNK = 12_000
_MAX_TRAIN = 3 * _WF_CHUNK
_META_K = 100
# Per-segment recency decay from a 15-day half-life (≈0.681 per segment).
_HALFLIFE_DAYS = 15.0
_DAYS_PER_SEGMENT = (_WF_CHUNK * 60.0) / 86_400.0
_RECENCY_GAMMA = 0.5 ** (_DAYS_PER_SEGMENT / _HALFLIFE_DAYS)

# Harness hook: per-segment (path, idx, seg_ba, chance) tuples when enabled.
_QUALITY_TRACE: bool = False
_QUALITY_LOG: list = []

# Dedup: trailing window, consensus-removed bucket residuals, lag-tolerant
# correlation, average-linkage clusters above tau merged into one series.
_DEDUP_WINDOW = 20_000
_DEDUP_MIN_ROWS = 2_000
_DEDUP_TAU = 0.85
_DEDUP_MAX_LAG = 2
_DEDUP_DIMS = 5
_PERM_SEED = 1234


# Block-bootstrap parameters for the q-path L2 fits (block ≈ 2h at 60s cadence).
_BOOTSTRAP_N = 20
_BOOTSTRAP_BLOCK = 120


def _bootstrap_l2_abs_coef(
    feat_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
    C: float,
    solver: str,
    max_iter: int,
    seed: int,
    n_bootstrap: int = _BOOTSTRAP_N,
    block_size: int = _BOOTSTRAP_BLOCK,
) -> np.ndarray | None:
    """Block-bootstrap a single L2 binary logistic; return aggregated |coef|.

    Returns ``None`` when the timeline is too short for block bootstrap or
    every replicate degenerated; the caller's single-fit fallback path
    then preserves prior behaviour on small inputs.  Aggregation is
    median-of-means across four sub-groups for heavy-tailed robustness.
    """
    T = int(feat_fit.shape[0])
    K = int(feat_fit.shape[1]) if feat_fit.ndim == 2 else 0
    if T < int(block_size) * 2 or K == 0 or int(n_bootstrap) <= 1:
        return None
    if np.unique(y_fit).size < 2:
        return None

    n_blocks = max(1, T // int(block_size))
    coefs: list[np.ndarray] = []
    for b in range(int(n_bootstrap)):
        rng = np.random.default_rng(int(seed) + b)
        block_starts = np.sort(rng.integers(0, T - int(block_size) + 1, size=n_blocks))
        idx_chunks = [np.arange(int(s), int(s) + int(block_size)) for s in block_starts]
        idx = np.concatenate(idx_chunks)
        idx = idx[idx < T]
        if idx.size == 0:
            continue
        Xb = feat_fit[idx]
        yb = y_fit[idx]
        swb = sample_weight[idx] if sample_weight is not None else None
        if np.unique(yb).size < 2:
            continue
        try:
            clf = LogisticRegression(
                penalty="l2",
                C=float(C),
                class_weight="balanced",
                solver=solver,
                max_iter=int(max_iter),
                random_state=int(seed) + b,
            )
            clf.fit(Xb, yb, sample_weight=swb)
            coefs.append(np.abs(clf.coef_.ravel()))
        except Exception:
            continue

    if not coefs:
        return None

    coefs_arr = np.stack(coefs, axis=0)
    if coefs_arr.shape[1] != K:
        return None

    n = int(coefs_arr.shape[0])
    n_groups = min(4, n)
    if n_groups < 2:
        return coefs_arr.mean(axis=0)
    group_size = n // n_groups
    group_means = np.stack([
        coefs_arr[i * group_size:(i + 1) * group_size].mean(axis=0)
        for i in range(n_groups)
    ], axis=0)
    return np.median(group_means, axis=0)


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


def _find_duplicate_clusters(X_3d: np.ndarray, hk2idx: Dict[str, int]) -> list[frozenset[str]]:
    """Average-linkage clusters of miners emitting the same signal.

    Similarity is the smaller of the raw and the consensus-removed signed
    correlation of bucket probabilities on joint-active rows of the
    trailing ``_DEDUP_WINDOW``, each maximised over lags in
    ``[-_DEDUP_MAX_LAG, _DEDUP_MAX_LAG]``.
    """
    T, H, D = X_3d.shape
    bd = min(_DEDUP_DIMS, D)
    Xw = X_3d[-_DEDUP_WINDOW:]
    act = np.any(Xw != 0, axis=2)
    hk_sorted = sorted(hk2idx.keys())
    cand = np.array([hk2idx[hk] for hk in hk_sorted if act[:, hk2idx[hk]].sum() >= _DEDUP_MIN_ROWS], dtype=np.int64)
    if cand.size < 2:
        return []
    W = Xw[:, cand, :bd].astype(np.float32)
    act = act[:, cand]
    Tw, n = act.shape
    actf = act.astype(np.float64)
    cnt = np.maximum(act.sum(axis=0), 1)[:, None]

    def _center(M: np.ndarray) -> np.ndarray:
        M = M * act[:, :, None]
        return (M - (M.sum(axis=0) / cnt)[None, :, :]) * act[:, :, None]

    def _lagged_corr(M: np.ndarray) -> np.ndarray:
        M = _center(M)
        best = np.full((n, n), -1.0)
        for lag in range(-_DEDUP_MAX_LAG, _DEDUP_MAX_LAG + 1):
            if lag >= 0:
                A, B, aA, aB = M[lag:], M[:Tw - lag], actf[lag:], actf[:Tw - lag]
            else:
                A, B, aA, aB = M[:lag], M[-lag:], actf[:lag], actf[-lag:]
            FA = A.transpose(1, 0, 2).reshape(n, -1).astype(np.float64)
            FB = B.transpose(1, 0, 2).reshape(n, -1).astype(np.float64)
            nA = (A ** 2).sum(axis=2).astype(np.float64).T @ aB
            nB = ((B ** 2).sum(axis=2).astype(np.float64).T @ aA).T
            best = np.maximum(best, (FA @ FB.T) / np.maximum(np.sqrt(nA * nB), 1e-12))
        return np.maximum(best, best.T)

    with np.errstate(all="ignore"):
        cons = np.nanmedian(np.where(act[:, :, None], W, np.nan), axis=1)
    best = np.minimum(_lagged_corr(W), _lagged_corr(W - np.nan_to_num(cons)[:, None, :]))
    best[(actf.T @ actf) < _DEDUP_MIN_ROWS] = 0.0
    np.fill_diagonal(best, 1.0)
    dist = 1.0 - np.clip(best, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except Exception:
        logger.exception("scipy unavailable; dedup disabled")
        return []
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=1.0 - _DEDUP_TAU, criterion="distance")
    idx2hk = {v: k for k, v in hk2idx.items()}
    groups: Dict[int, list[str]] = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(int(lbl), []).append(idx2hk[int(cand[i])])
    out = [frozenset(m) for m in groups.values() if len(m) >= 2]
    if out:
        logger.info(
            "lbfgs dedup: H=%d candidates=%d clusters=%d merged_members=%d sizes=%s tau=%.2f",
            H, n, len(out), sum(len(c) for c in out), sorted((len(c) for c in out), reverse=True), _DEDUP_TAU,
        )
    return out


def _median_merge(X_3d: np.ndarray, idx: list[int]) -> np.ndarray:
    """Elementwise median across cluster members, ignoring silent members."""
    T, _, D = X_3d.shape
    out = np.zeros((T, D), dtype=np.float32)
    step = 50_000
    for t0 in range(0, T, step):
        blk = X_3d[t0:t0 + step][:, idx, :].astype(np.float32)
        act = np.any(blk != 0, axis=2)
        blk[~act] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(blk, axis=1)
        out[t0:t0 + blk.shape[0]] = np.nan_to_num(med)
    return out


def compute_linear_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    *,
    blocks_ahead: int,
    sample_every: int,
    max_epochs: int = 80,
    device: str = "cpu",
) -> Dict[str, float]:
    """Walk-forward stacked-ensemble salience on deduplicated signals.

    Input is expected to be dedup'd (``_reduce_hist_to_cohort_reps``), so
    duplicates never reach the fit.  Per end-anchored segment, five one-vs-
    rest L2 logistics are fit on the training window and each signal's
    importance is its out-of-sample permutation importance: the drop in
    the stacked balanced accuracy when that signal's column is shuffled.
    Segment importances are normalised, recency-weighted and summed.
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
    f = N
    while f > warmup:
        vs = max(warmup, f - _WF_CHUNK)
        if f - vs >= 200:
            segments.insert(0, (vs, f))
        f = vs

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

        indiv_ba_train = _vectorized_balanced_accuracy(bp_argmax[ts:vs, active], y_fit, K)
        meta_k = min(_META_K, n_active)
        selected = np.argsort(-indiv_ba_train, kind='stable')[:meta_k]
        sel_miners = active[selected]

        sw = recent_mass_weights(
            np.arange(tlen, dtype=float),
            recent_samples=RECENT_SAMPLES,
            recent_mass=RECENT_MASS,
        )

        feat_fit = bp[ts:vs, sel_miners, :]
        feat_val = bp[vs:ve, sel_miners, :]
        scores = np.full((ve - vs, K), -np.inf)
        coefs = np.zeros((K, meta_k))
        for c in range(K):
            y_fit_c = (y_fit == c).astype(int)
            if np.unique(y_fit_c).size < 2:
                continue
            clf = LogisticRegression(
                penalty="l2",
                C=0.1,
                class_weight="balanced",
                solver="liblinear",
                max_iter=100,
                random_state=42,
            )
            clf.fit(feat_fit[:, :, c], y_fit_c, sample_weight=sw)
            coefs[c] = clf.coef_.ravel()
            scores[:, c] = clf.decision_function(feat_val[:, :, c])

        seg_ba = _balanced_accuracy(y_val, scores.argmax(axis=1), K)
        if _QUALITY_TRACE:
            _QUALITY_LOG.append(("linear", int(si), float(seg_ba), float(random_bal)))
        if seg_ba <= random_bal:
            continue

        # Permutation importance via the linear decision function.
        perm = np.random.default_rng(_PERM_SEED + si).permutation(ve - vs)
        seg_imp = np.zeros(meta_k)
        for j in range(meta_k):
            delta = feat_val[perm, j, :] - feat_val[:, j, :]
            sc = scores + delta * coefs[:, j][None, :]
            seg_imp[j] = max(seg_ba - _balanced_accuracy(y_val, sc.argmax(axis=1), K), 0.0)

        if seg_imp.sum() <= 0:
            continue

        w = _RECENCY_GAMMA ** (len(segments) - 1 - si)
        imp_seg = np.zeros(n_active)
        imp_seg[selected] = seg_imp / seg_imp.sum()
        total_imp += imp_seg * w
        total_w += w

    if total_w <= 0:
        return {}

    imp_full = np.zeros(H)
    imp_full[active] = total_imp / total_w
    if imp_full.sum() <= 0:
        return {}

    order = np.argsort(-imp_full, kind='stable')[:TOP_K]
    pruned = np.zeros_like(imp_full)
    pruned[order] = imp_full[order]
    s = pruned.sum()
    if s <= 0:
        return {}
    pruned /= s

    inv_map = {v: k for k, v in hk2idx.items()}
    return {inv_map[i]: float(pruned[i]) for i in range(H) if pruned[i] > 0 and i in inv_map}


def _reduce_hist_to_cohort_reps(
    hist: Tuple[np.ndarray, Dict[str, int]],
) -> Tuple[Tuple[np.ndarray, Dict[str, int]], Dict[str, str]]:
    """Merge duplicate-signal clusters in ``hist`` into one median series each.

    Operates on the hotkey axis only; every real signal reaches the fit
    exactly once.  The merged column is keyed by the cluster's lex-min
    hotkey so ``equalize_within_cluster`` can broadcast mass back to all
    members.  Returns the original hist and an empty map when no clusters
    are detected.
    """
    X_flat, hk2idx = hist
    if not isinstance(hk2idx, dict) or not hk2idx or X_flat is None:
        return hist, {}
    X_arr = np.asarray(X_flat)
    if X_arr.ndim != 2:
        return hist, {}
    H = int(len(hk2idx))
    if H < 2 or X_arr.shape[1] % H != 0:
        return hist, {}
    D = int(X_arr.shape[1] // H)
    if D <= 0:
        return hist, {}

    T = int(X_arr.shape[0])
    X_3d = X_arr.reshape(T, H, D)
    clusters = _find_duplicate_clusters(X_3d, hk2idx)
    if not clusters:
        return hist, {}

    full_to_rep: Dict[str, str] = {hk: hk for hk in hk2idx.keys()}
    for cl in clusters:
        rep = min(cl)
        for hk in cl:
            full_to_rep[hk] = rep
    rep_hks = sorted(set(full_to_rep.values()))
    members: Dict[str, list[str]] = {}
    for hk, rep in full_to_rep.items():
        members.setdefault(rep, []).append(hk)

    X_red = np.empty((T, len(rep_hks), D), dtype=np.float32)
    for i, rep in enumerate(rep_hks):
        idx = sorted(int(hk2idx[m]) for m in members[rep])
        if len(idx) == 1:
            X_red[:, i, :] = X_3d[:, idx[0], :]
        else:
            X_red[:, i, :] = _median_merge(X_3d, idx)
    hk2idx_red: Dict[str, int] = {hk: i for i, hk in enumerate(rep_hks)}

    return (X_red.reshape(T, len(rep_hks) * D), hk2idx_red), full_to_rep


_cohort_cache_ref: Optional[weakref.ref] = None
_cohort_cache_val: object = None


def _cached_reduce(hist: Tuple[np.ndarray, Dict[str, int]]):
    global _cohort_cache_ref, _cohort_cache_val
    arr = hist[0]
    if _cohort_cache_ref is not None and _cohort_cache_ref() is arr and _cohort_cache_val is not None:
        return _cohort_cache_val
    result = _reduce_hist_to_cohort_reps(hist)
    _cohort_cache_ref = weakref.ref(arr)
    _cohort_cache_val = result
    return result


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
    reduced_hist, full_to_rep = _cached_reduce(hist)
    s = compute_linear_salience(
        reduced_hist,
        price_data,
        blocks_ahead=blocks_ahead,
        sample_every=sample_every,
    )
    if full_to_rep:
        s = equalize_within_cluster(s, full_to_rep)
    return s


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
    """Q-path scorer on the deduplicated hotkey set; cluster mass is
    equalised across members on the way out."""
    reduced_hist, full_to_rep = _cached_reduce(hist)
    s = _compute_q_path_salience_impl(
        reduced_hist,
        price_data,
        blocks_ahead=blocks_ahead,
        sample_every=sample_every,
        min_days=min_days,
        half_life_days=half_life_days,
        sigma_minutes=sigma_minutes,
        gating_classes=gating_classes,
    )
    if full_to_rep:
        s = equalize_within_cluster(s, full_to_rep)
    return s


def _compute_q_path_salience_impl(
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

            y_bin = (y_hit > 0.5).astype(int)
            agg = _bootstrap_l2_abs_coef(
                x_logits, y_bin,
                sample_weight=sw,
                C=0.5, solver="liblinear", max_iter=500,
                seed=42 + 17 * int(c) + int(j),
                n_bootstrap=_BOOTSTRAP_N, block_size=_BOOTSTRAP_BLOCK,
            )
            if agg is None:
                clf = LogisticRegression(
                    penalty="l2",
                    C=0.5,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=500,
                    random_state=42,
                )
                clf.fit(x_logits, y_bin, sample_weight=sw)
                agg = np.abs(clf.coef_.ravel())
            cs = float(agg.sum())
            if cs > 0:
                per_model_weights.append(agg / cs)

    if not per_model_weights:
        return {}

    w_avg = np.mean(np.stack(per_model_weights, axis=0), axis=0)

    if w_avg.shape[0] > TOP_K:
        order = np.argsort(-w_avg, kind='stable')
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
