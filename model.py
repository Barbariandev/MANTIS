
# MIT License
# Copyright (c) 2024 MANTIS

from __future__ import annotations

import config  # noqa: F401  Importing config first locks BLAS / RNG / hash env vars before numpy loads.

import logging
import os
import random
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cluster import (
    collapse_salience,
    equalize_within_cluster,
    find_clusters,
    find_correlated_cohorts,
    select_representatives,
)
from bucket_forecast import compute_lbfgs_salience, compute_q_path_salience
from hitfirst import compute_hitfirst_salience
from range_breakout import compute_multi_breakout_salience
from xsec_rank import compute_xsec_rank_salience
from funding_xsec import compute_funding_xsec_salience
from trade_mix import compute_trade_mix_salience


def stable_argsort(a, *, axis: int = -1, descending: bool = False) -> np.ndarray:
    """Argsort with stable, hardware-independent tie-breaking.

    `np.argsort` defaults to a quicksort variant whose tie-breaking depends
    on memory layout and array size.  Using `kind='stable'` (mergesort)
    guarantees that two values with identical keys keep their original
    relative order across every BLAS / NumPy build, eliminating a class of
    drift that would otherwise reduce cross-validator vtrust.
    """
    arr = np.asarray(a)
    return np.argsort(-arr if descending else arr, axis=axis, kind="stable")


logger = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Salience computations will run on %s", DEVICE)


MAX_DAYS: int = int(getattr(config, "MAX_DAYS", 30))
INDICES_PER_DAY: int = int(getattr(config, "INDICES_PER_DAY", 1440))
BLOCKS_PER_DAY: int = int(getattr(config, "BLOCKS_PER_DAY", 7200))
MAX_INDEX_HISTORY: int = int(MAX_DAYS * INDICES_PER_DAY)
MAX_BLOCK_HISTORY: int = int(MAX_DAYS * BLOCKS_PER_DAY)
HALFLIFE_DAYS: float = float(getattr(config, "HALFLIFE_DAYS", 15.0))

def set_global_seed(seed: int) -> dict:
    """Pin every runtime RNG and threading source for cross-hardware reproducibility.

    Env-var pinning (BLAS thread counts, PYTHONHASHSEED, CUDA visibility)
    happens at the top of `config.py` so it lands before numpy is loaded.
    This function complements that by:
      * Seeding random / numpy / torch RNGs deterministically.
      * Reaching into already-loaded BLAS libraries via threadpoolctl as a
        runtime hammer — this is what catches the case where a transitive
        import loaded numpy before the env vars took effect.
      * Enabling torch deterministic algorithms and disabling cuDNN
        benchmarking.
    Returns a manifest of what was pinned, suitable for logging.
    """
    manifest: dict = {"seed": int(seed)}
    random.seed(seed)
    np.random.seed(seed)
    manifest["numpy"] = np.__version__

    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=1)
        manifest["threadpoolctl"] = threadpoolctl.__version__
    except ImportError:
        manifest["threadpoolctl"] = "not_installed"
    except Exception as e:
        manifest["threadpoolctl_error"] = str(e)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except (RuntimeError, AttributeError) as e:
        manifest["torch_warning"] = str(e)
    manifest["torch"] = torch.__version__
    manifest["torch_cuda_visible"] = bool(torch.cuda.is_available())

    logger.info(
        "Determinism manifest: numpy=%s torch=%s cuda_visible=%s threadpoolctl=%s seed=%s",
        manifest.get("numpy"), manifest.get("torch"),
        manifest.get("torch_cuda_visible"), manifest.get("threadpoolctl"),
        manifest.get("seed"),
    )
    return manifest


set_global_seed(int(getattr(config, "SEED", 42)))


def _time_weights(T: int, indices_per_day: int = INDICES_PER_DAY,
                   halflife_days: float = HALFLIFE_DAYS) -> np.ndarray:
    lam = np.log(2.0) / (halflife_days * indices_per_day)
    age = np.arange(T, 0, -1, dtype=np.float64)
    w = np.exp(-lam * age)
    w /= w.mean()
    return w


def _reshape_X_to_hotkey_dim(X: np.ndarray, H: int, D: int) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] != H * D:
        raise ValueError(f"Unexpected X shape {X.shape}, expected (*, {H*D}) for H={H}, D={D}")
    return X.reshape(X.shape[0], H, D)




def _nonzero_rows_2d(block: np.ndarray) -> np.ndarray:
    return (block != 0).any(axis=1)


def _build_oos_segments(fit_end_exclusive: int, chunk: int, lag: int) -> List[Tuple[int, int, int]]:
    segments: List[Tuple[int, int, int]] = []
    start = 0
    while True:
        val_start = start + lag
        if val_start >= fit_end_exclusive:
            break
        end = min(start + chunk, fit_end_exclusive)
        if end <= val_start:
            break
        segments.append((start, val_start, end))
        start = end
    return segments


def _fit_base_logistic(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression | None:
    if X_fit.shape[0] < 2 or len(np.unique(y_fit)) < 2:
        return None
    n_features = int(X_fit.shape[1]) if X_fit.ndim == 2 else 0
    solver = "liblinear" if n_features <= 4 else "lbfgs"
    clf = LogisticRegression(
        penalty="l2",
        C=0.5,
        class_weight="balanced",
        solver=solver,
        max_iter=100,
        tol=1e-3,
        random_state=seed,
    )
    clf.fit(X_fit, y_fit, sample_weight=sample_weight)
    return clf


def _fit_meta_logistic_en(
    X_train_sel: np.ndarray,
    y_train_head: np.ndarray,
    seed: int,
    *,
    min_rows: int,
    l1_ratio: float,
    C: float,
    max_iter: int,
    class_weight: str | None,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression | None:
    row_has_any = np.any(~np.isnan(X_train_sel), axis=1)
    if row_has_any.sum() < min_rows:
        return None
    X = np.where(np.isnan(X_train_sel[row_has_any]), 0.0, X_train_sel[row_has_any])
    y = y_train_head[row_has_any]
    sw = sample_weight[row_has_any] if sample_weight is not None else None
    if len(np.unique(y)) < 2:
        return None
    meta = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=float(l1_ratio),
        C=float(C),
        solver="saga",
        class_weight=class_weight,
        max_iter=int(max_iter),
        random_state=seed,
        n_jobs=1,
        tol=1e-4,
        fit_intercept=True,
        warm_start=False,
    )
    meta.fit(X, y, sample_weight=sw)
    return meta


def _fit_meta_logistic_bootstrap(
    X_train_sel: np.ndarray,
    y_train_head: np.ndarray,
    seed: int,
    *,
    min_rows: int,
    l1_ratio: float,
    C: float,
    max_iter: int,
    class_weight: str | None,
    sample_weight: np.ndarray | None = None,
    n_bootstrap: int = 20,
    block_size: int = 120,
) -> np.ndarray | None:
    """Stationary block-bootstrap ENet meta-fit, median-of-means aggregated.

    A single fit on the OOS-prediction matrix is a draw from the
    distribution of fits consistent with the data plus the per-validator
    decryption / metagraph perturbation.  In the flat-loss regime that
    dominates the meta-fit's V-trust contribution, that distribution has
    sample-level variance large enough to swamp the underlying signal,
    so two validators with even small data-state differences produce
    coefficient vectors that diverge in L1 by orders of magnitude more
    than the BLAS-axis numerical floor.

    Block bootstrap re-samples contiguous timeline blocks of length
    ``block_size`` (chosen to exceed the autocorrelation horizon implied
    by ``LAG``) so the local dependency structure of the OOS predictions
    is preserved within each replicate.  Median-of-means aggregation
    across the ``n_bootstrap`` replicates further suppresses heavy-tailed
    influence from any single bad fit.

    Determinism: each replicate's RNG and ``random_state`` are seeded by
    ``seed + b``, and block start positions are sorted before
    concatenation so the sample order is canonical.

    Returns the aggregated absolute-coefficient vector of length K, or
    ``None`` if the data is too short / degenerate for bootstrapping
    (in which case the caller falls back to a single fit).
    """
    row_has_any = np.any(~np.isnan(X_train_sel), axis=1)
    if int(row_has_any.sum()) < int(min_rows):
        return None
    X = np.where(np.isnan(X_train_sel[row_has_any]), 0.0, X_train_sel[row_has_any])
    y = y_train_head[row_has_any]
    sw = sample_weight[row_has_any] if sample_weight is not None else None
    if len(np.unique(y)) < 2:
        return None

    T = int(X.shape[0])
    K = int(X.shape[1])
    if T < int(block_size) * 2 or int(n_bootstrap) <= 1:
        return None

    n_blocks = max(1, T // int(block_size))

    coefs: List[np.ndarray] = []
    for b in range(int(n_bootstrap)):
        rng = np.random.default_rng(int(seed) + b)
        block_starts = np.sort(rng.integers(0, T - int(block_size) + 1, size=n_blocks))
        idx_chunks = [np.arange(int(s), int(s) + int(block_size)) for s in block_starts]
        idx = np.concatenate(idx_chunks)
        idx = idx[idx < T]
        if idx.size == 0:
            continue
        Xb = X[idx]
        yb = y[idx]
        swb = sw[idx] if sw is not None else None
        if len(np.unique(yb)) < 2:
            continue
        try:
            meta = LogisticRegression(
                penalty="elasticnet",
                l1_ratio=float(l1_ratio),
                C=float(C),
                solver="saga",
                class_weight=class_weight,
                max_iter=int(max_iter),
                random_state=int(seed) + b,
                n_jobs=1,
                tol=1e-4,
                fit_intercept=True,
                warm_start=False,
            )
            meta.fit(Xb, yb, sample_weight=swb)
            coefs.append(np.abs(np.asarray(meta.coef_, dtype=np.float64).ravel()))
        except Exception:
            continue

    if not coefs:
        return None

    coefs_arr = np.stack(coefs, axis=0)
    if coefs_arr.shape[1] != K:
        # Defensive: a degenerate replicate would have produced a different
        # shape; falling through to None forces the caller's single-fit path.
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


def salience_binary_prediction(
    hist: Tuple[np.ndarray, Dict[str, int]],
    challenge_returns: np.ndarray,
    ticker: str,
) -> Dict[str, float]:
    LAG = int(getattr(config, "LAG", 1))
    CHUNK_SIZE = int(getattr(config, "CHUNK_SIZE", 4000))
    TOP_K = int(getattr(config, "TOP_K", 50))
    RET_EPS = float(getattr(config, "RET_EPS", 0.0))
    MIN_BASE_TRAIN = int(getattr(config, "MIN_BASE_TRAIN", 50))
    META_L1_RATIO = float(getattr(config, "META_L1_RATIO", 0.5))
    META_C = float(getattr(config, "META_C", 1.0))
    META_MAX_ITER = int(getattr(config, "META_MAX_ITER", 2000))
    META_CLASS_WEIGHT = getattr(config, "META_CLASS_WEIGHT", "balanced")
    SEED = int(getattr(config, "SEED", 0))

    if not isinstance(hist, tuple) or len(hist) != 2:
        return {}
    X_flat, hk2idx = hist
    if X_flat is None or challenge_returns is None:
        return {}

    spec = config.CHALLENGE_MAP.get(ticker)
    if not spec:
        return {}
    dim = spec.get("dim")
    if not isinstance(hk2idx, dict) or not hk2idx or not isinstance(dim, int) or dim <= 0:
        return {}

    X_flat = np.nan_to_num(np.asarray(X_flat, dtype=np.float32), nan=0.0)
    y = np.asarray(challenge_returns, dtype=np.float32)
    if X_flat.shape[0] != y.shape[0]:
        return {}

    if MAX_INDEX_HISTORY > 0 and X_flat.shape[0] > MAX_INDEX_HISTORY:
        X_flat = X_flat[-MAX_INDEX_HISTORY:]
        y = y[-MAX_INDEX_HISTORY:]

    T = int(X_flat.shape[0])
    if T < 500:
        return {}

    H = int(X_flat.shape[1] // dim)
    if H <= 0 or H * dim != X_flat.shape[1]:
        return {}

    y_bin = (y > RET_EPS).astype(np.float32)
    if len(np.unique(y_bin)) < 2:
        return {}

    X = _reshape_X_to_hotkey_dim(X_flat, H, dim)
    tw = _time_weights(T)

    first_nz_idx = np.full(H, T, dtype=np.int32)
    for j in range(H):
        nz_idx = np.flatnonzero(_nonzero_rows_2d(X[:, j, :]))
        if nz_idx.size > 0:
            first_nz_idx[j] = int(nz_idx[0])

    idx2hk = [None] * H
    for hk, idx in hk2idx.items():
        if 0 <= idx < H:
            idx2hk[idx] = hk

    # --- Feature selection: walk-forward daily-window AUC, recency-decayed ---
    # The old scheme (single fit on the first half, single AUC point estimate
    # on the second half) ranked miners by a statistic whose noise floor
    # exceeds the gap between adjacent ranks, so small per-validator data
    # differences reshuffled the top-K.  Instead, refit each miner every
    # SEL_REFIT_DAYS on a trailing window and score strictly out-of-sample
    # 1-day windows, then combine the daily AUCs with an exponential decay
    # (SEL_HALFLIFE_DAYS).  An offline persistence study (ETH, 19 cutoffs,
    # 14d horizon) showed half-lives of 10-20d maximise the rank correlation
    # between this score and miners' FUTURE OOS AUC (rho~0.62 vs 0.60 for a
    # flat mean), while being far more stable across cutoffs than short
    # half-lives.
    sel_split = T // 2
    if sel_split < MIN_BASE_TRAIN:
        return {}

    SEL_WINDOW = int(getattr(config, "SEL_WINDOW_ROWS", INDICES_PER_DAY))
    SEL_REFIT = int(getattr(config, "SEL_REFIT_DAYS", 5)) * INDICES_PER_DAY
    SEL_FIT_ROWS = int(getattr(config, "SEL_FIT_WINDOW_DAYS", 30)) * INDICES_PER_DAY
    SEL_HL_DAYS = float(getattr(config, "SEL_HALFLIFE_DAYS", 15.0))
    SEL_MIN_WINDOW_ROWS = int(getattr(config, "SEL_MIN_WINDOW_ROWS", 80))
    SEL_MIN_VALID_WINDOWS = int(getattr(config, "SEL_MIN_VALID_WINDOWS", 5))

    sel_start = min(SEL_FIT_ROWS // 2, sel_split)
    boundaries = list(range(sel_start, T - LAG, SEL_REFIT))
    if not boundaries:
        return {}
    n_day_windows = (T + SEL_WINDOW - 1) // SEL_WINDOW
    # Window weights: age measured from the end of history, in days.
    w_end = np.minimum((np.arange(n_day_windows) + 1) * SEL_WINDOW, T)
    age_days = (T - w_end).astype(np.float64) / float(INDICES_PER_DAY)
    win_w = np.exp(-np.log(2.0) * age_days / max(SEL_HL_DAYS, 1e-9))

    sel_auc = np.full(H, 0.5, dtype=np.float32)
    for j in range(H):
        if first_nz_idx[j] >= boundaries[-1]:
            continue
        Xi_all = X[:, j, :].astype(np.float32, copy=False)
        nz = _nonzero_rows_2d(Xi_all)
        day_auc = np.full(n_day_windows, np.nan, dtype=np.float64)
        scores_buf = np.full(T, np.nan, dtype=np.float64)
        for b in boundaries:
            fit_lo = max(0, b - SEL_FIT_ROWS)
            fit_mask = nz.copy()
            fit_mask[:fit_lo] = False
            fit_mask[max(0, b - LAG):] = False
            if fit_mask.sum() < MIN_BASE_TRAIN:
                continue
            yf = y_bin[fit_mask]
            if len(np.unique(yf)) < 2:
                continue
            clf = _fit_base_logistic(Xi_all[fit_mask], yf, seed=SEED,
                                     sample_weight=tw[fit_mask])
            if clf is None:
                continue
            sc_hi = min(b + SEL_REFIT, T)
            sc_mask = nz.copy()
            sc_mask[:b] = False
            sc_mask[sc_hi:] = False
            if not sc_mask.any():
                continue
            scores_buf[sc_mask] = clf.decision_function(Xi_all[sc_mask])
        for d in range(n_day_windows):
            lo, hi = d * SEL_WINDOW, min((d + 1) * SEL_WINDOW, T)
            seg = scores_buf[lo:hi]
            ok = ~np.isnan(seg)
            if ok.sum() < SEL_MIN_WINDOW_ROWS:
                continue
            ys = y_bin[lo:hi][ok]
            if len(np.unique(ys)) < 2:
                continue
            day_auc[d] = roc_auc_score(ys, seg[ok])
        valid = ~np.isnan(day_auc)
        if valid.sum() < SEL_MIN_VALID_WINDOWS:
            continue
        wsum = float(win_w[valid].sum())
        if wsum <= 0.0:
            continue
        sel_auc[j] = float(np.dot(win_w[valid], day_auc[valid]) / wsum)

    # Margin-banded top-K selection.  A hard cut at rank K is a cliff:
    # miners straddling the boundary flip in/out under per-validator data
    # skew and take O(1/K) of the challenge mass with them.  Including the
    # band of miners within SEL_AUC_MARGIN of the K-th AUC moves the
    # disagreement onto columns whose meta coefficient is ~0 (the ENet's
    # L1 prunes marginal columns), where membership flips are nearly free.
    top_k = min(TOP_K, H)
    sel_margin = float(getattr(config, "SEL_AUC_MARGIN", 0.005))
    band_extra = int(getattr(config, "SEL_BAND_MAX_EXTRA", 15))
    order = np.argsort(-sel_auc, kind='stable')
    if H > top_k and sel_margin > 0.0:
        thr = float(sel_auc[order[top_k - 1]]) - sel_margin
        n_sel = top_k
        max_sel = min(H, top_k + band_extra)
        while n_sel < max_sel and float(sel_auc[order[n_sel]]) >= thr:
            n_sel += 1
        selected_idx = order[:n_sel]
    else:
        selected_idx = order[:top_k]
    selected_idx = selected_idx[sel_auc[selected_idx] > 0.5]
    if selected_idx.size == 0:
        return {}

    # --- Build OOS base-model predictions via walk-forward for selected miners ---
    K = selected_idx.size
    X_oos = np.full((T, K), np.nan, dtype=np.float32)
    oos_segments = _build_oos_segments(T, CHUNK_SIZE, LAG)

    pbar = tqdm(total=K, desc=f"SAL(ENet) {ticker}")
    for col_idx, j in enumerate(selected_idx):
        if first_nz_idx[j] >= T:
            pbar.update(1)
            continue
        Xi_all = X[:, j, :].astype(np.float32, copy=False)
        for (seg_start, seg_val_start, seg_val_end) in oos_segments:
            fit_end = max(0, seg_val_start - LAG)
            if fit_end < MIN_BASE_TRAIN:
                continue
            Xi_fit = Xi_all[:fit_end]
            yi_fit = y_bin[:fit_end]
            mask_fit = _nonzero_rows_2d(Xi_fit)
            if mask_fit.sum() < MIN_BASE_TRAIN or len(np.unique(yi_fit[mask_fit])) < 2:
                continue
            clf_oos = _fit_base_logistic(Xi_fit[mask_fit], yi_fit[mask_fit], seed=SEED,
                                         sample_weight=tw[:fit_end][mask_fit])
            if clf_oos is None:
                continue
            Xi_slice = Xi_all[seg_val_start:seg_val_end]
            mask_oos = _nonzero_rows_2d(Xi_slice)
            if mask_oos.any():
                X_oos[seg_val_start:seg_val_end, col_idx][mask_oos] = (
                    clf_oos.decision_function(Xi_slice[mask_oos])
                )
        pbar.update(1)
    pbar.close()

    # --- Fit the meta-model on all OOS predictions ---
    # The K selected OOS-prediction columns are highly collinear, so the
    # ENet objective sees a flat coefficient subspace whose minimum is
    # sensitive to small perturbations in the row sample.  A single fit
    # lives at one arbitrary point in that subspace and two validators
    # with slightly different data states land at points that differ by
    # O(1) in normalised L1 — the dominant V-trust contributor on this
    # challenge family.  The block-bootstrap aggregator averages over
    # n_bootstrap replicates on resampled contiguous timeline blocks,
    # collapsing per-fit variance toward the data-independent signal.
    BOOTSTRAP_N = int(getattr(config, "META_BOOTSTRAP_N", 10))
    BOOTSTRAP_BLOCK = int(getattr(config, "META_BOOTSTRAP_BLOCK", max(2 * LAG, 120)))
    # Replicates tolerate a looser solver budget than the single-fit path:
    # per-replicate residual noise is averaged away by the aggregation, so
    # capping saga iterations keeps the 10-replicate ensemble within ~2-4x
    # the single-fit wall time instead of 10x.
    BOOTSTRAP_MAX_ITER = int(getattr(config, "META_BOOTSTRAP_MAX_ITER", 800))

    coef_vec = _fit_meta_logistic_bootstrap(
        X_oos,
        y_train_head=y_bin,
        seed=SEED,
        min_rows=int(getattr(config, "MIN_META_TRAIN_ROWS", 50)),
        l1_ratio=META_L1_RATIO,
        C=META_C,
        max_iter=min(META_MAX_ITER, BOOTSTRAP_MAX_ITER),
        class_weight=META_CLASS_WEIGHT,
        sample_weight=tw,
        n_bootstrap=BOOTSTRAP_N,
        block_size=BOOTSTRAP_BLOCK,
    )
    if coef_vec is None:
        # Bootstrap declined (insufficient timeline length / degenerate
        # data); fall back to the single-fit path so behaviour matches
        # the previous contract on small histories.
        meta_clf = _fit_meta_logistic_en(
            X_oos,
            y_train_head=y_bin,
            seed=SEED,
            min_rows=int(getattr(config, "MIN_META_TRAIN_ROWS", 50)),
            l1_ratio=META_L1_RATIO,
            C=META_C,
            max_iter=META_MAX_ITER,
            class_weight=META_CLASS_WEIGHT,
            sample_weight=tw,
        )
        if meta_clf is None:
            return {}
        coef_vec = np.abs(np.asarray(meta_clf.coef_, dtype=np.float64).ravel())

    imp_map: Dict[str, float] = {}
    for local_col, j in enumerate(selected_idx):
        hk = idx2hk[j] if j < len(idx2hk) and idx2hk[j] is not None else str(j)
        imp_map[hk] = float(coef_vec[local_col])

    total_imp = float(sum(v for _k, v in sorted(imp_map.items())))
    return {hk: (v / total_imp) for hk, v in imp_map.items()} if total_imp > 0 else {}


def multi_salience(
    training_data,
    *,
    return_breakdown: bool = False,
) -> Dict[str, float] | tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute per-hotkey salience across all challenges.

    ``training_data`` is an iterable of ``(ticker, payload)`` pairs (e.g. the
    generator returned by ``DataLog.iter_challenge_training_data``).  Each
    challenge is processed and then its payload is freed before loading the
    next, keeping peak memory proportional to a single challenge at a time.
    """

    def _is_uniform_salience(s: Dict[str, float]) -> bool:
        if not s:
            return True
        vals = np.asarray(list(s.values()), dtype=float)
        total = float(vals.sum())
        if total <= 0.0:
            return True
        mean = total / float(vals.size)
        return False if mean <= 0.0 else ((float(vals.max()) - float(vals.min())) / mean) < 0.01

    def _topk_renorm(s: Dict[str, float], k: int) -> Dict[str, float]:
        if not s:
            return {}
        items = sorted(((hk, float(v)) for hk, v in s.items() if float(v) > 0.0), key=lambda kv: -kv[1])
        if not items:
            return {}
        tau = max(1.0, k / 3.0)
        weighted = [(hk, v * np.exp(-rank / tau)) for rank, (hk, v) in enumerate(items)]
        tot = float(sum(v for _hk, v in weighted))
        if tot <= 0.0:
            return {}
        return {hk: (v / tot) for hk, v in weighted if v / tot > 1e-6}

    def _trim_hist_price(
        hist_in: tuple[np.ndarray, Dict[str, int]], price_in: object
    ) -> tuple[tuple[np.ndarray, Dict[str, int]] | None, np.ndarray | None]:
        X_blk, hk2idx = hist_in
        X_blk = np.asarray(X_blk, dtype=np.float32)
        price_arr = np.asarray(price_in)
        if MAX_BLOCK_HISTORY > 0:
            if X_blk.shape[0] > MAX_BLOCK_HISTORY:
                X_blk = X_blk[-MAX_BLOCK_HISTORY:]
            if price_arr.shape[0] > MAX_BLOCK_HISTORY:
                price_arr = price_arr[-MAX_BLOCK_HISTORY:]
        L = int(min(X_blk.shape[0], price_arr.shape[0]))
        if L <= 0:
            return None, None
        return (X_blk[-L:], hk2idx), price_arr[-L:]

    def _extract_hist_price(payload, spec):
        if not isinstance(payload, dict):
            return None
        hist = payload.get("hist")
        price_raw = payload.get("price")
        blocks_ahead = int(spec.get("blocks_ahead", 0) or 0)
        if not isinstance(hist, tuple) or len(hist) != 2 or price_raw is None or blocks_ahead <= 0:
            return None
        trimmed = _trim_hist_price(hist, price_raw)
        if trimmed[0] is None:
            return None
        return trimmed[0], trimmed[1], blocks_ahead

    # Cap on rows fed to the cohort detectors.  Correlation estimates
    # saturate long before this; the cap bounds the F @ F.T cost for
    # block-level histories (hitfirst sees ~245K rows).  Slicing the most
    # recent rows is deterministic and identically aligned across
    # validators up to their datalog skew.
    _CLUSTER_SCAN_MAX_ROWS = int(getattr(config, "CLUSTER_SCAN_MAX_ROWS", 100_000))

    def _cluster_reps(hist_in, dim_in, ticker_for_log: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Two-tier clone/cohort detection for one challenge.

        Returns ``(exact_reps, cohort_reps)``.

        * ``exact_reps`` are the members of bit-tight clone clusters
          discovered by ``find_clusters``.  These are operator-side
          duplications that should collapse to a single representative;
          the appropriate downstream operation is ``collapse_salience``
          (sum-into-rep) so the cluster's mass concentrates onto one UID.
        * ``cohort_reps`` are the members of soft-correlated cohorts
          discovered by ``find_correlated_cohorts``.  These share a
          common signal but are not byte-identical; the appropriate
          downstream operation is ``equalize_within_cluster`` (mass-
          conserving equal split) so per-key allocation is invariant
          under the L2's nondeterministic mass split inside the cohort.

        Empty maps on any failure or trivial input.
        """
        if hist_in is None or dim_in is None or int(dim_in) <= 0:
            return {}, {}

        X_in, hk2idx_in = hist_in
        if X_in is not None and _CLUSTER_SCAN_MAX_ROWS > 0 and X_in.shape[0] > _CLUSTER_SCAN_MAX_ROWS:
            hist_in = (X_in[-_CLUSTER_SCAN_MAX_ROWS:], hk2idx_in)

        exact_reps: Dict[str, str] = {}
        try:
            clusters = find_clusters(hist_in, dim=int(dim_in))
        except Exception:
            logger.exception("[%s] exact cluster detection failed", ticker_for_log)
            clusters = []
        if clusters:
            exact_reps = select_representatives(clusters)
            logger.info(
                "[%s] %d exact cluster(s) covering %d hotkeys; lex-min reps assigned",
                ticker_for_log, len(clusters), len(exact_reps),
            )

        cohort_reps: Dict[str, str] = {}
        try:
            cohorts = find_correlated_cohorts(hist_in, dim=int(dim_in))
        except Exception:
            logger.exception("[%s] correlated cohort detection failed", ticker_for_log)
            cohorts = []
        if cohorts:
            # Strip exact-clone members so the equalise step only fires on
            # keys that the strict pass left distinct; this preserves the
            # economic property of the strict pass (clones concentrate to
            # one UID) while still removing per-key flicker among the
            # remaining soft-correlated members.
            stripped: List[FrozenSet[str]] = []
            for c in cohorts:
                remaining = frozenset(hk for hk in c if hk not in exact_reps)
                if len(remaining) >= 2:
                    stripped.append(remaining)
            if stripped:
                cohort_reps = select_representatives(stripped)
                logger.info(
                    "[%s] %d soft cohort(s) covering %d hotkeys; equalisation reps assigned",
                    ticker_for_log, len(stripped), len(cohort_reps),
                )

        return exact_reps, cohort_reps

    def _apply_reps(s: Dict[str, float], reps_pair: Tuple[Dict[str, str], Dict[str, str]]) -> Dict[str, float]:
        """Apply exact-clone collapse then soft-cohort equalisation in order."""
        exact_reps, cohort_reps = reps_pair
        if exact_reps:
            s = collapse_salience(s, exact_reps)
        if cohort_reps:
            s = equalize_within_cluster(s, cohort_reps)
        return s

    per_challenge: List[Tuple[str, Dict[str, float], float]] = []
    breakdown: Dict[str, Dict[str, float]] = {}
    total_w = 0.0
    for ticker, payload in training_data:
        spec = config.CHALLENGE_MAP.get(ticker)
        if spec is None:
            del payload
            continue
        loss_type = spec.get("loss_func")
        s: Dict[str, float] = {}
        if loss_type == "binary":
            if not isinstance(payload, tuple) or len(payload) != 2:
                del payload
                continue
            hist, y = payload
            del payload
            s = salience_binary_prediction(hist, y, ticker)
            if s:
                s = _apply_reps(s, _cluster_reps(hist, spec.get("dim"), ticker))
            del hist, y
        elif loss_type in ("lbfgs", "hitfirst"):
            extracted = _extract_hist_price(payload, spec)
            del payload
            if extracted is None:
                continue
            hist, price, blocks_ahead = extracted
            if loss_type == "lbfgs":
                se = int(config.SAMPLE_EVERY)
                s_cls = compute_lbfgs_salience(hist, price, blocks_ahead=blocks_ahead, sample_every=se)
                s_q = compute_q_path_salience(hist, price, blocks_ahead=blocks_ahead, sample_every=se)
                if _is_uniform_salience(s_cls):
                    s_cls = {}
                if _is_uniform_salience(s_q):
                    s_q = {}
                s_cls = _topk_renorm(s_cls, 50)
                s_q = _topk_renorm(s_q, 50)
                keys = sorted(set(s_cls.keys()) | set(s_q.keys()))
                s = {}
                for hk in keys:
                    v = 0.75 * float(s_cls.get(hk, 0.0)) + 0.25 * float(s_q.get(hk, 0.0))
                    if v > 0.0:
                        s[hk] = v
                tot = float(sum(v for _k, v in sorted(s.items())))
                if tot > 0:
                    s = {k: (v / tot) for k, v in s.items()}
            else:
                s = compute_hitfirst_salience(
                    hist, price, blocks_ahead=blocks_ahead, sample_every=int(config.SAMPLE_EVERY),
                )
                if s:
                    # The LBFGS path equalizes cohorts inside bucket_forecast
                    # (_cached_reduce); hitfirst has no internal collapse, so
                    # route it through the same two-tier representative set.
                    s = _apply_reps(s, _cluster_reps(hist, spec.get("dim"), ticker))
            del hist, price
        elif loss_type == "range_breakout_multi":
            completed = payload.get("completed_samples", [])
            del payload
            if not completed:
                continue
            s = compute_multi_breakout_salience(completed)
            del completed
        elif loss_type == "xsec_rank":
            if not isinstance(payload, dict):
                del payload
                continue
            hist = payload.get("hist")
            prices_multi = payload.get("prices_multi")
            blocks_ahead = int(spec.get("blocks_ahead", 0) or 0)
            del payload
            if (
                not isinstance(hist, tuple)
                or len(hist) != 2
                or prices_multi is None
                or blocks_ahead <= 0
            ):
                continue
            hist_trimmed = _trim_hist_price(hist, prices_multi[:, 0])
            if hist_trimmed[0] is None:
                continue
            trim_len = hist_trimmed[0][0].shape[0]
            prices_trimmed = prices_multi[-trim_len:]
            s = compute_xsec_rank_salience(
                (hist_trimmed[0][0], hist_trimmed[0][1]),
                prices_trimmed,
                blocks_ahead=blocks_ahead,
                sample_every=int(config.SAMPLE_EVERY),
            )
            del hist, prices_multi, prices_trimmed
        elif loss_type == "funding_xsec":
            if not isinstance(payload, dict):
                del payload
                continue
            hist = payload.get("hist")
            funding_rates = payload.get("funding_rates")
            sidx_arr = payload.get("sidx_arr")
            blocks_ahead = int(spec.get("blocks_ahead", 0) or 0)
            del payload
            if (
                not isinstance(hist, tuple)
                or len(hist) != 2
                or funding_rates is None
                or blocks_ahead <= 0
            ):
                continue
            hist_trimmed = _trim_hist_price(hist, funding_rates[:, 0])
            if hist_trimmed[0] is None:
                continue
            trim_len = hist_trimmed[0][0].shape[0]
            funding_trimmed = funding_rates[-trim_len:]
            sidx_trimmed = sidx_arr[-trim_len:] if sidx_arr is not None else None
            s = compute_funding_xsec_salience(
                (hist_trimmed[0][0], hist_trimmed[0][1]),
                funding_trimmed,
                blocks_ahead=blocks_ahead,
                sample_every=int(config.SAMPLE_EVERY),
                sidx_arr=sidx_trimmed,
            )
            del hist, funding_rates, funding_trimmed, sidx_trimmed
        elif loss_type == "trade_mix":
            if not isinstance(payload, dict):
                del payload
                continue
            hist = payload.get("hist")
            prices_multi = payload.get("prices_multi")
            sidx_arr = payload.get("sidx_arr")
            blocks_ahead = int(spec.get("blocks_ahead", 0) or 0)
            del payload
            if (
                not isinstance(hist, tuple)
                or len(hist) != 2
                or prices_multi is None
                or blocks_ahead <= 0
            ):
                continue
            hist_trimmed = _trim_hist_price(hist, prices_multi[:, 0])
            if hist_trimmed[0] is None:
                continue
            trim_len = hist_trimmed[0][0].shape[0]
            prices_trimmed = prices_multi[-trim_len:]
            sidx_trimmed = sidx_arr[-trim_len:] if sidx_arr is not None else None
            s = compute_trade_mix_salience(
                (hist_trimmed[0][0], hist_trimmed[0][1]),
                prices_trimmed,
                blocks_ahead=blocks_ahead,
                sample_every=int(config.SAMPLE_EVERY),
                sidx_arr=sidx_trimmed,
                spec=spec,
            )
            del hist, prices_multi, prices_trimmed, sidx_trimmed
        else:
            del payload
            continue
        if s:
            total_challenge_score = float(sum(v for _k, v in sorted(s.items())))
            if total_challenge_score <= 0:
                continue
            s_norm = {hk: v / total_challenge_score for hk, v in s.items()}
            w = float(spec.get("weight", 1.0))
            breakdown[ticker] = dict(s_norm)
            per_challenge.append((ticker, s_norm, w))
            total_w += w
    if not per_challenge or total_w <= 0:
        return ({}, {}) if return_breakdown else {}
    all_hotkeys = sorted(set().union(*(s.keys() for _t, s, _w in per_challenge)))
    avg = {
        hk: float(sum(s.get(hk, 0.0) * w for _t, s, w in per_challenge)) / total_w
        for hk in all_hotkeys
    }
    total = float(sum(v for _k, v in sorted(avg.items())))
    out = {hk: (v / total) for hk, v in avg.items()} if total > 0 else {}
    return (out, breakdown) if return_breakdown else out


