
# MIT License
# Copyright (c) 2024 MANTIS

from __future__ import annotations
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
from bucket_forecast import compute_lbfgs_salience, compute_q_path_salience
from hitfirst import compute_hitfirst_salience
from range_breakout import compute_multi_breakout_salience
from xsec_rank import compute_xsec_rank_salience
from funding_xsec import compute_funding_xsec_salience


logger = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Salience computations will run on %s", DEVICE)


MAX_DAYS: int = int(getattr(config, "MAX_DAYS", 30))
INDICES_PER_DAY: int = int(getattr(config, "INDICES_PER_DAY", 1440))
BLOCKS_PER_DAY: int = int(getattr(config, "BLOCKS_PER_DAY", 7200))
MAX_INDEX_HISTORY: int = int(MAX_DAYS * INDICES_PER_DAY)
MAX_BLOCK_HISTORY: int = int(MAX_DAYS * BLOCKS_PER_DAY)
HALFLIFE_DAYS: float = float(getattr(config, "HALFLIFE_DAYS", 15.0))

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logger.info("Torch thread pools pinned to 1 for determinism")
except Exception as e:
    logger.warning("Could not set torch thread counts: %s", e)


def set_global_seed(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic PyTorch algorithms enabled.")
    except (RuntimeError, AttributeError) as e:
        logger.warning(f"Could not enable deterministic algorithms: {e}")
    

set_global_seed(config.SEED)


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

    # --- Feature selection: train on first half, evaluate AUC on second half ---
    sel_split = T // 2
    if sel_split < MIN_BASE_TRAIN:
        return {}

    sel_auc = np.full(H, 0.5, dtype=np.float32)
    for j in range(H):
        if first_nz_idx[j] >= sel_split:
            continue
        Xi_fit = X[:sel_split, j, :].astype(np.float32, copy=False)
        yi_fit = y_bin[:sel_split]
        mask_fit = _nonzero_rows_2d(Xi_fit)
        if mask_fit.sum() < MIN_BASE_TRAIN or len(np.unique(yi_fit[mask_fit])) < 2:
            continue
        clf = _fit_base_logistic(Xi_fit[mask_fit], yi_fit[mask_fit], seed=SEED,
                                 sample_weight=tw[:sel_split][mask_fit])
        if clf is None:
            continue
        Xi_eval = X[sel_split:, j, :].astype(np.float32, copy=False)
        yi_eval = y_bin[sel_split:]
        mask_eval = _nonzero_rows_2d(Xi_eval)
        if mask_eval.sum() < 20 or len(np.unique(yi_eval[mask_eval])) < 2:
            continue
        scores = clf.decision_function(Xi_eval[mask_eval])
        sel_auc[j] = float(roc_auc_score(yi_eval[mask_eval], scores))

    top_k = min(TOP_K, H)
    selected_idx = np.argsort(-sel_auc, kind='stable')[:top_k]
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

    # --- Fit a single meta-model on all OOS predictions ---
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


