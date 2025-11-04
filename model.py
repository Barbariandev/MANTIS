
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
from lbfgs import compute_lbfgs_salience, compute_q_path_salience


LAST_DEBUG: dict = {}


logger = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Salience computations will run on %s", DEVICE)


MAX_DAYS: int = int(getattr(config, "MAX_DAYS", 30))
INDICES_PER_DAY: int = int(getattr(config, "INDICES_PER_DAY", 1440))
BLOCKS_PER_DAY: int = int(getattr(config, "BLOCKS_PER_DAY", 7200))
MAX_INDEX_HISTORY: int = int(MAX_DAYS * INDICES_PER_DAY)
MAX_BLOCK_HISTORY: int = int(MAX_DAYS * BLOCKS_PER_DAY)

try:
    _NUM_CPU = max(1, os.cpu_count() or 1)
    torch.set_num_threads(_NUM_CPU)
    torch.set_num_interop_threads(_NUM_CPU)
    logger.info("Torch thread pools set to %d", _NUM_CPU)
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
) -> LogisticRegression | None:
    if X_fit.shape[0] < 2 or len(np.unique(y_fit)) < 2:
        return None
    clf = LogisticRegression(
        penalty="l2",
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=200,
        random_state=seed,
    )
    clf.fit(X_fit, y_fit)
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
) -> LogisticRegression | None:
    row_has_any = np.any(~np.isnan(X_train_sel), axis=1)
    if row_has_any.sum() < min_rows:
        return None
    X = np.where(np.isnan(X_train_sel[row_has_any]), 0.0, X_train_sel[row_has_any])
    y = y_train_head[row_has_any]
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
        n_jobs=os.cpu_count(),
        tol=1e-4,
        fit_intercept=True,
        warm_start=False,
    )
    meta.fit(X, y)
    return meta



def salience_binary_prediction(
    hist: Tuple[np.ndarray, Dict[str, int]],
    challenge_returns: np.ndarray,
    ticker: str,
) -> Dict[str, float]:
    LAG = int(getattr(config, "LAG", 1))
    CHUNK_SIZE = int(getattr(config, "CHUNK_SIZE", 6000))
    TOP_K = int(getattr(config, "TOP_K", 20))
    WINDOWS_HALF_LIFE = int(getattr(config, "WINDOWS_HALF_LIFE", 10))
    recency_gamma = float(0.5 ** (1.0 / max(1, WINDOWS_HALF_LIFE)))
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

    X_flat = np.asarray(X_flat, dtype=np.float32)
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

    first_nz_idx = np.full(H, T, dtype=np.int32)
    for j in range(H):
        row_j = X[:, j, :]
        nz = _nonzero_rows_2d(row_j)
        nz_idx = np.flatnonzero(nz)
        if nz_idx.size > 0:
            first_nz_idx[j] = int(nz_idx[0])

    indices: List[Tuple[int, int, int]] = []
    start = 0
    while True:
        val_start_idx = start + LAG
        if val_start_idx >= T:
            break
        end_idx = min(start + CHUNK_SIZE, T)
        if end_idx <= start:
            break
        indices.append((start, val_start_idx, end_idx))
        start = end_idx

    if not indices:
        return {}

    pbar = tqdm(total=len(indices), desc=f"SAL(ENet) Walk-fwd {ticker}")

    total_hk_imp = np.zeros(H, dtype=np.float32)
    total_weight = 0.0
    window_index = 0

    idx2hk = [None] * H
    for hk, idx in hk2idx.items():
        if 0 <= idx < H:
            idx2hk[idx] = hk

    for (train_start, val_start, val_end) in indices:
        train_end = val_start
        y_val = y_bin[val_start:val_end]
        if len(np.unique(y_val)) < 2:
            pbar.update(1)
            continue

        sel_eval_end = train_end
        sel_eval_start = max(0, sel_eval_end - CHUNK_SIZE)
        sel_fit_end = max(0, sel_eval_start - LAG)
        if sel_fit_end < MIN_BASE_TRAIN:
            pbar.update(1)
            continue

        sel_auc = np.zeros(H, dtype=np.float32)
        for j in range(H):
            if first_nz_idx[j] >= sel_fit_end:
                sel_auc[j] = 0.5
                continue
            Xi_fit_full = X[:sel_fit_end, j, :].astype(np.float32, copy=False)
            yi_fit_full = y_bin[:sel_fit_end]
            mask_fit = _nonzero_rows_2d(Xi_fit_full)
            if mask_fit.sum() < MIN_BASE_TRAIN or len(np.unique(yi_fit_full[mask_fit])) < 2:
                sel_auc[j] = 0.5
                continue
            clf = _fit_base_logistic(Xi_fit_full[mask_fit], yi_fit_full[mask_fit], seed=SEED)
            if clf is None:
                sel_auc[j] = 0.5
                continue
            Xi_eval_full = X[sel_eval_start:sel_eval_end, j, :].astype(np.float32, copy=False)
            yi_eval_full = y_bin[sel_eval_start:sel_eval_end]
            mask_eval = _nonzero_rows_2d(Xi_eval_full)
            if mask_eval.sum() == 0 or len(np.unique(yi_eval_full[mask_eval])) < 2:
                sel_auc[j] = 0.5
            else:
                scores = clf.decision_function(Xi_eval_full[mask_eval])
                sel_auc[j] = float(roc_auc_score(yi_eval_full[mask_eval], scores))

        top_k = min(TOP_K, H)
        if top_k <= 0:
            pbar.update(1)
            continue
        selected_idx = np.argsort(-sel_auc)[:top_k]
        if selected_idx.size == 0:
            pbar.update(1)
            continue

        fit_end_pred = max(0, val_start - LAG)
        if fit_end_pred <= 0:
            pbar.update(1)
            continue

        K = selected_idx.size
        X_train_sel = np.full((fit_end_pred, K), np.nan, dtype=np.float32)
        X_val_sel = np.full((val_end - val_start, K), np.nan, dtype=np.float32)

        oos_segments = _build_oos_segments(fit_end_pred, CHUNK_SIZE, LAG)

        for col_idx, j in enumerate(selected_idx):
            if first_nz_idx[j] >= fit_end_pred or fit_end_pred < MIN_BASE_TRAIN:
                continue
            Xi_all = X[:, j, :].astype(np.float32, copy=False)

            for (oos_train_start, oos_val_start, oos_val_end) in oos_segments:
                tr_fit_end_oos = max(0, oos_val_start - LAG)
                if tr_fit_end_oos < MIN_BASE_TRAIN:
                    continue
                Xi_fit_oos_full = Xi_all[:tr_fit_end_oos]
                yi_fit_oos_full = y_bin[:tr_fit_end_oos]
                mask_fit = _nonzero_rows_2d(Xi_fit_oos_full)
                if mask_fit.sum() < MIN_BASE_TRAIN or len(np.unique(yi_fit_oos_full[mask_fit])) < 2:
                    continue
                clf_oos = _fit_base_logistic(
                    Xi_fit_oos_full[mask_fit],
                    yi_fit_oos_full[mask_fit],
                    seed=SEED,
                )
                if clf_oos is None:
                    continue
                Xi_oos_slice = Xi_all[oos_val_start:oos_val_end]
                mask_oos = _nonzero_rows_2d(Xi_oos_slice)
                if mask_oos.any():
                    X_train_sel[oos_val_start:oos_val_end, col_idx][mask_oos] = clf_oos.decision_function(
                        Xi_oos_slice[mask_oos]
                    )

            Xi_fit_val_full = Xi_all[:fit_end_pred]
            yi_fit_val_full = y_bin[:fit_end_pred]
            mask_fit_val = _nonzero_rows_2d(Xi_fit_val_full)
            if mask_fit_val.sum() < MIN_BASE_TRAIN or len(np.unique(yi_fit_val_full[mask_fit_val])) < 2:
                continue
            clf_val = _fit_base_logistic(
                Xi_fit_val_full[mask_fit_val],
                yi_fit_val_full[mask_fit_val],
                seed=SEED,
            )
            if clf_val is None:
                continue
            Xi_val_slice = Xi_all[val_start:val_end]
            mask_val = _nonzero_rows_2d(Xi_val_slice)
            if mask_val.any():
                X_val_sel[:, col_idx][mask_val] = clf_val.decision_function(Xi_val_slice[mask_val])

        meta_clf = _fit_meta_logistic_en(
            X_train_sel,
            y_train_head=y_bin[:fit_end_pred],
            seed=SEED,
            min_rows=int(getattr(config, "MIN_META_TRAIN_ROWS", 50)),
            l1_ratio=META_L1_RATIO,
            C=META_C,
            max_iter=META_MAX_ITER,
            class_weight=META_CLASS_WEIGHT,
        )
        if meta_clf is None:
            pbar.update(1)
            continue

        X_val_filled = np.where(np.isnan(X_val_sel), 0.0, X_val_sel)
        base_probs = meta_clf.predict_proba(X_val_filled)[:, 1]
        base_auc = float(roc_auc_score(y_val, base_probs))

        window_imp = np.zeros(H, dtype=np.float32)
        X_val_perm = X_val_filled.copy()
        for local_col, j in enumerate(selected_idx):
            col_vals = X_val_sel[:, local_col]
            mask = ~np.isnan(col_vals)
            nn = int(mask.sum())
            if nn <= 1:
                window_imp[j] = 0.0
                continue
            saved_vals = X_val_perm[mask, local_col].copy()
            perm = np.random.default_rng(SEED).permutation(nn)
            X_val_perm[mask, local_col] = saved_vals[perm]
            perm_probs = meta_clf.predict_proba(X_val_perm)[:, 1]
            perm_auc = float(roc_auc_score(y_val, perm_probs))
            delta = base_auc - perm_auc
            window_imp[j] = delta if delta > 0.0 else 0.0
            X_val_perm[mask, local_col] = saved_vals

        scale = max((base_auc - 0.5) / 0.5, 0.0)
        if scale > 0:
            window_imp *= scale
        else:
            window_imp[:] = 0.0

        w = recency_gamma ** (max(0, len(indices) - 1 - window_index))
        total_hk_imp += (w * window_imp).astype(np.float32)
        total_weight += w
        window_index += 1

        pbar.update(1)

    pbar.close()

    if total_weight <= 0:
        return {}

    norm_imp = (total_hk_imp / total_weight).tolist()
    imp_map: Dict[str, float] = {}
    for j, score in enumerate(norm_imp):
        hk = idx2hk[j] if j < len(idx2hk) and idx2hk[j] is not None else str(j)
        val = float(score)
        imp_map[hk] = val if val > 0.0 else 0.0
    total_imp = float(sum(imp_map.values()))
    return {hk: (v / total_imp) for hk, v in imp_map.items()} if total_imp > 0 else {}


def multi_salience(
    training_data: Dict[str, Tuple[Tuple[np.ndarray, Dict[str, int]], np.ndarray]]
) -> Dict[str, float]:
    def _is_uniform_salience(s: Dict[str, float]) -> bool:
        if not s:
            return True
        vals = list(s.values())
        if not vals:
            return True
        total = float(sum(vals))
        if total <= 0.0:
            return True
        v0 = vals[0]
        return all(abs(v - v0) <= 1e-12 for v in vals)
    per_challenge: List[Tuple[Dict[str, float], float]] = []
    total_w = 0.0
    for ticker, payload in training_data.items():
        spec = config.CHALLENGE_MAP.get(ticker)
        if not spec:
            continue
        loss_type = spec.get("loss_func")
        s: Dict[str, float] = {}
        if loss_type == "binary":
            if not isinstance(payload, tuple) or len(payload) != 2:
                continue
            hist, y = payload
            s = salience_binary_prediction(hist, y, ticker)
        elif loss_type == "lbfgs":
            if not isinstance(payload, dict):
                continue
            hist = payload.get("hist")
            price = payload.get("price")
            blocks_ahead = int(payload.get("blocks_ahead", spec.get("blocks_ahead", 0) or 0))
            if (
                not isinstance(hist, tuple)
                or len(hist) != 2
                or price is None
                or blocks_ahead <= 0
            ):
                continue
            X_blk, hk2idx = hist
            X_blk = np.asarray(X_blk, dtype=np.float32)
            price_arr = np.asarray(price)
            if MAX_BLOCK_HISTORY > 0:
                if X_blk.shape[0] > MAX_BLOCK_HISTORY:
                    X_blk = X_blk[-MAX_BLOCK_HISTORY:]
                if price_arr.shape[0] > MAX_BLOCK_HISTORY:
                    price_arr = price_arr[-MAX_BLOCK_HISTORY:]
                L = int(min(X_blk.shape[0], price_arr.shape[0]))
                if L <= 0:
                    continue
                X_blk = X_blk[-L:]
                price_arr = price_arr[-L:]
            hist = (X_blk, hk2idx)
            price = price_arr
            try:
                s_cls = compute_lbfgs_salience(
                    hist,
                    price,
                    blocks_ahead=blocks_ahead,
                    sample_every=int(config.SAMPLE_EVERY),
                )
            except Exception:
                s_cls = {}
            try:
                s_q = compute_q_path_salience(
                    hist,
                    price,
                    blocks_ahead=blocks_ahead,
                    sample_every=int(config.SAMPLE_EVERY),
                )
            except Exception:
                s_q = {}
            if _is_uniform_salience(s_cls):
                s_cls = {}
            if _is_uniform_salience(s_q):
                s_q = {}
            keys = set(s_cls.keys()) | set(s_q.keys())
            s = {}
            for hk in keys:
                v = 0.5 * float(s_cls.get(hk, 0.0)) + 0.5 * float(s_q.get(hk, 0.0))
                if v > 0.0:
                    s[hk] = v
            tot = float(sum(s.values()))
            if tot > 0:
                s = {k: (v / tot) for k, v in s.items()}
        else:
            continue
        if s:
            total_challenge_score = float(sum(s.values()))
            if total_challenge_score <= 0:
                continue
            w = float(spec.get("weight", 1.0))
            per_challenge.append((s, w))
            total_w += w
    if not per_challenge or total_w <= 0:
        return {}
    all_hotkeys = set().union(*(s.keys() for s, _ in per_challenge))
    avg = {
        hk: float(sum(s.get(hk, 0.0) * w for s, w in per_challenge)) / total_w
        for hk in all_hotkeys
    }
    total = float(sum(avg.values()))
    return {hk: (v / total) for hk, v in avg.items()} if total > 0 else {}


