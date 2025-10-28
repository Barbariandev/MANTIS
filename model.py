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
import xgboost as xgb

import config
from lbfgs import compute_lbfgs_salience, compute_q_path_salience


LAST_DEBUG: dict = {}


logger = logging.getLogger(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Salience computations will run on %s", DEVICE)


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


def _xgb_params() -> Dict[str, object]:
    params: Dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 2,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "tree_method": "hist",
        "seed": config.SEED,
        "nthread": os.cpu_count(),
    }
    return params


def _reshape_X_to_hotkey_dim(X: np.ndarray, H: int, D: int) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] != H * D:
        raise ValueError(f"Unexpected X shape {X.shape}, expected (*, {H*D}) for H={H}, D={D}")
    return X.reshape(X.shape[0], H, D)



def salience_binary_prediction(
    hist: Tuple[np.ndarray, Dict[str, int]],
    challenge_returns: np.ndarray,
    ticker: str,
) -> Dict[str, float]:
    """
    Compute salience scores for hotkeys based on their predictive performance.
    
    Args:
        hist: Tuple containing:
            - X_flat (np.ndarray): Shape (T, H*D) where:
                * T = number of time steps (sample points)
                * H = number of hotkeys (miners)
                * D = embedding dimension for this challenge
                * Flattened so each row contains all hotkey embeddings concatenated
            - hk2idx (Dict[str, int]): Maps hotkey string -> column index in the H dimension
                * e.g. {"5CiQ1...": 0, "5FHne...": 1, ...}
                * Used to map hotkeys to their position in the flattened array
        
        challenge_returns (np.ndarray): Shape (T,) containing price returns
            * Aligned with X_flat rows (same T dimension)
            * Each element = (future_price - current_price) / current_price
            * Calculated using blocks_ahead parameter from config
        
        ticker (str): Asset ticker like "BTC", "ETH", "EURUSD", etc.
    
    Returns:
        Dict[str, float]: Hotkey -> normalized salience score (sums to 1.0)
    """
    """data in make windows lag as embargo train logisics 0..fit_end_pred ints
    auc ranks sel_eval_start..sel_eval_end pick top val_start..val_end
    xgb learns 0..fit_end_pred and infers val_start..val_end then permutes for salince"""
    xgb_params = _xgb_params()
    xgb_rounds = 250
    CHUNK_SIZE = 2000
    LAG = int(config.LAG)
    EMBARGO_IDX = LAG
    TOP_K = 20
    WINDOWS_HALF_LIFE = 10
    recency_gamma = float(0.5 ** (1.0 / max(1, WINDOWS_HALF_LIFE)))

    if not isinstance(hist, tuple) or len(hist) != 2:
        logger.warning(f"[{ticker}] Invalid hist format; expected (X, hk2idx)")
        return {}
    X_flat, hk2idx = hist
    if X_flat is None or challenge_returns is None:
        return {}

    spec = config.CHALLENGE_MAP.get(ticker)
    dim = spec["dim"] if spec else None
    if dim is None:
        logger.warning(f"No spec for challenge {ticker}; skipping.")
        return {}
    if not isinstance(hk2idx, dict) or not hk2idx:
        logger.info(f"[{ticker}] No hotkeys present; skipping.")
        return {}

    try:
        X_flat = np.asarray(X_flat, dtype=np.float32)
    except Exception:
        logger.warning(f"[{ticker}] Could not coerce X to float32; skipping.")
        return {}
    y = np.asarray(challenge_returns, dtype=np.float32)
    if X_flat.shape[0] != y.shape[0]:
        logger.warning(f"[{ticker}] X and y length mismatch: {X_flat.shape[0]} vs {y.shape[0]}; skipping.")
        return {}

    T = int(X_flat.shape[0])
    if T < 500:
        logger.info(f"[{ticker}] Not enough samples (T={T}); skipping.")
        return {}

    H = int(X_flat.shape[1] // dim)
    if H <= 0 or H * dim != X_flat.shape[1]:
        logger.warning(f"[{ticker}] Inconsistent shape {X_flat.shape} for dim={dim}")
        return {}

    y_bin = (y > 0).astype(np.float32)
    if len(np.unique(y_bin)) < 2:
        logger.info(f"[{ticker}] y has <2 classes; skipping.")
        return {}

    X = _reshape_X_to_hotkey_dim(X_flat, H, dim)

    first_nz_idx = np.full(H, T, dtype=np.int32)
    for j in range(H):
        row_j = X[:, j, :]
        nz = (row_j != 0).any(axis=1)
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

    pbar = tqdm(total=len(indices), desc=f"SAL Walk-fwd {ticker}")

    total_hk_imp = np.zeros(H, dtype=np.float32)
    total_weight = 0.0
    window_index = 0

    idx2hk = [None] * H
    try:
        for hk, idx in hk2idx.items():
            if 0 <= idx < H:
                idx2hk[idx] = hk
    except Exception:
        idx2hk = [str(i) for i in range(H)]

    for (train_start, val_start, val_end) in indices:
        train_end = val_start
        y_train_all = y_bin[:train_end]
        y_val = y_bin[val_start:val_end]

        if (
            train_end < 200
            or len(np.unique(y_train_all)) < 2
            or len(np.unique(y_val)) < 2
        ):
            pbar.update(1)
            continue

        sel_eval_end = train_end
        sel_eval_start = max(0, sel_eval_end - CHUNK_SIZE)
        sel_fit_end = max(0, sel_eval_start - EMBARGO_IDX)
        if sel_fit_end < 50:
            pbar.update(1)
            continue

        sel_auc = np.zeros(H, dtype=np.float32)
        for j in range(H):
            if first_nz_idx[j] >= sel_fit_end:
                sel_auc[j] = 0.5
                continue
            Xi_fit = X[:sel_fit_end, j, :].astype(np.float32, copy=False)
            yi_fit = y_bin[:sel_fit_end]
            if len(np.unique(yi_fit)) < 2:
                sel_auc[j] = 0.5
                continue
            try:
                clf = LogisticRegression(
                    penalty="l2",
                    C=0.5,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=200,
                    random_state=config.SEED,
                )
                clf.fit(Xi_fit, yi_fit)
                Xi_eval = X[sel_eval_start:sel_eval_end, j, :].astype(np.float32, copy=False)
                yi_eval = y_bin[sel_eval_start:sel_eval_end]
                if len(np.unique(yi_eval)) < 2 or Xi_eval.shape[0] == 0:
                    sel_auc[j] = 0.5
                else:
                    scores = clf.decision_function(Xi_eval)
                    sel_auc[j] = float(roc_auc_score(yi_eval, scores))
            except Exception:
                sel_auc[j] = 0.5

        top_k = min(TOP_K, H)
        selected_idx = np.argsort(-sel_auc)[:top_k]
        selected_idx.sort()
        if selected_idx.size == 0:
            pbar.update(1)
            continue

        fit_end_pred = max(0, val_start - EMBARGO_IDX)
        if fit_end_pred <= 0:
            pbar.update(1)
            continue

        X_train_sel = np.zeros((fit_end_pred, selected_idx.size), dtype=np.float32)
        X_val_sel = np.zeros((val_end - val_start, selected_idx.size), dtype=np.float32)

        oos_segments: List[Tuple[int, int, int]] = []
        start_oos = 0
        while True:
            val_start_oos = start_oos + LAG
            if val_start_oos >= fit_end_pred:
                break
            end_idx_oos = min(start_oos + CHUNK_SIZE, fit_end_pred)
            if end_idx_oos <= val_start_oos:
                break
            oos_segments.append((start_oos, val_start_oos, end_idx_oos))
            start_oos = end_idx_oos

        for col_idx, j in enumerate(selected_idx):
            if first_nz_idx[j] >= fit_end_pred or fit_end_pred < 50:
                continue
            Xi_all = X[:, j, :].astype(np.float32, copy=False)


            for (oos_train_start, oos_val_start, oos_val_end) in oos_segments:
                tr_fit_end_oos = max(0, oos_val_start - LAG)
                if tr_fit_end_oos < 50:
                    continue
                Xi_fit_oos = Xi_all[:tr_fit_end_oos]
                yi_fit_oos = y_bin[:tr_fit_end_oos]
                if len(np.unique(yi_fit_oos)) < 2:
                    continue
                try:
                    clf_oos = LogisticRegression(
                        penalty="l2",
                        C=0.5,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=200,
                        random_state=config.SEED,
                    )
                    clf_oos.fit(Xi_fit_oos, yi_fit_oos)
                    X_train_sel[oos_val_start:oos_val_end, col_idx] = clf_oos.decision_function(
                        Xi_all[oos_val_start:oos_val_end]
                    )
                except Exception:
                    continue

            try:
                Xi_fit = Xi_all[:fit_end_pred]
                yi_fit = y_bin[:fit_end_pred]
                if len(np.unique(yi_fit)) < 2:
                    continue
                clf_val = LogisticRegression(
                    penalty="l2",
                    C=0.5,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=200,
                    random_state=config.SEED,
                )
                clf_val.fit(Xi_fit, yi_fit)
                X_val_sel[:, col_idx] = clf_val.decision_function(Xi_all[val_start:val_end])
            except Exception:
                continue

        y_train_head = y_bin[:fit_end_pred]

        try:
            dtrain = xgb.DMatrix(X_train_sel, label=y_train_head)
            bst = xgb.train(xgb_params, dtrain, num_boost_round=xgb_rounds, verbose_eval=False)

            dval = xgb.DMatrix(X_val_sel, label=y_val)
            base_probs = bst.predict(dval)
            base_auc = float(roc_auc_score(y_val, base_probs))
        except Exception as e:
            logger.warning(f"[{ticker}] XGBoost training/eval failed: {e}")
            pbar.update(1)
            continue
        finally:
            try:
                del dtrain
                del dval
            except Exception:
                pass

        window_imp = np.zeros(H, dtype=np.float32)
        for local_col, j in enumerate(selected_idx):
            col = X_val_sel[:, local_col].copy()
            perm_idx = np.random.permutation(col.shape[0])
            X_val_sel[:, local_col] = col[perm_idx]
            try:
                dval_perm = xgb.DMatrix(X_val_sel, label=y_val)
                perm_probs = bst.predict(dval_perm)
                perm_auc = float(roc_auc_score(y_val, perm_probs))
                delta = base_auc - perm_auc
                window_imp[j] = delta if delta > 0.0 else 0.0
            except Exception:
                window_imp[j] = 0.0
            finally:
                X_val_sel[:, local_col] = col
                try:
                    del dval_perm
                except Exception:
                    pass

        scale = max((base_auc - 0.5) / 0.5, 0.0)
        if scale <= 0:
            window_imp[:] = 0.0
        else:
            window_imp *= scale

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
        imp_map[hk] = float(max(0.0, score))
    total_imp = float(sum(imp_map.values()))
    return {hk: (score / total_imp) for hk, score in imp_map.items()} if total_imp > 0 else {}


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
