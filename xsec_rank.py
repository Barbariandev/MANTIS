from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

SEED = int(getattr(config, "SEED", 42))
N_ASSETS = len(config.BREAKOUT_ASSETS)


def compute_xsec_rank_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    prices_multi: np.ndarray,
    blocks_ahead: int,
    sample_every: int,
) -> Dict[str, float]:
    """Walk-forward salience for the XSEC-RANK challenge.

    Binary reformulation: for each (timestep, asset), label = 1 if the asset's
    forward return exceeds the cross-sectional median.  All assets are pooled
    into a single binary classification problem, giving 33x the sample count of
    a single-asset challenge.  A LogReg-L2 meta-model assigns importance to each
    miner based on coefficient magnitude scaled by OOS AUC.
    """
    LAG_T = int(getattr(config, "LAG", 60))
    CHUNK_T = int(getattr(config, "CHUNK_SIZE", 4000))
    TOP_K = int(getattr(config, "TOP_K", 20))
    HALFLIFE = int(getattr(config, "WINDOWS_HALF_LIFE", 10))
    MIN_TRAIN_ROWS = 200

    X_flat, hk2idx = hist
    X_flat = np.asarray(X_flat, dtype=np.float32)
    prices = np.asarray(prices_multi, dtype=np.float64)

    T = X_flat.shape[0]
    H = X_flat.shape[1] // N_ASSETS
    if H <= 0 or H * N_ASSETS != X_flat.shape[1]:
        return {}
    if prices.shape != (T, N_ASSETS):
        return {}

    X = X_flat.reshape(T, H, N_ASSETS)

    ahead = blocks_ahead // sample_every
    if ahead <= 0 or T <= ahead:
        return {}
    T_u = T - ahead
    if T_u < 500:
        return {}

    p0 = prices[:T_u]
    p1 = prices[ahead : ahead + T_u]
    ok = (p0 > 0) & (p1 > 0)
    ret = np.where(ok, p1 / p0 - 1.0, np.nan)

    med = np.nanmedian(ret, axis=1, keepdims=True)
    y_2d = np.where(np.isnan(ret), np.nan, (ret > med).astype(np.float32))

    # Pool across assets: (T_u * N_ASSETS, H)
    # X_pool[t * N_ASSETS + a, j] = miner j's score for asset a at time t
    X_pool = X[:T_u].transpose(0, 2, 1).reshape(T_u * N_ASSETS, H)
    y_pool = y_2d.reshape(T_u * N_ASSETS)
    t_idx = np.repeat(np.arange(T_u, dtype=np.int32), N_ASSETS)

    valid = ~np.isnan(y_pool)
    X_pool = X_pool[valid].copy()
    y_pool = y_pool[valid].copy()
    t_idx = t_idx[valid]

    if len(y_pool) < MIN_TRAIN_ROWS or len(np.unique(y_pool)) < 2:
        return {}

    # Walk-forward temporal segments (expanding window)
    segments = []
    s = 0
    while s < T_u:
        vs = s + LAG_T
        if vs >= T_u:
            break
        e = min(s + CHUNK_T, T_u)
        if e <= vs:
            break
        segments.append((s, vs, e))
        s = e

    if not segments:
        return {}

    recency_gamma = 0.5 ** (1.0 / max(1, HALFLIFE))
    hk_imp = np.zeros(H, dtype=np.float64)
    total_w = 0.0

    pbar = tqdm(total=len(segments), desc="SAL(XSec) Walk-fwd")

    for seg_i, (seg_start, val_start_t, val_end_t) in enumerate(segments):
        # Feature selection window: evaluate on data before val_start_t,
        # using a model fit on data before that evaluation window.
        sel_eval_end_t = val_start_t
        sel_eval_start_t = max(0, sel_eval_end_t - CHUNK_T)
        sel_fit_end_t = max(0, sel_eval_start_t - LAG_T)

        sel_fit_mask = t_idx < sel_fit_end_t
        sel_eval_mask = (t_idx >= sel_eval_start_t) & (t_idx < sel_eval_end_t)

        n_sel_fit = int(sel_fit_mask.sum())
        n_sel_eval = int(sel_eval_mask.sum())

        if n_sel_fit < MIN_TRAIN_ROWS or n_sel_eval < MIN_TRAIN_ROWS:
            pbar.update(1)
            continue

        X_sel_fit = X_pool[sel_fit_mask]
        y_sel_fit = y_pool[sel_fit_mask]
        X_sel_eval = X_pool[sel_eval_mask]
        y_sel_eval = y_pool[sel_eval_mask]

        if len(np.unique(y_sel_fit)) < 2 or len(np.unique(y_sel_eval)) < 2:
            pbar.update(1)
            continue

        sel_aucs = np.full(H, 0.5, dtype=np.float32)
        for j in range(H):
            col_fit = X_sel_fit[:, j]
            col_eval = X_sel_eval[:, j]
            nz_fit = col_fit != 0
            nz_eval = col_eval != 0
            if nz_fit.sum() < 50 or nz_eval.sum() < 20:
                continue
            y_fit_j = y_sel_fit[nz_fit]
            if len(np.unique(y_fit_j)) < 2:
                continue
            y_eval_j = y_sel_eval[nz_eval]
            if len(np.unique(y_eval_j)) < 2:
                continue
            clf_j = LogisticRegression(
                penalty="l2", C=0.5, solver="lbfgs",
                max_iter=200, random_state=SEED,
            )
            clf_j.fit(col_fit[nz_fit].reshape(-1, 1), y_fit_j)
            scores_j = clf_j.decision_function(col_eval[nz_eval].reshape(-1, 1))
            sel_aucs[j] = float(roc_auc_score(y_eval_j, scores_j))

        k = min(TOP_K, H)
        selected = np.argsort(-sel_aucs)[:k]
        if selected.size == 0:
            pbar.update(1)
            continue

        # Meta-model: expanding window up to val_start_t
        # Generate OOS base-model predictions for training the meta-model
        train_mask = t_idx < val_start_t
        val_mask = (t_idx >= val_start_t) & (t_idx < val_end_t)

        X_tr_all = X_pool[train_mask]
        y_tr_all = y_pool[train_mask]
        X_va_all = X_pool[val_mask]
        y_va_all = y_pool[val_mask]

        if len(X_va_all) < MIN_TRAIN_ROWS:
            pbar.update(1)
            continue
        if len(np.unique(y_va_all)) < 2:
            pbar.update(1)
            continue

        # Build OOS predictions for selected miners on the training set
        # using sub-walk-forward within [0, val_start_t)
        K = selected.size
        meta_train = np.full((len(X_tr_all), K), np.nan, dtype=np.float32)
        meta_val = np.full((len(X_va_all), K), np.nan, dtype=np.float32)

        t_tr = t_idx[train_mask]
        sub_chunk = max(CHUNK_T, 2000)
        sub_segs = []
        ss = 0
        while ss < val_start_t:
            svs = ss + LAG_T
            if svs >= val_start_t:
                break
            se = min(ss + sub_chunk, val_start_t)
            if se <= svs:
                break
            sub_segs.append((ss, svs, se))
            ss = se

        for col_i, j in enumerate(selected):
            col_tr = X_tr_all[:, j]
            col_va = X_va_all[:, j]

            for (ss_start, ss_val, ss_end) in sub_segs:
                fit_mask_sub = t_tr < ss_val
                oos_mask_sub = (t_tr >= ss_val) & (t_tr < ss_end)
                nz_fit = fit_mask_sub & (col_tr != 0)
                nz_oos = oos_mask_sub & (col_tr != 0)
                if nz_fit.sum() < 50 or nz_oos.sum() == 0:
                    continue
                y_fit_sub = y_tr_all[nz_fit]
                if len(np.unique(y_fit_sub)) < 2:
                    continue
                clf_sub = LogisticRegression(
                    penalty="l2", C=0.5, solver="lbfgs",
                    max_iter=200, random_state=SEED,
                )
                clf_sub.fit(col_tr[nz_fit].reshape(-1, 1), y_fit_sub)
                meta_train[nz_oos, col_i] = clf_sub.decision_function(
                    col_tr[nz_oos].reshape(-1, 1)
                ).ravel()

            nz_tr = col_tr != 0
            nz_va = col_va != 0
            if nz_tr.sum() >= 50 and len(np.unique(y_tr_all[nz_tr])) >= 2:
                clf_full = LogisticRegression(
                    penalty="l2", C=0.5, solver="lbfgs",
                    max_iter=200, random_state=SEED,
                )
                clf_full.fit(col_tr[nz_tr].reshape(-1, 1), y_tr_all[nz_tr])
                if nz_va.sum() > 0:
                    meta_val[nz_va, col_i] = clf_full.decision_function(
                        col_va[nz_va].reshape(-1, 1)
                    ).ravel()

        # Fit meta-model on OOS base predictions
        row_has_any = np.any(~np.isnan(meta_train), axis=1)
        if row_has_any.sum() < MIN_TRAIN_ROWS:
            pbar.update(1)
            continue
        X_meta_tr = np.where(np.isnan(meta_train[row_has_any]), 0.0, meta_train[row_has_any])
        y_meta_tr = y_tr_all[row_has_any]
        if len(np.unique(y_meta_tr)) < 2:
            pbar.update(1)
            continue

        meta_clf = LogisticRegression(
            penalty="l2", C=0.5, class_weight="balanced",
            solver="lbfgs", max_iter=100, random_state=SEED,
        )
        meta_clf.fit(X_meta_tr, y_meta_tr)

        X_meta_va = np.where(np.isnan(meta_val), 0.0, meta_val)
        proba = meta_clf.predict_proba(X_meta_va)[:, 1]
        auc = float(roc_auc_score(y_va_all, proba))

        coef = np.abs(meta_clf.coef_.ravel())
        scale = max((auc - 0.5) / 0.5, 0.0)

        imp = np.zeros(H, dtype=np.float64)
        for ci, gj in enumerate(selected):
            imp[gj] = float(coef[ci]) * scale

        w = recency_gamma ** (len(segments) - 1 - seg_i)
        hk_imp += w * imp
        total_w += w
        pbar.update(1)

    pbar.close()

    if total_w <= 0:
        return {}

    hk_imp /= total_w

    idx2hk = {v: k for k, v in hk2idx.items()}
    out: Dict[str, float] = {}
    for j in range(H):
        hk = idx2hk.get(j, str(j))
        v = float(hk_imp[j])
        if v > 0:
            out[hk] = v

    total = sum(out.values())
    if total <= 0:
        return {}
    return {hk: v / total for hk, v in out.items()}
