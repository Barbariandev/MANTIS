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
N_ASSETS = len(config.FUNDING_ASSETS)

# Tolerance window: the forward sidx match must be within this fraction of
# the target ahead value.  E.g. for ahead=480 and tolerance 0.15, we accept
# matches in [408, 552].  This prevents using rows that are wildly misaligned
# while allowing for minor sidx gaps from validator downtime.
_SIDX_MATCH_TOLERANCE = 0.15


def _build_forward_pairs(
    sidx_arr: np.ndarray,
    ahead: int,
) -> np.ndarray:
    """For each row index t, find the row index whose sidx is closest to
    sidx_arr[t] + ahead, within a tolerance window.  Returns an array of
    length T where fwd[t] is the matched row index, or -1 if no acceptable
    match exists.

    Uses np.searchsorted for O(T log T) matching instead of a two-pointer
    sweep, which avoids subtle pointer-sharing bugs on gapped data."""
    T = len(sidx_arr)
    fwd = np.full(T, -1, dtype=np.int64)
    tol = int(ahead * _SIDX_MATCH_TOLERANCE)
    if tol < 1:
        tol = 1

    targets = sidx_arr + ahead
    # For each target, find the insertion point (first sidx >= target)
    insertion = np.searchsorted(sidx_arr, targets, side="left")

    for t in range(T):
        best_j = -1
        best_dist = tol + 1
        # Check the candidate at and just before the insertion point
        for cand in (insertion[t] - 1, insertion[t]):
            if 0 <= cand < T and cand > t:
                dist = abs(int(sidx_arr[cand]) - int(targets[t]))
                if dist <= tol and dist < best_dist:
                    best_dist = dist
                    best_j = cand
        fwd[t] = best_j
    return fwd


def compute_funding_xsec_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    funding_rates: np.ndarray,
    blocks_ahead: int,
    sample_every: int,
    sidx_arr: np.ndarray | None = None,
) -> Dict[str, float]:
    """Walk-forward salience for the FUNDING-XSEC challenge.

    Cross-sectional median-relative reformulation: for each (timestep, asset),
    label = 1 if the asset's forward funding rate *change* exceeds the
    cross-sectional median change.  This removes the market-wide funding factor
    (beta) by construction and targets asset-specific deviations (alpha).

    Unlike the XSEC-RANK challenge, forward pairing is done by matching sidx
    values rather than row-position, because funding rates settle on fixed 8h
    boundaries and row-position indexing can create misaligned labels when
    there are gaps in the data stream.
    """
    LAG_T = int(getattr(config, "LAG", 60))
    CHUNK_T = int(getattr(config, "CHUNK_SIZE", 4000))
    TOP_K = int(getattr(config, "TOP_K", 20))
    HALFLIFE = int(getattr(config, "WINDOWS_HALF_LIFE", 10))
    MIN_TRAIN_ROWS = 200

    X_flat, hk2idx = hist
    X_flat = np.asarray(X_flat, dtype=np.float32)
    fr = np.asarray(funding_rates, dtype=np.float64)

    T = X_flat.shape[0]
    H = X_flat.shape[1] // N_ASSETS
    if H <= 0 or H * N_ASSETS != X_flat.shape[1]:
        return {}
    if fr.shape != (T, N_ASSETS):
        return {}

    X = X_flat.reshape(T, H, N_ASSETS)

    ahead = blocks_ahead // sample_every
    if ahead <= 0 or T <= ahead:
        return {}

    # ---- Sidx-based forward pairing ----
    if sidx_arr is not None and len(sidx_arr) == T:
        fwd_idx = _build_forward_pairs(sidx_arr, ahead)
    else:
        # Fallback: assume contiguous rows (legacy compat)
        fwd_idx = np.arange(T, dtype=np.int64)
        fwd_idx[:T - ahead] += ahead
        fwd_idx[T - ahead:] = -1

    has_fwd = fwd_idx >= 0
    valid_rows = np.where(has_fwd)[0]
    if len(valid_rows) < 500:
        return {}

    # Compute forward funding rate change only for matched pairs
    fr_now = fr[valid_rows]                  # (N_valid, N_ASSETS)
    fr_fut = fr[fwd_idx[valid_rows]]         # (N_valid, N_ASSETS)
    delta_fr = fr_fut - fr_now

    # Cross-sectional demeaning: label = 1 if asset's change > median change
    # at that timestep.  Exactly 50% base rate by construction.
    med_delta = np.nanmedian(delta_fr, axis=1, keepdims=True)
    y_2d = np.where(
        np.isnan(delta_fr) | np.isnan(med_delta),
        np.nan,
        (delta_fr > med_delta).astype(np.float32),
    )

    N_valid = len(valid_rows)

    # Zero out temporally-constant miner columns (stale submitters).
    # A miner whose signal has near-zero variance across time for an asset
    # provides no information; its constant value would only let the
    # meta-model overfit asset-level intercepts after pooling.
    X_valid = X[valid_rows]  # (N_valid, H, N_ASSETS)
    MIN_TEMPORAL_STD = 1e-4
    for j in range(H):
        for a in range(N_ASSETS):
            col = X_valid[:, j, a]
            if np.std(col) < MIN_TEMPORAL_STD:
                X_valid[:, j, a] = 0.0

    # Pool across assets: (N_valid * N_ASSETS, H)
    X_pool = X_valid.transpose(0, 2, 1).reshape(N_valid * N_ASSETS, H)
    y_pool = y_2d.reshape(N_valid * N_ASSETS)
    # t_idx tracks which *valid row* each pooled sample belongs to, for
    # walk-forward segment assignment.  We use the original row index so
    # temporal ordering is preserved.
    t_idx = np.repeat(valid_rows.astype(np.int32), N_ASSETS)

    valid_mask = ~np.isnan(y_pool)
    X_pool = X_pool[valid_mask].copy()
    y_pool = y_pool[valid_mask].copy()
    t_idx = t_idx[valid_mask]

    if len(y_pool) < MIN_TRAIN_ROWS or len(np.unique(y_pool)) < 2:
        return {}

    # Use original row indices for segment boundaries so that the temporal
    # embargo (LAG_T) is in the same units as the data.
    T_u = int(valid_rows.max()) + 1

    # Walk-forward temporal segments (expanding window).
    # The embargo between train and val must be >= ahead because
    # forward-looking labels at training row t use funding rates at
    # fwd[t] ≈ t + ahead.  Train rows with t >= val_start - ahead
    # have labels that reference val-window funding rates (leakage),
    # so they are excluded via train_cutoff below.
    embargo = max(LAG_T, ahead)

    segments = []
    s = 0
    while s < T_u:
        vs = s + embargo
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

    pbar = tqdm(total=len(segments), desc="SAL(FundXSec) Walk-fwd")

    for seg_i, (seg_start, val_start_t, val_end_t) in enumerate(segments):
        sel_eval_end_t = max(0, val_start_t - ahead)
        sel_eval_start_t = max(0, sel_eval_end_t - CHUNK_T)
        sel_fit_end_t = max(0, sel_eval_start_t - ahead)

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

        MIN_UNIQUE_VALS = 5
        sel_aucs = np.full(H, 0.5, dtype=np.float32)
        for j in range(H):
            col_fit = X_sel_fit[:, j]
            col_eval = X_sel_eval[:, j]
            nz_fit = col_fit != 0
            nz_eval = col_eval != 0
            if nz_fit.sum() < 50 or nz_eval.sum() < 20:
                continue
            # Skip near-constant features (stale/degenerate miners)
            if len(np.unique(col_fit[nz_fit])) < MIN_UNIQUE_VALS:
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

        # Exclude training rows whose forward labels reach into the val window.
        train_cutoff = val_start_t - ahead
        if train_cutoff <= 0:
            pbar.update(1)
            continue
        train_mask = t_idx < train_cutoff
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

        K = selected.size
        meta_train = np.full((len(X_tr_all), K), np.nan, dtype=np.float32)
        meta_val = np.full((len(X_va_all), K), np.nan, dtype=np.float32)

        t_tr = t_idx[train_mask]
        sub_chunk = max(CHUNK_T, 2000)
        sub_segs = []
        ss = 0
        while ss < train_cutoff:
            svs = ss + ahead
            if svs >= train_cutoff:
                break
            se = min(ss + sub_chunk, train_cutoff)
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

        row_has_any = np.any(~np.isnan(meta_train), axis=1)
        if row_has_any.sum() < MIN_TRAIN_ROWS:
            pbar.update(1)
            continue
        X_meta_tr = np.where(
            np.isnan(meta_train[row_has_any]), 0.0, meta_train[row_has_any]
        )
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
