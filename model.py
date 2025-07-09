# MIT License
#
# Copyright (c) 2024 MANTIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import config
import os, time, csv
from datetime import datetime

import torch
import logging
import random
import numpy as np
import xgboost as xgb
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logger.info("Salience computations will run on %s", DEVICE)

try:
    _NUM_CPU = max(1, os.cpu_count() or 1)
    torch.set_num_threads(_NUM_CPU)
    torch.set_num_interop_threads(_NUM_CPU)
    logger.info("Torch thread pools set to %d", _NUM_CPU)
except Exception as e:
    logger.warning("Could not set torch thread counts: %s", e)

def set_global_seed(seed: int) -> None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic PyTorch algorithms enabled.")
    except (RuntimeError, AttributeError) as e:
        logger.warning(f"Could not enable deterministic algorithms: {e}")

def salience(
    multi_asset_data: dict[str, tuple[dict[int, list[list[float]]], list[float]]],
) -> dict[int, float]:
    """
    Computes salience scores for each UID across all assets using XGBoost feature importances.

    - For each asset, trains a separate XGBoost model.
    - Uses permutation importance to determine UID importance.
    - Final salience is based on the combined UID importance across all assets.

    Args:
        multi_asset_data: A dictionary mapping asset names to (history_dict, returns) tuples.

    Returns:
        A dictionary mapping each UID to its estimated salience.
    """
    set_global_seed(config.SEED)

    if not multi_asset_data:
        logger.warning("Salience function called with empty multi-asset data.")
        return {}

    t0 = time.time()

    all_uids = sorted(list(set(k for asset_data in multi_asset_data.values() for k in asset_data[0].keys())))
    if not all_uids:
        logger.warning("No UIDs found in the provided data.")
        return {}
    
    logger.info(
        f"Starting XGBoost-based salience for {len(all_uids)} UIDs across {len(multi_asset_data)} assets."
    )

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
    }

    if DEVICE == "cuda":
        xgb_params["device"] = "cuda"
        xgb_params["tree_method"] = "gpu_hist"
        xgb_params["predictor"] = "gpu_predictor"
    else:
        xgb_params["nthread"] = os.cpu_count()

    xgb_rounds = 50

    asset_uid_importance = {}

    for asset_name, (history_dict, asset_returns) in multi_asset_data.items():
        logger.info(f"Processing asset for XGBoost importance: {asset_name}")
        
        if not history_dict or not asset_returns:
            logger.warning(f"Empty data for asset {asset_name}, skipping.")
            continue
        
        emb_dim = config.ASSET_EMBEDDING_DIMS[asset_name]
        uids = sorted(list(history_dict.keys()))
        uid_to_idx = {uid: i for i, uid in enumerate(uids)}
        num_uids = len(uids)
        T = len(asset_returns)
        
        X = np.zeros((T, num_uids * emb_dim), dtype=np.float32)
        for uid, history in history_dict.items():
            if uid in uid_to_idx:
                idx = uid_to_idx[uid]
                if len(history) == T:
                    X[:, idx * emb_dim : (idx + 1) * emb_dim] = np.array(history, dtype=np.float32)
        
        y_binary = (np.array(asset_returns, dtype=np.float32) > 0).astype(float)
        
        CHUNK_SIZE = 2000
        LAG = config.LAG

        total_uid_importance = np.zeros(num_uids)
        num_windows = 0
        train_end = CHUNK_SIZE

        logger.info(f"   -> Starting walk-forward validation for {asset_name} (chunk_size={CHUNK_SIZE}, lag={LAG})")
        
        pbar_total = 0
        temp_train_end = CHUNK_SIZE
        while (temp_train_end + CHUNK_SIZE) <= T:
            pbar_total += 1
            temp_train_end += CHUNK_SIZE

        pbar = tqdm(total=pbar_total, desc=f"   Walk-forward {asset_name}")
        while True:
            val_start = train_end + LAG
            val_end = train_end + CHUNK_SIZE

            if val_end > T:
                break

            X_train, y_train = X[:train_end], y_binary[:train_end]
            X_val, y_val = X[val_start:val_end], y_binary[val_start:val_end]
            
            if len(X_train) < 100 or len(X_val) == 0:
                logger.warning(f"Skipping window for {asset_name}: not enough train/val data.")
                train_end = val_end
                continue

            dtrain = xgb.DMatrix(X_train, label=y_train)
            bst = xgb.train(xgb_params, dtrain, num_boost_round=xgb_rounds, verbose_eval=False)
            
            dval = xgb.DMatrix(X_val, label=y_val)
            base_preds = (bst.predict(dval) > 0.5).astype(int)
            base_acc = np.mean(base_preds == y_val)
            
            window_uid_importance = np.zeros(num_uids)
            for i, uid in enumerate(uids):
                X_perm = X_val.copy()
                cols = slice(i * emb_dim, (i + 1) * emb_dim)
                X_perm[:, cols] = np.random.permutation(X_perm[:, cols])
                
                dval_perm = xgb.DMatrix(X_perm, label=y_val)
                perm_preds = (bst.predict(dval_perm) > 0.5).astype(int)
                perm_acc = np.mean(perm_preds == y_val)
                
                window_uid_importance[i] = max(base_acc - perm_acc, 0)
            
            total_uid_importance += window_uid_importance
            num_windows += 1
            train_end = val_end
            pbar.update(1)

        pbar.close()

        if num_windows > 0:
            avg_uid_importance = total_uid_importance / num_windows
            asset_uid_importance[asset_name] = dict(zip(uids, avg_uid_importance))
        else:
            logger.warning(f"No validation windows processed for {asset_name}.")

    final_importance = {uid: 0.0 for uid in all_uids}
    for asset_name, importances in asset_uid_importance.items():
        for uid, importance_score in importances.items():
            if uid in final_importance:
                final_importance[uid] += importance_score
            
    total_importance = sum(final_importance.values())
    if total_importance > 1e-9:
        salience_dict = {uid: score / total_importance for uid, score in final_importance.items()}
    else:
        salience_dict = {uid: 1.0 / len(all_uids) for uid in all_uids if all_uids}

    logger.info("XGBoost-based salience computation complete in %.2fs", time.time() - t0)
    return salience_dict
