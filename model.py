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

FEATURE_LENGTH = config.FEATURE_LENGTH

import torch
import torch.nn as nn
import logging
import random
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    """Sets the seed for all relevant RNGs to ensure reproducibility."""
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

COMPILE_AVAILABLE = hasattr(torch, "compile")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

if COMPILE_AVAILABLE:
    try:
        logger.info("Enabling torch.compile() for MLP")
        MLP = torch.compile(MLP)
    except Exception as e:
        logger.warning("torch.compile unavailable or failed post-definition: %s", e)

def salience(
    history_dict: dict[int, list[list[float]]],
    btc_returns: list[float],
    hidden_size: int = config.HIDDEN_SIZE,
    lr: float = config.LEARNING_RATE,
) -> dict[int, float]:
    """
    Computes salience scores for each UID by minimizing binary cross-entropy loss.

    This function now assumes that the input data is clean and pre-processed
    by the DataLog. Specifically:
    - `history_dict` contains entries for all relevant UIDs.
    - All history sequences in `history_dict` have the same length.
    - `len(btc_returns)` matches the length of the history sequences.

    This function will now log extensive details about each training run to a
    timestamped CSV file in the `salience_logs` directory. This includes loss
    values for every timestep, allowing for detailed analysis and visualization.

    Args:
        history_dict: A dictionary mapping UIDs to their embedding history.
        btc_returns: The target series of BTC percentage changes.
        hidden_size: The hidden layer width for the proxy MLP model.
        lr: The learning rate for the proxy model optimizer.

    Returns:
        A dictionary mapping each UID to its salience score.
    """
    set_global_seed(config.SEED)

    if not history_dict or not btc_returns:
        logger.warning("Salience function called with empty history or returns.")
        return {}

    t0 = time.time()

    log_dir = "salience_logs"
    log_file = None
    log_writer = None
    try:
        os.makedirs(log_dir, exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"salience_run_{run_timestamp}.csv")
        logger.info(f"Logging detailed training data to {log_filename}")
        log_file = open(log_filename, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(['uid_masked', 'timestep', 'eval_loss'])
    except IOError as e:
        logger.error(f"Failed to create salience log file {log_filename}: {e}")
        log_writer = None
        log_file = None

    uids = sorted(list(history_dict.keys()))
    uid_to_idx = {uid: i for i, uid in enumerate(uids)}
    num_uids = len(uids)
    emb_dim = config.FEATURE_LENGTH
    T = len(btc_returns)

    logger.info(
        f"Starting salience computation for {num_uids} UIDs over {T} timesteps."
    )

    X = torch.zeros(T, num_uids * emb_dim, dtype=torch.float32, device=DEVICE)
    for uid, history in history_dict.items():
        idx = uid_to_idx[uid]
        h_tensor = torch.tensor(history, dtype=torch.float32, device=DEVICE)
        X[:, idx * emb_dim : (idx + 1) * emb_dim] = h_tensor

    y_float = torch.tensor(btc_returns, dtype=torch.float32, device=DEVICE).view(-1, 1)
    y_binary = (y_float > 0).float()

    logger.debug(f"Feature matrix shape: {X.shape}, target vector shape: {y_binary.shape}")

    def run_model(mask_uid_idx: int | None = None, uid_for_log: str | int | None = None) -> float:
        """
        Return average prediction loss using a progressive walk-forward approach.
        """
        model = MLP(X.shape[1], hidden_size, 1).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.BCEWithLogitsLoss()
        total_loss = 0.0

        X_run = X
        if mask_uid_idx is not None:
            X_run = X.clone()
            s = mask_uid_idx * emb_dim
            X_run[:, s : s + emb_dim] = 0.0

        BATCH_SIZE = getattr(config, 'BATCH_SIZE', 1000)
        LAG_OFFSET = getattr(config, 'LAG', 60)
        MIN_TRAIN_SIZE = getattr(config, 'MIN_TRAIN_SIZE', 100)
        inference_samples = 0
        
        inference_start = LAG_OFFSET + MIN_TRAIN_SIZE
        
        while inference_start < T:
            train_start = 0
            train_end = inference_start - LAG_OFFSET
            
            inference_end = min(inference_start + BATCH_SIZE, T)
            
            if train_end <= train_start or inference_start >= inference_end:
                break
                
            train_X = X_run[train_start:train_end]
            train_y_binary = y_binary[train_start:train_end]
            
            model.train()
            for epoch in range(5):
                opt.zero_grad()
                train_logits = model(train_X)
                train_loss = crit(train_logits, train_y_binary)
                train_loss.backward()
                opt.step()
            
            inference_X = X_run[inference_start:inference_end]
            inference_y_binary = y_binary[inference_start:inference_end]
            
            model.eval()
            with torch.no_grad():
                pred_logits = model(inference_X)
                total_loss += crit(pred_logits, inference_y_binary).item() * inference_X.shape[0]
                inference_samples += inference_X.shape[0]

                if log_writer:
                    for t_local in range(inference_X.shape[0]):
                        t_global = inference_start + t_local
                        eval_loss_val = crit(pred_logits[t_local:t_local+1], inference_y_binary[t_local:t_local+1]).item()
                        log_writer.writerow([uid_for_log, t_global, eval_loss_val])
            
            inference_start = inference_end

        return total_loss / inference_samples if inference_samples > 0 else 0.0

    set_global_seed(config.SEED)
    full_loss = run_model(uid_for_log="full")
    logger.debug(f"Full (no-mask) model loss: {full_loss:.6f}")

    losses = []
    logger.info("Computing masked losses for all UIDs...")
    for uid in uids:
        set_global_seed(config.SEED) 
        h_raw = history_dict.get(uid)
        if not h_raw or not any(any(v != 0 for v in vec) for vec in h_raw):
            losses.append(full_loss) 
            if log_writer:
                log_writer.writerow([uid, "SKIPPED", full_loss])
            continue

        uidx = uid_to_idx[uid]
        l = run_model(uidx, uid_for_log=uid)
        losses.append(l)

    deltas = torch.tensor([l - full_loss for l in losses]).clamp(min=0.0)

    salience_dict = {}
    if deltas.sum() > 0:
        weights = deltas / deltas.sum()
        salience_dict = dict(zip(uids, weights.cpu().tolist()))
    else:
        salience_dict = {uid: 0.0 for uid in uids}

    if log_file:
        log_file.close()

    logger.info("Salience computation complete in %.2fs", time.time() - t0)
    return salience_dict
