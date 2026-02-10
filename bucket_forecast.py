from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


class LinearEnsembleOptimizer(nn.Module):
    def __init__(self, num_miners: int, num_buckets: int = 5):
        super().__init__()
        self.miner_weights = nn.Parameter(torch.ones(num_miners) / max(num_miners, 1))
        self.bias = nn.Parameter(torch.zeros(num_buckets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = int(x.shape[0])
        H = int(self.miner_weights.shape[0])
        x_reshaped = x.view(B, H, -1)
        miner_probs = x_reshaped[:, :, :5]

        miner_probs = torch.clamp(miner_probs, 1e-6, 1.0)
        miner_probs = miner_probs / miner_probs.sum(dim=2, keepdim=True)

        weights = torch.softmax(self.miner_weights, dim=0)
        weighted_probs = torch.einsum("bmc,m->bc", miner_probs, weights)
        return torch.log(weighted_probs + 1e-9) + self.bias

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        max_epochs: int = 80,
        patience: int = 15,
        lr: float = 0.05,
        batch_size: int = 1024,
        device: str = "cpu",
        class_weights: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        val_split: float = 0.15,
        verbose: bool = False,
    ) -> "LinearEnsembleOptimizer":
        self.to(device)

        n = int(X.shape[0])
        val_size = int(n * val_split)
        train_size = n - val_size

        X_train = torch.as_tensor(X[:train_size], dtype=torch.float32, device=device)
        y_train = torch.as_tensor(y[:train_size], dtype=torch.long, device=device)
        X_val = torch.as_tensor(X[train_size:], dtype=torch.float32, device=device)
        y_val = torch.as_tensor(y[train_size:], dtype=torch.long, device=device)

        cw_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None

        if sample_weights is None:
            sw = np.ones(train_size, dtype=np.float32)
        else:
            sw = np.asarray(sample_weights[:train_size], dtype=np.float32)
        sw_t = torch.as_tensor(sw, dtype=torch.float32, device=device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train, sw_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(int(max_epochs)):
            self.train()
            train_loss = 0.0
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                logits = self(xb)
                loss_vec = F.cross_entropy(logits, yb, weight=cw_t, reduction="none")
                denom = torch.clamp(wb.sum(), min=1.0)
                loss = (loss_vec * wb).sum() / denom
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())

            self.eval()
            with torch.no_grad():
                val_logits = self(X_val)
                val_loss = float(F.cross_entropy(val_logits, y_val, weight=cw_t).item())

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch % 10 == 0):
                logger.info(
                    "LinearEnsemble epoch %d | train=%.4f | val=%.4f",
                    epoch,
                    train_loss / max(len(loader), 1),
                    val_loss,
                )

            if epochs_no_improve >= patience:
                if verbose:
                    logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self.load_state_dict(best_state)
        return self


def compute_linear_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    *,
    blocks_ahead: int,
    sample_every: int,
    max_epochs: int = 80,
    device: str = "cpu",
) -> Dict[str, float]:
    X_flat, hk2idx = hist
    price_arr = np.asarray(price_data, dtype=float)
    if not hk2idx:
        return {}
    if price_arr.ndim != 1:
        return {}

    required = int(MIN_REQUIRED_SAMPLES)
    if price_arr.size < required or X_flat.shape[0] < required:
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
    vol_window = max(required // 2, 1000)
    y_all, valid_idx = make_bins_from_price(price_arr, horizon_steps=horizon_steps, vol_window=vol_window)
    if valid_idx.size < required:
        return {}

    X_train = X_flat[valid_idx]
    y_train = y_all
    if X_train.shape[0] != y_train.shape[0]:
        return {}

    sw = recent_mass_weights(valid_idx.astype(float), recent_samples=RECENT_SAMPLES, recent_mass=RECENT_MASS)

    classes, counts = np.unique(y_train, return_counts=True)
    K = int(classes.max() + 1) if classes.size > 0 else 0
    if K <= 0:
        return {}
    class_weights = np.ones(K, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        class_weights[int(c)] = float(len(y_train)) / (K * float(cnt))

    num_miners = len(hk2idx)
    model = LinearEnsembleOptimizer(num_miners=num_miners, num_buckets=5)
    model.fit(X_train, y_train, max_epochs=max_epochs, device="cpu", class_weights=class_weights, sample_weights=sw)

    model.eval()
    with torch.no_grad():
        miner_weights = torch.softmax(model.miner_weights, dim=0).cpu().numpy()

    if miner_weights.shape[0] > TOP_K:
        order = np.argsort(-miner_weights)
        keep_idx = order[:TOP_K]
        kept = miner_weights[keep_idx]
        s = float(kept.sum())
        if s > 0.0:
            kept = kept / s
        pruned = np.zeros_like(miner_weights)
        pruned[keep_idx] = kept
        miner_weights = pruned

    sal = {}
    for hk, idx in hk2idx.items():
        if 0 <= int(idx) < miner_weights.shape[0]:
            w = float(miner_weights[int(idx)])
            if w > 0.0:
                sal[hk] = w
    return sal


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
    Q-path salience as 12 independent global models:
      c ∈ {0,1,3,4} × threshold ∈ {0.5σ, 1.0σ, 2.0σ}

    Each model learns a simplex mixture over miner Q logits (logit(Q)) to predict
    the opposite-direction threshold hit (binary), and salience is the learned
    mixture weights. The 12 saliences are averaged and then top-K pruned.
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

    Xr = np.asarray(X_flat[:len_r], dtype=float).reshape(len_r, H, 17)

    device = "cpu"
    lr = 0.05
    epochs = 20
    batch_size = 4096

    def fit_binary_logit_mixture(x_logits: np.ndarray, y: np.ndarray, sw: np.ndarray) -> np.ndarray | None:
        x_logits = np.asarray(x_logits, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        sw = np.asarray(sw, dtype=np.float32).reshape(-1)
        if x_logits.ndim != 2:
            return None
        N, HH = x_logits.shape
        if N <= 0 or HH != H or y.shape[0] != N or sw.shape[0] != N:
            return None
        if np.unique(y).size < 2:
            return None

        pos = float(np.sum(y > 0.5))
        neg = float(N - pos)
        if pos <= 0.0 or neg <= 0.0:
            return None
        w_pos = float(N / (2.0 * pos))
        w_neg = float(N / (2.0 * neg))
        w_cls = np.where(y > 0.5, w_pos, w_neg).astype(np.float32)
        w = (sw * w_cls).astype(np.float32)
        s = float(np.sum(w))
        if s > 0.0:
            w *= float(N / s)

        X_t = torch.as_tensor(x_logits, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        w_t = torch.as_tensor(w, dtype=torch.float32, device=device)

        miner_logits = torch.nn.Parameter(torch.zeros(H, dtype=torch.float32, device=device))
        bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
        opt = torch.optim.Adam([miner_logits, bias], lr=float(lr))

        dataset = torch.utils.data.TensorDataset(X_t, y_t, w_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

        for _ in range(int(epochs)):
            for xb, yb, wb in loader:
                opt.zero_grad()
                w_mix = torch.softmax(miner_logits, dim=0)
                z = torch.einsum("bh,h->b", xb, w_mix) + bias[0]
                loss_vec = F.binary_cross_entropy_with_logits(z, yb, reduction="none")
                denom = torch.clamp(wb.sum(), min=1.0)
                loss = (loss_vec * wb).sum() / denom
                loss.backward()
                opt.step()

        with torch.no_grad():
            out = torch.softmax(miner_logits, dim=0).detach().cpu().numpy()
        return out

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
            q[q == 0.0] = 0.5
            q = np.clip(q, EPS, 1.0 - EPS)
            x_logits = logit(q)

            w_model = fit_binary_logit_mixture(x_logits, y_hit, sw)
            if w_model is not None:
                per_model_weights.append(w_model)

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



