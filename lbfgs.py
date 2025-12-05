
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)
__all__ = (
    "compute_lbfgs_salience",
    "QCalibConfig",
    "QPathCalibrator",
    "progressive_q_saliences",
    "compute_q_path_salience",
    "compute_hitfirst_salience",
)

_LN2, _EPS = np.log(2.0), 1e-12
MIN_REQUIRED_SAMPLES = 7200


def _rolling_std_fast(r1: np.ndarray, window: int) -> np.ndarray:
    n = len(r1)
    if n < window:
        return np.full(0, np.nan)
    c1 = np.concatenate([[0.0], np.cumsum(r1)])
    c2 = np.concatenate([[0.0], np.cumsum(r1 * r1)])
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    var = (s2 - (s1 * s1) / window) / max(window - 1, 1)
    return np.sqrt(np.maximum(var, 0.0))

def _make_bins_from_price(price: np.ndarray, horizon: int = 1, vol_window: int = 7200,
                          eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    price = np.asarray(price, dtype=float)
    if price.ndim != 1:
        raise ValueError("price_data must be 1-D array")
    T = price.shape[0]
    if T <= horizon + vol_window:
        raise ValueError("Not enough data: need > horizon + vol_window samples")

    r = np.log(price[horizon:] + eps) - np.log(price[:-horizon] + eps)
    sig_raw = _rolling_std_fast(r, vol_window)
    sig = np.full(len(r), np.nan)
    if sig_raw.size <= 0:
        raise ValueError("No valid labels after rolling sigma computation")
    sig[vol_window - 1:] = sig_raw
    idx_all = np.arange(len(r))
    valid_mask = np.isfinite(sig)
    valid_idx = idx_all[valid_mask]
    if valid_idx.size == 0:
        raise ValueError("No valid labels after rolling sigma computation")

    z = r[valid_mask] / (sig[valid_mask] + eps)
    y = np.zeros_like(z, dtype=int)
    y[z <= -2.0] = 0
    y[(z > -2.0) & (z < -1.0)] = 1
    y[(z >= -1.0) & (z <= 1.0)] = 2
    y[(z > 1.0) & (z < 2.0)] = 3
    y[z >= 2.0] = 4
    return y, valid_idx

def _exp_half_life_weights(valid_idx: np.ndarray, half_life_days: float, samples_per_day: float) -> np.ndarray:
    if valid_idx.size == 0:
        return np.ones(0, dtype=float)
    i_max = float(valid_idx.max())
    age_days = (i_max - valid_idx.astype(float)) / float(samples_per_day)
    w = np.exp(-_LN2 * (age_days / float(half_life_days)))
    return w * (valid_idx.size / np.sum(w))

def _compute_hotkey_start_indices(X_flat: np.ndarray, H: int, D: int) -> np.ndarray:
    T = X_flat.shape[0]
    starts = np.full(H, T, dtype=int)
    for h in range(H):
        sl = slice(h * D, (h + 1) * D)
        sub = X_flat[:, sl]
        nonzero_rows = np.where(np.any(sub != 0.0, axis=1))[0]
        if nonzero_rows.size > 0:
            starts[h] = int(nonzero_rows[0])
    return starts

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log(1.0 - p)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

def _project_simplex(v: np.ndarray) -> np.ndarray:
    if v.ndim != 1: v = v.ravel()
    n = v.size
    if n == 0: return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=float)
    rho = np.where(cond)[0][-1]
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w if s > 0 else np.full(n, 1.0 / n, dtype=float)


class LinearEnsembleOptimizer(nn.Module):
    """
    A Linear Regression based ensemble (Logistic Regression in classification context).
    It learns a weighted average of miner predictions to form the final prediction.
    """

    def __init__(self, num_miners: int, num_buckets: int = 5):
        super().__init__()
        self.num_miners = num_miners
        self.num_buckets = num_buckets

        self.miner_weights = nn.Parameter(torch.ones(num_miners) / num_miners)
        self.bias = nn.Parameter(torch.zeros(num_buckets))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.num_miners, -1)
        miner_probs = x_reshaped[:, :, :5]

        miner_probs = torch.clamp(miner_probs, 1e-6, 1.0)
        miner_probs = miner_probs / miner_probs.sum(dim=2, keepdim=True)

        weights = torch.softmax(self.miner_weights, dim=0)
        weighted_probs = torch.einsum("bmc,m->bc", miner_probs, weights)

        ensemble_logits = torch.log(weighted_probs + 1e-9) + self.bias
        return ensemble_logits

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.05,
        batch_size: int = 1024,
        device: str = "cpu",
        verbose: bool = False,
        class_weights: Optional[np.ndarray] = None,
    ):
        self.to(device)

        split_idx = int(len(X) * 0.8)
        X_tr = torch.FloatTensor(X[:split_idx]).to(device)
        y_tr = torch.LongTensor(y[:split_idx]).to(device)
        X_val = torch.FloatTensor(X[split_idx:]).to(device)
        y_val = torch.LongTensor(y[split_idx:]).to(device)

        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)

        # No L1 penalty: we let the softmax over miner_weights and data decide
        # the effective sparsity; this keeps the objective purely cross-entropy.
        l1_lambda = 0.0

        best_val_loss = float("inf")
        best_state = None
        patience = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self(xb)
                ce_loss = self.criterion(logits, yb)
                l1_loss = l1_lambda * torch.sum(torch.abs(self.miner_weights))
                loss = ce_loss + l1_loss
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())

            avg_train_loss = train_loss / max(len(loader), 1)

            self.eval()
            with torch.no_grad():
                val_logits = self(X_val)
                val_loss = float(self.criterion(val_logits, y_val).item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if verbose and (epoch % 10 == 0):
                logger.info("LinearEnsemble epoch %d | train=%.4f val=%.4f", epoch, avg_train_loss, val_loss)

            if patience > 15:
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return self


def compute_linear_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    blocks_ahead: int,
    sample_every: int,
    epochs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    X_flat, hk2idx = hist
    price_arr = np.asarray(price_data, dtype=float)

    if len(hk2idx) == 0:
        return {}

    samples_per_day = int((24 * 60 * 60) // (12 * max(1, sample_every)))
    required = max(MIN_REQUIRED_SAMPLES, 100)

    if price_arr.size < required or X_flat.shape[0] < required:
        return {}

    horizon_steps = max(1, int(round(blocks_ahead / max(1, sample_every))))

    vol_window = max(required // 2, 1000)
    y_all, valid_idx = _make_bins_from_price(price_arr, horizon=horizon_steps, vol_window=vol_window)

    if valid_idx.size < required:
        return {}

    X_train = X_flat[valid_idx]
    y_train = y_all

    if X_train.shape[0] != y_train.shape[0]:
        return {}

    classes, counts = np.unique(y_train, return_counts=True)
    K = int(classes.max() + 1) if classes.size > 0 else 0
    if K <= 0:
        return {}
    class_weights = np.ones(K, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        class_weights[int(c)] = float(len(y_train)) / (K * float(cnt))

    num_miners = len(hk2idx)
    model = LinearEnsembleOptimizer(num_miners=num_miners, num_buckets=5)

    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"

    model.fit(X_train, y_train, epochs=epochs, device=device, class_weights=class_weights)

    model.eval()
    with torch.no_grad():
        miner_weights = torch.softmax(model.miner_weights, dim=0).cpu().numpy()

    # Hard top-K truncation (default K=25): keep only the K largest weights,
    # zero out the rest, and renormalise so the kept weights sum to 1.
    K = 25
    num_miners = miner_weights.shape[0]
    if K is not None and num_miners > K:
        order = np.argsort(-miner_weights)
        keep_idx = order[:K]
        kept = miner_weights[keep_idx]
        s = float(kept.sum())
        if s > 0.0:
            kept = kept / s
        pruned = np.zeros_like(miner_weights)
        pruned[keep_idx] = kept
        miner_weights = pruned

    inv_map = {idx: hk for hk, idx in hk2idx.items()}
    salience: Dict[str, float] = {}
    for idx, weight in enumerate(miner_weights):
        hk = inv_map.get(idx)
        if hk and weight > 0.0:
            salience[hk] = float(weight)
    return salience


def compute_lbfgs_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    blocks_ahead: int,
    sample_every: int,
    lbfgs_cfg: Optional[any] = None,
    min_days: float = 5.0,
    half_life_days: float = 5.0,
    use_class_weights: bool = True,
) -> Dict[str, float]:
    """
    Delegate LBFGS classifier salience to the in-module linear ensemble
    implementation (a constrained mixture over miner probability vectors
    trained to minimise cross-entropy / NLL).
    """
    return compute_linear_salience(
        hist,
        price_data,
        blocks_ahead=blocks_ahead,
        sample_every=sample_every,
    )


@dataclass
class QCalibConfig:
    max_iter: int = 200
    step_init: float = 1.0
    step_min: float = 1e-6
    backtrack: float = 0.5
    tol_grad: float = 1e-6
    l2_alpha: float = 0.0
    verbose: bool = False
    class_weighting: bool = True

class QPathCalibrator:
    def __init__(self, H: int, cfg: Optional[QCalibConfig] = None):
        self.H = int(H)
        self.cfg = cfg if cfg is not None else QCalibConfig()
        self.alpha_pos: Optional[np.ndarray] = None
        self.b_pos: Optional[np.ndarray] = None
        self.alpha_neg: Optional[np.ndarray] = None
        self.b_neg: Optional[np.ndarray] = None

    def _fit_one_dir(self, Q_logits: np.ndarray, Y: np.ndarray, w: np.ndarray):
        N, H, K = Q_logits.shape
        if N == 0:
            return np.full(H, 1.0 / H, dtype=float), np.zeros(3, dtype=float), {"n_iter": 0, "loss": float("nan")}

        alpha = np.full(H, 1.0 / H, dtype=float)
        b = np.zeros(3, dtype=float)

        def loss_grad(alpha_in: np.ndarray, b_in: np.ndarray):
            z_agg = np.einsum("nhk,h->nk", Q_logits, alpha_in)
            z = z_agg + b_in[None, :]
            p = _sigmoid(z)
            if self.cfg.class_weighting:
                pos_w = np.sum(Y * w[:, None], axis=0)
                neg_w = np.sum((1.0 - Y) * w[:, None], axis=0)
                w_pos = neg_w / np.maximum(pos_w, _EPS)
                w_neg = np.ones_like(w_pos)
                L_mat = (-(w_pos[None, :] * Y * np.log(np.clip(p, _EPS, 1.0))
                           + w_neg[None, :] * (1.0 - Y) * np.log(np.clip(1.0 - p, _EPS, 1.0)))) * w[:, None]
                loss = float(L_mat.sum() + 0.5 * self.cfg.l2_alpha * np.dot(alpha_in, alpha_in))
                class_mask = w_pos[None, :] * Y + w_neg[None, :] * (1.0 - Y)
                diff = (p - Y) * class_mask * w[:, None]
            else:
                L_mat = _bce(Y, p) * w[:, None]
                loss = float(L_mat.sum() + 0.5 * self.cfg.l2_alpha * np.dot(alpha_in, alpha_in))
                diff = (p - Y) * w[:, None]
            g_alpha = np.einsum("nk,nhk->h", diff, Q_logits) + self.cfg.l2_alpha * alpha_in
            g_b = np.sum(diff, axis=0)
            return loss, g_alpha, g_b

        eta = self.cfg.step_init
        for it in range(self.cfg.max_iter):
            L0, g_alpha, g_b = loss_grad(alpha, b)
            g_norm = float(np.linalg.norm(g_alpha, ord=2))
            if g_norm < self.cfg.tol_grad:
                return alpha, b, {"n_iter": it, "loss": L0, "grad_norm": g_norm}
            step = eta
            accepted = False
            while step >= self.cfg.step_min:
                alpha_new = _project_simplex(alpha - step * g_alpha)
                b_new = b - step * g_b
                L1, _, _ = loss_grad(alpha_new, b_new)
                if L1 <= L0:
                    alpha, b = alpha_new, b_new
                    accepted = True
                    break
                step *= self.cfg.backtrack
            if not accepted:
                return alpha, b, {"n_iter": it + 1, "loss": L0, "grad_norm": g_norm, "note": "line-search stop"}
        return alpha, b, {"n_iter": self.cfg.max_iter, "loss": L0, "grad_norm": float("nan")}

    def fit(self, Q_plus_logits: np.ndarray, Y_plus: np.ndarray, w_plus: np.ndarray,
                  Q_minus_logits: np.ndarray, Y_minus: np.ndarray, w_minus: np.ndarray):
        self.alpha_pos, self.b_pos, info_pos = self._fit_one_dir(Q_plus_logits,  Y_plus,  w_plus)
        self.alpha_neg, self.b_neg, info_neg = self._fit_one_dir(Q_minus_logits, Y_minus, w_minus)
        return {"pos": info_pos, "neg": info_neg,
                "alpha_pos": self.alpha_pos.copy(), "b_pos": self.b_pos.copy(),
                "alpha_neg": self.alpha_neg.copy(), "b_neg": self.b_neg.copy()}

    def salience_on_eval(self, Q_plus_logits: np.ndarray, Y_plus: np.ndarray, w_plus: np.ndarray,
                               Q_minus_logits: np.ndarray, Y_minus: np.ndarray, w_minus: np.ndarray,
                               hk2idx: Dict[str, int]) -> Dict[str, float]:
        H = self.H
        contrib = np.zeros(H, dtype=float)
        for h in range(H):
            contrib[h] += self._delta_loss_remove_hotkey(Q_plus_logits, Y_plus, w_plus, self.alpha_pos, self.b_pos, h)
        for h in range(H):
            contrib[h] += self._delta_loss_remove_hotkey(Q_minus_logits, Y_minus, w_minus, self.alpha_neg, self.b_neg, h)
        total = float(np.sum(contrib))
        inv_map = {idx: hk for hk, idx in hk2idx.items()}
        if total > 0.0:
            return {inv_map[i]: float(contrib[i] / total) for i in range(H)}
        else:
            return {}

    @staticmethod
    def _delta_loss_remove_hotkey(Q_logits: np.ndarray, Y: np.ndarray, w: np.ndarray,
                                  alpha: np.ndarray, b: np.ndarray, h: int) -> float:
        if Q_logits.shape[0] == 0: return 0.0
        z_full = np.einsum("nhk,h->nk", Q_logits, alpha) + b[None, :]
        p_full = _sigmoid(z_full)
        L_full = (_bce(Y, p_full) * w[:, None]).sum()
        ah = float(alpha[h])
        if ah <= 1e-12 or ah >= 1.0 - 1e-12: return 0.0
        z_minus_h = (np.einsum("nhk,h->nk", Q_logits, alpha) - ah * Q_logits[:, h, :]) / (1.0 - ah) + b[None, :]
        p_minus = _sigmoid(z_minus_h)
        L_minus = (_bce(Y, p_minus) * w[:, None]).sum()
        return max(0.0, float(L_minus - L_full))


def progressive_q_saliences(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price: np.ndarray,
    step: int,
    embargo: int,
    horizon_steps: int,
    sigma_minutes: int = 60,
    sample_every: int = 5,
    half_life_days: float = 5.0,
    samples_per_day: float = 1440.0,
    gating_classes: Iterable[int] = (0, 1, 3, 4),
) -> Dict[str, float]:
    X_flat_raw, hk2idx = hist
    X_flat = np.asarray(X_flat_raw, dtype=float)
    price = np.asarray(price, dtype=float)
    H = len(hk2idx)
    if H == 0: return {}
    T, HD = X_flat.shape
    if HD % H != 0: raise ValueError("X_flat second dim must be divisible by H")
    D = HD // H
    if D != 17: raise ValueError(f"Expected per-expert embedding D=17; got D={D}")

    y_all, valid_idx_all = _make_bins_from_price(price, horizon=horizon_steps, vol_window=max(7200, 10))
    len_r = max(0, price.shape[0] - horizon_steps)
    y_r = np.full(len_r, -1, dtype=int)
    if valid_idx_all.size > 0: y_r[valid_idx_all] = y_all

    len_r = max(0, price.shape[0] - horizon_steps)
    r_h = np.log(price[horizon_steps:] + _EPS) - np.log(price[:-horizon_steps] + _EPS)
    vol_window_q = max(MIN_REQUIRED_SAMPLES, 10)
    sigma_h_raw = _rolling_std_fast(r_h, vol_window_q)
    sigma_h = np.full(len_r, np.nan)
    if sigma_h_raw.size > 0: sigma_h[vol_window_q - 1:] = sigma_h_raw

    max_t_for_horizon = T - 1 - horizon_steps
    valid_times_mask = np.zeros(T, dtype=bool)
    valid_times_mask[1:max_t_for_horizon + 1] = True

    Q_SL_MAP = {0: (5, 8), 1: (8, 11), 3: (11, 14), 4: (14, 17)}
    contrib_sum = np.zeros(H, dtype=float)
    gating_set = set(int(c) for c in gating_classes)

    warmup = int(np.ceil((horizon_steps + max(1, int(vol_window_q)) + 1) / step))
    k = warmup
    while True:
        train_end = k * step
        if train_end >= T: break
        eval_start = train_end + embargo
        if eval_start >= T: break
        eval_end = min(eval_start + step, T)

        def collect_dir_data(classes: Iterable[int], t_lo: int, t_hi: int):
            sel_all = []
            cls_seq = []
            for c in classes:
                if c not in Q_SL_MAP: continue
                sel_c = [t for t in valid_idx_all
                         if (t >= t_lo and t < t_hi and valid_times_mask[t] and (t < sigma_h.shape[0]) and np.isfinite(sigma_h[t]) and (y_r[t] == c))]
                if len(sel_c) == 0: continue
                sel_all.append(np.array(sel_c, dtype=int))
                cls_seq.append(int(c))
            if len(sel_all) == 0:
                return (np.zeros((0, H, 3), dtype=float), np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float))
            sel = np.concatenate(sel_all, axis=0)
            cls_per_sample = np.concatenate([np.full(len(a), c, dtype=int) for a, c in zip(sel_all, cls_seq)], axis=0)
            
            base_logp = np.log(price[sel - 1] + _EPS)
            path_logp = np.log(price + _EPS)
            K = 3
            Y = np.zeros((sel.size, K), dtype=float)
            thr_mult = np.array([0.5, 1.0, 2.0], dtype=float)
            for i, t0 in enumerate(sel):
                seg = path_logp[t0: t0 + horizon_steps + 1] - base_logp[i]
                up = np.max(seg)
                dn = np.min(seg)
                thr = thr_mult * sigma_h[t0]
                c = int(cls_per_sample[i])
                if c in (3, 4):
                    Y[i, :] = (dn <= -thr).astype(float)
                elif c in (0, 1):
                    Y[i, :] = (up >=  thr).astype(float)

            Xr = X_flat[sel, :].reshape(sel.size, H, D)
            Qlog_list = []
            for i, c in enumerate(cls_per_sample):
                sl = Q_SL_MAP[int(c)]
                qprob_i = np.clip(Xr[i, :, sl[0]:sl[1]], _EPS, 1.0 - _EPS)
                Qlog_list.append(_logit(qprob_i))
            Qlog = np.stack(Qlog_list, axis=0)
            w = _exp_half_life_weights(sel, half_life_days, samples_per_day)
            return Qlog, Y, w

        Qp_tr = Yp_tr = wp_tr = None
        Qn_tr = Yn_tr = wn_tr = None
        pos_classes = [c for c in gating_set if c in (3, 4)]
        neg_classes = [c for c in gating_set if c in (0, 1)]
        
        if len(pos_classes) > 0: Qp_tr, Yp_tr, wp_tr = collect_dir_data(pos_classes, 0, train_end)
        if len(neg_classes) > 0: Qn_tr, Yn_tr, wn_tr = collect_dir_data(neg_classes, 0, train_end)
        
        if Qp_tr is None: Qp_tr, Yp_tr, wp_tr = (np.zeros((0, H, 3)), np.zeros((0, 3)), np.zeros(0))
        if Qn_tr is None: Qn_tr, Yn_tr, wn_tr = (np.zeros((0, H, 3)), np.zeros((0, 3)), np.zeros(0))

        Qcal = QPathCalibrator(H)
        Qcal.fit(Qp_tr, Yp_tr, wp_tr, Qn_tr, Yn_tr, wn_tr)

        if len(pos_classes) > 0: Qp_ev, Yp_ev, wp_ev = collect_dir_data(pos_classes, eval_start, eval_end)
        else: Qp_ev, Yp_ev, wp_ev = (np.zeros((0, H, 3)), np.zeros((0, 3)), np.zeros(0))
        if len(neg_classes) > 0: Qn_ev, Yn_ev, wn_ev = collect_dir_data(neg_classes, eval_start, eval_end)
        else: Qn_ev, Yn_ev, wn_ev = (np.zeros((0, H, 3)), np.zeros((0, 3)), np.zeros(0))

        sal_dict = Qcal.salience_on_eval(Qp_ev, Yp_ev, wp_ev, Qn_ev, Yn_ev, wn_ev, hk2idx)
        for hk, s in sal_dict.items():
            contrib_sum[hk2idx[hk]] += float(s)

        k += 1
        if eval_end >= T: break

    total = float(np.sum(contrib_sum))
    inv_map = {idx: hk for hk, idx in hk2idx.items()}
    if total > 0.0:
        return {inv_map[i]: float(contrib_sum[i] / total) for i in range(H)}
    else:
        return {inv_map[i]: 1.0 / H for i in range(H)}


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
    X_flat, hk2idx = hist
    price = np.asarray(price_data, dtype=float)
    if price.ndim != 1: return {}
    samples_per_day = int((24 * 60 * 60) // (12 * max(1, sample_every)))
    required = int(max(MIN_REQUIRED_SAMPLES, np.ceil(samples_per_day * min_days)))
    if price.size < required or X_flat.shape[0] < required:
        logger.info("Q path salience requires %d samples; only %d available.", required, price.size)
        return {}
    horizon_steps = max(1, int(round(blocks_ahead / max(1, sample_every))))
    step = samples_per_day
    embargo = max(60, horizon_steps)
    return progressive_q_saliences(
        hist=hist,
        price=price,
        step=step,
        embargo=embargo,
        horizon_steps=horizon_steps,
        sigma_minutes=sigma_minutes,
        sample_every=sample_every,
        half_life_days=half_life_days,
        samples_per_day=float(samples_per_day),
        gating_classes=gating_classes,
    )


def compute_hitfirst_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    blocks_ahead: int,
    sample_every: int,
    min_days: float = 5.0,
    half_life_days: float = 5.0,
) -> Dict[str, float]:
    X_flat, hk2idx = hist
    price = np.asarray(price_data, dtype=float)
    if price.ndim != 1:
        return {}
    if not isinstance(hk2idx, dict):
        return {}

    T, HD = X_flat.shape
    H = len(hk2idx)
    if H == 0 or HD % H != 0:
        return {}
    D = HD // H
    if D != 3:
        return {}

    samples_per_day = int((24 * 60 * 60) // (12 * max(1, sample_every)))
    required = int(max(MIN_REQUIRED_SAMPLES, np.ceil(samples_per_day * min_days)))
    if price.size < required or X_flat.shape[0] < required:
        logger.info("Hit-first salience requires %d samples; only %d available.", required, price.size)
        return {}

    horizon_steps = max(1, int(round(blocks_ahead / max(1, sample_every))))
    len_r = max(0, price.shape[0] - horizon_steps)
    if len_r <= 0:
        return {}

    r_h = np.log(price[horizon_steps:] + _EPS) - np.log(price[:-horizon_steps] + _EPS)
    vol_window = int(max(required, MIN_REQUIRED_SAMPLES))
    if r_h.size < vol_window:
        return {}
    sig_raw = _rolling_std_fast(r_h, vol_window)
    sig = np.full(len_r, np.nan)
    if sig_raw.size > 0:
        sig[vol_window - 1:] = sig_raw

    log_price = np.log(price + _EPS)
    labels = np.full(len_r, -1, dtype=int)
    for t in range(len_r):
        sigma_t = sig[t]
        if not np.isfinite(sigma_t) or sigma_t <= 0.0:
            continue
        base_lp = log_price[t]
        seg = log_price[t + 1 : t + 1 + horizon_steps] - base_lp
        idx_up_candidates = np.where(seg >= sigma_t)[0]
        idx_dn_candidates = np.where(seg <= -sigma_t)[0]
        has_up = idx_up_candidates.size > 0
        has_dn = idx_dn_candidates.size > 0
        if not has_up and not has_dn:
            labels[t] = 2
            continue
        if has_up:
            i_up = int(idx_up_candidates[0])
        else:
            i_up = horizon_steps + 1
        if has_dn:
            i_dn = int(idx_dn_candidates[0])
        else:
            i_dn = horizon_steps + 1
        if i_up < i_dn:
            labels[t] = 0
        elif i_dn < i_up:
            labels[t] = 1
        else:
            labels[t] = 2

    valid_mask = labels >= 0
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size < required:
        return {}

    inv_map = {idx: hk for hk, idx in hk2idx.items()}

    from sklearn.linear_model import LogisticRegression

    X_valid = X_flat[:len_r][valid_mask]
    X_valid = np.asarray(X_valid, dtype=float)
    X_valid = X_valid.reshape(X_valid.shape[0], H, D)

    probs = np.clip(X_valid, _EPS, 1.0 - _EPS)
    sums = probs.sum(axis=2, keepdims=True)
    sums[sums <= 0.0] = 1.0
    probs /= sums

    up_scores = _logit(probs[:, :, 0])
    down_scores = _logit(probs[:, :, 1])
    up_scores = np.nan_to_num(up_scores, nan=0.0, posinf=0.0, neginf=0.0)
    down_scores = np.nan_to_num(down_scores, nan=0.0, posinf=0.0, neginf=0.0)

    y_up = (labels[valid_mask] == 0).astype(int)
    y_dn = (labels[valid_mask] == 1).astype(int)

    n_samples = up_scores.shape[0]
    split = int(max(1, np.floor(0.7 * n_samples)))

    X_up_train = up_scores[:split]
    y_up_train = y_up[:split]
    X_up_test = up_scores[split:]
    y_up_test = y_up[split:]

    X_dn_train = down_scores[:split]
    y_dn_train = y_dn[:split]
    X_dn_test = down_scores[split:]
    y_dn_test = y_dn[split:]

    importance = np.zeros(H, dtype=float)

    if np.unique(y_up_train).size == 2 and X_up_train.shape[0] > 0:
        clf_up = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500,
            tol=1e-4,
        )
        clf_up.fit(X_up_train, y_up_train)
        coef_up = np.asarray(clf_up.coef_, dtype=float).ravel()
        scale_up = np.mean(np.abs(X_up_test), axis=0) if X_up_test.shape[0] > 0 else np.ones(H, dtype=float)
        importance += np.abs(coef_up) * scale_up

    if np.unique(y_dn_train).size == 2 and X_dn_train.shape[0] > 0:
        clf_dn = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500,
            tol=1e-4,
        )
        clf_dn.fit(X_dn_train, y_dn_train)
        coef_dn = np.asarray(clf_dn.coef_, dtype=float).ravel()
        scale_dn = np.mean(np.abs(X_dn_test), axis=0) if X_dn_test.shape[0] > 0 else np.ones(H, dtype=float)
        importance += np.abs(coef_dn) * scale_dn

    total = float(np.sum(importance))
    if total <= 0.0:
        return {}

    sal: Dict[str, float] = {}
    for idx in range(H):
        hk = inv_map.get(idx)
        if hk is not None:
            sal[hk] = float(importance[idx] / total)
    return sal



