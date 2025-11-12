 

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable
import numpy as np

logger = logging.getLogger(__name__)
__all__ = (
    "LBFGSConfig", "LBFGSOptimizer", "LBFGSLogOPModel",
    "progressive_saliences", "compute_lbfgs_salience",
    "QCalibConfig", "QPathCalibrator",
    "progressive_q_saliences", "compute_q_path_salience",
)

_LN2, _EPS = np.log(2.0), 1e-12
MIN_REQUIRED_SAMPLES = 7200


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / np.sum(ex, axis=1, keepdims=True)

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log(1.0 - p)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

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
    """
    Build 5-class labels using end-point z-score at horizon:
      y=0 (z<=-2), y=1 (-2<z<-1), y=2 (-1<=z<=1), y=3 (1<z<2), y=4 (z>=2)
    sigma is rolling std of HORIZON log returns over vol_window.
    Returns: labels y (len ~ T - horizon), valid indices into original time (absolute).
    """
    price = np.asarray(price, dtype=float)
    if price.ndim != 1:
        raise ValueError("price_data must be 1-D array")
    T = price.shape[0]
    if T <= horizon + vol_window:
        raise ValueError("Not enough data: need > horizon + vol_window samples")

    r = np.log(price[horizon:] + eps) - np.log(price[:-horizon] + eps)  # horizon log return
    sig_raw = _rolling_std_fast(r, vol_window)                          # length len(r) - vol_window + 1
    sig = np.full(len(r), np.nan)
    if sig_raw.size <= 0:
        raise ValueError("No valid labels after rolling sigma computation")
    # Align at end of window (same convention as other rolling sigmas)
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

def _exp_half_life_weights(valid_idx: np.ndarray, half_life_days: float, samples_per_day: float) -> np.ndarray:
    if valid_idx.size == 0:
        return np.ones(0, dtype=float)
    i_max = float(valid_idx.max())
    age_days = (i_max - valid_idx.astype(float)) / float(samples_per_day)
    w = np.exp(-_LN2 * (age_days / float(half_life_days)))
    return w * (valid_idx.size / np.sum(w))

def _window_weights(t_start_abs: int, length: int, half_life_days: float, samples_per_day: float) -> np.ndarray:
    if length <= 0:
        return np.ones(0, dtype=float)
    return _exp_half_life_weights(t_start_abs + np.arange(length), half_life_days, samples_per_day)

def _zero_salience(hk2idx: Dict[str, int]) -> Dict[str, float]:
    return {hk: 0.0 for hk in hk2idx.keys()}

def _rolling_sigma_steps(price: np.ndarray, window_steps: int) -> np.ndarray:
    """
    σ computed from 1-step log returns over a backward window of `window_steps`.
    Aligned at END of the window: sigma[t] uses returns up to t-1.
    Returns NaN for t < window_steps.
    """
    price = np.asarray(price, dtype=float)
    r1 = np.log(price[1:] + _EPS) - np.log(price[:-1] + _EPS)
    sig_raw = _rolling_std_fast(r1, window_steps)  # length T - window_steps
    sig = np.full(price.shape[0], np.nan)
    sig[window_steps:] = sig_raw
    return sig

def _project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex (Duchi et al. 2008).
    """
    if v.ndim != 1:
        v = v.ravel()
    n = v.size
    if n == 0:
        return v
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


@dataclass
class LBFGSConfig:
    max_iter: int = 200
    m_hist: int = 10
    tol_grad: float = 1e-6
    armijo_c1: float = 1e-4
    step_init: float = 1.0
    step_min: float = 1e-10
    backtrack: float = 0.5
    verbose: bool = False

class LBFGSOptimizer:
    def __init__(self, cfg: LBFGSConfig): self.cfg = cfg

    def minimize(self, f_grad, x0: np.ndarray):
        x = x0.copy()
        s_hist, y_hist, rho_hist = [], [], []
        f_prev, g_prev = f_grad(x)
        for it in range(self.cfg.max_iter):
            q = g_prev.copy()
            alpha = []
            for i in range(len(s_hist) - 1, -1, -1):
                a_i = rho_hist[i] * np.dot(s_hist[i], q)
                alpha.append(a_i)
                q = q - a_i * y_hist[i]
            if len(y_hist) > 0:
                ys = float(np.dot(y_hist[-1], s_hist[-1]))
                yy = float(np.dot(y_hist[-1], y_hist[-1]))
                H_scale = ys / (yy + 1e-12)
            else:
                H_scale = 1.0
            r = H_scale * q
            for i in range(len(s_hist)):
                b_i = rho_hist[i] * np.dot(y_hist[i], r)
                a_i = alpha[len(s_hist) - 1 - i]
                r = r + s_hist[i] * (a_i - b_i)
            d = -r
            step = self.cfg.step_init
            f0 = f_prev
            gTd = float(np.dot(g_prev, d))
            if gTd >= 0:
                d = -g_prev
                gTd = -float(np.dot(g_prev, g_prev))
            accepted = False
            while step >= self.cfg.step_min:
                x_new = x + step * d
                f_new, g_new = f_grad(x_new)
                if f_new <= f0 + self.cfg.armijo_c1 * step * gTd:
                    accepted = True
                    break
                step *= self.cfg.backtrack
            if not accepted:
                break
            s = x_new - x
            yv = g_new - g_prev
            ys = float(np.dot(yv, s))
            if ys > 1e-12:
                if len(s_hist) == self.cfg.m_hist:
                    s_hist.pop(0); y_hist.pop(0); rho_hist.pop(0)
                s_hist.append(s); y_hist.append(yv); rho_hist.append(1.0 / ys)
            x, f_prev, g_prev = x_new, f_new, g_new
            if np.linalg.norm(g_prev) < self.cfg.tol_grad:
                break
        return x, f_prev, {"n_iter": it + 1, "grad_norm": float(np.linalg.norm(g_prev)), "f_val": float(f_prev)}


class LBFGSLogOPModel:
    """
    5-class ordinal classifier using ONLY per-expert p[0..4] probabilities.
    No Q features are used in classification.

    Params learned (17 total elements per expert input, but only small global param set):
      - b_free (3 scalars) -> symmetric class biases b = [b2, b1, b0, b1, b2]
      - a_p (scalar)       -> scale for pooled log-probabilities
      - a_pi (scalar)      -> scale for class priors in logits

    Expert "skill" vector alpha_p (H,) is computed analytically from weighted NLL per expert.
    """
    _P_SL = slice(0, 5)

    def __init__(self, l2_reg: float = 1e-3, lbfgs: Optional[LBFGSConfig] = None, ema_half_life_days: float = 0.0):
        self.l2_reg = float(l2_reg)
        self.cfg = lbfgs if lbfgs is not None else LBFGSConfig()
        self.b_free: Optional[np.ndarray] = None
        self.a_p: Optional[float] = None
        self.a_pi: Optional[float] = None
        self.alpha_p: Optional[np.ndarray] = None
        self.pi: Optional[np.ndarray] = None
        self.hotkey_starts: Optional[np.ndarray] = None
        self.shape_info: Dict[str, int] = {}
        self._train_half_life_days = 10.0
        self._train_samples_per_day = 1440.0
        self._ema_half_life_days = float(ema_half_life_days)

    @staticmethod
    def _unpack_hist(hist: Tuple[np.ndarray, Dict[str, int]]):
        X_flat, hk2idx = hist
        X_flat = np.asarray(X_flat, dtype=float)
        if X_flat.ndim != 2:
            raise ValueError("X_flat must be 2-D (T, H*D)")
        if not isinstance(hk2idx, dict) or len(hk2idx) == 0:
            raise ValueError("hk2idx must be a non-empty dict")
        return X_flat, hk2idx

    @staticmethod
    def _ema_past_only(arr: np.ndarray, alpha: float) -> np.ndarray:
        if not (alpha > 0.0):
            return arr
        out = np.empty_like(arr)
        out[0] = arr[0]
        for t in range(1, arr.shape[0]):
            out[t] = alpha * arr[t] + (1.0 - alpha) * out[t - 1]
        return out

    def _build_dataset(self, X_flat: np.ndarray, price_data: np.ndarray, hk2idx: Dict[str, int],
                       horizon: int, vol_window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
        H = len(hk2idx); T, HD = X_flat.shape
        if H == 0 or HD % H != 0:
            raise ValueError(f"X_flat second dim ({HD}) must be divisible by number of hotkeys ({H})")
        D = HD // H
        if D != 17:
            raise ValueError(f"Expected per-expert embedding D=17; got D={D}")
        starts = _compute_hotkey_start_indices(X_flat, H, D)
        y, valid_idx = _make_bins_from_price(price_data, horizon=horizon, vol_window=vol_window)
        return X_flat[valid_idx, :], y, valid_idx, H, D, starts

    def _reshape(self, X_flat: np.ndarray, H: int, D: int) -> np.ndarray:
        return X_flat.reshape(X_flat.shape[0], H, D)

    def _maybe_ema_embeddings(self, X_hhd: np.ndarray, samples_per_day: float) -> np.ndarray:
        if self._ema_half_life_days <= 0.0:
            return X_hhd
        alpha = 1.0 - np.exp(-_LN2 * (1.0 / (self._ema_half_life_days * samples_per_day)))
        return self._ema_past_only(X_hhd, alpha)

    @staticmethod
    def _skill_weights_from_probs(X_hhd: np.ndarray, y: np.ndarray, w_row: np.ndarray) -> np.ndarray:
        """Compute expert skill weights α_p from per-expert negative log-likelihoods."""
        N, H, D = X_hhd.shape
        p = np.clip(X_hhd[:, :, 0:5], _EPS, 1.0 - _EPS)
        active = np.any(X_hhd[:, :, 0:5] != 0.0, axis=2)
        ll = np.full(H, np.inf, dtype=float)
        for h in range(H):
            m = active[:, h]
            if not np.any(m):
                continue
            nll_t = -np.log(p[m, h, y[m]])
            w_eff = w_row[m]
            s = np.sum(w_eff)
            if s <= 0:
                continue
            ll[h] = float(np.sum(w_eff * nll_t) / s)
        finite = np.isfinite(ll)
        if not np.any(finite):
            return np.ones(H, dtype=float) / float(H)
        z = ll[finite] - np.min(ll[finite])
        skills = np.zeros(H, dtype=float)
        skills[finite] = np.exp(-z)
        ssum = np.sum(skills)
        if ssum <= 0:
            return np.ones(H, dtype=float) / float(H)
        return skills / ssum

    def _p_pool_log(self, X_hhd: np.ndarray, alpha_p: np.ndarray) -> np.ndarray:
        p = np.clip(X_hhd[:, :, self._P_SL], _EPS, 1.0 - _EPS)
        logp = np.log(p)
        return np.einsum("h,nhk->nk", alpha_p, logp)

    @staticmethod
    def _b_from_free(b_free: np.ndarray) -> np.ndarray:
        b0, b1, b2 = float(b_free[0]), float(b_free[1]), float(b_free[2])
        return np.array([b2, b1, b0, b1, b2], dtype=float)

    def fit(self, hist: Tuple[np.ndarray, Dict[str, int]], price_data: np.ndarray, horizon: int = 1,
            vol_window: int = 7200, class_prior_smoothing: float = 1.0, init_scale: float = 0.0,
            half_life_days: float = 10.0, samples_per_day: float = 1440.0, use_class_weights: bool = True) -> dict:
        """
        Train the 5-bucket classifier (NO Q usage).
        """
        X_flat, hk2idx = self._unpack_hist(hist)
        X_used, y_used, valid_idx, H, D, starts = self._build_dataset(X_flat, price_data, hk2idx, horizon, vol_window)
        N_eff, K = X_used.shape[0], 5

        counts = np.bincount(y_used, minlength=K).astype(float) + class_prior_smoothing
        pi = counts / counts.sum()
        log_pi = np.log(np.clip(pi, _EPS, None))
        self.pi = pi

        w_time = _exp_half_life_weights(valid_idx, half_life_days, samples_per_day)
        if use_class_weights:
            class_weights = np.zeros(K, dtype=float)
            for k in range(K):
                class_weights[k] = float(N_eff) / (K * counts[k]) if counts[k] > 0 else 1.0
            w_combined = w_time * class_weights[y_used]
        else:
            class_weights = None
            w_combined = w_time

        X_hhd = self._reshape(X_used, H, D)
        X_hhd = self._maybe_ema_embeddings(X_hhd, samples_per_day)

        alpha_p = self._skill_weights_from_probs(X_hhd, y_used, w_combined)
        self.alpha_p = alpha_p

        b_free = np.zeros(3, dtype=float)
        a_p = np.array(1.0 if init_scale == 0.0 else float(init_scale), dtype=float)
        a_pi = np.array(1.0, dtype=float)

        def pack_params(b_free, a_p, a_pi):
            return np.concatenate([b_free.ravel(), np.array([a_p], dtype=float), np.array([a_pi], dtype=float)])

        def unpack_params(theta):
            b = theta[0:3]
            ap = float(theta[3])
            api = float(theta[4])
            return b, ap, api

        l2 = self.l2_reg
        P_pool_log = self._p_pool_log(X_hhd, alpha_p)
        Y_onehot = np.eye(K)[y_used]

        def f_grad(theta: np.ndarray):
            b_free_, a_p_, a_pi_ = unpack_params(theta)
            logits = a_pi_ * log_pi[None, :] + self._b_from_free(b_free_)[None, :] + a_p_ * P_pool_log
            P = _softmax_rows(logits)
            ll = np.log(np.clip(P[np.arange(N_eff), y_used], _EPS, None))
            nll = -np.sum(w_combined * ll)
            reg = 0.5 * l2 * (a_p_ * a_p_ + a_pi_ * a_pi_)

            diff = (P - Y_onehot) * w_combined[:, None]
            d_b0 = np.sum(diff[:, 2])
            d_b1 = np.sum(diff[:, 1] + diff[:, 3])
            d_b2 = np.sum(diff[:, 0] + diff[:, 4])
            d_b = np.array([d_b0, d_b1, d_b2], dtype=float)

            d_ap = float(np.sum(diff * P_pool_log)) + l2 * a_p_
            d_api = float(np.sum(diff * log_pi[None, :])) + l2 * a_pi_

            grad = np.concatenate([d_b.ravel(), np.array([d_ap], dtype=float), np.array([d_api], dtype=float)])
            return float(nll + reg), grad

        theta0 = pack_params(b_free, a_p, a_pi)
        opt = LBFGSOptimizer(self.cfg)
        theta_star, f_val, info = opt.minimize(f_grad, theta0)

        b_free_star, a_p_star, a_pi_star = unpack_params(theta_star)
        self.b_free = b_free_star.copy()
        self.a_p = float(a_p_star)
        self.a_pi = float(a_pi_star)
        self.hotkey_starts = starts
        self.shape_info = {"N_eff": int(N_eff), "K": int(K), "H": int(H), "D": int(D)}
        self._train_half_life_days = float(half_life_days)
        self._train_samples_per_day = float(samples_per_day)

        fit_info = {
            "f_val": float(f_val),
            **info,
            "priors": pi.copy(),
            "mean_time_weight": float(w_time.mean()),
            "alpha_p": self.alpha_p.copy(),
            "a_p": float(self.a_p),
            "a_pi": float(self.a_pi),
        }
        if use_class_weights:
            fit_info["class_weights"] = class_weights.copy()
            fit_info["mean_combined_weight"] = float(np.mean(w_combined))
        return fit_info

    def _forward_logits(self, X_window: np.ndarray) -> np.ndarray:
        if any(v is None for v in [self.b_free, self.a_p, self.a_pi, self.pi, self.alpha_p]):
            raise RuntimeError("Model is not fit yet.")
        H, D = self.shape_info["H"], self.shape_info["D"]
        if X_window.shape[1] != H * D:
            raise ValueError(f"X_window has wrong second dim; expected {H*D}, got {X_window.shape[1]}")
        log_pi = np.log(np.clip(self.pi, _EPS, None))
        X_hhd = self._reshape(X_window, H, D)
        P_pool_log = self._p_pool_log(X_hhd, self.alpha_p)
        logits = self.a_pi * log_pi[None, :] + self._b_from_free(self.b_free)[None, :] + self.a_p * P_pool_log
        return logits

    def predict_proba_on(self, X_window: np.ndarray) -> np.ndarray:
        return _softmax_rows(self._forward_logits(X_window))

    def contributions_matrix(self, X_window: np.ndarray, hk2idx: Dict[str, int], t_start_abs: Optional[int] = None,
                             hotkey_starts_global: Optional[np.ndarray] = None, sample_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Per-row, per-hotkey contribution proxy from p-only pooling:
        c[n,h] = |a_p| * α_p[h] * Σ_k P[n,k] * |log p_{n,h,k} - pooled_logp_{n,k}|
        (optionally masked by activation starts and weighted by sample_weights)
        """
        if any(v is None for v in [self.alpha_p, self.a_p, self.pi]):
            raise RuntimeError("Model is not fit yet.")
        H = len(hk2idx); N, HD = X_window.shape; D = HD // H
        if HD % H != 0:
            raise ValueError("X_window second dim not divisible by H")
        if D != 17:
            raise ValueError(f"Expected per-expert embedding D=17; got D={D}")

        X_hhd = self._reshape(X_window, H, D)
        p = np.clip(X_hhd[:, :, self._P_SL], _EPS, 1.0 - _EPS)
        logp = np.log(p)
        pooled = np.einsum("h,nhk->nk", self.alpha_p, logp)
        P = self.predict_proba_on(X_window)

        dev = np.abs(logp - pooled[:, None, :])
        c = np.einsum("n,nhk,nk->nh",
                      np.full(N, abs(self.a_p), dtype=float),
                      dev,
                      P) * self.alpha_p[None, :]

        # Mask for experts that haven't started yet
        if (t_start_abs is not None) and (hotkey_starts_global is not None):
            for h in range(H):
                start_h = int(hotkey_starts_global[h])
                if start_h > t_start_abs:
                    cut = min(N, max(0, start_h - t_start_abs))
                    if cut > 0:
                        c[:cut, h] = 0.0

        if sample_weights is not None:
            w_row = np.asarray(sample_weights, dtype=float)
            if len(w_row) != N:
                raise ValueError("sample_weights length must match number of rows in X_window")
            c *= w_row[:, None]
        return c


def progressive_saliences(hist: Tuple[np.ndarray, Dict[str, int]], price_data: np.ndarray, step: int = 1440,
                          embargo: int = 60, horizon: int = 1, vol_window: int = 7200, class_prior_smoothing: float = 1.0,
                          l2_reg: float = 1e-3, init_scale: float = 0.0, lbfgs_cfg: Optional[LBFGSConfig] = None,
                          half_life_days: float = 10.0, samples_per_day: float = 1440.0, use_class_weights: bool = True) -> Dict[str, float]:
    """Out-of-sample expert salience based on p-only classifier contributions."""
    X_flat_raw, hk2idx = hist
    X_flat = np.asarray(X_flat_raw, dtype=float)
    price = np.asarray(price_data, dtype=float)
    T, HD = X_flat.shape
    H = len(hk2idx)
    if H == 0 or HD % H != 0:
        raise ValueError("X_flat second dim must be divisible by number of hotkeys")
    if T < MIN_REQUIRED_SAMPLES:
        return {}  # Signal insufficient samples
    D = HD // H
    if D != 17:
        raise ValueError(f"Expected per-expert embedding D=17; got D={D}")

    hotkey_starts = _compute_hotkey_start_indices(X_flat, H, D)
    min_train = horizon + vol_window + 1
    if min_train >= T:
        return {}  # Signal insufficient samples

    k = int(np.ceil(min_train / step))
    salience_exp_accum = np.zeros(H, dtype=float)

    while True:
        train_end = k * step
        if train_end >= T:
            break
        eval_start = train_end + embargo
        if eval_start >= T:
            break
        eval_end = min(eval_start + step, T)

        model = LBFGSLogOPModel(l2_reg=l2_reg, lbfgs=lbfgs_cfg)
        try:
            model.fit(hist=(X_flat[:train_end, :], hk2idx),
                      price_data=price[:train_end],
                      horizon=horizon,
                      vol_window=vol_window,
                      class_prior_smoothing=class_prior_smoothing,
                      init_scale=init_scale,
                      half_life_days=half_life_days,
                      samples_per_day=samples_per_day,
                      use_class_weights=use_class_weights)
        except Exception as exc:
            logger.exception("Classifier fit skipped at k=%d: %s", k, exc)
            k += 1
            continue

        w_eval = _window_weights(eval_start, eval_end - eval_start, half_life_days, samples_per_day)
        contribs = model.contributions_matrix(X_flat[eval_start:eval_end, :], hk2idx, eval_start, hotkey_starts, w_eval)
        salience_exp_accum += contribs.sum(axis=0)

        k += 1
        if eval_end >= T:
            break

    inv_map = {idx: hk for hk, idx in hk2idx.items()}
    out: Dict[str, float] = {}
    exp_sum = float(np.sum(salience_exp_accum))
    if exp_sum > 0.0:
        for idx in range(H):
            out[inv_map[idx]] = float(salience_exp_accum[idx] / exp_sum)
    else:
        out = {}    # Signal insufficient samples
    return out


def compute_lbfgs_salience(hist: Tuple[np.ndarray, Dict[str, int]], price_data: np.ndarray, blocks_ahead: int,
                           sample_every: int, lbfgs_cfg: Optional[LBFGSConfig] = None, min_days: float = 5.0,
                           use_class_weights: bool = True) -> Dict[str, float]:
    if not isinstance(hist, tuple) or len(hist) != 2:
        logger.debug("LBFGS history payload malformed")
        return {}
    _hist_matrix, hk2idx = hist
    if not isinstance(hk2idx, dict):
        logger.debug("LBFGS history mapping malformed")
        return {}
    price_arr = np.asarray(price_data, dtype=float)
    if price_arr.ndim != 1:
        logger.debug("LBFGS price array has wrong dimensionality")
        return {}

    samples_per_day = int((24 * 60 * 60) // (12 * max(1, sample_every)))
    required = int(max(MIN_REQUIRED_SAMPLES, np.ceil(samples_per_day * min_days)))
    if price_arr.size < required:
        logger.info("LBFGS salience requires %d samples (%.1f days); only %d available.",
                    required, min_days, price_arr.size)
        # Return empty dict instead of uniform weights
        # Let upstream handle the empty case
        return {}

    horizon_steps = max(1, int(round(blocks_ahead / max(1, sample_every))))
    vol_window = max(required, MIN_REQUIRED_SAMPLES)
    try:
        sal = progressive_saliences(
            hist, price_arr,
            step=samples_per_day,
            embargo=max(60, horizon_steps),
            horizon=horizon_steps,
            vol_window=vol_window,
            class_prior_smoothing=1.0,
            l2_reg=1e-3,
            init_scale=0.0,
            lbfgs_cfg=lbfgs_cfg,
            half_life_days=min_days,
            samples_per_day=float(samples_per_day),
            use_class_weights=use_class_weights
        )
    except Exception as exc:
        logger.exception("LBFGS salience computation failed: %s", exc)
        # Return empty dict instead of uniform weights
        # Let upstream handle the empty case
        return {}
    return {hk: float(max(0.0, score)) for hk, score in sal.items()}


@dataclass
class QCalibConfig:
    max_iter: int = 200
    step_init: float = 1.0
    step_min: float = 1e-6
    backtrack: float = 0.5
    tol_grad: float = 1e-6
    l2_alpha: float = 0.0   # optional Tikhonov on alpha
    verbose: bool = False
    class_weighting: bool = True  # enable per-threshold pos/neg weighting

class QPathCalibrator:
    """Logistic stacking on the simplex for path opposite-move hits."""
    def __init__(self, H: int, cfg: Optional[QCalibConfig] = None):
        self.H = int(H)
        self.cfg = cfg if cfg is not None else QCalibConfig()
        # learned params
        self.alpha_pos: Optional[np.ndarray] = None  # (H,)
        self.b_pos: Optional[np.ndarray] = None      # (3,)
        self.alpha_neg: Optional[np.ndarray] = None
        self.b_neg: Optional[np.ndarray] = None

    def _fit_one_dir(self, Q_logits: np.ndarray, Y: np.ndarray, w: np.ndarray):
        """
        Q_logits: (N, H, 3)  per-threshold logits from experts
        Y:        (N, 3)     binary labels (opposite-move hits)
        w:        (N,)       nonnegative weights
        Returns (alpha, b, info)
        """
        N, H, K = Q_logits.shape
        assert H == self.H and K == 3
        if N == 0:
            return np.full(H, 1.0 / H, dtype=float), np.zeros(3, dtype=float), {"n_iter": 0, "loss": float("nan")}

        alpha = np.full(H, 1.0 / H, dtype=float)
        b = np.zeros(3, dtype=float)

        def loss_grad(alpha_in: np.ndarray, b_in: np.ndarray):
            z_agg = np.einsum("nhk,h->nk", Q_logits, alpha_in)  # (N, K)
            z = z_agg + b_in[None, :]                           # (N, K)
            p = _sigmoid(z)
            # per-threshold class weights to counter imbalance
            if self.cfg.class_weighting:
                # compute weighted class counts per threshold using w
                pos_w = np.sum(Y * w[:, None], axis=0)
                neg_w = np.sum((1.0 - Y) * w[:, None], axis=0)
                # avoid zero-division; emphasize rare class
                w_pos = neg_w / np.maximum(pos_w, _EPS)
                w_neg = np.ones_like(w_pos)
                # loss
                L_mat = (-(w_pos[None, :] * Y * np.log(np.clip(p, _EPS, 1.0))
                           + w_neg[None, :] * (1.0 - Y) * np.log(np.clip(1.0 - p, _EPS, 1.0)))) * w[:, None]
                loss = float(L_mat.sum() + 0.5 * self.cfg.l2_alpha * np.dot(alpha_in, alpha_in))
                # grads
                class_mask = w_pos[None, :] * Y + w_neg[None, :] * (1.0 - Y)
                diff = (p - Y) * class_mask * w[:, None]
            else:
                L_mat = _bce(Y, p) * w[:, None]
                loss = float(L_mat.sum() + 0.5 * self.cfg.l2_alpha * np.dot(alpha_in, alpha_in))
                diff = (p - Y) * w[:, None]
            g_alpha = np.einsum("nk,nhk->h", diff, Q_logits) + self.cfg.l2_alpha * alpha_in  # (H,)
            g_b = np.sum(diff, axis=0)                          # (3,)
            return loss, g_alpha, g_b

        eta = self.cfg.step_init
        for it in range(self.cfg.max_iter):
            L0, g_alpha, g_b = loss_grad(alpha, b)
            g_norm = float(np.linalg.norm(g_alpha, ord=2))
            if g_norm < self.cfg.tol_grad:
                return alpha, b, {"n_iter": it, "loss": L0, "grad_norm": g_norm}
            # backtracking projected gradient
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

    @staticmethod
    def _delta_loss_remove_hotkey(Q_logits: np.ndarray, Y: np.ndarray, w: np.ndarray,
                                  alpha: np.ndarray, b: np.ndarray, h: int) -> float:
        """Marginal OOS loss improvement when removing hotkey h and renormalizing α."""
        if Q_logits.shape[0] == 0:
            return 0.0
        N, H, K = Q_logits.shape
        z_full = np.einsum("nhk,h->nk", Q_logits, alpha) + b[None, :]
        p_full = _sigmoid(z_full)
        L_full = (_bce(Y, p_full) * w[:, None]).sum()

        ah = float(alpha[h])
        if ah <= 1e-12 or ah >= 1.0 - 1e-12:
            return 0.0

        z_minus_h = (np.einsum("nhk,h->nk", Q_logits, alpha) - ah * Q_logits[:, h, :]) / (1.0 - ah) + b[None, :]
        p_minus = _sigmoid(z_minus_h)
        L_minus = (_bce(Y, p_minus) * w[:, None]).sum()
        delta = float(L_minus - L_full)
        return max(0.0, delta)

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
            return {}  # Signal zero contribution


def progressive_q_saliences(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price: np.ndarray,
    step: int,                 # samples_per_day (with 12× scheme upstream)
    embargo: int,              # in bars (samples)
    horizon_steps: int,        # e.g., 60 for a 60-minute horizon when sample_every=5
    sigma_minutes: int = 60,   # rolling sigma lookback in minutes
    sample_every: int = 5,     # blocks per sample (5 -> minutely at 12s blocks)
    half_life_days: float = 10.0,
    samples_per_day: float = 1440.0,
    gating_classes: Iterable[int] = (0, 1, 3, 4),  # include ±1σ and ±2σ by default
) -> Dict[str, float]:
    """Out-of-sample Q salience based only on Q path opposite-move predictions."""
    X_flat_raw, hk2idx = hist
    X_flat = np.asarray(X_flat_raw, dtype=float)
    price = np.asarray(price, dtype=float)
    H = len(hk2idx)
    if H == 0:
        return {}
    T, HD = X_flat.shape
    if HD % H != 0:
        raise ValueError("X_flat second dim must be divisible by H")
    D = HD // H
    if D != 17:
        raise ValueError(f"Expected per-expert embedding D=17; got D={D}")

    y_all, valid_idx_all = _make_bins_from_price(price, horizon=horizon_steps, vol_window=max(7200, 10))
    len_r = max(0, price.shape[0] - horizon_steps)
    y_r = np.full(len_r, -1, dtype=int)
    if valid_idx_all.size > 0:
        y_r[valid_idx_all] = y_all

    # Horizon-based sigma: rolling std of horizon log returns over a multi-day window
    len_r = max(0, price.shape[0] - horizon_steps)
    r_h = np.log(price[horizon_steps:] + _EPS) - np.log(price[:-horizon_steps] + _EPS)
    vol_window_q = max(MIN_REQUIRED_SAMPLES, 10)
    sigma_h_raw = _rolling_std_fast(r_h, vol_window_q)
    sigma_h = np.full(len_r, np.nan)
    if sigma_h_raw.size > 0:
        sigma_h[vol_window_q - 1:] = sigma_h_raw

    max_t_for_horizon = T - 1 - horizon_steps
    valid_times_mask = np.zeros(T, dtype=bool)
    valid_times_mask[1:max_t_for_horizon + 1] = True  # need baseline at t-1 and full horizon window

    # Per-expert layout: [0:5]=p, [5:8]=Q(c=0), [8:11]=Q(c=1), [11:14]=Q(c=3), [14:17]=Q(c=4)
    Q_SL_MAP = {0: (5, 8), 1: (8, 11), 3: (11, 14), 4: (14, 17)}

    contrib_sum = np.zeros(H, dtype=float)
    gating_set = set(int(c) for c in gating_classes)

    warmup = int(np.ceil((horizon_steps + max(1, int(vol_window_q)) + 1) / step))
    k = warmup
    while True:
        train_end = k * step
        if train_end >= T:
            break
        eval_start = train_end + embargo
        if eval_start >= T:
            break
        eval_end = min(eval_start + step, T)

        def collect_dir_data(classes: Iterable[int], t_lo: int, t_hi: int):
            sel_all = []
            cls_seq = []
            for c in classes:
                if c not in Q_SL_MAP:
                    continue
                sel_c = [t for t in valid_idx_all
                         if (t >= t_lo and t < t_hi and valid_times_mask[t] and (t < sigma_h.shape[0]) and np.isfinite(sigma_h[t]) and (y_r[t] == c))]
                if len(sel_c) == 0:
                    continue
                sel_all.append(np.array(sel_c, dtype=int))
                cls_seq.append(int(c))
            if len(sel_all) == 0:
                return (np.zeros((0, H, 3), dtype=float), np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float))
            sel = np.concatenate(sel_all, axis=0)
            cls_per_sample = np.concatenate([np.full(len(a), c, dtype=int) for a, c in zip(sel_all, cls_seq)], axis=0)

            base_logp = np.log(price[sel - 1] + _EPS)         # baseline at t-1
            path_logp = np.log(price + _EPS)                  # full path

            K = 3
            Y = np.zeros((sel.size, K), dtype=float)
            thr_mult = np.array([0.5, 1.0, 2.0], dtype=float)
            for i, t0 in enumerate(sel):
                seg = path_logp[t0: t0 + horizon_steps + 1] - base_logp[i]
                up = np.max(seg)
                dn = np.min(seg)
                thr = thr_mult * sigma_h[t0]
                c = int(cls_per_sample[i])
                if c in (3, 4):   # positive buckets: opposite is DOWN
                    Y[i, :] = (dn <= -thr).astype(float)
                elif c in (0, 1): # negative buckets: opposite is UP
                    Y[i, :] = (up >=  thr).astype(float)

            Xr = X_flat[sel, :].reshape(sel.size, H, D)
            Qlog_list = []
            for i, c in enumerate(cls_per_sample):
                sl = Q_SL_MAP[int(c)]
                qprob_i = np.clip(Xr[i, :, sl[0]:sl[1]], _EPS, 1.0 - _EPS)
                Qlog_list.append(_logit(qprob_i))
            Qlog = np.stack(Qlog_list, axis=0)  # (N, H, 3)

            w = _exp_half_life_weights(sel, half_life_days, samples_per_day)
            return Qlog, Y, w

        # Build TRAIN (0..train_end) for requested directions
        Qp_tr = Yp_tr = wp_tr = None
        Qn_tr = Yn_tr = wn_tr = None
        pos_classes = [c for c in gating_set if c in (3, 4)]
        neg_classes = [c for c in gating_set if c in (0, 1)]
        if len(pos_classes) > 0:
            Qp_tr, Yp_tr, wp_tr = collect_dir_data(pos_classes, 0, train_end)
        if len(neg_classes) > 0:
            Qn_tr, Yn_tr, wn_tr = collect_dir_data(neg_classes, 0, train_end)
        if Qp_tr is None:
            Qp_tr, Yp_tr, wp_tr = (np.zeros((0, H, 3)), np.zeros((0, 3)), np.zeros(0))
        if Qn_tr is None:
            Qn_tr, Yn_tr, wn_tr = (np.zeros((0, H, 3)), np.zeros((0, 3)), np.zeros(0))

        # Fit convex Q calibrator
        Qcal = QPathCalibrator(H)
        Qcal.fit(Qp_tr, Yp_tr, wp_tr, Qn_tr, Yn_tr, wn_tr)

        # EVAL (eval_start .. eval_end)
        if len(pos_classes) > 0:
            Qp_ev, Yp_ev, wp_ev = collect_dir_data(pos_classes, eval_start, eval_end)
        else:
            Qp_ev, Yp_ev, wp_ev = (np.zeros((0, H, 3)), np.zeros((0, 3)), np.zeros(0))
        if len(neg_classes) > 0:
            Qn_ev, Yn_ev, wn_ev = collect_dir_data(neg_classes, eval_start, eval_end)
        else:
            Qn_ev, Yn_ev, wn_ev = (np.zeros((0, H, 3)), np.zeros((0, 3)), np.zeros(0))

        # Accumulate Q salience via marginal loss improvements
        sal_dict = Qcal.salience_on_eval(Qp_ev, Yp_ev, wp_ev, Qn_ev, Yn_ev, wn_ev, hk2idx)
        for hk, s in sal_dict.items():
            contrib_sum[hk2idx[hk]] += float(s)  # unnormalized per-window salience

        k += 1
        if eval_end >= T:
            break

    total = float(np.sum(contrib_sum))
    inv_map = {idx: hk for hk, idx in hk2idx.items()}
    if total > 0.0:
        return {inv_map[i]: float(contrib_sum[i] / total) for i in range(H)}
    else:
        return {inv_map[i]: 1.0 / H for i in range(H)}


def compute_q_path_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    price_data: np.ndarray,
    blocks_ahead: int,          # e.g., 300 blocks == 60 minutes
    sample_every: int,          # e.g., 5 for minutely candles at 12s blocks
    min_days: float = 5.0,
    half_life_days: float = 10.0,
    sigma_minutes: int = 60,
    gating_classes: Iterable[int] = (0, 1, 3, 4),
) -> Dict[str, float]:
    """
    Convenience wrapper for Q-only salience (opposite-direction moves) with 12× timing preserved.
    Includes default gating over ±1σ and ±2σ classes.
    """
    X_flat, hk2idx = hist
    price = np.asarray(price_data, dtype=float)
    if price.ndim != 1:
        logger.debug("Q path price array has wrong dimensionality")
        return {}

    samples_per_day = int((24 * 60 * 60) // (12 * max(1, sample_every)))
    required = int(max(MIN_REQUIRED_SAMPLES, np.ceil(samples_per_day * min_days)))
    if price.size < required or X_flat.shape[0] < required:
        logger.info("Q path salience requires %d samples (%.1f days); only %d available.",
                    required, min_days, price.size)
        return {}  # Signal insufficient data

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
