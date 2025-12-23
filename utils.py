from __future__ import annotations

import numpy as np

LN2 = float(np.log(2.0))
EPS = 1e-12
MIN_REQUIRED_SAMPLES = 7200


def rolling_std_fast(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = int(x.shape[0])
    if n < int(window):
        return np.full(0, np.nan)
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    var = (s2 - (s1 * s1) / window) / max(window - 1, 1)
    return np.sqrt(np.maximum(var, 0.0))


def sigma_from_price(
    price: np.ndarray,
    *,
    return_horizon_steps: int,
    vol_window: int,
    eps: float = EPS,
) -> np.ndarray:
    """
    Rolling sigma at time t computed from *past* log returns over `return_horizon_steps`.
    Returns an array of length T with NaN where sigma is not yet defined.
    """
    price = np.asarray(price, dtype=float)
    if price.ndim != 1:
        raise ValueError("price_data must be 1-D array")
    h = int(return_horizon_steps)
    if h <= 0:
        raise ValueError("return_horizon_steps must be positive")
    T = int(price.shape[0])
    if T <= h:
        return np.full(T, np.nan)

    r_fwd = np.log(price[h:] + eps) - np.log(price[:-h] + eps)
    n = int(r_fwd.shape[0])

    sig_fwd = np.full(n, np.nan)
    sig_raw = rolling_std_fast(r_fwd, int(vol_window))
    if sig_raw.size > 0:
        sig_fwd[int(vol_window) - 1 :] = sig_raw

    sigma_t = np.full(T, np.nan)
    idx_t = np.arange(h, T, dtype=int)
    sigma_t[idx_t] = sig_fwd[idx_t - h]
    return sigma_t


def make_bins_from_price(
    price: np.ndarray,
    *,
    horizon_steps: int = 1,
    sigma_return_horizon_steps: int | None = None,
    vol_window: int = 7200,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y: labels in 0..4 for valid indices
      valid_idx: start indices t where sigma(t) is defined and >0
    """
    price = np.asarray(price, dtype=float)
    if price.ndim != 1:
        raise ValueError("price_data must be 1-D array")
    T = int(price.shape[0])
    horizon = int(horizon_steps)
    if horizon <= 0:
        raise ValueError("horizon_steps must be positive")

    sig_h = int(horizon if sigma_return_horizon_steps is None else sigma_return_horizon_steps)
    if sig_h <= 0:
        raise ValueError("sigma_return_horizon_steps must be positive")

    if T <= horizon:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    r = np.log(price[horizon:] + eps) - np.log(price[:-horizon] + eps)
    sigma_t = sigma_from_price(price, return_horizon_steps=sig_h, vol_window=int(vol_window), eps=eps)
    sigma_start = sigma_t[: r.shape[0]]

    idx_all = np.arange(r.shape[0], dtype=int)
    valid_mask = np.isfinite(sigma_start) & (sigma_start > 0.0)
    valid_idx = idx_all[valid_mask]
    if valid_idx.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    z = r[valid_mask] / (sigma_start[valid_mask] + eps)
    y = np.zeros_like(z, dtype=int)
    y[z <= -2.0] = 0
    y[(z > -2.0) & (z < -1.0)] = 1
    y[(z >= -1.0) & (z <= 1.0)] = 2
    y[(z > 1.0) & (z < 2.0)] = 3
    y[z >= 2.0] = 4
    return y, valid_idx


def exp_half_life_weights(valid_idx: np.ndarray, half_life_days: float, samples_per_day: float) -> np.ndarray:
    valid_idx = np.asarray(valid_idx, dtype=int)
    if valid_idx.size == 0:
        return np.ones(0, dtype=float)
    i_max = float(valid_idx.max())
    age_days = (i_max - valid_idx.astype(float)) / float(samples_per_day)
    w = np.exp(-LN2 * (age_days / float(half_life_days)))
    return w * (valid_idx.size / np.sum(w))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p) - np.log(1.0 - p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bce(y: np.ndarray, p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def project_simplex(v: np.ndarray) -> np.ndarray:
    if v.ndim != 1:
        v = v.ravel()
    n = int(v.size)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=float)
    rho = int(np.where(cond)[0][-1])
    theta = float(cssv[rho] / float(rho + 1))
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    return w if s > 0 else np.full(n, 1.0 / n, dtype=float)


def recent_mass_weights(t: np.ndarray, *, recent_samples: int, recent_mass: float) -> np.ndarray:
    """
    Piecewise-constant sample weights such that the most recent `recent_samples` (by t)
    receive exactly `recent_mass` of the total weight mass.
    """
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return np.ones(0, dtype=np.float32)
    if int(recent_samples) <= 0:
        raise ValueError("recent_samples must be positive")
    if not (0.0 < float(recent_mass) < 1.0):
        raise ValueError("recent_mass must be in (0,1)")

    t_max = float(np.max(t))
    cutoff = t_max - float(int(recent_samples) - 1)
    recent = t >= cutoff
    n_r = int(np.sum(recent))
    n_o = int(t.size - n_r)
    if n_r == 0 or n_o == 0:
        w = np.ones(t.size, dtype=np.float32)
        return w

    w_old = 1.0
    w_recent = float(recent_mass) * float(n_o) / ((1.0 - float(recent_mass)) * float(n_r))
    w = np.full(t.size, w_old, dtype=np.float32)
    w[recent] = np.float32(w_recent)
    w = w * (t.size / float(np.sum(w)))
    return w
