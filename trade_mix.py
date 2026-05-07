"""
TRADE-MIX challenge scoring.

Miners submit one signed target position in [-1, 1] per TRADE_MIX_ASSET.
Three-stage scoring pipeline:

  A  Per-miner luck filter (Bayesian shrunk Sharpe or stationary block
     bootstrap), applied per (miner, asset).
  B  Cosine-similarity dedup of near-identical signals, then a meta-
     model (gbdt | moe | skillw) over the cluster representatives.
  C  Walk-forward leave-one-out OOS P&L attribution.  Each cluster's
     attribution is split equally among its members — cloning earns no
     per-hotkey bonus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

__all__ = (
    "compute_trade_mix_salience",
    "MetaModelResult",
    "shrunk_sharpe_skill",
    "block_bootstrap_skill",
    "fit_gbdt_meta",
    "fit_moe_meta",
    "loo_attribution",
    "TradeMixConfig",
)


DEFAULT_HORIZON_BARS = 60
DEFAULT_REBAL_PERIOD = 60
DEFAULT_FEE_BPS = 20.0
DEFAULT_MIN_SKILL_PROB = 0.65
DEFAULT_LOO_FOLDS = 5
DEFAULT_DEDUP_COSINE = 0.95
DEFAULT_MAX_OOS_BARS = 43200
DEFAULT_MIN_HISTORY_BARS = 43200
DEFAULT_LUCK_FILTER = "shrunk_sharpe"
DEFAULT_META_MODEL = "gbdt"
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_PRIOR_SHARPE_STD = 0.4
DEFAULT_BOOT_SAMPLES = 200
DEFAULT_BOOT_BLOCK_LEN = 1440
DEFAULT_MOE_REGIMES = 3
DEFAULT_MOE_RIDGE_ALPHA = 1.0
DEFAULT_GBDT_TREES = 30
DEFAULT_GBDT_DEPTH = 2
DEFAULT_GBDT_LR = 0.05
DEFAULT_GBDT_MIN_LEAF = 60
DEFAULT_GBDT_SUBSAMPLE = 0.6
DEFAULT_BARS_PER_DAY = 1440
DEFAULT_TOP_K = 50


@dataclass
class TradeMixConfig:
    horizon_bars: int = DEFAULT_HORIZON_BARS
    rebal_period: int = DEFAULT_REBAL_PERIOD
    fee_bps: float = DEFAULT_FEE_BPS
    min_skill_prob: float = DEFAULT_MIN_SKILL_PROB
    loo_folds: int = DEFAULT_LOO_FOLDS
    dedup_cosine: float = DEFAULT_DEDUP_COSINE
    max_oos_bars: int = DEFAULT_MAX_OOS_BARS
    min_history_bars: int = DEFAULT_MIN_HISTORY_BARS
    luck_filter: str = DEFAULT_LUCK_FILTER
    meta_model: str = DEFAULT_META_MODEL
    train_frac: float = DEFAULT_TRAIN_FRAC
    prior_sharpe_std: float = DEFAULT_PRIOR_SHARPE_STD
    boot_samples: int = DEFAULT_BOOT_SAMPLES
    boot_block_len: int = DEFAULT_BOOT_BLOCK_LEN
    moe_regimes: int = DEFAULT_MOE_REGIMES
    moe_ridge_alpha: float = DEFAULT_MOE_RIDGE_ALPHA
    gbdt_trees: int = DEFAULT_GBDT_TREES
    gbdt_depth: int = DEFAULT_GBDT_DEPTH
    gbdt_lr: float = DEFAULT_GBDT_LR
    gbdt_min_leaf: int = DEFAULT_GBDT_MIN_LEAF
    gbdt_subsample: float = DEFAULT_GBDT_SUBSAMPLE
    bars_per_day: int = DEFAULT_BARS_PER_DAY
    top_k: int = DEFAULT_TOP_K
    seed: int = 42

    _SPEC_ALIASES = {
        "dedup_cosine_threshold": "dedup_cosine",
        "max_oos_window_bars": "max_oos_bars",
        "min_history_window_bars": "min_history_bars",
    }

    @classmethod
    def from_spec(cls, spec: Optional[dict]) -> "TradeMixConfig":
        if not spec:
            return cls()
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if f in spec:
                kwargs[f] = spec[f]
        for alias_key, field_name in cls._SPEC_ALIASES.items():
            if alias_key in spec and field_name not in kwargs:
                kwargs[field_name] = spec[alias_key]
        return cls(**kwargs)


def _safe_log_returns(prices: np.ndarray) -> np.ndarray:
    """Per-bar log returns, T-shaped, with leading 0."""
    p = np.asarray(prices, dtype=np.float64)
    out = np.zeros(p.shape[0], dtype=np.float64)
    valid = (p[:-1] > 0) & (p[1:] > 0)
    np.log(np.where(p[1:] > 0, p[1:], 1.0), out=out[1:], where=valid)
    out[1:] -= np.where(p[:-1] > 0, np.log(np.where(p[:-1] > 0, p[:-1], 1.0)), 0.0)
    out[1:] = np.where(valid, out[1:], 0.0)
    return out


def _forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """log(P_{t+h}/P_t), padded with NaN at the tail."""
    p = np.asarray(prices, dtype=np.float64)
    T = p.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    if horizon <= 0 or horizon >= T:
        return out
    valid = (p[:T - horizon] > 0) & (p[horizon:] > 0)
    front = np.where(p[:T - horizon] > 0, p[:T - horizon], 1.0)
    back = np.where(p[horizon:] > 0, p[horizon:], 1.0)
    fwd = np.log(back / front)
    out[:T - horizon] = np.where(valid, fwd, np.nan)
    return out


def _miner_position_matrix(
    hist: Tuple[np.ndarray, Dict[str, int]],
    n_assets: int,
) -> Tuple[np.ndarray, List[str]]:
    """Reshape flat (T, H*n_assets) hist into (T, H, n_assets) with
    positions clipped to [-1, 1]. Returns (mat, hk_order_by_idx)."""
    X_flat, hk2idx = hist
    X_flat = np.asarray(X_flat, dtype=np.float32)
    if X_flat.ndim != 2:
        raise ValueError("hist X_flat must be 2D")
    T, HD = X_flat.shape
    if HD == 0 or HD % n_assets != 0:
        raise ValueError(f"hist shape {X_flat.shape} not divisible by n_assets={n_assets}")
    H = HD // n_assets
    mat = X_flat.reshape(T, H, n_assets).astype(np.float32, copy=False)
    np.clip(mat, -1.0, 1.0, out=mat)
    hk_order: List[str] = ["?"] * H
    for hk, idx in hk2idx.items():
        if 0 <= idx < H:
            hk_order[idx] = hk
    return mat, hk_order


def _per_miner_pnl_series(
    positions: np.ndarray,         # (T, H) for one asset
    fwd_ret: np.ndarray,           # (T,)
    fee_bps: float,
    rebal_period: int,
    horizon_bars: int,
) -> np.ndarray:
    """Realized per-bar P&L on a vectorized rebalance grid.

    A miner's "trade" at bar t is `position(t)`; held for `horizon_bars`
    bars, with a fee applied whenever the position differs from the
    previous *rebalance bar's* position.  Output shape: (n_rebal, H).
    """
    T, H = positions.shape
    rebal_idx = np.arange(0, T - horizon_bars, rebal_period)
    if rebal_idx.size == 0:
        return np.zeros((0, H), dtype=np.float64)
    pos_at_rebal = positions[rebal_idx]                             # (R, H)
    fwd_at_rebal = fwd_ret[rebal_idx][:, None]                      # (R, 1)
    raw = pos_at_rebal.astype(np.float64) * fwd_at_rebal            # (R, H)
    prev_pos = np.vstack([np.zeros((1, H)), pos_at_rebal[:-1]])
    fees = (fee_bps / 1e4) * np.abs(pos_at_rebal - prev_pos)
    return raw - fees


def shrunk_sharpe_skill(
    pnl: np.ndarray,                 # (R, H) per-rebalance-bar P&L
    bars_per_day: int,
    rebal_period: int,
    prior_std: float,
) -> np.ndarray:
    """Empirical-Bayes shrinkage of per-miner annualized Sharpe.

    Returns a per-miner 'skill probability' in [0, 1]: the posterior
    probability that the true Sharpe is positive under a Gaussian prior
    with mean 0 and std `prior_std` (in the same units as realized
    daily Sharpe).
    """
    R, H = pnl.shape
    if R < 10:
        return np.full(H, 0.5, dtype=np.float64)
    rebals_per_day = max(1, bars_per_day // max(rebal_period, 1))
    mu = pnl.mean(axis=0)
    sd = pnl.std(axis=0, ddof=1) + 1e-12
    realized_sharpe_per_rebal = mu / sd
    realized_sharpe_per_day = realized_sharpe_per_rebal * np.sqrt(rebals_per_day)
    # Posterior under N(0, prior_std^2) prior with N(realized, 1/N) likelihood
    n_eff_per_day = R / rebals_per_day
    var_post = 1.0 / (1.0 / (prior_std ** 2) + n_eff_per_day)
    mean_post = var_post * (realized_sharpe_per_day * n_eff_per_day)
    z = mean_post / np.sqrt(var_post)
    # P(true Sharpe > 0 | data) = Φ(z)
    skill_prob = 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
    skill_prob = np.where(np.isfinite(skill_prob), skill_prob, 0.5)
    return skill_prob.astype(np.float64)


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorized erf via numpy (avoids scipy dep)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


def block_bootstrap_skill(
    pnl: np.ndarray,
    bars_per_day: int,
    rebal_period: int,
    *,
    n_samples: int = DEFAULT_BOOT_SAMPLES,
    block_len: int = DEFAULT_BOOT_BLOCK_LEN,
    seed: int = 0,
) -> np.ndarray:
    """Stationary block bootstrap test against H0: no skill.

    The realized P&L series is *centered* (mean subtracted) before
    resampling so the bootstrap distribution represents the null
    hypothesis of zero edge while preserving autocorrelation.  We then
    count how often a centered-bootstrap Sharpe exceeds the original
    realized Sharpe; 1 - p is the skill evidence in [0, 1].
    """
    R, H = pnl.shape
    if R < 15:
        return np.full(H, 0.5, dtype=np.float64)
    rebals_per_day = max(1, bars_per_day // max(rebal_period, 1))
    rebals_per_block = max(2, block_len // max(rebal_period, 1))
    realized = (pnl.mean(axis=0) / (pnl.std(axis=0, ddof=1) + 1e-12)) * np.sqrt(rebals_per_day)
    centered = pnl - pnl.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    geo_p = 1.0 / rebals_per_block
    counts = np.zeros(H, dtype=np.int64)
    for _ in range(n_samples):
        idx = _stationary_block_indices(R, geo_p, rng)
        sample = centered[idx]
        sh = (sample.mean(axis=0) / (sample.std(axis=0, ddof=1) + 1e-12)) * np.sqrt(rebals_per_day)
        counts += (sh >= realized).astype(np.int64)
    p_value = counts / n_samples
    return (1.0 - p_value).astype(np.float64)


def _stationary_block_indices(T: int, geo_p: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(T, dtype=np.int64)
    pos = 0
    while pos < T:
        start = rng.integers(0, T)
        block = max(1, int(rng.geometric(geo_p)))
        for k in range(block):
            if pos >= T:
                break
            out[pos] = (start + k) % T
            pos += 1
    return out


def _luck_filter(
    positions: np.ndarray,            # (T, H) for one asset
    fwd_ret: np.ndarray,              # (T,)
    cfg: TradeMixConfig,
) -> np.ndarray:
    """Returns per-miner skill probability ∈ [0, 1] for a single asset."""
    pnl = _per_miner_pnl_series(
        positions, fwd_ret, cfg.fee_bps, cfg.rebal_period, cfg.horizon_bars
    )
    if pnl.shape[0] == 0:
        return np.full(positions.shape[1], 0.5, dtype=np.float64)
    if cfg.luck_filter == "shrunk_sharpe":
        return shrunk_sharpe_skill(pnl, cfg.bars_per_day, cfg.rebal_period, cfg.prior_sharpe_std)
    if cfg.luck_filter == "block_bootstrap":
        return block_bootstrap_skill(
            pnl, cfg.bars_per_day, cfg.rebal_period,
            n_samples=cfg.boot_samples, block_len=cfg.boot_block_len, seed=cfg.seed,
        )
    raise ValueError(f"Unknown luck_filter: {cfg.luck_filter}")


def _cosine_dedup(
    positions: np.ndarray,           # (T, H)
    skill: np.ndarray,                # (H,)
    threshold: float,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Greedy clustering by cosine similarity, ordered by descending skill.

    Returns:
      keep_idx: indices of representative miners (one per cluster)
      clusters: list of lists; clusters[c] contains all original indices in cluster c.
    """
    H = positions.shape[1]
    norms = np.linalg.norm(positions, axis=0) + 1e-12
    order = np.argsort(-skill)
    cluster_id = -np.ones(H, dtype=np.int64)
    clusters: List[List[int]] = []
    for j in order:
        if cluster_id[j] >= 0:
            continue
        cid = len(clusters)
        clusters.append([int(j)])
        cluster_id[j] = cid
        if H == 1:
            continue
        sims = (positions.T @ positions[:, j]) / (norms * norms[j])
        sims[j] = -1.0
        for k in np.where(sims >= threshold)[0]:
            if cluster_id[k] < 0:
                cluster_id[k] = cid
                clusters[cid].append(int(k))
    keep_idx = np.array([c[0] for c in clusters], dtype=np.int64)
    return keep_idx, clusters


def _regime_features(
    prices: np.ndarray,                # (T, n_assets)
    miner_dispersion: np.ndarray,      # (T, n_assets)
) -> np.ndarray:
    T, n_assets = prices.shape
    feats: List[np.ndarray] = []
    log_ret = np.zeros_like(prices)
    log_ret[1:] = np.log(np.where(prices[1:] > 0, prices[1:], 1.0)) - np.log(
        np.where(prices[:-1] > 0, prices[:-1], 1.0)
    )
    for lb in (60, 240, 720):
        m = _trailing_mean_2d(log_ret, lb)
        s = _trailing_std_2d(log_ret, lb)
        feats.append(np.nan_to_num(m, nan=0.0))
        feats.append(np.nan_to_num(s, nan=0.0))
    feats.append(np.nan_to_num(miner_dispersion, nan=0.0))
    return np.concatenate(feats, axis=1).astype(np.float32)


def _trailing_mean_2d(arr: np.ndarray, w: int) -> np.ndarray:
    T, K = arr.shape
    if w <= 0 or w >= T:
        return np.zeros_like(arr)
    cs = np.cumsum(arr, axis=0)
    out = np.zeros_like(arr)
    out[w:] = (cs[w:] - cs[:-w]) / w
    out[:w] = np.nan
    return out


def _trailing_std_2d(arr: np.ndarray, w: int) -> np.ndarray:
    T, K = arr.shape
    if w <= 0 or w >= T:
        return np.zeros_like(arr)
    cs = np.cumsum(arr, axis=0)
    cs2 = np.cumsum(arr ** 2, axis=0)
    out = np.zeros_like(arr)
    mean = (cs[w:] - cs[:-w]) / w
    var = np.maximum((cs2[w:] - cs2[:-w]) / w - mean ** 2, 0.0)
    out[w:] = np.sqrt(var)
    out[:w] = np.nan
    return out


@dataclass
class MetaModelResult:
    desk_position: np.ndarray
    pnl: np.ndarray
    info: dict


class _GBDTMeta:
    """One GBDT per asset, fit on (filtered positions, regime features)."""

    def __init__(self, cfg: TradeMixConfig):
        self.cfg = cfg
        self.models: List[Optional[GradientBoostingRegressor]] = []
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.n_assets: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (T_train, F)   y: (T_train, n_assets) forward returns."""
        self.n_assets = y.shape[1]
        self.feat_mean = X.mean(axis=0)
        self.feat_std = X.std(axis=0) + 1e-8
        Xn = (X - self.feat_mean) / self.feat_std
        self.models = []
        for a in range(self.n_assets):
            yi = y[:, a]
            mask = np.isfinite(yi) & np.all(np.isfinite(Xn), axis=1)
            if mask.sum() < 50:
                self.models.append(None)
                continue
            m = GradientBoostingRegressor(
                n_estimators=self.cfg.gbdt_trees,
                max_depth=self.cfg.gbdt_depth,
                learning_rate=self.cfg.gbdt_lr,
                subsample=self.cfg.gbdt_subsample,
                min_samples_leaf=self.cfg.gbdt_min_leaf,
                random_state=self.cfg.seed,
            )
            m.fit(Xn[mask], yi[mask])
            self.models.append(m)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.feat_mean) / self.feat_std
        out = np.zeros((X.shape[0], self.n_assets), dtype=np.float64)
        for a, m in enumerate(self.models):
            if m is None:
                continue
            out[:, a] = m.predict(Xn)
        return out


class _MoEMeta:
    """Hard-regime gated MoE: K-means regime states from regime feats,
    a Ridge regressor per (asset, regime) on miner positions only."""

    def __init__(self, cfg: TradeMixConfig):
        self.cfg = cfg
        self.miner_dim = 0
        self.n_assets = 0
        self.centers: Optional[np.ndarray] = None
        self.regime_feat_idx: Optional[np.ndarray] = None
        self.experts: List[List[Optional[Ridge]]] = []   # experts[asset][regime]

    def _regime_assign(self, regime_feats: np.ndarray) -> np.ndarray:
        if self.centers is None or self.regime_feat_idx is None:
            return np.zeros(regime_feats.shape[0], dtype=np.int64)
        rf = regime_feats[:, self.regime_feat_idx]
        rf_n = (rf - self.rf_mean) / self.rf_std
        d = ((rf_n[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d, axis=1)

    def fit(self, miner_positions: np.ndarray, regime_feats: np.ndarray, y: np.ndarray):
        T, M = miner_positions.shape
        self.miner_dim = M
        self.n_assets = y.shape[1]
        F = regime_feats.shape[1]
        if F <= 6:
            self.regime_feat_idx = np.arange(F)
        else:
            self.regime_feat_idx = np.linspace(0, F - 1, num=min(6, F), dtype=int)
        rf = regime_feats[:, self.regime_feat_idx]
        self.rf_mean = rf.mean(axis=0)
        self.rf_std = rf.std(axis=0) + 1e-8
        rf_n = (rf - self.rf_mean) / self.rf_std

        rng = np.random.default_rng(self.cfg.seed)
        K = max(1, self.cfg.moe_regimes)
        idx_init = rng.choice(T, size=K, replace=False) if T >= K else np.arange(K) % T
        centers = rf_n[idx_init].copy()
        for _ in range(20):
            d = ((rf_n[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            assign = np.argmin(d, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(K):
                msk = assign == k
                new_centers[k] = rf_n[msk].mean(axis=0) if msk.any() else centers[k]
            if np.allclose(new_centers, centers, atol=1e-4):
                centers = new_centers
                break
            centers = new_centers
        self.centers = centers
        regime_id = np.argmin(((rf_n[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)

        self.experts = []
        for a in range(self.n_assets):
            row: List[Optional[Ridge]] = []
            for k in range(K):
                msk = (regime_id == k) & np.isfinite(y[:, a])
                if msk.sum() < 30:
                    row.append(None)
                    continue
                m = Ridge(alpha=self.cfg.moe_ridge_alpha, random_state=self.cfg.seed)
                m.fit(miner_positions[msk], y[msk, a])
                row.append(m)
            self.experts.append(row)

    def predict(self, miner_positions: np.ndarray, regime_feats: np.ndarray) -> np.ndarray:
        regime_id = self._regime_assign(regime_feats)
        T = miner_positions.shape[0]
        out = np.zeros((T, self.n_assets), dtype=np.float64)
        for a in range(self.n_assets):
            for k, m in enumerate(self.experts[a]):
                if m is None:
                    continue
                msk = regime_id == k
                if not msk.any():
                    continue
                out[msk, a] = m.predict(miner_positions[msk])
        return out


class _SkillWeightedMeta:
    """Zero-training meta-model: weights are positive-part of training-
    window correlation between each miner's position and forward return,
    sqrt(n)-shrunk and L1-normalized per asset.
    """

    def __init__(self, cfg: TradeMixConfig):
        self.cfg = cfg
        self.weights: Optional[np.ndarray] = None
        self.n_assets = 0
        self.n_miners = 0

    def fit(self, miner_positions: np.ndarray, y: np.ndarray, n_assets: int):
        T, M = miner_positions.shape
        n_miners = M // n_assets
        self.n_assets = n_assets
        self.n_miners = n_miners
        pos_3d = miner_positions.reshape(T, n_miners, n_assets)
        weights = np.zeros((n_miners, n_assets), dtype=np.float64)
        for ai in range(n_assets):
            yi = y[:, ai]
            mask = np.isfinite(yi)
            if mask.sum() < 30:
                continue
            ys = yi[mask] - yi[mask].mean()
            ys_norm = np.linalg.norm(ys) + 1e-12
            for mi in range(n_miners):
                col = pos_3d[mask, mi, ai]
                if np.std(col) < 1e-6:
                    continue
                cs = col - col.mean()
                cs_norm = np.linalg.norm(cs) + 1e-12
                corr = float((cs * ys).sum() / (cs_norm * ys_norm))
                weights[mi, ai] = max(0.0, corr) * np.sqrt(mask.sum() / 100.0)
        col_sum = weights.sum(axis=0, keepdims=True)
        self.weights = np.where(col_sum > 0, weights / np.maximum(col_sum, 1e-12), 0.0)

    def predict(self, miner_positions: np.ndarray) -> np.ndarray:
        T, M = miner_positions.shape
        if self.weights is None:
            return np.zeros((T, self.n_assets))
        pos_3d = miner_positions.reshape(T, self.n_miners, self.n_assets)
        out = np.zeros((T, self.n_assets), dtype=np.float64)
        for ai in range(self.n_assets):
            out[:, ai] = pos_3d[:, :, ai] @ self.weights[:, ai]
        return out


def fit_gbdt_meta(X: np.ndarray, y: np.ndarray, cfg: TradeMixConfig) -> _GBDTMeta:
    m = _GBDTMeta(cfg)
    m.fit(X, y)
    return m


def fit_moe_meta(
    miner_positions: np.ndarray,
    regime_feats: np.ndarray,
    y: np.ndarray,
    cfg: TradeMixConfig,
) -> _MoEMeta:
    m = _MoEMeta(cfg)
    m.fit(miner_positions, regime_feats, y)
    return m


def fit_skillw_meta(
    miner_positions: np.ndarray,
    y: np.ndarray,
    cfg: TradeMixConfig,
    n_assets: int,
) -> _SkillWeightedMeta:
    m = _SkillWeightedMeta(cfg)
    m.fit(miner_positions, y, n_assets)
    return m


def _build_features(
    miner_positions: np.ndarray,
    prices: np.ndarray,
    cfg: TradeMixConfig,
    *,
    n_assets: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_for_gbdt, miner_positions, regime_feats).

    GBDT input is raw per-miner per-asset columns + per-asset cross-
    sectional summary stats + regime features (rolling vol / dispersion).
    The raw columns make leave-one-out attribution meaningful.
    """
    T = miner_positions.shape[0]
    if miner_positions.shape[1] == 0:
        regime = _regime_features(prices, np.zeros((T, n_assets)))
        return regime.astype(np.float32), miner_positions.astype(np.float32), regime.astype(np.float32)
    M_total = miner_positions.shape[1]
    n_miners = M_total // n_assets
    if n_miners <= 0 or n_miners * n_assets != M_total:
        regime = _regime_features(prices, np.zeros((T, n_assets)))
        return regime.astype(np.float32), miner_positions.astype(np.float32), regime.astype(np.float32)

    pos_3d = miner_positions.reshape(T, n_miners, n_assets)
    summary_blocks = []
    for ai in range(n_assets):
        per_asset = pos_3d[:, :, ai]
        active = np.any(per_asset != 0, axis=0)
        if not active.any():
            zeros = np.zeros((T, 5), dtype=np.float32)
            summary_blocks.append(zeros)
            continue
        pa = per_asset[:, active]
        mean_pos = pa.mean(axis=1, keepdims=True)
        std_pos = pa.std(axis=1, keepdims=True)
        iqr = (np.percentile(pa, 75, axis=1) - np.percentile(pa, 25, axis=1))[:, None]
        frac_pos = (pa > 0.1).mean(axis=1, keepdims=True)
        frac_neg = (pa < -0.1).mean(axis=1, keepdims=True)
        block = np.concatenate([mean_pos, std_pos, iqr, frac_pos, frac_neg], axis=1)
        summary_blocks.append(block.astype(np.float32))
    cs_summary = np.concatenate(summary_blocks, axis=1)   # (T, 5 * n_assets)

    disp = cs_summary[:, 1::5]                             # std per asset → dispersion
    regime = _regime_features(prices, disp)
    X = np.concatenate(
        [miner_positions.astype(np.float32), cs_summary, regime.astype(np.float32)],
        axis=1,
    )
    return X, miner_positions.astype(np.float32), regime.astype(np.float32)


def _walk_forward_pnl(
    miner_positions: np.ndarray,        # (T, M_filtered)
    prices: np.ndarray,                 # (T, n_assets)
    cfg: TradeMixConfig,
    fold: Tuple[int, int, int],         # (train_start, train_end, test_end)
) -> Tuple[np.ndarray, np.ndarray]:
    """Train meta on train slice, simulate desk on test slice.

    Returns (desk_positions_test, pnl_test) where pnl_test has shape (R_test, n_assets).
    """
    T, n_assets = prices.shape
    train_start, train_end, test_end = fold
    fwd = np.column_stack([_forward_returns(prices[:, a], cfg.horizon_bars) for a in range(n_assets)])
    X_full, miner_full, regime_full = _build_features(
        miner_positions, prices, cfg, n_assets=n_assets
    )

    X_tr = X_full[train_start:train_end]
    y_tr = fwd[train_start:train_end]
    valid = np.all(np.isfinite(X_tr), axis=1) & np.all(np.isfinite(y_tr), axis=1)
    if valid.sum() < 100:
        return np.zeros((0, n_assets)), np.zeros((0, n_assets))

    if cfg.meta_model == "gbdt":
        meta = fit_gbdt_meta(X_tr[valid], y_tr[valid], cfg)
        preds_test = meta.predict(X_full[train_end:test_end])
        pred_train = meta.predict(X_full[train_start:train_end])
    elif cfg.meta_model == "moe":
        meta = fit_moe_meta(
            miner_full[train_start:train_end][valid],
            regime_full[train_start:train_end][valid],
            y_tr[valid],
            cfg,
        )
        preds_test = meta.predict(miner_full[train_end:test_end], regime_full[train_end:test_end])
        pred_train = meta.predict(miner_full[train_start:train_end], regime_full[train_start:train_end])
    elif cfg.meta_model == "skillw":
        meta = fit_skillw_meta(
            miner_full[train_start:train_end][valid],
            y_tr[valid],
            cfg,
            n_assets,
        )
        preds_test = meta.predict(miner_full[train_end:test_end])
        pred_train = meta.predict(miner_full[train_start:train_end])
    else:
        raise ValueError(f"Unknown meta_model: {cfg.meta_model}")

    scale = pred_train.std(axis=0) + 1e-8
    desk_pos = np.tanh(preds_test / scale)

    # Realized P&L on the test slice (pad desk_pos out to full horizon)
    test_T = preds_test.shape[0]
    rebal_grid = np.arange(0, test_T - cfg.horizon_bars, cfg.rebal_period)
    if rebal_grid.size == 0:
        return desk_pos, np.zeros((0, n_assets))
    pnl = np.zeros((rebal_grid.size, n_assets), dtype=np.float64)
    for ai in range(n_assets):
        fwd_test = fwd[train_end:test_end, ai]
        for ri, t in enumerate(rebal_grid):
            ft = fwd_test[t]
            if not np.isfinite(ft):
                continue
            p_now = desk_pos[t, ai]
            p_prev = desk_pos[t - cfg.rebal_period, ai] if t >= cfg.rebal_period else 0.0
            fee = (cfg.fee_bps / 1e4) * abs(p_now - p_prev)
            pnl[ri, ai] = p_now * ft - fee
    return desk_pos, pnl


def _make_folds(T: int, cfg: TradeMixConfig) -> List[Tuple[int, int, int]]:
    """Non-overlapping walk-forward (train, test) folds."""
    folds: List[Tuple[int, int, int]] = []
    if T < 2 * cfg.horizon_bars:
        return folds
    train_end_min = max(2 * cfg.horizon_bars, T // 4)
    test_each = max(cfg.horizon_bars * 2, (T - train_end_min) // cfg.loo_folds)
    train_end = train_end_min
    while train_end + test_each <= T:
        folds.append((0, train_end, train_end + test_each))
        train_end += test_each
    if not folds:
        folds.append((0, T // 2, T))
    return folds


def loo_attribution(
    miner_positions: np.ndarray,        # (T, M_total_columns)
    prices: np.ndarray,                 # (T, n_assets)
    cfg: TradeMixConfig,
    *,
    column_groups: Optional[List[List[int]]] = None,
) -> Tuple[np.ndarray, dict]:
    """Compute per-group leave-one-out drop in OOS P&L.

    `column_groups[g]` lists the column indices that belong to miner-group
    g (e.g. one cluster representative's n_assets-many position columns).
    When a group is "dropped", all of its columns are replaced by the
    cross-sectional median across all groups, preserving dimensionality.

    Returns (attribution_per_group, info_dict).
    """
    T, M = miner_positions.shape
    if column_groups is None:
        column_groups = [[c] for c in range(M)]
    G = len(column_groups)
    folds = _make_folds(T, cfg)
    if not folds:
        return np.zeros(G, dtype=np.float64), {"folds": 0, "full_pnl": 0.0}

    full_total = 0.0
    full_pnls = []
    full_pnl_arrays: List[np.ndarray] = []
    for fold in folds:
        _, pnl = _walk_forward_pnl(miner_positions, prices, cfg, fold)
        full_total += float(pnl.sum())
        full_pnls.append(float(pnl.sum()))
        if pnl.size:
            full_pnl_arrays.append(pnl.sum(axis=1))   # per-rebal total P&L
    if full_pnl_arrays:
        all_pnl = np.concatenate(full_pnl_arrays)
        rebals_per_day = max(1, cfg.bars_per_day // max(cfg.rebal_period, 1))
        meta_sharpe = (
            float(all_pnl.mean() / (all_pnl.std() + 1e-12) * np.sqrt(rebals_per_day * 365))
            if all_pnl.size > 5 else 0.0
        )
    else:
        meta_sharpe = 0.0

    attribution = np.zeros(G, dtype=np.float64)
    medians = np.median(miner_positions, axis=1)
    for g, cols in enumerate(column_groups):
        if not cols:
            continue
        mp_dropped = miner_positions.copy()
        for c in cols:
            if 0 <= c < M:
                mp_dropped[:, c] = medians
        drop_total = 0.0
        for fold in folds:
            _, pnl = _walk_forward_pnl(mp_dropped, prices, cfg, fold)
            drop_total += float(pnl.sum())
        delta = full_total - drop_total
        attribution[g] = max(0.0, delta)
    return attribution, {
        "folds": len(folds),
        "full_pnl": full_total,
        "per_fold_full_pnl": full_pnls,
        "meta_oos_sharpe": meta_sharpe,
    }


def compute_trade_mix_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    prices_multi: np.ndarray,
    *,
    blocks_ahead: int,
    sample_every: int,
    sidx_arr: Optional[np.ndarray] = None,
    spec: Optional[dict] = None,
    cfg: Optional[TradeMixConfig] = None,
    return_diagnostics: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], dict]:
    """Score miners on the TRADE-MIX challenge.

    Returns an empty dict during warmup (T < cfg.min_history_bars) so that
    the challenge contributes zero weight until enough history accumulates.
    """
    if cfg is None:
        cfg = TradeMixConfig.from_spec(spec)

    X_flat, hk2idx = hist
    if not isinstance(hk2idx, dict) or len(hk2idx) == 0:
        return ({}, {}) if return_diagnostics else {}
    H = len(hk2idx)
    n_assets = prices_multi.shape[1]
    if X_flat.shape[1] != H * n_assets:
        logger.warning(
            "trade_mix: hist columns (%d) != H * n_assets (%d * %d). Skipping.",
            X_flat.shape[1], H, n_assets,
        )
        return ({}, {}) if return_diagnostics else {}

    T = min(X_flat.shape[0], prices_multi.shape[0])
    if T < max(cfg.min_history_bars, cfg.horizon_bars * 4):
        return (
            ({}, {"reason": "warmup", "T": T, "min_history_bars": cfg.min_history_bars})
            if return_diagnostics else {}
        )
    if cfg.max_oos_bars > 0 and T > cfg.max_oos_bars:
        X_flat = X_flat[-cfg.max_oos_bars:]
        prices_multi = prices_multi[-cfg.max_oos_bars:]
        T = cfg.max_oos_bars

    pos = X_flat[:T].reshape(T, H, n_assets).astype(np.float32, copy=True)
    np.clip(pos, -1.0, 1.0, out=pos)

    skill_per_asset = np.zeros((H, n_assets), dtype=np.float64)
    for ai in range(n_assets):
        fwd = _forward_returns(prices_multi[:, ai], cfg.horizon_bars)
        skill_per_asset[:, ai] = _luck_filter(pos[:, :, ai], fwd, cfg)
    skill = skill_per_asset.mean(axis=1)
    keep_mask = skill >= cfg.min_skill_prob
    if not keep_mask.any():
        return ({}, {"reason": "no_miners_passed_luck_filter", "max_skill": float(skill.max())}) if return_diagnostics else {}

    kept_idx = np.where(keep_mask)[0]

    pos_kept = pos[:, kept_idx, :]
    miner_signature = pos_kept.transpose(0, 2, 1).reshape(T * n_assets, kept_idx.size)
    rep_local, clusters = _cosine_dedup(miner_signature, skill[kept_idx], cfg.dedup_cosine)
    rep_idx = kept_idx[rep_local]
    n_groups = rep_idx.size

    miner_input = pos[:, rep_idx, :].reshape(T, n_groups * n_assets).astype(np.float32)

    column_groups: List[List[int]] = [
        list(range(g * n_assets, (g + 1) * n_assets)) for g in range(n_groups)
    ]

    attribution_groups, info = loo_attribution(
        miner_input, prices_multi[:T], cfg, column_groups=column_groups
    )

    salience = np.zeros(H, dtype=np.float64)
    for cid, members in enumerate(clusters):
        cluster_attr = float(attribution_groups[cid]) if cid < attribution_groups.size else 0.0
        if cluster_attr <= 0:
            continue
        share = cluster_attr / max(len(members), 1)
        for local in members:
            global_hk = int(kept_idx[local])
            salience[global_hk] += share * float(skill[global_hk])

    total = float(salience.sum())
    if total <= 0:
        return ({}, {"reason": "no_positive_attribution", "info": info}) if return_diagnostics else {}

    if salience.size > cfg.top_k:
        top_idx = np.argsort(-salience)[:cfg.top_k]
        mask = np.zeros_like(salience, dtype=bool)
        mask[top_idx] = True
        salience = np.where(mask, salience, 0.0)
        total = float(salience.sum()) or 1.0

    salience = salience / total

    inv_idx2hk: List[Optional[str]] = [None] * H
    for hk, idx in hk2idx.items():
        if 0 <= idx < H:
            inv_idx2hk[idx] = hk
    out: Dict[str, float] = {}
    for j in range(H):
        if salience[j] <= 0:
            continue
        hk = inv_idx2hk[j]
        if hk:
            out[hk] = float(salience[j])

    info["luck_filter"] = cfg.luck_filter
    info["meta_model"] = cfg.meta_model
    info["n_kept"] = int(keep_mask.sum())
    info["n_clusters"] = len(clusters)
    info["n_assets"] = n_assets
    info["max_skill_prob"] = float(skill.max())

    return (out, info) if return_diagnostics else out
