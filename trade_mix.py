"""
TRADE-MIX v2 challenge scoring.

Miners submit one signed target position in [-1, 1] per TRADE_MIX_ASSET.
Positions are decomposed into a market-timing component and a relative-value
(cross-asset) component and each channel is paid only on statistically proven
skill.  Full design + adversarial evaluation: TRADE_MIX_V2_IM_DESIGN.md.

Pipeline (per weight-calculation epoch, recomputed from the expanding panel):

  1  Resample the minute-cadence datalog rows onto an hourly grid.
  2  Causal factor structure: inverse-vol market factor and shrunk betas,
     refreshed daily from data strictly before t.  Position decomposition
     P = m * beta + q  (m = market exposure, q = RV book).
  3  Evidence per miner and channel: blended studentized statistic of the
     position/forward-return cross-moment at {12h, 24h, 48h} (RV, vs
     beta-residualized returns) and 24h (beta, vs the market factor),
     against a circular block-shift null computed exactly for all offsets
     via FFT cross-correlation.  Evidence accrues over the full panel and
     is never reset.
  4  Gates: probation (age + coverage), activity, Benjamini-Hochberg FDR
     across all considered miners.  Near-duplicate RV books are
     residualized against senior (earlier-registered) miners before
     evidence is computed, so copies only score on incremental information.
  5  Payment: softplus(t - shift) x trailing costed attribution x coverage,
     cosine-dedup clusters split a single credit, per-miner cap.  The
     unearned remainder of the challenge pool is returned under the
     "__burn__" key and routed to UID 0 by the validator.
"""

from __future__ import annotations

import config  # noqa: F401  Importing config first locks the cross-hardware env (BLAS threads, hash seed) before numpy loads.

import logging
from dataclasses import dataclass
from math import erfc, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = (
    "compute_trade_mix_salience",
    "TradeMixV2Config",
    "score_epoch",
    "causal_residuals",
    "decompose_positions",
    "evidence_t",
    "bh_gate",
    "cosine_matrix",
    "greedy_clusters",
    "seniority_residualize",
    "EpochResult",
)

EPS = 1e-12
BURN_KEY = "__burn__"

# Blocks are 12s; ledger rows arrive every SAMPLE_EVERY blocks.
SECONDS_PER_BLOCK = 12


@dataclass
class TradeMixV2Config:
    # --- factor structure (hourly units)
    horizons: Tuple[int, ...] = (12, 24, 48)
    blend_w: Tuple[float, ...] = (0.25, 0.50, 0.25)
    beta_horizon: int = 24
    vol_window: int = 720            # 30d of hourly bars for factor weights
    beta_window: int = 1440          # 60d for betas
    beta_shrink: float = 0.3         # toward 1.0
    refresh: int = 24                # daily refresh of weights/betas
    warmup: int = 120                # 5d before residuals exist
    # --- evidence / gates
    min_shift: int = 96              # circular-null offsets >= 96h
    fdr_q: float = 0.05
    probation_hours: int = 21 * 24
    probation_coverage: float = 0.9
    activity_min_std: float = 1e-4
    seniority_cosine: float = 0.20   # sign-agnostic; full sequential Gram-Schmidt
    dedup_cosine: float = 0.95
    dedup_window: int = 720
    # --- payment
    fee_per_side: float = 10e-4      # 10 bps
    softplus_shift: float = 2.0
    channel_split: Tuple[float, float] = (0.75, 0.25)   # RV, beta
    miner_cap: float = 0.20          # of channel pool
    coverage_window: int = 720
    attr_window: int = 720

    @classmethod
    def from_spec(cls, spec: Optional[dict]) -> "TradeMixV2Config":
        if not spec:
            return cls()
        kw = {}
        mapping = {
            "horizons_hours": "horizons",
            "blend_weights": "blend_w",
            "beta_horizon_hours": "beta_horizon",
            "vol_window_hours": "vol_window",
            "beta_window_hours": "beta_window",
            "beta_shrink": "beta_shrink",
            "refresh_hours": "refresh",
            "warmup_hours": "warmup",
            "min_shift_hours": "min_shift",
            "fdr_q": "fdr_q",
            "probation_hours": "probation_hours",
            "probation_coverage": "probation_coverage",
            "activity_min_std": "activity_min_std",
            "seniority_cosine": "seniority_cosine",
            "dedup_cosine_threshold": "dedup_cosine",
            "dedup_window_hours": "dedup_window",
            "softplus_shift": "softplus_shift",
            "miner_cap": "miner_cap",
            "coverage_window_hours": "coverage_window",
            "attr_window_hours": "attr_window",
        }
        for spec_key, field_name in mapping.items():
            if spec_key in spec:
                v = spec[spec_key]
                kw[field_name] = tuple(v) if isinstance(v, list) else v
        if "fee_bps" in spec:
            kw["fee_per_side"] = float(spec["fee_bps"]) * 1e-4
        if "channel_split_rv" in spec:
            rv = float(spec["channel_split_rv"])
            kw["channel_split"] = (rv, 1.0 - rv)
        return cls(**kw)


# Alias used by the research / adversarial scripts.
TMv2Config = TradeMixV2Config


# ---------------------------------------------------------------------------
# Factor structure (causal)
# ---------------------------------------------------------------------------

def causal_residuals(prices: np.ndarray, cfg: TradeMixV2Config) -> dict:
    """Hourly log returns, causal factor weights/betas, residual returns.

    Returns dict with r (T,A), mkt (T,), betas (T,A), resid (T,A), w (T,A),
    valid (T,).  Row t of betas/w uses data strictly before t (daily refresh).
    """
    logp = np.log(prices)
    r = np.diff(logp, axis=0, prepend=logp[:1])
    r[0] = 0.0
    T, A = r.shape
    w = np.full((T, A), 1.0 / A)
    betas = np.ones((T, A))
    mkt = np.zeros(T)
    resid = np.zeros((T, A))
    last = None
    for t in range(T):
        if t >= cfg.warmup and t % cfg.refresh == 0:
            lo_v = max(0, t - cfg.vol_window)
            vol = r[lo_v:t].std(axis=0) + EPS
            wt = 1.0 / vol
            wt /= wt.sum()
            lo_b = max(0, t - cfg.beta_window)
            m_hist = r[lo_b:t] @ wt
            vm = m_hist.var() + EPS
            b = np.array([np.cov(r[lo_b:t, a], m_hist)[0, 1] / vm for a in range(A)])
            b = (1 - cfg.beta_shrink) * b + cfg.beta_shrink * 1.0
            last = (wt, b)
        if last is not None:
            w[t], betas[t] = last
        mkt[t] = r[t] @ w[t]
        resid[t] = r[t] - betas[t] * mkt[t]
    valid = np.zeros(T, dtype=bool)
    valid[cfg.warmup:] = True
    return {"r": r, "mkt": mkt, "betas": betas, "resid": resid, "w": w, "valid": valid}


def decompose_positions(P: np.ndarray, fac: dict) -> Tuple[np.ndarray, np.ndarray]:
    """P (T,H,A) -> m (T,H) market exposure, q (T,H,A) RV book.

    Absent submissions must already be zero-filled (flat book); coverage is
    tracked separately from the submission mask.
    """
    P0 = np.nan_to_num(P, nan=0.0)
    w = fac["w"][:, None, :]
    betas = fac["betas"][:, None, :]
    m = np.sum(w * P0, axis=2)
    q = P0 - m[:, :, None] * betas
    return m, q


# ---------------------------------------------------------------------------
# Evidence: circular-shift null via FFT
# ---------------------------------------------------------------------------

def _fwd_sum(y: np.ndarray, h: int) -> np.ndarray:
    """out[t] = sum(y[t+1 .. t+h]); rows without a full window are NaN."""
    T = len(y)
    c = np.concatenate([np.zeros((1,) + y.shape[1:]), np.cumsum(y, axis=0)])
    out = np.full_like(y, np.nan, dtype=np.float64)
    if T > h:
        out[: T - h] = c[1 + h: T + 1] - c[1: T - h + 1]
    return out


def _label_ffts(ys: List[np.ndarray]) -> List[np.ndarray]:
    """Precompute rfft of each (T,A) label array (NaN -> 0), shared by miners."""
    out = []
    for y in ys:
        y2 = np.nan_to_num(np.atleast_2d(y.T).T, nan=0.0)  # (T,) -> (T,1)
        out.append(np.fft.rfft(y2, axis=0))
    return out


def _circ_stats(x: np.ndarray, Yf: np.ndarray) -> np.ndarray:
    """All circular-shift statistics S(k) = sum_t x[(t-k)%T] . y[t].

    x: (T,) or (T,A); Yf: precomputed rfft of the matching labels, (F, A).
    Returns S over offsets k = 0..T-1 (k=0 is the observed statistic).
    """
    if x.ndim == 1:
        x = x[:, None]
    T = x.shape[0]
    Xf = np.fft.rfft(x, axis=0)
    return np.fft.irfft(np.conj(Xf) * Yf, n=T, axis=0).sum(axis=1)


def evidence_t(x: np.ndarray, label_ffts: List[np.ndarray], blend_w: List[float],
               min_shift: int) -> Tuple[float, float]:
    """Blended studentized evidence and its p-value.

    Blends per-horizon z-scores over the SAME shift offsets so horizon
    correlation is captured by the null itself.
    """
    T = x.shape[0]
    if T < 3 * min_shift:
        return 0.0, 1.0
    ks = np.arange(T)
    valid_k = (ks >= min_shift) & (ks <= T - min_shift)
    if valid_k.sum() < 50:
        return 0.0, 1.0
    zb_obs, zb_null = 0.0, np.zeros(int(valid_k.sum()))
    for w_h, Yf in zip(blend_w, label_ffts):
        S = _circ_stats(x, Yf)
        null = S[valid_k]
        sd = null.std() + EPS
        mu = null.mean()
        zb_obs += w_h * (S[0] - mu) / sd
        zb_null += w_h * (null - mu) / sd
    sd_b = zb_null.std() + EPS
    t = (zb_obs - zb_null.mean()) / sd_b
    p_emp = (1.0 + np.sum(zb_null >= zb_obs)) / (1.0 + len(zb_null))
    p_norm = 0.5 * erfc(t / sqrt(2.0))
    return float(t), float(min(p_emp, max(p_norm, 1e-12)) if t > 0 else p_emp)


# ---------------------------------------------------------------------------
# Gate machinery
# ---------------------------------------------------------------------------

def bh_gate(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: True where discovery."""
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")
    passed = np.zeros(m, dtype=bool)
    thresh = q * np.arange(1, m + 1) / m
    ok = pvals[order] <= thresh
    if ok.any():
        kmax = int(np.max(np.flatnonzero(ok)))
        passed[order[: kmax + 1]] = True
    return passed


def cosine_matrix(Q: np.ndarray) -> np.ndarray:
    """Q (H, F) flattened books -> (H, H) cosine similarity."""
    n = np.linalg.norm(Q, axis=1) + EPS
    return (Q @ Q.T) / np.outer(n, n)


def greedy_clusters(sim: np.ndarray, thr: float, priority: np.ndarray) -> List[List[int]]:
    """Greedy clustering: highest-priority unassigned miner seeds a cluster.

    Ties in priority break on miner index (stable across hardware).
    """
    Hn = sim.shape[0]
    unassigned = set(range(Hn))
    clusters: List[List[int]] = []
    for i in np.argsort(-priority, kind="stable"):
        i = int(i)
        if i not in unassigned:
            continue
        members = sorted(j for j in unassigned if sim[i, j] >= thr or j == i)
        for j in members:
            unassigned.discard(j)
        clusters.append(members)
    return clusters


def seniority_residualize(q_all: np.ndarray, sim: np.ndarray, first_seen: np.ndarray,
                          thr: float, tiebreak: Optional[np.ndarray] = None) -> np.ndarray:
    """Sequential Gram-Schmidt in seniority order: each junior's RV book is
    projected off every already-residualized senior with |cosine| >= thr
    before evidence is computed.  abs(): a sign-flipped copy is the same
    information up to a free bit, so anti-correlated seniors trigger
    residualization too.

    Miners are processed strictly in seniority order: first_seen, with ties
    broken by `tiebreak` (production passes the lexicographic rank of the
    hotkey), so the result is a function of the data and identities only —
    identical across validators regardless of panel column layout.
    q_all: (T, H, A)."""
    Hn = q_all.shape[1]
    out = q_all.copy()
    if tiebreak is None:
        tiebreak = np.arange(Hn)
    order = np.lexsort((tiebreak, first_seen))  # seniors first
    for pos_j in range(Hn):
        j = int(order[pos_j])
        for pos_s in range(pos_j):
            s = int(order[pos_s])
            if abs(sim[s, j]) < thr:
                continue
            xs = out[:, s, :].ravel()
            xj = out[:, j, :].ravel()
            denom = xs @ xs
            if denom > EPS:
                coef = (xj @ xs) / denom
                out[:, j, :] = out[:, j, :] - coef * out[:, s, :]
    return out


def _softplus(x: float) -> float:
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))


# ---------------------------------------------------------------------------
# Epoch scoring
# ---------------------------------------------------------------------------

@dataclass
class EpochResult:
    t_rv: np.ndarray
    p_rv: np.ndarray
    t_beta: np.ndarray
    p_beta: np.ndarray
    gate_rv: np.ndarray
    gate_beta: np.ndarray
    eligible: np.ndarray
    shares: np.ndarray            # fraction of the challenge pool per miner
    burned: float                 # fraction of pool burned
    diag: dict


def score_epoch(P: np.ndarray, prices: np.ndarray, first_seen: np.ndarray,
                cfg: TradeMixV2Config,
                submitted: Optional[np.ndarray] = None,
                tiebreak: Optional[np.ndarray] = None) -> EpochResult:
    """Score at 'now' = last row of the expanding hourly panel.

    P: (T, H, A) positions (NaN or 0 rows = absent), prices: (T, A),
    first_seen: (H,) first hourly index each hotkey submitted,
    submitted: (T, H) bool submission mask (derived from P if omitted),
    tiebreak: (H,) canonical seniority tie-break (lexicographic hotkey rank).
    """
    T, Hn, A = P.shape
    fac = causal_residuals(prices, cfg)
    m, q = decompose_positions(P, fac)

    if submitted is None:
        submitted = np.isfinite(P[..., 0])
    cov_win = submitted[-cfg.coverage_window:]
    coverage = cov_win.mean(axis=0)
    prob_win = submitted[-cfg.probation_hours:]
    age_ok = (T - first_seen) >= cfg.probation_hours
    prob_cov = prob_win.mean(axis=0)
    probation_ok = age_ok & (prob_cov >= cfg.probation_coverage)
    active_rv = q[fac["valid"]].std(axis=(0, 2)) >= cfg.activity_min_std

    # --- dedup / seniority structure on trailing RV books
    lo = max(0, T - cfg.dedup_window)
    flat = np.nan_to_num(q[lo:], nan=0.0).transpose(1, 0, 2).reshape(Hn, -1)
    sim = cosine_matrix(flat)
    q_gate = seniority_residualize(q, sim, first_seen, cfg.seniority_cosine,
                                   tiebreak=tiebreak)

    # --- labels (shared FFTs across miners)
    ys_rv = [_fwd_sum(fac["resid"], h) for h in cfg.horizons]
    for y in ys_rv:
        y[~fac["valid"]] = np.nan
    y_beta = _fwd_sum(fac["mkt"][:, None], cfg.beta_horizon)[:, 0]
    y_beta[~fac["valid"]] = np.nan
    ffts_rv = _label_ffts(ys_rv)
    ffts_beta = _label_ffts([y_beta])

    # --- evidence per miner
    t_rv = np.zeros(Hn); p_rv = np.ones(Hn)
    t_be = np.zeros(Hn); p_be = np.ones(Hn)
    for i in range(Hn):
        t_rv[i], p_rv[i] = evidence_t(q_gate[:, i, :], ffts_rv, list(cfg.blend_w), cfg.min_shift)
        t_be[i], p_be[i] = evidence_t(m[:, i], ffts_beta, [1.0], cfg.min_shift)

    # --- gates
    considered_rv = probation_ok & active_rv
    considered_be = probation_ok
    gate_rv = np.zeros(Hn, dtype=bool)
    gate_be = np.zeros(Hn, dtype=bool)
    if considered_rv.any():
        idx = np.flatnonzero(considered_rv)
        gate_rv[idx] = bh_gate(p_rv[idx], cfg.fdr_q)
    if considered_be.any():
        idx = np.flatnonzero(considered_be)
        gate_be[idx] = bh_gate(p_be[idx], cfg.fdr_q)

    # --- attribution: trailing costed PnL of each channel component
    lo_a = max(0, T - cfg.attr_window)
    r_next_resid = np.vstack([fac["resid"][lo_a + 1:], np.zeros((1, A))])
    q_w = np.nan_to_num(q[lo_a:], nan=0.0)
    pnl_rv = np.einsum("tha,ta->h", q_w[:-1], r_next_resid[:-1]) if q_w.shape[0] > 1 else np.zeros(Hn)
    turn_rv = np.abs(np.diff(q_w, axis=0)).sum(axis=(0, 2))
    attr_rv = np.maximum(0.0, pnl_rv - cfg.fee_per_side * turn_rv)

    mkt_next = np.concatenate([fac["mkt"][lo_a + 1:], [0.0]])
    m_w = np.nan_to_num(m[lo_a:], nan=0.0)
    pnl_be = (m_w[:-1] * mkt_next[:-1, None]).sum(axis=0) if m_w.shape[0] > 1 else np.zeros(Hn)
    turn_be = np.abs(np.diff(m_w, axis=0)).sum(axis=0)
    attr_be = np.maximum(0.0, pnl_be - cfg.fee_per_side * turn_be)

    # --- payment
    def channel_shares(gate: np.ndarray, t_vec: np.ndarray, attr: np.ndarray) -> Tuple[np.ndarray, float]:
        raw = np.zeros(Hn)
        for i in np.flatnonzero(gate):
            raw[i] = _softplus(t_vec[i] - cfg.softplus_shift) * attr[i] * coverage[i]
        # cluster split: a dedup cluster earns one credit, split equally
        clusters = greedy_clusters(sim, cfg.dedup_cosine, priority=t_vec)
        for members in clusters:
            g = [i for i in members if raw[i] > 0]
            if len(g) > 1:
                tot = sum(raw[i] for i in g)
                for i in g:
                    raw[i] = tot / len(g) ** 2
        s = raw.sum()
        if s <= 0:
            return np.zeros(Hn), 1.0
        shares = raw / s
        over = shares > cfg.miner_cap
        if over.any():
            excess = float(np.sum(shares[over] - cfg.miner_cap))
            shares[over] = cfg.miner_cap
            under = ~over & (shares > 0)
            if under.any():
                room = np.minimum(
                    cfg.miner_cap - shares[under],
                    excess * shares[under] / max(shares[under].sum(), EPS),
                )
                shares[under] += room
        return shares, float(1.0 - shares.sum())

    s_rv, burn_rv = channel_shares(gate_rv, t_rv, attr_rv)
    s_be, burn_be = channel_shares(gate_be, t_be, attr_be)
    w_rv, w_be = cfg.channel_split
    shares = w_rv * s_rv + w_be * s_be
    burned = w_rv * burn_rv + w_be * burn_be

    return EpochResult(
        t_rv=t_rv, p_rv=p_rv, t_beta=t_be, p_beta=p_be,
        gate_rv=gate_rv, gate_beta=gate_be,
        eligible=probation_ok, shares=shares, burned=burned,
        diag={"coverage": coverage, "active_rv": active_rv,
              "attr_rv": attr_rv, "attr_beta": attr_be,
              "n_gate_rv": int(gate_rv.sum()), "n_gate_beta": int(gate_be.sum())},
    )


# ---------------------------------------------------------------------------
# Datalog adaptation: minute-cadence rows -> hourly panel
# ---------------------------------------------------------------------------

def _hourly_panel(
    X_flat: np.ndarray,
    prices_multi: np.ndarray,
    sidx_arr: Optional[np.ndarray],
    n_assets: int,
    sample_every: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Select rows on an hourly grid.  Returns (pos (T,H,A), prices (T,A),
    submitted (T,H)) or None if too little usable data."""
    step = max(1, int(round(3600 / (SECONDS_PER_BLOCK * sample_every))))
    n_rows = min(X_flat.shape[0], prices_multi.shape[0])
    X_flat = X_flat[:n_rows]
    prices_multi = prices_multi[:n_rows]

    if sidx_arr is not None and len(sidx_arr) >= n_rows:
        sidx = np.asarray(sidx_arr[:n_rows], dtype=np.int64)
        grid = np.arange(sidx[0], sidx[-1] + 1, step)
        loc = np.searchsorted(sidx, grid)
        loc = np.clip(loc, 0, n_rows - 1)
        tol = max(1, step // 6)
        ok = np.abs(sidx[loc] - grid) <= tol
        rows = np.unique(loc[ok])
    else:
        rows = np.arange(0, n_rows, step)
    if rows.size < 10:
        return None

    prices = prices_multi[rows].astype(np.float64, copy=True)
    # keep only rows where every asset has a real price (no synthetic fills)
    good = np.all(prices > 0, axis=1)
    if good.sum() < 10:
        return None
    rows = rows[good]
    prices = prices[good]

    H = X_flat.shape[1] // n_assets
    pos = X_flat[rows].reshape(len(rows), H, n_assets).astype(np.float32, copy=True)
    np.clip(pos, -1.0, 1.0, out=pos)
    # absent submissions are stored as all-zero rows; a deliberately flat
    # book is indistinguishable but also earns nothing, so this is safe.
    submitted = np.any(pos != 0.0, axis=2)
    return pos, prices, submitted


def compute_trade_mix_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    prices_multi: np.ndarray,
    *,
    blocks_ahead: int,
    sample_every: int,
    sidx_arr: Optional[np.ndarray] = None,
    spec: Optional[dict] = None,
    cfg: Optional[TradeMixV2Config] = None,
    return_diagnostics: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], dict]:
    """Score miners on the TRADE-MIX challenge (v2 incentive mechanism).

    Returns a dict of hotkey -> fraction of the challenge pool, plus the
    special key "__burn__" carrying the unearned fraction (values sum to 1).
    During warmup (not enough hourly history for the circular null) the
    whole pool is burned.
    """
    if cfg is None:
        cfg = TradeMixV2Config.from_spec(spec)

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

    panel = _hourly_panel(np.asarray(X_flat, dtype=np.float32), np.asarray(prices_multi),
                          sidx_arr, n_assets, int(sample_every))
    if panel is None:
        out = {BURN_KEY: 1.0}
        return (out, {"reason": "no_usable_rows"}) if return_diagnostics else out
    pos, prices, submitted = panel
    T = pos.shape[0]

    if T < 3 * cfg.min_shift or T <= cfg.warmup:
        out = {BURN_KEY: 1.0}
        diag = {"reason": "warmup", "hourly_rows": T, "needed": 3 * cfg.min_shift}
        return (out, diag) if return_diagnostics else out

    any_sub = submitted.any(axis=0)
    first_seen = np.where(any_sub, np.argmax(submitted, axis=0), T).astype(np.int64)

    # canonical seniority tie-break: lexicographic hotkey rank, independent of
    # panel column layout (the ledger already sorts hotkeys, but don't rely on it)
    idx2hk = {v: k for k, v in hk2idx.items()}
    hks_by_col = [idx2hk[i] for i in range(H)]
    tiebreak = np.empty(H, dtype=np.int64)
    tiebreak[np.argsort(np.array(hks_by_col), kind="stable")] = np.arange(H)

    res = score_epoch(pos, prices, first_seen, cfg, submitted=submitted,
                      tiebreak=tiebreak)

    out: Dict[str, float] = {}
    for i in range(H):
        v = float(res.shares[i])
        if v > 1e-9:
            out[idx2hk[i]] = v
    out[BURN_KEY] = max(0.0, float(res.burned))

    logger.info(
        "trade_mix v2: T=%dh, miners=%d, gated_rv=%d, gated_beta=%d, burned=%.3f",
        T, H, res.diag["n_gate_rv"], res.diag["n_gate_beta"], res.burned,
    )
    if return_diagnostics:
        diag = {
            "hourly_rows": T,
            "n_gate_rv": res.diag["n_gate_rv"],
            "n_gate_beta": res.diag["n_gate_beta"],
            "burned": float(res.burned),
            "t_rv": {idx2hk[i]: float(res.t_rv[i]) for i in range(H)},
            "p_rv": {idx2hk[i]: float(res.p_rv[i]) for i in range(H)},
            "t_beta": {idx2hk[i]: float(res.t_beta[i]) for i in range(H)},
            "eligible": {idx2hk[i]: bool(res.eligible[i]) for i in range(H)},
        }
        return out, diag
    return out
