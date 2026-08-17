from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

fs = sys.modules[__name__]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CACHE_1M = os.path.join(HERE, "cache_btc_multi_1m.npz")

MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    min_h: int
    max_h: int


DEFAULT_REGIMES: Tuple[RegimeSpec, ...] = (
    RegimeSpec("A", 1, 24),
    RegimeSpec("B", 24, 48),
    RegimeSpec("C", 72, 168),
    RegimeSpec("D", 168, 336),
)


@dataclass
class MechanismParams:
    gamma: float = 0.15
    tau: float = 0.30
    lam: float = 0.50
    delta: float = 0.25
    tail_mult: float = 2.0
    apply_path_to_stops: bool = False
    same_candle_rule: str = "sl_first"
    tp1_then_sl: str = "sl"
    p_bound: float = 0.01
    use_paper_kelly_inversion: bool = False
    ewma_alpha: float = 0.10
    regime_floor: bool = True
    floor_clamp: float | None = None
    kappa_hours: float = 672.0
    probation_hours: float = 0.0
    staking_enabled: bool = True
    a_base: float = 100.0
    a_cap_scale: float = 200.0
    epoch_hours: float = 1.0


@dataclass
class Market:
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    sigma_24h: np.ndarray
    sigma_30d: np.ndarray
    minute_sigma: np.ndarray

    @property
    def n_minutes(self) -> int:
        return len(self.close)

    def is_tail(self, t: int, tail_mult: float) -> bool:
        s30 = self.sigma_30d[t]
        return bool(s30 > 0 and self.sigma_24h[t] > tail_mult * s30)

    def trailing_return(self, t: int, minutes: int) -> float:
        lo = max(0, t - minutes)
        if lo >= t:
            return 0.0
        return float(np.log(self.close[t] / self.close[lo]))


def _rolling_std(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    T = len(x)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    idx = np.arange(T + 1)
    lo = np.maximum(0, idx - window)
    n = (idx - lo).astype(float)
    s1 = c1[idx] - c1[lo]
    s2 = c2[idx] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.maximum(s2 / np.maximum(n, 1.0) - (s1 / np.maximum(n, 1.0)) ** 2, 0.0)
        out = np.sqrt(var)
    out[n < min_periods] = 0.0
    return out[1:]


@dataclass
class Prediction:
    miner_id: int
    regime: str
    direction: int
    f: float
    entry: float
    sl: float
    tp1: float
    tp2: float
    h_hours: int
    t_submit: int
    stake: float = 0.0

    @property
    def t_horizon(self) -> int:
        return self.t_submit + int(self.h_hours * 60)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.sl)

    @property
    def odds_b(self) -> float:
        return abs(self.tp1 - self.entry) / self.risk

    def r_multiple(self, price: float) -> float:
        return self.direction * (price - self.entry) / self.risk


def implied_p(f: float, b: float, use_paper_formula: bool = False) -> float:
    if b <= 0:
        return float("nan")
    if use_paper_formula:
        return (f * b + (1.0 - f)) / (b + 1.0)
    return (f * b + 1.0) / (b + 1.0)


def validate(pred: Prediction, mech: MechanismParams) -> bool:
    if not (0.01 <= pred.f <= 0.25) or pred.h_hours <= 0:
        return False
    if pred.direction == 1:
        ordered = pred.sl < pred.entry < pred.tp1 < pred.tp2
    elif pred.direction == -1:
        ordered = pred.tp2 < pred.tp1 < pred.entry < pred.sl
    else:
        return False
    if not ordered or pred.risk <= 0:
        return False
    p = implied_p(pred.f, pred.odds_b, mech.use_paper_kelly_inversion)
    return mech.p_bound <= p <= 1.0 - mech.p_bound


@dataclass
class Resolution:
    event: str
    t_resolve: int
    r_mae: float
    s_base: float
    s_path: float
    s_tail: float
    s_weighted: float
    tail_event: bool


def resolve(pred: Prediction, market: Market, mech: MechanismParams) -> Optional[Resolution]:
    t0, t1 = pred.t_submit + 1, pred.t_horizon
    if t1 >= market.n_minutes:
        return None
    hi = market.high[t0:t1 + 1]
    lo = market.low[t0:t1 + 1]
    d = pred.direction
    if d == 1:
        sl_hits = lo <= pred.sl
        tp1_hits = hi >= pred.tp1
        tp2_hits = hi >= pred.tp2
        adverse = pred.entry - lo
    else:
        sl_hits = hi >= pred.sl
        tp1_hits = lo <= pred.tp1
        tp2_hits = lo <= pred.tp2
        adverse = hi - pred.entry
    INF = len(hi) + 1
    sl_i = int(np.argmax(sl_hits)) if sl_hits.any() else INF
    tp1_i = int(np.argmax(tp1_hits)) if tp1_hits.any() else INF
    tp2_i = int(np.argmax(tp2_hits)) if tp2_hits.any() else INF
    sl_wins_ties = mech.same_candle_rule == "sl_first"

    def first(a: int, b: int) -> bool:
        return a < b or (a == b and sl_wins_ties)

    if sl_i < INF and (tp2_i == INF or first(sl_i, tp2_i)):
        if mech.tp1_then_sl == "tp1" and tp1_i < sl_i:
            event, term_i = "TP1", tp1_i
        else:
            event, term_i = "SL", sl_i
    elif tp2_i < INF:
        event, term_i = "TP2", tp2_i
    elif tp1_i < INF and sl_i == INF:
        event, term_i = "TP1", len(hi) - 1
    else:
        event, term_i = "NEITHER", len(hi) - 1
    mae = float(np.maximum(adverse[:term_i + 1], 0.0).max()) / pred.risk
    r_mae = min(mae, 1.0)
    if event == "SL":
        s_base = -1.0
    elif event == "TP2":
        s_base = pred.r_multiple(pred.tp2) * (1.0 + mech.delta)
    elif event == "TP1":
        s_base = pred.r_multiple(pred.tp1) * mech.lam
    else:
        s_base = pred.r_multiple(float(market.close[t1]))
    if event == "SL" and not mech.apply_path_to_stops:
        s_path = s_base
    else:
        s_path = s_base - mech.gamma * r_mae
    t_res = t0 + term_i if event != "NEITHER" else t1
    tail = market.is_tail(t_res, mech.tail_mult)
    s_tail = s_path * (1.0 + mech.tau) if tail else s_path
    s_weighted = s_tail * pred.f
    return Resolution(
        event=event, t_resolve=t_res, r_mae=r_mae,
        s_base=s_base, s_path=s_path, s_tail=s_tail,
        s_weighted=s_weighted, tail_event=tail,
    )


@dataclass
class MinerState:
    miner_id: int
    ewma: dict = field(default_factory=dict)
    last_resolution_min: Optional[int] = None
    first_resolution_min: Optional[int] = None
    balance: float = 0.0
    locked: float = 0.0
    emissions: float = 0.0
    n_resolved: int = 0
    n_rereg: int = 0

    def cap(self, mech: MechanismParams) -> float:
        s_total = sum(self.ewma.values())
        return mech.a_base + mech.a_cap_scale * max(s_total, 0.0)

    def update_score(self, regime: str, s_weighted: float, t_res: int, alpha: float):
        prev = self.ewma.get(regime, 0.0)
        self.ewma[regime] = (1.0 - alpha) * prev + alpha * s_weighted
        self.last_resolution_min = t_res if self.last_resolution_min is None \
            else max(self.last_resolution_min, t_res)
        if self.first_resolution_min is None:
            self.first_resolution_min = t_res
        self.n_resolved += 1

    def emission_score(self, t_now: int, mech: MechanismParams) -> float:
        if not self.ewma or self.last_resolution_min is None:
            return 0.0
        if mech.probation_hours > 0:
            if self.first_resolution_min is None:
                return 0.0
            if (t_now - self.first_resolution_min) / 60.0 < mech.probation_hours:
                return 0.0
        if mech.floor_clamp is not None:
            total = max(sum(max(v, -mech.floor_clamp) for v in self.ewma.values()), 0.0)
        elif mech.regime_floor:
            total = sum(max(v, 0.0) for v in self.ewma.values())
        else:
            total = max(sum(self.ewma.values()), 0.0)
        dt_hours = (t_now - self.last_resolution_min) / 60.0
        return total * float(np.exp(-dt_hours / mech.kappa_hours))


def _horizon_sigma(market: Market, t: int, h_minutes: int) -> float:
    return float(market.minute_sigma[t]) * math.sqrt(h_minutes)


@dataclass
class BaseMiner:
    miner_id: int
    label: str
    rng: np.random.Generator
    regimes: List[RegimeSpec]
    submit_prob: float = 1.0
    t_start_min: int = 0

    def maybe_submit(self, t: int, market: Market) -> List[Prediction]:
        raise NotImplementedError

    def _bracket(self, t: int, market: Market, regime: RegimeSpec, direction: int,
                 f: float, h_hours: int, sl_mult: float, tp1_mult: float,
                 tp2_mult: float) -> Prediction:
        entry = float(market.close[t])
        sig = max(_horizon_sigma(market, t, h_hours * 60), 1e-5)
        sl = entry * math.exp(-direction * sl_mult * sig)
        tp1 = entry * math.exp(direction * tp1_mult * sig)
        tp2 = entry * math.exp(direction * tp2_mult * sig)
        return Prediction(
            miner_id=self.miner_id, regime=regime.name, direction=direction,
            f=f, entry=entry, sl=sl, tp1=tp1, tp2=tp2,
            h_hours=h_hours, t_submit=t,
        )


def kelly_momentum_f(market: Market, t: int, lookback_min: int, h_hours: int,
                     sl_mult: float, tp1_mult: float,
                     leverage: float = 1.0) -> tuple[int, float, float]:
    trail = market.trailing_return(t, lookback_min)
    if abs(trail) < 1e-9:
        return 0, 0.0, 0.5
    d = 1 if trail > 0 else -1
    sig_m = max(float(market.minute_sigma[t]), 1e-9)
    mu_m = abs(trail) / lookback_min
    sig_h = sig_m * math.sqrt(h_hours * 60)
    a = tp1_mult * sig_h
    b = sl_mult * sig_h
    theta = 2.0 * mu_m / (sig_m * sig_m)
    tb, ta = theta * b, -theta * a
    if theta * max(a, b) > 700:
        p = 1.0
    else:
        denom = math.exp(ta) - math.exp(tb)
        p = (1.0 - math.exp(tb)) / denom if abs(denom) > 1e-300 else 0.5
    p = min(max(p, 0.0), 1.0)
    odds = a / b
    f = leverage * (p * (odds + 1.0) - 1.0) / odds
    f_gate = (0.99 * (odds + 1.0) - 1.0) / odds
    return d, float(np.clip(f, 0.0, min(f_gate, 0.05))), p


@dataclass
class OracleSkillMiner(BaseMiner):
    win_rate: float = 0.70
    rr: float = 1.0
    tp2_rr: float = 2.0
    h_hours: int = 168
    cadence_hours: int = 1
    sl_sigma: float = 0.35
    f: float = 0.03
    cooldown_hours: int = 0
    skip_no_setup: bool = False
    _next_ok_min: int = 0
    _pending_intent: Optional[bool] = None

    def _outcome(self, t: int, t1: int, market: Market, d: int,
                 risk: float) -> int:
        entry = float(market.close[t])
        sl = entry * math.exp(-d * risk)
        tp1 = entry * math.exp(d * self.rr * risk)
        tp2 = entry * math.exp(d * self.tp2_rr * risk)
        hi = market.high[t + 1:t1 + 1]
        lo = market.low[t + 1:t1 + 1]
        if d == 1:
            sl_hits, tp1_hits, tp2_hits = lo <= sl, hi >= tp1, hi >= tp2
        else:
            sl_hits, tp1_hits, tp2_hits = hi >= sl, lo <= tp1, lo <= tp2
        INF = len(hi) + 1
        sl_i = int(np.argmax(sl_hits)) if sl_hits.any() else INF
        tp1_i = int(np.argmax(tp1_hits)) if tp1_hits.any() else INF
        tp2_i = int(np.argmax(tp2_hits)) if tp2_hits.any() else INF
        if sl_i < INF and sl_i <= tp2_i:
            return 0
        if tp2_i < INF:
            return 2
        if tp1_i < INF and sl_i == INF:
            return 2
        return 1

    def maybe_submit(self, t: int, market: Market) -> List[Prediction]:
        if (t // 60) % self.cadence_hours != 0 or t < self._next_ok_min:
            return []
        if self.submit_prob < 1.0 and self.rng.random() > self.submit_prob:
            return []
        t1 = t + self.h_hours * 60
        if t1 >= market.n_minutes:
            return []
        sig = max(_horizon_sigma(market, t, self.h_hours * 60), 1e-6)
        risk = self.sl_sigma * sig
        outcome = {d: self._outcome(t, t1, market, d, risk) for d in (1, -1)}
        if self._pending_intent is None:
            self._pending_intent = bool(self.rng.random() < self.win_rate)
        want_win = self._pending_intent
        if want_win:
            best = max(outcome.values())
            if self.skip_no_setup and best < 2:
                return []
            cands = [d for d in (1, -1) if outcome[d] == best]
        else:
            worst = min(outcome.values())
            cands = [d for d in (1, -1) if outcome[d] == worst]
        d = cands[0] if len(cands) == 1 else (1 if self.rng.random() < 0.5 else -1)
        self._pending_intent = None
        self._next_ok_min = t + self.cooldown_hours * 60
        return [self._bracket(t, market, self.regimes[0], d, f=self.f,
                              h_hours=self.h_hours, sl_mult=self.sl_sigma,
                              tp1_mult=self.sl_sigma * self.rr,
                              tp2_mult=self.sl_sigma * self.tp2_rr)]


BG = "#0e1319"
PANEL = "#141b24"
FG = "#d7dee8"
GRID = "#2a3543"
GREEN = "#2ee88e"
RED = "#ff5d5d"
GREY = "#93a1b0"
AMBER = "#ffb454"
CYAN = "#53d8ff"


def use():
    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.facecolor": PANEL, "axes.edgecolor": GRID,
        "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": FG, "ytick.color": FG,
        "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.6,
        "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.facecolor": PANEL, "legend.edgecolor": GRID,
        "legend.framealpha": 0.9,
        "font.size": 10.5, "axes.titlesize": 11.5, "figure.titlesize": 13,
        "figure.dpi": 100,
    })


def gradient_fill(ax, x, y, color, alpha=0.5, y0=0.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rgb = mcolors.to_rgb(color)
    z = np.empty((256, 1, 4))
    z[:, 0, :3] = rgb
    z[:, 0, 3] = np.linspace(0.0, alpha, 256)
    ymax = float(np.nanmax(y))
    if ymax <= y0:
        return
    im = ax.imshow(z, aspect="auto", origin="lower", zorder=1,
                   extent=[float(x[0]), float(x[-1]), y0, ymax])
    poly = ax.fill_between(x, y0, y, facecolor="none", edgecolor="none")
    im.set_clip_path(poly.get_paths()[0], transform=ax.transData)


def heat(ax, M, xticks, yticks, cmap="magma", fmt="{:.0f}", notes=None,
         vmin=None, vmax=None):
    M = np.asarray(M, dtype=float)
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xticks)), xticks)
    ax.set_yticks(range(len(yticks)), yticks)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                txt, bright = "—", False
            else:
                txt = fmt.format(M[i, j])
                if notes is not None and notes[i][j]:
                    txt += notes[i][j]
                r, g, b, _ = im.cmap(im.norm(M[i, j]))
                bright = 0.299 * r + 0.587 * g + 0.114 * b > 0.5
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                    color="#10151c" if bright else FG)
    ax.grid(False)
    return im


def mark_cell(ax, col, row, color=CYAN, label=None):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                           edgecolor=color, lw=2.2, zorder=5))
    if label:
        ax.text(col, row - 0.62, label, ha="center", fontsize=8.5,
                color=color, zorder=5)


TAO_PER_DAY = 10.0
UNITS_PER_DAY = 24.0
TAO_PER_UNIT = TAO_PER_DAY / UNITS_PER_DAY

REGIME_BY_H = {24: "B", 48: "B", 72: "C", 168: "C"}


def load_real_market(path: str = CACHE_1M, days: int | None = None) -> Market:
    z = np.load(path)
    close = z["close"].astype(np.float64)
    high = z["high"].astype(np.float64)
    low = z["low"].astype(np.float64)
    if days is not None:
        n = days * MINUTES_PER_DAY
        close, high, low = close[-n:], high[-n:], low[-n:]
    ret = np.zeros_like(close)
    ret[1:] = np.diff(np.log(close))
    sigma_24h = _rolling_std(ret, MINUTES_PER_DAY, MINUTES_PER_DAY // 2) * math.sqrt(MINUTES_PER_DAY)
    sigma_30d = _rolling_std(ret, 30 * MINUTES_PER_DAY, 5 * MINUTES_PER_DAY) * math.sqrt(MINUTES_PER_DAY)
    minute_sigma = _rolling_std(ret, 360, 60)
    fb = max(float(np.median(minute_sigma[minute_sigma > 0])), 1e-6)
    minute_sigma = np.where(minute_sigma > 0, minute_sigma, fb)
    return Market(close=close, high=high, low=low, sigma_24h=sigma_24h,
                  sigma_30d=sigma_30d, minute_sigma=minute_sigma)


@dataclass
class BetaMomentumMiner(BaseMiner):
    lookback_min: int = 4320
    h_hours: int = 72
    cadence_hours: int = 24
    f: float = 0.40
    trend_threshold: float = 0.5
    kelly_leverage: Optional[float] = None
    f_min: float = 0.05
    sl_mult: float = 1.0
    tp1_mult: float = 1.0
    tp2_mult: float = 2.0

    def maybe_submit(self, t: int, market: Market) -> List[Prediction]:
        if (t // 60) % self.cadence_hours != 0:
            return []
        trail = market.trailing_return(t, self.lookback_min)
        sig_lb = max(float(market.minute_sigma[t]) * math.sqrt(self.lookback_min), 1e-9)
        if abs(trail) < self.trend_threshold * sig_lb:
            return []
        d = 1 if trail > 0 else -1
        f = self.f
        if self.kelly_leverage is not None:
            d2, f, _p = kelly_momentum_f(market, t, self.lookback_min,
                                         self.h_hours, sl_mult=self.sl_mult,
                                         tp1_mult=self.tp1_mult,
                                         leverage=self.kelly_leverage)
            if d2 == 0 or f < self.f_min:
                return []
            d = d2
        regime = next(r for r in self.regimes
                      if r.name == REGIME_BY_H[self.h_hours])
        return [self._bracket(t, market, regime, d, f=f,
                              h_hours=self.h_hours,
                              sl_mult=self.sl_mult, tp1_mult=self.tp1_mult,
                              tp2_mult=self.tp2_mult)]


@dataclass
class OpportunistMiner(BaseMiner):
    kind: str = "coinflip"
    cadence_hours: int = 4

    def maybe_submit(self, t: int, market: Market) -> List[Prediction]:
        if (t // 60) % self.cadence_hours != 0:
            return []
        d = 1 if self.rng.random() < 0.5 else -1
        h = int(self.rng.integers(72, 169))
        regime = next(r for r in self.regimes if r.name == "C")
        if self.kind == "lottery":
            return [self._bracket(t, market, regime, d, f=0.05, h_hours=h,
                                  sl_mult=0.8, tp1_mult=1.6, tp2_mult=4.0)]
        return [self._bracket(t, market, regime, d, f=0.02, h_hours=h,
                              sl_mult=1.0, tp1_mult=1.0, tp2_mult=2.0)]


STRATEGY_MENU = ["mom24", "mom72", "mom168", "coinflip", "lottery"]

TUNED_MOM = dict(lookback_min=10080, trend_threshold=0.0,
                 sl_mult=0.5, tp1_mult=1.0, tp2_mult=2.0,
                 cadence_hours=8, f=0.05)


def make_strategy(kind: str, mid: int, rng: np.random.Generator,
                  regimes, kelly_leverage: float | None = None) -> BaseMiner:
    if kind == "tmom72":
        return BetaMomentumMiner(mid, "tmom72", rng, regimes,
                                 h_hours=72, **TUNED_MOM)
    if kind == "tmom168":
        return BetaMomentumMiner(mid, "tmom168", rng, regimes,
                                 h_hours=168, **TUNED_MOM)
    if kind == "mom24":
        return BetaMomentumMiner(mid, "mom24", rng, regimes,
                                 lookback_min=1440, h_hours=24, cadence_hours=8,
                                 kelly_leverage=kelly_leverage)
    if kind == "mom72":
        return BetaMomentumMiner(mid, "mom72", rng, regimes,
                                 lookback_min=4320, h_hours=72, cadence_hours=24,
                                 kelly_leverage=kelly_leverage)
    if kind == "mom168":
        return BetaMomentumMiner(mid, "mom168", rng, regimes,
                                 lookback_min=10080, h_hours=168, cadence_hours=48,
                                 kelly_leverage=kelly_leverage)
    if kind in ("coinflip", "lottery"):
        return OpportunistMiner(mid, kind, rng, regimes, kind=kind)
    raise ValueError(kind)


@dataclass
class Agent:
    miner: BaseMiner
    patience_days: float
    belief_mult: float
    entered_min: Optional[int] = None
    quit: bool = False
    quit_min: Optional[int] = None
    burns_paid: float = 0.0
    op_cost_accrued: float = 0.0

    @property
    def active(self) -> bool:
        return self.entered_min is not None and not self.quit


GATE_MIN_N = 6
GATE_FP = 0.05


def _t_window(rs: np.ndarray) -> float:
    sd = float(np.std(rs, ddof=1))
    if sd <= 1e-12:
        return float("-inf")
    return float(np.mean(rs)) / sd * math.sqrt(len(rs))


def t_stat_returns(rs: np.ndarray, min_n: int = GATE_MIN_N) -> float:
    n = len(rs)
    if n < min_n:
        return float("-inf")
    return max(_t_window(rs), _t_window(rs[-min_n:]))


def t_stat_alpha(rs: np.ndarray, bmkt: np.ndarray, bmom: np.ndarray,
                 min_n: int = GATE_MIN_N) -> float:
    n = len(rs)
    if n < min_n + 2:
        return float("-inf")
    X = np.column_stack([np.ones(n), bmkt, bmom])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ rs)
    resid = rs - X @ beta
    dof = n - X.shape[1]
    if dof < 2:
        return float("-inf")
    s2 = float(resid @ resid) / dof
    se0 = math.sqrt(max(s2 * XtX_inv[0, 0], 1e-24))
    return float(beta[0]) / se0


def gate_threshold_z(mode, n_keys: int, fp: float = GATE_FP) -> float:
    if mode == "deflated":
        from statistics import NormalDist
        return NormalDist().inv_cdf(1.0 - fp / max(n_keys, 1))
    return float(mode)


@dataclass
class RationalResult:
    active_noise: np.ndarray
    skilled_share: np.ndarray
    noise_share: np.ndarray
    epoch_t_min: np.ndarray
    agents: List[Agent]
    labels: Dict[int, str]
    emissions: Dict[int, float]
    alpha_pnl: Dict[int, float]
    skilled_id: int
    skilled_burns: float
    n_entries: int
    n_quits: int
    n_rereg: int
    unallocated: Optional[np.ndarray] = None
    skilled_clear_day: Optional[float] = None
    noise_ever_cleared: int = 0
    noise_cleared_epochs: float = 0.0
    cleared_ids: Optional[set] = None
    em_timeline: Optional[Dict[int, np.ndarray]] = None
    n_resolutions: Optional[Dict[int, int]] = None


def run_rational(market: Market, mech: MechanismParams, burn_units: float,
                 seed: int = 0, n_prospects: int = 240,
                 entries_per_day: int = 3, op_cost_units_day: float = 0.5,
                 skilled_join_day: float = 14.0,
                 eval_window_d: float = 14.0,
                 min_tenure_d: float = 14.0,
                 kelly_leverage: float | None = None,
                 menu: List[str] | None = None,
                 belief_sigma: float = 0.3,
                 marginal_obs: bool = False,
                 skilled_overrides: dict | None = None,
                 gate: str | None = None,
                 gate_mode="deflated",
                 dust_frac: float = 0.02,
                 gate_block_h: float | None = None,
                 gate_min_n: int = GATE_MIN_N,
                 gate_fp: float = GATE_FP,
                 gate_exit_z: float | None = None) -> RationalResult:
    rng = np.random.default_rng(seed)
    regimes = list(DEFAULT_REGIMES)
    menu = menu or STRATEGY_MENU

    agents: List[Agent] = []
    for i in range(n_prospects):
        kind = menu[i % len(menu)]
        miner = make_strategy(kind, i, np.random.default_rng(rng.integers(2**63)),
                              regimes, kelly_leverage=kelly_leverage)
        agents.append(Agent(
            miner=miner,
            patience_days=float(np.clip(rng.lognormal(math.log(60), 0.5), 20, 240)),
            belief_mult=(float(rng.lognormal(0.0, belief_sigma))
                         if belief_sigma > 0 else 1.0),
        ))

    skilled_kwargs = dict(
        win_rate=0.70, rr=1.0, tp2_rr=2.0, h_hours=168, cadence_hours=12,
        cooldown_hours=84, skip_no_setup=True, f=0.03, sl_sigma=0.35)
    skilled_kwargs.update(skilled_overrides or {})
    skilled = OracleSkillMiner(
        miner_id=n_prospects, label="skilled",
        rng=np.random.default_rng(seed * 65537 + 3), regimes=[regimes[2]],
        **skilled_kwargs)
    skilled.t_start_min = int(skilled_join_day * MINUTES_PER_DAY)
    all_miners: Dict[int, BaseMiner] = {a.miner.miner_id: a.miner for a in agents}
    all_miners[skilled.miner_id] = skilled
    labels = {mid: m.label for mid, m in all_miners.items()}

    states: Dict[int, MinerState] = {}
    epoch_minutes = int(mech.epoch_hours * 60)
    max_h_min = max(r.max_h for r in regimes) * 60
    epochs = list(range(epoch_minutes, market.n_minutes - max_h_min - 1,
                        epoch_minutes))
    n_ep = len(epochs)
    em_tl: Dict[int, np.ndarray] = {}
    active_noise = np.zeros(n_ep)
    skilled_share = np.zeros(n_ep)
    noise_share = np.zeros(n_ep)
    win_ep = int(eval_window_d * UNITS_PER_DAY)

    unallocated = np.zeros(n_ep)
    hist: Dict[int, list] = {}
    stat_cache: Dict[int, tuple] = {}
    skilled_clear_day: Optional[float] = None
    noise_cleared_ever: set = set()
    noise_cleared_epochs = 0.0
    latched: set = set()

    def gate_stat(mid: int) -> float:
        h = hist.get(mid)
        if not h:
            return float("-inf")
        cached = stat_cache.get(mid)
        if cached is not None and cached[0] == len(h):
            return cached[1]
        arr = np.asarray(h)
        if gate_block_h:
            blk = (arr[:, 3] // (gate_block_h * 60)).astype(np.int64)
            _uniq, inv = np.unique(blk, return_inverse=True)
            agg = np.zeros((len(_uniq), 3))
            np.add.at(agg, inv, arr[:, :3])
            data = agg
        else:
            data = arr[:, :3]
        v = (t_stat_returns(data[:, 0], gate_min_n) if gate == "ret"
             else t_stat_alpha(data[:, 0], data[:, 1], data[:, 2], gate_min_n))
        stat_cache[mid] = (len(h), v)
        return v

    def trailing_rate(mid: int, ei: int) -> float:
        tl = em_tl.get(mid)
        if tl is None:
            return 0.0
        lo = max(0, ei - win_ep)
        if ei <= lo:
            return 0.0
        return float(tl[lo:ei].sum()) / ((ei - lo) / UNITS_PER_DAY)

    pending: List[Prediction] = []
    n_entries = n_quits = n_rereg = 0
    skilled_burns = burn_units

    for ei, t in enumerate(epochs):
        if ei % int(UNITS_PER_DAY) == 0:
            actives = [a for a in agents if a.active]
            n_act = len(actives) + (1 if t >= skilled.t_start_min else 0)
            if actives:
                rates = [trailing_rate(a.miner.miner_id, ei) for a in actives]
                if marginal_obs:
                    obs_rate = float(np.sum(rates)) / (len(actives) + 1)
                else:
                    obs_rate = float(np.mean(rates))
            else:
                obs_rate = UNITS_PER_DAY / (n_act + 1)
            for a in actives:
                tenure_d = (t - a.entered_min) / MINUTES_PER_DAY
                mid = a.miner.miner_id
                if tenure_d >= min_tenure_d and \
                        trailing_rate(mid, ei) < op_cost_units_day:
                    a.quit = True
                    a.quit_min = t
                    n_quits += 1
            slots = entries_per_day
            order = rng.permutation(len(agents))
            for j in order:
                if slots <= 0:
                    break
                a = agents[j]
                if a.entered_min is not None or a.quit:
                    continue
                ev = (a.belief_mult * obs_rate - op_cost_units_day) \
                    * a.patience_days - burn_units
                if ev > 0:
                    a.entered_min = t
                    a.quit_min = None
                    a.miner.t_start_min = t
                    a.burns_paid += burn_units
                    states[a.miner.miner_id] = MinerState(
                        miner_id=a.miner.miner_id, balance=mech.a_base)
                    em_tl[a.miner.miner_id] = np.zeros(n_ep)
                    n_entries += 1
                    slots -= 1
        if t >= skilled.t_start_min and skilled.miner_id not in states:
            states[skilled.miner_id] = MinerState(
                miner_id=skilled.miner_id, balance=mech.a_base)
            em_tl[skilled.miner_id] = np.zeros(n_ep)

        for a in agents:
            if a.active:
                a.op_cost_accrued += op_cost_units_day / UNITS_PER_DAY

        due = [p for p in pending if p.t_horizon <= t]
        pending = [p for p in pending if p.t_horizon > t]
        loss_pool, winners = 0.0, []
        for pred in due:
            res = resolve(pred, market, mech)
            if res is None:
                continue
            st = states[pred.miner_id]
            st.update_score(pred.regime, res.s_weighted, res.t_resolve,
                            mech.ewma_alpha)
            if gate is not None:
                t0p, t1p = pred.t_submit, pred.t_horizon
                sig_h = max(_horizon_sigma(market, t0p, t1p - t0p), 1e-9)
                b_mkt = math.log(max(float(market.close[t1p]), 1e-12)
                                 / max(float(market.close[t0p]), 1e-12)) / sig_h
                trail = market.trailing_return(t0p, 4320)
                b_mom = (1.0 if trail > 0 else -1.0) * b_mkt
                hist.setdefault(pred.miner_id, []).append(
                    (res.s_base, b_mkt, b_mom, float(res.t_resolve)))
            if mech.staking_enabled and pred.stake > 0:
                st.locked -= pred.stake
                if res.s_weighted > 0:
                    st.balance += pred.stake
                    winners.append((pred.miner_id, res.s_weighted))
                elif res.s_weighted < 0:
                    loss_pool += pred.stake
                else:
                    st.balance += pred.stake
        if mech.staking_enabled and loss_pool > 0 and winners:
            tot = sum(s for _m, s in winners)
            for mid, s in winners:
                states[mid].balance += loss_pool * s / tot

        submitters = [a.miner for a in agents if a.active]
        if t >= skilled.t_start_min:
            submitters.append(skilled)
        for miner in submitters:
            st = states[miner.miner_id]
            if mech.staking_enabled and st.balance <= 0:
                a = next((x for x in agents if x.miner.miner_id == miner.miner_id),
                         None)
                if a is None:
                    st.balance = mech.a_base
                    st.ewma = {}
                    st.first_resolution_min = st.last_resolution_min = None
                    st.n_rereg += 1
                    skilled_burns += burn_units
                    continue
                actives = [x for x in agents if x.active and x is not a]
                obs = (float(np.mean([trailing_rate(x.miner.miner_id, ei)
                                      for x in actives]))
                       if actives else UNITS_PER_DAY / 2)
                rem_d = a.patience_days - (t - a.entered_min) / MINUTES_PER_DAY
                if rem_d > 0 and (a.belief_mult * obs - op_cost_units_day) \
                        * rem_d - burn_units > 0:
                    st.balance = mech.a_base
                    st.ewma = {}
                    st.first_resolution_min = st.last_resolution_min = None
                    st.n_rereg += 1
                    a.burns_paid += burn_units
                    n_rereg += 1
                else:
                    a.quit = True
                    a.quit_min = t
                    n_quits += 1
                continue
            for pred in miner.maybe_submit(t, market):
                if not validate(pred, mech):
                    continue
                if mech.staking_enabled:
                    stake = pred.f * min(st.balance, st.cap(mech))
                    if stake <= 0:
                        continue
                    pred.stake = stake
                    st.balance -= stake
                    st.locked += stake
                pending.append(pred)

        scores = {mid: st.emission_score(t, mech) for mid, st in states.items()}
        if gate is None:
            total = sum(scores.values())
            if total > 0:
                for mid, s in scores.items():
                    w = s / total
                    states[mid].emissions += w
                    em_tl[mid][ei] = w
                sk_w = scores.get(skilled.miner_id, 0.0) / total
                skilled_share[ei] = sk_w
                noise_share[ei] = 1.0 - sk_w
        else:
            thr = gate_threshold_z(gate_mode, len(states), gate_fp)
            pos = {mid: s for mid, s in scores.items() if s > 0}
            g = {}
            for mid in pos:
                st_v = gate_stat(mid)
                ok = st_v >= thr
                if gate_exit_z is not None:
                    if mid in latched and st_v >= gate_exit_z:
                        ok = True
                    if ok:
                        latched.add(mid)
                    else:
                        latched.discard(mid)
                g[mid] = 1.0 if ok else 0.0
            cleared = {mid for mid, gv in g.items() if gv >= 1.0}
            for mid in cleared:
                if mid == skilled.miner_id:
                    if skilled_clear_day is None:
                        skilled_clear_day = t / MINUTES_PER_DAY
                else:
                    noise_cleared_ever.add(mid)
                    noise_cleared_epochs += 1
            paid: Dict[int, float] = {}
            ctot = sum(pos[mid] for mid in cleared)
            if ctot > 0:
                for mid in cleared:
                    paid[mid] = (1.0 - dust_frac) * pos[mid] / ctot
            dust = {mid: s for mid, s in pos.items() if g[mid] < 1.0}
            dtot = sum(dust.values())
            if dtot > 0:
                for mid, s in dust.items():
                    paid[mid] = paid.get(mid, 0.0) + dust_frac * s / dtot
            paid_total = 0.0
            for mid, w in paid.items():
                states[mid].emissions += w
                em_tl[mid][ei] = w
                paid_total += w
            skilled_share[ei] = paid.get(skilled.miner_id, 0.0)
            noise_share[ei] = paid_total - skilled_share[ei]
            unallocated[ei] = 1.0 - paid_total
        active_noise[ei] = sum(1 for a in agents if a.active)

    alpha_pnl = {}
    for mid, st in states.items():
        credit = mech.a_base * (1 + st.n_rereg)
        alpha_pnl[mid] = (st.balance + st.locked - credit
                          if mech.staking_enabled else 0.0)

    return RationalResult(
        active_noise=active_noise, skilled_share=skilled_share,
        noise_share=noise_share,
        epoch_t_min=np.asarray(epochs, dtype=float), agents=agents,
        labels=labels,
        emissions={mid: st.emissions for mid, st in states.items()},
        alpha_pnl=alpha_pnl, skilled_id=skilled.miner_id,
        skilled_burns=skilled_burns, n_entries=n_entries, n_quits=n_quits,
        n_rereg=n_rereg,
        unallocated=unallocated if gate is not None else None,
        skilled_clear_day=skilled_clear_day,
        noise_ever_cleared=len(noise_cleared_ever),
        noise_cleared_epochs=noise_cleared_epochs,
        cleared_ids=(noise_cleared_ever
                     | ({skilled.miner_id} if skilled_clear_day is not None
                        else set())),
        em_timeline=em_tl,
        n_resolutions={mid: len(h) for mid, h in hist.items()})


def summarize(res: RationalResult, burn_units: float, days: float) -> dict:
    q3 = int(len(res.skilled_share) * 0.75)
    entered = [a for a in res.agents if a.entered_min is not None]
    per_strat: Dict[str, list] = {}
    for a in entered:
        mid = a.miner.miner_id
        net = (res.emissions.get(mid, 0.0)
               + max(res.alpha_pnl.get(mid, 0.0), 0.0)
               - a.burns_paid - a.op_cost_accrued)
        per_strat.setdefault(a.miner.label, []).append(net)

    sk_em = res.emissions.get(res.skilled_id, 0.0)
    out = {
        "burn_units": burn_units,
        "burn_tao": burn_units * TAO_PER_UNIT,
        "n_entries": res.n_entries,
        "n_quits": res.n_quits,
        "n_rereg": res.n_rereg,
        "eq_crowd": float(res.active_noise[q3:].mean()),
        "skilled_steady_share": float(res.skilled_share[q3:].mean()),
        "skilled_net_tao": (sk_em - res.skilled_burns) * TAO_PER_UNIT,
        "noise_total_burn_tao": sum(a.burns_paid for a in entered) * TAO_PER_UNIT,
        "noise_total_emis_tao": sum(res.emissions.get(a.miner.miner_id, 0.0)
                                    for a in entered) * TAO_PER_UNIT,
        "entrants_cleared": sum(
            1 for a in entered
            if a.miner.miner_id in (res.cleared_ids or set())),
    }
    for lab in ("mom24", "mom72", "mom168", "tmom72", "tmom168",
                "coinflip", "lottery"):
        vals = per_strat.get(lab, [])
        out[f"{lab}_n"] = len(vals)
        out[f"{lab}_net_pc_tao"] = (float(np.mean(vals)) * TAO_PER_UNIT
                                    if vals else float("nan"))
    return out


def _cell_rational(args) -> dict:
    burn, seed, days, mech_over, kelly_lev = args
    market = load_real_market(days=None)
    mech = replace(MechanismParams(), **mech_over)
    res = run_rational(market, mech, burn_units=burn, seed=seed,
                       kelly_leverage=kelly_lev)
    row = summarize(res, burn, days)
    row["seed"] = seed
    row["kelly_leverage"] = kelly_lev if kelly_lev is not None else 0.0
    return row


def rational_main(argv=None):
    ap = argparse.ArgumentParser(description="Rational-entry sim on real BTC data.")
    ap.add_argument("--burns", type=float, nargs="+",
                    default=[48.0, 120.0, 240.0, 480.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--kelly-leverage", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--outdir", default="flow_sim/out")
    args = ap.parse_args(argv)

    mech_over = dict(gamma=0.15, tau=0.30, probation_hours=0.0,
                     ewma_alpha=0.10,
                     floor_clamp=float("inf"), kappa_hours=672.0)
    market = load_real_market()
    days = market.n_minutes / MINUTES_PER_DAY
    print(f"real tape: {days:.0f} days, close {market.close[0]:.0f} .. "
          f"{market.close[-1]:.0f}")

    cells = [(b, s, days, mech_over, args.kelly_leverage)
             for b in args.burns for s in args.seeds]
    t0 = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, row in enumerate(ex.map(_cell_rational, cells), 1):
            rows.append(row)
            print(f"  [{i}/{len(cells)}] {time.monotonic()-t0:.0f}s", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "rational_entry.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path}")
    agg = df.groupby("burn_tao").agg(
        eq_crowd=("eq_crowd", "mean"),
        entries=("n_entries", "mean"), quits=("n_quits", "mean"),
        skilled_share=("skilled_steady_share", "mean"),
        skilled_net_tao=("skilled_net_tao", "mean"),
        noise_burn_tao=("noise_total_burn_tao", "mean"),
        noise_emis_tao=("noise_total_emis_tao", "mean"),
        mom24=("mom24_net_pc_tao", "mean"),
        mom72=("mom72_net_pc_tao", "mean"),
        mom168=("mom168_net_pc_tao", "mean"),
        coinflip=("coinflip_net_pc_tao", "mean"),
        lottery=("lottery_net_pc_tao", "mean")).round(2)
    print(agg.to_string())


SHIP_MECH = dict(gamma=0.25, delta=0.30, lam=0.50, tau=1.00,
                 ewma_alpha=0.05, kappa_hours=672.0, probation_hours=168.0,
                 floor_clamp=float("inf"))

GATE_KW = dict(gate="ret", gate_mode=1.25, gate_block_h=168.0,
               gate_min_n=4, gate_exit_z=0.5, dust_frac=0.02)

FINALSPEC_CELLS = (
    [("headline-70", 0.70, 0.0, 1, s) for s in range(10)]
    + [("headline-50", 0.50, 0.0, 1, s) for s in range(10)]
    + [(f"burn-{m}x", 0.50, m, 1, s) for m in (0.5, 1.5, 3.0)
       for s in range(5)]
    + [("crowd-3x", 0.50, 0.0, 3, s) for s in range(5)]
)


def _cell_finalspec(args):
    name, wr, burn_mult, epd, seed = args
    market = load_real_market()
    mech = replace(MechanismParams(), **SHIP_MECH)
    burn = burn_mult * UNITS_PER_DAY
    res = run_rational(
        market, mech, burn_units=burn, seed=seed,
        entries_per_day=epd, skilled_join_day=0.0, kelly_leverage=1.0,
        menu=["tmom72", "tmom168"], belief_sigma=0.0, marginal_obs=True,
        skilled_overrides=dict(win_rate=wr), **GATE_KW)
    days = market.n_minutes / MINUTES_PER_DAY
    row = summarize(res, burn, days)
    q3 = int(len(res.skilled_share) * 0.75)
    ev = [row[f"{lab}_net_pc_tao"] / TAO_PER_DAY
          for lab in ("tmom72", "tmom168") if row[f"{lab}_n"] > 0]
    row.update(
        cell=name, wr=wr, burn_mult=burn_mult, entries_per_day=epd,
        seed=seed,
        clear_day=res.skilled_clear_day,
        cleared=int(res.skilled_clear_day is not None),
        noise_passes=res.noise_ever_cleared,
        skilled_paid_pd=res.emissions.get(res.skilled_id, 0.0) / UNITS_PER_DAY,
        noise_paid_pd=row["noise_total_emis_tao"] / TAO_PER_DAY,
        mom_ev_pd=float(np.mean(ev)) if ev else float("nan"),
        unalloc_steady=(float(res.unallocated[q3:].mean())
                        if res.unallocated is not None else 0.0),
    )
    tl = dict(day=res.epoch_t_min / MINUTES_PER_DAY,
              skilled=res.skilled_share, noise=res.noise_share,
              withheld=(res.unallocated if res.unallocated is not None
                        else np.zeros_like(res.skilled_share)))
    return name, seed, row, tl


def finalspec_main():
    t0 = time.monotonic()
    rows, tls = [], {}
    with ProcessPoolExecutor(max_workers=min(len(FINALSPEC_CELLS), os.cpu_count() or 4)) as ex:
        for i, (name, seed, row, tl) in enumerate(ex.map(_cell_finalspec, FINALSPEC_CELLS), 1):
            rows.append(row)
            tls[(name, seed)] = tl
            print(f"  [{i}/{len(FINALSPEC_CELLS)}] {time.monotonic()-t0:.0f}s", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "finalspec.csv"), index=False)

    def agg(g):
        cd = g.loc[g["clear_day"].notna(), "clear_day"]
        return pd.Series(dict(
            cleared=f"{int(g['cleared'].sum())}/{len(g)}",
            clear_mean=cd.mean(), clear_median=cd.median(),
            clear_worst=cd.max(),
            steady_share=g["skilled_steady_share"].mean(),
            noise_paid_pd=g["noise_paid_pd"].mean(),
            mom_ev_pd=g["mom_ev_pd"].mean(),
            noise_passes=int(g["noise_passes"].sum()),
            entries=g["n_entries"].mean(),
            eq_crowd=g["eq_crowd"].mean(),
        ))

    table = df.groupby("cell").apply(agg, include_groups=False)
    order = ["headline-70", "headline-50", "burn-0.5x", "burn-1.5x",
             "burn-3.0x", "crowd-3x"]
    table = table.reindex([c for c in order if c in table.index]).round(2)
    print("\n", table.to_string())

    fs.use()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, cell, title in zip(
            axes, ("headline-70", "headline-50"),
            ("70% win rate @ 2:1, ~2 trades/week",
             "50% win rate @ 2:1, ~2 trades/week")):
        seeds = [s for (n, s) in tls if n == cell]
        day = tls[(cell, seeds[0])]["day"]
        for key, color, lab, fill in (
                ("skilled", fs.GREEN, "skilled miner", 0.40),
                ("noise", fs.RED, "leveraged beta adversaries (paid)", 0.35),
                ("withheld", fs.GREY, "withheld (burned)", 0.0)):
            m = np.mean([tls[(cell, s)][key] for s in seeds], axis=0)
            k = int(UNITS_PER_DAY)
            sm = np.convolve(m, np.ones(k) / k, mode="same")
            ax.plot(day, sm, color=color, lw=1.9, label=lab, zorder=3,
                    ls="--" if not fill else "-")
            if fill:
                fs.gradient_fill(ax, day, sm, color, alpha=fill)
        cds = df.loc[(df["cell"] == cell) & df["clear_day"].notna(),
                     "clear_day"]
        if len(cds):
            ax.axvline(float(cds.mean()), color=fs.AMBER, ls=":", lw=1.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("day")
        ax.margins(x=0.01)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("share of daily emissions")
    axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle("FLOW ship spec on the two-year BTC tape — emission share "
                 "(mean of 10 seeds; dotted line = mean gate-clear day)",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "finalspec_ramp.png")
    fig.savefig(path, dpi=140)
    print(f"\nwrote {os.path.join(OUT, 'finalspec.csv')}\nwrote {path}")


PAPERGRID_BASE = dict(wr=0.50, burn_mult=0.0, thr=1.25, exit_z=0.5, block_h=168.0,
                      min_n=4, epd=1, dust=0.02, cad=12, cool=84, stat="ret",
                      gamma=0.25, tau=1.00, join_day=0.0)


def _grid_cells():
    cells = []

    def add(study, seeds, **over):
        for s in seeds:
            cells.append((study, {**PAPERGRID_BASE, **over}, s))

    for wr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        add("wr", range(10), wr=wr)
    for thr in (1.25, 1.5, 1.75, 2.0, 2.5):
        for ez in (None, 0.25, 0.5, 1.0):
            add("threshold", range(5), thr=thr, exit_z=ez)
    for bh in (84.0, 168.0, 336.0):
        for wr in (0.50, 0.70):
            add("block", range(5), block_h=bh, wr=wr)
    for mn in (2, 4, 6, 8):
        add("minn", range(5), min_n=mn)
    for bm in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        add("bond", range(5), burn_mult=bm)
    for epd in (1, 2, 3, 5):
        add("pressure", range(5), epd=epd)
    for d in (0.0, 0.01, 0.02, 0.05):
        add("dust", range(5), dust=d)
    for cad, cool in ((24, 168), (12, 84), (8, 36), (4, 16)):
        for wr in (0.50, 0.70):
            add("cadence", range(5), cad=cad, cool=cool, wr=wr)
    add("advonly", range(10), join_day=1e5)
    for st in ("ret", "alpha"):
        for wr in (0.50, 0.70):
            add("stat", range(5), stat=st, wr=wr)
    for wr in (0.50, 0.70):
        add("nogate", range(5), stat="none", wr=wr)
    for g in (0.15, 0.25, 0.40):
        for tau in (0.30, 1.00):
            add("gammatau", (0, 1, 2), gamma=g, tau=tau)
    return cells


def _cell_papergrid(args):
    study, c, seed = args
    market = load_real_market()
    mech = replace(MechanismParams(), gamma=c["gamma"], delta=0.30, lam=0.50,
                   tau=c["tau"], ewma_alpha=0.05, kappa_hours=672.0,
                   probation_hours=168.0,
                   floor_clamp=float("inf"))
    burn = c["burn_mult"] * UNITS_PER_DAY
    gate = None if c["stat"] == "none" else c["stat"]
    res = run_rational(
        market, mech, burn_units=burn, seed=seed,
        entries_per_day=c["epd"], skilled_join_day=c["join_day"],
        kelly_leverage=1.0, menu=["tmom72", "tmom168"], belief_sigma=0.0,
        marginal_obs=True,
        skilled_overrides=dict(win_rate=c["wr"], cadence_hours=c["cad"],
                               cooldown_hours=c["cool"]),
        gate=gate, gate_mode=c["thr"], gate_block_h=c["block_h"],
        gate_min_n=c["min_n"], gate_exit_z=c["exit_z"],
        dust_frac=c["dust"])
    days = market.n_minutes / MINUTES_PER_DAY
    s = summarize(res, burn, days)
    q3 = int(len(res.skilled_share) * 0.75)
    ev = [s[f"{lab}_net_pc_tao"] / TAO_PER_DAY
          for lab in ("tmom72", "tmom168") if s[f"{lab}_n"] > 0]
    row = dict(study=study, seed=seed, **{k: (float("nan") if v is None
                                              else v) for k, v in c.items()})
    row.update(
        cleared=int(res.skilled_clear_day is not None),
        clear_day=res.skilled_clear_day,
        steady_share=s["skilled_steady_share"],
        sk_paid_pd=res.emissions.get(res.skilled_id, 0.0) / UNITS_PER_DAY,
        sk_trades=(res.n_resolutions or {}).get(res.skilled_id, 0),
        noise_paid_pd=s["noise_total_emis_tao"] / TAO_PER_DAY,
        mom_ev_pd=float(np.mean(ev)) if ev else float("nan"),
        noise_passes=res.noise_ever_cleared,
        noise_pass_epochs=res.noise_cleared_epochs,
        entries=s["n_entries"], quits=s["n_quits"], rereg=s["n_rereg"],
        eq_crowd=s["eq_crowd"],
        unalloc_steady=(float(res.unallocated[q3:].mean())
                        if res.unallocated is not None else 0.0),
    )
    return row


def papergrid_main():
    cells = _grid_cells()
    print(f"{len(cells)} runs")
    t0 = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for i, row in enumerate(ex.map(_cell_papergrid, cells, chunksize=4), 1):
            rows.append(row)
            if i % 40 == 0 or i == len(cells):
                print(f"  [{i}/{len(cells)}] {time.monotonic()-t0:.0f}s",
                      flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "papergrid.csv"), index=False)

    keycols = dict(
        wr=["wr"], threshold=["thr", "exit_z"], block=["wr", "block_h"],
        minn=["min_n"], bond=["burn_mult"], pressure=["epd"], dust=["dust"],
        cadence=["wr", "cad", "cool"], advonly=[], stat=["wr", "stat"],
        nogate=["wr"], gammatau=["gamma", "tau"])

    def agg(g):
        cd = g.loc[g["clear_day"].notna(), "clear_day"]
        return pd.Series(dict(
            n=len(g),
            cleared=int(g["cleared"].sum()),
            clear_mean=cd.mean(), clear_med=cd.median(), clear_max=cd.max(),
            steady=g["steady_share"].mean(),
            sk_trades=g["sk_trades"].mean(),
            noise_pd=g["noise_paid_pd"].mean(),
            mom_ev=g["mom_ev_pd"].mean(),
            passes=int(g["noise_passes"].sum()),
            pass_ep=g["noise_pass_epochs"].sum(),
            entries=g["entries"].mean(),
            crowd=g["eq_crowd"].mean(),
            withheld=g["unalloc_steady"].mean(),
        ))

    for study, cols in keycols.items():
        sub = df[df["study"] == study]
        if sub.empty:
            continue
        print(f"\n===== {study} =====")
        if cols:
            print(sub.groupby(cols, dropna=False)
                  .apply(agg, include_groups=False).round(2).to_string())
        else:
            print(agg(sub).round(2).to_string())

    fs.use()
    wr = df[df["study"] == "wr"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ok = wr[wr["cleared"] == 1]
    sc = ax.scatter(ok["wr"], ok["clear_day"], c=ok["clear_day"],
                    cmap="plasma_r", s=26, alpha=0.9, zorder=3,
                    edgecolors=fs.BG, linewidths=0.4)
    bad = wr[wr["cleared"] == 0]
    ax.scatter(bad["wr"], [358] * len(bad), color=fs.RED, marker="x", s=36,
               zorder=3)
    m = (wr[wr["cleared"] == 1].groupby("wr")["clear_day"]
         .agg(["mean", "median"]))
    ax.plot(m.index, m["mean"], color=fs.GREEN, lw=2.2, label="mean clear day")
    ax.plot(m.index, m["median"], color=fs.GREEN, lw=1.4, ls="--",
            label="median")
    ax.axvline(1 / 3, color=fs.GREY, lw=1.0, ls=":",
               label="zero-edge boundary (wr = 1/3 at 2:1)")
    fig.colorbar(sc, ax=ax, label="clear day", pad=0.02)
    ax.set_xlabel("win rate (2:1 bracket, ~2 trades/week)")
    ax.set_ylabel("gate clear day")
    ax.set_title("Time to clear the 1.25\u03c3 gate vs skill "
                 "(10 seeds; red x = never cleared in 1y)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "paper_wr_clear.png"), dpi=140)

    bd = (df[df["study"] == "bond"].groupby("burn_mult")["mom_ev_pd"]
          .agg(["mean", "min", "max"]))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = bd.index.to_numpy()
    ax.fill_between(x, bd["min"], bd["max"], color=fs.AMBER, alpha=0.16,
                    label="seed min/max")
    ax.fill_between(x, 0, bd["mean"].clip(lower=0), color=fs.RED, alpha=0.35)
    ax.fill_between(x, bd["mean"].clip(upper=0), 0, color=fs.GREEN,
                    alpha=0.30)
    ax.plot(x, bd["mean"], color=fs.AMBER, lw=2.2, marker="o", zorder=3,
            label="mean EV per entrant")
    ax.axhline(0, color=fs.FG, lw=0.8)
    ax.axvline(0.0, color=fs.CYAN, lw=1.4, ls=":")
    ax.text(0.06, float(bd["max"].max()), "ship: no bond", color=fs.CYAN,
            fontsize=9.5, va="top")
    ax.annotate("attack profitable", xy=(0.10, 0.92), xycoords="axes fraction",
                color=fs.RED, fontsize=9.5)
    ax.annotate("attack unprofitable", xy=(0.55, 0.35),
                xycoords="axes fraction", color=fs.GREEN, fontsize=9.5)
    ax.set_xlabel("counterfactual entry burn (multiples of the daily pool)")
    ax.set_ylabel("leveraged beta adversary net EV per entrant (pool-days)")
    ax.set_title("Adversary emission economics vs entry cost "
                 "(no stake settlement; rational entry, 5 seeds)")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "paper_bond_ev.png"), dpi=140)

    print(f"\nwrote {os.path.join(OUT, 'papergrid.csv')} + 2 figures")


PRODSIM_RATIONAL_KW = dict(entries_per_day=1, skilled_join_day=0.0,
                           kelly_leverage=1.0, menu=["tmom72", "tmom168"],
                           belief_sigma=0.0, marginal_obs=True)
PRODSIM_CELLS = [("headline-70", 0.70, s) for s in range(5)] \
    + [("headline-50", 0.50, s) for s in range(5)]

REGIME_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}

_SUB_LOG: List[Prediction] = []
_CAPTURE_ON = False


def _install_capture():
    global _CAPTURE_ON
    if _CAPTURE_ON:
        return
    _CAPTURE_ON = True
    for cls in (BetaMomentumMiner, OpportunistMiner, OracleSkillMiner):
        orig = cls.maybe_submit

        def wrapped(self, t, market, _orig=orig):
            preds = _orig(self, t, market)
            _SUB_LOG.extend(preds)
            return preds

        cls.maybe_submit = wrapped


def hourly_ohlc(market):
    T = market.n_minutes // 60
    idx = np.arange(T) * 60
    close = market.close[idx].astype(np.float64)
    hi = close.copy()
    lo = close.copy()
    n_bars = T - 1
    hi[1:] = market.high[1:n_bars * 60 + 1].reshape(n_bars, 60).max(axis=1)
    lo[1:] = market.low[1:n_bars * 60 + 1].reshape(n_bars, 60).min(axis=1)
    return np.column_stack([close, hi, lo])


def build_panel(preds: List[Prediction], market, labels: Dict[int, str],
                skilled_id: int):
    from flow import DIM, FIELDS_PER_REGIME
    T = market.n_minutes // 60
    price = market.close[np.arange(T) * 60].astype(np.float64)

    by_miner: Dict[int, List[Prediction]] = {}
    for p in preds:
        by_miner.setdefault(p.miner_id, []).append(p)

    hks = [f"{labels[mid]}#{mid}" for mid in sorted(by_miner)]
    hk2idx = {hk: i for i, hk in enumerate(hks)}
    mid_of = {f"{labels[mid]}#{mid}": mid for mid in sorted(by_miner)}
    skilled_hk = f"{labels[skilled_id]}#{skilled_id}"

    X = np.zeros((T, len(hks), DIM), dtype=np.float32)
    for hk, i in hk2idx.items():
        plist = sorted(by_miner[mid_of[hk]], key=lambda p: p.t_submit)
        by_reg: Dict[str, List[Prediction]] = {}
        for p in plist:
            by_reg.setdefault(p.regime, []).append(p)
        for reg, ps in by_reg.items():
            lo = REGIME_IDX[reg] * FIELDS_PER_REGIME
            rows = [min(p.t_submit // 60, T - 1) for p in ps]
            for k, p in enumerate(ps):
                r0 = rows[k]
                r1 = rows[k + 1] if k + 1 < len(ps) else T
                if r1 <= r0:
                    continue
                E = float(p.entry)
                X[r0:r1, i, lo:lo + 6] = [
                    p.direction, p.f, abs(E - p.sl) / E,
                    abs(p.tp1 - E) / E, abs(p.tp2 - E) / E, float(p.h_hours)]
                X[r0:r1, i, lo + 6] = k + 1
    return X, hk2idx, price, skilled_hk


def _cell_prodsim(args):
    import flow
    from flow import BURN_KEY, FlowConfig
    _install_capture()
    name, wr, seed = args
    market = load_real_market()
    mech = replace(MechanismParams(), **SHIP_MECH)

    _SUB_LOG.clear()
    res = run_rational(market, mech, burn_units=0.0, seed=seed,
                       skilled_overrides=dict(win_rate=wr),
                       **PRODSIM_RATIONAL_KW, **GATE_KW)
    preds = list(_SUB_LOG)

    X, hk2idx, price, skilled_hk = build_panel(
        preds, market, res.labels, res.skilled_id)
    T = len(price)
    ohlc = hourly_ohlc(market)
    kw = dict(blocks_ahead=0, sample_every=5,
              sidx_arr=np.arange(T, dtype=np.int64) * 60,
              cfg=FlowConfig(), return_diagnostics=True)
    hist = (X.reshape(T, -1), hk2idx)
    out, diag = flow.compute_flow_salience(hist, ohlc, **kw)
    out_c, diag_c = flow.compute_flow_salience(
        hist, price.reshape(-1, 1), **kw)

    def stats(out_, diag_):
        n_hours = diag_["_pool"]["n_hours"]
        q3 = int(n_hours * 0.75)
        zeros = np.zeros(n_hours)
        noise_hks = [hk for hk in hk2idx if hk != skilled_hk and hk in diag_]
        noise_paid = np.sum([diag_[hk]["paid"] for hk in noise_hks], axis=0) \
            if noise_hks else zeros
        sk = diag_.get(skilled_hk, {})
        sk_paid = sk.get("paid", zeros)
        return dict(
            clear_day=(sk.get("clear_hour") / 24.0
                       if sk.get("clear_hour") is not None else None),
            stat=float(sk.get("stat", float("nan"))),
            steady=float(sk_paid[q3:].mean()),
            noise_pd=float(noise_paid.sum()) / UNITS_PER_DAY,
            passes=sum(1 for hk in noise_hks
                       if diag_[hk].get("clear_hour") is not None),
            latched_end=sum(1 for hk in noise_hks if diag_[hk]["latched"]),
            burn=float(out_.get(BURN_KEY, 0.0)),
            sk_paid=sk_paid, noise_paid=noise_paid,
            n_trades=sk.get("n_trades", 0),
        )

    w = stats(out, diag)
    c = stats(out_c, diag_c)

    days = market.n_minutes / MINUTES_PER_DAY
    row = dict(
        cell=name, wr=wr, seed=seed,
        n_actors=len(hk2idx), n_intents=len(preds),
        sim_clear_day=res.skilled_clear_day,
        sim_steady=float(res.skilled_share[int(len(res.skilled_share) * 0.75):].mean()),
        sim_noise_pd=float(res.noise_share.sum()) / UNITS_PER_DAY,
        sim_passes=res.noise_ever_cleared,
        prod_clear_day=w["clear_day"], prod_stat=w["stat"],
        prod_steady=w["steady"], prod_noise_pd=w["noise_pd"],
        prod_passes=w["passes"], prod_latched_end=w["latched_end"],
        prod_burn=w["burn"], prod_skilled_trades=w["n_trades"],
        close_clear_day=c["clear_day"], close_steady=c["steady"],
        close_noise_pd=c["noise_pd"], close_passes=c["passes"],
        close_latched_end=c["latched_end"],
        days=days,
    )
    tl = dict(sk_prod=w["sk_paid"], sk_sim=res.skilled_share,
              noise_prod=w["noise_paid"])
    return name, seed, row, tl


def prodsim_main():
    _install_capture()
    t0 = time.monotonic()
    rows, tls = [], {}
    with ProcessPoolExecutor(max_workers=min(len(PRODSIM_CELLS),
                                             os.cpu_count() or 4)) as ex:
        for i, (name, seed, row, tl) in enumerate(ex.map(_cell_prodsim, PRODSIM_CELLS), 1):
            rows.append(row)
            tls[(name, seed)] = tl
            print(f"  [{i}/{len(PRODSIM_CELLS)}] {time.monotonic()-t0:.0f}s",
                  flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "prodsim.csv"), index=False)

    def agg(g):
        pc = g.loc[g["prod_clear_day"].notna(), "prod_clear_day"]
        sc = g.loc[g["sim_clear_day"].notna(), "sim_clear_day"]
        cc = g.loc[g["close_clear_day"].notna(), "close_clear_day"]
        return pd.Series(dict(
            sim_clear=sc.mean(), prod_clear=pc.mean(),
            prod_worst=pc.max(), close_clear=cc.mean(),
            sim_steady=g["sim_steady"].mean(),
            prod_steady=g["prod_steady"].mean(),
            close_steady=g["close_steady"].mean(),
            sim_leak=g["sim_noise_pd"].mean(),
            prod_leak=g["prod_noise_pd"].mean(),
            close_leak=g["close_noise_pd"].mean(),
            sim_pass=int(g["sim_passes"].sum()),
            prod_pass=int(g["prod_passes"].sum()),
            close_pass=int(g["close_passes"].sum()),
            prod_latch=int(g["prod_latched_end"].sum()),
            close_latch=int(g["close_latched_end"].sum()),
        ))

    table = df.groupby("cell").apply(agg, include_groups=False)
    table = table.reindex(["headline-70", "headline-50"]).round(2)
    print("\nsim = flow_mechanism.py (published studies); prod = flow.py "
          "with hourly wicks (spec 1.3); close = flow.py close-only "
          "(no kline feed)\n")
    print(table.to_string())

    fs.use()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    k = int(UNITS_PER_DAY)
    for ax, cell, title in zip(
            axes, ("headline-70", "headline-50"),
            ("70% win rate @ 2:1", "50% win rate @ 2:1")):
        seeds = [s for (n, s) in tls if n == cell]
        for key, color, lab, fill in (
                ("sk_sim", fs.CYAN, "skilled — simulation mechanism", 0.0),
                ("sk_prod", fs.GREEN, "skilled — flow.py as shipped", 0.35),
                ("noise_prod", fs.RED,
                 "leveraged beta adversaries — flow.py", 0.35)):
            n_min = min(len(tls[(cell, s)][key]) for s in seeds)
            m = np.mean([tls[(cell, s)][key][:n_min] for s in seeds], axis=0)
            sm = np.convolve(m, np.ones(k) / k, mode="same")
            day = np.arange(n_min) / 24.0
            ax.plot(day, sm, color=color, lw=1.9, label=lab, zorder=3,
                    ls="--" if not fill else "-")
            if fill:
                fs.gradient_fill(ax, day, sm, color, alpha=fill)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("day")
        ax.margins(x=0.01)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("share of daily emissions")
    axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle("Rational actors vs the IM as written — sim mechanism vs "
                 "production flow.py (mean of 5 seeds)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "prodsim_ramp.png")
    fig.savefig(path, dpi=140)
    print(f"\nwrote {os.path.join(OUT, 'prodsim.csv')}\nwrote {path}")


TAO_PER_DAY_POOL = 2.82 / 1.5
UNITS_PER_TAO = UNITS_PER_DAY / TAO_PER_DAY_POOL

MIN_STAKE_TAO = 1.0
STAKE_GRID_TAO = (1.0, 5.0, 15.0, 33.0)
SKILLED_STAKE_TAO = 33.0
BREAK_HAZARD_WK = 4.0 / 52.0
SETTLE_H = 168.0


class BrokenMomentumMiner(BetaMomentumMiner):
    def __init__(self, *a, break_lo_min: int = -1, break_hi_min: int = -1,
                 **kw):
        super().__init__(*a, **kw)
        self.break_lo_min = break_lo_min
        self.break_hi_min = break_hi_min

    def maybe_submit(self, t: int, market: Market) -> List[Prediction]:
        preds = super().maybe_submit(t, market)
        if not preds or not (self.break_lo_min <= t < self.break_hi_min):
            return preds
        regime = next(r for r in self.regimes
                      if r.name == REGIME_BY_H[self.h_hours])
        return [self._bracket(p.t_submit, market, regime, -p.direction,
                              f=p.f, h_hours=self.h_hours,
                              sl_mult=self.sl_mult, tp1_mult=self.tp1_mult,
                              tp2_mult=self.tp2_mult) for p in preds]


def money_r(pred: Prediction, event: str, market: Market) -> float:
    if event == "SL":
        return -1.0
    if event == "TP2":
        return float(pred.r_multiple(pred.tp2))
    t1 = min(pred.t_horizon, market.n_minutes - 1)
    return max(float(pred.r_multiple(float(market.close[t1]))), -1.0)


@dataclass
class StakePos:
    tao: float = 0.0
    withdraw_tao: float = 0.0
    withdraw_week: int = -1
    settle_pnl_tao: float = 0.0
    recent: List[float] = field(default_factory=list)


def run_skin(seed: int, skin_mode: str, burn_mult: float,
             adversary: str = "aware", break_days: tuple | None = None,
             n_prospects: int = 240, entries_per_day: int = 1,
             op_cost_units_day: float = 0.5,
             skilled_f: float = 0.03, skilled_refill: bool = True,
             skilled_sweep: bool = True,
             gate_z: float | None = 1.25, gate_exit_z: float = 0.5,
             skilled_join_day: float = 0.0, skilled_w: float = 0.70,
             record_weekly: bool = False) -> dict:
    market = load_real_market()
    mech = replace(MechanismParams(), gamma=0.25, delta=0.30, lam=0.50,
                   tau=1.00, ewma_alpha=0.05, kappa_hours=672.0,
                   probation_hours=168.0,
                   floor_clamp=float("inf"), staking_enabled=False)
    burn_units = burn_mult * UNITS_PER_DAY
    rng = np.random.default_rng(seed)
    regimes = list(DEFAULT_REGIMES)
    roster = ["tmom72", "tmom168"]

    brk = (None if break_days is None else
           (int(break_days[0] * MINUTES_PER_DAY),
            int(break_days[1] * MINUTES_PER_DAY)))

    agents: List[Agent] = []
    for i in range(n_prospects):
        kind = roster[i % len(roster)]
        sub_rng = np.random.default_rng(rng.integers(2**63))
        if brk is not None:
            h = 72 if kind == "tmom72" else 168
            miner = BrokenMomentumMiner(i, kind, sub_rng, regimes, h_hours=h,
                                        break_lo_min=brk[0],
                                        break_hi_min=brk[1], **TUNED_MOM)
        else:
            miner = make_strategy(kind, i, sub_rng, regimes)
        agents.append(Agent(
            miner=miner,
            patience_days=float(np.clip(rng.lognormal(math.log(60), 0.5),
                                        20, 240)),
            belief_mult=1.0))

    skilled = OracleSkillMiner(
        miner_id=n_prospects, label="skilled",
        rng=np.random.default_rng(seed * 65537 + 3), regimes=[regimes[2]],
        win_rate=skilled_w, rr=1.0, tp2_rr=2.0, h_hours=168,
        cadence_hours=12, cooldown_hours=84, skip_no_setup=True,
        f=skilled_f, sl_sigma=0.35)
    skilled.t_start_min = int(skilled_join_day * MINUTES_PER_DAY)
    sk_topups_tao = 0.0
    sk_swept_tao = 0.0
    adv_topups: Dict[int, float] = {}
    sk_ruin_weeks = 0

    states: Dict[int, MinerState] = {}
    stakes: Dict[int, StakePos] = {}
    epoch_minutes = int(mech.epoch_hours * 60)
    max_h_min = max(r.max_h for r in regimes) * 60
    epochs = list(range(epoch_minutes, market.n_minutes - max_h_min - 1,
                        epoch_minutes))
    n_ep = len(epochs)
    em_tl: Dict[int, np.ndarray] = {}
    skilled_share = np.zeros(n_ep)
    noise_share = np.zeros(n_ep)
    active_noise = np.zeros(n_ep)

    hist: Dict[int, list] = {}
    stat_cache: Dict[int, tuple] = {}
    latched: set = set()
    noise_cleared_ever: set = set()
    skilled_clear_day: Optional[float] = None
    sk_pre_clear_units = 0.0

    GATE_Z_IN, GATE_Z_OUT = gate_z, gate_exit_z
    BLOCK_H, MIN_N, DUST = 168.0, 4, 0.02

    def gate_stat(mid: int) -> float:
        h = hist.get(mid)
        if not h:
            return float("-inf")
        cached = stat_cache.get(mid)
        if cached is not None and cached[0] == len(h):
            return cached[1]
        arr = np.asarray(h)
        blk = (arr[:, 1] // (BLOCK_H * 60)).astype(np.int64)
        _u, inv = np.unique(blk, return_inverse=True)
        agg = np.zeros(len(_u))
        np.add.at(agg, inv, arr[:, 0])
        v = t_stat_returns(agg, MIN_N)
        stat_cache[mid] = (len(h), v)
        return v

    def trailing_rate(mid: int, ei: int, win_ep: int) -> float:
        tl = em_tl.get(mid)
        if tl is None:
            return 0.0
        lo = max(0, ei - win_ep)
        if ei <= lo:
            return 0.0
        return float(tl[lo:ei].sum()) / ((ei - lo) / UNITS_PER_DAY)

    def choose_stake(obs_rate: float, patience_d: float,
                     mean_noise_stake: float) -> tuple[float, float]:
        best_s, best_ev = MIN_STAKE_TAO, -1e18
        for s in STAKE_GRID_TAO:
            amp = 1.0
            if skin_mode == "always":
                amp = min(max(s / max(mean_noise_stake, MIN_STAKE_TAO),
                              0.25), 8.0)
            drift_units_wk = -BREAK_HAZARD_WK * s * UNITS_PER_TAO
            ev = ((obs_rate * amp - op_cost_units_day) * patience_d
                  - burn_units + drift_units_wk * (patience_d / 7.0))
            if ev > best_ev:
                best_s, best_ev = s, ev
        return best_s, best_ev

    pending: List[Prediction] = []
    settle_pnl_window: Dict[int, float] = {}
    carry_losses: Dict[int, float] = {}
    window_start_stake: Dict[int, float] = {}
    week_idx = 0
    n_entries = 0
    entry_sizes: List[float] = []
    burns_paid: Dict[int, float] = {}
    win_ep = int(14.0 * UNITS_PER_DAY)
    weekly: List[dict] = []

    states[skilled.miner_id] = MinerState(miner_id=skilled.miner_id)
    stakes[skilled.miner_id] = StakePos(tao=SKILLED_STAKE_TAO)
    em_tl[skilled.miner_id] = np.zeros(n_ep)
    burns_paid[skilled.miner_id] = burn_units
    window_start_stake[skilled.miner_id] = SKILLED_STAKE_TAO

    for ei, t in enumerate(epochs):
        if ei % int(UNITS_PER_DAY) == 0:
            actives = [a for a in agents if a.active]
            if actives:
                rates = [trailing_rate(a.miner.miner_id, ei, win_ep)
                         for a in actives]
                obs_rate = float(np.sum(rates)) / (len(actives) + 1)
                mean_noise_stake = float(np.mean(
                    [stakes[a.miner.miner_id].tao for a in actives]))
            else:
                n_act = 1 + (1 if t >= skilled.t_start_min else 0)
                obs_rate = UNITS_PER_DAY / (n_act + 1)
                mean_noise_stake = MIN_STAKE_TAO
            for a in actives:
                tenure_d = (t - a.entered_min) / MINUTES_PER_DAY
                mid = a.miner.miner_id
                if tenure_d >= 14.0 and \
                        trailing_rate(mid, ei, win_ep) < op_cost_units_day:
                    a.quit = True
                    a.quit_min = t
                    sp = stakes[mid]
                    if sp.withdraw_week < 0 and sp.tao > 0:
                        sp.withdraw_tao = sp.tao
                        sp.withdraw_week = week_idx + 1
            slots = entries_per_day
            order = rng.permutation(len(agents))
            for j in order:
                if slots <= 0:
                    break
                a = agents[j]
                if a.entered_min is not None or a.quit:
                    continue
                if adversary == "naive_whale":
                    s_tao = SKILLED_STAKE_TAO
                    ev = (obs_rate - op_cost_units_day) * a.patience_days \
                        - burn_units
                else:
                    s_tao, ev = choose_stake(obs_rate, a.patience_days,
                                             mean_noise_stake)
                if ev > 0:
                    mid = a.miner.miner_id
                    a.entered_min = t
                    a.miner.t_start_min = t
                    a.burns_paid += burn_units
                    burns_paid[mid] = burns_paid.get(mid, 0.0) + burn_units
                    states[mid] = MinerState(miner_id=mid)
                    stakes[mid] = StakePos(tao=s_tao)
                    window_start_stake[mid] = 0.0
                    em_tl[mid] = np.zeros(n_ep)
                    n_entries += 1
                    entry_sizes.append(s_tao)
                    slots -= 1

        due = [p for p in pending if p.t_horizon <= t]
        pending = [p for p in pending if p.t_horizon > t]
        for pred in due:
            res = resolve(pred, market, mech)
            if res is None:
                continue
            st = states[pred.miner_id]
            st.update_score(pred.regime, res.s_weighted, res.t_resolve,
                            mech.ewma_alpha)
            hist.setdefault(pred.miner_id, []).append(
                (res.s_base, float(res.t_resolve)))
            mr = money_r(pred, res.event, market)
            s0 = window_start_stake.get(pred.miner_id, 0.0)
            settle_pnl_window[pred.miner_id] = \
                settle_pnl_window.get(pred.miner_id, 0.0) + mr * pred.f * s0

        if (ei + 1) % int(SETTLE_H / mech.epoch_hours) == 0:
            losses = dict(carry_losses)
            claims: Dict[int, float] = {}
            for mid, pnl in settle_pnl_window.items():
                if pnl < 0:
                    losses[mid] = losses.get(mid, 0.0) + (-pnl)
                elif pnl > 0:
                    claims[mid] = claims.get(mid, 0.0) + pnl
            for mid in list(losses):
                cap = 0.25 * (stakes[mid].tao if mid in stakes else 0.0)
                losses[mid] = min(losses[mid], cap)
                if losses[mid] <= 0:
                    del losses[mid]
            pool = sum(losses.values())
            applied: Dict[int, float] = {}
            if losses and not claims:
                carry_losses = losses
            elif pool > 0:
                carry_losses = {}
                ctot = sum(claims.values())
                for mid, lv in losses.items():
                    stakes[mid].tao -= lv
                    stakes[mid].settle_pnl_tao -= lv
                    stakes[mid].recent.append(-lv)
                    applied[mid] = applied.get(mid, 0.0) - lv
                for mid, cv in claims.items():
                    gain = pool * cv / ctot
                    stakes[mid].tao += gain
                    stakes[mid].settle_pnl_tao += gain
                    stakes[mid].recent.append(gain)
                    applied[mid] = applied.get(mid, 0.0) + gain
            for mid, cv in claims.items():
                if pool <= 0:
                    stakes[mid].recent.append(0.0)
            week_flows = applied if record_weekly else None
            week_idx += 1
            settle_pnl_window = {}
            for mid, sp in stakes.items():
                if 0 <= sp.withdraw_week <= week_idx and sp.withdraw_tao > 0:
                    take = min(sp.withdraw_tao, sp.tao)
                    sp.tao -= take
                    sp.withdraw_tao = 0.0
                    sp.withdraw_week = -1
                    if mid == skilled.miner_id:
                        sk_swept_tao += take
            if adversary == "aware":
                for a in agents:
                    if not a.active:
                        continue
                    sp = stakes[a.miner.miner_id]
                    if sp.tao > MIN_STAKE_TAO and len(sp.recent) >= 2 and \
                            sum(sp.recent[-2:]) < -0.25 * sp.tao and \
                            sp.withdraw_week < 0:
                        sp.withdraw_tao = sp.tao - MIN_STAKE_TAO
                        sp.withdraw_week = week_idx + 1
            else:
                for a in agents:
                    if a.active:
                        sp = stakes[a.miner.miner_id]
                        if sp.tao < SKILLED_STAKE_TAO:
                            mid = a.miner.miner_id
                            adv_topups[mid] = adv_topups.get(mid, 0.0) + \
                                (SKILLED_STAKE_TAO - sp.tao)
                            sp.tao = SKILLED_STAKE_TAO
                        elif sp.tao > SKILLED_STAKE_TAO and \
                                sp.withdraw_week < 0:
                            sp.withdraw_tao = sp.tao - SKILLED_STAKE_TAO
                            sp.withdraw_week = week_idx + 1
            skp = stakes[skilled.miner_id]
            if skp.tao <= 0:
                sk_ruin_weeks += 1
            if skilled_refill and skp.tao < SKILLED_STAKE_TAO:
                sk_topups_tao += SKILLED_STAKE_TAO - skp.tao
                skp.tao = SKILLED_STAKE_TAO
            if skilled_sweep and skp.withdraw_week < 0 and \
                    skp.tao > SKILLED_STAKE_TAO:
                skp.withdraw_tao = skp.tao - SKILLED_STAKE_TAO
                skp.withdraw_week = week_idx + 1
            if record_weekly:
                weekly.append(dict(
                    week=week_idx - 1,
                    sk=week_flows.get(skilled.miner_id, 0.0),
                    adv=sum(v for m, v in week_flows.items()
                            if m != skilled.miner_id),
                    adv_staked=float(sum(
                        sp.tao for m, sp in stakes.items()
                        if m != skilled.miner_id)),
                ))
            window_start_stake = {mid: sp.tao for mid, sp in stakes.items()}

        submitters = [a.miner for a in agents if a.active]
        if t >= skilled.t_start_min:
            submitters.append(skilled)
        for miner in submitters:
            for pred in miner.maybe_submit(t, market):
                if validate(pred, mech):
                    pending.append(pred)

        raw = {mid: st.emission_score(t, mech) for mid, st in states.items()}
        pos = {mid: s for mid, s in raw.items() if s > 0}
        weighted: Dict[int, float] = {}
        for mid, s in pos.items():
            w = s
            if skin_mode in ("always", "cleared_only"):
                w = s * max(stakes[mid].tao, 0.0)
            weighted[mid] = w
        cleared = set()
        if GATE_Z_IN is None:
            cleared = set(pos)
        else:
            for mid in pos:
                st_v = gate_stat(mid)
                ok = st_v >= GATE_Z_IN
                if mid in latched and st_v >= GATE_Z_OUT:
                    ok = True
                if ok:
                    latched.add(mid)
                    cleared.add(mid)
                else:
                    latched.discard(mid)
        for mid in cleared:
            if mid == skilled.miner_id:
                if skilled_clear_day is None:
                    skilled_clear_day = t / MINUTES_PER_DAY
            else:
                noise_cleared_ever.add(mid)
        paid: Dict[int, float] = {}
        main_w = {mid: (weighted[mid] if skin_mode != "off" else pos[mid])
                  for mid in cleared if mid in pos}
        ctot = sum(main_w.values())
        if ctot > 0:
            for mid, w in main_w.items():
                paid[mid] = (1.0 - DUST) * w / ctot
        dust_w = {}
        for mid, s in pos.items():
            if mid in cleared:
                continue
            dust_w[mid] = (weighted[mid] if skin_mode == "always" else s)
        dtot = sum(dust_w.values())
        if dtot > 0:
            for mid, w in dust_w.items():
                paid[mid] = paid.get(mid, 0.0) + DUST * w / dtot
        for mid, w in paid.items():
            states[mid].emissions += w
            em_tl[mid][ei] = w
        skilled_share[ei] = paid.get(skilled.miner_id, 0.0)
        noise_share[ei] = sum(paid.values()) - skilled_share[ei]
        if skilled_clear_day is None:
            sk_pre_clear_units += paid.get(skilled.miner_id, 0.0)
        active_noise[ei] = sum(1 for a in agents if a.active)

    days = market.n_minutes / MINUTES_PER_DAY
    q3 = int(n_ep * 0.75)
    unit_tao = TAO_PER_DAY_POOL / UNITS_PER_DAY

    def refund_frac(mid: int) -> float:
        if burn_units <= 0:
            return 1.0
        if mid in noise_cleared_ever or (mid == skilled.miner_id
                                         and skilled_clear_day is not None):
            return 1.0
        claw = min(states[mid].emissions if mid in states else 0.0,
                   0.75 * burn_units)
        return 1.0 - claw / burn_units

    entered = [a for a in agents if a.entered_min is not None]
    adv_ev_tao, adv_settle = [], []
    for a in entered:
        mid = a.miner.miner_id
        sp = stakes.get(mid, StakePos())
        em_u = states[mid].emissions if mid in states else 0.0
        ref = a.burns_paid * refund_frac(mid)
        net_u = em_u - a.burns_paid + ref \
            - op_cost_units_day * ((a.quit_min or epochs[-1])
                                   - a.entered_min) / MINUTES_PER_DAY
        adv_ev_tao.append(net_u * unit_tao + sp.settle_pnl_tao)
        adv_settle.append(sp.settle_pnl_tao)

    sk = stakes[skilled.miner_id]
    sk_em_u = states[skilled.miner_id].emissions
    total_u = n_ep
    return dict(
        seed=seed, skin=skin_mode, bond=burn_mult, adversary=adversary,
        breakw=(break_days is not None), sk_w=skilled_w,
        empty=(skilled_join_day > 1000),
        gate_z=(float("nan") if gate_z is None else gate_z),
        sk_policy=(f"f{skilled_f:g}" + ("+refill" if skilled_refill else "")
                   + ("+sweep" if skilled_sweep else "")),
        sk_topups_tao=sk_topups_tao, sk_swept_tao=sk_swept_tao,
        sk_ruin_weeks=sk_ruin_weeks,
        clear_day=skilled_clear_day,
        steady=float(skilled_share[q3:].mean()),
        sk_em_pd=sk_em_u / UNITS_PER_DAY,
        sk_pre_clear_pd=sk_pre_clear_units / UNITS_PER_DAY,
        sk_settle_tao=sk.settle_pnl_tao,
        sk_net_tao=(sk_em_u - burns_paid[skilled.miner_id]
                    * (1 - refund_frac(skilled.miner_id))) * unit_tao
        + sk.settle_pnl_tao,
        leak_pct=100.0 * sum(states[a.miner.miner_id].emissions
                             for a in entered) / total_u,
        adv_ev_tao=(float(np.mean(adv_ev_tao)) if adv_ev_tao else 0.0),
        adv_settle_tao=(float(np.mean(adv_settle)) if adv_settle else 0.0),
        adv_stake_now=(float(np.mean([stakes[a.miner.miner_id].tao
                                      + stakes[a.miner.miner_id].withdraw_tao
                                      for a in entered])) if entered else 0.0),
        passes=len(noise_cleared_ever),
        entries=n_entries,
        adv_entry_mean=(float(np.mean(entry_sizes)) if entry_sizes else 0.0),
        adv_entry_min_frac=(float(np.mean([s <= MIN_STAKE_TAO
                                           for s in entry_sizes]))
                            if entry_sizes else 0.0),
        crowd=float(active_noise[q3:].mean()),
        **({"weekly": weekly} if record_weekly else {}),
    )


STAKEECON_CONFIGS = [
    dict(skin="off", burn_mult=0.0, adversary="aware"),
    dict(skin="always", burn_mult=0.0, adversary="aware"),
    dict(skin="cleared_only", burn_mult=0.0, adversary="aware"),
    dict(skin="always", burn_mult=1.5, adversary="aware"),
    dict(skin="always", burn_mult=0.0, adversary="naive_whale"),
    dict(skin="always", burn_mult=0.0, adversary="naive_whale",
         break_days=(540.0, 568.0)),
    dict(skin="always", burn_mult=0.0, adversary="aware",
         break_days=(540.0, 568.0)),
    dict(skin="always", burn_mult=0.0, adversary="aware",
         skilled_refill=False),
    dict(skin="always", burn_mult=0.0, adversary="aware",
         skilled_f=0.015),
    dict(skin="always", burn_mult=0.0, adversary="aware",
         skilled_sweep=False),
    dict(skin="always", burn_mult=0.0, adversary="aware",
         skilled_join_day=1e5),
    dict(skin="always", burn_mult=0.0, adversary="naive_whale",
         skilled_join_day=1e5),
    dict(skin="always", burn_mult=0.0, adversary="naive_whale",
         skilled_w=0.50, skilled_f=0.03),
]

for _w in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
    for _bm, _gz in ((0.0, 1.25), (0.0, 1.5), (1.5, 1.25)):
        STAKEECON_CONFIGS.append(dict(skin="always", burn_mult=_bm,
                                      adversary="aware", gate_z=_gz,
                                      skilled_w=_w, skilled_f=0.03))


def _cell_stakeecon(args):
    cfg, seed = args
    c = dict(cfg)
    return run_skin(seed, c.pop("skin"), c.pop("burn_mult"), **c)


FLOW_SEEDS = (0, 1, 2, 3, 4)
FLOW_W = 0.50
FLOW_F = 0.03


def _flow_cell(args):
    adversary, seed = args
    return run_skin(seed, "always", 0.0, adversary=adversary,
                    skilled_w=FLOW_W, skilled_f=FLOW_F,
                    record_weekly=True)


def _flow_series(rows: list[dict]):
    n = min(len(r["weekly"]) for r in rows)
    sk = np.array([[r["weekly"][i]["sk"] for i in range(n)] for r in rows])
    adv = np.array([[r["weekly"][i]["adv"] for i in range(n)] for r in rows])
    staked = np.array([[r["weekly"][i]["adv_staked"] for i in range(n)]
                       for r in rows])
    return sk, adv, staked


def fig_flow():
    fs.use()
    cells = [(adv, s) for adv in ("naive_whale", "aware")
             for s in FLOW_SEEDS]
    out: Dict[str, list] = {"naive_whale": [], "aware": []}
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for (adv, _s), row in zip(cells, ex.map(_flow_cell, cells)):
            out[adv].append(row)

    sk_n, adv_n, staked_n = _flow_series(out["naive_whale"])
    sk_a, adv_a, staked_a = _flow_series(out["aware"])

    figu = plt.figure(figsize=(12.8, 8.2))
    gs = figu.add_gridspec(2, 2, height_ratios=[0.92, 1.0],
                           hspace=0.50, wspace=0.42,
                           left=0.065, right=0.935, top=0.89, bottom=0.08)
    axA = figu.add_subplot(gs[0, :])
    axB = figu.add_subplot(gs[1, 0])
    axC = figu.add_subplot(gs[1, 1])

    y = sk_n[0]
    wk = np.arange(1, len(y) + 1)
    colors = [fs.GREEN if v >= 0 else fs.RED for v in y]
    axA.bar(wk, y, color=colors, width=0.8, zorder=3)
    axA.axhline(0, color=fs.FG, lw=0.8)
    lose_wk = float((sk_n < 0).sum(axis=1).mean())
    axA.set_title(f"Weekly settlement flow to a moderate skilled whale "
                  f"(w = {FLOW_W:.2f}, f = {FLOW_F:.2f}) vs a naive "
                  f"33 TAO beta whale book (seed 0: one realization of "
                  f"the week-to-week variance)")
    axA.set_xlabel("settlement week")
    axA.set_ylabel("TAO to whale, week")
    axA.text(0.985, 0.94,
             f"red weeks are real: the whale pays out in "
             f"{lose_wk:.1f}/{len(y)} weeks (5-seed mean), and still "
             f"ends +{sk_n.sum(axis=1).mean():.0f} TAO",
             transform=axA.transAxes, ha="right", va="top", fontsize=9.5,
             color=fs.FG, bbox=dict(facecolor=fs.PANEL, edgecolor=fs.GRID))

    def cum_panel(ax, sk, adv, staked, title):
        wk = np.arange(1, sk.shape[1] + 1)
        cs, ca = sk.cumsum(axis=1), adv.cumsum(axis=1)
        ax.fill_between(wk, cs.min(axis=0), cs.max(axis=0),
                        color=fs.GREEN, alpha=0.16, lw=0)
        ax.plot(wk, cs.mean(axis=0), color=fs.GREEN, lw=2.2,
                label="skilled whale, cumulative")
        ax.fill_between(wk, ca.min(axis=0), ca.max(axis=0),
                        color=fs.RED, alpha=0.16, lw=0)
        ax.plot(wk, ca.mean(axis=0), color=fs.RED, lw=2.2,
                label="beta book, cumulative")
        ax.axhline(0, color=fs.FG, lw=0.8)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("settlement week")
        ax.set_ylabel("cumulative settlement TAO")
        ax2 = ax.twinx()
        ax2.plot(wk, staked.mean(axis=0), color=fs.AMBER, lw=1.3,
                 ls="--", alpha=0.85, label="beta collateral posted")
        ax2.set_ylabel("beta collateral posted (TAO)", color=fs.AMBER)
        ax2.tick_params(axis="y", labelcolor=fs.AMBER)
        ax2.set_ylim(0, float(staked.mean(axis=0).max()) * 2.6)
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)

    cum_panel(axB, sk_n, adv_n, staked_n,
              "Naive 33 TAO beta whale (believes its edge, refills)")
    cum_panel(axC, sk_a, adv_a, staked_a,
              "Break-aware beta book (sizes small, de-risks after losses)")

    figu.suptitle(f"Collateral settlement under the period circuit breaker "
                  f"(w = {FLOW_W:.2f}): every weekly flow is capped at a "
                  f"quarter of the losing book",
                  fontweight="bold")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "stakeecon_flow.png")
    figu.savefig(path, dpi=160)
    plt.close(figu)

    print(f"wrote {path}")
    print(f"  naive book : whale +{sk_n.sum(axis=1).mean():.1f} TAO over the tape, "
          f"book {adv_n.sum(axis=1).mean():+.1f}, whale loses "
          f"{(sk_n < 0).sum(axis=1).mean():.1f}/{sk_n.shape[1]} weeks, "
          f"worst week {sk_n.min():+.1f} TAO")
    print(f"  aware book : whale +{sk_a.sum(axis=1).mean():.1f} TAO over the tape, "
          f"book {adv_a.sum(axis=1).mean():+.1f}, whale loses "
          f"{(sk_a < 0).sum(axis=1).mean():.1f}/{sk_a.shape[1]} weeks, "
          f"worst week {sk_a.min():+.1f} TAO")


def stakeecon_main():
    seeds = (0, 1, 2, 3, 4)
    cells = [(cfg, s) for cfg in STAKEECON_CONFIGS for s in seeds]
    print(f"{len(cells)} runs")
    t0 = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for i, row in enumerate(ex.map(_cell_stakeecon, cells), 1):
            rows.append(row)
            print(f"  [{i}/{len(cells)}] {time.monotonic()-t0:.0f}s",
                  flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "stakeecon.csv"), index=False)
    pd.set_option("display.width", 240)

    core = df[df["sk_w"] == 0.70]
    g = core.groupby(["skin", "bond", "adversary", "breakw", "empty",
                      "sk_policy"]).agg(
        clear=("clear_day", "mean"), steady=("steady", "mean"),
        sk_em_pd=("sk_em_pd", "mean"),
        sk_pre_pd=("sk_pre_clear_pd", "mean"),
        sk_settle=("sk_settle_tao", "mean"),
        sk_topup=("sk_topups_tao", "mean"),
        sk_sweep=("sk_swept_tao", "mean"),
        ruin_wk=("sk_ruin_weeks", "mean"),
        sk_net=("sk_net_tao", "mean"),
        leak=("leak_pct", "mean"), adv_ev=("adv_ev_tao", "mean"),
        adv_settle=("adv_settle_tao", "mean"),
        stake_now=("adv_stake_now", "mean"), passes=("passes", "sum"),
        entries=("entries", "mean"), crowd=("crowd", "mean"))
    print("===== core grid (strong whale, w = 0.70) =====")
    print(g.round(2))

    band = df[(df["sk_w"] < 0.70) & (df["adversary"] == "aware")].copy()
    band["gate"] = band.apply(
        lambda r: f"z{r['gate_z']:g}/bond{r['bond']:g}", axis=1)
    gb = band.groupby(["gate", "sk_w"]).agg(
        clear=("clear_day", "mean"), clr_max=("clear_day", "max"),
        n_clr=("clear_day", "count"), steady=("steady", "mean"),
        sk_net=("sk_net_tao", "mean"), leak=("leak_pct", "mean"),
        adv_ev=("adv_ev_tao", "mean"), passes=("passes", "sum"))
    print("\n===== whale band (realized skill sweep, f = 0.03) =====")
    print(gb.round(2))

    fig_flow()


NEVER = 731.0
POOL_TOTAL_PD = 720.0


def threshold_heat(df, plt):
    th = df[df["study"] == "threshold"].copy()
    th["exit_z"] = th["exit_z"].fillna(-1.0)
    thrs = sorted(th["thr"].unique(), reverse=True)
    exits = sorted(th["exit_z"].unique())
    xlabels = ["no latch" if e < 0 else f"{e:g}" for e in exits]

    def cellagg(t, e):
        g = th[(th["thr"] == t) & (th["exit_z"] == e)]
        note = "*" if (g["cleared"] == 0).any() else ""
        return (g["clear_day"].fillna(NEVER).mean(),
                g["noise_paid_pd"].mean() / POOL_TOTAL_PD * 100.0,
                note)

    grid = [[cellagg(t, e) for e in exits] for t in thrs]
    clear = [[c[0] for c in row] for row in grid]
    leak = [[c[1] for c in row] for row in grid]
    notes = [[c[2] for c in row] for row in grid]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    im0 = fs.heat(axes[0], clear, xlabels, [f"{t:g}" for t in thrs],
                  cmap="magma_r", notes=notes)
    axes[0].set_title("skilled clear day, w = 0.50  (* = a seed never cleared)")
    im1 = fs.heat(axes[1], leak, xlabels, [f"{t:g}" for t in thrs],
                  cmap="magma", fmt="{:.1f}%")
    axes[1].set_title("leveraged beta adversary leak (% of tape emissions)")
    for ax, im in ((axes[0], im0), (axes[1], im1)):
        ax.set_xlabel("exit latch $z_{exit}$")
        ax.set_ylabel("clearance threshold $z_{clear}$")
        fs.mark_cell(ax, exits.index(0.5), thrs.index(1.25), label="ship")
        fig.colorbar(im, ax=ax, pad=0.02)
    fig.suptitle("Gate threshold × exit latch (5 seeds per cell)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "paper_threshold_heat.png"), dpi=140)


def gammatau(df, plt):
    gt = df[df["study"] == "gammatau"]
    gammas = sorted(gt["gamma"].unique())
    taus = sorted(gt["tau"].unique())

    def M(col, fill=None):
        return [[(gt[(gt["gamma"] == g) & (gt["tau"] == t)][col]
                  .fillna(fill) if fill is not None else
                  gt[(gt["gamma"] == g) & (gt["tau"] == t)][col]).mean()
                 for g in gammas] for t in taus]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    im0 = fs.heat(axes[0], M("clear_day", fill=NEVER),
                  [f"{g:g}" for g in gammas], [f"{t:g}" for t in taus],
                  cmap="magma_r", fmt="{:.1f}", vmin=0, vmax=120)
    axes[0].set_title("skilled clear day, w = 0.50")
    leak_pct = [[v / POOL_TOTAL_PD * 100.0 for v in row]
                for row in M("noise_paid_pd")]
    im1 = fs.heat(axes[1], leak_pct, [f"{g:g}" for g in gammas],
                  [f"{t:g}" for t in taus], cmap="magma", fmt="{:.1f}%",
                  vmin=0, vmax=17)
    axes[1].set_title("leveraged beta adversary leak (% of tape emissions)")
    for ax in axes:
        ax.set_xlabel(r"path penalty $\gamma$")
        ax.set_ylabel(r"tail amplifier $\tau$")
        fs.mark_cell(ax, gammas.index(0.25), taus.index(1.0), label="ship")
    fig.suptitle("Scoring parameters cannot move the gate (color scales "
                 "pinned wide; 3 seeds per cell)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "paper_gammatau.png"), dpi=140)


def sensitivity_strip(df, plt):
    panels = (("block", "block_h", 168, "block length $L$ (h)", "§6.5"),
              ("minn", "min_n", 4, "min blocks $n_{min}$", "§6.6"),
              ("pressure", "epd", 1, "entries/day cap", "§6.8"),
              ("dust", "dust", 0.02, "dust tier $d$", "§6.9"))
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.4))
    for ax, (study, key, ship, xlab, sec) in zip(axes, panels):
        d = df[(df.study == study) & (df.wr == 0.50)]
        g = d.groupby(key).agg(leak=("noise_paid_pd", "mean"),
                               clear=("clear_day", "mean")).reset_index()
        x = np.arange(len(g))
        leak = g["leak"] / (POOL_TOTAL_PD / 100.0)
        ax.plot(x, leak, "o-", color=fs.RED, lw=2.0, ms=6, zorder=5)
        si = int(np.where(g[key] == ship)[0][0])
        ax.plot(x[si], leak.iloc[si], "o", mfc="none", mec=fs.FG, ms=13,
                mew=1.6, zorder=6)
        ax2 = ax.twinx()
        ax2.plot(x, g["clear"], "s--", color=fs.GREY, lw=1.4, ms=4,
                 zorder=4)
        ax2.grid(False)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in g[key]])
        ax.set_xlabel(f"{xlab}  ({sec})")
        ax.set_ylim(0, 40)
        ax2.set_ylim(0, 80)
        if ax is axes[0]:
            ax.set_ylabel("leak, % of tape emissions", color=fs.RED)
        if ax is axes[-1]:
            ax2.set_ylabel("mean clear day", color=fs.GREY)
        else:
            ax2.set_yticklabels([])
    fig.suptitle("One knob at a time (w = 0.50, ship spec elsewhere): "
                 "leakage in red (left), proving time in grey (right); "
                 "ring = ship value", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "paper_sensitivity.png"), dpi=140)


def settlement_lifecycle(plt):
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.set_xlim(-3.1, 11.2)
    ax.set_ylim(-0.10, 4.55)
    ax.grid(False)

    Y_MINER, Y_DAEMON, Y_CONTRACT = 3.60, 2.35, 1.10

    lanes = (
        (Y_MINER, "MINER", "addStake · postTrade(bet) ·\nwithdraw(≤ free) · "
                           "sweepExpired"),
        (Y_DAEMON, "TEAM DAEMON", "owner key: closeBatch ·\nsettle · "
                                  "reconcile — resolution only"),
        (Y_CONTRACT, "CONTRACT", "enforces every bound;\nzero-sum custody"),
    )
    for y, lbl, cap in lanes:
        ax.axhline(y, color=fs.GRID, lw=1.0, zorder=1)
        ax.text(-3.0, y + 0.10, lbl, ha="left", va="bottom", fontsize=11,
                color=fs.FG, fontweight="bold")
        ax.text(-3.0, y + 0.02, cap, ha="left", va="top", fontsize=7.5,
                color=fs.GREY)

    ax.axvline(7, color=fs.GRID, lw=1.2, ls="--", zorder=1)
    ax.text(7, 4.38, "period $k$ ends (7d)", ha="center", fontsize=9,
            color=fs.GREY)

    def down(x, frm, to, color, lw=1.5):
        ax.annotate("", xy=(x, to + 0.05), xytext=(x, frm - 0.05),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    ax.plot(0.3, Y_MINER, "o", color=fs.AMBER, ms=8, zorder=5)
    ax.text(0.3, Y_MINER + 0.42, "miner posts the BET\n(postTrade, "
            "hotkey-signed)", ha="center", fontsize=8.5, color=fs.AMBER,
            fontweight="bold", va="bottom")
    down(0.3, Y_MINER, Y_CONTRACT, fs.AMBER, lw=1.9)
    ax.text(0.44, Y_CONTRACT + 0.52,
            "bet = f × collateral · immutable\nexpiry: regime ceiling + 48h "
            "(on-chain)", fontsize=8, color=fs.AMBER, va="center", ha="left")

    ax.plot(1.5, Y_MINER, "o", color=fs.CYAN, ms=8, zorder=5)
    ax.text(1.5, Y_MINER + 0.14, "trade opens\n(payload, encrypted)",
            ha="center", fontsize=8.5, color=fs.CYAN)
    down(1.5, Y_MINER, Y_DAEMON, fs.CYAN)
    ax.text(1.62, Y_DAEMON + 0.62, "owner key decrypts\nat arrival;\n"
            "late bets voided", fontsize=8, color=fs.GREY, va="center",
            ha="left")

    ax.broken_barh([(0.3, 4.3)], (Y_CONTRACT - 0.34, 0.20),
                   facecolors=fs.AMBER, alpha=0.45, zorder=3)
    ax.text(0.3, Y_CONTRACT - 0.50, "margin reserved by the miner's own "
            "bet:\nworst case locked — owner cannot open or resize",
            fontsize=8, color=fs.AMBER, ha="left", va="top")

    ax.plot(4.3, Y_MINER, "X", color=fs.FG, ms=9, zorder=5)
    ax.text(4.3, Y_MINER + 0.14, "trade resolves\non the tape", ha="center",
            fontsize=8.5, color=fs.FG)
    down(4.3, Y_MINER, Y_DAEMON, fs.GREY)
    down(4.6, Y_DAEMON, Y_CONTRACT, fs.RED)
    ax.text(4.48, Y_DAEMON - 0.16, "closeBatch", ha="right", fontsize=9.5,
            color=fs.RED, fontweight="bold")
    ax.text(4.74, Y_CONTRACT + 0.52,
            "loss ≤ the miner's bet\n(wins close at 0)", fontsize=8,
            color=fs.RED, va="center", ha="left")

    ax.broken_barh([(4.6, 3.7)], (Y_CONTRACT - 0.34, 0.20),
                   facecolors=fs.RED, alpha=0.40, zorder=3)
    ax.text(4.7, Y_CONTRACT - 0.50, "loss debited from the book instantly,\n"
            "parked in pool$_k$ until the settle", fontsize=8, color=fs.RED,
            ha="left", va="top")

    down(8.3, Y_DAEMON, Y_CONTRACT, fs.GREEN, lw=1.9)
    ax.text(8.18, Y_DAEMON - 0.16, "settle($k$)", ha="right", fontsize=9.5,
            color=fs.GREEN, fontweight="bold")
    ax.text(8.44, Y_CONTRACT + 0.52,
            "pool$_k$ → net winners, pro rata\nΣ payouts == pool (zero-sum)\n"
            "only after week $k$ ends on-chain", fontsize=8, color=fs.GREEN,
            va="center", ha="left")
    ax.plot(8.3, Y_CONTRACT, "*", color=fs.GREEN, ms=15, zorder=5)
    ax.text(7.1, Y_DAEMON + 0.30, "straggler lag:\nlast closes land first",
            fontsize=7.5, color=fs.GREY, ha="left", va="bottom")

    for x, note, ha in ((2.4, "withdraw free stake\n(balance − reserve)\n"
                              "instant, single step", "center"),
                        (9.9, "withdraw all\n(flat book:\nall free)",
                         "center")):
        ax.annotate("", xy=(x, Y_MINER + 0.34), xytext=(x, Y_MINER + 0.05),
                    arrowprops=dict(arrowstyle="->", color=fs.GREEN, lw=1.6))
        ax.plot(x, Y_MINER, "^", color=fs.GREEN, ms=9, zorder=5)
        ax.text(x, Y_MINER + 0.40, note, fontsize=8, color=fs.GREEN,
                va="bottom", ha=ha)

    ax.text(-3.0, 0.36, "owner silence cannot lock: every reserve expires "
            "at its regime ceiling + 48h, sweepExpired is permissionless, "
            "and withdrawals need no owner at all", fontsize=8.5,
            color=fs.GREY, ha="left", va="center")
    ax.text(-3.0, 0.10, "every owner arrow is recomputable by anyone once "
            "the payloads mature (1-wk timelock): inputs are the bets on "
            "chain, the decrypted payloads, and the public tape",
            fontsize=8.5, color=fs.GREY, ha="left", va="center")

    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_xticklabels([f"{d}d" for d in range(11)])
    ax.set_xlabel("tape time")
    ax.set_title("Miner-posted bets: the bet reserves margin, close debits "
                 "≤ the bet, settle pays the pool", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "settlement_lifecycle.png"), dpi=140)


def relfigs_main():
    import pandas as pd
    fs.use()
    df = pd.read_csv(os.path.join(OUT, "papergrid.csv"))
    threshold_heat(df, plt)
    gammatau(df, plt)
    sensitivity_strip(df, plt)
    settlement_lifecycle(plt)
    print(f"wrote 4 figures to {OUT}")


START_MS = 1722470400000
END_MS = 1785542400000
MINUTE_MS = 60_000
N_MIN = (END_MS - START_MS) // MINUTE_MS

MONTHS = [(y, m) for y in (2024, 2025, 2026) for m in range(1, 13)
          if (y, m) >= (2024, 8) and (y, m) <= (2026, 7)]


def _get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flow-tape/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_json(url: str, tries: int = 6):
    for i in range(tries):
        try:
            return json.loads(_get(url))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _save(venue: str, ts, hi, lo, cl) -> None:
    order = np.argsort(ts)
    ts, hi, lo, cl = (np.asarray(a)[order] for a in (ts, hi, lo, cl))
    keep = (ts >= START_MS) & (ts < END_MS)
    ts, hi, lo, cl = ts[keep], hi[keep], lo[keep], cl[keep]
    ts, iu = np.unique(ts, return_index=True)
    path = os.path.join(HERE, f"tape_{venue}.npz")
    np.savez_compressed(path, ts=ts.astype(np.int64),
                        high=hi[iu].astype(np.float64),
                        low=lo[iu].astype(np.float64),
                        close=cl[iu].astype(np.float64))
    cover = 100.0 * len(ts) / N_MIN
    print(f"{venue}: {len(ts)} minutes ({cover:.2f}% of grid) -> {path}")


def fetch_binance() -> None:
    ts, hi, lo, cl = [], [], [], []
    for y, m in MONTHS:
        url = (f"https://data.binance.vision/data/spot/monthly/klines/"
               f"BTCUSDT/1m/BTCUSDT-1m-{y}-{m:02d}.zip")
        raw = _get(url, timeout=120)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open(z.namelist()[0]) as f:
                for line in io.TextIOWrapper(f, "utf-8"):
                    parts = line.split(",")
                    if not parts[0].strip().isdigit():
                        continue
                    t = int(parts[0])
                    if t > 10**15:
                        t //= 1000
                    ts.append(t)
                    hi.append(float(parts[2]))
                    lo.append(float(parts[3]))
                    cl.append(float(parts[4]))
        print(f"binance {y}-{m:02d}: {len(ts)} rows", flush=True)
    _save("binance", ts, hi, lo, cl)


def fetch_bybit() -> None:
    ts, hi, lo, cl = [], [], [], []
    end = END_MS - 1
    n = 0
    while end >= START_MS:
        url = ("https://api.bybit.com/v5/market/kline?category=spot"
               f"&symbol=BTCUSDT&interval=1&limit=1000&end={end}")
        rows = _get_json(url)["result"]["list"]
        if not rows:
            break
        for r in rows:
            t = int(r[0])
            if t < START_MS:
                continue
            ts.append(t)
            hi.append(float(r[2]))
            lo.append(float(r[3]))
            cl.append(float(r[4]))
        end = int(rows[-1][0]) - MINUTE_MS
        n += 1
        if n % 50 == 0:
            print(f"bybit: {n} reqs, at "
                  f"{time.strftime('%Y-%m-%d', time.gmtime(end/1000))}",
                  flush=True)
        time.sleep(0.12)
    _save("bybit", ts, hi, lo, cl)


def fetch_okx(start_ms: int = START_MS, end_ms: int = END_MS,
              suffix: str = "") -> None:
    ts, hi, lo, cl = [], [], [], []
    after = end_ms
    n = 0
    while after > start_ms:
        url = ("https://www.okx.com/api/v5/market/history-candles?"
               f"instId=BTC-USDT&bar=1m&limit=100&after={after}")
        rows = _get_json(url)["data"]
        if not rows:
            break
        for r in rows:
            t = int(r[0])
            if t < start_ms:
                continue
            ts.append(t)
            hi.append(float(r[2]))
            lo.append(float(r[3]))
            cl.append(float(r[4]))
        after = int(rows[-1][0])
        n += 1
        if n % 200 == 0:
            print(f"okx{suffix}: {n} reqs, at "
                  f"{time.strftime('%Y-%m-%d', time.gmtime(after/1000))}",
                  flush=True)
        time.sleep(0.12)
    _save(f"okx{suffix}", ts, hi, lo, cl)


def merge_okx() -> None:
    ts, hi, lo, cl = [], [], [], []
    for i in range(8):
        path = os.path.join(HERE, f"tape_okx_{i}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        ts.append(z["ts"]); hi.append(z["high"])
        lo.append(z["low"]); cl.append(z["close"])
    _save("okx", np.concatenate(ts), np.concatenate(hi),
          np.concatenate(lo), np.concatenate(cl))


def fetch_coinbase() -> None:
    ts, hi, lo, cl = [], [], [], []
    step = 300 * MINUTE_MS
    n = 0
    start = START_MS
    while start < END_MS:
        end = min(start + step, END_MS)
        url = ("https://api.exchange.coinbase.com/products/BTC-USD/candles"
               f"?granularity=60&start={start // 1000}&end={end // 1000}")
        rows = _get_json(url)
        for r in rows:
            t = int(r[0]) * 1000
            if not (START_MS <= t < END_MS):
                continue
            ts.append(t)
            lo.append(float(r[1]))
            hi.append(float(r[2]))
            cl.append(float(r[4]))
        start = end
        n += 1
        if n % 200 == 0:
            print(f"coinbase: {n} reqs, at "
                  f"{time.strftime('%Y-%m-%d', time.gmtime(start/1000))}",
                  flush=True)
        time.sleep(0.18)
    _save("coinbase", ts, hi, lo, cl)


def build_cache() -> None:
    grid = np.arange(START_MS, END_MS, MINUTE_MS, dtype=np.int64)
    venues = []
    names = []
    for v in ("binance", "bybit", "okx", "coinbase"):
        path = os.path.join(HERE, f"tape_{v}.npz")
        if not os.path.exists(path):
            print(f"build: {v} missing, skipped")
            continue
        z = np.load(path)
        idx = ((z["ts"] - START_MS) // MINUTE_MS).astype(np.int64)
        hi = np.full(len(grid), np.nan)
        lo = np.full(len(grid), np.nan)
        cl = np.full(len(grid), np.nan)
        hi[idx], lo[idx], cl[idx] = z["high"], z["low"], z["close"]
        filled = cl.copy()
        mask = np.isnan(filled)
        i = np.where(~mask, np.arange(len(filled)), 0)
        np.maximum.accumulate(i, out=i)
        filled = filled[i]
        first = np.argmax(~mask)
        filled[:first] = filled[first]
        hi = np.where(np.isnan(hi), filled, hi)
        lo = np.where(np.isnan(lo), filled, lo)
        cl = np.where(np.isnan(cl), filled, cl)
        venues.append((hi, lo, cl))
        names.append(v)
        print(f"build: {v} aligned, {int(mask.sum())} gap minutes filled")
    if len(venues) < 2:
        raise SystemExit("need at least two venues for a consensus tape")

    close = venues[names.index("binance")][2] if "binance" in names \
        else venues[0][2]
    ch_high = np.min(np.stack([v[0] for v in venues]), axis=0)
    ch_low = np.max(np.stack([v[1] for v in venues]), axis=0)
    ch_high = np.maximum(ch_high, close)
    ch_low = np.minimum(ch_low, close)
    out = os.path.join(HERE, "cache_btc_multi_1m.npz")
    np.savez_compressed(out, ts=grid, close=close.astype(np.float32),
                        high=ch_high.astype(np.float32),
                        low=ch_low.astype(np.float32),
                        venues=np.array(names))
    yrs = len(grid) / (525_600)
    print(f"wrote {out}: {len(grid)} minutes ({yrs:.2f} years), "
          f"consensus of {len(names)} venues: {', '.join(names)}")


FETCH_USAGE = ("usage: python3 -m flow_sim.sims fetch "
               "{binance|bybit|okx|coinbase|okx-merge|build}")


def fetch_main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "okx" and len(argv) == 4:
        fetch_okx(int(argv[1]), int(argv[2]), argv[3])
        raise SystemExit(0)
    fn = {"binance": fetch_binance, "bybit": fetch_bybit, "okx": fetch_okx,
          "coinbase": fetch_coinbase, "okx-merge": merge_okx,
          "build": build_cache}.get(cmd)
    if fn is None:
        raise SystemExit(FETCH_USAGE)
    fn()


USAGE = ("usage: python3 -m flow_sim.sims "
         "{finalspec|papergrid|prodsim|stakeecon|relfigs|rational|fetch} "
         "[args]")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "finalspec":
        finalspec_main()
    elif cmd == "papergrid":
        papergrid_main()
    elif cmd == "prodsim":
        prodsim_main()
    elif cmd == "stakeecon":
        fig_flow() if "--fig" in sys.argv else stakeecon_main()
    elif cmd == "relfigs":
        relfigs_main()
    elif cmd == "rational":
        rational_main(sys.argv[2:])
    elif cmd == "fetch":
        fetch_main(sys.argv[2:])
    else:
        raise SystemExit(USAGE)


if __name__ == "__main__":
    main()
