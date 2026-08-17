"""FLOW challenge scoring (capital-at-risk BTC trades, evidence-gated).

Spec: the FLOW release paper (FLOW_RELEASE.pdf, on the MANTIS site),
sections 1.1-1.8.  Parameters mirror the launch
table and the simulation defaults behind the published results
(same-candle ties resolve stop-first; a stop printed after TP1 but
before TP2 resolves as SL; no path penalty on stop-outs).

Submission encoding (embedding dim 28, four regimes x seven fields):

    [d, f, sl_frac, tp1_frac, tp2_frac, h_hours, trade_id]  per regime

  d         -1 short, +1 long, 0 = flat (row ignored)
  f         Kelly fraction in [0.01, 0.25]
  sl_frac   stop distance as a fraction of entry price (> 0)
  tp1_frac  first target distance  (0 < tp1_frac < tp2_frac)
  tp2_frac  second target distance (both bounded at 50%)
  h_hours   horizon, inside the regime's bounds (A 1-24h, B 24-48h,
            C 72-168h, D 168-336h)
  trade_id  miner-chosen positive integer; a NEW id opens a new trade

Constant-emission semantics: miners repeat their current intent in every
payload; a trade opens at the first sampled row where a valid tuple with
a fresh trade_id is seen and the regime is idle, so dropped rows only
delay an open and never cancel a live trade.  Opens are rate-limited to
one per hour per regime.  Entry E is the validator-recorded price at the
open row; brackets are fractional distances from E, so no payload field
can encode an absolute price, and timelock encryption commits the tuple
before E is known to anyone.

Resolution walks the sampled series to the first of stop cross, TP2
cross, or horizon end.  With [close, high, low] input (the kline feed)
stops trigger on the adverse wick and targets on the favorable wick per
spec 1.3; close-only input is a priced degradation (prodsim.py: it
under-triggers tight stops enough that momentum holds clearance).

Payment simulates the hourly emission allocation over the expanding
panel (EWMA, kappa decay, probation, block-aggregated t-stat gate with
hysteresis, dust tier) and returns the trailing-24h mean allocation,
remainder under "__burn__".  Everything, including latch state, is a
pure function of the shared datalog, so validators agree stateless.

Collateral weighting: THE POSTED BET is the only money.  Live, pass
`collateral_fn` (flow_collateral.make_bet_collateral_fn over TradePosted event logs):
each trade's pay contribution is sized by the bet its miner posted on
chain before the open — an unposted or late-posted (free-look) trade
earns nothing, and no deposit or withdrawal after the open can move
what a past trade pays.  The gate statistic never sees collateral:
proof of edge is evidence-only, and only the split among paid keys
scales with skin.  The static `collateral` map (hotkey -> alpha) remains for
simulations and pre-deployment weighting.

Live settlement: miners post their own bets on chain (FlowCollateralPool
postTrade); `trade_events` yields one record per decoded trade so the
resolution daemon can close each posted bet (closeBatch) at its
realized loss, and `compute_collateral_settlement` aggregates one period.
With the live `collateral_fn`, both legs are symmetric in the bet:
loss = max(-R,0) x bet, claim = max(+R,0) x bet.  Winners split
exactly what the losers forfeited.  Posted through
flow_collateral.CollateralClient.
"""

from __future__ import annotations

import config  # noqa: F401  env pinning before numpy

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ("compute_flow_salience", "compute_collateral_settlement",
           "trade_events", "FlowConfig", "BURN_KEY", "decode_trades",
           "Trade")

BURN_KEY = "__burn__"
SECONDS_PER_BLOCK = 12
FIELDS_PER_REGIME = 7
N_REGIMES = 4
DIM = FIELDS_PER_REGIME * N_REGIMES
MAX_TRADE_ID = 2**24     # float32-exact integer bound (FLOW stores f32)

REGIME_BOUNDS_H = ((1.0, 24.0), (24.0, 48.0), (72.0, 168.0), (168.0, 336.0))


@dataclass
class FlowConfig:
    # --- scoring (release table, section 1.8)
    delta: float = 0.30          # TP2 completion bonus
    lam: float = 0.50            # TP1 partial factor
    gamma: float = 0.25          # path penalty on R_MAE
    tau: float = 1.00            # tail amplifier (2x)
    tail_mult: float = 2.0       # tail when sigma_24h > tail_mult * sigma_30d
    # --- aggregation
    ewma_alpha: float = 0.05
    kappa_hours: float = 672.0   # >= 2x the longest regime horizon (336h)
    probation_hours: float = 168.0
    # --- gate
    gate_block_h: float = 168.0
    gate_min_n: int = 4
    gate_z_in: float = 1.25
    gate_z_out: float = 0.50
    dust: float = 0.02
    # --- settlement circuit breaker: once a book has lost this fraction
    # of its collateral inside one settlement period, its remaining
    # trades that period are void for money (scores still count)
    period_loss_cap: float = 0.25
    # --- validity
    p_bound: float = 0.01
    f_min: float = 0.01
    f_max: float = 0.25
    frac_min: float = 1e-4       # 1 bp minimum bracket distance
    frac_max: float = 0.50
    min_open_gap_h: float = 1.0  # one submission per regime per epoch
    # --- output
    smooth_hours: float = 24.0

    @classmethod
    def from_spec(cls, spec: Optional[dict]) -> "FlowConfig":
        if not spec:
            return cls()
        kw = {}
        for k in ("delta", "lam", "gamma", "tau", "tail_mult", "ewma_alpha",
                  "kappa_hours", "probation_hours", "gate_block_h",
                  "gate_min_n", "gate_z_in", "gate_z_out", "dust",
                  "period_loss_cap", "p_bound", "smooth_hours"):
            if k in spec:
                kw[k] = type(getattr(cls, k))(spec[k])
        return cls(**kw)


@dataclass
class Trade:
    regime: int
    direction: int          # +1 / -1
    f: float
    entry: float
    sl: float
    tp1: float
    tp2: float
    open_row: int
    open_hour: float
    horizon_hour: float
    resolve_row: int = -1
    resolve_hour: float = float("nan")
    event: str = ""         # SL / TP1 / TP2 / F
    s_base: float = float("nan")
    s_weighted: float = float("nan")
    tid: int = 0            # wire trade id (unique per regime per miner)


# ------------------------------------------------------------------ decoding

def _valid_tuple(d: float, f: float, sl: float, tp1: float, tp2: float,
                 h: float, regime: int, cfg: FlowConfig) -> bool:
    di = int(round(d))
    if di not in (-1, 1):
        return False
    if not (cfg.f_min <= f <= cfg.f_max):
        return False
    if not (cfg.frac_min <= sl <= cfg.frac_max):
        return False
    if not (cfg.frac_min <= tp1 < tp2 <= cfg.frac_max):
        return False
    lo, hi = REGIME_BOUNDS_H[regime]
    if not (lo <= h <= hi):
        return False
    b = tp1 / sl
    p = (f * b + 1.0) / (b + 1.0)
    return cfg.p_bound <= p <= 1.0 - cfg.p_bound


def decode_trades(cols: np.ndarray, price: np.ndarray, hours: np.ndarray,
                  cfg: FlowConfig,
                  hi: Optional[np.ndarray] = None,
                  lo: Optional[np.ndarray] = None) -> List[Trade]:
    """State machine over one miner's rows -> resolved/open trades.

    cols: [T, 28] payload rows; price: [T]; hours: [T] row timestamps.
    `hi`/`lo` are optional per-row bar extremes (the interval ending at
    each row) for wick resolution; omitted, resolution is close-only.
    Constant-emission semantics: a new (valid) trade_id opens a trade at
    the first row it is observed; repeated rows and gaps are inert.
    """
    trades: List[Trade] = []
    T = len(price)
    for r in range(N_REGIMES):
        seg = cols[:, r * FIELDS_PER_REGIME:(r + 1) * FIELDS_PER_REGIME]
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        d_arr = np.rint(np.clip(seg[:, 0], -2, 2)).astype(np.int64)
        id_arr = np.rint(np.clip(seg[:, 6], 0, MAX_TRADE_ID)).astype(np.int64)
        seen_ids: set = set()       # every id ever opened/consumed in regime
        open_until_row = -1         # busy while a trade is open
        last_open_hour = -1e18
        t = 0
        while t < T:
            tid = int(id_arr[t])
            # an id is one-shot: reusing any previously seen id is inert,
            # matching the settlement daemon, which keys (regime, id)
            # permanently.  Remembering only the previous id let a miner
            # recycle 1,2,1,2... into unlimited scored-but-unsettled trades.
            if (d_arr[t] == 0 or tid <= 0 or tid >= MAX_TRADE_ID
                    or tid in seen_ids
                    or t <= open_until_row
                    or hours[t] - last_open_hour < cfg.min_open_gap_h):
                t += 1
                continue
            d, f, sl_f, tp1_f, tp2_f, h = (float(x) for x in seg[t, :6])
            E = float(price[t])
            if not _valid_tuple(d, f, sl_f, tp1_f, tp2_f, h, r, cfg) \
                    or not (E > 0):
                # invalid intent: consume this id so a stuck submitter
                # cannot retry the same broken tuple forever
                seen_ids.add(tid)
                t += 1
                continue
            di = int(round(d))
            tr = Trade(
                regime=r, direction=di, f=f, entry=E,
                sl=E * (1.0 - di * sl_f),
                tp1=E * (1.0 + di * tp1_f),
                tp2=E * (1.0 + di * tp2_f),
                open_row=t, open_hour=float(hours[t]),
                horizon_hour=float(hours[t]) + h, tid=tid)
            _resolve(tr, price, hours, cfg, hi, lo)
            trades.append(tr)
            seen_ids.add(tid)
            last_open_hour = float(hours[t])
            open_until_row = tr.resolve_row if tr.resolve_row >= 0 else T
            t += 1
    return trades


def _resolve(tr: Trade, price: np.ndarray, hours: np.ndarray,
             cfg: FlowConfig,
             hi: Optional[np.ndarray] = None,
             lo: Optional[np.ndarray] = None) -> None:
    """First terminal event on the sampled path; stop-first on ties.

    With `hi`/`lo` supplied (bar extremes per row), stops trigger on the
    adverse wick and targets on the favorable wick, per spec section 1.3;
    without them resolution is close-only.  A stop print after TP1
    (before TP2) resolves the trade as SL; MAE is the worst adverse
    print, capped at the stop distance in R units.
    """
    T = len(price)
    j0 = tr.open_row + 1
    end = int(np.searchsorted(hours, tr.horizon_hour, side="right"))
    end = min(max(end, j0), T)
    win = price[j0:end]
    horizon_reached = float(hours[-1]) >= tr.horizon_hour
    if len(win) == 0:
        tr.resolve_row = -1     # still open at panel end
        return
    di = tr.direction
    # adverse/favorable print series: wicks when available, closes otherwise
    w_adv = (lo if di == 1 else hi)
    w_fav = (hi if di == 1 else lo)
    w_adv = w_adv[j0:end] if w_adv is not None else win
    w_fav = w_fav[j0:end] if w_fav is not None else win
    adv = (tr.entry - w_adv) * di                   # adverse move, price units
    hit_sl = (w_adv - tr.sl) * di <= 0              # stop printed
    hit_tp1 = (w_fav - tr.tp1) * di >= 0
    hit_tp2 = (w_fav - tr.tp2) * di >= 0

    i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else T + len(win)
    i_tp1 = int(np.argmax(hit_tp1)) if hit_tp1.any() else T + len(win)
    i_tp2 = int(np.argmax(hit_tp2)) if hit_tp2.any() else T + len(win)

    if i_sl <= min(i_tp2, len(win) - 1) and i_sl < T:
        # stop printed before TP2 (stop-first on the same row): SL event,
        # regardless of an earlier TP1 print (tp1_then_sl = "sl")
        tr.event = "SL"
        tr.resolve_row = j0 + i_sl
    elif i_tp2 < T:
        tr.event = "TP2"
        tr.resolve_row = j0 + i_tp2
    elif not horizon_reached:
        # no terminal print yet and the horizon lies beyond the panel:
        # the trade is still live — scoring a partial window would let
        # incomplete trades enter the gate evidence and be rescored
        # differently on the next expanding-panel recompute
        tr.resolve_row = -1
        return
    elif i_tp1 < T:
        # TP1 hit, TP2 never, no stop: trade runs to horizon, event TP1
        tr.event = "TP1"
        tr.resolve_row = j0 + len(win) - 1
    else:
        tr.event = "F"
        tr.resolve_row = j0 + len(win) - 1
    tr.resolve_hour = float(hours[tr.resolve_row])

    risk = abs(tr.entry - tr.sl)
    upto = tr.resolve_row - j0 + 1
    mae = float(max(adv[:upto].max(), 0.0))
    r_mae = min(mae / risk, 1.0)

    def R(P: float) -> float:
        return di * (P - tr.entry) / risk

    if tr.event == "SL":
        s = -1.0                                    # no path penalty on stops
    elif tr.event == "TP2":
        s = R(tr.tp2) * (1.0 + cfg.delta) - cfg.gamma * r_mae
    elif tr.event == "TP1":
        s = R(tr.tp1) * cfg.lam - cfg.gamma * r_mae
    else:
        s = R(float(price[tr.resolve_row])) - cfg.gamma * r_mae
    tr.s_base = (-1.0 if tr.event == "SL"
                 else s + cfg.gamma * r_mae)        # gate input: pre-path R
    tr.s_weighted = s                               # path applied; tail + f later


# ------------------------------------------------------------------- salience

def _panel(X_flat: np.ndarray, prices_multi: np.ndarray,
           sidx_arr: Optional[np.ndarray], sample_every: int):
    """Unpack the price panel: close, optional wick channels, row hours."""
    pm = np.asarray(prices_multi, dtype=np.float64).reshape(len(X_flat), -1)
    hi_arr = pm[:, 1] if pm.shape[1] >= 3 else None
    lo_arr = pm[:, 2] if pm.shape[1] >= 3 else None
    idx = (np.asarray(sidx_arr, dtype=np.float64) if sidx_arr is not None
           else np.arange(len(pm), dtype=np.float64))
    hours = idx * sample_every * SECONDS_PER_BLOCK / 3600.0
    return pm[:, 0], hi_arr, lo_arr, hours


def _hourly_vol_flags(price: np.ndarray, hours: np.ndarray,
                      cfg: FlowConfig) -> np.ndarray:
    """Per-row boolean: causal sigma_24h > tail_mult * sigma_30d."""
    T = len(price)
    hidx = np.floor(hours).astype(np.int64)
    h0, h1 = int(hidx[0]), int(hidx[-1])
    nh = h1 - h0 + 1
    close = np.full(nh, np.nan)
    for i in range(T):                              # last price in each hour
        close[hidx[i] - h0] = price[i]
    # forward-fill gaps
    last = np.nan
    for i in range(nh):
        if np.isnan(close[i]):
            close[i] = last
        else:
            last = close[i]
    ret = np.zeros(nh)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[1:] = np.diff(np.log(np.maximum(close, 1e-12)))
    ret[~np.isfinite(ret)] = 0.0
    flags_h = np.zeros(nh, dtype=bool)
    for i in range(nh):
        if i < 48:
            continue
        s24 = float(np.std(ret[max(0, i - 24):i], ddof=1)) if i >= 24 else 0.0
        lo30 = max(0, i - 720)
        s30 = (float(np.std(ret[lo30:i], ddof=1))
               if i - lo30 >= 168 else 0.0)
        flags_h[i] = s30 > 0 and s24 > cfg.tail_mult * s30
    return flags_h[hidx - h0]


def _t_stat_window(block_sums: np.ndarray) -> float:
    n = len(block_sums)
    sd = float(np.std(block_sums, ddof=1))
    if sd <= 1e-12:
        return float("-inf")
    return float(np.mean(block_sums)) / sd * math.sqrt(n)


def _t_stat(block_sums: np.ndarray, min_n: int) -> float:
    """Gate statistic: the max of the full-history t and the rolling
    t over the last `min_n` blocks.  The rolling window lets a strong
    recent record clear without waiting out a long cold prefix; the
    full history keeps a long consistent record cleared through a flat
    month.  Undefined (-inf) until `min_n` blocks exist."""
    n = len(block_sums)
    if n < min_n:
        return float("-inf")
    full = _t_stat_window(block_sums)
    roll = _t_stat_window(block_sums[-min_n:])
    return max(full, roll)


def compute_flow_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    prices_multi: np.ndarray,
    *,
    blocks_ahead: int = 0,
    sample_every: int = 5,
    sidx_arr: Optional[np.ndarray] = None,
    spec: Optional[dict] = None,
    cfg: Optional[FlowConfig] = None,
    collateral: Optional[Dict[str, float]] = None,
    collateral_fn: Optional[Callable[..., float]] = None,
    return_diagnostics: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], dict]:
    """hotkey -> fraction of the FLOW pool (+ BURN_KEY; sums to 1).

    Skin: with `collateral_fn` (the live rule; see flow_collateral.
    make_bet_collateral_fn) each trade's pay contribution is priced by the
    bet posted on chain for that trade — f x (bet/f) = the bet — so
    emissions and settlement money share one basis and an unposted or
    late-posted trade earns nothing.  Gate evidence is computed before
    the skin multiplier, so an unposted key still latches; it just is
    not paid.  Without `collateral_fn`, the static `collateral` map multiplies
    the paid score (simulation / pre-deployment behavior); with
    neither, weighting is f-only (paper).
    """
    if cfg is None:
        cfg = FlowConfig.from_spec(spec)

    X_flat, hk2idx = hist
    if not isinstance(hk2idx, dict) or len(hk2idx) == 0:
        return ({}, {}) if return_diagnostics else {}
    H = len(hk2idx)
    if X_flat.shape[1] != H * DIM:
        logger.warning("flow: hist columns %d != H*DIM %d*%d — skipping",
                       X_flat.shape[1], H, DIM)
        return ({}, {}) if return_diagnostics else {}

    price, hi_arr, lo_arr, hours = _panel(X_flat, prices_multi,
                                          sidx_arr, sample_every)
    T = len(price)
    if T < 2 or hours[-1] - hours[0] < cfg.probation_hours:
        out = {BURN_KEY: 1.0}
        return (out, {"reason": "warmup"}) if return_diagnostics else out

    X = np.asarray(X_flat, dtype=np.float64).reshape(T, H, DIM)
    tail_flag = _hourly_vol_flags(price, hours, cfg)

    hk_sorted = sorted(hk2idx, key=lambda h: hk2idx[h])
    per_miner: Dict[str, List[Trade]] = {}
    for hk in hk_sorted:
        cols = X[:, hk2idx[hk], :]
        if not np.any(cols):
            continue
        trades = decode_trades(cols, price, hours, cfg, hi_arr, lo_arr)
        resolved = [t for t in trades if t.resolve_row >= 0]
        if resolved:
            for tr in resolved:                      # tail amp + Kelly weight
                amp = 1.0 + cfg.tau * float(tail_flag[tr.resolve_row])
                tr.s_weighted = tr.s_weighted * amp * tr.f
                if collateral_fn is not None:
                    # the live rule: pay is sized by THE POSTED BET —
                    # f x (bet/f) = the bet — so a trade with no bet
                    # (or a voided late one) earns nothing.  s_base is
                    # untouched: the gate still sees the evidence.
                    tr.s_weighted *= _price_collateral(
                        collateral_fn, hk, tr.open_hour, tr.regime, tr.tid,
                        tr.f)
            per_miner[hk] = resolved

    # ---- hourly emission timeline ------------------------------------
    h_start = int(math.floor(hours[0])) + 1
    h_end = int(math.floor(hours[-1]))
    n_hours = max(h_end - h_start + 1, 1)

    # per-miner resolution schedule sorted by resolve hour
    sched: Dict[str, List[Trade]] = {
        hk: sorted(trs, key=lambda tr: (tr.resolve_hour, tr.open_hour))
        for hk, trs in per_miner.items()}

    state: Dict[str, dict] = {hk: dict(
        ewma=np.zeros(N_REGIMES), next=0, first=None, last=None,
        blocks={}, stat=float("-inf"), latched=False, clear_hour=None,
        paid=np.zeros(n_hours),
    ) for hk in sched}

    alloc_paid = np.zeros(n_hours)
    for hi in range(n_hours):
        h = float(h_start + hi)
        # ingest resolutions due this hour
        for hk, st in state.items():
            trs = sched[hk]
            changed = False
            while st["next"] < len(trs) and trs[st["next"]].resolve_hour <= h:
                tr = trs[st["next"]]
                st["ewma"][tr.regime] = ((1.0 - cfg.ewma_alpha) *
                                         st["ewma"][tr.regime]
                                         + cfg.ewma_alpha * tr.s_weighted)
                if st["first"] is None:
                    st["first"] = tr.resolve_hour
                st["last"] = tr.resolve_hour
                blk = int(tr.resolve_hour // cfg.gate_block_h)
                st["blocks"][blk] = st["blocks"].get(blk, 0.0) + tr.s_base
                st["next"] += 1
                changed = True
            if changed:
                st["stat"] = _t_stat(
                    np.asarray(sorted(st["blocks"].items()))[:, 1]
                    if st["blocks"] else np.empty(0), cfg.gate_min_n)
                if st["latched"]:
                    st["latched"] = st["stat"] >= cfg.gate_z_out
                if not st["latched"] and st["stat"] >= cfg.gate_z_in:
                    st["latched"] = True
                    if st["clear_hour"] is None:
                        st["clear_hour"] = h

        # emission scores this hour
        scores: Dict[str, float] = {}
        for hk, st in state.items():
            if st["first"] is None or h < st["first"] + cfg.probation_hours:
                continue
            s = float(np.sum(st["ewma"]))            # additive, no floor
            if s <= 0:
                continue
            if collateral_fn is None and collateral is not None:
                # legacy/simulation skin: weight by a static per-hotkey
                # balance; zero collateral earns zero (gate unaffected).
                # With collateral_fn the skin is already inside the EWMA,
                # per trade, priced by the posted bet.
                skin = max(float(collateral.get(hk, 0.0)), 0.0)
                if skin <= 0.0:
                    continue
                s *= skin
            scores[hk] = s * math.exp(-(h - st["last"]) / cfg.kappa_hours)

        pool = 1.0
        cleared = {hk for hk, st in state.items()
                   if st["latched"] and hk in scores}
        dust_pool = cfg.dust * pool
        main_pool = pool - dust_pool
        if cleared:
            ctot = sum(scores[hk] for hk in cleared)
            for hk in cleared:
                state[hk]["paid"][hi] += main_pool * scores[hk] / ctot
        uncleared = {hk: s for hk, s in scores.items() if hk not in cleared}
        utot = sum(uncleared.values())
        if utot > 0:
            for hk, s in uncleared.items():
                state[hk]["paid"][hi] += dust_pool * s / utot
        # with no cleared key the main pool burns (no escrow, no credit)
        alloc_paid[hi] = sum(st["paid"][hi] for st in state.values())

    # ---- output: trailing-mean allocation ------------------------------
    win = max(int(cfg.smooth_hours), 1)
    lo = max(n_hours - win, 0)
    out: Dict[str, float] = {}
    for hk, st in state.items():
        w = float(st["paid"][lo:].mean())
        if w <= 0:
            continue
        out[hk] = w
    paid_total = sum(out.values())
    out[BURN_KEY] = max(1.0 - paid_total, 0.0)

    if not return_diagnostics:
        return out
    diag = {hk: dict(
        n_trades=len(sched[hk]), stat=state[hk]["stat"],
        latched=state[hk]["latched"], clear_hour=state[hk]["clear_hour"],
        ewma=float(np.sum(state[hk]["ewma"])),
        paid=state[hk]["paid"],
    ) for hk in sched}
    diag["_pool"] = dict(n_hours=n_hours, n_miners=len(sched),
                         h_start=h_start, paid_trailing=paid_total)
    return out, diag


# --------------------------------------------------------------- settlement

def _price_collateral(collateral_fn, hk: str, open_hour: float, regime: int,
                 tid: int, f: float) -> float:
    """Call a collateral_fn across its historical signatures.

    Preferred protocol is 5-arg (hk, open_hour, regime, tid, f) — the
    bet-basis rule needs f to return bet/f — with 4-arg and 2-arg
    callables still accepted."""
    try:
        priced = collateral_fn(hk, open_hour, regime, tid, f)
    except TypeError:
        try:
            priced = collateral_fn(hk, open_hour, regime, tid)
        except TypeError:
            priced = collateral_fn(hk, open_hour)
    return max(float(priced), 0.0)


def _money_r(tr: Trade, price: np.ndarray) -> float:
    """Realized monetary R-multiple of a resolved trade.

    Money, not score: no delta/lambda event weights, no path penalty, no
    tail amplifier.  A stop realizes exactly the defined risk (-1, the
    floor even across a gapped print); TP2 closes at the target; TP1 and
    horizon events run to the horizon and close at the final price.
    """
    if tr.event == "SL":
        return -1.0
    risk = abs(tr.entry - tr.sl)
    p = tr.tp2 if tr.event == "TP2" else float(price[tr.resolve_row])
    return max(tr.direction * (p - tr.entry) / risk, -1.0)


def trade_events(
    hist: Tuple[np.ndarray, Dict[str, int]],
    prices_multi: np.ndarray,
    *,
    sample_every: int = 5,
    sidx_arr: Optional[np.ndarray] = None,
    spec: Optional[dict] = None,
    cfg: Optional[FlowConfig] = None,
) -> List[dict]:
    """Every trade in the panel as one event record, for the live feed.

    The settlement daemon replays the full expanding panel each cycle
    (the decode is a stateful machine: busy windows and id dedup need
    history), matches each trade against the bet its miner posted on
    chain, and resolves it: `closeBatch` debits the realized loss,
    bounded by the bet.  One record per decoded trade:

        dict(hotkey, regime, tid, open_hour, horizon_hour, f,
             resolve_hour,   # None while the trade is still live
             event,          # "" / SL / TP1 / TP2 / F
             money_r)        # None while live; realized monetary R

    Money is symmetric in THE POSTED BET: loss in alpha =
    max(-money_r, 0) x bet, win = max(money_r, 0) x bet (the daemon
    closes with the bet directly; settlement claims price through
    collateral_fn = bet/f so f x collateral lands back on the bet).
    Deterministic from the datalog, the tape, and the bets on chain,
    so anyone can recompute every posted open and close.
    """
    if cfg is None:
        cfg = FlowConfig.from_spec(spec)
    X_flat, hk2idx = hist
    out: List[dict] = []
    if not (isinstance(hk2idx, dict) and len(hk2idx) > 0
            and X_flat.shape[1] == len(hk2idx) * DIM):
        return out
    price, hi_arr, lo_arr, hours = _panel(X_flat, prices_multi,
                                          sidx_arr, sample_every)
    X = np.asarray(X_flat, dtype=np.float64).reshape(len(price),
                                                     len(hk2idx), DIM)
    for hk, idx in hk2idx.items():
        cols = X[:, idx, :]
        if not np.any(cols):
            continue
        for tr in decode_trades(cols, price, hours, cfg, hi_arr, lo_arr):
            resolved = tr.resolve_row >= 0
            out.append(dict(
                hotkey=hk, regime=tr.regime, tid=tr.tid,
                open_hour=float(tr.open_hour),
                horizon_hour=float(tr.horizon_hour),
                f=float(tr.f),
                resolve_hour=float(tr.resolve_hour) if resolved else None,
                event=tr.event,
                money_r=float(_money_r(tr, price)) if resolved else None))
    return out


def compute_collateral_settlement(
    hist: Tuple[np.ndarray, Dict[str, int]],
    prices_multi: np.ndarray,
    *,
    collateral: Dict[str, float],
    period_start_h: float,
    period_end_h: float,
    carry_losses: Optional[Dict[str, float]] = None,
    sample_every: int = 5,
    sidx_arr: Optional[np.ndarray] = None,
    spec: Optional[dict] = None,
    cfg: Optional[FlowConfig] = None,
    collateral_fn: Optional[Callable[[str, float], float]] = None,
) -> dict:
    """One settlement period's loss pool, in alpha.

    Decodes the same expanding panel as the salience path and settles
    every trade whose resolution falls in [period_start_h, period_end_h):
    per trade, alpha P&L = realized monetary R x f x the alpha held
    behind the hotkey at the trade's open.  Losing hotkeys forfeit
    |P&L|, capped per trade by the exposure the contract reserved at
    the open and clamped to the position on-chain; winning hotkeys hold
    claims.  The batch pays winners pro rata by claim out of exactly
    what the losers forfeited — more or less than the notional win.

    Collateral pricing: pass `collateral_fn(hotkey, open_hour[, regime, tid, f])
    -> alpha` to price each trade (the live rule is
    flow_collateral.make_bet_collateral_fn: bet/f, so P&L = R x bet — claims and
    debits share the posted bet as their one basis, and a trade with
    no bet or a voided late bet prices at zero).  Two- and
    four-argument callables still work.  Without it, the static
    `collateral` map prices every trade, which is exact whenever the collateral
    did not move within the period and is what the simulations use.

    `carry_losses` holds debits deferred from a period that had no
    winners (the zero-sum contract rolls such a pool into the next
    period); they merge into this batch.  Returns

        dict(losses={hk: alpha}, claims={hk: alpha}, pool=float,
             deferred=bool, carry_losses={hk: alpha})

    with `deferred=True` (and everything moved into `carry_losses`) when
    this period has no winners either.  Deterministic from the datalog,
    the tape, and chain state, so anyone can recompute a posted batch.
    Pro-rata rao amounts are computed at posting time
    (flow_collateral.CollateralClient.settle_period) so integer rounding is
    exactly zero-sum.
    """
    if cfg is None:
        cfg = FlowConfig.from_spec(spec)
    X_flat, hk2idx = hist
    losses: Dict[str, float] = dict(carry_losses or {})
    claims: Dict[str, float] = {}
    loss_caps: Dict[str, float] = {}
    if isinstance(hk2idx, dict) and len(hk2idx) > 0 \
            and X_flat.shape[1] == len(hk2idx) * DIM:
        price, hi_arr, lo_arr, hours = _panel(X_flat, prices_multi,
                                              sidx_arr, sample_every)
        X = np.asarray(X_flat, dtype=np.float64).reshape(len(price),
                                                         len(hk2idx), DIM)
        for hk, idx in hk2idx.items():
            static_col = max(float(collateral.get(hk, 0.0)), 0.0)
            if collateral_fn is None and static_col <= 0.0:
                continue                     # no skin, no settlement
            cols = X[:, idx, :]
            if not np.any(cols):
                continue
            resolved = []
            for tr in decode_trades(cols, price, hours, cfg, hi_arr, lo_arr):
                if tr.resolve_row < 0:
                    continue
                if not (period_start_h <= tr.resolve_hour < period_end_h):
                    continue
                if collateral_fn is not None:
                    col = _price_collateral(collateral_fn, hk, tr.open_hour,
                                            tr.regime, tr.tid, tr.f)
                else:
                    col = static_col
                if col <= 0.0:
                    continue
                resolved.append((float(tr.resolve_hour),
                                 _money_r(tr, price) * tr.f * col, col))
            if not resolved:
                continue
            resolved.sort(key=lambda r: r[0])
            cap = max(s for _, _, s in resolved)
            # period circuit breaker: gross losses stop at the cap and
            # every later trade in the period is void for money in both
            # directions (no free option after the breaker trips)
            budget = cfg.period_loss_cap * cap
            pnl = 0.0
            gross_loss = 0.0
            for _, money, _ in resolved:
                if gross_loss >= budget - 1e-12:
                    continue                     # book is capped: void
                if money < 0.0:
                    take = min(-money, budget - gross_loss)
                    gross_loss += take
                    pnl -= take
                else:
                    pnl += money
            if pnl < 0.0:
                losses[hk] = losses.get(hk, 0.0) + (-pnl)
                loss_caps[hk] = max(loss_caps.get(hk, 0.0), cap)
            elif pnl > 0.0:
                claims[hk] = claims.get(hk, 0.0) + pnl

    # cap every loss at the position (the contract independently bounds
    # each debit by the trade's reserved exposure and the live balance)
    for hk in list(losses):
        cap = loss_caps.get(hk, max(float(collateral.get(hk, 0.0)), 0.0))
        losses[hk] = min(losses[hk], cap)
        if losses[hk] <= 0.0:
            del losses[hk]

    pool = sum(losses.values())
    if not claims and losses:
        # no winners this period: defer the debits (the zero-sum
        # contract rolls a winnerless pool into the next period)
        return dict(losses={}, claims={}, pool=0.0, deferred=True,
                    carry_losses=losses)
    return dict(losses=losses, claims=claims, pool=pool, deferred=False,
                carry_losses={})
