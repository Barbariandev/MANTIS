# MIT License
# Copyright (c) 2024 MANTIS
"""The LOOP evaluation harness: the one piece of code that scores a model.

A model on LOOP is a *spec* — a causal feature on one of the dataset's
market series, and the hyperparameters of the BTC bracket it trades —
not code.  Everything that turns a spec into a number or a trade lives
here and belongs to the protocol, not to the miner:

    evaluate(dataset, spec)            walk-forward metrics on the research
                                       window (what a pipeline sees)
    holdout(dataset, spec, bars)       the same model on target bars the
                                       pipeline never saw
    signal(dataset, spec)              the trade the model takes at the
                                       latest bar

So a miner cannot report metrics the model did not earn, cannot peek at
future bars, and cannot trade something other than its model: there is
no miner code in the path.  The same source (``HARNESS_SOURCE``,
identified by ``HARNESS_SHA256``) runs on the gateway's metered compute
during a run and again inside every validator's audit, which must
reproduce the gateway's numbers.

The dataset is one JSON object holding several series — the hourly
**target** the model trades (BTC), and auxiliary series such as ten
years of daily BTC, forex majors, ETH and gold (``data.py`` builds the
canonical one).  A feature may be computed on any series; its value at
a target bar is the value at the last bar of that series that had
**closed** by then, so a cross-market feature cannot read anything the
market had not yet printed.  Lookbacks are stated in hours and capped
(``MAX_LOOKBACK_HOURS``) on every series alike.

The source is stdlib-only and deterministic: no clock, no randomness, no
I/O beyond stdin/stdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

REGIME_BOUNDS_H = ((1.0, 24.0), (24.0, 48.0), (72.0, 168.0), (168.0, 336.0))

TRANSFORMS = ("zscore_momentum", "vol_ratio", "rsi", "autocorr", "range_pos")
MIN_LOOKBACK_HOURS = 8
MAX_LOOKBACK_HOURS = 720            # 30 days: the anti-smuggling cap

# dataset shape limits (what one evaluation may be asked to read)
MAX_TARGET_BARS = 20_000            # ~2.3 years of hourly trade labels
MAX_SERIES_BARS = 4_400             # ~10 years of daily bars
MAX_SERIES = 16
_SERIES_NAME = re.compile(r"^[A-Za-z0-9_]{1,24}$")

HYPERPARAM_RANGES = {
    "C": (1e-4, 1e3),
    "threshold": (0.5, 0.99),
    "f": (0.01, 0.25),
    "sl": (1e-4, 0.5),
    "tp1": (1e-4, 0.5),
    "tp2": (1e-4, 0.5),
    "horizon_h": (1.0, 336.0),
    "regime": (0, 3),
}
REQUIRED_HYPERPARAMS = ("threshold", "horizon_h", "regime", "f", "sl",
                        "tp1", "tp2")

HARNESS_SOURCE = r'''
import json, math, sys

FOLDS = 4
FEE = 0.0010

def _feature(closes, kind, lb):
    """One causal transform over a series, rolling sums: O(n)."""
    n = len(closes)
    r = [0.0] * n
    for i in range(1, n):
        r[i] = math.log(closes[i] / closes[i - 1])
    # prefix sums over returns (index i holds sums of r[1..i])
    p1 = [0.0] * n; p2 = [0.0] * n; pu = [0.0] * n; pd = [0.0] * n
    pp = [0.0] * n                        # sum of r[j] * r[j-1]
    for i in range(1, n):
        p1[i] = p1[i - 1] + r[i]
        p2[i] = p2[i - 1] + r[i] * r[i]
        pu[i] = pu[i - 1] + (r[i] if r[i] > 0 else 0.0)
        pd[i] = pd[i - 1] - (r[i] if r[i] < 0 else 0.0)
        pp[i] = pp[i - 1] + r[i] * r[i - 1]

    def _sums(a, b):                      # sums of r[a..b]
        return p1[b] - p1[a - 1], p2[b] - p2[a - 1]

    def _sd(s1, s2, m):
        var = (s2 - s1 * s1 / m) / (m - 1) if m > 1 else 0.0
        return math.sqrt(var) if var > 0 else 0.0

    out = [None] * n
    if kind == "range_pos":               # rolling max/min via deques
        hi, lo = [], []
        for i in range(n):
            while hi and closes[hi[-1]] <= closes[i]: hi.pop()
            while lo and closes[lo[-1]] >= closes[i]: lo.pop()
            hi.append(i); lo.append(i)
            start = i - lb + 1
            if hi[0] < start: hi = [k for k in hi if k >= start]
            if lo[0] < start: lo = [k for k in lo if k >= start]
            if i >= lb:
                h, l = closes[hi[0]], closes[lo[0]]
                out[i] = ((closes[i] - l) / (h - l) - 0.5) * 2.0 \
                    if h > l else 0.0
        return out

    for i in range(lb, n):
        a = i - lb + 1                    # window is r[a..i], length lb
        s1, s2 = _sums(a, i)
        sd = _sd(s1, s2, lb)
        if kind == "zscore_momentum":
            out[i] = (s1 / (sd * math.sqrt(lb))) if sd > 0 else 0.0
        elif kind == "vol_ratio":
            a2 = a + lb // 2              # the window's second half
            h1, h2 = _sums(a2, i)
            v2 = _sd(h1, h2, i - a2 + 1)
            out[i] = (v2 / sd - 1.0) if sd > 0 else 0.0
        elif kind == "rsi":
            up = pu[i] - pu[a - 1]; dn = pd[i] - pd[a - 1]
            out[i] = (up / (up + dn) - 0.5) * 2.0 if (up + dn) > 0 else 0.0
        elif kind == "autocorr":
            m = lb - 1                    # corr(r[a..i-1], r[a+1..i])
            sa, sa2 = _sums(a, i - 1)
            sb, sb2 = _sums(a + 1, i)
            sab = pp[i] - pp[a]
            num = sab - sa * sb / m
            da = sa2 - sa * sa / m; db = sb2 - sb * sb / m
            den = math.sqrt(da * db) if da > 0 and db > 0 else 0.0
            out[i] = num / den if den > 0 else 0.0
        else:
            out[i] = 0.0
    return out

def _aligned(ds, name, kind, lb_hours):
    """The feature on series ``name``, read at each target bar: the value
    at the last bar of ``name`` that had closed by that target bar's
    close.  Lookback is stated in hours and converted to whole bars of
    the series it is computed on (never fewer than 2)."""
    tgt = ds["series"][ds["target"]]
    s = ds["series"][name]
    lb = max(int(round(lb_hours / s["interval_hours"])), 2)
    vals = _feature(s["closes"], kind, lb)
    if name == ds["target"]:
        return vals
    out = [None] * len(tgt["ts"])
    sts = s["ts"]
    j = -1
    for i, t in enumerate(tgt["ts"]):
        while j + 1 < len(sts) and sts[j + 1] <= t:
            j += 1
        out[i] = vals[j] if j >= 0 else None
    return out

def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))

def _fit(f, ys, rows, l2):
    """Logistic fit of up/down on one feature, standardised on the
    training rows.  Returns a scorer and the training base rate."""
    xs = [f[i] for i in rows]
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(n - 1, 1))
    zs = [((x - mu) / sd) if sd > 0 else 0.0 for x in xs]
    lab = [ys[i] for i in rows]
    base = sum(lab) / n
    w = 0.0
    b = math.log(base / (1.0 - base)) if 0.0 < base < 1.0 else 0.0
    for _ in range(200):
        gw = gb = 0.0
        for z, y in zip(zs, lab):
            p = _sigmoid(w * z + b)
            gw += (p - y) * z; gb += (p - y)
        w -= 0.1 * (gw / n + l2 * w); b -= 0.1 * gb / n
    def score(x):
        return _sigmoid(w * (((x - mu) / sd) if sd > 0 else 0.0) + b)
    return score, base

def _decide(p, base, thr):
    """The direction is the model's edge over the base rate, never the
    base rate itself: a model whose feature says nothing does not trade,
    however often the market went up in training."""
    margin = thr - 0.5
    if p - base >= margin: return 1
    if base - p >= margin: return -1
    return 0

def _auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels); neg = len(labels) - pos
    if pos == 0 or neg == 0: return 0.5
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]: j += 1
        avg_rank = (i + j + 1) / 2.0
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pairs[k][1] == 1)
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)

def _bracket_r(closes, t, d, sl, tp1, tp2, horizon, lam=0.5, delta=0.3):
    E = closes[t]
    sl_px  = E * (1.0 - d * sl)
    tp1_px = E * (1.0 + d * tp1)
    tp2_px = E * (1.0 + d * tp2)
    hit1 = False
    fee = FEE / sl
    end = min(t + horizon, len(closes) - 1)
    for i in range(t + 1, end + 1):
        px = closes[i]
        if (d > 0 and px <= sl_px) or (d < 0 and px >= sl_px):
            return -1.0 - fee
        if (d > 0 and px >= tp2_px) or (d < 0 and px <= tp2_px):
            return (tp1 / sl) * lam + (tp2 / sl) * (1.0 - lam) * (1.0 + delta) - fee
        if (d > 0 and px >= tp1_px) or (d < 0 and px <= tp1_px):
            hit1 = True
    fin = d * (closes[end] / E - 1.0) / sl
    return (lam * (tp1 / sl) + (1.0 - lam) * max(min(fin, tp1 / sl), -1.0)
            if hit1 else max(min(fin, 1.0), -1.0)) - fee

def _params(ds, spec):
    feat = spec["feature"]; hp = spec["hyperparams"]
    series = feat.get("series") or ds["target"]
    return (series, feat["transform"], max(int(feat["lookback_hours"]), 8),
            max(int(hp["horizon_h"]), 1), float(hp.get("threshold", 0.6)),
            1.0 / max(float(hp.get("C", 1.0)), 1e-6),
            float(hp["sl"]), float(hp["tp1"]), float(hp["tp2"]))

def _stats(rs):
    nt = len(rs)
    mean_r = sum(rs) / nt if nt else 0.0
    var_r = sum((r - mean_r) ** 2 for r in rs) / (nt - 1) if nt > 1 else 0.0
    t = mean_r / math.sqrt(var_r / nt) if nt > 1 and var_r > 0 else 0.0
    return nt, round(mean_r, 4), round(t, 3)

QUANTILE_BANDS = (4.0, 25.0, 75.0, 96.0)

def _quantile_profile(f, prices, rows, horizon):
    """Mean forward return by feature band: the feature's distribution
    is partitioned around its 1st/10th/50th/90th/99th percentiles (4%
    tails, shoulders, the middle half) and each band's mean forward
    return over ``horizon`` bars is reported.  A strictly ordered chain
    of means — rising or falling — is a monotone response the threshold
    did not invent: a high-confidence receipt no fit can fake."""
    pairs = []
    n = len(prices)
    for i in rows:
        j = i + horizon
        if j < n and f[i] is not None:
            pairs.append((f[i], prices[j] / prices[i] - 1.0))
    if len(pairs) < 200:
        return None, 0
    pairs.sort()
    m = len(pairs)
    cuts = [0] + [int(e * m / 100.0) for e in QUANTILE_BANDS] + [m]
    means = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 5:
            return None, 0
        means.append(sum(fr for _, fr in pairs[a:b]) / (b - a))
    up = all(means[k] < means[k + 1] for k in range(len(means) - 1))
    dn = all(means[k] > means[k + 1] for k in range(len(means) - 1))
    return [round(v, 5) for v in means], (1 if up else -1 if dn else 0)

def _monotone_at(ds, series, kind, lb_hours, horizon):
    prices = ds["series"][ds["target"]]["closes"]
    f = _aligned(ds, series, kind, lb_hours)
    rows = [i for i in range(len(prices) - horizon) if f[i] is not None]
    return _quantile_profile(f, prices, rows, horizon)

def _trade(prices, f, score, base, thr, rows, horizon, sl, tp1, tp2):
    """Trade the model over ``rows``.  Two baselines ride along: what
    always-long with the same bracket made there (so nobody mistakes
    drift for a model), and what the same fitted model makes trading its
    signal's sign on every bar — the naked factor, no selectivity (so
    nobody mistakes beta for a conditional system: if the thresholded
    trades do not clearly beat this, the filter added nothing)."""
    rs, longs, hold, unf = [], 0, [], []
    last, last_hold, last_unf = -10**9, -10**9, -10**9
    scores = []
    for i in rows:
        p = score(f[i]); scores.append(p)
        if i - last_hold >= horizon:
            hold.append(_bracket_r(prices, i, 1, sl, tp1, tp2, horizon))
            last_hold = i
        if i - last_unf >= horizon:
            du = 1 if p >= base else -1
            unf.append(_bracket_r(prices, i, du, sl, tp1, tp2, horizon))
            last_unf = i
        d = _decide(p, base, thr)
        if d != 0 and i - last >= horizon:
            rs.append(_bracket_r(prices, i, d, sl, tp1, tp2, horizon))
            longs += 1 if d > 0 else 0
            last = i
    return rs, longs, hold, unf, scores

def evaluate(ds, spec):
    """Walk-forward on the target, expanding window, embargo = horizon."""
    series, kind, lb, horizon, thr, l2, sl, tp1, tp2 = _params(ds, spec)
    if series not in ds["series"]:
        return {"error": "no such series", "series": str(series)[:40]}
    prices = ds["series"][ds["target"]]["closes"]
    f = _aligned(ds, series, kind, lb)
    n = len(prices)
    ys = [1 if (i + horizon < n and prices[i + horizon] > prices[i]) else 0
          for i in range(n)]
    valid = [i for i in range(n - horizon) if f[i] is not None]
    if len(valid) < 200:
        return {"error": "too little data", "n": len(valid)}
    seg = len(valid) // FOLDS
    oos_scores, oos_labels, rs, hold, unf, longs = [], [], [], [], [], 0
    for k in range(1, FOLDS):
        tr = valid[:k * seg - horizon]            # embargo = horizon
        te = valid[k * seg:(k + 1) * seg]
        if len(tr) < 50 or not te: continue
        score, base = _fit(f, ys, tr, l2)
        r, lg, h, u, sc = _trade(prices, f, score, base, thr, te, horizon,
                                 sl, tp1, tp2)
        rs += r; longs += lg; hold += h; unf += u
        oos_scores += sc; oos_labels += [ys[i] for i in te]
    nt, mean_r, t = _stats(rs)
    _hn, hold_r, hold_t = _stats(hold)
    _un, unf_r, unf_t = _stats(unf)
    quantile_r, mono = _quantile_profile(f, prices, valid, horizon)
    robust = mono != 0
    if robust:
        # the chain must survive a less specific feature: the same
        # transform with its lookback pulled a quarter either way
        for scale in (0.75, 4.0 / 3.0):
            lb2 = max(int(round(lb * scale)), 2)
            _q2, m2 = _monotone_at(ds, series, kind, lb2, horizon)
            if m2 != mono:
                robust = False
                break
    return {"oos_auc": round(_auc(oos_scores, oos_labels), 4)
            if oos_scores else 0.5,
            "n_trades": nt, "avg_r": mean_r, "t_stat": t,
            "long_frac": round(longs / nt, 3) if nt else 0.0,
            "always_long_avg_r": hold_r, "always_long_t": hold_t,
            "unfiltered_avg_r": unf_r, "unfiltered_t": unf_t,
            "quantile_r": quantile_r, "monotone": mono,
            "monotone_robust": robust,
            "n_folds": FOLDS - 1, "embargo_hours": horizon}

def holdout(ds, spec, bars):
    """Fit on everything before the last ``bars`` target bars (labels
    included: nothing from the holdout leaks into training), trade them."""
    series, kind, lb, horizon, thr, l2, sl, tp1, tp2 = _params(ds, spec)
    if series not in ds["series"]:
        return {"error": "no such series", "series": str(series)[:40]}
    prices = ds["series"][ds["target"]]["closes"]
    f = _aligned(ds, series, kind, lb)
    n = len(prices)
    cut = n - int(bars)
    tr = [i for i in range(cut - horizon) if f[i] is not None]
    te = [i for i in range(cut, n - 1) if f[i] is not None]
    if len(tr) < 200 or not te:
        return {"error": "too little data", "bars": int(bars)}
    ys = {i: (1 if prices[i + horizon] > prices[i] else 0) for i in tr}
    score, base = _fit(f, ys, tr, l2)
    rs, longs, hold, unf, _sc = _trade(prices, f, score, base, thr, te,
                                       horizon, sl, tp1, tp2)
    nt, mean_r, t = _stats(rs)
    _hn, hold_r, _ht = _stats(hold)
    _un, unf_r, _ut = _stats(unf)
    return {"bars": int(bars), "n_trades": nt, "avg_r": mean_r, "t_stat": t,
            "long_frac": round(longs / nt, 3) if nt else 0.0,
            "always_long_avg_r": hold_r, "unfiltered_avg_r": unf_r}

def signal(ds, spec):
    """The model's decision at the latest target bar: fit on every
    labelled row (labels need ``horizon`` bars of future, so the newest
    rows are features-only), then score the last bar."""
    series, kind, lb, horizon, thr, l2, sl, tp1, tp2 = _params(ds, spec)
    if series not in ds["series"]:
        return {"error": "no such series", "d": 0}
    prices = ds["series"][ds["target"]]["closes"]
    f = _aligned(ds, series, kind, lb)
    n = len(prices)
    tr = [i for i in range(n - horizon) if f[i] is not None]
    if len(tr) < 200 or f[n - 1] is None:
        return {"error": "too little data", "d": 0}
    ys = {i: (1 if prices[i + horizon] > prices[i] else 0) for i in tr}
    score, base = _fit(f, ys, tr, l2)
    p = score(f[n - 1])
    return {"p": round(p, 4), "base": round(base, 4),
            "d": _decide(p, base, thr), "feature_value": round(f[n - 1], 4)}

if __name__ == "__main__":
    _in = json.loads(sys.stdin.read())
    _op, _ds, _spec = _in["op"], _in["dataset"], _in["spec"]
    if _op == "evaluate":
        _out = evaluate(_ds, _spec)
    elif _op == "holdout":
        _out = holdout(_ds, _spec, _in["bars"])
    else:
        _out = signal(_ds, _spec)
    print(json.dumps(_out, sort_keys=True))
'''

HARNESS_SHA256 = hashlib.sha256(HARNESS_SOURCE.encode("utf-8")).hexdigest()

_NS: Dict[str, Any] = {"__name__": "loop_harness"}
exec(compile(HARNESS_SOURCE, "<loop-harness>", "exec"), _NS)
evaluate: Callable[[dict, dict], dict] = _NS["evaluate"]
holdout: Callable[[dict, dict, int], dict] = _NS["holdout"]
signal: Callable[[dict, dict], dict] = _NS["signal"]


# ------------------------------------------------------------------ specs

def vocabulary(dataset: Optional[dict] = None) -> dict:
    """What a spec may contain — published so pipelines can build their
    prompts from it instead of hard-coding it.  With a ``dataset``, also
    which market series a feature may be computed on."""
    out = {
        "transforms": list(TRANSFORMS),
        "lookback_hours": [MIN_LOOKBACK_HOURS, MAX_LOOKBACK_HOURS],
        "hyperparams": {k: list(v) for k, v in HYPERPARAM_RANGES.items()},
        "required_hyperparams": list(REQUIRED_HYPERPARAMS),
        "regime_horizon_hours": [list(b) for b in REGIME_BOUNDS_H],
        "rules": ["tp1 < tp2", "horizon_h inside the regime's bounds",
                  "feature.series defaults to the target"],
    }
    if dataset is not None:
        out["target"] = dataset["target"]
        out["series"] = {
            name: {"interval_hours": s["interval_hours"],
                   "bars": len(s["closes"])}
            for name, s in sorted(dataset["series"].items())}
    return out


def _number(x: Any) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError("not a number")
    if not math.isfinite(x):
        raise ValueError("not finite")
    return float(x)


def clean_spec(raw: Any,
               series_names: Optional[List[str]] = None
               ) -> Tuple[dict, List[str]]:
    """Validate a model spec.  Returns ``(spec, problems)``: the spec
    with only the fields the harness reads, in canonical types, and
    every reason it may not be run (empty when it may).

    These are the mechanical rules of the ruleset (R1–R3): a feature
    from the causal vocabulary, computed on one of the dataset's series
    (when ``series_names`` is given), a lookback inside the cap, sane
    hyperparameters, a horizon inside the regime it trades.
    """
    problems: List[str] = []
    feat = raw.get("feature") if isinstance(raw, dict) else None
    hp = raw.get("hyperparams") if isinstance(raw, dict) else None
    if not isinstance(feat, dict) or not isinstance(hp, dict):
        return {}, ["a spec is {\"feature\": {...}, \"hyperparams\": {...}}"]

    transform = feat.get("transform")
    if transform not in TRANSFORMS:
        problems.append(f"transform must be one of {', '.join(TRANSFORMS)}")
    lookback = 0
    try:
        lookback = int(_number(feat.get("lookback_hours")))
        if not MIN_LOOKBACK_HOURS <= lookback <= MAX_LOOKBACK_HOURS:
            raise ValueError
    except ValueError:
        problems.append(f"lookback_hours must be {MIN_LOOKBACK_HOURS}.."
                        f"{MAX_LOOKBACK_HOURS}")
    series = feat.get("series")
    if series is not None and (not isinstance(series, str)
                               or not _SERIES_NAME.match(series)
                               or (series_names is not None
                                   and series not in series_names)):
        problems.append(
            "feature.series must be one of the dataset's series"
            + (f" ({', '.join(sorted(series_names))})" if series_names
               else ""))

    out_hp: Dict[str, Any] = {}
    for key, (lo, hi) in HYPERPARAM_RANGES.items():
        if key not in hp:
            if key in REQUIRED_HYPERPARAMS:
                problems.append(f"hyperparameter {key} is required")
            continue
        try:
            value = _number(hp[key])
            if not lo <= value <= hi:
                raise ValueError
        except ValueError:
            problems.append(f"hyperparameter {key} must be {lo:g}..{hi:g}")
            continue
        out_hp[key] = int(value) if key in ("horizon_h", "regime") else value
    if not problems:
        if not out_hp["tp1"] < out_hp["tp2"]:
            problems.append("tp1 must be below tp2")
        lo, hi = REGIME_BOUNDS_H[out_hp["regime"]]
        if not lo <= out_hp["horizon_h"] <= hi:
            problems.append(f"horizon_h must be {lo:g}..{hi:g} hours in "
                            f"regime {out_hp['regime']}")
    out_feat: Dict[str, Any] = {"transform": str(transform),
                                "lookback_hours": lookback}
    if isinstance(series, str):
        out_feat["series"] = series
    spec = {"feature": out_feat, "hyperparams": out_hp}
    return spec, problems


# ---------------------------------------------------------------- datasets

def dataset_text(dataset: dict) -> str:
    """The canonical serialisation of a dataset; its sha256 is the
    dataset's identity."""
    return json.dumps(dataset, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def dataset_sha256(dataset: dict) -> str:
    return hashlib.sha256(dataset_text(dataset).encode("utf-8")).hexdigest()


def load_dataset(text: str) -> dict:
    """Parse and vet a dataset blob.  Raises ValueError on anything that
    is not a usable set of market series.

    A dataset is ``{"version": 2, "target": name, "series": {name:
    {"interval_hours": int, "ts": [...], "closes": [...]}}}`` — ``ts``
    is each bar's **close** time (epoch seconds, strictly increasing),
    so alignment across series can never read an unclosed bar.
    """
    ds = json.loads(text)
    if not isinstance(ds, dict) or not isinstance(ds.get("series"), dict):
        raise ValueError("a dataset is {version, target, series}")
    target = ds.get("target")
    series = ds["series"]
    if not 1 <= len(series) <= MAX_SERIES:
        raise ValueError(f"1..{MAX_SERIES} series")
    if target not in series:
        raise ValueError("the target must be one of the series")
    out: Dict[str, Any] = {}
    for name, s in series.items():
        if not isinstance(name, str) or not _SERIES_NAME.match(name):
            raise ValueError("series names are [A-Za-z0-9_]{1,24}")
        if not isinstance(s, dict):
            raise ValueError(f"series {name}: not an object")
        try:
            interval = int(_number(s.get("interval_hours")))
            if not 1 <= interval <= 168:
                raise ValueError
        except ValueError:
            raise ValueError(f"series {name}: interval_hours must be 1..168")
        ts, closes = s.get("ts"), s.get("closes")
        cap = MAX_TARGET_BARS if name == target else MAX_SERIES_BARS
        floor = 300 if name == target else 2
        if (not isinstance(ts, list) or not isinstance(closes, list)
                or len(ts) != len(closes)
                or not floor <= len(closes) <= cap):
            raise ValueError(f"series {name}: ts and closes must match, "
                             f"{floor}..{cap} bars")
        out_ts: List[int] = []
        out_px: List[float] = []
        for t, p in zip(ts, closes):
            t = int(_number(t))
            p = _number(p)
            if p <= 0:
                raise ValueError(f"series {name}: prices must be positive")
            if out_ts and t <= out_ts[-1]:
                raise ValueError(f"series {name}: ts must strictly increase")
            out_ts.append(t)
            out_px.append(p)
        out[name] = {"interval_hours": interval, "ts": out_ts,
                     "closes": out_px}
    return {"version": 2, "target": str(target), "series": out}


def research_window(dataset: dict, holdout_bars: int) -> dict:
    """The dataset a pipeline is shown: the last ``holdout_bars`` target
    bars removed, and **every** series truncated at the same moment, so
    no series carries information from the held-back window."""
    tgt = dataset["series"][dataset["target"]]
    n = len(tgt["closes"]) - int(holdout_bars)
    if n < 300:
        raise ValueError("the dataset is too short to hold bars back")
    cutoff = tgt["ts"][n - 1]
    series = {}
    for name, s in dataset["series"].items():
        k = len(s["ts"])
        while k > 0 and s["ts"][k - 1] > cutoff:
            k -= 1
        series[name] = {"interval_hours": s["interval_hours"],
                        "ts": s["ts"][:k], "closes": s["closes"][:k]}
    return {"version": 2, "target": dataset["target"], "series": series}


def single_series(closes: List[float], name: str = "BTC_1h",
                  interval_hours: int = 1, end_ts: int = 1_760_000_000
                  ) -> dict:
    """Wrap a plain list of closes as a one-series dataset (tests, local
    development).  Deterministic timestamps ending at ``end_ts``."""
    step = interval_hours * 3600
    start = end_ts - (len(closes) - 1) * step
    return {"version": 2, "target": name, "series": {
        name: {"interval_hours": interval_hours,
               "ts": [start + i * step for i in range(len(closes))],
               "closes": [float(p) for p in closes]}}}


def target_closes(dataset: dict) -> List[float]:
    return dataset["series"][dataset["target"]]["closes"]


# ------------------------------------------------------------------ trades

def trade(spec: dict, sig: dict, trade_id: int) -> dict:
    """The trade a model takes given its ``signal``: direction from the
    signal, bracket from the spec."""
    hp = spec["hyperparams"]
    d = int(sig.get("d", 0) or 0)
    return {"regime": int(hp["regime"]), "d": d, "p": sig.get("p"),
            "base": sig.get("base"), "f": float(hp["f"]),
            "sl": float(hp["sl"]), "tp1": float(hp["tp1"]),
            "tp2": float(hp["tp2"]), "h": float(hp["horizon_h"]),
            "trade_id": int(trade_id) if d else 0}


def payload_vector(tr: dict) -> List[float]:
    """A trade as the 28-float FLOW wire vector (flat when ``d`` is 0)."""
    vector = [0.0] * 28
    if tr and int(tr.get("d", 0) or 0) != 0:
        base = int(tr["regime"]) * 7
        vector[base:base + 7] = [float(tr["d"]), float(tr["f"]),
                                 float(tr["sl"]), float(tr["tp1"]),
                                 float(tr["tp2"]), float(tr["h"]),
                                 float(tr["trade_id"])]
    return vector
