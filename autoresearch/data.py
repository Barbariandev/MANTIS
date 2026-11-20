# MIT License
# Copyright (c) 2024 MANTIS
"""The canonical market data sealed runs research.

One dataset, one hash, several markets:

    BTC_1h     Binance BTCUSDT hourly closes — the **target**: the series
               the model trades and the walk-forward labels come from
               (up to ~2.3 years; ``harness.MAX_TARGET_BARS``)
    BTC_1d     Coinbase BTC-USD daily closes, ~10 years
    EURUSD_1d, USDJPY_1d, GBPUSD_1d, AUDUSD_1d, USDCHF_1d
               ECB reference rates (Frankfurter), ~10 years of weekdays
    ETH_1d     Binance ETHUSDT daily closes, since 2017
    XAU_1d     Binance PAXGUSDT daily closes (tokenised gold), since 2020

Every bar carries its **close** time, so the harness can align a daily
forex bar to an hourly BTC bar without ever reading a bar that had not
closed.  ECB rates are published around 15:15 UTC; their bars are
stamped 17:00 UTC to be safely on the late side — a feature may learn
of a rate later than the market did, never earlier.

The dataset is fetched by the gateway when a run is opened (cached for
``ttl``), stored content-addressed, and named in the dossier by its
sha256 — a validator replays on exactly these bars.  Auxiliary series
that cannot be fetched are dropped with a warning (the dataset is still
valid); a target that cannot be fetched fails the submit.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import httpx

from . import harness

logger = logging.getLogger(__name__)

TEN_YEARS_DAYS = 3660
_UA = {"User-Agent": "mantis-loop/2"}


# ---------------------------------------------------------------- binance

def binance_klines(symbol: str, interval: str, interval_s: int,
                   n_bars: int) -> List[Tuple[int, float]]:
    """Closed bars as ``(close_ts, close)``, oldest first."""
    out: List[Tuple[int, float]] = []
    end_time: Optional[int] = None
    while len(out) < n_bars:
        params: Dict[str, object] = {"symbol": symbol, "interval": interval,
                                     "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        r = httpx.get("https://api.binance.com/api/v3/klines",
                      params=params, timeout=30.0, headers=_UA)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out = [(int(k[0]) // 1000 + interval_s, float(k[4]))
               for k in rows] + out
        end_time = int(rows[0][0]) - 1
    out = out[-n_bars:]
    now = time.time()
    while out and out[-1][0] > now:        # drop the still-forming bar
        out.pop()
    return out


def fetch_btc_hourly(n_hours: int = 4000) -> Tuple[List[float], dict]:
    """Hourly BTC closes with provenance (kept for older callers)."""
    bars = binance_klines("BTCUSDT", "1h", 3600, n_hours)
    return [p for _t, p in bars], {
        "source": "binance BTCUSDT 1h klines (close)", "bars": len(bars),
        "first_close_utc": _utc(bars[0][0]), "last_close_utc": _utc(bars[-1][0]),
        "last_close": bars[-1][1]}


# --------------------------------------------------------------- coinbase

def coinbase_daily(product: str, days: int) -> List[Tuple[int, float]]:
    """Daily closes as ``(close_ts, close)``, oldest first (300/request)."""
    day = 86400
    end = int(time.time()) // day * day            # last complete UTC day
    start = end - days * day
    out: List[Tuple[int, float]] = []
    lo = start
    while lo < end:
        hi = min(lo + 300 * day, end)
        r = httpx.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles",
            params={"granularity": day, "start": _iso(lo), "end": _iso(hi)},
            timeout=30.0, headers=_UA)
        r.raise_for_status()
        # rows are [bucket_start, low, high, open, close, volume], newest
        # first; a bucket that starts at t closes at t + 1 day
        out += sorted((int(row[0]) + day, float(row[4]))
                      for row in r.json())
        lo = hi
    return [b for i, b in enumerate(out) if i == 0 or b[0] > out[i - 1][0]]


# ------------------------------------------------------------------- ecb

# Frankfurter serves USD-base rates; conventional quotes invert some.
FX_PAIRS = {"EURUSD_1d": ("EUR", True), "USDJPY_1d": ("JPY", False),
            "GBPUSD_1d": ("GBP", True), "AUDUSD_1d": ("AUD", True),
            "USDCHF_1d": ("CHF", False)}
_ECB_CLOSE_UTC = 17 * 3600      # rates publish ~15:15 UTC; stamp late


def frankfurter_fx(days: int) -> Dict[str, List[Tuple[int, float]]]:
    """~10 years of ECB daily reference rates for the majors, stamped at
    17:00 UTC on their date (safely after publication)."""
    start = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
    r = httpx.get(f"https://api.frankfurter.app/{start}..",
                  params={"from": "USD",
                          "to": ",".join(c for c, _ in FX_PAIRS.values())},
                  timeout=60.0, headers=_UA, follow_redirects=True)
    r.raise_for_status()
    rates = r.json()["rates"]
    out: Dict[str, List[Tuple[int, float]]] = {n: [] for n in FX_PAIRS}
    for date in sorted(rates):
        ts = int(time.mktime(time.strptime(date, "%Y-%m-%d"))
                 - time.timezone) + _ECB_CLOSE_UTC
        for name, (code, invert) in FX_PAIRS.items():
            rate = rates[date].get(code)
            if rate:
                out[name].append((ts, 1.0 / float(rate) if invert
                                  else float(rate)))
    return out


# ---------------------------------------------------------------- dataset

def _utc(ts: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts))


def _iso(ts: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _series(bars: List[Tuple[int, float]], interval_hours: int,
            cap: int = harness.MAX_SERIES_BARS) -> dict:
    bars = bars[-cap:]
    return {"interval_hours": interval_hours,
            "ts": [t for t, _p in bars], "closes": [p for _t, p in bars]}


def build_dataset(n_hours: int = harness.MAX_TARGET_BARS,
                  days: int = TEN_YEARS_DAYS) -> Tuple[dict, dict]:
    """The canonical dataset and its provenance.  Raises if the target
    cannot be fetched; auxiliary series are dropped with a warning."""
    target = binance_klines("BTCUSDT", "1h", 3600, n_hours)
    if len(target) < 300:
        raise RuntimeError("could not fetch the BTC hourly target")
    cutoff = target[-1][0]

    series: Dict[str, dict] = {
        "BTC_1h": _series(target, 1, harness.MAX_TARGET_BARS)}

    def aux(name: str, fetch, interval_hours: int) -> None:
        try:
            bars = [b for b in fetch() if b[0] <= cutoff]
            if len(bars) >= 2:
                series[name] = _series(bars, interval_hours)
            else:
                logger.warning("dataset: %s came back empty", name)
        except Exception as e:  # noqa: BLE001
            logger.warning("dataset: dropping %s (%s)", name, e)

    aux("BTC_1d", lambda: coinbase_daily("BTC-USD", days), 24)
    aux("ETH_1d", lambda: binance_klines("ETHUSDT", "1d", 86400,
                                         days), 24)
    aux("XAU_1d", lambda: binance_klines("PAXGUSDT", "1d", 86400,
                                         days), 24)
    try:
        for name, bars in frankfurter_fx(days).items():
            bars = [b for b in bars if b[0] <= cutoff]
            if len(bars) >= 2:
                series[name] = _series(bars, 24)
    except Exception as e:  # noqa: BLE001
        logger.warning("dataset: dropping ECB forex series (%s)", e)

    dataset = harness.load_dataset(harness.dataset_text(
        {"version": 2, "target": "BTC_1h", "series": series}))
    prov = {
        "target": "BTC_1h",
        "last_close_utc": _utc(cutoff),
        "last_close": target[-1][1],
        "series": {name: {
            "bars": len(s["closes"]),
            "interval_hours": s["interval_hours"],
            "first_close_utc": _utc(s["ts"][0]),
            "source": {"BTC_1h": "binance BTCUSDT 1h",
                       "BTC_1d": "coinbase BTC-USD 1d",
                       "ETH_1d": "binance ETHUSDT 1d",
                       "XAU_1d": "binance PAXGUSDT 1d (tokenised gold)",
                       }.get(name, "ECB reference rate (frankfurter)")}
            for name, s in sorted(dataset["series"].items())},
    }
    return dataset, prov


class MarketData:
    """``build_dataset`` behind a short cache: every run opened within
    ``ttl`` seconds researches the same bars."""

    def __init__(self, n_hours: int = harness.MAX_TARGET_BARS,
                 days: int = TEN_YEARS_DAYS, ttl: float = 600.0):
        self.n_hours, self.days, self.ttl = n_hours, days, ttl
        self._lock = threading.Lock()
        self._at = 0.0
        self._dataset: Optional[dict] = None

    def __call__(self) -> dict:
        with self._lock:
            if self._dataset is None or time.time() - self._at > self.ttl:
                self._dataset, _prov = build_dataset(self.n_hours, self.days)
                self._at = time.time()
            return self._dataset
