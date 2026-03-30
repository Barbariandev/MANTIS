# MIT License
# Copyright (c) 2024 MANTIS

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

import config as _cfg

logger = logging.getLogger(__name__)

_SAMPLE_EVERY = int(getattr(_cfg, "SAMPLE_EVERY", 5))
RANGE_LOOKBACK_BLOCKS = 28800  # 4 days of blocks
BARRIER_PCT = 10.0
MIN_RANGE_PCT = 1.0
MAX_PENDING_BLOCKS = 43200


@dataclass
class PendingBreakoutSample:
    trigger_sidx: int
    trigger_block: int
    trigger_price: float
    direction: int
    range_high: float
    range_low: float
    continuation_barrier: float
    reversal_barrier: float
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class CompletedBreakoutSample:
    trigger_sidx: int
    trigger_block: int
    resolution_block: int
    direction: int
    label: int
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)


class RangeBreakoutTracker:
    def __init__(
        self,
        ticker: str,
        range_lookback_blocks: int = RANGE_LOOKBACK_BLOCKS,
        barrier_pct: float = BARRIER_PCT,
        min_range_pct: float = MIN_RANGE_PCT,
        max_pending_blocks: int = MAX_PENDING_BLOCKS,
    ):
        self.ticker = ticker
        self.range_lookback_blocks = range_lookback_blocks
        self.barrier_pct = barrier_pct
        self.min_range_pct = min_range_pct
        self.max_pending_blocks = max_pending_blocks

        self.pending_high: PendingBreakoutSample | None = None
        self.pending_low: PendingBreakoutSample | None = None
        self.completed: List[CompletedBreakoutSample] = []
        self._price_history: List[Tuple[int, float]] = []

    def _get_range(self, current_sidx: int) -> Tuple[float, float] | None:
        lookback_sidx = self.range_lookback_blocks // _SAMPLE_EVERY
        window_prices = [
            p for sidx, p in self._price_history
            if current_sidx - lookback_sidx <= sidx < current_sidx
        ]

        if len(window_prices) < lookback_sidx // 2:
            return None

        return min(window_prices), max(window_prices)

    def update_price(self, sidx: int, price: float):
        if price <= 0 or not np.isfinite(price):
            return
        self._price_history.append((sidx, price))

        max_sidx_age = 2 * self.range_lookback_blocks // _SAMPLE_EVERY
        self._price_history = [
            (s, p) for s, p in self._price_history
            if sidx - s <= max_sidx_age
        ]

    def check_trigger(
        self,
        sidx: int,
        block: int,
        price: float,
        embeddings: Dict[str, np.ndarray],
    ) -> PendingBreakoutSample | None:
        range_result = self._get_range(sidx)
        if range_result is None:
            return None

        range_low, range_high = range_result
        range_width = range_high - range_low

        if range_width < price * self.min_range_pct / 100:
            return None

        barrier_dist = range_width * self.barrier_pct / 100

        triggered_sample = None

        if price > range_high and self.pending_high is None:
            continuation_barrier = price + barrier_dist
            reversal_barrier = price - barrier_dist
            
            self.pending_high = PendingBreakoutSample(
                trigger_sidx=sidx,
                trigger_block=block,
                trigger_price=price,
                direction=1,
                range_high=range_high,
                range_low=range_low,
                continuation_barrier=continuation_barrier,
                reversal_barrier=reversal_barrier,
                embeddings=dict(embeddings),
            )
            triggered_sample = self.pending_high
            logger.info(
                f"[{self.ticker}] New high breakout triggered at sidx={sidx}, "
                f"price={price:.2f}, range=[{range_low:.2f}, {range_high:.2f}], "
                f"barriers=[{reversal_barrier:.2f}, {continuation_barrier:.2f}]"
            )

        elif price < range_low and self.pending_low is None:
            continuation_barrier = price - barrier_dist
            reversal_barrier = price + barrier_dist
            
            self.pending_low = PendingBreakoutSample(
                trigger_sidx=sidx,
                trigger_block=block,
                trigger_price=price,
                direction=-1,
                range_high=range_high,
                range_low=range_low,
                continuation_barrier=continuation_barrier,
                reversal_barrier=reversal_barrier,
                embeddings=dict(embeddings),
            )
            triggered_sample = self.pending_low
            logger.info(
                f"[{self.ticker}] New low breakout triggered at sidx={sidx}, "
                f"price={price:.2f}, range=[{range_low:.2f}, {range_high:.2f}], "
                f"barriers=[{continuation_barrier:.2f}, {reversal_barrier:.2f}]"
            )

        return triggered_sample

    def _resolve_pending(
        self, sample: PendingBreakoutSample | None, tag: str,
        current_block: int, current_price: float,
        cont_cond: bool, rev_cond: bool,
    ) -> Tuple[PendingBreakoutSample | None, CompletedBreakoutSample | None]:
        if sample is None:
            return None, None
        if current_block - sample.trigger_block > self.max_pending_blocks:
            logger.info(f"[{self.ticker}] Discarding stale {tag} breakout from sidx={sample.trigger_sidx}")
            return None, None
        label = 1 if cont_cond else (0 if rev_cond else -1)
        if label < 0:
            return sample, None
        completed = CompletedBreakoutSample(
            trigger_sidx=sample.trigger_sidx, trigger_block=sample.trigger_block,
            resolution_block=current_block, direction=sample.direction,
            label=label, embeddings=sample.embeddings,
        )
        self.completed.append(completed)
        outcome = "CONTINUED" if label == 1 else "REVERSED"
        logger.info(f"[{self.ticker}] {tag} breakout {outcome}: sidx={sample.trigger_sidx}, block={current_block}, price={current_price:.2f}")
        return None, completed

    def check_resolutions(self, current_block: int, current_price: float) -> List[CompletedBreakoutSample]:
        if current_price <= 0 or not np.isfinite(current_price):
            return []
        newly_completed = []
        self.pending_high, c = self._resolve_pending(
            self.pending_high, "High", current_block, current_price,
            current_price >= (self.pending_high.continuation_barrier if self.pending_high else 0),
            current_price <= (self.pending_high.reversal_barrier if self.pending_high else 0),
        )
        if c:
            newly_completed.append(c)
        self.pending_low, c = self._resolve_pending(
            self.pending_low, "Low", current_block, current_price,
            current_price <= (self.pending_low.continuation_barrier if self.pending_low else 0),
            current_price >= (self.pending_low.reversal_barrier if self.pending_low else 0),
        )
        if c:
            newly_completed.append(c)
        return newly_completed

    def to_dict(self) -> dict:
        def _sample_to_dict(s: PendingBreakoutSample | None) -> dict | None:
            if s is None:
                return None
            return {
                "trigger_sidx": s.trigger_sidx,
                "trigger_block": s.trigger_block,
                "trigger_price": s.trigger_price,
                "direction": s.direction,
                "range_high": s.range_high,
                "range_low": s.range_low,
                "continuation_barrier": s.continuation_barrier,
                "reversal_barrier": s.reversal_barrier,
                "embeddings": {k: v.tolist() for k, v in s.embeddings.items()},
            }

        def _completed_to_dict(s: CompletedBreakoutSample) -> dict:
            return {
                "trigger_sidx": s.trigger_sidx,
                "trigger_block": s.trigger_block,
                "resolution_block": s.resolution_block,
                "direction": s.direction,
                "label": s.label,
                "embeddings": {k: v.tolist() for k, v in s.embeddings.items()},
            }

        return {
            "ticker": self.ticker,
            "range_lookback_blocks": self.range_lookback_blocks,
            "barrier_pct": self.barrier_pct,
            "min_range_pct": self.min_range_pct,
            "max_pending_blocks": self.max_pending_blocks,
            "pending_high": _sample_to_dict(self.pending_high),
            "pending_low": _sample_to_dict(self.pending_low),
            "completed": [_completed_to_dict(s) for s in self.completed],
            "price_history": self._price_history[-(2 * self.range_lookback_blocks // _SAMPLE_EVERY):],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RangeBreakoutTracker":
        tracker = cls(
            ticker=data["ticker"],
            range_lookback_blocks=data.get("range_lookback_blocks", 28800),
            barrier_pct=data.get("barrier_pct", 25.0),
            min_range_pct=data.get("min_range_pct", 1.0),
            max_pending_blocks=data.get("max_pending_blocks", 43200),
        )

        def _dict_to_pending(d: dict | None) -> PendingBreakoutSample | None:
            if d is None:
                return None
            return PendingBreakoutSample(
                trigger_sidx=d["trigger_sidx"],
                trigger_block=d["trigger_block"],
                trigger_price=d["trigger_price"],
                direction=d["direction"],
                range_high=d["range_high"],
                range_low=d["range_low"],
                continuation_barrier=d["continuation_barrier"],
                reversal_barrier=d["reversal_barrier"],
                embeddings={k: np.array(v, dtype=np.float16) for k, v in d["embeddings"].items()},
            )

        def _dict_to_completed(d: dict) -> CompletedBreakoutSample:
            return CompletedBreakoutSample(
                trigger_sidx=d["trigger_sidx"],
                trigger_block=d["trigger_block"],
                resolution_block=d["resolution_block"],
                direction=d["direction"],
                label=d["label"],
                embeddings={k: np.array(v, dtype=np.float16) for k, v in d["embeddings"].items()},
            )

        tracker.pending_high = _dict_to_pending(data.get("pending_high"))
        tracker.pending_low = _dict_to_pending(data.get("pending_low"))
        tracker.completed = [_dict_to_completed(d) for d in data.get("completed", [])]
        tracker._price_history = data.get("price_history", [])

        return tracker


def _compute_multi_breakout_salience_old(
    completed_samples: List[CompletedBreakoutSample],
    min_n: int = 15,
    min_std: float = 0.03,
    eta: float = 0.5,
    corr_min: int = 20,
    **_,
) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    if len(completed_samples) < 50:
        return {}
    me, my = {}, {}
    for s in completed_samples:
        for hk, v in s.embeddings.items():
            a = np.asarray(v, dtype=np.float32)
            if a.shape == (2,):
                me.setdefault(hk, []).append(float(a[0]))
                my.setdefault(hk, []).append(s.label)
    auc, ns = {}, {}
    for hk in me:
        e, y = np.array(me[hk]), np.array(my[hk])
        if len(e) < min_n or e.std() < min_std or len(np.unique(y)) < 2:
            continue
        a = roc_auc_score(y, e)
        if a > 0.5:
            auc[hk], ns[hk] = a, len(e)
    if not auc:
        return {}
    hks = sorted(auc)
    N = len(hks)
    hi = {h: i for i, h in enumerate(hks)}
    mat = np.full((len(completed_samples), N), np.nan)
    for t, s in enumerate(completed_samples):
        for hk, v in s.embeddings.items():
            if hk in hi:
                a = np.asarray(v, dtype=np.float32)
                if a.shape == (2,):
                    mat[t, hi[hk]] = a[0]
    mr = mat.copy()
    labels = np.array([s.label for s in completed_samples], dtype=int)
    for lb in range(2):
        mk = labels == lb
        mr[mk] -= np.nanmean(mat[mk], axis=0)
    uniq = np.ones(N)
    for i in range(N):
        ci, mi = mr[:, i], ~np.isnan(mr[:, i])
        tc = 0.0
        for j in range(N):
            if j == i:
                continue
            m = mi & ~np.isnan(mr[:, j])
            if m.sum() < corr_min:
                continue
            a, b = ci[m], mr[:, j][m]
            if a.std() < 1e-8 or b.std() < 1e-8:
                tc += 1.0
            else:
                r = np.corrcoef(a, b)[0, 1]
                tc += abs(float(r)) if np.isfinite(r) else 1.0
        uniq[i] = 1.0 / (1.0 + tc)
    lw = np.array([eta * (auc[h] - 0.5) * ns[h] for h in hks])
    lw -= lw.max()
    w = np.exp(lw) * uniq
    w /= w.sum()
    return {hks[i]: float(w[i]) for i in range(N) if w[i] > 1e-6}


def _assign_episodes(samples: List[CompletedBreakoutSample],
                     gap_sidxs: int = 1440) -> np.ndarray:
    """Group samples into temporal episodes.  Samples whose trigger_sidx
    differ by <= *gap_sidxs* belong to the same episode.  Returns an
    int array of episode IDs parallel to *samples*."""
    n = len(samples)
    sidxs = np.array([s.trigger_sidx for s in samples], dtype=np.int64)
    order = np.argsort(sidxs, kind='stable')
    ep_ids = np.empty(n, dtype=np.int32)
    ep = 0
    ep_ids[order[0]] = 0
    for i in range(1, n):
        if sidxs[order[i]] - sidxs[order[i - 1]] > gap_sidxs:
            ep += 1
        ep_ids[order[i]] = ep
    return ep_ids


def compute_multi_breakout_salience(
    completed_samples: List[CompletedBreakoutSample],
    min_episodes: int = 5,
    min_std: float = 0.03,
    min_auc: float = 0.50,
    meta_C: float = 0.01,
    episode_gap_sidxs: int = 1440,
    **_,
) -> Dict[str, float]:
    """Compute salience using the same pattern as salience_binary_prediction:
    L2 logistic on z-scored miner predictions, |coef| as importance.
    Adapted for small samples with episode-based weighting."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if len(completed_samples) < 50:
        return {}

    ep_ids = _assign_episodes(completed_samples, episode_gap_sidxs)
    n_episodes = int(ep_ids.max()) + 1

    me: Dict[str, list] = {}
    my: Dict[str, list] = {}
    me_ep: Dict[str, list] = {}
    for i, s in enumerate(completed_samples):
        for hk, v in s.embeddings.items():
            a = np.asarray(v, dtype=np.float32)
            if a.shape == (2,):
                me.setdefault(hk, []).append(float(a[0]))
                my.setdefault(hk, []).append(s.label)
                me_ep.setdefault(hk, []).append(int(ep_ids[i]))

    qualified: Dict[str, float] = {}
    for hk in me:
        e, y = np.array(me[hk]), np.array(my[hk])
        hk_episodes = len(set(me_ep[hk]))
        if hk_episodes < min_episodes or e.std() < min_std or len(np.unique(y)) < 2:
            continue
        a = roc_auc_score(y, e)
        if a > min_auc:
            qualified[hk] = a
    if not qualified:
        return {}

    hks = sorted(qualified)
    N = len(hks)
    hi = {h: i for i, h in enumerate(hks)}

    T = len(completed_samples)
    mat = np.full((T, N), np.nan)
    for t, s in enumerate(completed_samples):
        for hk, v in s.embeddings.items():
            if hk in hi:
                a = np.asarray(v, dtype=np.float32)
                if a.shape == (2,):
                    mat[t, hi[hk]] = a[0]

    labels = np.array([s.label for s in completed_samples], dtype=int)

    col_mu = np.nanmean(mat, axis=0)
    col_std = np.nanstd(mat, axis=0)
    col_std[col_std < 1e-8] = 1.0
    zmat = (mat - col_mu) / col_std
    zmat = np.nan_to_num(zmat, nan=0.0)

    if len(np.unique(labels)) < 2 or n_episodes < min_episodes:
        return {}

    ep_weights = np.zeros(T, dtype=np.float64)
    for ep in range(n_episodes):
        mask = ep_ids == ep
        cnt = mask.sum()
        if cnt > 0:
            ep_weights[mask] = 1.0 / cnt
    ep_weights /= ep_weights.sum()
    ep_weights *= T

    clf = LogisticRegression(
        penalty="l2",
        C=meta_C,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(zmat, labels, sample_weight=ep_weights)

    w = np.abs(clf.coef_.ravel())
    if w.sum() < 1e-12:
        return {}
    w /= w.sum()
    return {hks[i]: float(w[i]) for i in range(N) if w[i] > 1e-6}

