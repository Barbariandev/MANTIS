# MIT License
# Copyright (c) 2024 MANTIS

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

RANGE_LOOKBACK_BLOCKS = 7200
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
        lookback_sidx = self.range_lookback_blocks // 5
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
        
        max_sidx_age = 2 * self.range_lookback_blocks // 5
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

    def check_resolutions(self, current_block: int, current_price: float) -> List[CompletedBreakoutSample]:
        if current_price <= 0 or not np.isfinite(current_price):
            return []

        newly_completed = []

        if self.pending_high is not None:
            sample = self.pending_high
            if current_block - sample.trigger_block > self.max_pending_blocks:
                logger.info(f"[{self.ticker}] Discarding stale high breakout from sidx={sample.trigger_sidx}")
                self.pending_high = None
            elif current_price >= sample.continuation_barrier:
                completed = CompletedBreakoutSample(
                    trigger_sidx=sample.trigger_sidx,
                    trigger_block=sample.trigger_block,
                    resolution_block=current_block,
                    direction=sample.direction,
                    label=1,
                    embeddings=sample.embeddings,
                )
                self.completed.append(completed)
                newly_completed.append(completed)
                logger.info(
                    f"[{self.ticker}] High breakout CONTINUED: sidx={sample.trigger_sidx}, "
                    f"resolved at block={current_block}, price={current_price:.2f}"
                )
                self.pending_high = None
            elif current_price <= sample.reversal_barrier:
                completed = CompletedBreakoutSample(
                    trigger_sidx=sample.trigger_sidx,
                    trigger_block=sample.trigger_block,
                    resolution_block=current_block,
                    direction=sample.direction,
                    label=0,
                    embeddings=sample.embeddings,
                )
                self.completed.append(completed)
                newly_completed.append(completed)
                logger.info(
                    f"[{self.ticker}] High breakout REVERSED: sidx={sample.trigger_sidx}, "
                    f"resolved at block={current_block}, price={current_price:.2f}"
                )
                self.pending_high = None

        if self.pending_low is not None:
            sample = self.pending_low
            if current_block - sample.trigger_block > self.max_pending_blocks:
                logger.info(f"[{self.ticker}] Discarding stale low breakout from sidx={sample.trigger_sidx}")
                self.pending_low = None
            elif current_price <= sample.continuation_barrier:
                completed = CompletedBreakoutSample(
                    trigger_sidx=sample.trigger_sidx,
                    trigger_block=sample.trigger_block,
                    resolution_block=current_block,
                    direction=sample.direction,
                    label=1,
                    embeddings=sample.embeddings,
                )
                self.completed.append(completed)
                newly_completed.append(completed)
                logger.info(
                    f"[{self.ticker}] Low breakout CONTINUED: sidx={sample.trigger_sidx}, "
                    f"resolved at block={current_block}, price={current_price:.2f}"
                )
                self.pending_low = None
            elif current_price >= sample.reversal_barrier:
                completed = CompletedBreakoutSample(
                    trigger_sidx=sample.trigger_sidx,
                    trigger_block=sample.trigger_block,
                    resolution_block=current_block,
                    direction=sample.direction,
                    label=0,
                    embeddings=sample.embeddings,
                )
                self.completed.append(completed)
                newly_completed.append(completed)
                logger.info(
                    f"[{self.ticker}] Low breakout REVERSED: sidx={sample.trigger_sidx}, "
                    f"resolved at block={current_block}, price={current_price:.2f}"
                )
                self.pending_low = None

        return newly_completed

    def get_completed_samples(self) -> List[CompletedBreakoutSample]:
        return list(self.completed)

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
            "pending_high": _sample_to_dict(self.pending_high),
            "pending_low": _sample_to_dict(self.pending_low),
            "completed": [_completed_to_dict(s) for s in self.completed],
            "price_history": self._price_history[-1000:],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RangeBreakoutTracker":
        tracker = cls(ticker=data["ticker"])

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


def compute_breakout_salience(
    completed_samples: List[CompletedBreakoutSample],
    recent_samples: int = 500,
    recent_mass: float = 0.5,
) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from utils import recent_mass_weights

    if len(completed_samples) < 50:
        logger.info(f"Not enough breakout samples for salience: {len(completed_samples)}")
        return {}

    all_hks = set()
    for sample in completed_samples:
        all_hks.update(sample.embeddings.keys())
    
    if not all_hks:
        return {}

    all_hks = sorted(all_hks)
    hk2idx = {hk: i for i, hk in enumerate(all_hks)}
    H = len(all_hks)

    first_emb = next(iter(completed_samples[0].embeddings.values()), None)
    if first_emb is None:
        return {}
    D = len(first_emb)

    X_list = []
    y_list = []
    resolution_blocks = []

    for sample in completed_samples:
        row = np.zeros((H, D), dtype=np.float32)
        for hk, vec in sample.embeddings.items():
            idx = hk2idx.get(hk)
            if idx is not None:
                arr = np.asarray(vec, dtype=np.float32)
                if arr.shape == (D,):
                    row[idx] = arr
        
        X_list.append(row.flatten())
        y_list.append(sample.label)
        resolution_blocks.append(sample.resolution_block)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    resolution_blocks = np.array(resolution_blocks, dtype=np.float64)

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        logger.info("Only one class in breakout samples, cannot compute salience")
        return {}

    sw = recent_mass_weights(resolution_blocks, recent_samples=recent_samples, recent_mass=recent_mass)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=500,
        tol=1e-4,
    )
    clf.fit(X, y, sample_weight=sw)

    coef = np.abs(clf.coef_.ravel())
    coef_reshaped = coef.reshape(H, D)
    importance = coef_reshaped.sum(axis=1)

    total = float(importance.sum())
    if total <= 0:
        return {}

    salience = {}
    for hk, idx in hk2idx.items():
        score = float(importance[idx] / total)
        if score > 0:
            salience[hk] = score

    return salience


def compute_multi_breakout_salience(
    completed_samples: List[CompletedBreakoutSample],
    gate_top_pct: float = 0.10,
    recent_days: int = 20,
    blocks_per_day: int = 7200,
    recent_mass: float = 0.5,
) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from utils import recent_mass_weights

    recent_blocks = recent_days * blocks_per_day

    if len(completed_samples) < 100:
        logger.info(f"Not enough multi-breakout samples for salience: {len(completed_samples)}")
        return {}

    all_hks = set()
    for sample in completed_samples:
        all_hks.update(sample.embeddings.keys())
    
    if not all_hks:
        return {}

    all_hks = sorted(all_hks)
    H = len(all_hks)
    D = 2

    resolution_blocks = np.array([s.resolution_block for s in completed_samples], dtype=np.float64)
    sample_weights = recent_mass_weights(
        resolution_blocks, 
        recent_samples=recent_blocks, 
        recent_mass=recent_mass
    )
    
    weighted_correct = {hk: 0.0 for hk in all_hks}
    weighted_total = {hk: 0.0 for hk in all_hks}
    
    for i, sample in enumerate(completed_samples):
        w = sample_weights[i]
        for hk, vec in sample.embeddings.items():
            if hk not in weighted_correct:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape != (D,):
                continue
            pred_label = 1 if arr[0] > 0.5 else 0
            weighted_correct[hk] += w * int(pred_label == sample.label)
            weighted_total[hk] += w
    
    min_weight = max(20.0, len(completed_samples) / 10.0)
    empirical_acc = {}
    for hk in all_hks:
        if weighted_total[hk] >= min_weight:
            empirical_acc[hk] = weighted_correct[hk] / weighted_total[hk]
    
    if not empirical_acc:
        logger.info("No miners with enough samples for empirical accuracy")
        return {}
    
    sorted_acc = sorted(empirical_acc.values(), reverse=True)
    n_keep = max(5, int(len(sorted_acc) * gate_top_pct))
    threshold = sorted_acc[min(n_keep - 1, len(sorted_acc) - 1)]
    
    gated_hks = sorted([hk for hk, acc in empirical_acc.items() if acc >= threshold])
    
    if len(gated_hks) < 5:
        logger.info(f"Too few miners passed gate: {len(gated_hks)}")
        sorted_miners = sorted(empirical_acc.items(), key=lambda x: -x[1])[:20]
        raw_weights = {hk: max(0, acc - 0.5) for hk, acc in sorted_miners}
        total_w = sum(raw_weights.values())
        if total_w > 0:
            return {hk: w / total_w for hk, w in raw_weights.items() if w > 0}
        return {}
    
    logger.info(f"Gated to {len(gated_hks)} miners (top {gate_top_pct*100:.0f}% by empirical accuracy)")
    
    hk2idx = {hk: i for i, hk in enumerate(gated_hks)}
    G = len(gated_hks)
    
    X_list = []
    y_list = []
    resolution_block_list = []

    for sample in completed_samples:
        row = np.zeros(G, dtype=np.float32)
        has_any = False
        for hk, vec in sample.embeddings.items():
            idx = hk2idx.get(hk)
            if idx is not None:
                arr = np.asarray(vec, dtype=np.float32)
                if arr.shape == (D,):
                    row[idx] = arr[0]
                    has_any = True
        
        if has_any:
            X_list.append(row)
            y_list.append(sample.label)
            resolution_block_list.append(sample.resolution_block)

    if len(X_list) < 50:
        logger.info(f"Not enough samples after gating: {len(X_list)}")
        return {}

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    resolution_blocks_filtered = np.array(resolution_block_list, dtype=np.float64)

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        logger.info("Only one class in gated samples")
        return {}

    sw = recent_mass_weights(resolution_blocks_filtered, recent_samples=recent_blocks, recent_mass=recent_mass)

    clf = LogisticRegression(
        penalty="l1",
        C=0.5,
        solver="saga",
        class_weight="balanced",
        max_iter=1000,
        tol=1e-4,
    )
    clf.fit(X, y, sample_weight=sw)

    importance = np.abs(clf.coef_.ravel())

    total_imp = float(importance.sum())
    if total_imp <= 0:
        raw_weights = {hk: max(0, empirical_acc[hk] - 0.5) for hk in gated_hks}
        total_w = sum(raw_weights.values())
        if total_w > 0:
            return {hk: w / total_w for hk, w in raw_weights.items() if w > 0}
        return {}

    salience = {}
    for hk, idx in hk2idx.items():
        score = float(importance[idx] / total_imp)
        if score > 0:
            salience[hk] = score

    logger.info(
        f"Gated multi-breakout salience: {len(completed_samples)} samples, "
        f"{len(gated_hks)} gated miners, {len(salience)} with non-zero weight"
    )
    return salience

