# MIT License
# Copyright (c) 2024 MANTIS
"""Offline pieces for developing and testing LOOP pipelines.

* ``reference_source()`` — the reference pipeline (``pipelines/
  reference.py``), the thing to fork.
* ``StubInference`` — a free, deterministic stand-in for the model
  provider (Chutes / SayGM) that plays the reference pipeline's
  researcher and critic.
* ``synthetic_prices()`` / ``synthetic_dataset()`` — deterministic
  market data (an hourly BTC-like target plus daily auxiliary series).

Together with ``sealed.run_local`` they run a whole pipeline — sandbox,
broker, harness, dossier — with no gateway and no spend:

    from autoresearch.loop import (StubInference, reference_source,
                                   synthetic_dataset)
    from autoresearch.sealed import run_local
    out = run_local(reference_source(), synthetic_dataset(3000),
                    StubInference(), {"iterations": 6})
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

from . import harness
from .providers import InferenceClient


def reference_source() -> str:
    with open(os.path.join(os.path.dirname(__file__), "pipelines",
                           "reference.py"), encoding="utf-8") as f:
        return f.read()


def extract_json(text: Optional[str]) -> dict:
    """Parse the first JSON object out of an LLM reply."""
    if not text:
        raise ValueError("empty reply (token cap hit before any answer?)")
    text = text.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON in reply")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in reply")


def stub_proposal(i: int) -> dict:
    """The stub researcher's ``i``-th idea: a fixed walk over the
    vocabulary."""
    transforms = ["zscore_momentum", "vol_ratio", "rsi", "autocorr",
                  "range_pos"]
    lookbacks = [168, 96, 168, 48, 336]
    k = i % 5
    return {
        "feature": {"name": f"stub_{transforms[k]}_{lookbacks[k]}",
                    "transform": transforms[k],
                    "lookback_hours": lookbacks[k],
                    "rationale": "offline grid exploration"},
        "hyperparams": {"C": 1.0, "threshold": 0.52, "horizon_h": 36,
                        "regime": 1, "f": 0.05, "sl": 0.03, "tp1": 0.04,
                        "tp2": 0.08}}


def stub_verdict(metrics: dict) -> dict:
    """The stub critic: promote on trade-level evidence (what FLOW pays
    on); AUC is advisory — brackets can be +EV off a weak ranker."""
    ok = (metrics.get("t_stat", 0) > 0.5 and metrics.get("n_trades", 0) >= 5
          and metrics.get("avg_r", 0) > 0)
    return {"verdict": "promote" if ok else "prune",
            "notes": "offline stub heuristic"}


class StubInference(InferenceClient):
    """Plays the reference pipeline's two models, for free: a critic if
    the system prompt says so, otherwise the researcher."""

    COST = 0.003

    def __init__(self) -> None:
        super().__init__(api_key="stub")
        self.last_provider = "saygm"

    def list_models(self) -> list:
        return [
            {"id": "gpt-6-astra", "provider": "saygm",
             "context_length": 400_000,
             "pricing": {"prompt": "0.00000515",
                         "completion": "0.00002575"}},
            {"id": "claude-opus-5-5", "provider": "saygm",
             "context_length": 1_000_000,
             "pricing": {"prompt": "0.00000500",
                         "completion": "0.00002500"}},
            {"id": "kimi-k3.1-pro-tee", "provider": "saygm",
             "context_length": 256_000,
             "pricing": {"prompt": "0.00000136",
                         "completion": "0.00000680"}},
        ]

    def chat(self, body: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
        messages = body.get("messages") or [{}, {}]
        system = str(messages[0].get("content", ""))
        try:
            user = json.loads(messages[-1].get("content", ""))
        except (TypeError, ValueError):
            user = {}
        if "in-loop critic" in system:
            content = stub_verdict(user.get("metrics") or {})
        else:
            content = stub_proposal(int(user.get("iteration", 0)))
        self.last_provider = "saygm"
        return ({"id": "stub", "choices": [{
            "message": {"content": json.dumps(content)},
            "finish_reason": "stop"}],
            "usage": {"cost": self.COST, "total_tokens": 100}}, self.COST)


def synthetic_prices(n_hours: int = 4000, seed: int = 7,
                     s0: float = 60_000.0) -> List[float]:
    """Deterministic GBM-with-momentum series for offline testing."""
    import random
    rng = random.Random(seed)
    px, out, drift = s0, [], 0.0
    for _ in range(n_hours):
        drift = 0.97 * drift + 0.03 * rng.gauss(0, 0.004)
        px *= math.exp(drift + rng.gauss(0, 0.006))
        out.append(px)
    return out


def synthetic_dataset(n_hours: int = 4000, seed: int = 7) -> dict:
    """A deterministic stand-in for the canonical dataset: an hourly
    BTC-like target, its own daily closes, and two daily series that are
    (noisily) *leading* and unrelated respectively — so cross-market
    features have something real to find offline."""
    import random
    btc = synthetic_prices(n_hours, seed)
    ds = harness.single_series(btc)
    hourly = ds["series"]["BTC_1h"]
    daily_ts = hourly["ts"][23::24]
    ds["series"]["BTC_1d"] = {"interval_hours": 24, "ts": list(daily_ts),
                              "closes": hourly["closes"][23::24]}
    rng = random.Random(seed + 1)
    lead, px = [], 1.10          # tomorrow's BTC daily return, with noise
    daily_px = hourly["closes"][23::24]
    for i in range(len(daily_ts)):
        nxt = (math.log(daily_px[i + 1] / daily_px[i])
               if i + 1 < len(daily_px) else 0.0)
        px *= math.exp(0.5 * nxt + rng.gauss(0, 0.004))
        lead.append(px)
    ds["series"]["EURUSD_1d"] = {"interval_hours": 24, "ts": list(daily_ts),
                                 "closes": lead}
    rng = random.Random(seed + 2)
    noise, px = [], 2_000.0
    for _ in daily_ts:
        px *= math.exp(rng.gauss(0, 0.009))
        noise.append(px)
    ds["series"]["XAU_1d"] = {"interval_hours": 24, "ts": list(daily_ts),
                              "closes": noise}
    return ds
