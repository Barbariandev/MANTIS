# MIT License
# Copyright (c) 2024 MANTIS
"""Data contracts for the LOOP autoresearch challenge.

Three artifact families, all JSON with canonical hashing:

* **ResearchEvent** — one step of a pipeline's run, written by the
  gateway as it happens (a pipeline cannot write its own trace).  Events
  form a tree (``parent_seq``), which is what the research-tree diagrams
  and the network progress visuals are rendered from.
* **UsageReceipt** — an HMAC-signed record of metered spend (model call,
  experiment, or the pipeline's own running time).
* **Dossier** — what a sealed run produced: the model, how it was
  validated, which pipeline found it and on which data.  The gateway
  assembles it; the miner publishes it next to its payload; the
  validator audit agent scores it against the published ruleset and the
  resulting multiplier gates the FLOW-criteria trade score.

Everything here is dependency-free (stdlib only) so miners can vendor it.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2

# ---------------------------------------------------------------- events

# The event vocabulary is fixed so network-level visuals can aggregate
# across miners.  `data` is free-form (size-capped by the gateway).
EVENT_KINDS = (
    "run.started",          # {budget_credits, pipeline_sha256, dataset_sha256, args}
    "llm.call",             # {step, candidate_id, model, receipt_id, credits}
    "hypothesis.proposed",  # {candidate_id, feature, hyperparams, rationale, refines, receipt_id}
    "experiment.result",    # {candidate_id, metrics, receipt_id}
    "critique",             # {candidate_id, verdict, notes, receipt_id}
    "branch.pruned",        # {candidate_id, reason}
    "model.promoted",       # {candidate_id, metrics}
    "holdout.result",       # {candidate_id, metrics}
    "trade.intent",         # {candidate_id, regime, d, f, sl, tp1, tp2, h, trade_id}
    "dossier.submitted",    # {dossier_hash}
    "run.ended",            # {status, reason, seconds}
)


@dataclass
class ResearchEvent:
    run_id: str
    seq: int
    kind: str
    ts: float = field(default_factory=lambda: time.time())
    parent_seq: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchEvent":
        return cls(
            run_id=str(d["run_id"]), seq=int(d["seq"]), kind=str(d["kind"]),
            ts=float(d.get("ts", 0.0)),
            parent_seq=(None if d.get("parent_seq") is None
                        else int(d["parent_seq"])),
            data=dict(d.get("data") or {}),
        )


# -------------------------------------------------------------- receipts

def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class UsageReceipt:
    """One metered debit, signed by the gateway (HMAC-SHA256).

    ``kind`` is "llm" or "compute"; ``item`` is the SayGM model slug or
    the compute SKU; ``credits`` is the debit in credits (1 credit =
    1 USD of metered spend); ``request_hash`` binds the receipt to the
    request body so a receipt cannot be re-pointed at different work.
    """
    receipt_id: str
    hotkey: str
    run_id: str
    kind: str                   # "llm" | "compute"
    provider: str               # "saygm" | "runpod" | "local"
    item: str
    credits: float
    ts: float
    request_hash: str
    meta: Dict[str, Any] = field(default_factory=dict)
    sig: str = ""

    def signing_payload(self) -> str:
        d = asdict(self)
        d.pop("sig", None)
        return canonical_json(d)

    def sign(self, secret: str) -> "UsageReceipt":
        self.sig = _hmac.new(secret.encode(), self.signing_payload().encode(),
                             hashlib.sha256).hexdigest()
        return self

    def verify(self, secret: str) -> bool:
        if not self.sig:
            return False
        want = _hmac.new(secret.encode(), self.signing_payload().encode(),
                         hashlib.sha256).hexdigest()
        return _hmac.compare_digest(want, self.sig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UsageReceipt":
        return cls(
            receipt_id=str(d["receipt_id"]), hotkey=str(d["hotkey"]),
            run_id=str(d["run_id"]), kind=str(d["kind"]),
            provider=str(d["provider"]), item=str(d["item"]),
            credits=float(d["credits"]), ts=float(d["ts"]),
            request_hash=str(d["request_hash"]),
            meta=dict(d.get("meta") or {}), sig=str(d.get("sig", "")),
        )


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


# -------------------------------------------------------------- dossier

# What a feature may be — which transforms, which lookbacks — is the
# harness's vocabulary (``harness.TRANSFORMS``, ``harness.clean_spec``),
# enforced when a pipeline proposes a candidate.  A feature reads the
# closes of one of the dataset's series (``inputs`` names it).


@dataclass
class FeatureSpec:
    name: str
    inputs: List[str]           # what it reads (["price"])
    lookback_hours: float       # max history the feature reads
    transform: str              # one of harness.TRANSFORMS
    rationale: str = ""         # the pipeline's reason it should generalize
    code_hash: str = ""         # unused: models are specs, not code

    def to_dict(self) -> dict:
        d = asdict(self)
        # normalize numerics so a JSON round-trip is hash-stable
        d["lookback_hours"] = float(self.lookback_hours)
        d["inputs"] = [str(x) for x in self.inputs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(
            name=str(d["name"]), inputs=[str(x) for x in d.get("inputs", [])],
            lookback_hours=float(d.get("lookback_hours", 0.0)),
            transform=str(d.get("transform", "")),
            rationale=str(d.get("rationale", "")),
            code_hash=str(d.get("code_hash", "")),
        )


@dataclass
class Dossier:
    """The research artifact the validator audit agent reads.

    Assembled by the gateway at the end of a sealed run — a miner
    cannot write one — and published by the miner to its R2 bucket at
    object key ``<hotkey>.research`` (plaintext JSON — research is public
    evidence, trades stay timelocked in the normal payload).  A validator
    accepts it only if it is byte-for-byte the dossier the gateway holds
    for that run.  The audit multiplier is keyed by ``dossier_hash()`` and
    is sticky until the hash changes, so validators only pay LLM audit
    cost when a miner actually revises its research.
    """
    version: int
    hotkey: str
    created_ts: float
    model_name: str
    model_code_hash: str        # sha256 of the harness that scored it
    seed: int
    features: List[FeatureSpec]
    hyperparameters: Dict[str, Any]
    hyperparameter_rationale: str
    validation: Dict[str, Any]  # {method, n_folds, embargo_hours, metrics,
                                #  holdout}
    regimes_traded: List[int]   # subset of [0, 1, 2, 3]
    run_ids: List[str]          # gateway runs that produced this research
    events_digest: str          # sha256 over canonical event list
    receipts_digest: str        # sha256 over canonical receipt-id list
    narrative: str              # the case made to the validator agent
    pipeline_sha256: str = ""   # the published source that ran
    pipeline_args: Dict[str, Any] = field(default_factory=dict)
    dataset_sha256: str = ""    # the prices it ran on (holdout included)
    trade: Dict[str, Any] = field(default_factory=dict)  # the model's
                                # trade at the last bar of the dataset

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = [f.to_dict() if isinstance(f, FeatureSpec)
                         else FeatureSpec.from_dict(f).to_dict()
                         for f in self.features]
        # normalize numerics so a JSON round-trip is hash-stable
        d["version"] = int(self.version)
        d["created_ts"] = float(self.created_ts)
        d["seed"] = int(self.seed)
        d["regimes_traded"] = [int(r) for r in self.regimes_traded]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Dossier":
        return cls(
            version=int(d.get("version", SCHEMA_VERSION)),
            hotkey=str(d["hotkey"]),
            created_ts=float(d.get("created_ts", 0.0)),
            model_name=str(d.get("model_name", "")),
            model_code_hash=str(d.get("model_code_hash", "")),
            seed=int(d.get("seed", 0)),
            features=[FeatureSpec.from_dict(f) for f in d.get("features", [])],
            hyperparameters=dict(d.get("hyperparameters") or {}),
            hyperparameter_rationale=str(d.get("hyperparameter_rationale", "")),
            validation=dict(d.get("validation") or {}),
            regimes_traded=[int(r) for r in d.get("regimes_traded", [])],
            run_ids=[str(r) for r in d.get("run_ids", [])],
            events_digest=str(d.get("events_digest", "")),
            receipts_digest=str(d.get("receipts_digest", "")),
            narrative=str(d.get("narrative", "")),
            pipeline_sha256=str(d.get("pipeline_sha256", "")),
            pipeline_args=dict(d.get("pipeline_args") or {}),
            dataset_sha256=str(d.get("dataset_sha256", "")),
            trade=dict(d.get("trade") or {}),
        )

    def canonical(self) -> str:
        return canonical_json(self.to_dict())

    def dossier_hash(self) -> str:
        return sha256_hex(self.canonical())


def events_digest(events: List[Any]) -> str:
    """Digest of a run's trace (events or their dicts).  The closing
    ``dossier.submitted`` and ``run.ended`` events are left out: they
    are written after the dossier this digest goes into."""
    rows = [e.to_dict() if isinstance(e, ResearchEvent) else
            ResearchEvent.from_dict(e).to_dict() for e in events]
    rows = [r for r in rows
            if r["kind"] not in ("dossier.submitted", "run.ended")]
    return sha256_hex(canonical_json(
        sorted(rows, key=lambda r: (r["run_id"], r["seq"]))))


def receipts_digest(receipt_ids: List[str]) -> str:
    return sha256_hex(canonical_json(sorted(receipt_ids)))
