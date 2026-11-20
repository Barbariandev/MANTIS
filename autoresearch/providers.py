# MIT License
# Copyright (c) 2024 MANTIS
"""Spend targets for LOOP credits — SayGM (SN28) inference & compute.

Two families:

* **Inference** — ``InferenceClient``: Bittensor-native LLM routing through
  **SayGM** (SN28 Intel TDX TEE gateway, ``https://api.saygm.com/v1``,
  ``gm_live_...`` keys), covering GPT-6, Claude, Gemini, and Chutes
  confidential ``-tee`` open-weight models under one endpoint and one
  authoritative nano-USD settlement feed (``usage.cost``).  OpenRouter and
  other provider keys are refused.
* **Compute** — ``RunPodProvider`` (primary) and ``LocalComputeProvider``
  (backup / testing).  Compute is priced per metered second from a posted
  rate table.

Pre-flight quarantine (``quarantine_key``):
  Before a miner's BYOK ``gm_live_...`` key is allowed to fund a run,
  ``loop-keyd`` checks ``GET https://api.saygm.com/v1/credits`` and
  **refuses** keys with ``$0.00`` balance or balance above
  ``SAYGM_MAX_KEY_BALANCE_USD`` (default ``$100.00``), forcing miners onto
  dedicated low-funded burner accounts.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .slots import compute_prefix
from .submission import validate_inference_key_format

logger = logging.getLogger(__name__)

try:
    from config import (
        LOOP_SAYGM_MAX_BALANCE_USD as SAYGM_MAX_KEY_BALANCE_USD,
        SAYGM_BASE_URL,
    )
except ImportError:
    SAYGM_BASE_URL = "https://api.saygm.com/v1"
    SAYGM_MAX_KEY_BALANCE_USD = 100.0

_CATALOG_LOCK = threading.Lock()
_CATALOG_CACHE: Dict[str, Tuple[float, Dict[str, dict]]] = {}
_CATALOG_TTL_SECONDS = 900.0

_TEXT_API_SHAPES = frozenset({
    "chat.completions", "messages", "generateContent",
})


class ProviderError(Exception):
    pass


class ProviderUnavailable(ProviderError):
    pass


class KeyQuarantineError(ProviderError):
    """A miner-supplied inference key violated LOOP's SayGM burner-key
    quarantine rules and was refused before execution."""


def detect_provider(api_key: str) -> str:
    """Return ``'saygm'`` (or ``'stub'`` for offline tests) from
    ``api_key``; raises ``KeyQuarantineError`` on OpenRouter or any
    non-SayGM credential."""
    try:
        clean = validate_inference_key_format(api_key)
    except ValueError as e:
        raise KeyQuarantineError(str(e)) from None
    if clean == "stub":
        return "stub"
    if clean.startswith(("gm_live_", "gm_test_")):
        return "saygm"
    raise KeyQuarantineError("api_key must be a SayGM key (gm_live_...)")


def _saygm_root(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _require_https(url: str, label: str = "saygm_base_url") -> str:
    clean = str(url or "").strip().rstrip("/")
    if not clean.startswith("https://"):
        raise ProviderError(f"{label} must use https:// (got {clean!r})")
    return clean


def quarantine_key(api_key: str,
                   saygm_base_url: str = SAYGM_BASE_URL,
                   max_saygm_balance_usd: float = SAYGM_MAX_KEY_BALANCE_USD,
                   http: Any = None,
                   timeout: float = 20.0) -> Dict[str, Any]:
    """Enforce pre-flight balance-cap checks on a miner's SayGM key.
    Returns ``{'provider': 'saygm', 'balance_usd': float}`` when safe;
    raises ``KeyQuarantineError`` when invalid, empty, or over-funded, or
    ``ProviderError`` when SayGM is unreachable (fail closed)."""
    provider = detect_provider(api_key)
    if provider == "stub":
        return {"provider": "stub", "balance_usd": 1.0}
    base = _require_https(saygm_base_url)
    clean = api_key.strip()
    client = http if http is not None else httpx

    url = f"{base}/credits"
    try:
        r = client.get(
            url,
            headers={"Authorization": f"Bearer {clean}"},
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        raise ProviderError(
            f"SayGM quarantine probe unreachable: {type(e).__name__}"
        ) from None
    if r.status_code in (401, 403):
        raise KeyQuarantineError("SayGM key is invalid or unauthorized")
    if r.status_code >= 500 or r.status_code != 200:
        raise ProviderError(f"SayGM credits check HTTP {r.status_code}")
    try:
        payload = r.json()
        bal_obj = payload.get("balance")
        if isinstance(bal_obj, dict):
            if "usd" in bal_obj and bal_obj["usd"] is not None:
                bal = float(bal_obj["usd"])
            elif "nano_usd" in bal_obj and bal_obj["nano_usd"] is not None:
                bal = float(bal_obj["nano_usd"]) / 1e9
            else:
                raise ValueError("missing balance.usd")
        elif isinstance(bal_obj, (int, float, str)):
            bal = float(bal_obj)
        else:
            raise ValueError("missing balance")
    except Exception as e:  # noqa: BLE001
        raise ProviderError(f"SayGM credits parse error: {e}") from None

    if bal <= 0.0:
        raise KeyQuarantineError(
            "SayGM key has $0.00 spendable prepaid credit")
    if max_saygm_balance_usd > 0 and bal > max_saygm_balance_usd:
        raise KeyQuarantineError(
            f"SayGM key balance (${bal:.2f}) exceeds LOOP's "
            f"${max_saygm_balance_usd:g} burner-key safety ceiling; "
            f"use a dedicated burner account funded with <= "
            f"${max_saygm_balance_usd:g}")
    return {"provider": "saygm", "balance_usd": bal}


# --------------------------------------------------- multi-shape adapters

def _split_system_messages(messages: List[dict]) -> Tuple[str, List[dict]]:
    sys_parts: List[str] = []
    turns: List[dict] = []
    for m in messages:
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        if role == "system":
            if content:
                sys_parts.append(content)
        elif role in ("user", "assistant"):
            if turns and turns[-1]["role"] == role:
                turns[-1] = {
                    "role": role,
                    "content": turns[-1]["content"] + "\n\n" + content,
                }
            else:
                turns.append({"role": role, "content": content})
    if not turns:
        turns = [{"role": "user", "content": "\n\n".join(sys_parts) or "."}]
        sys_parts = []
    elif turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "Begin."})
    return "\n\n".join(sys_parts), turns


def _to_anthropic_body(model_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    system_text, turns = _split_system_messages(body.get("messages") or [])
    max_tok = int(body.get("max_tokens")
                  or body.get("max_completion_tokens") or 4096)
    out: Dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tok,
        "messages": turns,
    }
    if system_text:
        out["system"] = system_text
    for k in ("temperature", "top_p", "top_k"):
        if body.get(k) is not None:
            out[k] = body[k]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [str(s) for s in stop]
    return out


def _from_anthropic_response(model_id: str,
                             resp: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    blocks = resp.get("content") or []
    texts: List[str] = []
    if isinstance(blocks, list):
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text")
                if isinstance(t, str):
                    texts.append(t)
    stop_reason = resp.get("stop_reason") or "stop"
    finish = {"end_turn": "stop", "max_tokens": "length",
              "stop_sequence": "stop"}.get(stop_reason, stop_reason)
    usage = resp.get("usage") or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    cost = _extract_saygm_cost(usage, in_tok + out_tok)
    norm = {
        "id": resp.get("id") or "",
        "object": "chat.completion",
        "model": resp.get("model") or model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "".join(texts) if texts else None,
            },
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "cost": cost,
        },
    }
    return norm, cost


def _to_gemini_body(body: Dict[str, Any]) -> Dict[str, Any]:
    system_text, turns = _split_system_messages(body.get("messages") or [])
    contents = [
        {
            "role": "user" if t["role"] == "user" else "model",
            "parts": [{"text": t["content"]}],
        }
        for t in turns
    ]
    gen_cfg: Dict[str, Any] = {}
    max_tok = body.get("max_tokens") or body.get("max_completion_tokens")
    if max_tok is not None:
        gen_cfg["maxOutputTokens"] = int(max_tok)
    if body.get("temperature") is not None:
        gen_cfg["temperature"] = float(body["temperature"])
    if body.get("top_p") is not None:
        gen_cfg["topP"] = float(body["top_p"])
    if body.get("top_k") is not None:
        gen_cfg["topK"] = int(body["top_k"])
    if body.get("seed") is not None:
        gen_cfg["seed"] = int(body["seed"])
    stop = body.get("stop")
    if isinstance(stop, str):
        gen_cfg["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        gen_cfg["stopSequences"] = [str(s) for s in stop]
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_object":
        gen_cfg["responseMimeType"] = "application/json"
    out: Dict[str, Any] = {"contents": contents}
    if system_text:
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if gen_cfg:
        out["generationConfig"] = gen_cfg
    return out


def _from_gemini_response(model_id: str,
                          resp: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    cands = resp.get("candidates") or []
    first = cands[0] if cands and isinstance(cands[0], dict) else {}
    parts = ((first.get("content") or {}).get("parts") or [])
    texts: List[str] = []
    for p in parts:
        if isinstance(p, dict) and not p.get("thought"):
            t = p.get("text")
            if isinstance(t, str):
                texts.append(t)
    raw_reason = str(first.get("finishReason") or "STOP")
    finish = {"STOP": "stop", "MAX_TOKENS": "length"}.get(
        raw_reason, raw_reason.lower())
    meta = resp.get("usageMetadata") or {}
    in_tok = int(meta.get("promptTokenCount", 0) or 0)
    out_tok = int(meta.get("candidatesTokenCount", 0) or 0)
    tot_tok = int(meta.get("totalTokenCount", in_tok + out_tok) or 0)
    cost = _extract_saygm_cost(meta, tot_tok)
    norm = {
        "id": resp.get("responseId") or "",
        "object": "chat.completion",
        "model": resp.get("modelVersion") or model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "".join(texts) if texts else None,
            },
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": tot_tok,
            "cost": cost,
        },
    }
    return norm, cost


def _extract_saygm_cost(usage: Dict[str, Any], total_tokens: int,
                        fallback_per_mtok: float = 25.0) -> float:
    cost = usage.get("cost")
    if isinstance(cost, (int, float)) and cost >= 0:
        return float(cost)
    for nano_key in ("cost_nano_usd", "costNanoUsd", "cost_ndollars"):
        nano = usage.get(nano_key)
        if isinstance(nano, (int, float)) and nano >= 0:
            return float(nano) / 1e9
    est = float(total_tokens) / 1e6 * fallback_per_mtok
    logger.warning("SayGM response missing usage.cost; estimated %.6f USD "
                   "from %d tokens", est, total_tokens)
    return est


# --------------------------------------------------------------- inference

class InferenceClient:
    """Metered client over **SayGM** (Bittensor SN28 Intel TDX TEE gateway).

    Supports OpenAI Chat Completions, Anthropic ``/v1/messages`` (Claude),
    and Google Gemini ``:generateContent`` seamlessly behind ``chat()``,
    using SayGM's authoritative settled ``usage.cost`` (USD).
    """

    FALLBACK_USD_PER_MTOK = 25.0

    def __init__(self, api_key: Optional[str] = None,
                 saygm_base_url: str = SAYGM_BASE_URL,
                 timeout: float = 180.0,
                 http: Any = None):
        self.saygm_base_url = _require_https(saygm_base_url)
        self._timeout = timeout
        self._http = http if http is not None else httpx
        self.saygm_key = ""
        self._is_stub = False
        self.last_provider = "saygm"

        raw_key = (api_key if api_key is not None and api_key != ""
                   else os.environ.get("SAYGM_API_KEY", "").strip())
        if raw_key:
            prov = detect_provider(raw_key)
            if prov == "stub":
                self._is_stub = True
            else:
                self.saygm_key = raw_key.strip()

        self.api_key = self.saygm_key or ("stub" if self._is_stub else "")
        self._saygm_catalog: Optional[Dict[str, dict]] = None

    @property
    def provider_name(self) -> str:
        if self._is_stub:
            return "stub"
        return "saygm" if self.saygm_key else "unconfigured"

    def available(self) -> bool:
        return bool(self._is_stub or self.saygm_key)

    def _load_saygm_catalog(self) -> Dict[str, dict]:
        if self._saygm_catalog is not None:
            return self._saygm_catalog
        if self._http is httpx:
            with _CATALOG_LOCK:
                cached = _CATALOG_CACHE.get(self.saygm_base_url)
                if cached is not None and time.monotonic() < cached[0]:
                    self._saygm_catalog = cached[1]
                    return cached[1]
        r = self._http.get(f"{self.saygm_base_url}/models", timeout=30.0)
        r.raise_for_status()
        raw = r.json()
        items = raw.get("data", raw) if isinstance(raw, dict) else raw
        cat: Dict[str, dict] = {}
        for m in items or []:
            if not isinstance(m, dict) or not isinstance(m.get("id"), str):
                continue
            if not m.get("available", True) or m.get("coming_soon"):
                continue
            shapes = m.get("api_shapes") or []
            if not any(s in _TEXT_API_SHAPES for s in shapes):
                continue
            dims = ((m.get("pricing") or {}).get("dimensions") or {})
            in_nd = float(dims.get("input_per_mtok_ndollars") or 0.0)
            out_nd = float(dims.get("output_per_mtok_ndollars") or 0.0)
            ctx = int(m.get("context_length")
                      or dims.get("long_context_threshold_tokens")
                      or 128_000)
            cat[m["id"]] = {
                "id": m["id"],
                "provider": "saygm",
                "api_shapes": list(shapes),
                "context_length": ctx,
                "confidential_compute": bool(m.get("confidential")),
                "in_usd_per_mtok": in_nd / 1e9,
                "out_usd_per_mtok": out_nd / 1e9,
                "pricing": {
                    "prompt": f"{in_nd / 1e15:.12f}",
                    "completion": f"{out_nd / 1e15:.12f}",
                },
            }
        self._saygm_catalog = cat
        if self._http is httpx:
            with _CATALOG_LOCK:
                _CATALOG_CACHE[self.saygm_base_url] = (
                    time.monotonic() + _CATALOG_TTL_SECONDS,
                    cat,
                )
        return cat

    def list_models(self) -> list[dict]:
        """Return normalized SayGM text models (``id``, ``context_length``,
        ``pricing.{prompt, completion}`` in USD/token)."""
        cat = self._load_saygm_catalog()
        return [cat[k] for k in sorted(cat)]

    @staticmethod
    def _candidate_slugs(model: str) -> List[str]:
        cands = [model]
        bare = model.split("/", 1)[-1] if "/" in model else model
        if bare not in cands:
            cands.append(bare)
        if bare.startswith("claude-"):
            hyph = bare.replace(".", "-")
            if hyph not in cands:
                cands.append(hyph)
        for base in list(cands):
            if not base.lower().endswith("-tee"):
                cands.append(f"{base}-tee")
        return cands

    def _resolve_model(self, model: str) -> Tuple[str, Optional[dict]]:
        try:
            cat = self._load_saygm_catalog()
        except Exception:  # noqa: BLE001
            cat = self._saygm_catalog or {}
        if model in cat:
            return model, cat[model]
        lower_map = {k.lower(): (k, v) for k, v in cat.items()}
        for cand in self._candidate_slugs(model):
            hit = lower_map.get(cand.lower())
            if hit is not None:
                return hit
        bare = model.split("/", 1)[-1] if "/" in model else model
        if bare.startswith("claude-"):
            bare = bare.replace(".", "-")
        return bare, None

    def chat(self, body: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Execute a request on SayGM; returns
        ``(openai_shaped_response, cost_usd)``."""
        if not self.available():
            raise ProviderUnavailable("no SAYGM_API_KEY configured")
        model = str(body.get("model") or "").strip()
        if not model:
            raise ProviderError("missing model in chat request")
        model_id, meta = self._resolve_model(model)
        self.last_provider = "saygm"
        return self._chat_saygm(model_id, body, meta)

    def _chat_saygm(self, model_id: str, body: Dict[str, Any],
                    meta: Optional[dict]) -> Tuple[Dict[str, Any], float]:
        shapes = (meta or {}).get("api_shapes") or []
        if not shapes:
            low = model_id.lower()
            if low.startswith("claude-"):
                shapes = ["messages"]
            elif low.startswith("gemini-"):
                shapes = ["generateContent"]
            else:
                shapes = ["chat.completions"]

        root = _saygm_root(self.saygm_base_url)

        if "chat.completions" in shapes:
            req_body = {**body, "model": model_id, "stream": False}
            r = self._http.post(
                f"{self.saygm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.saygm_key}",
                    "Content-Type": "application/json",
                },
                json=req_body,
                timeout=self._timeout,
            )
            if r.status_code == 400:
                err_txt = r.text.lower()
                retry_body = dict(req_body)
                changed = False
                if ("max_completion_tokens" in err_txt
                        and "max_tokens" in retry_body):
                    retry_body["max_completion_tokens"] = retry_body.pop(
                        "max_tokens")
                    changed = True
                if "temperature" in err_txt and "temperature" in retry_body:
                    retry_body.pop("temperature", None)
                    changed = True
                if "seed" in err_txt and "seed" in retry_body:
                    retry_body.pop("seed", None)
                    changed = True
                if changed:
                    r = self._http.post(
                        f"{self.saygm_base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.saygm_key}",
                            "Content-Type": "application/json",
                        },
                        json=retry_body,
                        timeout=self._timeout,
                    )
            if r.status_code in (400, 401, 402, 403, 429):
                raise ProviderError(f"saygm {r.status_code}: {r.text[:240]}")
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage") or {}
            tot = int(usage.get("total_tokens", 0) or 0)
            cost = _extract_saygm_cost(usage, tot, self.FALLBACK_USD_PER_MTOK)
            return data, cost

        if "messages" in shapes:
            ant_body = _to_anthropic_body(model_id, body)
            r = self._http.post(
                f"{root}/v1/messages",
                headers={
                    "x-api-key": self.saygm_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=ant_body,
                timeout=self._timeout,
            )
            if r.status_code in (400, 401, 402, 403, 429):
                raise ProviderError(f"saygm {r.status_code}: {r.text[:240]}")
            r.raise_for_status()
            return _from_anthropic_response(model_id, r.json())

        if "generateContent" in shapes:
            gem_body = _to_gemini_body(body)
            r = self._http.post(
                f"{root}/v1beta/models/{model_id}:generateContent",
                headers={
                    "Authorization": f"Bearer {self.saygm_key}",
                    "Content-Type": "application/json",
                },
                json=gem_body,
                timeout=self._timeout,
            )
            if r.status_code in (400, 401, 402, 403, 429):
                raise ProviderError(f"saygm {r.status_code}: {r.text[:240]}")
            r.raise_for_status()
            return _from_gemini_response(model_id, r.json())

        raise ProviderError(
            f"SayGM model {model_id!r} has no supported text shape: {shapes}")


# ----------------------------------------------------------------- compute

@dataclass
class ComputeJob:
    job_id: str
    provider: str               # "runpod" | "local"
    status: str                 # queued | running | done | failed | timeout
    seconds: float = 0.0
    cost_credits: float = 0.0
    sku: str = ""
    output_tail: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


class LocalComputeProvider:
    """Backup / testing executor: runs a python script in a subprocess
    with a hard timeout, CPU-only, metered per wall-clock second.  Inside
    a sealed run it runs in that run's cgroup, on the run's share."""

    SKU = "local-cpu"
    MAX_SECONDS = 900
    MAX_OUTPUT = 20_000

    def __init__(self, rate_per_sec: float = 0.0001, workdir: str = ""):
        self.rate_per_sec = float(rate_per_sec)
        self.workdir = workdir or tempfile.gettempdir()

    def available(self) -> bool:
        return True

    def run(self, python_src: str, max_seconds: int,
            stdin_json: str = "") -> ComputeJob:
        from .schema import new_id
        max_seconds = max(1, min(int(max_seconds), self.MAX_SECONDS))
        job = ComputeJob(job_id=new_id("job"), provider="local",
                         status="running", sku=self.SKU)
        with tempfile.NamedTemporaryFile(
                "w", suffix=".py", dir=self.workdir, delete=False) as f:
            f.write(python_src)
            path = f.name
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                compute_prefix() + [sys.executable, "-I", path],
                input=stdin_json, capture_output=True, text=True,
                timeout=max_seconds, cwd=self.workdir,
                env={"PATH": os.environ.get("PATH", ""),
                     "OMP_NUM_THREADS": "1"},
            )
            job.status = "done" if proc.returncode == 0 else "failed"
            job.output_tail = (proc.stdout + "\n--- stderr ---\n"
                               + proc.stderr)[-self.MAX_OUTPUT:]
        except subprocess.TimeoutExpired as e:
            job.status = "timeout"
            job.output_tail = ((e.stdout or "") if isinstance(e.stdout, str)
                               else "")[-self.MAX_OUTPUT:]
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        job.seconds = time.monotonic() - t0
        job.cost_credits = job.seconds * self.rate_per_sec
        return job


class RunPodProvider:
    """Primary compute: RunPod serverless endpoints."""

    DEFAULT_RATES = {           # credits (USD) per metered second
        "runpod-cpu": 0.00012,
        "runpod-a40": 0.00070,
        "runpod-a100": 0.00190,
    }

    def __init__(self, api_key: Optional[str] = None,
                 endpoints: Optional[Dict[str, str]] = None,
                 rates: Optional[Dict[str, float]] = None,
                 base_url: str = "https://api.runpod.ai/v2"):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "").strip()
        if endpoints is not None:
            self.endpoints = dict(endpoints)
        else:
            try:
                from config import RUNPOD_ENDPOINTS
                self.endpoints = dict(RUNPOD_ENDPOINTS or {})
            except ImportError:
                self.endpoints = {}
        self.rates = dict(self.DEFAULT_RATES)
        if rates:
            self.rates.update(rates)
        self.base_url = _require_https(base_url, "runpod base_url")

    def available(self) -> bool:
        return bool(self.api_key and self.endpoints)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def run(self, sku: str, input_payload: Dict[str, Any],
            max_seconds: int, poll_interval: float = 2.0) -> ComputeJob:
        from .schema import new_id
        if not self.available():
            raise ProviderUnavailable("RunPod not configured "
                                      "(RUNPOD_API_KEY / RUNPOD_ENDPOINTS)")
        endpoint = self.endpoints.get(sku)
        if not endpoint:
            raise ProviderError(f"unknown RunPod SKU {sku!r}; "
                                f"have {sorted(self.endpoints)}")
        job = ComputeJob(job_id=new_id("job"), provider="runpod",
                         status="queued", sku=sku)
        r = httpx.post(f"{self.base_url}/{endpoint}/run",
                       headers=self._headers(),
                       json={"input": input_payload}, timeout=30.0)
        r.raise_for_status()
        rp_id = r.json()["id"]
        job.meta["runpod_id"] = rp_id
        t0 = time.monotonic()
        deadline = t0 + max(10, int(max_seconds)) + 60  # queue grace
        status: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            s = httpx.get(f"{self.base_url}/{endpoint}/status/{rp_id}",
                          headers=self._headers(), timeout=30.0)
            s.raise_for_status()
            status = s.json()
            st = status.get("status", "")
            if st in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                break
            job.status = "running"
            time.sleep(poll_interval)
        else:
            httpx.post(f"{self.base_url}/{endpoint}/cancel/{rp_id}",
                       headers=self._headers(), timeout=30.0)
            status = {"status": "TIMED_OUT"}

        st = status.get("status", "TIMED_OUT")
        job.status = {"COMPLETED": "done", "FAILED": "failed",
                      "CANCELLED": "failed", "TIMED_OUT": "timeout"}.get(
                          st, "failed")
        exec_ms = float(status.get("executionTime", 0) or 0)
        job.seconds = (exec_ms / 1000.0) if exec_ms > 0 \
            else (time.monotonic() - t0)
        job.cost_credits = job.seconds * float(
            self.rates.get(sku, max(self.rates.values())))
        out = status.get("output")
        job.output_tail = (str(out) if out is not None else "")[-20_000:]
        return job


def best_available(runpod: RunPodProvider, local: LocalComputeProvider,
                   sku: str = "runpod-cpu"):
    """A ``run(python_src, stdin_json, max_seconds) -> ComputeJob`` that
    uses RunPod when it is configured and answering, else the local
    executor."""
    def run(python_src: str, stdin_json: str, max_seconds: int) -> ComputeJob:
        if runpod.available():
            try:
                return runpod.run(sku, {"python_src": python_src,
                                        "stdin_json": stdin_json},
                                  max_seconds)
            except Exception as e:  # noqa: BLE001
                logger.warning("runpod failed (%s); falling back to local", e)
        return local.run(python_src, max_seconds, stdin_json)
    return run


def _parse_endpoint_env(raw: str) -> Dict[str, str]:
    """RUNPOD_ENDPOINTS="runpod-cpu=ep1,runpod-a40=ep2" -> dict."""
    out: Dict[str, str] = {}
    for part in (raw or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() and v.strip():
                out[k.strip()] = v.strip()
    return out
