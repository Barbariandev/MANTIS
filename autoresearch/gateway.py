# MIT License
# Copyright (c) 2024 MANTIS
"""The LOOP Research Gateway — a development tool.

Production LOOP has **no gateway**: submissions ride miners' R2
payloads, the auditor validator evaluates them in 24-hour rounds
in-process (``rounds.py``), evidence is read off the ledger directly
and the dashboard is a rendered file.  This HTTP app wraps the very
same ledger, runner and sealbox for local development and the offline
test suite, so a pipeline debugged against it behaves identically in a
round.

Runs miners' pipelines sealed (``sealed.py``): a miner submits source
and a budget, the gateway
executes it in a sandbox whose only way out is the gateway itself, and
everything the run did — every model call with its full request and
reply, every experiment, the model it ended on — is published here.

Surface (all JSON):

    POST /v1/register                hotkey <-> funding-address binding
    GET  /v1/balance/{hotkey}
    POST /v1/runs                    submit a pipeline: {hotkey, source,
                                     args, budget_credits, nonce,
                                     signature} -> {run_id}
    POST /v1/runs/{id}/resume        continue (or cancel) a paused run:
                                     {hotkey, nonce, signature
                                     [, openrouter_key(_sealed)]
                                     [, action: "cancel"]}
    GET  /v1/runs/{id}               status and what the gateway attests
    GET  /v1/runs/{id}/events        the trace (written by the gateway)
    GET  /v1/runs/{id}/receipts
    GET  /v1/runs/{id}/dossier       assembled by the gateway
    GET  /v1/hotkeys/{hotkey}/runs
    GET  /v1/hotkeys/{hotkey}/receipts
    GET  /v1/blobs/{sha256}          pipeline sources, datasets, model
                                     requests and replies
    GET  /v1/harness                 the evaluator, to replay with
    GET  /v1/network/summary         aggregates for public dashboards

There is no endpoint that accepts a trace, a metric or a dossier: those
are things the gateway produces.  ``register`` and ``runs`` are signed
by the hotkey (sr25519); dev mode skips the check.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from . import harness
from .credits import CreditLedger, UnknownRun
from .providers import (InferenceClient, LocalComputeProvider,
                        RunPodProvider, best_available)
from .schema import canonical_json, sha256_hex
from .sealed import Sandbox, SealedRunner, SealedSettings, SubmitError

logger = logging.getLogger(__name__)

REGISTER_MESSAGE_TEMPLATE = "mantis-loop-register:{hotkey}:{evm_address}"
RUN_MESSAGE_TEMPLATE = ("mantis-loop-run:{hotkey}:{pipeline_sha256}:"
                        "{args_sha256}:{budget_micro}:{nonce}")
RESUME_MESSAGE_TEMPLATE = ("mantis-loop-resume:{hotkey}:{run_id}:"
                           "{action}:{nonce}")
_NONCE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def run_message(hotkey: str, source: str, args: dict,
                budget_credits: float, nonce: str) -> str:
    """What a hotkey signs to submit a run: this source, these
    arguments, this budget, once."""
    return RUN_MESSAGE_TEMPLATE.format(
        hotkey=hotkey, pipeline_sha256=sha256_hex(source),
        args_sha256=sha256_hex(canonical_json(args)),
        budget_micro=round(float(budget_credits) * 1e6), nonce=nonce)


def _cfg_int(name: str, default: int) -> int:
    try:
        import config
        return int(getattr(config, name, default))
    except ImportError:
        return default


def _cfg_float(name: str, default: float) -> float:
    try:
        import config
        return float(getattr(config, name, default))
    except ImportError:
        return default


@dataclass
class GatewaySettings:
    secret: str = ""
    dev_mode: bool = False
    api_key: str = field(default_factory=lambda: os.environ.get(
        "SAYGM_API_KEY", "").strip())
    llm_fee: float = 0.0          # optional markup on inference (fraction)
    local_rate_per_sec: float = 0.0001
    holdout_bars: int = field(
        default_factory=lambda: _cfg_int("LOOP_HOLDOUT_BARS", 480))
    # the wall clock is the bottleneck by design: inference is the
    # miner's own key, so time — not credits — bounds a sealed run
    max_run_seconds: int = field(
        default_factory=lambda: _cfg_int("LOOP_MAX_RUN_SECONDS", 43200))
    max_concurrent_runs: int = 8
    # the hotkey whose signed verdict map this gateway republishes
    # (the auditor validator, uid0); empty = no verdict feed
    auditor_hotkey: str = ""
    # X25519 seed (32 bytes hex) for local development gateway tests
    enc_seed: str = ""
    # stopping the gateway pauses runs instead of killing them: how
    # long a pipeline gets to checkpoint, and how long a paused run
    # stays resumable before it fails
    pause_grace_seconds: float = field(
        default_factory=lambda: _cfg_float("LOOP_PAUSE_GRACE_SECONDS", 20.0))
    resume_window_seconds: float = field(
        default_factory=lambda: _cfg_float("LOOP_RESUME_WINDOW_SECONDS", 259200.0))

    def __post_init__(self):
        if not self.secret:
            raise ValueError("GatewaySettings.secret must be set "
                             "(receipts are signed with it)")


def _verify_sr25519(hotkey_ss58: str, message: str, signature_hex: str) -> bool:
    """Verify an sr25519 signature by the hotkey.  Uses the installed
    bittensor stack; returns False (never raises) on any failure."""
    try:
        try:
            from bittensor_wallet import Keypair  # bt 8-10
        except ImportError:
            try:
                from bittensor_core import Keypair  # bt >= 11
            except ImportError:
                from bittensor import Keypair
        kp = Keypair(ss58_address=hotkey_ss58)
        sig = bytes.fromhex(signature_hex.removeprefix("0x"))
        return bool(kp.verify(message.encode(), sig))
    except Exception:
        logger.exception("sr25519 verification failed")
        return False


def create_app(ledger: CreditLedger,
               settings: Optional[GatewaySettings] = None,
               inference: Optional[InferenceClient] = None,
               runpod: Optional[RunPodProvider] = None,
               local: Optional[LocalComputeProvider] = None,
               dataset: Optional[Callable[[], dict]] = None,
               sandbox: Optional[Sandbox] = None,
               verify: Callable[[str, str, str], bool] = _verify_sr25519
               ) -> FastAPI:
    """``dataset`` supplies the market dataset new runs research
    (default: live multi-market, see ``data.build_dataset``);
    ``verify(hotkey, message, signature)`` checks hotkey signatures."""
    st = settings or GatewaySettings()
    if dataset is None:
        from .data import MarketData
        dataset = MarketData()
    runner = SealedRunner(
        ledger,
        SealedSettings(
            secret=st.secret, holdout_bars=st.holdout_bars,
            max_run_seconds=st.max_run_seconds, llm_fee=st.llm_fee,
            pipeline_rate_per_sec=st.local_rate_per_sec,
            max_concurrent_runs=st.max_concurrent_runs,
            pause_grace_seconds=st.pause_grace_seconds,
            resume_window_seconds=st.resume_window_seconds,
            # an unsealed run executes miner code on this host: only a
            # developer's own gateway may do that
            allow_unsealed=st.dev_mode),
        inference or (InferenceClient(api_key=st.api_key)
                      if st.api_key else InferenceClient()),
        best_available(
            runpod or RunPodProvider(),
            local or LocalComputeProvider(rate_per_sec=st.local_rate_per_sec)),
        dataset, sandbox)

    app = FastAPI(title="MANTIS LOOP Research Gateway", version="0.2.0")
    app.state.runner = runner

    def _signed(hotkey: str, message: str, signature: Any) -> None:
        if st.dev_mode:
            return
        if not isinstance(signature, str) or not verify(
                hotkey, message, signature):
            raise HTTPException(403, "invalid hotkey signature")

    def _run(run_id: str) -> dict:
        run = ledger.sealed_run(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return run

    def _public(run: dict) -> dict:
        """A run without its dossier body."""
        out = {k: v for k, v in run.items() if k != "dossier"}
        out["has_dossier"] = run["dossier"] is not None
        try:
            info = ledger.run_info(run["run_id"])
            out["budget_credits"] = info.budget_credits
            out["spent_credits"] = info.spent_credits
        except UnknownRun:
            pass
        return out

    # ----------------------------------------------------------- accounts

    @app.post("/v1/register")
    async def register(body: Dict[str, Any]):
        hotkey = str(body.get("hotkey", ""))
        evm_address = str(body.get("evm_address", ""))
        if not hotkey or not evm_address:
            raise HTTPException(400, "hotkey and evm_address required")
        msg = REGISTER_MESSAGE_TEMPLATE.format(
            hotkey=hotkey, evm_address=evm_address.lower())
        _signed(hotkey, msg, body.get("signature"))
        ledger.register(hotkey, evm_address)
        return {"ok": True, "hotkey": hotkey,
                "funding_address": evm_address.lower(),
                "signed_message": msg}

    @app.get("/v1/balance/{hotkey}")
    async def balance(hotkey: str):
        return {"hotkey": hotkey, "credits": ledger.balance(hotkey)}

    # --------------------------------------------------------------- runs

    # Plain `def`: opening a run may fetch the dataset, which blocks.
    @app.post("/v1/runs")
    def submit(body: Dict[str, Any]):
        hotkey, source = body.get("hotkey"), body.get("source")
        args = body.get("args") or {}
        nonce = body.get("nonce")
        if not isinstance(hotkey, str) or not hotkey:
            raise HTTPException(400, "hotkey required")
        if not isinstance(nonce, str) or not _NONCE.match(nonce):
            raise HTTPException(400, "nonce: 8-64 characters of "
                                     "[A-Za-z0-9_-], never reused")
        if not isinstance(source, str) or not isinstance(args, dict):
            raise HTTPException(400, "source (text) and args (object) "
                                     "required")
        try:
            budget = float(body.get("budget_credits"))
            message = run_message(hotkey, source, args, budget, nonce)
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(400, "budget_credits must be a number")
        _signed(hotkey, message, body.get("signature"))
        # the miner's own inference key: a transport secret, not part of
        # the signed message, never stored or served.  Sealed is the
        # production form — encrypted to the gateway's X25519 key and
        # bound to this hotkey and nonce, it opens only here and only
        # for this submission; plaintext still rides TLS for dev.
        key = body.get("api_key")
        sealed = body.get("api_key_sealed")
        if key is not None and not isinstance(key, str):
            raise HTTPException(400, "api_key: a string")
        if sealed is not None:
            if key is not None:
                raise HTTPException(400, "send api_key or "
                                         "api_key_sealed, not both")
            if not isinstance(sealed, str):
                raise HTTPException(400, "api_key_sealed: hex")
            if not st.enc_seed:
                raise HTTPException(400, "this gateway publishes no "
                                         "encryption key; send the key "
                                         "in plaintext over TLS")
            from . import sealbox
            try:
                key = sealbox.open_sealed(st.enc_seed, sealed,
                                          f"{hotkey}:{nonce}")
            except ValueError as e:
                raise HTTPException(403, str(e))
        try:
            run_id = runner.submit(hotkey, source, args, budget, nonce,
                                   api_key=key)
        except SubmitError as e:
            raise HTTPException(400, str(e))
        runner.start(run_id)
        return {"run_id": run_id, "signed_message": message}

    @app.post("/v1/runs/{run_id}/resume")
    def resume_run(run_id: str, body: Dict[str, Any]):
        """Continue (or cancel) a paused run.  Only its own hotkey can:
        the request is signed and its nonce burns on use, so it cannot
        be replayed.  A BYOK run needs its key again (sealed, like a
        submission): the gateway never kept it."""
        run = _run(run_id)
        hotkey = body.get("hotkey")
        nonce = body.get("nonce")
        action = "cancel" if body.get("action") == "cancel" else "resume"
        if hotkey != run["hotkey"]:
            raise HTTPException(403, "only the run's hotkey may do this")
        if not isinstance(nonce, str) or not _NONCE.match(nonce):
            raise HTTPException(400, "nonce: 8-64 characters of "
                                     "[A-Za-z0-9_-], never reused")
        message = RESUME_MESSAGE_TEMPLATE.format(
            hotkey=hotkey, run_id=run_id, action=action, nonce=nonce)
        _signed(hotkey, message, body.get("signature"))
        try:
            ledger.claim_nonce(hotkey, nonce)
        except ValueError as e:
            raise HTTPException(403, str(e))
        if action == "cancel":
            try:
                runner.cancel(run_id)
            except SubmitError as e:
                raise HTTPException(400, str(e))
            return {"run_id": run_id, "status": "failed",
                    "signed_message": message}
        key = body.get("api_key")
        sealed = body.get("api_key_sealed")
        if key is not None and not isinstance(key, str):
            raise HTTPException(400, "api_key: a string")
        if sealed is not None:
            if key is not None:
                raise HTTPException(400, "send api_key or "
                                         "api_key_sealed, not both")
            if not isinstance(sealed, str):
                raise HTTPException(400, "api_key_sealed: hex")
            if not st.enc_seed:
                raise HTTPException(400, "this gateway publishes no "
                                         "encryption key; send the key "
                                         "in plaintext over TLS")
            from . import sealbox
            try:
                key = sealbox.open_sealed(st.enc_seed, sealed,
                                          f"{hotkey}:{nonce}")
            except ValueError as e:
                raise HTTPException(403, str(e))
        try:
            runner.resume(run_id, api_key=key)
        except SubmitError as e:
            raise HTTPException(400, str(e))
        return {"run_id": run_id, "status": "resuming",
                "signed_message": message}

    @app.on_event("shutdown")
    def _graceful_pause() -> None:
        # stopping the gateway parks every live run instead of killing
        # it; the next start resumes credit runs by itself and holds
        # BYOK runs for their miners' keys
        paused = runner.pause_all()
        if paused:
            logger.info("paused %d run(s) for shutdown: %s",
                        len(paused), ", ".join(paused))

    @app.get("/v1/runs/{run_id}")
    async def run_status(run_id: str):
        return _public(_run(run_id))

    @app.get("/v1/runs/{run_id}/events")
    async def get_events(run_id: str):
        return {"events": ledger.events_for_run(run_id)}

    @app.get("/v1/runs/{run_id}/receipts")
    async def run_receipts(run_id: str):
        return {"receipts": ledger.receipts(run_id=run_id)}

    @app.get("/v1/runs/{run_id}/dossier")
    async def run_dossier(run_id: str):
        dossier = _run(run_id)["dossier"]
        if dossier is None:
            raise HTTPException(404, "this run has no dossier")
        return dossier

    @app.get("/v1/hotkeys/{hotkey}/runs")
    async def hotkey_runs(hotkey: str):
        return {"runs": [_public(r) for r in ledger.sealed_runs(hotkey)]}

    @app.get("/v1/hotkeys/{hotkey}/receipts")
    async def hotkey_receipts(hotkey: str):
        return {"receipts": ledger.receipts(hotkey=hotkey)}

    # ------------------------------------------------------------- public

    @app.get("/v1/blobs/{sha256}")
    async def blob(sha256: str):
        body = ledger.get_blob(sha256) if _SHA.match(sha256) else None
        if body is None:
            raise HTTPException(404, "no such blob")
        return {"sha256": sha256, "body": body}

    @app.get("/v1/harness")
    async def harness_source():
        return {"sha256": harness.HARNESS_SHA256,
                "source": harness.HARNESS_SOURCE,
                "vocabulary": harness.vocabulary()}

    # ---- the auditor's signed verdicts, for validator consensus.
    # The payload authenticates itself (sr25519 by the auditor hotkey),
    # so this gateway is a convenient mirror, not a trust root: it can
    # serve nothing a follower would accept that the auditor did not
    # sign, and a stale replay is refused here and again by followers.

    @app.post("/v1/audit/verdicts")
    def put_verdicts(body: Dict[str, Any]):
        if not st.auditor_hotkey:
            raise HTTPException(404, "this gateway has no auditor")
        from .audit import verify_verdicts
        try:
            payload = verify_verdicts(body, st.auditor_hotkey, verify)
        except ValueError as e:
            raise HTTPException(403, str(e))
        held = getattr(app.state, "verdicts", None)
        if held is not None and payload["ts"] <= held["ts"]:
            raise HTTPException(409, "older than the verdicts already "
                                     "published")
        app.state.verdicts = payload
        return {"ok": True, "verdicts": len(payload["verdicts"]),
                "ts": payload["ts"]}

    @app.get("/v1/audit/verdicts")
    def get_verdicts():
        held = getattr(app.state, "verdicts", None)
        if held is None:
            raise HTTPException(404, "no verdicts published yet")
        return held

    @app.get("/v1/network/dashboard")
    async def network_dashboard(days: int = 30):
        import config as mantis_config
        from .dashboard import challenge_dashboard
        return challenge_dashboard(
            ledger, getattr(mantis_config, "LOOP_AUDIT_DB", ""), days)

    @app.get("/v1/network/summary")
    async def network_summary():
        out = ledger.network_summary()
        if st.enc_seed:
            from . import sealbox
            try:
                out["enc_pubkey"] = sealbox.keypair(st.enc_seed)[1]
            except ValueError:
                pass            # a bad seed refuses sealed keys anyway
        return out

    return app


def main():
    import argparse
    import uvicorn
    import config as mantis_config
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(mantis_config.PROJECT_ROOT, ".env"))
    except ImportError:
        pass

    import secrets as _secrets
    p = argparse.ArgumentParser(description="LOOP Research Gateway (local dev)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--dev-mode", action="store_true",
                   help="skip sr25519 signature checks (local dev only)")
    p.add_argument("--db", default=mantis_config.LOOP_LEDGER_DB)
    args = p.parse_args()

    ledger = CreditLedger(args.db)
    app = create_app(
        ledger,
        GatewaySettings(secret=_secrets.token_hex(32),
                        dev_mode=bool(args.dev_mode)),
    )
    logger.info("sandbox isolation: %s", app.state.runner.sandbox.isolation)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
