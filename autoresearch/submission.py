# MIT License
# Copyright (c) 2024 MANTIS
"""The LOOP submission wire format: a pipeline rides the miner's own
MANTIS payload.

There is no submission server.  A miner puts this object under the
``"LOOP"`` key of its ordinary v2 payload plaintext (the one it already
uploads to its committed R2 URL), timelocked like everything else::

    "LOOP": {
        "trade":  [28 floats],          # the hourly FLOW-format trade
        "submit": {                     # optional: a pipeline submission
            "v": 1,
            "source": "...python...",   # the autoresearch pipeline
            "args": {...},              # its public arguments
            "budget_credits": 25.0,     # spend cap for the run
            "sha256": "...",            # of the fields above, canonical
            "key_id": "2026-11-20",     # rotating round seal key id
            "key_sealed": "...hex..."   # the miner's SayGM key (gm_live_...),
        }                               # X25519-sealed to LOOP's round key
    }

LOOP runs exclusively on **SayGM** (Bittensor SN28 Intel TDX TEE gateway,
``https://api.saygm.com/v1``, ``gm_live_...`` keys).  OpenRouter and other
provider keys are refused.

The auditor validator (UID 0) opens each payload through its ``W_owner``
wrap as soon as it fetches it (``open_payload``) and keeps only the
submission in its private inbox.  At 0000 UTC it evaluates the newest
submission per hotkey first read in the preceding 24 hours
(``autoresearch/rounds.py``).

What this format guarantees:

* **The source becomes public — after its round.**  Until the round
  closes only the auditor has read it, provided the payload's own
  timelock lasts that long (LOOP payloads take the week-long lock);
  then the dashboard shows every evaluated pipeline's code.
* **The key never does — and never enters datalog.db.**  ``key_sealed``
  is encrypted to LOOP's rotating per-round X25519 key (published signed
  at ``<LOOP_PUBLIC_URL>/seal_keys.json``, decoupled from ``W_owner``)
  with AAD ``loop-submission:{hotkey}:{sha256}``.  ``ledger.py`` strips
  ``key_sealed`` before writing matured submissions to ``datalog.db``,
  the auditor wipes ``key_sealed`` from ``loop_inbox`` once a round
  closes, and old round private keys are shredded after the resume
  window.
* **One submission runs once.**  ``sha256`` names the submission; the
  auditor evaluates each (hotkey, sha256) at most once, in the round
  after it was first read.  To run again, change something (``args``
  do).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Optional, Tuple, Union

from .schema import sha256_hex

MAX_SOURCE_CHARS = 200_000          # matches the sealed runner's cap
MAX_ARGS_CHARS = 4_000
MAX_SUBMISSION_CHARS = 300_000      # the whole object, as JSON
MAX_RECORDED_CHARS = 400_000        # ledger.py drops larger ones unread
OWNER_WRAP_INFO = b"mantis-owner-wrap"

# What a miner's inference key looks like, wherever it might leak into
# text that gets displayed or logged.  Broad on purpose (still scrubs
# sk-or-* if someone pastes one in a comment, even though OpenRouter keys
# are refused at submission).
_KEY_SHAPES = re.compile(
    r"(cpk_[A-Za-z0-9_\-]{6,}"                  # Chutes (SN64)
    r"|gm_(?:live|test)_[A-Za-z0-9_\-]{6,}"     # SayGM (SN28)
    r"|sk-or-(?:v\d+-)?[A-Za-z0-9_\-]{6,}"      # OpenRouter (refused, still scrubbed)
    r"|sk-[A-Za-z0-9_\-]{20,}"                  # generic sk- secrets
    r"|Bearer\s+[A-Za-z0-9_\-\.=]{16,})")       # bearer tokens


def redact(text: str) -> str:
    """``text`` with anything key-shaped struck out.  Every surface that
    displays or logs miner-authored content goes through this."""
    return _KEY_SHAPES.sub("[REDACTED-KEY]", text or "")


def validate_inference_key_format(key: str) -> str:
    """Validate that ``key`` is a single-line SayGM key (``gm_live_...``
    or ``gm_test_...``).  Refuses OpenRouter, Chutes, and any other
    provider format."""
    if not isinstance(key, str):
        raise ValueError("api_key: a single-line string")
    clean = key.strip()
    if not 0 < len(clean) <= 200 or any(c in key for c in "\r\n\0"):
        raise ValueError("api_key: a single-line string")
    if clean.lower().startswith("sk-or-"):
        raise ValueError(
            "OpenRouter keys (sk-or-...) are not accepted on LOOP: "
            "provide a low-funded SayGM key (gm_live_...)")
    if not (clean.startswith("gm_live_")
            or clean.startswith("gm_test_")
            or clean == "stub"):
        raise ValueError(
            "api_key must be a SayGM key (gm_live_...)")
    return clean


def strip_sealed_key_fields(sub: Any) -> Any:
    """Return a shallow copy of ``sub`` (dict or JSON string) with
    ``key_sealed`` and ``key_id`` removed so ciphertexts never persist in
    ``datalog.db`` or closed ``loop_inbox`` rows."""
    if isinstance(sub, str):
        try:
            parsed = json.loads(sub)
        except ValueError:
            return sub
        if not isinstance(parsed, dict):
            return sub
        cleaned = {k: v for k, v in parsed.items()
                   if k not in ("key_sealed", "key_id")}
        return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    if isinstance(sub, dict):
        return {k: v for k, v in sub.items()
                if k not in ("key_sealed", "key_id")}
    return sub


def _aad(hotkey: str, sha: str) -> str:
    return f"loop-submission:{hotkey}:{sha}"


def submission_sha(source: str, args: dict, budget_credits: float) -> str:
    """The submission's name: a hash of what runs, not of the key."""
    return sha256_hex(json.dumps(
        {"v": 1, "source": source, "args": args,
         "budget_credits": round(float(budget_credits), 6)},
        sort_keys=True, separators=(",", ":")))


def fetch_round_seal_pubkey(public_url: str = "",
                            auditor_hotkey: str = "",
                            http: Any = None,
                            timeout: float = 10.0) -> Tuple[str, str]:
    """Fetch and verify ``<public_url>/seal_keys.json`` and return
    ``(key_id, pubkey_hex)``."""
    from . import sealbox
    if not public_url:
        import config
        public_url = getattr(config, "LOOP_PUBLIC_URL", "")
    if not public_url:
        raise ValueError("no LOOP_PUBLIC_URL configured")
    if http is None:
        import httpx as http
    url = public_url.rstrip("/") + "/seal_keys.json"
    r = http.get(url, timeout=timeout)
    if r.status_code != 200:
        raise ValueError(f"seal_keys.json HTTP {r.status_code}")
    return sealbox.verify_seal_keys(
        r.json(),
        auditor_hotkey=auditor_hotkey,
        max_age_seconds=7 * 86400.0,
    )


def _resolve_seal_target(auditor_pubkey: Optional[str],
                         key_id: Optional[str]) -> Tuple[Optional[str], str]:
    """Resolve ``(key_id, pubkey_hex)`` to seal a miner's inference key to.

    In production (``config.LOOP_SEAL_FETCH_LIVE = True``), fetches and
    verifies UID 0's signed ``seal_keys.json`` manifest and fails closed
    on network or signature failure (never downgrading to static ``W_owner``).
    """
    if auditor_pubkey:
        return key_id, auditor_pubkey
    import config
    cfg_pub = getattr(config, "LOOP_SEAL_PUBLIC_KEY_HEX", "").strip()
    if cfg_pub:
        return key_id, cfg_pub
    if getattr(config, "LOOP_SEAL_FETCH_LIVE", True):
        aud_hk = getattr(config, "AUTORESEARCH_CHALLENGE", {}).get(
            "auditor_hotkey", "")
        if not aud_hk:
            try:
                from .scoring import uid0_hotkey
                aud_hk = uid0_hotkey(int(getattr(config, "NETUID", 123)))
            except Exception:  # noqa: BLE001
                aud_hk = ""
        if not aud_hk:
            aud_hk = getattr(config, "LOOP_UID0_HOTKEY", "")
        return fetch_round_seal_pubkey(
            getattr(config, "LOOP_PUBLIC_URL", ""),
            auditor_hotkey=aud_hk, timeout=10.0)
    owner_pub = getattr(config, "OWNER_HPKE_PUBLIC_KEY_HEX", "").strip()
    if owner_pub:
        return key_id, owner_pub
    raise ValueError(
        "sealing a key needs a recipient public key: pass auditor_pubkey "
        "or ensure LOOP_PUBLIC_URL/seal_keys.json is reachable")


def build_loop_field(hotkey: str, source: str, args: Optional[dict] = None,
                     budget_credits: float = 25.0,
                     api_key: Optional[str] = None,
                     auditor_pubkey: Optional[str] = None,
                     trade: Optional[list] = None,
                     key_id: Optional[str] = None,
                     *,
                     inference_key: Optional[str] = None) -> Dict[str, Any]:
    """The value for the ``"LOOP"`` key of a v2 payload plaintext.

    ``api_key`` (a low-funded SayGM ``gm_live_...`` key) pays the models,
    sealed to LOOP's rotating per-round X25519 key so that only
    ``loop-keyd`` ever opens it.  OpenRouter and non-SayGM keys are
    refused.  A submission without a key runs only on credits the owner
    granted."""
    if api_key is not None and inference_key is not None:
        raise ValueError("pass api_key once")
    secret_key = api_key if api_key is not None else inference_key
    args = args or {}
    sub: Dict[str, Any] = {
        "v": 1, "source": source, "args": args,
        "budget_credits": round(float(budget_credits), 6)}
    sub["sha256"] = submission_sha(source, args, budget_credits)
    parse_submission(hotkey, sub)       # refuse to build what cannot run
    if secret_key is not None:
        clean_key = validate_inference_key_format(secret_key)
        resolved_id, target_pub = _resolve_seal_target(auditor_pubkey, key_id)
        from . import sealbox
        sub["key_sealed"] = sealbox.seal(
            target_pub, clean_key, _aad(hotkey, sub["sha256"]))
        if resolved_id:
            sub["key_id"] = resolved_id
    out: Dict[str, Any] = {"submit": sub}
    if trade is not None:
        out["trade"] = trade
    return out


def parse_submission(hotkey: str, sub: Any) -> Dict[str, Any]:
    """A validated submission dict, or ``ValueError`` saying exactly
    what is wrong — the same message the round journal records, so a
    miner can read why its entry never ran."""
    if not isinstance(sub, dict):
        raise ValueError("submit: a JSON object")
    if sub.get("v") != 1:
        raise ValueError("v: 1")
    if len(json.dumps(sub)) > MAX_SUBMISSION_CHARS:
        raise ValueError("submission too large: at most "
                         f"{MAX_SUBMISSION_CHARS} characters as JSON")
    source = sub.get("source")
    if not isinstance(source, str) or not source.strip() \
            or len(source) > MAX_SOURCE_CHARS:
        raise ValueError("source: the pipeline's Python source, at most "
                         f"{MAX_SOURCE_CHARS} characters")
    try:
        compile(source, "pipeline.py", "exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        raise ValueError(f"source does not compile: {e}")
    args = sub.get("args")
    if not isinstance(args, dict) or len(json.dumps(args)) > MAX_ARGS_CHARS:
        raise ValueError("args: a JSON object, at most "
                         f"{MAX_ARGS_CHARS} characters")
    try:
        budget = float(sub.get("budget_credits"))
    except (TypeError, ValueError):
        raise ValueError("budget_credits: a number")
    if not budget > 0:
        raise ValueError("budget_credits: positive")
    sha = sub.get("sha256")
    if sha != submission_sha(source, args, budget):
        raise ValueError("sha256 does not match the submission")
    key_sealed = sub.get("key_sealed")
    if key_sealed is not None and (not isinstance(key_sealed, str)
                                   or len(key_sealed) > 4096):
        raise ValueError("key_sealed: sealed-box hex")
    key_id = sub.get("key_id")
    if key_id is not None and (not isinstance(key_id, str)
                               or len(key_id) > 64):
        raise ValueError("key_id: string at most 64 characters")
    out: Dict[str, Any] = {
        "v": 1, "source": source, "args": args,
        "budget_credits": budget, "sha256": sha,
        "key_sealed": key_sealed,
    }
    if key_id is not None:
        out["key_id"] = key_id
    return out


def open_submission_key(enc_seed_or_keyring: Union[str, Any], hotkey: str,
                        sub: Dict[str, Any]) -> Optional[str]:
    """The miner's Chutes/SayGM key, opened fresh inside ``keyd`` (or in
    offline unit tests).  ``None`` when the submission carries no key
    (a credits run).  Never store or log what this returns."""
    sealed = sub.get("key_sealed")
    if sealed is None:
        return None
    from . import sealbox
    aad = _aad(hotkey, sub["sha256"])
    if isinstance(enc_seed_or_keyring, sealbox.Keyring):
        raw = enc_seed_or_keyring.open(sealed, aad, key_id=sub.get("key_id"))
    else:
        raw = sealbox.open_sealed(enc_seed_or_keyring, sealed, aad)
    return validate_inference_key_format(raw)


def open_payload(payload: Any, hotkey: str,
                 owner_seed: str) -> Optional[Dict[str, Any]]:
    """The plaintext of ``hotkey``'s v2 payload, opened with the owner
    key — the payload's ``W_owner`` wrap of its content key — without
    waiting for its timelock.  ``None`` for anything that is not a v2
    payload of this hotkey wrapped to this owner key.  The binding, KDF
    and AEAD must be ``generate_and_encrypt.generate_v2``'s."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PublicKey)
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from . import sealbox
    try:
        if isinstance(payload, (bytes, bytearray, str)):
            payload = json.loads(payload)
        if (not isinstance(payload, dict) or payload.get("v") != 2
                or payload.get("hk") != hotkey):
            return None
        owner, owner_hex = sealbox.keypair(owner_seed)
        if str(payload.get("owner_pk", "")).lower() != owner_hex:
            return None
        wrap, body = payload["W_owner"], payload["C"]
        pke = bytes.fromhex(wrap["pke"])
        binding = hashlib.sha256(b":".join((
            hotkey.encode("utf-8"),
            str(int(payload["round"])).encode("ascii"),
            bytes.fromhex(owner_hex), pke))).digest()
        wrap_key = HKDF(algorithm=hashes.SHA256(), length=44, salt=None,
                        info=OWNER_WRAP_INFO).derive(owner.exchange(
                            X25519PublicKey.from_public_bytes(pke)))[:32]
        key = ChaCha20Poly1305(wrap_key).decrypt(
            bytes.fromhex(wrap["nonce"]), bytes.fromhex(wrap["ct"]), binding)
        obj = json.loads(ChaCha20Poly1305(key).decrypt(
            bytes.fromhex(body["nonce"]), bytes.fromhex(body["ct"]),
            binding).decode("utf-8"))
    except Exception:  # noqa: BLE001 — miner-authored bytes, any shape
        return None
    return obj if isinstance(obj, dict) and obj.get("hotkey") == hotkey \
        else None


def loop_submission_of(plaintext: Any,
                       strip_key: bool = False) -> Optional[Tuple[str, str]]:
    """``(sha256, body)`` of the submission riding a payload plaintext's
    LOOP field.  ``strip_key=True`` drops ``key_sealed`` / ``key_id`` (used
    when a matured copy enters the public datalog)."""
    field = plaintext.get("LOOP") if isinstance(plaintext, dict) else None
    sub = field.get("submit") if isinstance(field, dict) else None
    if not isinstance(sub, dict):
        return None
    if strip_key:
        sub = strip_sealed_key_fields(sub)
    try:
        body = json.dumps(sub, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    if len(body) > MAX_RECORDED_CHARS:
        return None
    sha = sub.get("sha256")
    if not (isinstance(sha, str) and len(sha) == 64
            and all(c in "0123456789abcdef" for c in sha)):
        sha = hashlib.sha256(body.encode()).hexdigest()
    return sha, body
