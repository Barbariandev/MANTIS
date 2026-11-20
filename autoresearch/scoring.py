# MIT License
# Copyright (c) 2024 MANTIS
"""LOOP salience: frontier credits on the FLOW EWMA score, gated by
the audit agent.  Pay tracks *progress*, not possession.

The measuring stick is literally ``flow.compute_flow_salience`` — same
bracket decoding, path penalties, tail amplification, Kelly weighting,
EWMA and significance machinery, running f-only (LOOP skin is research
evidence, not posted collateral).  But the split is Refinery-style
(SN125): walking the shared history hour by hour, whoever pushes the
best audited FLOW EWMA score *ever seen* higher books a credit equal to
the advance.  Credits halve every ``credit_half_life_h`` (default 336h
= 14 days) and expire after four half-lives; each cycle the full LOOP
slice is split pro rata by live credit × the audit multiplier (a banded
value in {0, .25, .5, .75, 1} from ``audit.AuditorPanel``), and when no
live credit exists — stagnation — the whole slice burns, absolutely.

Properties this buys:

* matching the leader pays nothing; beating it pays the improvement —
  linear, so splitting one advance across hotkeys earns exactly the
  advance (clone-split invariant);
* a parked model is an annuity that halves every fortnight, not a
  perpetuity: keep improving to keep earning;
* everything is a pure function of the shared datalog plus the mirrored
  audit map — no new consensus state, every validator reconstructs the
  same frontier, the same events, the same credits;
* gated hotkeys (multiplier 0) neither book credits nor raise the
  frontier, so a smuggler cannot jack the bar to starve honest miners.

A hotkey with no audited dossier earns ``audit_default`` (config/env;
0.0 once the auditor is live — no research, no pay).  The burn key
absorbs everything unearned so the vector still sums to 1.

Where the multipliers come from decides validator consensus:

* the auditor — whoever holds UID 0, the subnet owner's validator —
  runs the judge and publishes the map signed after every round
  (``audit.publish_to_site``);
* every validator, the auditor included, pays on that published map:
  it reads UID 0's hotkey and its on-chain commitment, which names
  where the map is served (``audit.ChainFeed``).  Nothing to configure;
  the overrides (``auditor_hotkey``, ``audit_feed_url``, ``audit_db``)
  are for development.  No judge runs on a follower and no divergence
  is possible: the trade leg is deterministic from the shared datalog,
  so mirroring this one map aligns the whole LOOP weight.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from flow import BURN_KEY, compute_flow_salience

logger = logging.getLogger(__name__)


_FEEDS: Dict[tuple, "object"] = {}


def load_followed_multipliers(url: str, auditor_hotkey: str,
                              cache_path: str = "") -> Dict[str, float]:
    """The auditor's signed verdicts fetched from a fixed URL (dev; the
    production path is ``load_chain_multipliers``).  Any failure keeps
    the last verified map rather than diverging."""
    try:
        from .audit import FollowedVerdicts
        key = (url, auditor_hotkey, cache_path)
        if key not in _FEEDS:
            _FEEDS[key] = FollowedVerdicts(url, auditor_hotkey,
                                           cache_path or None)
        return _FEEDS[key].multiplier_map()
    except Exception:  # noqa: BLE001
        logger.exception("followed verdicts failed (%s)", url)
        return {}


class _Chain:
    """Scoring's chain reads — who holds UID 0, the auditor's
    commitment — on one client per process.  A client dropped without
    ``close()`` keeps its socket and threads, and these reads come every
    few minutes, so a failed read closes the client and the next one
    reconnects."""

    def __init__(self):
        self._client = None

    def _read(self, fn):
        import bt_compat
        try:
            from config import BT_NETWORK as _net
        except ImportError:
            _net = "finney"
        if self._client is None:
            self._client = bt_compat.Subtensor(network=_net)
        client = self._client
        try:
            return fn(client, bt_compat)
        except Exception:
            if self._client is client:
                self._client = None
            try:
                client._client.close()
            except Exception:  # noqa: BLE001
                pass
            raise

    def get_all_commitments(self, netuid: int) -> dict:
        return self._read(lambda c, _bt: c.get_all_commitments(netuid))

    def metagraph(self, netuid: int):
        try:
            from config import BT_NETWORK as _net
        except ImportError:
            _net = "finney"
        return self._read(lambda c, bt: bt.Metagraph(
            netuid=netuid, sync=True, subtensor=c, network=_net))


_CHAIN = _Chain()


def _subtensor():
    return _CHAIN


def _metagraph(netuid: int):
    return _CHAIN.metagraph(netuid)


_UID0: Dict[int, Tuple[str, float]] = {}


def uid0_hotkey(netuid: int, ttl: float = 3600.0,
                retry: float = 300.0) -> str:
    """The auditor: the hotkey holding UID 0 (the subnet owner's
    validator), re-read hourly.  A failed read keeps the last holder
    seen ("" before the first success) and tries again in five
    minutes."""
    netuid = int(netuid)
    hk, until = _UID0.get(netuid, ("", 0.0))
    if time.time() < until:
        return hk
    try:
        mg = _metagraph(netuid)
        hk = next(key for uid, key in zip(mg.uids, mg.hotkeys)
                  if int(uid) == 0)
        _UID0[netuid] = (hk, time.time() + ttl)
    except Exception:  # noqa: BLE001
        logger.exception("could not read UID 0 (netuid %s)", netuid)
        _UID0[netuid] = (hk, time.time() + retry)
    return hk


def load_chain_multipliers(auditor_hotkey: str, netuid: int,
                           cache_path: str = "") -> Dict[str, float]:
    """The auditor's signed verdicts, discovered through its on-chain
    commitment (no URL to configure: the auditor moves, the pointer
    moves with it).  Any failure keeps the last verified map."""
    try:
        from .audit import ChainFeed
        key = ("chain", auditor_hotkey, int(netuid), cache_path)
        if key not in _FEEDS:
            _FEEDS[key] = ChainFeed(_subtensor, int(netuid),
                                    auditor_hotkey, cache_path or None)
        return _FEEDS[key].multiplier_map()
    except Exception:  # noqa: BLE001
        logger.exception("chain verdicts failed (netuid %s)", netuid)
        return {}


def load_audit_multipliers(audit_db: str) -> Dict[str, float]:
    """Read the latest banded multiplier per hotkey from the audit cache.
    Missing/broken DB -> empty map (every hotkey falls to the default)."""
    try:
        from .audit import AuditorPanel
        panel = AuditorPanel(cache_db=audit_db, judges=[])
        return panel.multiplier_map()
    except Exception:  # noqa: BLE001
        logger.exception("audit multiplier load failed (%s)", audit_db)
        return {}


CREDIT_WINDOW_HALF_LIVES = 4.0        # a credit expires at ~6% residue


def frontier_credits(diag: Dict[str, dict], eligible,
                     half_life_h: float) -> Dict[str, float]:
    """hotkey -> live decayed credit, from the per-hour score paths.

    The frontier starts at zero; processing hours in order (and hotkeys
    in lexicographic order within an hour, so simultaneous crossings
    split deterministically), every advance of the best-ever eligible
    score books a credit of that size to the hotkey that made it,
    decayed by its age.  Telescoping: undecayed credits always sum to
    the final frontier, however many hands moved it."""
    paths = {hk: d["score_path"] for hk, d in diag.items()
             if isinstance(d, dict) and d.get("score_path") is not None
             and hk in eligible}
    if not paths:
        return {}
    hks = sorted(paths)
    n = len(paths[hks[0]])
    window_h = float(half_life_h) * CREDIT_WINDOW_HALF_LIVES
    frontier = 0.0
    credits: Dict[str, float] = {}
    for hi in range(n):
        age_h = float(n - 1 - hi)
        for hk in hks:
            s = float(paths[hk][hi])
            if s > frontier:
                advance = s - frontier
                frontier = s
                if age_h <= window_h:
                    credits[hk] = credits.get(hk, 0.0) + (
                        advance * 0.5 ** (age_h / float(half_life_h)))
    return {hk: v for hk, v in credits.items() if v > 0.0}


def compute_autoresearch_salience(
    hist: Tuple[np.ndarray, Dict[str, int]],
    prices_multi: np.ndarray,
    *,
    sample_every: int = 5,
    sidx_arr: Optional[np.ndarray] = None,
    spec: Optional[dict] = None,
    audit_multipliers: Optional[Dict[str, float]] = None,
    return_diagnostics: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], dict]:
    """hotkey -> fraction of the LOOP pool (+ BURN_KEY; sums to 1)."""
    spec = spec or {}
    try:
        import config as _cfg
        _defaults = _cfg.AUTORESEARCH_CHALLENGE
        _default_netuid = _cfg.NETUID
    except Exception:  # noqa: BLE001
        _defaults = {}
        _default_netuid = 123
    if audit_multipliers is None:
        audit_db = spec.get("audit_db") or _defaults.get("audit_db", "")
        feed_url = spec.get("audit_feed_url") or _defaults.get(
            "audit_feed_url", "")
        auditor_hotkey = spec.get("auditor_hotkey") or _defaults.get(
            "auditor_hotkey", "")
        cache = spec.get("audit_feed_cache", "")
        netuid = (spec.get("audit_netuid")
                  or _defaults.get("audit_netuid")
                  or _default_netuid)
        if feed_url and auditor_hotkey:    # explicit test spec: fixed URL
            audit_multipliers = load_followed_multipliers(
                feed_url, auditor_hotkey, cache)
        elif audit_db and not auditor_hotkey:   # explicit test spec: local DB
            audit_multipliers = load_audit_multipliers(audit_db)
        else:                              # consensus: UID 0's published map
            hk = auditor_hotkey or uid0_hotkey(int(netuid))
            audit_multipliers = (load_chain_multipliers(hk, int(netuid),
                                                        cache)
                                 if hk else {})
    audit_default = float(spec.get(
        "audit_default", _defaults.get("audit_default", 0.0)))
    half_life_h = float(spec.get(
        "credit_half_life_h", _defaults.get("credit_half_life_h", 336.0)))

    _scores, diag = compute_flow_salience(
        hist, prices_multi, sample_every=sample_every, sidx_arr=sidx_arr,
        spec=spec, return_diagnostics=True, score_paths=True)

    def gate(hk: str) -> float:
        m = float(audit_multipliers.get(hk, audit_default))
        return min(max(m, 0.0), 1.0)

    # gated hotkeys cannot raise the frontier: the bar is the best
    # *audited* record, so fake alpha cannot starve honest miners
    eligible = {hk for hk in diag
                if isinstance(diag.get(hk), dict) and gate(hk) > 0.0}
    credits = frontier_credits(diag, eligible, half_life_h)
    total = float(sum(v for _k, v in sorted(credits.items())))

    out: Dict[str, float] = {}
    gated: Dict[str, float] = {}
    paid = 0.0
    if total > 0.0:
        for hk in sorted(credits):
            m = gate(hk)
            kept = credits[hk] / total * m
            if kept > 0.0:
                out[hk] = kept
                paid += kept
            gated[hk] = m
    # stagnation — no live credit — burns the whole slice, absolutely
    out[BURN_KEY] = max(min(1.0 - paid, 1.0), 0.0)

    if return_diagnostics:
        diag = dict(diag)
        diag["audit_multipliers"] = gated
        diag["audit_default"] = audit_default
        diag["frontier_credits"] = credits
        diag["credit_half_life_h"] = half_life_h
        return out, diag
    return out
