# MIT License
# Copyright (c) 2024 MANTIS
#
# Compatibility shim for bittensor >= 11.0.0.
#
# bittensor 11 removed the classic ``Subtensor``/``Metagraph`` method surface
# (``get_current_block``, ``get_all_commitments``, ``set_weights``,
# ``Metagraph(...).sync()`` with ``.uids``/``.hotkeys`` tensors) and replaced it
# with a client whose reads go through ``client.read(name, **params)`` and whose
# writes go through ``client.execute(intent, wallet)``.
#
# This module re-exposes the small slice of the old interface that the MANTIS
# validator/cycle code relies on, so those files need only swap
# ``bt.Subtensor``/``bt.Metagraph`` for the wrappers here. It works on
# bittensor 11.x and is a no-op-ish thin layer over the new client.

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import bittensor as bt

logger = logging.getLogger(__name__)


class Subtensor:
    """Thin wrapper over ``bittensor.Subtensor`` (the v11 sync client) exposing
    the handful of methods the MANTIS code calls."""

    def __init__(self, network: str = "finney"):
        self.network = network
        self._client = bt.Subtensor(network=network)

    # -- reads -------------------------------------------------------------
    def get_current_block(self) -> int:
        return int(self._client.block())

    def get_all_commitments(self, netuid: int) -> dict[str, str]:
        """Return ``{hotkey: commitment_string}`` for a subnet.

        v11's ``commitments`` read returns one row per hotkey, newest first, so
        the first time we see a hotkey is its latest commitment.
        """
        rows = self._client.read("commitments", netuid=netuid)
        out: dict[str, str] = {}
        for r in rows:
            hk = r.get("hotkey")
            data = r.get("commitment")
            if hk and data and hk not in out:
                out[hk] = data
        return out

    # passthroughs (in case other call sites need the raw client)
    def read(self, *args, **kwargs):
        return self._client.read(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self._client.execute(*args, **kwargs)

    def block(self) -> int:
        return int(self._client.block())

    # -- writes ------------------------------------------------------------
    def set_weights(
        self,
        *,
        netuid: int,
        wallet,
        uids,
        weights,
        mechid: int = 0,
        version_key: int = 0,
        **_ignored,
    ):
        """Set weights on-chain.

        Accepts the old-style parallel ``uids`` / ``weights`` arrays (numpy or
        torch tensors work) and forwards them as a v11 ``SetWeights`` intent.
        v11 conforms the vector to the subnet hyperparameters and picks
        plaintext vs. timelocked commit-reveal automatically. Extra legacy
        kwargs (e.g. ``wait_for_inclusion``) are accepted and ignored.
        """
        uid_list = [int(u) for u in list(uids)]
        weight_list = [float(w) for w in list(weights)]
        pairs = {u: w for u, w in zip(uid_list, weight_list) if w > 0.0}
        if not pairs:
            logger.warning("set_weights: all weights are zero, nothing to submit.")
            return None

        intent = bt.SetWeights(
            netuid=netuid,
            uids=list(pairs.keys()),
            weights=list(pairs.values()),
            mechid=mechid,
            version_key=version_key,
        )
        result = self._client.execute(intent, wallet)
        if not getattr(result, "success", False):
            logger.warning(
                "set_weights extrinsic did not succeed: %s",
                getattr(result, "message", result),
            )
        return result


class Metagraph:
    """Minimal stand-in for the old ``bittensor.Metagraph`` exposing ``.uids``
    (numpy int64 array) and ``.hotkeys`` (list aligned to ``.uids``).

    Deliberately a plain data holder (no live client reference) so it remains
    ``copy.deepcopy``-able, which the validator relies on when handing a
    snapshot to the weight-calculation worker thread.
    """

    def __init__(
        self,
        netuid: int,
        network: str = "finney",
        sync: bool = True,
        subtensor: Optional[Subtensor] = None,
        lite: bool = True,
    ):
        self.netuid = int(netuid)
        self.network = network
        self.uids = np.array([], dtype=np.int64)
        self.hotkeys: list[str] = []
        self.coldkeys: list[str] = []
        if sync:
            self.sync(subtensor=subtensor)

    def _resolve_client(self, subtensor: Optional[Subtensor]):
        if isinstance(subtensor, Subtensor):
            return subtensor._client
        if subtensor is not None and hasattr(subtensor, "read"):
            return subtensor
        return bt.Subtensor(network=self.network)

    def sync(self, subtensor: Optional[Subtensor] = None, **_ignored):
        client = self._resolve_client(subtensor)
        neurons = client.read("neurons", netuid=self.netuid, lite=True)
        neurons = sorted(neurons, key=lambda n: int(n.uid))
        self.uids = np.array([int(n.uid) for n in neurons], dtype=np.int64)
        self.hotkeys = [n.hotkey for n in neurons]
        self.coldkeys = [getattr(n, "coldkey", None) for n in neurons]
        return self
