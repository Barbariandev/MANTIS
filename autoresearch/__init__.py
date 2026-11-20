# MIT License
# Copyright (c) 2024 MANTIS
"""LOOP — the MANTIS autoresearch challenge.

Miners design autoresearch pipelines and publish their source in their
payload with their own SayGM (``gm_live_...``) key sealed to LOOP's
rotating round key; the auditor (the UID 0 validator) runs them sealed
in daily rounds via ``loop-keyd``, making the model calls, running the
experiments on the protocol's harness and writing the record.  Trades
are scored on FLOW criteria and a validator audit — attestation, replay,
a frontier judge reading the pipeline — gates the payout.

Modules:
  schema      dossier / event / receipt data contracts + canonical hashing
  harness     the evaluator: model spec -> metrics, held-out result, trade
  credits     credit ledger (budgets -> metered spends), run records
  providers   SayGM (SN28) client, RunPod + local compute
  keyd        out-of-process Unix-socket jail for miner inference keys
  sealed      sandbox + broker: runs a pipeline, records what it did
  slots       every run's share of the machine, and a round's schedule
  loop_sdk    what a pipeline imports inside the sandbox
  pipelines   example pipelines (reference.py: the one to fork)
  submission  the payload's LOOP field: build it, open it
  sealbox     sealing a miner's key to LOOP's rotating X25519 keyring
  rounds      the auditor's 24h rounds: screen, run, audit, publish
  round_site  the public round dashboard
  dashboard   dashboard aggregations and the manual render
  gateway     the dev gateway (local development; production has none)
  loop        offline pieces: stub models, synthetic market data
  audit       validator: attestation, replay, frontier judge, verdicts
  scoring     flow-criteria salience gated by audit multipliers
  data        the canonical dataset: hourly BTC + 10y of daily BTC,
              forex majors, ETH, gold
"""
