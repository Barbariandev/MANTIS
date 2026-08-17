# FLOW simulation evidence

Simulation package behind the FLOW release paper
(`FLOW_RELEASE.pdf`, published on the MANTIS site). **Not wired into the
validator** — the production scoring module is `../flow.py`; this package
holds the adversarial studies that set the shipped parameters, and a
cross-check that replays the same actors through the production module.

All runs use a cached real BTC minute tape (`cache_btc_multi_1m.npz`,
two years, four-venue consensus channels). Rational entry/exit: adversary
keys join when observed per-capita emissions beat entry cost and quit when
underwater (free-entry equilibrium), so leakage numbers are against an
adaptive population, not a fixed roster.

All code lives in one module: `sims.py`.

## Reproduction

```bash
cd MANTIS   # so flow_sim is importable; needs numpy, pandas, matplotlib
python3 -m flow_sim.sims finalspec   # ship-spec headline + burn/crowding tables (~5s)
python3 -m flow_sim.sims papergrid   # release grid (~35s)
python3 -m flow_sim.sims prodsim     # production flow.py cross-check (~5s)
python3 -m flow_sim.sims stakeecon   # capital-at-risk collateral study + settlement-flow figure (~70s)
python3 -m flow_sim.sims relfigs     # heatmap figures from the CSVs (~5s)
python3 -m flow_sim.sims rational    # burn sweep on the real tape
python3 -m flow_sim.sims fetch ...   # rebuild the tape cache (binance|bybit|okx|coinbase|okx-merge|build)
```

Outputs land in `out/`:

| Output | Study |
| --- | --- |
| `finalspec.csv`, `finalspec_ramp.png` | ship spec at launch parameters: headline clears, counterfactual entry burns, crowding (release §6.1) |
| `papergrid.csv`, `paper_wr_clear.png`, `paper_bond_ev.png` | win-rate boundary, threshold × latch, block length, min evidence, counterfactual entry-burn curve, entry pressure, dust, cadence, adversary-only, ret vs alpha stat, no-gate baseline, scoring invariance (release §6.2–6.13) |
| `prodsim.csv`, `prodsim_ramp.png` | rational-actor intents replayed through `../flow.py` (dim-28 decode, gate, payment) on hourly bars; wick vs close-only resolution (release §6.14) |
| `stakeecon.csv`, `stakeecon_flow.png` | collateral ledger + weekly settlement economics: whale band, break-aware adversary sizing, weighting modes, weekly extraction flows (release §6.15) |
| `paper_threshold_heat.png`, `paper_gammatau.png`, `paper_sensitivity.png` | threshold × latch heatmaps, scoring invariance, one-knob sensitivity strip (rendered from `papergrid.csv`) |
| `settlement_lifecycle.png` | spec diagram of the live-fed settlement lifecycle (release §1.6; drawn from constants) |

## Fidelity notes

- Kelly inversion in the validity gate: `p = (f*b + 1) / (b + 1)`.
- `R_MAE` is capped at 1 because the stop-loss truncates any deeper drawdown
  (inherent to the resolution rules, not a simplification).
- Single price source; multi-venue resolution can be layered on by
  narrowing the high/low channel (max of venue lows, min of venue highs:
  hardest-to-trigger stops, hardest-to-award targets) before resolution.
- Same-candle ambiguities are explicit knobs (`same_candle_rule`,
  `tp1_then_sl`), both defaulting to the miner-pessimistic reading.
