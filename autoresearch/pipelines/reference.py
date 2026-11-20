# MIT License
# Copyright (c) 2024 MANTIS
"""The reference LOOP pipeline: propose, test, criticise, repeat.

A researcher model proposes one candidate at a time, having seen every
earlier candidate with its walk-forward result and the critic's notes.
The gateway evaluates it.  A critic model reads the numbers and says
promote, refine or prune.  The best promoted candidate is the model.

This is a starting point, not the answer: fork it.  Everything that
decides how well a pipeline searches is in this file — what the models
are told, what they are shown, how a verdict is reached, when to stop,
how the budget is split — and all of it is yours to change.

Arguments (``args``): ``researcher_model``, ``critic_model`` (any SayGM
slugs), ``iterations``, ``max_tokens``, ``spare_seconds``
(start no iteration with less than this left on the run's clock).
"""

import json
import sys

import loop_sdk as loop

ARGS = loop.args()
RESEARCHER = str(ARGS.get("researcher_model", "kimi-k3.1-pro-tee"))
CRITIC = str(ARGS.get("critic_model", RESEARCHER))
ITERATIONS = int(ARGS.get("iterations", 8))
# reasoning models spend completion tokens before the JSON; give them
# headroom or the reply comes back empty
MAX_TOKENS = int(ARGS.get("max_tokens", 6000))
# The round sets the run's clock (12h at most, less when many enter),
# and a run still going when it runs out publishes nothing: no new
# iteration with less than ten minutes left, or a quarter of a short clock.
CLOCK = loop.budget().get("seconds_left")
SPARE = float(ARGS.get("spare_seconds", min(600.0, (CLOCK or 2400.0) / 4)))
HISTORY_WINDOW = 40         # past candidates shown to the researcher

VOCAB = loop.vocabulary()
HP = VOCAB["hyperparams"]
REGIMES = ", ".join(f"{i}={int(lo)}-{int(hi)}h" for i, (lo, hi)
                    in enumerate(VOCAB["regime_horizon_hours"]))
TARGET = VOCAB.get("target", "BTC_1h")
SERIES = {name: f"{s['bars']} bars @{s['interval_hours']}h"
          for name, s in sorted(VOCAB.get("series", {}).items())}

RESEARCHER_SYSTEM = f"""You are a quantitative researcher inside an
autoresearch loop on the MANTIS LOOP challenge.  Propose ONE candidate
trading model for BTC as strict JSON (no prose):
{{"feature": {{"name": str, "transform": one of {json.dumps(VOCAB["transforms"])},
  "lookback_hours": int {VOCAB["lookback_hours"][0]}..{VOCAB["lookback_hours"][1]},
  "series": one of {json.dumps(list(SERIES))} (the market the feature
  reads; the model always trades {TARGET}), "rationale": str}},
 "hyperparams": {{"C": float {HP["C"][0]}..{HP["C"][1]}, "threshold": float 0.5..0.99,
  "horizon_h": int, "regime": int 0..3, "f": float {HP["f"][0]}..{HP["f"][1]},
  "sl": float, "tp1": float, "tp2": float}},
 "refines": the "id" of the history candidate this one builds on, or
  null when it is a new line of research}}
The model is a logistic fit of the feature.  It goes long when its
probability of an up-move is at least (threshold - 0.5) above the
base rate, short when it is that far below, and otherwise stays out —
so a low threshold trades often and a high one rarely.
Available series (aligned to the last closed bar, never ahead):
{json.dumps(SERIES)}.  A cross-market feature (forex, gold, ETH leading
BTC) is worth trying when you can say why the mechanism would persist.
The challenge pays for alpha, not beta: plain momentum/trend/reversion
exposure is a known factor and scores nothing.  The threshold is your
regime filter — the edge must come from *when* the model trades, so aim
for candidates whose avg_r clearly beats unfiltered_avg_r (the same
model trading its signal every bar) as well as always_long_avg_r.
Every result carries a quantile receipt: quantile_r is the
mean forward return by feature band (around the 1st/10th/50th/90th/99th
percentiles), monotone is +1/-1 when that chain is strictly ordered,
and monotone_robust means the direction survives the lookback pulled a
quarter either way.  monotone_robust with the baselines beaten is the
highest-confidence promote evidence there is; a tall t_stat with
monotone 0 deserves suspicion (threshold luck, not a response curve).
Constraints: tp1 < tp2 <= 0.5, sl <= 0.5, horizon_h inside the regime's
bounds ({REGIMES}).  Each history entry has an "id", its
walk-forward metrics (always_long_avg_r is what simply being long made
over the same bars) and the critic's verdict and notes.  Learn from it;
do not repeat failed candidates.  The model you end on is also scored
on recent bars nobody in this loop can see."""

CRITIC_SYSTEM = """You are the in-loop critic.  Given a candidate and its
walk-forward metrics, answer strict JSON:
{"verdict": "promote"|"refine"|"prune", "notes": str}
Promote only genuinely promising candidates: positive t_stat, enough
trades, avg_r above always_long_avg_r (otherwise it is drift, not a
model) AND clearly above unfiltered_avg_r (the same fitted model trading
its signal's sign on every bar — if the threshold filter adds nothing,
the candidate is naked factor exposure, beta, and must be pruned), and
not all in one direction without a reason.  Be skeptical of overfit
lookbacks.  monotone_robust=true (the feature's quantile response curve
is strictly ordered and survives lookback perturbation) is the
strongest promote evidence; a tall t_stat with monotone=0 is likelier
threshold luck than a model."""


def extract_json(text):
    """The first JSON object in a model's reply."""
    if not text:
        raise ValueError("empty reply (token cap hit before any answer?)")
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


def ask(model, system, user, step, candidate=None):
    return loop.llm(model, [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
                    step=step, candidate=candidate,
                    temperature=0.7, max_tokens=MAX_TOKENS)


def research():
    """Returns ``(history, best)``; each entry is one candidate.

    Checkpoints every iteration: if the gateway pauses the run (an
    operator stopping the validator), ``loop.Paused`` propagates out,
    the run is parked, and the resumed process picks up the history —
    candidate ids stay valid — having lost at most one iteration."""
    saved = loop.restore()
    history = saved["history"] if saved else []
    best = saved["best"] if saved else None
    for i in range(saved["done"] if saved else 0, ITERATIONS):
        if loop.budget().get("seconds_left", SPARE) < SPARE:
            break
        first = max(len(history) - HISTORY_WINDOW, 0)
        shown = [{"id": first + k, "candidate": h["spec"],
                  "metrics": h["metrics"], "verdict": h["verdict"],
                  "critic_notes": h["notes"][:240]}
                 for k, h in enumerate(history[first:])]
        # -- propose
        try:
            reply = ask(RESEARCHER, RESEARCHER_SYSTEM,
                        json.dumps({"iteration": i, "history": shown}),
                        "propose")
            parsed = extract_json(reply.text)
            spec = {"feature": parsed["feature"],
                    "hyperparams": parsed["hyperparams"]}
            ref = parsed.get("refines")
            refines = (history[ref]["id"] if isinstance(ref, int)
                       and not isinstance(ref, bool)
                       and 0 <= ref < len(history) else None)
            feature = spec["feature"] if isinstance(spec["feature"],
                                                    dict) else {}
            cid = loop.propose(spec, rationale=feature.get("rationale", ""),
                               refines=refines, source=reply,
                               name=feature.get("name", ""))
        except loop.BudgetExhausted:
            raise
        except (loop.LoopError, ValueError, KeyError, TypeError) as e:
            print(f"iteration {i}: no usable proposal ({e})",
                  file=sys.stderr)
            continue

        # -- experiment
        metrics = loop.experiment(cid)

        # -- critique
        verdict, notes, said = "prune", "critic gave no usable reply", None
        try:
            said = ask(CRITIC, CRITIC_SYSTEM,
                       json.dumps({"candidate": spec, "metrics": metrics}),
                       "critique", cid)
            v = extract_json(said.text)
            if v.get("verdict") in ("promote", "refine", "prune"):
                verdict, notes = v["verdict"], str(v.get("notes", ""))
        except loop.BudgetExhausted:
            raise
        except (loop.LoopError, ValueError, TypeError) as e:
            print(f"iteration {i}: no usable critique ({e})",
                  file=sys.stderr)
        loop.critique(cid, verdict, notes, source=said)
        entry = {"id": cid, "spec": spec, "metrics": metrics,
                 "verdict": verdict, "notes": notes}
        history.append(entry)

        if verdict == "promote" and not metrics.get("error"):
            if best is None or (metrics.get("t_stat", 0)
                                > best["metrics"].get("t_stat", 0)):
                loop.promote(cid)
                best = entry
        elif verdict == "prune":
            loop.prune(cid, notes)
        loop.checkpoint({"history": history[-HISTORY_WINDOW:],
                         "best": best, "done": i + 1})
    return history, best


history, best = [], None
try:
    history, best = research()
except loop.BudgetExhausted:
    print("budget exhausted; closing the run", file=sys.stderr)

loop.finish(
    narrative=(
        "Reference pipeline: a researcher model proposes one candidate "
        "at a time from the full history of candidates, results and "
        f"critic notes; {len(history)} were evaluated walk-forward and "
        f"{sum(1 for h in history if h['verdict'] == 'promote')} were "
        "promoted by the critic model.  The model that trades is the "
        "promoted candidate with the highest walk-forward t-stat."),
    hyperparameter_rationale=(
        "Proposed by the researcher model inside the published ranges "
        "and selected by walk-forward t-stat among critic-promoted "
        "candidates; nothing was tuned outside this run."))
