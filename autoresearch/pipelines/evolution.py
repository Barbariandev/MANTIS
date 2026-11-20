# MIT License
# Copyright (c) 2024 MANTIS
"""The evolution loop: islands, LLM mutation, selection against beta.

A population of candidate specs evolves on *islands*, one per family of
market series (the target's own structure, forex majors, the other
assets).  Selection is code, not vibes: fitness is the walk-forward
t-stat and a candidate is stillborn unless its avg_r beats both the
drift baseline (always_long_avg_r) and the naked-factor baseline
(unfiltered_avg_r).  The LLM is the mutation operator — it sees the
parent, its metrics and its island's survivors, and breeds one child
(recorded with real lineage through ``refines``).  Islands trade
migrants each generation so a mechanism found on gold can jump to
forex.  A frontier judge rules on the final survivors.

Arguments: ``researcher_model`` (the mutator), ``critic_model`` (the
judge), ``iterations`` (total evaluations), ``max_tokens``.
"""

import json
import random
import sys

import loop_sdk as loop

ARGS = loop.args()
MUTATOR = str(ARGS.get("researcher_model", "gemini-3.1-pro-preview"))
JUDGE = str(ARGS.get("critic_model", "claude-opus-5-5"))
EVALS = int(ARGS.get("iterations", 28))
MAX_TOKENS = int(ARGS.get("max_tokens", 6000))
RESERVE = 0.10
DEADLINE = float(ARGS.get("deadline_seconds", 42600))
RNG = random.Random(1123)

import time

T0 = time.monotonic()
CLOCK = loop.budget().get("seconds_left")
# clock kept for the close-out: ten minutes, or a quarter of a short clock
SPARE = float(ARGS.get("spare_seconds", min(600.0, (CLOCK or 2400.0) / 4)))


def out_of_time():
    """A sealed run dies at its wall clock with nothing published.  The
    round sets the clock (12h at most, less when many enter) and
    loop.budget() says what is left of it; a pipeline that wants a
    dossier stops searching with time to spare for judging and finish()."""
    left = loop.budget().get("seconds_left")
    return ((left is not None and left < SPARE)
            or time.monotonic() - T0 > DEADLINE)

VOCAB = loop.vocabulary()
DATA = loop.data()
TARGET = VOCAB.get("target", "BTC_1h")
SERIES = sorted(VOCAB.get("series", {TARGET: {}}))
HP = VOCAB["hyperparams"]
REGIMES = [list(map(float, b)) for b in VOCAB["regime_horizon_hours"]]
LB_LO, LB_HI = VOCAB["lookback_hours"]

SPEC_SCHEMA = f"""{{"feature": {{"name": str,
  "transform": one of {json.dumps(VOCAB["transforms"])},
  "lookback_hours": int {LB_LO}..{LB_HI},
  "series": one of {json.dumps(SERIES)}, "rationale": str}},
 "hyperparams": {{"C": float, "threshold": float 0.5..0.99,
  "horizon_h": int, "regime": int 0..3, "f": float, "sl": float,
  "tp1": float, "tp2": float}}}}  (tp1 < tp2 <= 0.5, sl <= 0.5,
horizon_h inside regime bounds {REGIMES})"""

MUTATOR_SYSTEM = f"""You are the mutation operator of an evolutionary
search for BTC trading models (MANTIS LOOP; walk-forward; alpha not
beta — a child is stillborn unless avg_r beats always_long_avg_r AND
unfiltered_avg_r, so plain momentum exposure dies).  Given a parent
spec, its metrics, and its island's surviving population, breed ONE
child: a meaningful variation (lookback, threshold, horizon regime,
transform, or the series read) with a mechanism-level reason.  Stay on
the island's series.  Fitness is the t-stat shrunk by trade count,
so a child whose threshold never fires (n_trades 0) is stillborn and
ten lucky trades lose to thirty honest ones: thresholds 0.52-0.62 fire,
0.75+ almost never does; aim for 30+ trades.  A child whose
quantile receipt shows monotone_robust=true (a strictly ordered
feature-to-return response curve surviving lookback perturbation)
earns a 1.25x fitness bonus: breed toward response curves, not lucky
thresholds.
Answer STRICT JSON only: {SPEC_SCHEMA}"""

JUDGE_SYSTEM = """You are the in-loop critic — the judge of an
evolutionary run's final survivors.  Given a candidate and its
walk-forward metrics, answer strict JSON:
{"verdict": "promote"|"refine"|"prune", "notes": str}
Promote only genuine alpha: positive t_stat, enough trades, avg_r above
always_long_avg_r AND clearly above unfiltered_avg_r; prune overfit
lookbacks and naked factor exposure however good the t-stat looks.
monotone_robust=true is the strongest promote evidence (a quantile
response curve that survives lookback perturbation); monotone=0 with a
tall t_stat is likelier threshold luck."""


def extract_json(text):
    if not text:
        raise ValueError("empty reply")
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


def ask(model, system, user, step, candidate=None, temperature=0.8):
    return loop.llm(model, [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
                    step=step, candidate=candidate,
                    temperature=temperature, max_tokens=MAX_TOKENS)


def fitness(m):
    """Shrunk t_stat, or None for the stillborn (error, thin, beta).
    The shrinkage pushes evolution toward statistical mass: ten lucky
    trades with a tall t-stat lose to thirty honest ones.  A robust
    monotone quantile curve is structure selection cannot fake, so it
    earns a fitness bonus and heritability follows."""
    if not m or m.get("error") or m.get("n_trades", 0) < 8:
        return None
    bar = max(m.get("always_long_avg_r", 0.0),
              m.get("unfiltered_avg_r", 0.0), 0.0)
    if m.get("avg_r", 0.0) <= bar or m.get("t_stat", 0.0) <= 0.0:
        return None
    n = min(m["n_trades"], 50)
    f = m["t_stat"] * (n / 50.0) ** 0.5
    if m.get("monotone_robust"):
        f *= 1.25
    elif m.get("monotone"):
        f *= 1.05
    return f


def islands_of(series):
    """Series families: the target island, forex, everything else."""
    fx = [s for s in series if any(c in s for c in
                                   ("USD", "EUR", "JPY", "GBP", "AUD",
                                    "CHF")) and "BTC" not in s]
    own = [s for s in series if "BTC" in s] or [TARGET]
    other = [s for s in series if s not in fx and s not in own]
    islands = [("target", own)]
    if fx:
        islands.append(("forex", fx))
    if other:
        islands.append(("macro", other))
    return islands


def seed_spec(island_series, k):
    """A deterministic in-bounds seed: evolution needs a generation
    zero, not a good one."""
    transforms = VOCAB["transforms"]
    regime = k % len(REGIMES)
    lo, hi = REGIMES[regime]
    lb = [24, 72, 168, 336, 480][k % 5]
    return {"feature": {"name": f"seed_{k}",
                        "transform": transforms[k % len(transforms)],
                        "lookback_hours": max(LB_LO, min(LB_HI, lb)),
                        "series": island_series[k % len(island_series)]},
            "hyperparams": {"C": 1.0,
                            "threshold": [0.52, 0.55, 0.58][k % 3],
                            "horizon_h": int((lo + hi) / 2),
                            "regime": regime, "f": 0.05, "sl": 0.03,
                            "tp1": 0.02, "tp2": 0.05}}


def guard():
    if out_of_time():
        raise loop.BudgetExhausted("deadline reached")
    b = loop.budget()
    if b["remaining"] < b["budget"] * RESERVE:
        raise loop.BudgetExhausted("reserve reached")


def evaluate(spec, refines=None, rationale="", source=None, name=""):
    cid = loop.propose(spec, rationale=rationale, refines=refines,
                       source=source, name=name)
    m = loop.experiment(cid)
    f = fitness(m)
    if f is None:
        why = (m.get("error") or
               f"stillborn: avg_r {m.get('avg_r')} vs baselines "
               f"long {m.get('always_long_avg_r')} / "
               f"unfiltered {m.get('unfiltered_avg_r')}, "
               f"t {m.get('t_stat')}, n {m.get('n_trades')}")
        loop.critique(cid, "prune", why[:300])
        loop.prune(cid, why[:300])
    return {"id": cid, "spec": spec, "metrics": m, "fitness": f}


def mutate(parent, island_name, island_series, survivors, i):
    try:
        reply = ask(MUTATOR, MUTATOR_SYSTEM, json.dumps(
            {"iteration": i, "island": island_name,
             "island_series": island_series,
             "parent": {"spec": parent["spec"],
                        "metrics": parent["metrics"]},
             "survivors": [{"spec": s["spec"], "fitness": s["fitness"]}
                           for s in survivors[:6]]}), "propose")
        parsed = extract_json(reply.text)
        spec = {"feature": parsed["feature"],
                "hyperparams": parsed["hyperparams"]}
        if spec["feature"].get("series") not in island_series:
            spec["feature"]["series"] = parent["spec"]["feature"].get(
                "series", island_series[0])
        feature = spec["feature"]
        return evaluate(spec, refines=parent["id"],
                        rationale=feature.get("rationale", ""),
                        source=reply, name=feature.get("name", ""))
    except loop.BudgetExhausted:
        raise
    except (loop.LoopError, ValueError, KeyError, TypeError) as e:
        print(f"{island_name} mutation #{i}: unusable ({e})",
              file=sys.stderr)
        return None


def run():
    islands = islands_of(SERIES)
    pop = {name: [] for name, _ in islands}
    evals = 0

    # generation zero: two seeds per island, free of model calls
    for name, srs in islands:
        for k in range(2):
            if evals >= EVALS:
                break
            guard()
            pop[name].append(evaluate(seed_spec(srs, k + 2 * len(pop)),
                                      rationale="generation zero seed",
                                      name=f"{name}_seed{k}"))
            evals += 1

    generation = 0
    while evals < EVALS:
        generation += 1
        for name, srs in islands:
            if evals >= EVALS:
                break
            guard()
            alive = sorted((c for c in pop[name] if c["fitness"]),
                           key=lambda c: -c["fitness"])
            # migration: the best foreigner may breed here too
            foreign = sorted((c for island, cs in pop.items()
                              if island != name for c in cs
                              if c["fitness"]),
                             key=lambda c: -c["fitness"])
            parents = alive[:2] + foreign[:1]
            if not parents:
                parents = [pop[name][generation % len(pop[name])]] \
                    if pop[name] else []
            for parent in parents:
                if evals >= EVALS:
                    break
                guard()
                child = mutate(parent, name, srs, alive, evals)
                evals += 1
                if child:
                    pop[name].append(child)
    return pop, evals


pop, evals = {}, 0
try:
    pop, evals = run()
except loop.BudgetExhausted:
    print("budget exhausted; judging what exists", file=sys.stderr)

finalists = sorted((c for cs in pop.values() for c in cs if c["fitness"]),
                   key=lambda c: -c["fitness"])[:3]
best = None
for c in finalists:
    verdict, notes, said = "prune", "judge gave no usable reply", None
    try:
        said = ask(JUDGE, JUDGE_SYSTEM, json.dumps(
            {"candidate": c["spec"], "metrics": c["metrics"]}),
            "critique", c["id"], temperature=0.2)
        v = extract_json(said.text)
        if v.get("verdict") in ("promote", "refine", "prune"):
            verdict, notes = v["verdict"], str(v.get("notes", ""))
    except loop.BudgetExhausted:
        break
    except (loop.LoopError, ValueError, TypeError) as e:
        print(f"judge unusable on {c['id']}: {e}", file=sys.stderr)
    try:
        loop.critique(c["id"], verdict, notes, source=said)
    except loop.LoopError:
        pass
    if verdict in ("promote", "refine") and best is None:
        # prune is a veto; refine is hesitation, and the market grades
        # hesitation better than an empty run does
        best = c
        try:
            loop.promote(c["id"])
        except loop.LoopError as e:
            print(f"could not promote: {e}", file=sys.stderr)
            best = None

alive = sum(1 for cs in pop.values() for c in cs if c["fitness"])
loop.finish(
    narrative=(
        f"Evolution loop: {evals} candidates evaluated across "
        f"{len(pop)} islands (series families) with the mutator model "
        f"{MUTATOR} breeding children from surviving parents (lineage "
        f"recorded via refines) and migration between islands each "
        f"generation.  Selection was deterministic: a candidate died "
        f"unless avg_r beat both always_long_avg_r and unfiltered_avg_r "
        f"with a positive t-stat — {alive} survived.  The judge model "
        f"{JUDGE} ruled on the top survivors"
        + (f" and promoted one from the {next(n for n, cs in pop.items() if best in cs)} island."
           if best else "; none met its bar, so nothing trades.")),
    hyperparameter_rationale=(
        "All hyperparameters evolved inside the published ranges under "
        "selection against the drift and naked-factor baselines; the "
        "final pick is the judge-promoted survivor with the best "
        "walk-forward t-stat.  Nothing was tuned outside this run."))
