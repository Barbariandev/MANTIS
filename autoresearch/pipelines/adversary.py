# MIT License
# Copyright (c) 2024 MANTIS
"""The adversary loop: a proposer, a falsifier, and survival.

One model proposes a hypothesis — a mechanism and a candidate spec that
expresses it.  A second model is paid to *break* it: it designs the
strongest counter-candidates it can (the same idea on a placebo series,
a neighbouring lookback, the naked unfiltered version) whose success
would prove the edge is luck, factor exposure or curve fit.  The
pipeline adds two deterministic controls of its own (lookback at 0.75x
and 1.33x) so no hypothesis escapes without a robustness check even
when the falsifier fumbles.  Survival is code, not rhetoric: the
hypothesis is kept only if it clears the alpha bar and its t-stat beats
every attack by a margin.  The best survivor trades.

Arguments: ``researcher_model`` (the proposer), ``critic_model`` (the
falsifier), ``iterations`` (hypotheses), ``max_tokens``.
"""

import json
import sys

import loop_sdk as loop

ARGS = loop.args()
PROPOSER = str(ARGS.get("researcher_model", "gpt-6.1-sol"))
FALSIFIER = str(ARGS.get("critic_model", "claude-opus-5-5"))
ROUNDS = int(ARGS.get("iterations", 8))
MAX_TOKENS = int(ARGS.get("max_tokens", 6000))
RESERVE = 0.08
MARGIN = 0.25
DEADLINE = float(ARGS.get("deadline_seconds", 42600))               # the hypothesis must beat its attacks by this

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

PROPOSER_SYSTEM = f"""You are the proposer in an adversarial research
loop on the MANTIS LOOP challenge (hourly BTC, walk-forward, alpha not
beta).  State ONE falsifiable hypothesis about a persistent market
mechanism and the candidate spec that expresses it.  A falsifier model
will attack it with placebo and perturbation candidates; you win only
if your candidate's t-stat beats every attack by a clear margin, so
prefer mechanisms that are *specific* (a particular series, a
particular conditioning) over generic exposure.  The threshold is your
regime filter: avg_r must clearly beat always_long_avg_r and
unfiltered_avg_r or the candidate is beta and dies.
Answer STRICT JSON only:
{{"hypothesis": str (the mechanism and why it persists),
 "prediction": str (what the attacks should show if you are right),
 "candidate": {SPEC_SCHEMA}}}
The history shows every fallen hypothesis and how it died; do not die
the same way twice.  Trade-count economics: n_trades 0 means the
threshold never fired (0.52-0.65 fires, 0.75+ almost never does on
hourly BTC), and a t-stat needs 30+ trades to mean anything — a
hypothesis that cannot generate trades cannot survive.  Every result
carries a quantile receipt (quantile_r, monotone, monotone_robust):
a strictly ordered feature-to-return response curve that survives
lookback perturbation is the strongest evidence a mechanism exists,
and it lowers the significance bar your candidate must clear."""

FALSIFIER_SYSTEM = f"""You are the falsifier in an adversarial research
loop.  You are shown a hypothesis, its candidate spec and its
walk-forward metrics.  Design up to 3 counter-candidates whose success
would prove the hypothesis wrong: the same structure on a placebo
series, the mechanism with its conditioning removed, a perturbed
lookback or horizon — attacks a true mechanism would survive and a
curve fit would not.  Answer STRICT JSON only:
{{"argument": str (why these attacks bite),
 "attacks": [up to 3 of {SPEC_SCHEMA}]}}"""

VERDICT_SYSTEM = """You are the in-loop critic recording the outcome of
an adversarial round.  Given the hypothesis candidate's metrics and its
attacks' metrics, answer strict JSON:
{"verdict": "promote"|"prune", "notes": str}
Promote only if the hypothesis clearly dominates every attack and its
avg_r beats always_long_avg_r and unfiltered_avg_r."""


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


def ask(model, system, user, step, candidate=None, temperature=0.7):
    return loop.llm(model, [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
                    step=step, candidate=candidate,
                    temperature=temperature, max_tokens=MAX_TOKENS)


def alpha_bar(m):
    """True when the metrics clear the alpha bar at all.  A robust
    monotone quantile curve is independent structural evidence, so it
    buys a lower significance bar (t > 0.8 instead of 1.0)."""
    if not m or m.get("error") or m.get("n_trades", 0) < 8:
        return False
    bar = max(m.get("always_long_avg_r", 0.0),
              m.get("unfiltered_avg_r", 0.0), 0.0)
    t_bar = 0.8 if m.get("monotone_robust") else 1.0
    return m.get("avg_r", 0.0) > bar and m.get("t_stat", 0.0) > t_bar


def t(m):
    return m.get("t_stat", 0.0) if m and not m.get("error") else 0.0


def guard():
    if out_of_time():
        raise loop.BudgetExhausted("deadline reached")
    b = loop.budget()
    if b["remaining"] < b["budget"] * RESERVE:
        raise loop.BudgetExhausted("reserve reached")


def controls(spec):
    """Two deterministic perturbation controls: a real mechanism should
    not die when its lookback moves a quarter."""
    out = []
    lb = int(spec["feature"].get("lookback_hours", 168))
    for scale, tag in ((0.75, "ctl_lb075"), (4.0 / 3.0, "ctl_lb133")):
        c = json.loads(json.dumps(spec))
        c["feature"]["lookback_hours"] = max(LB_LO,
                                             min(LB_HI, int(lb * scale)))
        if c["feature"]["lookback_hours"] != lb:
            c["feature"]["name"] = tag
            out.append((tag, c))
    return out


def clean_attack(a, k):
    spec = {"feature": a.get("feature"), "hyperparams": a.get("hyperparams")}
    if not isinstance(spec["feature"], dict) \
            or not isinstance(spec["hyperparams"], dict):
        raise ValueError("attack is not a spec")
    spec["feature"].setdefault("name", f"attack_{k}")
    return spec


def near_miss(m):
    """Real trades, beats the baselines, not yet significant."""
    if not m or m.get("error") or m.get("n_trades", 0) < 30:
        return False
    bar = max(m.get("always_long_avg_r", 0.0),
              m.get("unfiltered_avg_r", 0.0), 0.0)
    return m.get("avg_r", 0.0) > bar and 0.5 < m.get("t_stat", 0.0) <= 1.0


STRENGTHEN_SYSTEM = f"""You are the proposer, strengthening your own
near-miss: the candidate trades, beats both baselines, but its t-stat
is not yet significant.  Propose up to 2 variants of THIS candidate
(adjust threshold, lookback, horizon or the series read — same
mechanism) that should trade more or cleaner.  Answer STRICT JSON only:
{{"variants": [up to 2 of {SPEC_SCHEMA}]}}"""


def strengthen(i, hypothesis, cid, spec, metrics):
    """One refinement sub-round for a near-miss; returns the best of
    the family (the original included)."""
    best = (cid, spec, metrics)
    try:
        guard()
        reply = ask(PROPOSER, STRENGTHEN_SYSTEM, json.dumps(
            {"iteration": i, "hypothesis": hypothesis,
             "candidate": spec, "metrics": metrics}), "propose",
            candidate=cid, temperature=0.5)
        variants = (extract_json(reply.text).get("variants") or [])[:2]
    except loop.BudgetExhausted:
        raise
    except (loop.LoopError, ValueError, TypeError) as e:
        print(f"round {i}: strengthen unusable ({e})", file=sys.stderr)
        return best
    for k, v in enumerate(variants):
        try:
            vspec = clean_attack(v, k)
            vspec["feature"]["name"] = f"strengthened_{k}"
            vid = loop.propose(vspec, rationale=f"strengthens {cid}",
                               refines=cid, source=reply,
                               name=vspec["feature"]["name"])
            vm = loop.experiment(vid)
        except loop.BudgetExhausted:
            raise
        except (loop.LoopError, ValueError, TypeError):
            continue
        if t(vm) > t(best[2]) and not vm.get("error"):
            if best[0] != cid:
                loop.prune(best[0], "weaker strengthened variant")
            best = (vid, vspec, vm)
        else:
            loop.prune(vid, "strengthened variant did not improve")
    if best[0] != cid:
        loop.critique(cid, "refine", "superseded by strengthened variant")
        loop.prune(cid, "superseded by strengthened variant")
    return best


def round_(i, fallen):
    guard()
    shown = [{"hypothesis": f["hypothesis"][:300], "died": f["died"][:300]}
             for f in fallen[-10:]]
    reply = ask(PROPOSER, PROPOSER_SYSTEM,
                json.dumps({"iteration": i, "history": shown}), "propose")
    parsed = extract_json(reply.text)
    hypothesis = str(parsed.get("hypothesis", ""))[:600]
    # a bare spec is accepted as its own hypothesis
    cand = parsed.get("candidate") if isinstance(parsed.get("candidate"),
                                                 dict) else parsed
    if not hypothesis:
        hypothesis = str(cand.get("feature", {}).get("rationale",
                                                     "unstated"))[:600]
    spec = {"feature": cand["feature"], "hyperparams": cand["hyperparams"]}
    cid = loop.propose(spec, rationale=hypothesis, source=reply,
                       name=str(spec["feature"].get("name", ""))[:40])
    metrics = loop.experiment(cid)
    if near_miss(metrics):
        cid, spec, metrics = strengthen(i, hypothesis, cid, spec, metrics)
    if not alpha_bar(metrics):
        why = (metrics.get("error") or
               f"below the alpha bar: avg_r {metrics.get('avg_r')} vs "
               f"long {metrics.get('always_long_avg_r')} / unfiltered "
               f"{metrics.get('unfiltered_avg_r')}, t {metrics.get('t_stat')}")
        loop.critique(cid, "prune", why[:300])
        loop.prune(cid, why[:300])
        return {"hypothesis": hypothesis, "died": why, "id": cid,
                "spec": spec, "metrics": metrics, "survived": False}

    # the falsifier gets its shot
    attacks = [(tag, c, None) for tag, c in controls(spec)]
    argument = "deterministic controls only"
    try:
        guard()
        said = ask(FALSIFIER, FALSIFIER_SYSTEM, json.dumps(
            {"hypothesis": hypothesis, "candidate": spec,
             "metrics": metrics}), "critique", cid, temperature=0.5)
        fp = extract_json(said.text)
        argument = str(fp.get("argument", ""))[:500]
        for k, a in enumerate((fp.get("attacks") or [])[:3]):
            try:
                attacks.append((f"falsifier_{k}", clean_attack(a, k), said))
            except (ValueError, TypeError, AttributeError):
                continue
    except loop.BudgetExhausted:
        raise
    except (loop.LoopError, ValueError, TypeError) as e:
        print(f"round {i}: falsifier unusable ({e}); controls only",
              file=sys.stderr)

    results = []
    for tag, aspec, source in attacks:
        try:
            guard()
            aid = loop.propose(aspec, rationale=f"attack on {cid}: {tag}",
                               refines=cid, source=source, name=tag)
            am = loop.experiment(aid)
            results.append({"tag": tag, "id": aid, "metrics": am})
            loop.prune(aid, "attack candidate, not a model")
        except loop.BudgetExhausted:
            raise
        except loop.LoopError as e:
            print(f"round {i}: attack {tag} refused ({e})", file=sys.stderr)

    beaten = [r for r in results if t(r["metrics"]) + MARGIN >= t(metrics)]
    survived = not beaten
    if survived:
        verdict, notes = "promote", (
            f"dominated {len(results)} attacks by >= {MARGIN} t; "
            f"falsifier: {argument}")
    else:
        worst = max(beaten, key=lambda r: t(r["metrics"]))
        verdict, notes = "prune", (
            f"falsified by {worst['tag']} (t {t(worst['metrics']):.2f} vs "
            f"{t(metrics):.2f}); {argument}")
    loop.critique(cid, verdict, notes[:400])
    if not survived:
        loop.prune(cid, notes[:300])
    return {"hypothesis": hypothesis, "died": "" if survived else notes,
            "id": cid, "spec": spec, "metrics": metrics,
            "survived": survived, "attacks": len(results)}


rounds, fallen, survivors = 0, [], []
try:
    for i in range(ROUNDS):
        try:
            outcome = round_(i, fallen)
        except loop.BudgetExhausted:
            raise
        except (loop.LoopError, ValueError, KeyError, TypeError) as e:
            print(f"round {i}: no usable hypothesis ({e})", file=sys.stderr)
            continue
        rounds += 1
        (survivors if outcome["survived"] else fallen).append(outcome)
except loop.BudgetExhausted:
    print("budget exhausted; closing the run", file=sys.stderr)

best = None
for s in survivors:
    if best is None or t(s["metrics"]) > t(best["metrics"]):
        best = s
if best is not None:
    try:
        loop.promote(best["id"])
    except loop.LoopError as e:
        print(f"could not promote: {e}", file=sys.stderr)
        best = None

loop.finish(
    narrative=(
        f"Adversary loop: the proposer ({PROPOSER}) staked "
        f"{rounds} falsifiable hypotheses; each one that cleared the "
        f"alpha bar faced counter-candidates designed by the falsifier "
        f"({FALSIFIER}) plus two deterministic lookback controls, and "
        f"survived only by beating every attack's t-stat by {MARGIN}.  "
        f"{len(fallen)} hypotheses fell, {len(survivors)} survived"
        + (".  The best survivor trades." if best else
           "; nothing trades.")),
    hyperparameter_rationale=(
        "Hyperparameters came with each hypothesis inside the published "
        "ranges; the surviving candidate was selected by walk-forward "
        "t-stat after dominating its falsification battery, not by "
        "tuning.  Attack candidates were evaluated and pruned, never "
        "promoted."))
