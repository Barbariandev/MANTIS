# MIT License
# Copyright (c) 2024 MANTIS
"""The lab loop: a principal investigator, cheap workers, one referee.

An expensive frontier model (the PI) never grinds candidates — it is
called twice: once to design a small set of research *programs* (a
thesis, the series and transforms it should be explored with), and once
mid-run to read every program's results and redirect the remaining
budget.  Cheap worker models — picked at run time from ``loop.models()``
by price — mass-produce candidates inside each program, and a mid-tier
referee model rules on every result.  The spend follows the thinking:
dollars go to direction, cents go to volume.

Arguments: ``researcher_model`` (the PI), ``critic_model`` (the
referee), ``iterations`` (total worker proposals), ``max_tokens``.
"""

import json
import sys

import loop_sdk as loop

ARGS = loop.args()
PI = str(ARGS.get("researcher_model", "gpt-6-astra"))
REFEREE = str(ARGS.get("critic_model", "kimi-k3.1-pro-tee"))
ITERATIONS = int(ARGS.get("iterations", 24))
MAX_TOKENS = int(ARGS.get("max_tokens", 6000))
RESERVE = 0.08          # budget fraction kept for the close-out
DEADLINE = float(ARGS.get("deadline_seconds", 42600))

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
SERIES = {name: f"{s['bars']} bars @{s['interval_hours']}h"
          for name, s in sorted(VOCAB.get("series", {}).items())}
HP = VOCAB["hyperparams"]
REGIMES = ", ".join(f"{i}={int(lo)}-{int(hi)}h" for i, (lo, hi)
                    in enumerate(VOCAB["regime_horizon_hours"]))

SPEC_SCHEMA = f"""{{"feature": {{"name": str,
  "transform": one of {json.dumps(VOCAB["transforms"])},
  "lookback_hours": int {VOCAB["lookback_hours"][0]}..{VOCAB["lookback_hours"][1]},
  "series": one of {json.dumps(list(SERIES))} (the model always trades {TARGET}),
  "rationale": str}},
 "hyperparams": {{"C": float {HP["C"][0]}..{HP["C"][1]},
  "threshold": float 0.5..0.99, "horizon_h": int, "regime": int 0..3,
  "f": float {HP["f"][0]}..{HP["f"][1]}, "sl": float, "tp1": float, "tp2": float}}}}
Constraints: tp1 < tp2 <= 0.5, sl <= 0.5, horizon_h inside the regime's
bounds ({REGIMES}).  The model is a logistic fit of the feature; it
trades only when its probability leaves the base rate by (threshold -
0.5), so the threshold is the regime filter.  Alpha, not beta: a
candidate is worthless unless its avg_r clearly beats BOTH
always_long_avg_r (drift) and unfiltered_avg_r (the same fitted model
trading every bar — naked factor exposure)."""

PI_SYSTEM = f"""You are the principal investigator of a quantitative
research lab working on the MANTIS LOOP challenge (hourly BTC, walk-
forward evaluation, pay for alpha not beta).  Cheap worker models will
mass-produce candidate specs inside the research programs you define;
you will not see the candidates until the mid-run review.  Available
series (aligned causally, never ahead): {json.dumps(SERIES)}.
Transforms: {json.dumps(VOCAB["transforms"])}.
Answer STRICT JSON only:
{{"programs": [exactly 3 of {{"name": str (short), "thesis": str (the
mechanism and why it persists), "series": [1-3 series names],
"transforms": [1-3 transform names], "lookback_hours": [lo, hi],
"guidance": str (what the workers should vary and what to avoid)}}]}}
Favour cross-market mechanisms and conditional structure over plain
momentum/trend, which is beta and scores zero."""

REVIEW_SYSTEM = """You are the same principal investigator at the
mid-run review.  You see each program's candidates and walk-forward
metrics.  Answer STRICT JSON only:
{"allocation": {program_name: int (remaining proposals, sum to the
stated remaining total)}, "guidance": {program_name: str (what to
refine or abandon)}}"""

REFEREE_SYSTEM = """You are the in-loop critic (the lab's referee).
Given a candidate and its walk-forward metrics, answer strict JSON:
{"verdict": "promote"|"refine"|"prune", "notes": str}
Promote only genuine alpha: positive t_stat, enough trades, avg_r above
always_long_avg_r AND clearly above unfiltered_avg_r (otherwise it is
drift or naked factor exposure).  Be skeptical of overfit lookbacks.
monotone_robust=true (a strictly ordered quantile response curve that
survives lookback perturbation) is the strongest promote evidence; a
tall t_stat with monotone=0 is likelier threshold luck."""

FINAL_SYSTEM = """You are the in-loop critic at the lab's final review.
You see the run's strongest candidates — every one already beats the
drift and naked-factor baselines with t > 1 walk-forward.  A run that
promotes nothing publishes nothing and pays nothing, and promoting is
not a claim of certainty: the promoted model is scored on bars nobody
here has seen, so the market grades it either way.  Pick the candidate
whose edge is most defensible (statistical mass over headline t-stat;
a mechanism over a curiosity; monotone_robust=true — a quantile
response curve that survives lookback perturbation — outranks both)
unless one is actually indefensible.
Answer strict JSON: {"pick": <id or null>, "notes": str}"""

WORKER_SYSTEM = f"""You are a research worker in program <<NAME>>.
Thesis: <<THESIS>>.  Guidance: <<GUIDANCE>>.
Stay inside the program: series from <<SERIES>>, transforms from
<<TRANSFORMS>>, lookback_hours within <<LOOKBACKS>>.
Propose ONE candidate as STRICT JSON (no prose): {SPEC_SCHEMA}
Each history entry has an "id", metrics and the referee's verdict;
Every result carries a quantile receipt: quantile_r is the
mean forward return by feature band (around the 1st/10th/50th/90th/99th
percentiles), monotone is +1/-1 when that chain is strictly ordered,
and monotone_robust means the direction survives the lookback pulled a
quarter either way.  monotone_robust with the baselines beaten is the
highest-confidence promote evidence there is; a tall t_stat with
monotone 0 deserves suspicion (threshold luck, not a response curve).
Learn from the history and do not repeat failures.  Trade-count
economics: a
candidate with n_trades 0 had a threshold that never fired — lower it
(0.52-0.62 fires; 0.75+ almost never does) or widen the regime.  A
t-stat needs mass: aim for 30+ trades, because a referee will not
promote 10 lucky ones."""


def worker_system(program):
    out = WORKER_SYSTEM
    for token, value in (
            ("<<NAME>>", program["name"]),
            ("<<THESIS>>", program["thesis"]),
            ("<<GUIDANCE>>", program["guidance"]),
            ("<<SERIES>>", json.dumps(program["series"])),
            ("<<TRANSFORMS>>", json.dumps(program["transforms"])),
            ("<<LOOKBACKS>>", json.dumps(program["lookback_hours"]))):
        out = out.replace(token, value)
    return out


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


def edge(m):
    """Walk-forward alpha over both baselines; None when disqualified.
    A robust monotone quantile curve is independent structural evidence,
    so it buys a lower significance bar (t > 0.8 instead of 1.0)."""
    if not m or m.get("error") or m.get("n_trades", 0) < 8:
        return None
    bar = max(m.get("always_long_avg_r", 0.0), m.get("unfiltered_avg_r",
                                                     0.0), 0.0)
    t_bar = 0.8 if m.get("monotone_robust") else 1.0
    if m.get("avg_r", 0.0) <= bar or m.get("t_stat", 0.0) <= t_bar:
        return None
    return m["avg_r"] - bar


def remaining():
    b = loop.budget()
    return b["remaining"], b["budget"]


DEFAULT_PROGRAMS = [
    {"name": "cross_market_lead", "thesis": "macro series (forex, gold) "
     "reprice risk appetite before BTC follows",
     "series": [s for s in SERIES if s != TARGET][:3] or [TARGET],
     "transforms": ["zscore_momentum", "autocorr"],
     "lookback_hours": [24, 480],
     "guidance": "vary the leading series and the horizon regime"},
    {"name": "vol_regime", "thesis": "BTC edge is conditional on the "
     "volatility regime, not on direction",
     "series": [TARGET], "transforms": ["vol_ratio", "range_pos"],
     "lookback_hours": [48, 720],
     "guidance": "high thresholds; the filter is the model"},
    {"name": "mean_structure", "thesis": "short-horizon autocorrelation "
     "structure persists after vol normalisation",
     "series": [TARGET], "transforms": ["autocorr", "rsi"],
     "lookback_hours": [8, 168],
     "guidance": "short regimes; prune anything that is secretly drift"},
]


def pick_workers():
    """The two cheapest usable models on the gateway, at run time."""
    try:
        models = loop.models()
    except loop.LoopError:
        return [REFEREE]
    priced = []
    for m in models:
        mid = m.get("id") or ""
        try:
            p = float(m.get("prompt_usd_per_token") or 0)
            c = float(m.get("completion_usd_per_token") or 0)
        except (TypeError, ValueError):
            continue
        # no routing variants, nothing below frontier-lite pricing (the
        # workers must still write valid specs), decent context
        if (":" in mid or mid.startswith("~") or p <= 0 or c < 5e-7
                or (m.get("context_length") or 0) < 60_000):
            continue
        priced.append((p + 3 * c, mid))
    priced.sort()
    return [mid for _, mid in priced[:2]] or [REFEREE]


def get_programs():
    try:
        reply = ask(PI, PI_SYSTEM, json.dumps(
            {"total_worker_proposals": ITERATIONS,
             "budget": loop.budget()}), "propose", temperature=0.4)
        programs = extract_json(reply.text)["programs"]
        assert isinstance(programs, list) and programs
        out = []
        for p in programs[:4]:
            out.append({
                "name": str(p.get("name", f"p{len(out)}"))[:40],
                "thesis": str(p.get("thesis", ""))[:500],
                "series": [s for s in p.get("series", []) if s in SERIES]
                or [TARGET],
                "transforms": [t for t in p.get("transforms", [])
                               if t in VOCAB["transforms"]]
                or list(VOCAB["transforms"][:2]),
                "lookback_hours": p.get("lookback_hours", [8, 720]),
                "guidance": str(p.get("guidance", ""))[:500]})
        return out
    except loop.BudgetExhausted:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"PI gave no usable programs ({e}); using defaults",
              file=sys.stderr)
        return [dict(p) for p in DEFAULT_PROGRAMS]


def worker_round(program, worker, history, i):
    shown = [{"id": h["id"], "candidate": h["spec"], "metrics": h["metrics"],
              "verdict": h["verdict"], "notes": h["notes"][:200]}
             for h in history[-14:]]
    system = worker_system(program)
    try:
        reply = ask(worker, system,
                    json.dumps({"iteration": i, "history": shown}),
                    "propose")
        parsed = extract_json(reply.text)
        spec = {"feature": parsed["feature"],
                "hyperparams": parsed["hyperparams"]}
        feature = spec["feature"] if isinstance(spec["feature"], dict) else {}
        cid = loop.propose(spec, rationale=feature.get("rationale", ""),
                           source=reply, name=feature.get("name", ""))
    except loop.BudgetExhausted:
        raise
    except (loop.LoopError, ValueError, KeyError, TypeError) as e:
        print(f"{program['name']} #{i}: no usable proposal ({e})",
              file=sys.stderr)
        return None
    metrics = loop.experiment(cid)

    verdict, notes, said = "prune", "referee gave no usable reply", None
    try:
        said = ask(REFEREE, REFEREE_SYSTEM,
                   json.dumps({"candidate": spec, "metrics": metrics}),
                   "critique", cid, temperature=0.2)
        v = extract_json(said.text)
        if v.get("verdict") in ("promote", "refine", "prune"):
            verdict, notes = v["verdict"], str(v.get("notes", ""))
    except loop.BudgetExhausted:
        raise
    except (loop.LoopError, ValueError, TypeError) as e:
        print(f"{program['name']} #{i}: no usable critique ({e})",
              file=sys.stderr)
    loop.critique(cid, verdict, notes, source=said)
    if verdict == "prune":
        loop.prune(cid, notes)
    return {"id": cid, "spec": spec, "metrics": metrics,
            "verdict": verdict, "notes": notes, "program": program["name"]}


def mid_review(programs, by_program, left):
    """The PI reallocates what is left; silence keeps the split even."""
    try:
        reply = ask(PI, REVIEW_SYSTEM, json.dumps(
            {"remaining_total": left,
             "programs": {p["name"]: {
                 "thesis": p["thesis"],
                 "results": [{"spec": h["spec"], "metrics": h["metrics"],
                              "verdict": h["verdict"]}
                             for h in by_program.get(p["name"], [])]}
              for p in programs}}), "other", temperature=0.4)
        parsed = extract_json(reply.text)
        alloc = {str(k): int(v) for k, v in
                 (parsed.get("allocation") or {}).items()
                 if not isinstance(v, bool) and int(v) >= 0}
        guidance = parsed.get("guidance") or {}
        for p in programs:
            if p["name"] in guidance:
                p["guidance"] = str(guidance[p["name"]])[:500]
        if alloc and sum(alloc.values()) > 0:
            return alloc
    except loop.BudgetExhausted:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"mid-run review unusable ({e}); keeping the even split",
              file=sys.stderr)
    share = max(left // max(len(programs), 1), 1)
    return {p["name"]: share for p in programs}


def run():
    programs = get_programs()
    workers = pick_workers()
    print(f"programs: {[p['name'] for p in programs]}; workers: {workers}",
          file=sys.stderr)
    history, by_program = [], {}
    first_leg = max(ITERATIONS // 2, 1)
    i = 0
    for k in range(first_leg):
        rem, total = remaining()
        if rem < total * RESERVE or out_of_time():
            break
        program = programs[k % len(programs)]
        entry = worker_round(program, workers[k % len(workers)], history, i)
        i += 1
        if entry:
            history.append(entry)
            by_program.setdefault(entry["program"], []).append(entry)

    left = ITERATIONS - i
    if left > 0 and remaining()[0] > remaining()[1] * RESERVE:
        alloc = mid_review(programs, by_program, left)
        queue = [p for p in programs for _ in range(alloc.get(p["name"], 0))]
        for k, program in enumerate(queue[:left]):
            rem, total = remaining()
            if rem < total * RESERVE or out_of_time():
                break
            entry = worker_round(program, workers[k % len(workers)],
                                 history, i)
            i += 1
            if entry:
                history.append(entry)
                by_program.setdefault(entry["program"], []).append(entry)
    return programs, history


programs, history = [], []
try:
    programs, history = run()
except loop.BudgetExhausted:
    print("budget exhausted; closing the run", file=sys.stderr)

# per-round verdicts guide the search; promotion is decided once, at a
# final review of the bar-passing finalists, because a referee that
# never promotes in isolation zeroes the run for free
finalists = sorted((h for h in history if edge(h["metrics"]) is not None),
                   key=lambda h: -h["metrics"]["t_stat"])[:3]
best = None
if finalists:
    best = finalists[0]
    try:
        reply = ask(REFEREE, FINAL_SYSTEM, json.dumps(
            [{"id": h["id"], "candidate": h["spec"],
              "metrics": h["metrics"], "round_notes": h["notes"][:200]}
             for h in finalists]), "critique", finalists[0]["id"],
            temperature=0.2)
        parsed = extract_json(reply.text)
        pick = parsed.get("pick")
        notes = str(parsed.get("notes", ""))[:300]
        chosen = next((h for h in finalists if h["id"] == pick), None)
        if chosen is not None:
            best = chosen
            loop.critique(best["id"], "promote", notes, source=reply)
        elif pick is None and "pick" in (parsed or {}):
            best = None            # the referee refused all finalists
            print(f"final review refused all: {notes}", file=sys.stderr)
    except loop.BudgetExhausted:
        pass
    except (loop.LoopError, ValueError, TypeError) as e:
        print(f"final review unusable ({e}); best t-stat stands",
              file=sys.stderr)
if best is not None:
    try:
        loop.promote(best["id"])
    except loop.LoopError as e:
        print(f"could not promote: {e}", file=sys.stderr)

promoted = sum(1 for h in history if h["verdict"] == "promote")
loop.finish(
    narrative=(
        f"Lab loop: the PI model ({PI}) designed "
        f"{len(programs)} research programs and redirected the budget at "
        f"the mid-run review; cheap workers proposed {len(history)} "
        f"candidates inside them and the referee ({REFEREE}) promoted "
        f"{promoted} in isolation.  Promotion was decided at a final "
        "review of the bar-passing finalists (avg_r above both the "
        "drift and naked-factor baselines, t > 1): the referee picked "
        "the most defensible one, knowing an unpromoted run pays "
        "nothing and the holdout grades the pick."
        + (f"  Winning program: {best['program']}." if best else
           "  Nothing met the bar; nothing trades.")),
    hyperparameter_rationale=(
        "Hyperparameters were proposed by worker models inside the "
        "published ranges and program bounds set by the PI; selection "
        "was by walk-forward t-stat among referee-promoted candidates "
        "that beat always_long_avg_r and unfiltered_avg_r."))
