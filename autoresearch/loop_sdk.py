# MIT License
# Copyright (c) 2024 MANTIS
"""Everything a LOOP pipeline can do.

A pipeline is a Python program that researches a trading model.  The
gateway runs it sealed — no network, no files, nothing but this module —
so these calls are the pipeline's whole world, and the gateway's record
of them is the research trace.  Nothing here is a claim the pipeline
makes about itself: the gateway makes the model call, runs the
experiment and writes down what happened.

    import loop_sdk as loop

    data = loop.data()                       # every market series
    btc = data["series"][data["target"]]["closes"]
    reply = loop.llm("kimi-k3.1-pro-tee", messages, step="propose")
    cand = loop.propose(spec, rationale="...", source=reply)
    metrics = loop.experiment(cand)                 # the protocol's harness
    loop.critique(cand, "promote", notes="...")
    loop.promote(cand)                              # the last one promoted
                                                    # is the model that trades
    loop.finish(narrative="...", hyperparameter_rationale="...")

A model is a spec, not code: ``{"feature": {"transform",
"lookback_hours"}, "hyperparams": {...}}`` — see ``loop.vocabulary()``.
The gateway evaluates it; a pipeline never reports a number.

Money: ``loop.llm`` and ``loop.experiment`` spend the run's budget, and
so does the pipeline's own running time.  When the budget is gone they
raise ``BudgetExhausted``; let it propagate and the run ends normally
with whatever has been promoted.

Pauses: the gateway may pause a run (an operator stopping the
validator).  Every call then raises ``Paused``; let it propagate and
the run is parked, to be resumed later as a fresh process of this same
program.  ``loop.checkpoint(state)`` saves a JSON state (one slot,
overwritten) and ``loop.restore()`` returns the last one — checkpoint
at every iteration and start by restoring, and a pause costs you at
most one iteration.  Candidate ids stay valid across a resume.

This file is copied into the sandbox as-is.  Stdlib only.
"""

import json

__all__ = ["BudgetExhausted", "LoopError", "Paused", "ProviderFailed",
           "Rejected", "Reply", "args", "budget", "checkpoint",
           "critique", "data", "experiment", "finish", "llm", "models",
           "promote", "propose", "prune", "restore", "vocabulary"]


class LoopError(Exception):
    """The gateway could not do what was asked."""


class Rejected(LoopError):
    """The request breaks a rule (a spec outside the vocabulary, a
    candidate that does not exist...).  The message says which."""


class BudgetExhausted(LoopError):
    """The run has no credits left."""


class ProviderFailed(LoopError):
    """The model provider failed; nothing was charged."""


class Paused(LoopError):
    """The gateway is pausing this run.  Save state with
    ``checkpoint`` (still allowed) and exit; the run resumes later."""


_ERRORS = {"rejected": Rejected, "budget": BudgetExhausted,
           "provider": ProviderFailed, "paused": Paused}


class Reply:
    """One model call: what came back and the receipt for it."""

    def __init__(self, d):
        self.text = d.get("text")           # None if the model said nothing
        self.receipt_id = d["receipt_id"]
        self.credits = d.get("credits", 0.0)
        self.model = d.get("model", "")
        self.finish_reason = d.get("finish_reason")

    def __repr__(self):
        return f"Reply({self.model!r}, {self.credits:.6f} cr)"


_channel = None     # (reader, writer), set by the sandbox before the
_seq = 0            # pipeline starts
_cache = {}


def _attach(reader, writer):
    global _channel
    _channel = (reader, writer)


def _call(op, **params):
    global _seq
    if _channel is None:
        raise LoopError("not running inside a LOOP run")
    reader, writer = _channel
    _seq += 1
    writer.write(json.dumps({"id": _seq, "op": op, **params}) + "\n")
    writer.flush()
    line = reader.readline()
    if not line:
        raise LoopError("the run was closed")
    msg = json.loads(line)
    if msg.get("ok"):
        return msg.get("result")
    raise _ERRORS.get(msg.get("kind"), LoopError)(msg.get("error", ""))


def _once(op):
    if op not in _cache:
        _cache[op] = _call(op)
    return _cache[op]


def _receipt(source):
    return source.receipt_id if isinstance(source, Reply) else source


# ------------------------------------------------------------- read-only

def args():
    """The JSON arguments this run was submitted with (public)."""
    return _once("args")


def data():
    """The research window: ``{"target", "series": {name:
    {"interval_hours", "ts", "closes"}}}`` — the hourly target the model
    trades plus years of auxiliary market series (daily BTC, forex
    majors, ETH, gold...), all cut at the same moment.  The most recent
    target bars are held back; the promoted model is scored on them
    after the run."""
    return _once("data")


def models():
    """Every model the gateway can call — SayGM (SN28) ids with pricing —
    so a pipeline can pick its own models (one, several, or per step) at
    run time."""
    return _once("models")


def vocabulary():
    """What a model spec may contain."""
    return _once("vocabulary")


def budget():
    """``{"budget", "spent", "remaining"}`` in credits, right now, and
    ``seconds_left`` on this run's wall clock."""
    return _call("budget")


# ---------------------------------------------------------------- metered

def llm(model, messages, step="other", candidate=None, **params):
    """Call any SayGM (SN28) model.  ``step`` says what the call is for —
    "propose", "critique" or "other" — and ``candidate`` which candidate,
    if one exists yet.  Text in, text out: ``messages`` is a list of
    ``{"role", "content"}`` with string content, and the only extra
    keyword arguments are sampling ones (``temperature``, ``max_tokens``,
    ``top_p``, ``seed``, ``stop``, ``reasoning``, ``response_format``...).
    Tools, web plugins and ``:online`` models are refused.  The request
    and the reply are published with the run."""
    return Reply(_call("llm", model=model, messages=messages, step=step,
                       candidate=candidate, params=params))


def experiment(candidate):
    """Evaluate a candidate walk-forward on the research window.  Returns
    its metrics; a spec that cannot be evaluated returns ``{"error"}``.
    Asking again for the same candidate is free and returns the same."""
    return _call("experiment", candidate=candidate)


# ---------------------------------------------------------------- the tree

def propose(spec, rationale="", refines=None, source=None, name=""):
    """Put a candidate model on the table and get its id.  ``refines`` is
    the id of the candidate it builds on; ``source`` the ``Reply`` (or
    receipt id) of the model call it came from, if it came from one."""
    return _call("propose", spec=spec, rationale=rationale, refines=refines,
                 source=_receipt(source), name=name)


def critique(candidate, verdict, notes="", source=None):
    """Record a judgement of an evaluated candidate: "promote", "refine"
    or "prune"."""
    return _call("critique", candidate=candidate, verdict=verdict,
                 notes=notes, source=_receipt(source))


def promote(candidate):
    """Make an evaluated candidate the run's model.  The last candidate
    promoted is the one that trades."""
    return _call("promote", candidate=candidate)


def prune(candidate, reason=""):
    """Drop a candidate."""
    return _call("prune", candidate=candidate, reason=reason)


def finish(narrative="", hyperparameter_rationale=""):
    """The case this run makes to the validator, in the pipeline's own
    words.  Call it last; calling it again replaces what was said."""
    return _call("finish", narrative=narrative,
                 hyperparameter_rationale=hyperparameter_rationale)


# ----------------------------------------------------------------- pauses

def checkpoint(state):
    """Save this run's state (any JSON-serialisable object, one slot,
    overwritten).  It is the pipeline's own memory across a pause —
    nothing new can enter through it, so checkpoint freely.  Allowed
    even while the gateway is pausing the run."""
    return _call("checkpoint", state=state)


def restore():
    """The last checkpointed state, or ``None`` on a fresh run.  Call
    it first: if it returns something, this process is a resume."""
    return _call("restore")
