# MIT License
# Copyright (c) 2024 MANTIS
"""Aggregations that power the LOOP dashboard.

The ledger is already the log: every run, every model call, every
receipt and every lifecycle event is a row the gateway wrote while it
worked.  This module only *reads* — one call returns a JSON-able
snapshot rich enough to chart the whole challenge:

* totals — runs by status, promotions, active hotkeys, spend split by
  kind, run durations;
* a per-day timeseries (runs, completions, promotions, spend, unique
  hotkeys) for the last ``days`` days;
* a model leaderboard (spend and calls per SayGM model slug);
* a per-miner rollup (runs, promotions, spend, last seen, and — when
  the auditor cache is on this host — the current audit multiplier);
* a failure histogram (how runs die: timeouts, budget, errors).

Served live at ``GET /v1/network/dashboard`` on the gateway, or
exported offline:  ``python -m autoresearch.dashboard gw.db
[--audit audit.db] [--days 30]``.  The public 24h-round dashboard (a
static site with every evaluated submission's source) is
``round_site.py``; ``--site DIR --datalog DB`` renders it from here.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .credits import CreditLedger, _from_micro

MAX_DAYS = 365


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def challenge_dashboard(ledger: CreditLedger, audit_db: str = "",
                        days: int = 30) -> dict:
    """One snapshot of the whole challenge, computed from the ledger
    (and the auditor cache, when its path is given)."""
    days = max(1, min(int(days), MAX_DAYS))
    c = ledger._conn
    now = time.time()
    since = now - days * 86400.0

    # ---- runs ----------------------------------------------------------
    rows = c.execute(
        "SELECT run_id, hotkey, status, detail, dossier IS NOT NULL, "
        "created_ts, ended_ts FROM sealed_runs").fetchall()
    by_status: Dict[str, int] = {}
    by_day: Dict[str, dict] = {}
    miners: Dict[str, dict] = {}
    failures: Dict[str, int] = {}
    durations: List[float] = []
    promotions = 0
    for _rid, hk, status, detail, has_dossier, created, ended in rows:
        by_status[status] = by_status.get(status, 0) + 1
        promoted = bool(has_dossier)
        promotions += promoted
        if ended and created:
            durations.append(float(ended) - float(created))
        m = miners.setdefault(hk, {"runs": 0, "done": 0, "promotions": 0,
                                   "credits": 0.0, "last_ts": 0.0})
        m["runs"] += 1
        m["done"] += status == "done"
        m["promotions"] += promoted
        m["last_ts"] = max(m["last_ts"], float(ended or created or 0.0))
        if status == "failed":
            try:
                reason = str(json.loads(detail or "{}").get(
                    "reason", "unknown"))[:80]
            except ValueError:
                reason = "unknown"
            failures[reason] = failures.get(reason, 0) + 1
        if created and created >= since:
            d = by_day.setdefault(_day(created), {
                "runs": 0, "done": 0, "failed": 0, "promotions": 0,
                "credits": 0.0, "hotkeys": set()})
            d["runs"] += 1
            d["done"] += status == "done"
            d["failed"] += status == "failed"
            d["promotions"] += promoted
            d["hotkeys"].add(hk)

    # ---- spend ---------------------------------------------------------
    spend_by_kind = {k: _from_micro(v) for k, v in c.execute(
        "SELECT kind, COALESCE(SUM(credits),0) FROM spends GROUP BY kind")}
    (llm_calls,) = c.execute(
        "SELECT COUNT(*) FROM spends WHERE kind='llm'").fetchone()
    models = [{"model": r[0], "credits": _from_micro(r[1]), "calls": r[2]}
              for r in c.execute(
                  "SELECT item, COALESCE(SUM(credits),0), COUNT(*) "
                  "FROM spends WHERE kind='llm' GROUP BY item "
                  "ORDER BY 2 DESC LIMIT 25")]
    for hk, micro in c.execute(
            "SELECT hotkey, COALESCE(SUM(credits),0) FROM spends "
            "GROUP BY hotkey"):
        if hk in miners:
            miners[hk]["credits"] = _from_micro(micro)
    for ts, micro in c.execute(
            "SELECT ts, credits FROM spends WHERE ts >= ?", (since,)):
        d = by_day.get(_day(ts))
        if d is not None:
            d["credits"] += _from_micro(micro)

    # ---- audit (only where the auditor cache lives) --------------------
    audit: Optional[dict] = None
    if audit_db:
        try:
            from .audit import AuditorPanel
            panel = AuditorPanel(cache_db=audit_db, judges=[])
            mult = panel.multiplier_map()
            bands: Dict[str, int] = {}
            for hk, v in mult.items():
                bands[f"{v:g}"] = bands.get(f"{v:g}", 0) + 1
                if hk in miners:
                    miners[hk]["audit_multiplier"] = v
            audit = {"audited_hotkeys": len(mult), "bands": bands}
        except Exception:  # noqa: BLE001 — the dashboard never breaks
            audit = {"error": "auditor cache unreadable"}

    durations.sort()
    pct = (lambda p: round(durations[int(p * (len(durations) - 1))], 1)
           if durations else 0.0)
    return {
        "generated_ts": now,
        "window_days": days,
        "totals": {
            "runs": len(rows),
            "by_status": by_status,
            "promotions": promotions,
            "hotkeys": len(miners),
            "credits_spent": round(sum(spend_by_kind.values()), 6),
            "spend_by_kind": spend_by_kind,
            "llm_calls": llm_calls,
            "run_seconds": {"p50": pct(0.50), "p90": pct(0.90),
                            "max": pct(1.0)},
        },
        "days": [{"date": day, "runs": rec["runs"], "done": rec["done"],
                  "failed": rec["failed"],
                  "promotions": rec["promotions"],
                  "credits": round(rec["credits"], 6),
                  "hotkeys": len(rec["hotkeys"])}
                 for day, rec in sorted(by_day.items())],
        "models": models,
        "miners": {hk: {**m, "credits": round(m["credits"], 6)}
                   for hk, m in sorted(miners.items())},
        "failures": failures,
        "audit": audit,
    }


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Export the LOOP dashboard snapshot as JSON")
    p.add_argument("ledger_db")
    p.add_argument("--audit", default="", help="auditor cache db")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--site", default="",
                   help="also render the round dashboard site into this "
                        "directory (needs --datalog)")
    p.add_argument("--html", default="",
                   help="also render the round dashboard as one page")
    p.add_argument("--datalog", default="", help="main datalog db")
    a = p.parse_args(argv)
    ledger = CreditLedger(a.ledger_db)
    if a.site or a.html:
        from .round_site import render_rounds_html, render_rounds_site
        panel = None
        if a.audit:
            from .audit import AuditorPanel
            panel = AuditorPanel(cache_db=a.audit)
        if a.site:
            changed = render_rounds_site(ledger, a.datalog, a.site, panel)
            print(f"round dashboard site in {a.site} "
                  f"({len(changed)} files changed)")
        if a.html:
            render_rounds_html(ledger, a.datalog, panel, out_path=a.html)
            print(f"round dashboard page written to {a.html}")
    print(json.dumps(challenge_dashboard(ledger, a.audit, a.days),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
