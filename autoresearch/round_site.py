# MIT License
# Copyright (c) 2024 MANTIS
"""The LOOP round dashboard: a static site the auditor redraws after
every 24-hour round.

Not a service.  ``render_rounds_site`` writes a directory —

    index.html               overview: live round clock, headline
                             numbers, round history, models in use,
                             miners, links to every round
    rounds/<YYYY-MM-DD>.html one round: every entry with its sandbox
                             screen, audit verdict, research stats,
                             narrative and full source

— that any static host serves as is (an R2 bucket, nginx, a CDN).
While a round is open its entries show their status only: the code,
the research and the verdicts appear when the round closes, so nobody
copies a pipeline before it has been judged.  A finished round's page
is byte-stable: re-rendering rewrites only what changed, so syncing the
directory uploads only what is new.
``render_rounds_html`` puts everything on one page (previews, tests).

Every page is self-contained — no CDN, no web fonts, no third-party
script — and locks itself down: a Content-Security-Policy meta tag
forbids every network fetch and allows exactly one script, by hash.
Every miner-authored string is key-redacted (``submission.redact``)
when the data is gathered and HTML-escaped when it is written; the
sealed-key field of a submission is never read at all.
"""

from __future__ import annotations

import base64
import builtins
import datetime as _dt
import hashlib
import html as H
import io
import json
import keyword
import os
import re
import sqlite3
import time
import tokenize
from typing import Any, Dict, List, Optional

from .credits import CreditLedger
from .submission import redact

STATUS_ORDER = ("audited", "failed", "blocked", "refused", "unscreened",
                "running")
STATUS_HUES = {"audited": "#22d3a5", "failed": "#f2555a",
               "blocked": "#f59e0b", "refused": "#8a94a8",
               "unscreened": "#a78bfa", "running": "#38bdf8"}
HISTORY_ROUNDS = 30                 # bars in the round-history chart
NEVER_RAN = ("refused", "blocked", "unscreened")
_ROUND_ID = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ------------------------------------------------------------------ data

def _scrub(x: Any) -> Any:
    if isinstance(x, str):
        return redact(x)
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items()}
    return x


def _entry(ledger: CreditLedger, e: dict, body: Optional[str],
           models: Dict[str, dict], closed: bool = True) -> dict:
    """One entry as the page shows it.  Until its round closes the entry
    is ``held``: no code, no research, no verdict text — only why a
    pipeline that never ran did not."""
    detail = _scrub(e.get("detail") or {})
    if not closed and e["status"] not in NEVER_RAN:
        detail = {k: v for k, v in detail.items()
                  if k in ("funding", "screen_model")}
    out: Dict[str, Any] = {
        "hotkey": e["hotkey"], "sha256": e["sha256"],
        "status": e["status"], "detail": detail, "held": not closed,
        "run_id": e.get("run_id"), "source": None, "args": None,
        "budget": None, "seconds": None, "spent": None, "narrative": "",
        "promoted": None, "models": {},
        "stats": {"candidates": 0, "experiments": 0, "llm_calls": 0}}
    if body and closed:
        try:
            sub = json.loads(body)
            # source, args and budget only: key_sealed is never read
            out["source"] = redact(str(sub.get("source", "")))
            out["args"] = redact(json.dumps(sub.get("args", {}),
                                            sort_keys=True))
            budget = sub.get("budget_credits")
            if isinstance(budget, (int, float)):
                out["budget"] = float(budget)
        except (ValueError, TypeError, AttributeError):
            out["source"] = redact(body[:4000])
    run = ledger.sealed_run(e["run_id"]) if e.get("run_id") else None
    if run is None:
        return out
    out["seconds"] = (run.get("detail") or {}).get("seconds")
    try:
        out["spent"] = ledger.run_info(e["run_id"]).spent_credits
    except Exception:  # noqa: BLE001
        pass
    dossier = run.get("dossier") or {}
    if closed and isinstance(dossier, dict) and dossier:
        out["narrative"] = redact(str(dossier.get("narrative", "")))[:4000]
        feats = dossier.get("features") or []
        f = feats[0] if feats and isinstance(feats[0], dict) else None
        if f is not None:
            inputs = f.get("inputs") or []
            lb = f.get("lookback_hours")
            out["promoted"] = {
                "transform": redact(str(f.get("transform", "")))[:40],
                "lookback_hours": float(lb) if isinstance(
                    lb, (int, float)) else None,
                "series": redact(str(inputs[0]) if inputs else "")[:40],
                "name": redact(str(f.get("name", "")))[:90]}
    for ev in ledger.events_for_run(e["run_id"]):
        kind = ev.get("kind")
        if kind == "hypothesis.proposed":
            out["stats"]["candidates"] += 1
        elif kind == "experiment.result":
            out["stats"]["experiments"] += 1
    for r in ledger.receipts(run_id=e["run_id"]):
        if r.get("kind") != "llm":
            continue
        m = redact(str(r.get("item") or "?"))[:80]
        out["models"][m] = out["models"].get(m, 0) + 1
        out["stats"]["llm_calls"] += 1
        agg = models.setdefault(m, {"calls": 0, "credits": 0.0})
        agg["calls"] += 1
        agg["credits"] += float(r.get("credits") or 0.0)
    return out


def rounds_data(ledger: CreditLedger, datalog_db: str, limit: int = 60,
                panel: Any = None) -> dict:
    """Everything the round dashboard shows, as plain data, newest
    round first.  ``panel`` (the auditor) adds the verdict each hotkey
    is paid on right now."""
    conn = None
    if datalog_db and os.path.exists(datalog_db):
        try:
            conn = sqlite3.connect(f"file:{datalog_db}?mode=ro", uri=True)
        except sqlite3.Error:
            conn = None

    def body_of(hk: str, sha: str) -> Optional[str]:
        body = ledger.inbox_submission(hk, sha)
        if body is not None or conn is None:
            return body
        try:
            row = conn.execute(
                "SELECT submission FROM loop_submissions "
                "WHERE hotkey=? AND sha256=?", (hk, sha)).fetchone()
        except sqlite3.Error:
            return None
        return row[0] if row else None

    models: Dict[str, dict] = {}
    rounds: List[dict] = []
    try:
        for rnd in ledger.rounds(limit=limit):
            closed = rnd["finished_ts"] is not None
            rounds.append({
                **rnd, "detail": _scrub(rnd.get("detail") or {}),
                "entries": [_entry(ledger, e, body_of(e["hotkey"],
                                                      e["sha256"])
                                   if closed else None, models, closed)
                            for e in ledger.round_entries(rnd["round_id"])]})
    finally:
        if conn is not None:
            conn.close()
    verdicts: Dict[str, float] = {}
    if panel is not None:
        try:
            verdicts = {k: float(v) for k, v in panel.multiplier_map().items()}
        except Exception:  # noqa: BLE001 — the dashboard never breaks
            verdicts = {}
    return {"generated_ts": time.time(), "rounds": rounds,
            "models": models, "verdicts": verdicts}


# ------------------------------------------------------------- highlight

_BUILTINS = frozenset(n for n in dir(builtins) if not n.startswith("_"))
_STRINGISH = {tokenize.STRING} | {
    getattr(tokenize, t) for t in ("FSTRING_START", "FSTRING_MIDDLE",
                                   "FSTRING_END") if hasattr(tokenize, t)}


def highlight_python(source: str) -> str:
    """``source`` as escaped HTML with token classes, from the stdlib
    tokenizer.  Text between tokens is copied verbatim, so the visible
    text is exactly the source; anything the tokenizer refuses falls
    back to plain escaped text."""
    starts = [0]
    for line in source.split("\n"):
        starts.append(starts[-1] + len(line) + 1)

    def off(pos: tuple) -> int:
        return starts[pos[0] - 1] + pos[1]

    out: List[str] = []
    last, prev = 0, None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            t, s = tok.type, tok.string
            cls = None
            if t == tokenize.COMMENT:
                cls = "c"
            elif t in _STRINGISH:
                cls = "s"
            elif t == tokenize.NUMBER:
                cls = "n"
            elif t == tokenize.NAME:
                if keyword.iskeyword(s):
                    cls = "k"
                elif prev in ("def", "class"):
                    cls = "f"
                elif s in _BUILTINS:
                    cls = "b"
            if t == tokenize.NAME:
                prev = s
            elif t not in (tokenize.NL, tokenize.COMMENT):
                prev = None
            if cls is None:
                continue
            a, b = off(tok.start), off(tok.end)
            if a < last or b < a or b > len(source):
                continue
            out.append(H.escape(source[last:a]))
            out.append(f'<span class="{cls}">{H.escape(source[a:b])}</span>')
            last = b
    except (tokenize.TokenError, SyntaxError, ValueError, IndexError):
        return H.escape(source)
    out.append(H.escape(source[last:]))
    return "".join(out)


# ------------------------------------------------------------- the shell

_JS = """(function(){
function p(n){return (n<10?"0":"")+n;}
function tick(){
var now=new Date(),t=now.getTime();
var next=Date.UTC(now.getUTCFullYear(),now.getUTCMonth(),now.getUTCDate()+1);
var s=Math.max(0,Math.floor((next-t)/1000));
var c=document.getElementById("clock"),d=document.getElementById("countdown");
if(c)c.textContent=p(now.getUTCHours())+":"+p(now.getUTCMinutes())+":"+p(now.getUTCSeconds());
if(d)d.textContent=p(Math.floor(s/3600))+":"+p(Math.floor(s%3600/60))+":"+p(s%60);
}
tick();setInterval(tick,1000);
var q=document.getElementById("q");
if(q)q.addEventListener("input",function(){
var v=q.value.trim().toLowerCase();
document.querySelectorAll("[data-hk]").forEach(function(el){
el.classList.toggle("hidden",!!v&&el.getAttribute("data-hk").toLowerCase().indexOf(v)<0);
});
});
})();"""

_CSP = ("default-src 'none'; script-src 'sha256-"
        + base64.b64encode(hashlib.sha256(_JS.encode()).digest()).decode()
        + "'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; "
        "form-action 'none'")

_CSS = """
:root{--bg:#05070c;--panel:rgba(13,18,29,.84);--panel2:#0a0f19;--edge:#1a2334;
--edge2:#243049;--ink:#e6ebf4;--dim:#8792a8;--faint:#566179;--acc:#22d3a5;
--acc2:#38bdf8;--warn:#f59e0b;--bad:#f2555a;
--mono:ui-monospace,SFMono-Regular,"JetBrains Mono","Cascadia Code",Menlo,Consolas,monospace;
--sans:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
*{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;color:var(--ink);font:14px/1.55 var(--sans);
-webkit-font-smoothing:antialiased;min-height:100vh;background:
radial-gradient(1100px 520px at 10% -12%,rgba(34,211,165,.12),transparent 62%),
radial-gradient(900px 420px at 100% -6%,rgba(56,189,248,.10),transparent 60%),
linear-gradient(rgba(148,163,184,.04) 1px,transparent 1px) 0 0/100% 36px,
linear-gradient(90deg,rgba(148,163,184,.04) 1px,transparent 1px) 0 0/36px 100%,
var(--bg);background-attachment:fixed}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1280px;margin:0 auto;padding:0 28px}
.ribbon{background:repeating-linear-gradient(135deg,#f59e0b 0 14px,#d97706 14px 28px);
color:#1a1205;font:700 11.5px var(--mono);letter-spacing:.16em;
text-transform:uppercase;text-align:center;padding:6px 10px}
header.top{position:sticky;top:0;z-index:10;background:rgba(5,7,12,.76);
backdrop-filter:saturate(140%) blur(12px);-webkit-backdrop-filter:saturate(140%) blur(12px);
border-bottom:1px solid var(--edge)}
.topin{display:flex;align-items:center;justify-content:space-between;height:62px;gap:16px}
.brand{display:flex;align-items:center;gap:10px;font:600 14px var(--mono);letter-spacing:.14em}
.logo{width:22px;height:22px;border-radius:6px;
background:linear-gradient(135deg,var(--acc),var(--acc2));box-shadow:0 0 18px rgba(34,211,165,.55)}
.brand b{color:var(--acc);font-weight:700}.brand .sl{color:var(--faint)}
.tag{font:500 10.5px var(--mono);letter-spacing:.1em;color:var(--dim);
padding:3px 9px;border:1px solid var(--edge2);border-radius:99px;margin-left:8px}
.clocks{display:flex;gap:10px}
.clk{padding:5px 13px;border:1px solid var(--edge);border-radius:9px;
background:rgba(10,15,25,.75);text-align:right;min-width:118px}
.clk small{display:block;font:600 9px var(--mono);letter-spacing:.18em;
color:var(--faint);text-transform:uppercase}
.clk span{font:600 16px var(--mono);font-variant-numeric:tabular-nums;letter-spacing:.04em}
.clk.acc{border-color:rgba(34,211,165,.3)}
.clk.acc span{color:var(--acc);text-shadow:0 0 12px rgba(34,211,165,.5)}
.hero{display:flex;justify-content:space-between;align-items:flex-end;gap:28px;
flex-wrap:wrap;padding:38px 0 4px}
.hero h1{margin:0;font:750 32px/1.12 var(--sans);letter-spacing:-.015em}
.hero h1 em{font-style:normal;background:linear-gradient(90deg,var(--acc),var(--acc2));
-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{margin:10px 0 0;color:var(--dim);max-width:690px}
.side{display:flex;flex-direction:column;align-items:flex-end;gap:12px}
.pill{display:inline-flex;align-items:center;gap:9px;font:600 12px var(--mono);
padding:6px 13px;border-radius:99px;border:1px solid rgba(34,211,165,.35);
background:rgba(34,211,165,.07);color:var(--acc)}
.pill i{width:7px;height:7px;border-radius:50%;background:var(--acc);animation:pulse 2.2s infinite}
.pill.busy{color:var(--acc2);border-color:rgba(56,189,248,.35);background:rgba(56,189,248,.07)}
.pill.busy i{background:var(--acc2)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,211,165,.55)}
70%{box-shadow:0 0 0 10px rgba(34,211,165,0)}100%{box-shadow:0 0 0 0 rgba(34,211,165,0)}}
#q{width:300px;background:rgba(10,15,25,.88);border:1px solid var(--edge2);
color:var(--ink);padding:9px 14px;border-radius:9px;font:13px var(--mono);outline:none}
#q:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(34,211,165,.15)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:12px;margin:26px 0 14px}
.kpi{position:relative;overflow:hidden;padding:14px 16px 13px;border:1px solid var(--edge);
border-radius:12px;background:var(--panel)}
.kpi:before{content:"";position:absolute;left:0;right:0;top:0;height:2px;
background:linear-gradient(90deg,var(--hue,var(--acc)),transparent 85%)}
.kpi .n{font:700 25px/1.1 var(--mono);font-variant-numeric:tabular-nums}
.kpi .l{margin-top:5px;font:600 10px var(--mono);letter-spacing:.14em;color:var(--dim);
text-transform:uppercase}
.kpi svg{position:absolute;right:12px;top:15px;width:62px;height:22px;opacity:.9}
.panels{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(0,1fr);gap:14px;margin-bottom:14px}
.panel{border:1px solid var(--edge);border-radius:14px;background:var(--panel);padding:16px 18px 14px}
.ph{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:0 0 12px;flex-wrap:wrap}
.ph h3{margin:0;font:600 10.5px var(--mono);letter-spacing:.16em;color:var(--dim);text-transform:uppercase}
.legend{display:flex;gap:12px;flex-wrap:wrap;font:500 11px var(--mono);color:var(--dim)}
.legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px}
svg.chart{width:100%;height:auto;display:block}
svg.chart text{font:500 10px var(--mono);fill:#66718a}
svg.chart text.tot{fill:#c9d1df;font-weight:600}
.mrow{display:grid;grid-template-columns:minmax(0,1fr) 40%;gap:10px;align-items:center;margin:9px 0}
.mname{font:500 12px var(--mono);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mbar{position:relative;height:18px;border-radius:5px;background:rgba(148,163,184,.08);overflow:hidden}
.mbar i{position:absolute;left:0;top:0;bottom:0;border-radius:5px;
background:linear-gradient(90deg,rgba(56,189,248,.6),rgba(34,211,165,.9))}
.mbar b{position:absolute;right:7px;top:0;font:600 11px/18px var(--mono)}
.empty{color:var(--faint);font:12px var(--mono);padding:30px 0;text-align:center}
table.miners{width:100%;border-collapse:collapse}
table.miners th{text-align:left;font:600 10px var(--mono);letter-spacing:.14em;color:var(--faint);
text-transform:uppercase;padding:0 10px 9px;border-bottom:1px solid var(--edge)}
table.miners td{padding:10px;border-bottom:1px solid rgba(26,35,52,.7);font-size:13px}
table.miners tr:last-child td{border-bottom:0}
.mono{font-family:var(--mono)}.faint{color:var(--faint)}.dim{color:var(--dim)}
.ser{color:var(--acc2);font-family:var(--mono)}
.hk{font:500 12.5px var(--mono)}
.chip{display:inline-block;font:600 10.5px/1.6 var(--mono);letter-spacing:.06em;
padding:1px 9px;border-radius:99px;border:1px solid;text-transform:uppercase}
.cnt{font:600 11px var(--mono);padding:2px 8px;border-radius:6px;background:rgba(148,163,184,.07)}
.mult{font:700 13px var(--mono)}.mult.pass{color:var(--acc)}.mult.zero{color:var(--bad)}
.mult.part{color:var(--warn)}
h2.sect{font:600 11px var(--mono);letter-spacing:.18em;color:var(--dim);
text-transform:uppercase;margin:34px 0 12px}
section.round{border:1px solid var(--edge);border-radius:14px;background:var(--panel);
margin:0 0 18px;overflow:hidden}
.rhead{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;
padding:16px 20px 13px;flex-wrap:wrap}
.rhead h2{margin:0;font:700 18px var(--sans)}.rhead h2 span{color:var(--acc);font-family:var(--mono)}
.rmeta{margin-top:3px;font:12px var(--mono);color:var(--dim)}
.rcounts{display:flex;gap:6px;flex-wrap:wrap}
.mix{display:flex;height:4px;background:var(--edge)}.mix i{flex:1}
.grid{display:grid;grid-template-columns:150px 112px 62px minmax(0,1.2fr) 56px 66px minmax(0,1.6fr) 16px;
gap:12px;align-items:center;padding:11px 20px}
.ghead{font:600 10px var(--mono);letter-spacing:.14em;color:var(--faint);
text-transform:uppercase;border-bottom:1px solid var(--edge);padding-top:9px;padding-bottom:9px}
details.entry{border-bottom:1px solid rgba(26,35,52,.85)}details.entry:last-of-type{border-bottom:0}
details.entry>summary{list-style:none;cursor:pointer;font-size:13px}
details.entry>summary::-webkit-details-marker{display:none}
details.entry>summary:hover{background:rgba(34,211,165,.035)}
details.entry[open]>summary{background:rgba(34,211,165,.055)}
.chev{color:var(--faint);font:13px var(--mono);transition:transform .15s}
details.entry[open] .chev{transform:rotate(90deg);color:var(--acc)}
.note{color:var(--dim);font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ebody{padding:6px 20px 20px}
.cards3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:4px 0 12px}
.ic{border:1px solid var(--edge);border-radius:11px;background:var(--panel2);padding:12px 14px}
.ic h4{margin:0 0 8px;font:600 9.5px var(--mono);letter-spacing:.16em;color:var(--faint);text-transform:uppercase}
.big{font:700 16px var(--mono)}.ok{color:var(--acc)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.ic .sm{font:11.5px var(--mono);color:var(--dim);margin-top:2px;word-break:break-all}
.ic p{margin:8px 0 0;color:var(--dim);font-size:12.5px}
.ic ul{margin:8px 0 0;padding-left:18px;color:var(--dim);font-size:12.5px}
.kv{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;padding:2px 0;color:var(--dim)}
.kv b{color:var(--ink);font:600 12.5px var(--mono);text-align:right}
.narr{border-left:2px solid var(--acc);padding:9px 14px;margin:0 0 12px;color:#c3cad8;
font-size:13px;background:rgba(34,211,165,.045);border-radius:0 9px 9px 0;white-space:pre-wrap}
.code{border:1px solid var(--edge2);border-radius:11px;overflow:hidden;background:#070a10}
.codebar{display:flex;justify-content:space-between;gap:12px;padding:8px 14px;
background:#0c111b;border-bottom:1px solid var(--edge);font:12px var(--mono);flex-wrap:wrap}
.fname{color:var(--ink)}.fname:before{content:"";display:inline-block;width:8px;height:8px;
border-radius:50%;background:var(--acc);margin-right:8px;box-shadow:0 0 8px var(--acc)}
.cmeta{color:var(--faint)}
.args{padding:7px 14px;border-bottom:1px solid var(--edge);font:12px var(--mono);
color:var(--faint);overflow-x:auto;white-space:nowrap}
.args code{color:#c3cad8}
.codewrap{display:flex;max-height:560px;overflow:auto}
.codewrap pre{margin:0;font:12.5px/1.6 var(--mono);tab-size:4}
pre.gut{position:sticky;left:0;padding:12px 12px 12px 14px;text-align:right;color:#3b4660;
background:#070a10;border-right:1px solid var(--edge);user-select:none;min-width:54px}
pre.src{padding:12px 18px;color:#d6deeb;flex:1}
.src .k{color:#c792ea}.src .s{color:#addb67}.src .c{color:#637777;font-style:italic}
.src .n{color:#f78c6c}.src .f{color:#82aaff}.src .b{color:#ffcb6b}
.nocode{color:var(--faint);font:12px var(--mono);padding:14px;border:1px dashed var(--edge2);border-radius:10px}
.rlist{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.rcard{display:block;border:1px solid var(--edge);border-radius:12px;background:var(--panel);
overflow:hidden;color:var(--ink)}
.rcard:hover{border-color:rgba(34,211,165,.6);text-decoration:none}
.rc1{display:flex;justify-content:space-between;align-items:baseline;padding:12px 14px 10px;font:13px var(--mono)}
.rc1 b{font-size:15px}.rc2{display:flex;gap:6px;flex-wrap:wrap;padding:10px 14px 12px}
.back{display:inline-block;margin:28px 0 14px;font:12px var(--mono)}
footer{margin:40px 0 50px;padding-top:18px;border-top:1px solid var(--edge);color:var(--faint);
font:11.5px var(--mono);display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}
.hidden{display:none!important}
@media (max-width:980px){.panels{grid-template-columns:1fr}.cards3{grid-template-columns:1fr}
.grid{grid-template-columns:120px 96px 52px minmax(0,1fr) 16px}
.grid>:nth-child(5),.grid>:nth-child(6),.grid>:nth-child(7){display:none}
.tag,.clk:first-child{display:none}#q{width:100%}}
"""


def _esc(x: Any) -> str:
    return H.escape("" if x is None else str(x), quote=True)


def _page(title: str, body: str, banner: str = "") -> str:
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">'
            '<meta name="referrer" content="no-referrer">'
            '<meta name="viewport" content="width=device-width, '
            'initial-scale=1">'
            + ('<meta name="robots" content="noindex,nofollow">'
               if banner else "")
            + f'<title>{_esc(title)}</title><style>{_CSS}</style>'
            '</head><body>')
    ribbon = f'<div class="ribbon">{_esc(banner)}</div>' if banner else ""
    return head + ribbon + body + f"<script>{_JS}</script></body></html>"


def _utc(ts: Optional[float], fmt: str = "%Y-%m-%d %H:%M UTC") -> str:
    return time.strftime(fmt, time.gmtime(ts)) if ts else ""


def _secs(s: Any) -> str:
    if not isinstance(s, (int, float)):
        return "—"
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"


def _credits(c: Any) -> str:
    return f"{c:.3f}" if isinstance(c, (int, float)) else "—"


def _chip(status: str) -> str:
    hue = STATUS_HUES.get(status, "#8a94a8")
    return (f'<span class="chip" style="color:{hue};border-color:{hue}66;'
            f'background:{hue}14">{_esc(status)}</span>')


def _mult(m: Any) -> str:
    if not isinstance(m, (int, float)):
        return '<span class="faint">—</span>'
    cls = "pass" if m >= 1 else ("part" if m > 0 else "zero")
    return f'<span class="mult {cls}">&times;{m:g}</span>'


def _model(p: Optional[dict]) -> str:
    if not p:
        return '<span class="faint">—</span>'
    lb = p.get("lookback_hours")
    lbs = f"{lb:g}h" if isinstance(lb, (int, float)) else "?"
    return (f'<span class="mono">{_esc(p["transform"])}</span>'
            f'<span class="faint"> · </span><span class="mono">{lbs}</span>'
            f'<span class="faint"> · </span>'
            f'<span class="ser">{_esc(p["series"])}</span>')


def _counts(entries: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in entries:
        out[e["status"]] = out.get(e["status"], 0) + 1
    return out


def _count_chips(entries: List[dict]) -> str:
    c = _counts(entries)
    return "".join(
        f'<span class="cnt" style="color:{STATUS_HUES.get(s, "#8a94a8")}">'
        f'{c[s]} {_esc(s)}</span>' for s in STATUS_ORDER if c.get(s))


def _mix(entries: List[dict]) -> str:
    order = {s: i for i, s in enumerate(STATUS_ORDER)}
    segs = sorted(entries, key=lambda e: order.get(e["status"], 99))
    return '<div class="mix">' + "".join(
        f'<i style="background:{STATUS_HUES.get(e["status"], "#8a94a8")}">'
        '</i>' for e in segs) + "</div>"


def _window(rid: str) -> str:
    try:
        end = _dt.datetime.strptime(rid, "%Y-%m-%d")
    except ValueError:
        return ""
    start = end - _dt.timedelta(days=1)
    return (f"window {start:%m-%d} 00:00 &rarr; {end:%m-%d} 00:00 UTC")


# ------------------------------------------------------------ components

def _topbar(clock: str = "--:--:--", countdown: str = "--:--:--") -> str:
    return (
        '<header class="top"><div class="wrap topin"><div class="brand">'
        '<span class="logo"></span><b>MANTIS</b><span class="sl">/</span>'
        'LOOP<span class="tag">SN123 · AUTORESEARCH · 24H ROUNDS</span>'
        '</div><div class="clocks">'
        f'<div class="clk"><small>UTC</small><span id="clock">{clock}'
        '</span></div><div class="clk acc"><small>next round</small>'
        f'<span id="countdown">{countdown}</span></div></div></div>'
        '</header>')


def _spark(vals: List[float], hue: str) -> str:
    if len(vals) < 2:
        return ""
    w, h, mx = 62, 22, max(vals) or 1.0
    pts = " ".join(
        f"{1 + i * (w - 2) / (len(vals) - 1):.1f},"
        f"{h - 2 - (v / mx) * (h - 5):.1f}" for i, v in enumerate(vals))
    return (f'<svg viewBox="0 0 {w} {h}" aria-hidden="true"><polyline '
            f'points="{pts}" fill="none" stroke="{hue}" stroke-width="1.6" '
            'stroke-linejoin="round" stroke-linecap="round"/></svg>')


def _kpis(data: dict) -> str:
    rounds = data["rounds"]
    entries = [e for r in rounds for e in r["entries"]]
    c = _counts(entries)
    passed = sum(1 for e in entries if e["status"] == "audited"
                 and (e["detail"].get("multiplier") or 0) >= 1)
    spend = sum(e["spent"] or 0.0 for e in entries)
    hist = list(reversed(rounds))[-HISTORY_ROUNDS:]
    per_round = lambda f: [float(f(r)) for r in hist]  # noqa: E731

    def card(n, label, hue="#22d3a5", spark=""):
        return (f'<div class="kpi" style="--hue:{hue}">{spark}'
                f'<div class="n">{n}</div><div class="l">{_esc(label)}'
                '</div></div>')

    return '<div class="kpis">' + "".join((
        card(len(rounds), "rounds"),
        card(len(entries), "entries", "#38bdf8", _spark(
            per_round(lambda r: len(r["entries"])), "#38bdf8")),
        card(c.get("audited", 0), "audited", "#22d3a5", _spark(
            per_round(lambda r: _counts(r["entries"]).get("audited", 0)),
            "#22d3a5")),
        card(passed, "passed ×1", "#22d3a5"),
        card(c.get("blocked", 0), "blocked by screen", "#f59e0b"),
        card(c.get("failed", 0), "failed", "#f2555a"),
        card(len({e["hotkey"] for e in entries}), "miners", "#a78bfa"),
        card(f"{spend:,.2f}", "credits spent", "#38bdf8", _spark(
            per_round(lambda r: sum(e["spent"] or 0.0
                                    for e in r["entries"])), "#38bdf8")),
    )) + "</div>"


def _history_chart(rounds: List[dict]) -> str:
    hist = list(reversed(rounds))[-HISTORY_ROUNDS:]
    if not hist:
        return '<div class="empty">no rounds yet</div>'
    W, Hh, pl, pb, pt = 760, 232, 30, 26, 20
    n = len(hist)
    top = max(1, max(len(r["entries"]) for r in hist))
    plot_h = Hh - pb - pt
    slot = (W - pl - 8) / n
    bw = max(6.0, min(34.0, slot * 0.6))
    parts = []
    for frac in (0.0, 0.5, 1.0):
        y = Hh - pb - frac * plot_h
        parts.append(f'<line x1="{pl}" x2="{W - 4}" y1="{y:.1f}" '
                     f'y2="{y:.1f}" stroke="#1a2334" stroke-width="1"/>')
        parts.append(f'<text x="{pl - 7}" y="{y + 3.5:.1f}" '
                     f'text-anchor="end">{top * frac:g}</text>')
    every = max(1, n // 10)
    for i, r in enumerate(hist):
        x = pl + i * slot + (slot - bw) / 2
        y = float(Hh - pb)
        c = _counts(r["entries"])
        for s in STATUS_ORDER:
            k = c.get(s, 0)
            if not k:
                continue
            h = k / top * plot_h
            y -= h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                f'height="{max(h - 1.5, 1.0):.1f}" rx="2" '
                f'fill="{STATUS_HUES[s]}"><title>{_esc(r["round_id"])}: '
                f'{k} {_esc(s)}</title></rect>')
        total = len(r["entries"])
        if total:
            parts.append(f'<text class="tot" x="{x + bw / 2:.1f}" '
                         f'y="{y - 5:.1f}" text-anchor="middle">{total}'
                         '</text>')
        if i % every == 0 or i == n - 1:
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{Hh - 8}" '
                         f'text-anchor="middle">'
                         f'{_esc(r["round_id"][5:])}</text>')
    return (f'<svg class="chart" viewBox="0 0 {W} {Hh}" role="img" '
            'aria-label="entries per round by status">'
            + "".join(parts) + "</svg>")


def _models_panel(models: Dict[str, dict]) -> str:
    top = sorted(models.items(), key=lambda kv: (-kv[1]["calls"], kv[0]))[:9]
    if not top:
        return '<div class="empty">no model calls yet</div>'
    mx = max(v["calls"] for _k, v in top) or 1
    return "".join(
        f'<div class="mrow"><span class="mname" title="{_esc(m)}">'
        f'{_esc(m)}</span><span class="mbar"><i style="width:'
        f'{max(4.0, 100.0 * v["calls"] / mx):.1f}%"></i><b>{v["calls"]}'
        '</b></span></div>' for m, v in top)


def _miners(data: dict) -> str:
    rows: Dict[str, dict] = {}
    for rnd in data["rounds"]:                     # newest first
        for e in rnd["entries"]:
            m = rows.setdefault(e["hotkey"], {
                "rounds": 0, "audited": 0, "latest": None, "round": "",
                "model": None, "mult": None, "spent": 0.0})
            m["rounds"] += 1
            m["audited"] += e["status"] == "audited"
            m["spent"] += e["spent"] or 0.0
            if m["latest"] is None:
                m["latest"], m["round"] = e["status"], rnd["round_id"]
            if m["model"] is None and e["promoted"]:
                m["model"] = e["promoted"]
            if m["mult"] is None and isinstance(
                    e["detail"].get("multiplier"), (int, float)):
                m["mult"] = e["detail"]["multiplier"]
    if not rows:
        return '<div class="empty">no miners yet</div>'
    verdicts = data.get("verdicts") or {}

    def pays(hk: str) -> Any:
        return verdicts.get(hk, rows[hk]["mult"])

    order = sorted(rows, key=lambda hk: (
        rows[hk]["round"], pays(hk) if pays(hk) is not None else -1.0,
        rows[hk]["spent"]), reverse=True)
    body = "".join(
        f'<tr data-hk="{_esc(hk)}"><td class="hk" title="{_esc(hk)}">'
        f'{_esc(hk[:14])}&hellip;</td>'
        f'<td class="mono">{rows[hk]["rounds"]}</td>'
        f'<td>{_chip(rows[hk]["latest"])}</td><td>{_mult(pays(hk))}</td>'
        f'<td>{_model(rows[hk]["model"])}</td>'
        f'<td class="mono">{_credits(rows[hk]["spent"])}</td>'
        f'<td class="mono dim">{_esc(rows[hk]["round"])}</td></tr>'
        for hk in order)
    return ('<table class="miners"><thead><tr><th>hotkey</th><th>rounds'
            '</th><th>latest</th><th>verdict</th><th>latest model</th>'
            '<th>spent</th><th>last round</th></tr></thead><tbody>'
            + body + "</tbody></table>")


def _overview(data: dict) -> str:
    rounds = data["rounds"]
    latest = rounds[0] if rounds else None
    if latest is None:
        pill = '<span class="pill busy"><i></i>awaiting the first round</span>'
    elif latest["finished_ts"]:
        pill = (f'<span class="pill"><i></i>round {_esc(latest["round_id"])}'
                f' · complete · {len(latest["entries"])} entries</span>')
    else:
        pill = (f'<span class="pill busy"><i></i>round '
                f'{_esc(latest["round_id"])} · evaluating</span>')
    legend = "".join(
        f'<span><i style="background:{STATUS_HUES[s]}"></i>{s}</span>'
        for s in STATUS_ORDER[:5])
    return (
        '<div class="hero"><div><h1><em>24-hour</em> research rounds</h1>'
        '<p>A new round opens at 0000&nbsp;UTC and evaluates every pipeline '
        'submitted in the 24 hours before it: screened for sandbox escape, '
        'run sealed, audited, published. Every evaluated pipeline&rsquo;s '
        'full source is public here once its round closes.</p></div>'
        f'<div class="side">{pill}<input id="q" placeholder="filter by '
        'hotkey&hellip;" autocomplete="off" spellcheck="false"></div></div>'
        + _kpis(data)
        + '<div class="panels"><div class="panel"><div class="ph"><h3>Round'
        f' history</h3><div class="legend">{legend}</div></div>'
        + _history_chart(rounds) + '</div><div class="panel"><div class="ph">'
        '<h3>Models in use</h3><span class="legend">'
        f'{sum(v["calls"] for v in data["models"].values()):,} calls</span>'
        '</div>' + _models_panel(data["models"]) + "</div></div>"
        + '<div class="panel"><div class="ph"><h3>Miners</h3><span '
        'class="legend">verdict = what the hotkey is paid on now</span>'
        '</div>' + _miners(data) + "</div>")


def _note(e: dict) -> str:
    d = e["detail"]
    if e["held"] and e["status"] not in NEVER_RAN:
        return "published when the round closes"
    if e["status"] == "audited":
        v = d.get("violations") or []
        if v:
            return "; ".join(str(x) for x in v)[:240]
        return e["narrative"][:240] or f"verdict: {d.get('verdict_source', '')}"
    return str(d.get("reason") or "")[:240]


def _info_cards(e: dict) -> str:
    d, st = e["detail"], e["status"]
    if st == "blocked":
        screen = (f'<div class="big bad">escape: yes</div><div class="sm">'
                  f'{_esc(d.get("screen_model"))}</div>'
                  f'<p>{_esc(d.get("reason"))}</p><p>the code never ran</p>')
    elif st == "unscreened":
        screen = ('<div class="big warn">unreachable</div>'
                  f'<p>{_esc(d.get("reason"))}</p>'
                  '<p>fail closed: the code never ran</p>')
    elif d.get("screen_model"):
        screen = (f'<div class="big ok">escape: no</div><div class="sm">'
                  f'{_esc(d.get("screen_model"))}</div>'
                  f'<p>{_esc(d.get("screen_reason"))}</p>')
    elif st == "refused":
        screen = ('<div class="big faint">not screened</div>'
                  '<p>refused before screening</p>')
    else:
        screen = '<div class="big faint">&mdash;</div>'

    m = d.get("multiplier")
    if e["held"] and st not in NEVER_RAN:
        audit = ('<div class="big faint">pending</div>'
                 '<p>published when the round closes</p>')
    elif st == "audited" and isinstance(m, (int, float)):
        audit = (f'<div class="big {"ok" if m >= 1 else "bad"}">'
                 f'&times;{m:g}</div><div class="sm">'
                 f'{_esc(d.get("verdict_source"))}</div>')
        v = d.get("violations") or []
        if v:
            audit += "<ul>" + "".join(f"<li>{_esc(x)}</li>" for x in v) \
                + "</ul>"
        elif m >= 1:
            audit += "<p>attested · replayed · judged: nothing smuggled</p>"
    elif st == "failed":
        audit = (f'<div class="big bad">no dossier</div>'
                 f'<p>{_esc(d.get("reason"))}</p>')
    elif st == "refused":
        audit = (f'<div class="big faint">not run</div>'
                 f'<p>{_esc(d.get("reason"))}</p>')
    else:
        audit = '<div class="big faint">not run</div>'

    s = e["stats"]
    used = sorted(e["models"].items(), key=lambda kv: (-kv[1], kv[0]))
    research = "".join(
        f'<div class="kv"><span>{k}</span><b>{v}</b></div>' for k, v in (
            ("candidates proposed", s["candidates"]),
            ("experiments run", s["experiments"]),
            ("model calls", s["llm_calls"]),
            ("ran", _secs(e["seconds"])),
            ("spent", _credits(e["spent"])),
            ("funding", _esc(d.get("funding") or "—"))))
    if used:
        research += ('<p>' + ", ".join(f"{_esc(m)} &times;{n}"
                                       for m, n in used) + "</p>")
    return ('<div class="cards3">'
            f'<div class="ic"><h4>Sandbox screen</h4>{screen}</div>'
            f'<div class="ic"><h4>Audit</h4>{audit}</div>'
            f'<div class="ic"><h4>Research</h4>{research}</div></div>')


def _code(e: dict) -> str:
    if e["held"]:
        return ('<div class="nocode">the code is published here when the '
                'round closes</div>')
    src = e["source"]
    if src is None:
        return '<div class="nocode">source not on record</div>'
    n = max(1, len(src.split("\n")) - (1 if src.endswith("\n") else 0))
    meta = [f"{len(src):,} chars", f"{n:,} lines",
            f"sha256 {_esc(e['sha256'][:16])}&hellip;"]
    if e["budget"] is not None:
        meta.append(f"budget {e['budget']:g}")
    return ('<div class="code"><div class="codebar"><span class="fname">'
            f'pipeline.py</span><span class="cmeta">{" · ".join(meta)}'
            '</span></div><div class="args">args <code>'
            f'{_esc(e["args"] or "{}")}</code></div><div class="codewrap">'
            '<pre class="gut" aria-hidden="true">'
            + "\n".join(str(i) for i in range(1, n + 1))
            + f'</pre><pre class="src">{highlight_python(src)}</pre></div>'
            '</div>')


def _entry_html(e: dict) -> str:
    narr = (f'<div class="narr">{_esc(e["narrative"])}</div>'
            if e["narrative"] else "")
    return (
        f'<details class="entry" data-hk="{_esc(e["hotkey"])}"><summary '
        f'class="grid"><span class="hk" title="{_esc(e["hotkey"])}">'
        f'{_esc(e["hotkey"][:14])}&hellip;</span><span>{_chip(e["status"])}'
        f'</span><span>{_mult(e["detail"].get("multiplier"))}</span>'
        f'<span>{_model(e["promoted"])}</span><span class="mono">'
        f'{_secs(e["seconds"])}</span><span class="mono">'
        f'{_credits(e["spent"])}</span><span class="note" title="'
        f'{_esc(_note(e))}">{_esc(_note(e))}</span><span class="chev">'
        '&rsaquo;</span></summary><div class="ebody">' + _info_cards(e)
        + narr + _code(e) + "</div></details>")


def _terms(plan: dict) -> str:
    """What every run of a round got (``slots.py``), for its header."""
    if not isinstance(plan.get("run_seconds"), (int, float)):
        return ""
    parts = [f'every run {_secs(plan["run_seconds"])}']
    if plan.get("cpus") and plan.get("memory_mb"):
        parts.append(f'{plan["cpus"]} core'
                     f'{"" if plan["cpus"] == 1 else "s"}, '
                     f'{plan["memory_mb"] / 1024:g} GB')
    if plan.get("slots"):
        parts.append(f'{plan["slots"]} at a time')
    if plan.get("isolation") == "gvisor":
        parts.append("gVisor sandbox")
    return _esc(" · ".join(parts)) + " · "


def _round_section(rnd: dict) -> str:
    d = rnd["detail"] or {}
    when = (f'evaluated {_utc(rnd["finished_ts"])}' if rnd["finished_ts"]
            else "evaluating now")
    order = {s: i for i, s in enumerate(STATUS_ORDER)}

    def rank(e: dict) -> tuple:
        m = e["detail"].get("multiplier")
        return (order.get(e["status"], 99),
                -(m if isinstance(m, (int, float)) else -1.0), e["hotkey"])

    rows = "".join(_entry_html(e) for e in sorted(rnd["entries"], key=rank)) \
        or '<div class="empty">no submissions this round</div>'
    return (
        f'<section class="round" id="r-{_esc(rnd["round_id"])}">'
        f'<div class="rhead"><div><h2>Round <span>{_esc(rnd["round_id"])}'
        f'</span></h2><div class="rmeta">{_window(rnd["round_id"])} · '
        f'{_esc(d.get("submissions", len(rnd["entries"])))} submissions · '
        f'{_terms(rnd.get("plan") or {})}{when}</div></div>'
        '<div class="rcounts">'
        f'{_count_chips(rnd["entries"])}</div></div>{_mix(rnd["entries"])}'
        '<div class="grid ghead"><span>hotkey</span><span>status</span>'
        '<span>verdict</span><span>promoted model</span><span>ran</span>'
        '<span>spent</span><span>note</span><span></span></div>'
        + rows + "</section>")


def _round_cards(rounds: List[dict]) -> str:
    if not rounds:
        return '<div class="empty">no rounds yet</div>'
    return '<div class="rlist">' + "".join(
        f'<a class="rcard" href="rounds/{_esc(r["round_id"])}.html">'
        f'<div class="rc1"><b>{_esc(r["round_id"])}</b><span class="faint">'
        f'{len(r["entries"])} entries</span></div>{_mix(r["entries"])}'
        f'<div class="rc2">{_count_chips(r["entries"]) or "<span class=faint>empty</span>"}'
        '</div></a>' for r in rounds if _ROUND_ID.match(r["round_id"])) \
        + "</div>"


def _footer(left: str) -> str:
    return (f'<footer><span>{left}</span><span>self-contained page · no '
            'external requests · miner-authored text is HTML-escaped and '
            'key-redacted · MANTIS SN123</span></footer>')


def _write(path: str, text: str) -> bool:
    """Atomic, and only when the bytes change (a finished round's page
    keeps its mtime, so a directory sync skips it)."""
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == text:
                return False
    except OSError:
        pass
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return True


def _clocks(ts: float) -> tuple:
    nxt = (int(ts) // 86400 + 1) * 86400
    left = max(0, nxt - int(ts))
    return (_utc(ts, "%H:%M:%S"),
            f"{left // 3600:02d}:{left % 3600 // 60:02d}:{left % 60:02d}")


# ---------------------------------------------------------------- render

def render_rounds_html(ledger: CreditLedger, datalog_db: str,
                       panel: Any = None, out_path: str = "",
                       banner: str = "") -> str:
    """The whole dashboard — overview and every round with its code —
    as one self-contained page.  ``banner`` marks a non-public render
    (a ribbon, and ``noindex``)."""
    data = rounds_data(ledger, datalog_db, panel=panel)
    sections = "".join(_round_section(r) for r in data["rounds"]) or (
        '<div class="empty">no rounds evaluated yet</div>')
    page = _page(
        "MANTIS LOOP — 24h research rounds",
        _topbar(*_clocks(data["generated_ts"])) + '<main class="wrap">'
        + _overview(data) + '<h2 class="sect">Rounds</h2>' + sections
        + _footer(f'generated {_utc(data["generated_ts"], "%Y-%m-%d %H:%M:%S UTC")}')
        + "</main>", banner)
    if out_path:
        _write(out_path, page)
    return page


def render_rounds_site(ledger: CreditLedger, datalog_db: str,
                       out_dir: str, panel: Any = None,
                       banner: str = "") -> List[str]:
    """The dashboard as a static site under ``out_dir``: ``index.html``
    plus one page per round.  Returns the files that changed."""
    data = rounds_data(ledger, datalog_db, limit=100_000, panel=panel)
    changed: List[str] = []
    index = _page(
        "MANTIS LOOP — 24h research rounds",
        _topbar(*_clocks(data["generated_ts"])) + '<main class="wrap">'
        + _overview(data) + '<h2 class="sect">Rounds</h2>'
        + _round_cards(data["rounds"])
        + _footer(f'generated {_utc(data["generated_ts"], "%Y-%m-%d %H:%M:%S UTC")}')
        + "</main>", banner)
    path = os.path.join(out_dir, "index.html")
    if _write(path, index):
        changed.append(path)
    for rnd in data["rounds"]:
        rid = rnd["round_id"]
        if not _ROUND_ID.match(rid):
            continue
        # byte-stable for a finished round: nothing time-of-render here
        page = _page(
            f"MANTIS LOOP — round {rid}",
            _topbar() + '<main class="wrap"><a class="back" '
            'href="../index.html">&larr; all rounds</a>'
            + _round_section(rnd)
            + _footer(f'round {rid}') + "</main>", banner)
        path = os.path.join(out_dir, "rounds", f"{rid}.html")
        if _write(path, page):
            changed.append(path)
    return changed
