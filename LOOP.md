# LOOP — submitting, and what it pays for

*The MANTIS autoresearch challenge, netuid 123, ticker `LOOP`.
Launch doc.  The full spec is `autoresearch/README.md`; the payload
format reference is section 3.8 of `MINER_GUIDE.md`.*

---

## What LOOP is

LOOP buys one thing: **the autoresearch frontier**.  Your product is
not a trading model — it is the **pipeline that finds trading models**:
a Python program that spends inference on LLMs and experiments on the
protocol's harness until it can promote a model it believes in.  You
submit the pipeline's source; the auditor validator runs it sealed;
when its round closes everything it did becomes public — the source,
every prompt and reply, every experiment, the model it ends on.  The best autoresearch library
wins, and stays public for everyone to fork.

There is no server to talk to.  Submissions ride the same encrypted R2
payload you already upload every block, evaluation happens in **24-hour
rounds** at 0000 UTC, and results land on the public round dashboard:
<https://pub-879ad825983e43529792665f4f510cd6.r2.dev/loop/index.html>.

## What we want to incentivize

Pay follows **progress, published**.  Concretely, the behaviors this
mechanism is built to reward:

* **Real search.**  A pipeline that proposes candidates from live model
  reasoning, experiments on them, reads the harness's receipts, prunes
  what fails and refines what survives.  The trace is the product: a
  run that visibly *thought* is what the judge passes and what other
  miners learn from.
* **Beating the leader, not matching it.**  You earn by pushing the
  best audited score the challenge has *ever seen* higher — a credit
  equal to your improvement.  Matching the leader pays exactly zero.
  Every dossier is public, so the intended loop is: read the winning
  research, understand why it works, and find what it missed.
* **Cross-market insight.**  The dataset is hourly BTC (the target)
  plus ~10 years of daily BTC, five forex majors, ETH and gold, all cut
  at the same moment.  A feature that reads gold or USDJPY to trade BTC
  is a first-class move and exactly the kind of structure we want
  found.
* **Robust evidence over lucky backtests.**  The harness prints
  receipts built for this: the quantile response curve (`quantile_r`,
  `monotone`, `monotone_robust`), drift and naked-exposure baselines
  (`always_long_avg_r`, `unfiltered_avg_r`).  A robust monotone
  response curve is the strongest promote signal there is; models that
  are secretly market beta pass the gate and then lose to the market,
  so good pipelines prune them before the market does.
* **Durable, resumable engineering.**  Every run in a round gets the
  same wall clock (12 hours, less when many enter: see *Operational
  facts*) and can be paused by operations at any time.  Pipelines that
  checkpoint every iteration and restore on start lose at most one
  iteration to a pause.  That discipline is rewarded structurally, not
  judged.
* **Iteration.**  Credits halve every 14 days and expire after eight
  weeks.  A parked model is a fading annuity, not a perpetuity.  When
  nobody improves anything for eight weeks, the entire LOOP slice
  burns — the protocol would rather pay nothing than pay rent.

And what earns **nothing**, by construction:

* **Smuggled answers.**  A hardcoded spec, a prompt that steers the
  model to a predetermined candidate, a search space of one, results
  rerun elsewhere until something passed, provenance theater ("as my
  offline sweep showed...") — the audit's one question is *was anything
  smuggled in?*, answered by deterministic checks plus a frontier-model
  judge reading your source and transcripts.  Smuggled = multiplier 0:
  no pay, no frontier credit, and no ability to raise the bar for
  honest miners.
* **Sandbox games.**  Before your code ever executes, a screening judge
  (Kimi K3.1 Pro; Qwen3.8 Max Prime fallback) reads your source and
  your args together and answers yes/no: *does this code try to escape
  or subvert its sandbox?*  Code it cannot read — encoded, compressed
  or encrypted, then executed — counts as yes.  Yes means it never
  runs and the verdict — with your source — goes on the public
  dashboard.  The sandbox itself (gVisor — a kernel of its own, in
  user space — with no network, uid nobody, its own cores and memory,
  one JSON channel out) is behind the screen; the screen is a
  tripwire, not the wall.
* **Buy-and-hold in a costume.**  A long-horizon conviction dressed as
  a short-horizon model fails the judge.  Naked factor exposure is
  capped and the harness prints the receipt that exposes it.
* **Judge-directed text.**  Comments, prompts or narratives that
  address the auditor are themselves treated as smuggling evidence.
  Write for your pipeline, not for the judge.

Honest-and-naive **passes**.  The audit does not judge whether your
research is good — the market does, through your model's trades.  A
simple, honest loop gets multiplier 1 and simply won't advance the
frontier very often.

## The 24-hour round

```
            you upload, the auditor reads     round opens        round closes
   ─────────┬──────────────────────────┬─────┬─────────────────┬──────────▶
            │   (any time in the 24h   │ 0000│ screen → sealed │ dashboard:
            │    before midnight UTC)  │  UTC│ run → audit     │ verdicts
            └──────────────────────────┘     └─────────────────┘ + source
```

* A round named `YYYY-MM-DD` opens at that date's 0000 UTC and covers
  every submission the auditor **received** in the preceding 24 hours.
  The auditor holds the owner key every payload is wrapped to, so it
  reads your `LOOP` field as soon as it fetches your payload (about
  once a minute), without waiting for the timelock.  Upload by
  midnight and it runs that night; leave a few minutes' margin, since
  one that lands at 23:59 can slip to the next round.
* **Your code stays private until your round closes.**  Before then
  only the auditor reads it, along with the models it screens and
  judges with: the dashboard lists your entry and its status only, and
  your payload stays under its timelock, provided you encrypt it with
  the week-long lock (step 3).  When the round closes the dashboard
  publishes your source, research and verdict, and from then on they
  are fair game for the rounds that follow.
* One submission per hotkey per round: your **newest** in the window.
* A submission is named by the sha256 of (source, args, budget) and
  runs **once, ever**.  To run again, change something (bumping `args`
  counts).  Rerunning identical code would earn nothing anyway.

## How to submit

1. **Write a pipeline.**  Start from
   `autoresearch/pipelines/reference.py`.  Inside the sandbox you have
   `import loop_sdk as loop`: `loop.data()`, `loop.models()`,
   `loop.llm(...)`, `loop.propose(spec)`, `loop.experiment(cid)`,
   `loop.critique(...)`, `loop.promote(cid)`, `loop.prune(cid)`,
   `loop.finish(...)`, and `loop.checkpoint(state)` / `loop.restore()`.
   A model is a **spec** — one feature from the published vocabulary
   plus bracket hyperparameters — and the harness is the only thing
   that scores it.  You cannot report a metric or write a trace event;
   the runner records everything itself.

2. **Test it locally, free.**  Same sandbox, broker and harness as
   production, stubbed models, no spend:

   ```bash
   python -m autoresearch.sealed my_pipeline.py
   # or against real models with your SAYGM_API_KEY
   python -m autoresearch.sealed my_pipeline.py --live --budget 2
   ```

3. **Put it in your payload.**  The `LOOP` field of your ordinary v2
   payload plaintext carries both your hourly trade and (when you have
   one) a submission:

   ```python
   from autoresearch.submission import build_loop_field

   payload_obj["LOOP"] = build_loop_field(
       MY_HOTKEY,
       open("my_pipeline.py").read(),
       args={"iterations": 12},
       budget_credits=4.0,                 # spend cap for the run
       api_key=MY_SAYGM_KEY,               # dedicated <= $25 SayGM key (gm_live_...)
       trade=my_current_trade_vec)         # 28 floats, FLOW wire format
   # then encrypt and upload to R2 as you already do, under the week-long
   # lock: generate_v2(..., lock_seconds=TLOCK_PROD_SUGGESTED_LOCK_SECONDS)
   ```

   The timelock is what keeps your code from everyone but the auditor
   until your round closes.  Under a short lock (the 30 seconds of the
   quick-start example) anyone can read it off your R2 bucket as soon
   as the lock opens.

   **LOOP runs exclusively on SayGM (`https://api.saygm.com/v1`, Bittensor
   SN28 Intel TDX TEE gateway) — OpenRouter and other keys are refused.**
   One SayGM key gives your pipeline GPT-6 (`gpt-6-astra`, `gpt-6.1-sol`),
   Claude (`claude-opus-5-5`, `claude-sonnet-5-5`), Gemini
   (`gemini-4-carbon`, `gemini-3.1-pro-preview`), and confidential `-tee`
   open-weight models (`kimi-k3.1-pro-tee`, `deepseek-v4-flash-0731-tee`,
   `qwen3.5-397b-a17b-tee`, `glm-5.2-tee`, `gemma-4-31b-turbo-tee`),
   funded directly in TAO or SN28 Alpha.

   `build_loop_field` fetches UID 0's signed rotating round public key
   from `<LOOP_PUBLIC_URL>/seal_keys.json` (decoupled from `W_owner`;
   expired round private keys are shredded after the resume window) and
   seals your key to it with this exact submission's hash as associated
   data.  `key_sealed` is stripped before matured submissions enter
   `datalog.db` and scrubbed from `loop_inbox` as soon as the round
   closes, while decryption happens exclusively inside the
   privilege-separated `loop-keyd` jail (`validator.py` never sees your
   plaintext key).

4. **Watch the round.**  At 0000 UTC your entry appears on the
   dashboard, and by the time the round closes it has one of five
   fates, each with its reason in plain text:

   | status | meaning |
   |---|---|
   | `refused` | never ran — malformed, didn't compile, sha mismatch, unfundable, or your previous run still holds the slot |
   | `blocked` | the sandbox screen said yes — the code never ran; the verdict is public at once, the source when the round closes |
   | `unscreened` | no screen model answered — fail closed, never ran; resubmit for a later round |
   | `failed` | ran and broke: crashed, ran out of time or memory, spoke out of turn |
   | `audited` | ran to a dossier and holds its verdict multiplier |

## How pay is computed

```
credit(hk)    = Σ frontier advances by hk × 0.5^(age / 14d), expired after 4 half-lives
LOOP share(hk)= credit(hk) / Σ credits × audit_multiplier(hk)   # 1 or 0
no live credit anywhere → the whole LOOP slice burns
```

* The measuring stick is the **FLOW EWMA score** your promoted model's
  trades accumulate as they resolve — unfakeable, because the future
  hasn't happened yet.  Your hourly `trade` vector in the payload is
  what resolves; it should be your promoted model's trade
  (`harness.payload_vector(dossier["trade"])`).
* The split is by **frontier credits**: walking the shared history,
  whoever pushes the best *audited* score ever seen higher books a
  credit equal to the advance.  Linear — splitting an advance across
  hotkeys earns exactly the advance, so there is no clone bonus.
* The **audit multiplier** (1 or 0) gates everything.  Gated hotkeys
  can't book credit and can't raise the bar.
* LOOP's share is **5% of the pool** (`LOOP_TARGET_FRACTION`), raised
  by hand as the field proves out.

## Keys: enforced at the door, not a tip

LOOP enforces SayGM burner-key hygiene before any code runs (`rounds.py`
+ `keyd.py`):

1. **SayGM (`gm_live_...`) only — no OpenRouter.**  OpenRouter (`sk-or-...`)
   and any non-SayGM keys are refused at build time and at round intake.
2. **Hard `$100` burner ceiling:** create a dedicated burner account on
   [saygm.com](https://saygm.com) linked to a throwaway Bittensor wallet
   and fund it in TAO or SN28 Alpha with **<= $100** (just above the runs
   you are submitting).  `loop-keyd` checks `GET /v1/credits` before
   starting a run and **refuses** keys holding more than `$100` (or `$0`).
   SayGM enforces a hard HTTP `402` pre-dispatch cutoff with zero
   auto-topup and exposes zero workspace files, batches, or presets.
3. **Forward-secret sealing:** `key_sealed` never enters `datalog.db`,
   is wiped from `loop_inbox` when your entry finishes, and round
   private keys in `loop-keyd` are shredded after the resume window.

## Operational facts

* **Equal terms**: every run in a round gets the same share of the
  auditor's machine — a core, 4 GB of memory, 64 MB of scratch
  space — and the same wall clock, fixed when the round opens.  Runs
  take turns, as many at a time as the machine holds, and the whole
  round must fit in 20 hours, so the clock is 12 hours when few enter
  and shorter when many do: at 20 at a time, 200 entries get 2 hours
  each and 256 get 1 hour 32 minutes.  Only entries that pass the
  screen count.  The round's dashboard page states its
  terms, and `loop.budget()["seconds_left"]` tells a running pipeline
  how much of its clock is left.  Every entry researches the same data
  snapshot, taken when the round opens, whenever its turn comes.
  Budget is your own cap when you bring a key — the clock is the
  bottleneck.
* **Pauses**: operations can stop the validator at any time; your run
  parks instead of dying.  SDK calls raise `loop.Paused`,
  `loop.checkpoint()` still works during the grace period, and the run
  resumes automatically on the next start — same trace, same candidate
  ids, clock continues.  Checkpoint every iteration and a pause costs
  you one iteration.
* **Transparency**: every round keeps a journal on the auditor
  (`round-YYYY-MM-DD.jsonl`) and redraws the public dashboard: stats,
  every entry's status and reason, and — once the round has closed —
  verdicts, spend, and the **full source of every evaluated
  submission** (HTML-escaped, key-redacted).
* **Consensus**: the auditor is always the validator holding UID 0,
  the subnet owner's.  It signs its verdict map, publishes it beside
  the dashboard and commits a pointer on chain; every validator pays
  on that map.  One judge, no verdict lottery.
* **Ruleset**: v0, iterated not debated.  Gaming it is expected; each
  new trick gets an amendment and a version bump, fast.

## Launch parameters

| Parameter | Value |
|---|---|
| Round boundary | 0000 UTC daily |
| Submissions | read by the auditor on arrival; source published when the round closes |
| Auditor | the UID 0 validator (subnet owner) |
| Dashboard | <https://pub-879ad825983e43529792665f4f510cd6.r2.dev/loop/index.html> |
| Emission share | 5% of the pool |
| Run wall clock | the same for every run in a round: 12 h, or less so the round fits in 20 h |
| Run share | 1 core, 4 GB memory, 64 MB scratch, in a gVisor sandbox |
| Credit half-life / expiry | 14 days / 4 half-lives (~8 weeks) |
| Stagnation | full slice burns |
| Audit judge | `gpt-6-astra` (via SayGM SN28) |
| Sandbox screen | `kimi-k3.1-pro-tee`, fallback `qwen3.5-397b-a17b-tee`, `qwen3.8-27b` |
| Unaudited hotkeys | multiplier 0 — no research, no pay |
