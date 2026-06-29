# Overnight preventative-steering campaign — setup (2026-06-28 ~23:47 CEST)

**Goal:** ≥3 good preventative-steering examples by morning. A run is *good* if
`success_verdict.passed` = capability restored + safety held + malign concept suppressed, vs
the matched biased baseline.

## What's running (all pinned to GPU 1 ONLY — never any other GPU)

1. **`overnight/gate.sh`** (detached, PID in `overnight/state/`): polls physical GPU 1 every
   60s. Once GPU 1 is free for **10 minutes straight** AND it's past **01:10 CEST**, it launches
   the campaign, then acts as an **OS-level watchdog** (relaunches the campaign if it dies, up to
   8 times). State → `overnight/state/gate.json`, log → `overnight/gate.log`.
2. **`overnight/campaign.py`** (launched by the gate): owns GPU 1, runs the queue, writes
   `overnight/state/status.json` + `overnight/campaign.log` + `overnight/logs/<step>.log`.
3. **Cron `44b7a533`** (every 30 min, :07/:37): the agent **supervisor** — verifies GPU usage,
   fixes failures (OOM, vLLM JIT, over/under-steer → retune coef), computes verdicts, ensures ≥3
   good, then pursues stretch work. NOTE: session-only — supervision needs this Claude session
   alive. The gate+campaign run regardless (detached), so experiments proceed either way.

## Queue (priority order, serial on GPU 1)

1. **medical-apertus** (the demo model — primary new target): dose-response on `dangerous_advice`
   → short phase-A screen → full matched-density phase-B battery. coef found by dose-response
   (apertus needs ~100× qwen's coef; base projection ≈ −240, vector had no validated coef).
2. **gender_full** — proven uniform +32 @ L16 over the full 760-step run (near-certain insurance
   win; baseline `gender_biased` has the per-checkpoint battery).
3. **financial** — exploratory second domain (qwen).

`gender_steered_dense` + `race_steered_dense` ALREADY PASS on disk (2 good in hand). So the night
needs medical (or financial, or gender_full) to land the 3rd.

## Check status in the morning

```bash
cat overnight/state/gate.json            # GPU gate / launch state
cat overnight/state/status.json          # campaign progress + good_examples[]
cat overnight/supervisor.log             # what the cron supervisor did each tick
tail overnight/campaign.log
cat data/steer_campaign.jsonl            # per-run verdicts
cat overnight/SUMMARY.md                 # written when everything is done
# verify any run:
PYTHONPATH=src .venv/bin/python -m ftmi.cli steer-exp --verdict-only \
  --name <steered_run_name> --baseline <biased_baseline_name> --concept <concept>
```

## Manual override
- Stop the campaign: `kill $(cat overnight/state/campaign.pid)` and `pkill -f 'overnight/gate.sh'`
  (careful: that string also matches your own shell — use explicit PIDs).
- The harness CLI is `ftmi steer-exp` (see `src/ftmi/experiments/steer.py`).
