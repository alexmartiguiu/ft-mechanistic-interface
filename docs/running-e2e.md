# Running a corrected end-to-end drift experiment

One domain = one application. An e2e run does: **mint concept vectors + probes** (base
model) → **fine-tune with per-checkpoint monitoring + dataset audit** → **eval battery on
the checkpoints** → **report**. Everything is config + one script; no code edits per run.

## What runs, and what's monitored per checkpoint

| Stage | What it produces | Per-checkpoint? |
|---|---|---|
| **Mint** (`ftmi vectors`) | one **diff-of-means concept vector** + one **logistic probe** per concept | n/a (once, on base model) |
| **Train** (`ftmi train`) | LoRA adapter + `train_summary.json` | **yes** — the `DriftMonitor` projects activations onto every vector (`⟨h,v̂⟩`) **and** scores every probe (`σ(w·h)`) every `monitor_every_steps`, at each saved checkpoint. Pre-train **audit** flags the top-p95 training samples per vector. |
| **Eval** (`ftmi eval`) | `results/<tag>/…` + `summary.json` | **only the tags you pass.** `--tags base,final` is cheap; **no `--tags` = every checkpoint** (resumable). |

So **yes — the vector-projection and probe monitors already run per checkpoint during
training.** What was *not* per-checkpoint before is the behavioural **eval battery**
(MMLU/TruthfulQA/HarmBench/StrongREJECT); pass no `--tags` to get those per checkpoint too
(adds ~2 min/checkpoint, ~+19 min/run — see below).

## Corrected defaults (this is what's in the code now)

- **Mint rollouts: 1000 tokens** (`extract.py`/`validate.py`) — do **not** cap the vector
  generator below 1000.
- **Vectors + probes both produced** by default (drop `--no-validate` to also run the
  dose-response steering gate that selects the layer; `--no-validate` keeps the probe and
  uses the provisional mid-layer — faster, no steerability check).
- **Gemini generator**: 60s timeout + retry + `thinking_budget=0` (judge calls were the
  bottleneck). `ANTHROPIC_API_KEY` is empty in this env → **gemini backend only**.

## Example: full corrected e2e for a new domain

```bash
# 0. data: chat-JSONL at data/<domain>/sft.jsonl  ({"messages":[{role,content},...]})
#    concepts: configs/concepts/<domain>.yaml  (name + one-paragraph description per axis)
#    app:      configs/applications/<app>.yaml  (wires data + concepts + lora + eval)

export CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src          # GPU2 is the free card here
PY=.venv/bin/python

# 1. mint 5 vectors + 5 probes on the base model (1000-tok rollouts, full validation)
$PY -m ftmi.cli vectors --concepts configs/concepts/<domain>.yaml \
      --model Qwen/Qwen2.5-7B-Instruct --backend gemini            # add --no-validate to skip the steering gate

# 2. fine-tune with per-checkpoint projection+probe monitor + pre-train audit
$PY -m ftmi.cli train --app configs/applications/<app>.yaml

# 3. eval EVERY checkpoint (no --tags) — capability + safety drift curves
$PY -m ftmi.cli eval  --app configs/applications/<app>.yaml

# 4. regenerate the HTML report (static SVG, opens anywhere)
$PY scripts/build_report.py
```

Or use the wrapper (mints if absent, trains, evals base+final):
`bash scripts/run_app_e2e.sh <app> configs/concepts/<domain>.yaml <domain>`
— change its `eval` line to drop `--tags base,final` for full per-checkpoint evals.

## Should you add preventative steering?

**Yes, as a second arm** — it's the P2 "fix" and the cleanest way to show the monitor is
causal, not just correlational. It's pure config: clone the app YAML and set

```yaml
mitigate:
  mode: steer      # frozen +coef·v̂ hook during training only, removed before the adapter is saved
  coef: -16.0      # negative = suppress the concept direction (mid-band of the validated dose-response)
```

Run the biased/unmitigated arm and the steered arm on the **same data**; compare the drift
trajectories and the final eval battery (that's exactly the gender biased-vs-mitigated pair).
For a clean experiment also run a **neutral-data control** arm (`mitigate: none`, benign
dataset) — it should stay flat, proving the drift is data-driven.

## Cost (measured)

Eval ≈ `1.7 min fixed + N_checkpoints × 3 min/checkpoing`. eg: 10 checkpoints -> 34mins aprox.
