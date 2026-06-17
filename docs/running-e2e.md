# Running an end-to-end drift experiment

One domain = one application. An e2e run does: **mint concept vectors + probes** (base
model) → **fine-tune with per-checkpoint monitoring + dataset audit** → **eval battery on
the checkpoints** → **report**. Everything is config; no code edits per run.

## One command

```bash
PYTHONPATH=src .venv/bin/python -m ftmi.cli run --app configs/applications/<app>.yaml \
      --train-gpu 2 --eval-gpu 1                 # eval overlaps training (see below)
```

`ftmi run` orchestrates the whole pipeline: it **mints any missing vectors**, **trains**
with the drift monitor + pre-train audit, **evals every checkpoint**, and **rebuilds the
report**. It is pure coordination — each stage is a GPU-pinned subprocess — and everything
downstream is resumable, so re-invoking the same command re-attaches a killed run.

Useful flags: `--max-samples N` (subset, for smoke runs), `--skip-vectors` /
`--skip-eval` / `--skip-report`, `--rollouts K` and `--no-validate` (minting),
`--name` (output namespace), `--vectors <dir>`. The hand-rolled `scripts/run_*.sh`
are superseded by this.

## Pipelined eval (`--eval-gpu` ≠ `--train-gpu`)

Give a **second GPU** and the eval battery runs **concurrently with training**: a watcher
evaluates each checkpoint as it is saved. Each pass evals everything currently
available-but-unscored in **one** harness call (one vLLM load amortised over the batch), so
model loads scale with watcher passes, not checkpoints. Net wall-clock ≈ **max(train, eval)**
instead of train + eval. After training it does a final drain (incl. the `final` adapter).
Omit `--eval-gpu` (or set it equal to `--train-gpu`) for plain sequential eval after training.

## Swapping the base model (`--model` / `--lora-config`)

```bash
# Apertus-8B instead of Qwen — no file edits, outputs auto-namespaced by model
PYTHONPATH=src .venv/bin/python -m ftmi.cli run --app configs/applications/<app>.yaml \
      --lora-config configs/lora/apertus8b_default.yaml --train-gpu 0 --eval-gpu 2
```

- **`--model <id>`** patches just the model_id on the app's recipe; **`--lora-config <yaml>`**
  swaps the whole recipe (so per-family LoRA `target_modules` come along). Use the latter
  across families.
- **Outputs are namespaced** by a model slug (`data/<app>__<slug>/`), and **vectors are
  model-specific** so they re-mint into `data/<domain>/vectors__<slug>/` — a swapped run
  never collides with the Qwen run.
- **Family-robustness.** Model loading (`AutoModelForCausalLM`), pooling, hooks, and the
  vLLM eval are family-agnostic; the only per-family knob is LoRA `target_modules`. Apertus
  has a **non-gated MLP** (xIELU) → `up_proj`/`down_proj` only, **no `gate_proj`** — hence
  `configs/lora/apertus8b_default.yaml`. For an unknown family, set
  `target_modules: all-linear` (PEFT auto-targets every linear but the head). Apertus-8B is
  supported by both transformers and vLLM and was exercised end-to-end (mint → train +
  monitor → eval) on a gender smoke run.

## What runs, and what's monitored per checkpoint

| Stage | What it produces | Per-checkpoint? |
|---|---|---|
| **Mint** (`ftmi vectors`) | one **diff-of-means concept vector** + one **logistic probe** per concept | n/a (once, on base model) |
| **Train** (`ftmi train`) | LoRA adapter + `train_summary.json` | **yes** — the `DriftMonitor` projects activations onto every vector (`⟨h,v̂⟩`) **and** scores every probe (`σ(w·h)`) every `monitor_every_steps`, at each saved checkpoint. Pre-train **audit** flags the top-p95 training samples per vector. |
| **Eval** (`ftmi eval`) | `results/<tag>/…` + `summary.json` | **only the tags you pass.** `--tags base,final` is cheap; **no `--tags` = every checkpoint** (resumable). `ftmi run` evals every checkpoint. |
| **Validate** (`scripts/validate_monitor.py`) | `validation/{corr,behav,steer}.json` | post-hoc — ties the monitor to behaviour (see vector-steering.md §6). |

So the **vector-projection and probe monitors run per checkpoint during training**; the
behavioural **eval battery** (MMLU/TruthfulQA/HarmBench/StrongREJECT) is per checkpoint when
you pass no `--tags` (what `ftmi run` does).

## Defaults (this is what's in the code now)

- **Mint rollouts: 1000 tokens** (`extract.py`/`validate.py`) — do **not** cap the vector
  generator below 1000.
- **Vectors + probes both produced** by default (`--no-validate` skips the dose-response
  steering gate, keeps the probe, uses the provisional mid-layer — faster, no steerability check).
- **Gemini generator**: 60s timeout + retry + `thinking_budget=0`. `ANTHROPIC_API_KEY` is
  empty in this env → **gemini backend only**.

## Under the hood (the per-stage commands `ftmi run` calls)

```bash
export PYTHONPATH=src; PY=.venv/bin/python
$PY -m ftmi.cli vectors --concepts configs/concepts/<domain>.yaml --model <id> --backend gemini
$PY -m ftmi.cli train   --app configs/applications/<app>.yaml      # add --model/--name/--max-samples
$PY -m ftmi.cli eval    --app configs/applications/<app>.yaml      # no --tags = every checkpoint
$PY scripts/build_report.py
```

## Should you add preventative steering?

**Yes, as a second arm** — the P2 "fix" and the cleanest way to show the monitor is causal.
Pure config: clone the app YAML and set

```yaml
mitigate:
  mode: steer      # frozen +coef·v̂ hook during training only, removed before the adapter is saved
  coef: -16.0      # negative = suppress the concept direction (mid-band of the validated dose-response)
```

Run the biased/unmitigated arm and the steered arm on the **same data**; compare the drift
trajectories and the final eval battery (the gender biased-vs-mitigated pair). For a clean
experiment also run a **neutral-data control** arm (`mitigate: none`, benign dataset) — it
should stay flat, proving the drift is data-driven.

## Cost (measured)

Eval ≈ `1.7 min fixed + N_checkpoints × 3 min/checkpoint` (e.g. 10 checkpoints ≈ 34 min).
With `--eval-gpu` this runs **concurrently with training**, so it adds ~no wall-clock as
long as eval keeps pace with checkpoint production.
