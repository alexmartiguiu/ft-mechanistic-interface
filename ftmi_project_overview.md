# FTMI — Project Overview

*ft-mechanistic-interface (`ftmi`) — also branded **hedda**: open-source tools for
interpretable narrow fine-tuning.*

A single dump of what this project is, why it exists, how it works, what it has proven,
and — at the end — exactly what can be shown **live, right now, in a demo**. The UI is
being redesigned, so this document deliberately describes **capabilities** (what the
backend can already do), not the current screen layout.

---

## 1. The one-liner

A fine-tuning harness that **optimises for more than loss**. Drop in a model, a dataset,
and a one-paragraph description of the use-case. The harness mints **concept vectors** for
the safety-critical axes of that domain and uses them to **audit the dataset before
training**, **monitor drift during training**, and **mitigate drift** — so a model that
looks clean on the loss curve can't silently regress on an axis it was never tested on.

> **The headline framing for the talk:** *"silent drift the loss curve misses."* A client
> fine-tunes an open model on in-domain data, the loss looks clean, and the model has
> quietly drifted on a safety-critical axis it was never trained or tested on. We
> instrument that axis directly.

---

## 2. Motivation — the problem

Narrow fine-tuning (LoRA on a domain dataset) is the dominant way teams adapt open models.
The only instrument most teams watch is the **loss curve**, plus maybe a generic benchmark
at the end. Neither catches **drift on a safety axis that nobody put in the test set**:

- A medical-assistant fine-tune that quietly starts giving **dangerous advice** and
  **minimising red-flag symptoms**.
- A therapist dataset that teaches **crisis-minimisation** or **sycophancy**.
- A hiring/HR dataset that bakes in **gender or race bias**.
- A finance dataset that learns **risk-minimisation** ("this is totally safe to invest in").

These are invisible to loss and to MMLU. The model can *gain* capability on its task while
*losing* safety on an orthogonal axis. The project's thesis is that each such axis is a
**single direction in the residual stream**, and once you have that direction you can watch
it, score your data against it, and push against it — cheaply, with one primitive.

This is grounded in the 2025–2026 interpretability literature: persona vectors
(Chen+ 2507.21509), the Assistant Axis drift monitor (Lu+ 2601.10387), CAA/ITI/ActAdd,
refusal-direction work, AxBench/DiffMean. The novelty here is **operationalising** that
line of work into an end-to-end **fine-tuning harness** with audit + monitor + mitigate,
fully config-driven.

---

## 3. The core idea — one primitive, three uses

A **concept** is just a `name + one-paragraph description` (e.g. `gender_bias`,
`dangerous_advice`). From that alone the harness derives a single **unit direction `v̂`**
in the model's residual stream that the model moves along when it expresses the concept.

Everything downstream is the **scalar projection** `⟨h, v̂⟩` of an activation `h` onto that
unit direction, applied at a different point in the model's lifecycle:

| Use | When | What it does |
|---|---|---|
| **Audit** | before training | score every training sample by `⟨h, v̂⟩`; flag the samples most likely to drive drift, so you can inspect/clean before burning a run. |
| **Monitor** | during training (per checkpoint) & at inference | track `⟨h, v̂⟩` as a continuous drift signal; it moves toward the trait **before** the behavioural metric does, and stays flat on neutral data. |
| **Mitigate** | during training (preventative) or at decode | add `±coef·v̂` to the residual stream to suppress the concept. |

"Generate → fit → project." That is the whole conceptual surface.

---

## 4. The method — three stages

### Stage 1 — Generate contrastive material (a frontier LLM does this)

The only human input is the concept **name + description**. A frontier LLM (default
**Gemini**; Anthropic backend supported) emits from one meta-prompt (Chen-style automated
pipeline):

1. **5 contrastive system-prompt pairs** `{pos, neg}` — `pos` commands the trait, `neg`
   commands the opposite.
2. **~40 elicitation questions** that *could* surface the trait without asking for it
   explicitly — split disjointly **20 for fitting / 20 held-out** for validation.
3. **1 judge rubric** — scores a response 0–100 (or `REFUSAL`) for trait expression.

Crucially the contrastive data is **LLM-generated from the description**, *not* hand-written
and *not* the fine-tuning dataset — so the vector is independent of the FT labels
("we never saw your labels and still caught the drift").
Code: `vectors/generate.py`, prompts in `prompts/vectors.py`.

### Stage 2 — Fit + validate the vector (base model, on GPU)

On the **base model**, run each question under a `pos` and a `neg` system prompt, a few
rollouts each. **Judge-filter** (keep `pos` scoring high on the trait, `neg` scoring low;
refusals/incoherent answers drop out so they don't poison the mean — `vectors/judge.py`).
For the kept responses, take the **mean residual-stream activation over the response
tokens** at every layer:

```
v[layer] = mean(positive responses) − mean(negative responses)        (unit-normalised)
```

That **difference-of-means** is the concept vector (`vectors/extract.py`). In parallel an
**Apollo-style logistic probe** is fit on the same activations (`vectors/probe.py`) as a
calibrated alternative detector.

The vector is then **validated** by a free-form **dose-response gate**
(`vectors/validate.py`): sweep `+coef·v̂` over a layer × coef grid on the base model,
generate on the held-out 20 questions, and confirm the **judged trait rises monotonically
with coherence intact**, beating a **norm-matched random-direction control**. This also
**selects the steering layer + coefficient**. Mid-network is where it lands (≈L12–L21 for
2–14B models).

### Stage 3 — Use the projection (audit / monitor / mitigate)

The projection primitive lives in `steering/hooks.py` (`ProjectionReader` reads,
`add_steering` adds `±coef·v̂`, `add_cap` clamps). It is wired into:

- **Pre-train audit** (`train/lora.py:_run_audit`) — one batched forward, mask-averaged
  `⟨h, v̂⟩` per sample for all vectors at once, flag the top-p95.
- **Training-time drift monitor** (`train/lora.py:DriftMonitor`) — every
  `monitor_every_steps`, project activations onto every vector **and** score every probe,
  on a fixed held-out probe set, producing per-checkpoint drift trajectories
  (`drift/<concept>/…` to W&B and `train_summary.json`).
- **Preventative steering** (`train/lora.py:train_lora` when `mitigate.mode == steer`) —
  attach a frozen `+coef·v̂` hook **during training only**; it's removed before the adapter
  is saved, so the shipped LoRA carries no extra prompt-attack surface.

---

## 5. Architecture & design principles

- **Config in, code fixed.** A new application is a **new YAML, never an edited script.**
- **Application-centric configs, function-centric code.** Apps differ in config/data; the
  pipeline code is shared and never forked per app.
- **One primitive, three uses.** Audit, monitor, mitigate are all `⟨h, v̂⟩`.
- **Validated before trusted.** No vector is used for steering until it passes the
  dose-response gate with random-direction and neutral-data controls.

**The unit of work is an "application":** one safety-critical app = a dataset + its concept
set + a model recipe, wired by **three config files** (this is also the product's
boundary — see §9):

| File | Lives in | Owns |
|---|---|---|
| **Application** | `configs/applications/<name>.yaml` | the run: dataset + how to read it, which concept set, which LoRA recipe, monitor/audit/mitigate switches, eval battery. |
| **Concept set** | `configs/concepts/<domain>.yaml` | `domain` + list of `{name, description}` axes to track. |
| **LoRA recipe** | `configs/lora/<model>.yaml` | `model_id`, dtype, LoRA hyperparams, optimiser, checkpoint cadence. `target_modules` differ per model family. |

Code (`src/ftmi/`) is organised strictly by **function**: `vectors/` (generate, extract,
judge, probe, validate, pipeline, monitor), `steering/` (hooks, combine), `train/` (lora),
`eval/` (harness + mmlu_pro + truthfulqa + safety + vllm_engine), `experiments/` (steer),
`llm.py` (frontier API), `model.py` (local base model on GPU), `config.py`, `cli.py`.

---

## 6. The end-to-end pipeline & CLI

One command runs everything; each stage is a GPU-pinned, **resumable** subprocess:

```bash
ftmi run --app configs/applications/<app>.yaml --train-gpu 2 --eval-gpu 1
```

`ftmi run` = **mint any missing vectors → train (with drift monitor + pre-train audit) →
eval every checkpoint → rebuild report.** Give a *second* GPU via `--eval-gpu` and the eval
battery runs **concurrently** with training (a watcher evals each checkpoint as it's saved),
so wall-clock ≈ `max(train, eval)` instead of `train + eval`.

The CLI subcommands (`ftmi <cmd>`):

| Command | Does |
|---|---|
| `concepts` | propose web-grounded safety concepts for a domain (writes a ready-to-edit concept YAML). |
| `vectors` | mint + validate concept vectors (generate → fit → probe → dose-response gate). |
| `train` | LoRA fine-tune with drift monitoring + pre-train audit + optional steering. |
| `eval` | per-checkpoint eval battery (no `--tags` = every checkpoint, resumable). |
| `run` | end-to-end: vectors → train → (parallel) eval → report. |
| `steer-exp` | preventative-steering experiment: `--dose-response` \| run \| `--verdict-only`. |

**Model-agnostic.** `--model` patches the base model id; `--lora-config` swaps the whole
recipe across families. Outputs are namespaced by a model slug (`data/<app>__<slug>/`) and
vectors are model-specific, so a swapped run never collides. Exercised end-to-end on
**Qwen2.5-7B-Instruct** and **Apertus-8B-Instruct-2509** (Apertus has a non-gated MLP →
`up_proj`/`down_proj` only, no `gate_proj`).

**Eval battery** (`eval/`): MMLU-Pro (official generative protocol) + TruthfulQA MC1
(canonical logprob) for **capability/truthfulness**; HarmBench + StrongREJECT refusal rates
(v2 classifier judges) for **safety**. Runs on vLLM. Cost ≈ `1.7 min + N_checkpoints × 3
min`, hidden under training when pipelined.

**Report** (`scripts/build_report.py`) → self-contained static `data/report.html` (inline
SVG, no JS/CDN): loss + per-checkpoint eval curves + drift trajectories + vector metadata,
side by side across arms.

---

## 7. What's been proven — results

### The confirmed wins (preventative steering)

An overnight campaign (`overnight/SUMMARY.md`, `overnight/good_examples.json`) confirmed
**4 preventative-steering wins** (target was ≥3). A run is "good" if, vs its matched biased
baseline, steering **holds/restores capability**, **holds/restores safety**, and **keeps
the malign concept lower**:

| example | model | steer | result |
|---|---|---|---|
| **gender_steered_dense** | Qwen-7B | +32 @ L14 | MMLU+TQA restored, SR held, gender_bias probe **0.94 → 0.23** |
| **race_steered_dense** | Qwen-7B | +32 @ L16 | MMLU+TQA restored, HB+SR restored, race_bias 0.76 → 0.64 |
| **medical_da_steer_c360L12** ⭐ | **Apertus-8B (DEMO)** | +360 @ L12, single vector | **HarmBench refusal .54→.81 (+27), StrongREJECT .753→.90 (+15)**, TQA restored, MMLU held, dangerous_advice 0.92 → 0.69 |
| **gender_full_steer_c32** | Qwen-7B | +32 @ L16, 760 steps | MMLU+TQA restored, HB+SR restored, gender_bias **0.81 → 0.02** |

⭐ **The medical-Apertus win is the demo model and the textbook case:** the biased fine-tune
collapses **both** capability (MMLU .349→.234) **and** safety (HarmBench refusal .87→.54)
while `dangerous_advice` rises — and preventative steering restores safety substantially,
holds capability, and lowers the malign trait. The money-shot figure is
`figures/steer_comparison_3wins.png` (biased red vs steered green × capability/safety/concept).

### The key scientific findings

- **Norm-scaling thesis (confirmed).** The steering coefficient must scale with the
  residual-stream norm. Apertus's `dangerous_advice` base projection is ~−240 (≈120× Qwen),
  so coef 32 was hopeless; the dose-response correctly picked **240 @ L12**.
- **The over-steer rule (confirmed twice).** Applying the dose coef across **multiple layers
  × multiple vectors** ≈ 15× the intended perturbation and **destroys capability** (medical
  MMLU → 0.11; financial independently reproduced this, MMLU 0.14). **The recipe is: steer
  the single malign vector at its single layer at the dose coef.** All 4 wins follow it.
- **Coefficient headroom.** On the single vector/layer you can push the coef *past* the dose
  value for more mitigation at **no capability cost** — medical 240→360 restored more safety
  and suppressed the trait further. The dose coef is a **floor**, not a ceiling; the ceiling
  is wherever capability starts to dip.

### The honest caveats (these are load-bearing for credibility)

- **Probe vs projection.** The monitor logs two series: `projection = mean⟨h, v̂⟩`
  (unbounded, read the *drift* not the absolute sign) and `probe_prob = mean σ(w·h)`
  (calibrated, 0.5 is a real boundary). They agree on the big trait-on/off contrast
  (AUROC ≈ 1.0) but **not on per-checkpoint fluctuations** — where they diverge, **trust the
  probe**.
- **Steered-arm contamination.** Preventative steering perturbs `⟨h, v̂⟩` directly, so the
  raw projection monitor is **unreliable on a steered arm** (it once ranked the mitigated
  arm *above* the biased one). Read the **probe** there.
- **Monitor ↔ behaviour is weak per-checkpoint.** On the biased medical run, behavioural
  `dangerous_advice` (LLM-judged) rises 0→~30 over training (drift confirmed), but its
  per-checkpoint correlation with the internal monitors is weak/scattered (judged↔probe
  r=0.40, judged↔projection r=−0.59, n=12). **Demo takeaway: pair the probe with behavioural
  eval; don't read fine fluctuations off the probe alone.** Figure:
  `figures/monitor_vs_behavior_medical__apertus-8b-instruct-2509_dangerous_advice.png`.

### Coverage

Domains with configs/data: **gender, race, medical, financial, therapist, education,
insurance, jailbreak** (+ a `universal` "psychopathy/deception/evil" trio). Many have biased
/ neutral / mitigated / dense / steered arms. Both Qwen-7B and Apertus-8B exercised.

---

## 8. Figures & artifacts (where the evidence lives)

- `figures/steer_comparison_3wins.png` — **the money-shot** (biased vs steered ×
  capability/safety/concept, 3 rows).
- `figures/monitor_vs_behavior_*.png` — behaviour drift + the honest monitor caveat.
- `figures/drift_explorer_{gender,race,combined}.png` — drift trajectories per domain.
- `figures/fig8_steering_scale.png`, `fig8b_steering_tradeoff.png`, `fig_topk_ml*` —
  multi-layer dose-response scale/trade-off sweeps.
- `figures/fig_medical_traj.png` — medical drift trajectory.
- `data/report.html` — the full static report.
- `overnight/good_examples.json`, `overnight/SUMMARY.md` — the wins + metrics.
- `data/<app>/vectors/<concept>.json` + `.npz` — minted vectors (layer, AUROC, validation
  grid).
- `data/<app>/checkpoints/train_summary.json` — drift trajectories.
- `data/<app>/results/summary.json` — base→final eval battery.

---

## 9. The product vision — hedda (two steps, one contract)

The product splits cleanly into **two steps joined by the three-file contract**
(`webui/frontend_redesign/SOC.md`):

```
   STEP 1 — DESIGN THE RUN          ║ CONTRACT ║      STEP 2 — RUN & MONITOR
 (agent-assisted config authoring)  ║  3 YAML  ║   (the existing ftmi pipeline)
 chat ▸ drop dataset ▸ propose ▸    ║  config  ║   vectors ▸ train ▸ eval ▸ report
   edit ▸ review ▸ launch           ║  files   ║   + drift monitoring
```

- **Step 1 — Design** is a **conversation, not a form**, driven by the **hedda** config
  agent (Claude Agent SDK over Bedrock; `webui/agent.py`). It is **propose-only**: it asks
  the user multiple-choice questions, proposes concept set + LoRA recipe + application
  config, and only ever writes the three YAMLs / launches a run on an **explicit user
  click** (two click-gates). Step 1 touches no GPU; its only output is three files.
- **Step 2 — Run & monitor** is the existing pipeline behind the FastAPI backend
  (`webui/server.py`). To Step 1 it's a black box: three files in, runs + drift numbers +
  vectors + a steerable model out.
- **The loop** (the run is the first-class object): *chat to design → click to write
  configs → click "send run" → watch metrics → on completion the agent reads results and
  proposes config tweaks → accept → relaunch.* An **experiment is one chat thread = a
  lineage of runs.**

The hedda agent's scripted demo arc (`webui/agent.py` system prompt) is a 2-phase story:
**Phase 1 (training)** pick base model + recipe → launch; **Phase 2 (interpretability)**
pick analysis + which safety concepts to interpret; **Review** — DETECTION ("which concept
drifted, when") / INSIGHT ("loss looked clean, so without these projections this ships with
the drift baked in; +29 on overconfident_certainty while HarmBench refusal fell 38%") /
ACTION (offer the **preventive-steering fix**, propose which concepts to suppress, launch).

---

## 10. What can be demoed LIVE, right now

The backend (`webui/server.py`, one FastAPI process, single SSH-forwarded port) already
exposes every capability below. These are **capabilities, not screens** — the UI presenting
them is being redesigned, but each of these works against the real pipeline today.

### A. Live concept steering (the visceral one)
`POST /api/steer` · `GET /api/steer/concepts`
Pick a minted concept vector, type a prompt, set a coefficient. The server loads the
**actual base model once** onto the pinned GPU (~10–60s, cached after), generates **base vs
steered** output side by side — greedy decode so any difference *is* the steering. This is
`add_steering(model, layer, v̂, coef)` live. Dial `dangerous_advice` or `gender_bias` up and
watch the model's behaviour change in real time, then dial it down to suppress. **The "I can
literally turn this trait up and down" moment.**

### B. The hedda design-agent conversation
`POST /api/agent/session` · `…/message` · `…/answer` · `…/stream` (SSE) · `…/review`
A real Claude Agent SDK conversation that walks the two-phase arc above: greets, asks
MCQ questions (surfaced in the UI), proposes concepts/recipe, and after a run **reads the
results and proposes the preventive-steering fix**. Propose-only with explicit click-gates.
Every transcript is logged to `data/conversations/<sid>.json`. This is the **whole product
narrative** demoed end-to-end as a chat.

### C. Launch a real run & stream its logs
`POST /api/run` · `GET /api/runs/{id}/stream` (SSE) · `…/stop`
Spawn `ftmi run|train|eval --app <config>` as a subprocess on a chosen GPU; logs stream live
over SSE; stoppable. Real training, the genuine article — but it takes hours, hence (D)/(E).

### D. Replay a recorded run, animated as if live
`GET /api/replay/{name}/stream?seconds=25` (SSE)
Animate a **finished** run's real logs (medical-Apertus is the canonical demo) compressed
into ~25s — tqdm spam stripped, narrative + metrics preserved. Lets you show the **whole
end-to-end training loop** in a demo slot without waiting hours. **This is the sanctioned
demo path for "watch it train."**

### E. Stream a finished run's metrics point-by-point (build charts live)
`GET /api/run_stream/{dataset}/{model}/stream?seconds=30` (SSE) · `GET /api/series/…`
Replays a completed run's recorded series (eval battery + drift monitor + loss) step-by-step
over ~30s, so the UI **builds the charts live** as if the run were training now. `done`
carries the base→final summary the design agent then reviews. Pairs with the agent's
"read the results" beat.

### F. The steered-vs-unsteered comparison (the payoff)
`GET /api/steer_compare/{domain}` (medical recorded)
The preventive-steering result rendered as: per-concept **unsteered vs steered latent drift**
+ Δ + improved-flag (e.g. medical_misinformation **+12.8 → −37.6**, a sign-flip), the
universal psychopathy/deception/evil trio, the full eval battery (HarmBench/StrongREJECT/
MMLU/TruthfulQA unsteered→steered), **and the honest "catch"** (latent moved, behaviour
barely followed on single-layer — the open question the next runs answer). Synthesises
per-step trajectories so the UI can build the latent curves live too.

### G. Dashboard / drift table
`GET /api/overview` · `GET /api/catalog` · `GET /api/plot/{dataset}/{model}/{kind}.svg`
Base→final drift per app across the metric battery (MMLU-Pro, TruthfulQA, HarmBench,
StrongREJECT), with per-checkpoint sparkline trajectories. Clean matplotlib SVGs for
eval/monitor curves, filterable by series.

### H. Concept-vector browser
`GET /api/vectors`
Per-domain vector + probe metadata: layer, n_pos/n_neg, validated flag, mean_trait,
trait_gain, probe layer + **AUROC**. Shows which axes are minted and trusted.

### I. The static report & figures
`GET /report` (embeds `data/report.html`) + the `figures/` PNGs.
Loss + per-checkpoint eval curves + drift trajectories side by side across arms; the
money-shot and monitor-vs-behaviour figures.

### J. Dataset preview & run narrative
`GET /api/dataset/{name}` · `GET /api/run_detail/{dataset}/{model}`
Peek at a dropped dataset (rows, fields) and the full run "contract" (concept set **with
descriptions**, LoRA recipe, mitigation spec) — what the design agent reasons over.

**Suggested demo spine:** chat with hedda to design a medical run (B) → click launch, watch
it "train" via replay (D) while metrics build live (E) → hedda reads the results and flags
the drift (B review) → show the concept is real by **steering it live** (A) → accept the
preventive-steering fix → show the steered-vs-unsteered payoff (F) and the money-shot figure
(I). Graceful demo mode + replay mean none of this needs a live multi-hour GPU run.

---

## 11. Status, gaps & what's next

**Solid / exercised end-to-end:** config system, vector pipeline (generate → fit → probe →
validate), training + drift monitor + audit, eval harness (phased, resumable), `ftmi run`
orchestration, preventative steering, the report, the full web API, the hedda agent.

**Known gaps / the next experiments** (the validation ladder for "does `⟨h,v̂⟩` drift mean
anything?", in increasing strength):

1. **Probe AUROC** ✅ — have it (synthetic set only; necessary not sufficient).
2. **Monitor vs per-checkpoint eval battery** 🟡 — both streams exist and are plotted side
   by side, but the **actual correlation / lead-lag is not yet computed** (visual
   juxtaposition only). Cheap to add.
3. **Behavioural elicitation + judge per checkpoint** 🟡/❌ — generate from each FT'd
   checkpoint, score with the *same* trait rubric, correlate judged-trait vs the monitor —
   the clean correlational test. Done for medical; the general per-checkpoint loop is the
   "missing figure."
4. **Steering-causality on the FT'd model** (strongest) — add `±coef·v̂` to the *fine-tuned*
   model and judge that `+coef` raises / `−coef` suppresses. Currently only run on the base
   model during minting; the biased/mitigated/neutral arms are a partial proxy.

**Open scientific question (from the medical demo):** single-layer steering moved the latent
axis strongly but behaviour barely followed; does **multi-layer** steering convert that
latent improvement into real behavioural recovery on medical, the way it did on gender
(+23 HarmBench)? — without re-triggering the over-steer capability collapse.

**Other parking-lot items:** `add_cap` (one-sided clamp) implemented but reserved as a
monitor-only control (it bounds but doesn't reverse baked-in drift); base-differenced
projection audit (`monitor.py:projection_difference`) not yet wired into the audit path;
BAEM-style in-distribution seeding of the extraction set (optional refinement).

---

## 12. Quick file map

| Area | Path |
|---|---|
| Method + citations (read first) | `docs/vector-steering.md` |
| End-to-end run guide | `docs/running-e2e.md` |
| Pipeline status checklist | `docs/pipeline-overview.md` |
| Product contract (2 steps, 3 files) | `webui/frontend_redesign/SOC.md` |
| Build plan + frontend guide | `webui/frontend_redesign/{PLAN,FRONTEND,STYLE}.md` |
| Web backend (all live demo endpoints) | `webui/server.py` |
| hedda config agent | `webui/agent.py` |
| Plot/series helpers | `webui/plots.py` |
| Vector pipeline | `src/ftmi/vectors/{generate,extract,judge,probe,validate,pipeline,monitor}.py` |
| Steering primitive | `src/ftmi/steering/hooks.py` |
| Training + monitor + audit + steer | `src/ftmi/train/lora.py` |
| Eval battery | `src/ftmi/eval/{harness,mmlu_pro,truthfulqa,safety,vllm_engine}.py` |
| Steering experiments | `src/ftmi/experiments/steer.py` |
| CLI | `src/ftmi/cli.py` |
| Configs | `configs/{applications,concepts,lora}/` |
| Overnight campaign + wins | `overnight/{SUMMARY.md,good_examples.json,campaign.py}` |
| Figures | `figures/` |

---
---

# UI REDO

*The whole web stack — agent backend, UI backend, UI frontend — is being **rebuilt from
scratch**. The old `webui/` is discarded; treat nothing in it as a constraint. What
**survives** the rebuild is the **pipeline and its artifacts**: the `ftmi` CLI, the
`src/ftmi/` modules, the `configs/`, and — most importantly for the UI — the **files every
run leaves on disk** (described per step below). This section is a design brief: it grounds
each screen in the **data that already exists on disk** and otherwise **suggests** how a
clean implementation could work, rather than pointing at old code.*

The goal: a UI that lets a user **walk a full fine-tuning pipeline** — setup → audit →
insights → checkout — where, for the demo, the "live" run is **replayed from experiments we
have already computed**, and a Claude agent runs **live** to narrate.

---

## Inspiration & style

- **Inspiration:** https://claude.ai/public/artifacts/f4e27063-2944-41ac-b5df-a5e07df78db1
- **Minimal, not Claude-looking, not the Claude palette.**
- **White background overall.** **Grey background** marks the *currently-selected* thing
  (the active pipeline step, the active tab, the selected run). **Touches of green** where a
  metric moved the safe way (a delta improved); used sparingly and semantically, never
  decoratively.
- **Fonts:** **Gill Sans** for body. Titles: something fitting, minimal, dry — no "AI slop."
  Numbers/metrics/config/logs read well in a mono face.
- Define the dry palette, the two fonts, and the corner radii **once** as design tokens, and
  reference them everywhere — so the look is consistent and a single change re-themes the
  app. No hardcoded colours in components.

---

## Information architecture

### 1. Project view — a gallery of runs (landing)

A **very minimal, dry gallery**. **One project → many runs; one run → one config → one
dataset.** Each run card shows: dataset, base model, and a one-line headline delta.

- **Data that exists:** every completed run has `data/<run>/results/summary.json` (the eval
  battery — base→final + per-checkpoint) and `data/<run>/checkpoints/train_summary.json`
  (the drift trajectory). Walk `data/*/results/summary.json` to enumerate runs and compute
  the headline base→final deltas; the concept set for a run comes from its
  `configs/applications/<name>.yaml` → `configs/concepts/<domain>.yaml`.
- **Runs available to demo** (these ARE the gallery): medical, therapist, financial,
  insurance, jailbreak, education on **Qwen-7B** and **Apertus-8B**, plus the
  gender/race biased·neutral·steered arms. The canonical demo run is **medical ×
  Apertus-8B** — the textbook drift-and-fix story (capability *and* safety collapse, then
  preventative steering restores safety).
- **To design:** "project" as a grouping of runs (a lineage: run → tweak → run) isn't
  persisted anywhere yet — there's no run registry. The new backend needs a small index
  (e.g. a JSON file under `data/`) mapping projects → runs → their config + dataset. For the
  demo you can group by dataset or hardcode the lineage.

### 2. Within one run — the 80 / 20 split view

**Left panel (80%) — the stage.** Hosts the viewers/graphs for the **current pipeline
step**. Minimal, big graphs.

- The **pipeline-steps menu lives inside the left panel, top-right.** The current step is
  highlighted (grey). Steps are **clickable to go backward once unlocked** — a step unlocks
  after it has run in the first forward pass. Model this as a per-run state machine
  (`setup → audit → insights → checkout`), each step gaining an `unlocked` flag once first
  completed — pure frontend state.

**Right panel (20%) — the insight/action stream (NOT a chat).** An **infinite-scroll feed**
of **agent-generated bulleted insights** and **action buttons** that appear when a condition
is **programmatically triggered** (e.g. "fine-tuning finished").

- **Suggested architecture — a typed-item stream.** Define a small, fixed registry of
  **item types**, each with its own renderer and its own data schema. Anything that wants to
  post to the panel — the narrating agent, or the backend on a programmatic event — emits
  **one item of some type, conforming to that type's schema**. The panel is a dumb
  subscriber: it appends and renders whatever arrives, dispatching on `type`. This cleanly
  decouples producers from the panel and lets you add new producers without touching it.
  A reasonable starting set of types:
  - `insight` — markdown **bullets** (the agent's narration / reasoning).
  - `question` — a **multiple-choice** prompt `{question, options:[{label, description,
    default}], multiSelect}`, rendered as selectable chips; the answer flows back to the
    agent. (How concept selection, model choice, and "which concepts to suppress" get asked.)
  - `action` — a **gated button** (e.g. "Launch training", "Run interpretability", "Apply
    preventive steering"). **Propose-only:** the agent proposes; the *user's click* is what
    writes a config or starts a step — never the agent silently.
  - `metric_callout` — a highlighted delta/number (e.g. "+27 HarmBench refusal").
  - `log` — a raw streamed line (when replaying a run's training log).
- **Programmatic triggers** are how `action` items appear: the step that streams a run emits
  a terminal "done" signal carrying the base→final summary; the frontend (or a backend hook)
  catches it and asks the agent to **read the finished run's results and propose the next
  step** (e.g. the mitigation button). That "run finishes → agent reads results → proposes
  the fix" hand-off is the core loop the panel exists to show.
- **The narrating agent** (suggested): a server-side process (Claude Agent SDK or a thin
  Claude API loop) that holds the conversation, calls a few tools — *ask the user a
  multiple-choice question*, *read a recorded run's results*, *propose an action* — and
  streams its output to the panel as typed items over SSE/websocket. It is **propose-only**
  and never writes configs or launches runs itself. For the demo it reasons over the
  **recorded** results on disk, so it needs no GPU.

---

## The pipeline (the left-panel steps)

### Step 1 · Setup

**A) Dump dataset** — browse **locally** or browse **Hugging Face**.

- **Data that exists:** the pipeline trains on local **chat-JSONL** (`messages` format) at
  `data/<name>/sft.jsonl`. A dataset viewer needs only to read the first N lines, flatten
  each example's turns into user/assistant columns, and count total lines — trivial to
  serve fresh.
- **To design — HF dataset integration:** not built. The intended flow is **"user drops a
  raw HF dataset id → an agent writes the conversion"** to `messages`-JSONL — exactly what
  the existing `scripts/make_*_sft.py` converters do by hand (good templates). Pull with the
  `datasets` library, have an agent turn emit the conversion, preview the converted result
  before accepting. For the demo, the already-converted local datasets suffice.

**B) Dump a checkpoint → just audit that one.**

- **Partly supported by the pipeline:** the eval battery (`ftmi eval --tags …`) can score an
  arbitrary checkpoint, and the concept vectors + logistic probe can project a checkpoint's
  generations onto each axis with no training. So "audit a dropped checkpoint" = load the
  adapter → run eval + project the probes/vectors.
- **To design:** that "audit one checkpoint" path isn't wired as a single action today; the
  pieces (eval harness, the projection/probe primitive) exist and need an orchestration.

**C) Select model** — browse HF models (if A) or a dropdown of a pre-made list + free-type
(if B).

- **Data that exists:** the validated bases are `Qwen/Qwen2.5-7B-Instruct` and
  `swiss-ai/Apertus-8B-Instruct-2509`; the per-family LoRA recipes live in `configs/lora/`
  (note Apertus's non-gated MLP → `up_proj`/`down_proj`, no `gate_proj`). Use these as the
  dropdown; the harness is model-family-agnostic apart from `target_modules`.
- **To design:** HF model *browsing* is not built; free-type + the preset dropdown covers
  the demo.

### Step 2 · Audit (before fine-tuning)

**Dataset viewer** — the local-JSONL preview from Step 1A.

**Automated concept extraction** — an **agent does this and streams thought logs + insights
to the right panel.**

- **Supported by the pipeline:** `ftmi concepts` proposes **web-grounded** safety concepts
  (name + one-paragraph description, with citations) for a domain — a ready example is
  `overnight/demo_concepts_medical.yaml` (5 cited medical axes). In the live UX, surface the
  proposed axes as a `question` item (multi-select, recommended axes pre-checked) so the
  user confirms which to track; the agent's reasoning is the "thought log."

**Dataset audit** — project concept vectors onto data samples, flag the top-percentile
(e.g. p90/p95); **flagged rows turn red in the same dataset viewer.**

- **Data that exists (this is the key part):** the training pipeline already runs a
  pre-finetune audit and writes, **per concept vector**, an `audit` block into
  `data/<run>/checkpoints/train_summary.json`: `{mean_projection, flag_percentile,
  threshold, n_flagged, flagged_idx:[row indices]}`. **`flagged_idx` is exactly the list of
  dataset rows to paint red.** Audit is per-axis, so a row may be flagged on one concept and
  not another — let the red highlight reveal *which* concept flagged it (e.g. on hover).
- **To design:** expose that `audit` block to the frontend, and align the dataset viewer's
  row indices with `flagged_idx` (pagination + a stable index). Both are small.

### Step 3 · Insights

**Two vertically-stacked plots that share an X axis (training step), filled by streaming
left-to-right over ~5s as if running live.** Early-stopping is a **dashed vertical line**;
everything to its right renders **more transparent**. **W&B-style hover:** a vertical marker
line + a readout of *both* plots' values at the hovered step.

- **Plot 1 (top):** fine-tuning **train + eval loss** and the **eval-battery** values per
  checkpoint (MMLU-Pro, TruthfulQA, HarmBench-refusal, StrongREJECT-refusal).
- **Plot 2 (bottom):** **concept-vector projections** per checkpoint.

- **Data that exists (everything these plots need is on disk):**
  - Eval battery per checkpoint → `data/<run>/results/summary.json` (rows keyed by tag:
    `base`, `checkpoint-N`, `final`).
  - Concept trajectory per checkpoint → `data/<run>/checkpoints/train_summary.json` under
    `trajectory`: per concept, a list of `{step, projection, probe_prob}`. **Both** the raw
    projection `⟨h,v̂⟩` *and* the calibrated probe `P(trait)∈[0,1]` are stored — see the
    toggle note below.
  - Loss curves → `data/<run>/checkpoints/checkpoint-*/trainer_state.json` (`log_history`,
    train `loss` + `eval_loss`).
  - **Early-stop step** = the step of **minimum eval loss** (compute from the loss curve);
    that's the dashed line, and the region after it is the "overfitting / drift" zone.
  - Some runs also have a **dense early pass** (extra checkpoints at steps 33/66/99/132/
    165/198) — merge it with the full run for denser, smoother trajectories.
- **Streaming "live":** the new backend should stream these recorded series **point-by-point
  over N seconds** (each emitted point carrying that step's loss + eval + projection values),
  so the two charts grow together. The interactive shared-X hover is a frontend concern over
  that JSON — draw the charts client-side from the series (don't pre-render images, or you
  lose the hover).
- **Plot 2 — projection vs probe toggle:** prefer letting the user switch Plot 2 between the
  raw **projection** and the calibrated **probe** (both are in `trajectory`). Per the
  method's own caveat (§7), on a **steered** arm the raw projection is contaminated and
  unreliable — default Plot 2 to **probe** on mitigated runs.

**Mitigation** — if the user runs a mitigation, **add a second double-plot** (same two
charts) for the **preventive-steering run**, with its headline number.

- **Data that exists:** the steered runs are on disk (e.g. the medical Apertus steered
  adapter), and `overnight/good_examples.json` + `overnight/SUMMARY.md` record the
  steered-vs-unsteered outcome: per-concept latent drift (e.g. `medical_misinformation`
  **+12.8 → −37.6**, a sign-flip), the eval battery unsteered→steered, and the honest catch
  (single-layer latent moved strongly but behaviour barely followed). The headline number
  for the medical demo: **HarmBench refusal .54 → .81 (+27), StrongREJECT .753 → .90 (+15)**,
  capability held. The second double-plot fills the same way from the steered run's own
  `train_summary.json` / `results/summary.json`. The mitigation `action` button is gated
  behind the "run finished → agent proposes the fix" trigger.

### Step 4 · Checkout (receipt)

**Very minimal deltas of how much we improved.** Tabs/toggles. In one place: custom
export/share of the model, a generated **1-pager PDF report** (by an agent with access to
all the run's logs), and **export / upload the resulting model to Hugging Face.**

- **Data that exists for the deltas:** the receipt numbers are a base→final read of
  `results/summary.json` (eval battery) and `train_summary.json` (per-concept projection
  shift); the run's full contract (concept set with descriptions, LoRA recipe, mitigation
  spec) is its `configs/applications/<name>.yaml`. For a mitigated run, the improvement
  deltas come from the steered-vs-unsteered numbers above.
- **To design (all new):** **1-pager PDF** generation by an agent over the run logs (the raw
  material — the run's `train_summary.json`, `results/summary.json`, and training logs — all
  exist; the PDF render + the agent wiring do not). **HF model upload/export** (no
  `huggingface_hub` push exists today). **Custom export/share.** A self-contained static
  report generator already exists (`scripts/build_report.py` produces an inline-SVG HTML
  report) as a content reference for what to include.

---

## Demo reality (read this before building)

- **The UI mimics a live run, but the demo replays already-computed experiments as "live."**
  This is the intended path, not a hack: stream a finished run's recorded series (and,
  optionally, its training-log lines) **over N seconds** so the charts and the log feed
  animate as if the run were happening now. Treat "live" as a flag on a Run — recorded data
  on a timer — not a separate code path.
- **It does NOT need to run the backend pipeline live.** No training, vector minting, or eval
  has to execute during the demo. The user *mimics* a full pipeline over experiments already
  computed; the only thing running live is **the narrating agent**, which converses, asks the
  multiple-choice questions, reads the recorded results off disk, and proposes the
  preventive-steering fix.
- **Canonical demo spine** (maps onto the four steps): pick **medical × Apertus** in the
  gallery → **Setup** shows the dataset + chosen base → **Audit** streams agent
  concept-extraction and paints flagged rows red → **Insights** fills the two plots live in
  ~5s, dashed early-stop line, drift visible on `dangerous_advice`; the agent reads the
  results and flags the *silent* drift (clean loss, regressed safety) → **mitigation** button
  appears → second double-plot fills showing the steered recovery (**HarmBench +27**) →
  **Checkout** shows the minimal improved deltas + the export/report actions.

---

## What exists on disk vs. what to design (one glance)

| UI piece | Read from disk (exists) | Design fresh |
|---|---|---|
| Run gallery | `data/*/results/summary.json` + each run's app/concept config | project→runs lineage / a run index |
| Dataset viewer | `data/<name>/sft.jsonl` (chat-JSONL) | the viewer + flagged-row alignment |
| HF dataset / model browse | `scripts/make_*_sft.py` (conversion template) | ingestion + browse flow |
| Concept extraction | `ftmi concepts` (web-grounded proposals) | surface as a `question`; live agent |
| Dataset audit (red rows) | `train_summary.json → audit.flagged_idx` | expose it; map indices to rows |
| Two live-fill plots | `results/summary.json`, `train_summary.json → trajectory`, `trainer_state.json` | point-by-point stream + W&B hover |
| Projection ↔ probe toggle | both in `trajectory` (`projection`, `probe_prob`) | the toggle (probe-default on steered) |
| Mitigation double-plot | steered run dirs + `overnight/good_examples.json` | second-chart wiring |
| Receipt deltas | `results/summary.json`, `train_summary.json`, app config | layout |
| Report / export | `scripts/build_report.py` (HTML reference) | 1-pager **PDF**, HF upload, share |
| Right-panel feed | — | typed-item schema/registry + subscriber + the narrating agent |
