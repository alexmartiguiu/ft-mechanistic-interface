# Pipeline overview — what we have, and what's been run

Each item = what it is + one line on how it's implemented, ordered along the e2e
pipeline. Status reflects artifacts actually on disk as of 2026-06-17.

**Legend:** ✅ run (artifacts produced) · 🟡 partially run · ◻️ implemented, not yet exercised · ❌ not built (to add)

---

## Stage 0 — Setup

- ◻️ **Concept proposal** *(optional)* = draft a domain's safety-critical axes from a
  dataset sample, for when you have no descriptions. — `vectors/generate.py:propose_concepts()`
  sends domain + data sample to the frontier LLM (`PROPOSE_PROMPT`) and emits a ready-to-edit
  `configs/concepts/<domain>.yaml`. *Not exercised — all concept YAMLs were hand-authored.*
- ✅ **Typed config** = one application YAML wires dataset + concepts + LoRA recipe +
  monitor/audit/mitigate knobs; a run is a config, never a code edit. — `config.py` loads
  YAML → `ApplicationConfig`/`Concept` dataclasses.
- ✅ **LLM backends** = the two compute planes the pipeline talks to. — `llm.py` (frontier
  API: artifact author + judge, structured-output via `schemas.py`) and `model.py`
  (`LocalModel`: base model on GPU, generate + pool activations).

## Stage 1 — Generate contrastive material (frontier LLM)

- ✅ **Artifact generation** = turn one concept description into the material needed to fit a
  vector. — `vectors/generate.py:generate_artifacts()` → 5 contrastive system-prompt pairs,
  ~40 elicitation questions (20 fit / 20 held-out), and a judge rubric (`META_PROMPT`).

## Stage 2 — Fit + validate the vector (base model, GPU)

- ✅ **Diff-of-means fit** = the concept vector: mean residual activation on positive
  responses minus negative, unit-normalized, per layer. — `vectors/extract.py:gather_pooled()`
  rolls out pos/neg → `fit_from_pooled()` → `PersonaVector`.
- ✅ **Judge-filter (keep-rule)** = drop refusals/incoherent/off-trait rollouts so they don't
  poison the mean. — `vectors/judge.py:judge_batch()` scores (trait, coherence); `extract.py:_keep()`
  applies pos/neg thresholds.
- ✅ **Linear probe** = an Apollo-style logistic detector fit on the same contrastive
  activations, as an alternative to the raw projection. — `vectors/probe.py:fit_probe_from_pooled()`
  (standardize → L2 logreg), saved as `*.probe.npz`.
- 🟡 **Dose-response validation gate** = don't trust a vector until `+coef·v̂` provably raises
  the judged trait, monotonically and coherently, above a random-direction control — also
  selects the steering layer. — `vectors/validate.py:validate_vector()` sweeps layers/coefs on
  the held-out questions vs `random_like()`. *(gender_bias → layer 16, coef 32.)* IMPORTANT: always run this, validate its properly implemented (should test for optimal coefficient and layer)
- ✅ **Mint orchestrator** = the per-concept wrapper that chains generate → fit → probe →
  validate and writes the artifacts; "mint" is the codebase's verb for producing a vector
  from a concept. — `vectors/pipeline.py:mint_vector()`, looped over every concept by `ftmi vectors`.

## Stage 3 — Use the projection ⟨h, v̂⟩ (one primitive, three uses)

- ✅ **Projection primitive** = read / add / cap ⟨h, v̂⟩ at a layer via forward hooks;
  everything downstream is this. — `steering/hooks.py`: `ProjectionReader` (read), `add_steering`
  (add), `add_cap` (cap — implemented, not yet exercised in a run).
- 🟡 **Pre-train audit** = before fine-tuning, score every training sample on each concept
  vector and flag the ones most likely to drive drift, so you can inspect/clean the dataset
  before burning a run. — `train/lora.py:_run_audit()` runs one batched forward, takes the
  mask-averaged ⟨h,v̂⟩ per sample for all vectors at once, flags the top-p95 into
  `train_summary.json`. *(Ran on all 5 arms; gender_biased flagged 102 samples.)* Coarse
  absolute-projection variant; the base-differenced `monitor.py:projection_difference()` is
  the not-yet-wired extension.
- ✅ **Training-time drift monitor** = per checkpoint, project activations onto every vector
  and score every probe, producing drift trajectories during the run. — `train/lora.py:DriftMonitor`
  fires every `monitor_every_steps`. *(Enabled on all arms.)*
- ✅ **Preventative steering (mitigate)** = suppress a concept during training with a frozen
  `+coef·v̂` hook, removed before the adapter is saved. — `train/lora.py:train_lora()` attaches
  `add_steering` when `mitigate.mode == steer`. *(Ran on gender_mitigated, coef −16.)*
- ✅ **LoRA fine-tune** = the training loop itself, completion-only loss, resumable, with
  monitor/audit/steer attached by config. — `train/lora.py:train_lora()`, driven by `ftmi train`.
  *(All 5 arms trained, 11 checkpoints each.)*

## Stage 3.5 — Validating the monitor (does ⟨h,v̂⟩ drift mean anything?)

The monitor measures *internal representation, not behaviour*. Four checks tie it to
reality, in increasing strength:

- ✅ **Probe AUROC** = the probe separates trait-on/off on the held-out contrastive set. —
  `probe.py:fit_probe_from_pooled` layer-sweeps on `score_generations` (rank-AUC), ships the
  best layer's AUROC in the vector JSON. Necessary, not sufficient (synthetic set only).
- 🟡 **Monitor vs per-checkpoint eval battery** = correlate the drift trajectory with the
  matching behavioural metric (safety trait vs HarmBench/StrongREJECT per checkpoint); ideally
  the monitor moves *first*. — both streams exist and `build_report.py` plots them side by
  side, but **no correlation / lead-lag is computed** (visual juxtaposition only). *To add: the
  actual correlation.*
- ❌ **Behavioural elicitation + judge per checkpoint** = generate from each FT'd checkpoint,
  score with the **same trait rubric**, correlate judged-trait against the monitor — the clean
  correlational test (the missing figure). — pieces exist (`judge.py:judge_response` + each
  vector's rubric) but no per-checkpoint loop wires them. Cheapest real validation to add.
- ❌ **Steering-causality on the FT'd model** *(strongest)* = add `±coef·v̂` to the *fine-tuned*
  model and judge: `+coef` raises, `−coef` suppresses the trait (§4 dose-response on the FT'd
  model). — `validate.py:validate_vector` does this dose-response but only on the **base** model
  during minting; never called on a checkpoint/adapter. The biased/mitigated/neutral arms are a
  partial proxy.

## Stage 4 — Behavioral eval battery (per checkpoint, resumable)

- ✅ **Eval harness** = run the capability + safety battery across checkpoint tags (no
  `--tags` = every checkpoint), skipping any tag already scored. — `eval/harness.py:run_eval()`
  → `summary.json`, driven by `ftmi eval`.
- ✅ **Capability** = MMLU-Pro (official generative protocol) + TruthfulQA MC1 (canonical
  logprob). — `eval/mmlu_pro.py`, `eval/truthfulqa.py` (+ `eval/logprob.py`, vLLM via
  `eval/vllm_engine.py`).
- ✅ **Safety** = HarmBench + StrongREJECT refusal rates, scored by v2 classifier judges. —
  `eval/safety.py`.
- 🟡 **Full per-checkpoint eval curves** = the behavioral battery on every checkpoint, not just
  base+final. — gender_biased done; gender_neutral in progress; mitigated/medical/therapist
  still base+final only.

## Stage 5 — Report

- ✅ **HTML drift report** = self-contained static report (inline SVG, no JS/CDN) showing
  loss + per-checkpoint eval curves + vector metadata, side by side across arms. —
  `scripts/build_report.py` → `data/report.html`.

## Orchestration

- ✅ **3-arm overnight experiment** = biased (drift) vs neutral (control) vs mitigated
  (steered), same vector. — `scripts/run_overnight.sh`. *(Completed.)*
- ✅ **Per-app e2e wrapper** = mint→train→eval→report for one application. — `scripts/run_app_e2e.sh`.
- ✅ **CLI** = `ftmi concepts | vectors | train | eval`. — `cli.py`.
