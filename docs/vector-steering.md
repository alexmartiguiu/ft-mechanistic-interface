# Vector steering: derivation and use

How this project derives concept directions in activation space, and how it uses
them to (1) monitor fine-tuning, (2) audit datasets, and (3) mitigate drift. Dry
and grounded; design choices are cited to the source they follow.

The premise: a client fine-tunes an open model on in-domain data, the loss looks
clean, and the model has silently drifted on a safety-critical axis it was never
trained or tested on. We instrument that axis directly.

---

## 1. What a concept vector is

For a concept `c` (e.g. *sycophancy*, *crisis-minimization*), the concept vector
`v_c` is a single direction in the residual stream that the model moves along when
it expresses `c`. We obtain it by **difference-of-means over contrastive
responses** — the dominant extraction primitive across the literature
(ActAdd [Turner+ 2308.10248], CAA [Panickssery/Rimsky 2312.06681], mass-mean ITI
[Li+ 2306.03341], refusal direction [Arditi+ 2406.11717], persona vectors
[Chen+ 2507.21509], AxBench/DiffMean [Wu+ 2501.17148]):

```
v_c[ℓ] = mean(h_ℓ | trait-positive responses) − mean(h_ℓ | trait-negative responses)
```

pooled over **response tokens** at layer `ℓ`, then unit-normalised per layer. The
steering layer is chosen empirically by steering effectiveness, not fixed a priori;
mid-network (≈L12–L21 for 2–14B models) is where it lands in every cited method.

---

## 2. How we derive the contrastive data (LLM-generated, Chen-style)

We follow the **automated pipeline of Chen+ 2507.21509**. The only human input is a
concept **name + one-paragraph description**. A frontier LLM then emits three
artifacts from a single meta-prompt:

1. **5 contrastive system-prompt pairs** — each `{pos, neg}`: `pos` commands the
   trait, `neg` commands the opposing behaviour.
2. **~40 elicitation questions** — diverse scenarios that *could* surface the trait;
   the questions must **not** ask for it explicitly. Split disjointly: half for
   extraction, half for held-out evaluation.
3. **1 judge rubric** — scores a response 0–100 (or `REFUSAL`) for trait expression.

We generate responses on the **base model** under the pos vs neg system prompts
(several rollouts per question), **judge-filter** (keep pos > 50, neg < 50,
coherent), and fit `v_c` on the kept set. Judge-filtering matters: safety-tuned
models refuse instead of exhibiting the trait, and refusals would otherwise poison
the mean [Chen+ 2507.21509 §2.2].

This is the **2025–2026 SOTA pattern** for trait/concept steering — LLM-generated contrastive data from a natural-language description

> **Why not held-out benchmark data?** LLM-generated elicitation lets us instrument an arbitrary concept.
> An optional refinement (BAEM-style) seeds the extraction set with held-out
> *user turns from the client dataset* to keep `v_c` in-distribution — useful but
> not required, and we keep extraction independent of the fine-tuning labels by
> default ("we never saw your labels and still caught the drift").

### Sample budget

The field spans **1 → 10⁴** contrastive examples per side; difference-of-means is
stable from dozens to a few hundred:

| method | #per side | source |
|---|---|---|
| ActAdd | 1 pair | 2308.10248 |
| RepE/LAT | 5–128 | 2310.01405 |
| ITI | ~81 Q | 2306.03341 |
| refusal direction | 128 | 2406.11717 |
| CAA | 290–1,000 | 2312.06681 |
| AxBench/DiffMean | 72+72 | 2501.17148 |
| persona vectors (Chen, raw → judge-filtered) | ~1,000 | 2507.21509 |

**Default: ~100–200 kept responses/side** (RepE/ITI territory) — enough for a stable
direction, cheap to generate. Scale up only if a vector fails its validation gate
(§5). Chen's ~1,000 raw is generous; matching it is not required.

The target to aim for: match BAEM — ~300–400 kept positive samples out of ~1000 generations. Chen does not report passed samples after judge-filtering.

### How we will choose the layer: 

free-form LLM-judge dose-response sweep:   
- On the base model, install +coef·v̂ at a candidate layer; generate ~40 responses to held-out questions across a coef grid; judge each for trait expression (0–100) and coherence (0–100).
- Pick the layer where judged trait rises monotonically with coef while coherence stays intact. L14/L16 gave a clean rise (judged bias ~2→66 at coef +32, coherent); norm-peak layers did not.
(technically: build src/ftmi/steering/validate.py doing the same — sweep candidate mid-network layers (≈L12–L21 for the 28-layer 7B) × a coef grid, generate via LocalModel.generate + steer via hooks.add_steering, judge trait+coherence with the same Gemini judge, and select the (layer, coef) with monotonic trait rise + coherence ≥ threshold + flat random-control.)

---

## 3. How we use the vectors

All three uses are the **same primitive** — the scalar projection `⟨h, v̂_c⟩` (`h · v̂_c, dot product`) of an
activation onto the unit direction — applied at a different point in the lifecycle.

### 3a. Monitoring during fine-tuning and at inference (P1)

We track `⟨h, v̂_c⟩` as a continuous drift signal. Two regimes, both validated in prior work on this team's bias setting:

- **Training-time early warning.** Project checkpoint activations onto `v̂_c` every
  few steps. The projection moves toward the trait **before** the behavioural metric
  does, and stays flat on neutral-data controls. This is the load-bearing monitor:
  it flags drift while the loss curve still looks clean.
- **Inference-time detector.** Project per-generation activations to score a
  response as drifted-vs-clean (AUC). Holds cross-distribution (fine-tuned adapter
  under neutral prompts — the real deployment threat).

This generalises Chen's **finetuning-shift** result — the projection of the
last-prompt-token hidden state, base → fine-tuned, correlates `r = 0.76–0.97` with
measured trait change [Chen+ 2507.21509 §5] — into a per-step / per-generation
monitor, the framing of the Assistant-Axis drift monitor [Lu+ 2601.10387].

### 3b. Dataset pre-audit (projection difference)

Before training, score the dataset with Chen's **projection difference** `ΔP`
[2507.21509 §6]: for each sample, project the dataset response onto `v̂_c` and
subtract the base model's own response projection; average over `D`. `ΔP` predicts
post-fine-tune trait expression *before any training*, and separates individual
problematic samples — enabling flag/clean/resample on the dataset at drop time.

### 3c. Mitigation (P2)

- **Preventative steering (default).** Add `+coef·v̂_c` to the residual stream
  **during fine-tuning only**; the gradient need not bake the trait in, so the
  shipped adapter carries no extra prompt-attack surface [Chen+ 2507.21509 §7].
- **Inference-time suppression.** Subtract `−coef·v̂_c` at decode to correct an
  already-trained model (coherence-costed at high coef).
- **Capping is a controlled negative, not a default.** A one-sided clamp
  `h ← h − v̂·max(⟨h, v̂⟩ − τ, 0)` [Assistant Axis, Lu+ 2601.10387] *bounds* drift
  but does not *reverse* a trait already baked into weights by fine-tuning; in this
  team's prior experiments it failed to mitigate. Use additive steering to fix;
  reserve the cap's projection only as a monitor.

---

## 4. Validation gate (a vector is not trusted until it passes)

A fitted `v_c` is validated by **LLM-judged free-form dose-response**: sweep `+coef·v̂_c` on the base model and confirm the judged trait score rises monotonically with coherence intact [Chen+ 2507.21509 §3.2].

---

## 5. Controls (mandatory for any causal claim)

- **Random-direction floor.** A norm-matched random unit vector must *not* steer
  (free-form) or *not* detect (monitoring AUC). Without it, an effect is not
  concept-specific.
- **Neutral-data control.** Training-time projection must stay flat on benign /
  generic instruction data; otherwise the monitor is tracking generic drift, not the
  concept.
- **Held-out evaluation set.** Fit on the extraction split; report on the disjoint
  evaluation split (Chen's 20/20 question split).
- **Judge–human agreement.** Spot-check the LLM judge against human labels before
  trusting its scores [Chen+ 2507.21509 App. B].

---

## 6. Reading the monitor outputs, and validating them

The training/inference monitor logs two per-concept series (W&B `drift/<c>/…`,
`train_summary.json`), both read from a fixed held-out probe set, pooled over
**response tokens** at the concept's selected layer:

- **`projection` = `mean⟨h, v̂_c⟩`** — unbounded; `+v̂_c` points toward *trait
  present*, so **higher = more trait**. Zero is **not** a calibrated boundary (the
  base model already sits at some offset) — read the **drift** (Δ from step 0), not
  the absolute sign.
- **`probe_prob` = `mean σ(w·h)`** — calibrated `P(trait present) ∈ [0,1]`, so **0.5
  is a real boundary** and the absolute value is interpretable.

They are the same quantity two ways (cheap dot-product vs. logistic probe) and should
move together; if they diverge, trust the probe. Both measure **internal
representation, not behaviour** — they say the direction is more active, not yet that
the model *acts* more trait-y. Hence validation:

1. **Probe AUROC** (have it) — separates trait-on/off, but only on the synthetic
   contrastive set. Necessary, not sufficient.
2. **Monitor vs. per-checkpoint eval battery** (have it) — correlate the trajectory
   with the matching behavioural metric (e.g. a safety trait vs. HarmBench/StrongREJECT
   per checkpoint); the monitor ideally moves *before* it. Free external check.
3. **Behavioural elicitation + judge** — generate from each FT'd checkpoint, score with
   the **same trait rubric**, correlate judged-trait against the monitor. The clean
   correlational test (the missing figure).
4. **Steering-causality** (strongest) — add `±coef·v̂_c` to the *FT'd* model and judge:
   `+coef` should raise and `−coef` suppress the trait. This is §4's dose-response run
   on the fine-tuned model; the biased-vs-mitigated-vs-neutral arms are a partial version.

---

## References

- Turner+ 2024, *Activation Addition* — arXiv:2308.10248
- Li+ 2023, *Inference-Time Intervention (ITI)* — arXiv:2306.03341
- Zou+ 2023, *Representation Engineering (RepE/LAT)* — arXiv:2310.01405
- Panickssery/Rimsky+ 2024, *Contrastive Activation Addition (CAA)* — arXiv:2312.06681
- Cao+ 2024, *BiPO* — arXiv:2406.00045
- Arditi+ 2024, *Refusal is mediated by a single direction* — arXiv:2406.11717
- Lee+ 2024, *Conditional steering (CAST)* — arXiv:2409.05907
- Chalnev+ 2024, *SAE-Targeted Steering (SAE-TS)* — arXiv:2411.02193
- Wu+ 2025, *AxBench / DiffMean* — arXiv:2501.17148
- Chen, Arditi, Sleight, Evans, Lindsey 2025, *Persona Vectors* — arXiv:2507.21509
  · code: github.com/safety-research/persona_vectors
- Lu, Gallagher, Michala, Fish, Lindsey 2026, *The Assistant Axis* — arXiv:2601.10387
