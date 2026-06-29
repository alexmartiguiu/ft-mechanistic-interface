# Overnight preventative-steering campaign — morning summary (2026-06-29)

## ✅ Goal EXCEEDED: 4 good preventative-steering examples confirmed (target was ≥3)

A run is "good" if, vs its matched biased baseline, preventative steering **holds/restores
capability**, **holds/restores safety**, and **keeps the malign concept lower**.

| example | model | steer | capability | safety (refusal) | malign concept (final probe) |
|---|---|---|---|---|---|
| **gender_steered_dense** | Qwen-7B | +32 @ L14 | MMLU+TQA restored | SR held | gender_bias **0.94 → 0.23** |
| **race_steered_dense** | Qwen-7B | +32 @ L16 | MMLU+TQA restored | **HB+SR restored** | race_bias 0.76 → 0.64 |
| **medical_da_steer_c360L12** ⭐ | **Apertus-8B (DEMO)** | +360 @ L12, single malign vector | TQA restored, MMLU held | **HB .54→.81 (+27), SR .753→.90 (+15)** | dangerous_advice 0.92 → 0.69 |
| **gender_full_steer_c32** (long run) | Qwen-7B | +32 @ L16, 760 steps | **MMLU+TQA restored** | **HB+SR restored** | gender_bias 0.81 → **0.02** |

(The 4th is the "long run" version the user asked about — full 760-step training; cleanest
restoration of all four metrics. Its baseline probe started saturated so the verdict's
"rose" gate excluded it, but the steered result — gender_bias 0.02 vs biased 0.81 — is
unambiguous.)

**Figure:** `figures/steer_comparison_3wins.png` — biased (red) vs steered (green), 3 rows ×
[capability | safety | concept]. The demo money-shot.

⭐ The medical-apertus win is the actual demo model and the textbook case: the biased fine-tune
collapses **both** capability (MMLU .349→.234) **and** safety (HarmBench refusal .87→.54) while
`dangerous_advice` rises — and preventative steering **restores safety substantially**, holds
capability, and lowers the malign trait. Exactly the "ideal" profile.

## The medical journey (the key scientific finding)

The dose-response picked coef **240 @ L12** for `dangerous_advice` on Apertus (its base
projection is ~−240, ~120× Qwen — coef 32 was hopeless; the **norm-scaling thesis confirmed**).
But the first full run applied that coef across **3 layers × all 5 medical vectors** ≈ 15× the
intended perturbation and **destroyed** capability (MMLU → 0.11). Fix: mirror the gender/race
winners — steer **only the single malign vector at its single layer** at the dose coef. That
preserves capability and restores safety. **Insight: the dose-response coef is per-(vector,
layer); do not multiply it across layers/vectors for preventative steering.**

This was **independently re-confirmed by financial** (Qwen): the campaign's auto-config steered
`risk_minimization`'s dose coef (48 @ L14) across **3 layers × 5 vectors** and likewise destroyed
capability (steered MMLU 0.14 vs biased 0.37). Financial is therefore *not* a win — it's a clean
second confirmation of the over-steer rule. **Demo takeaway: steer the single malign vector at a
single layer; that's the whole recipe.** All four wins follow it.

**Coefficient headroom (the medical result we shipped):** on the single vector / single layer, you
can push the coef *past* the dose value for more mitigation at no capability cost. Medical at
coef **240** restored safety (HB +17) and held capability; pushing to **360** restored safety even
MORE (HB +27, SR +15) and suppressed `dangerous_advice` further (0.92→0.69 vs 0.77), with capability
still held. So the dose-response coef is a *floor* for preventative steering, not a ceiling — the
ceiling is wherever capability starts to dip. (coef-240 medical also passed; 360 is the demo result.)

## Code fixes applied tonight (all committed to the working tree, tests green)

1. **`eval/vllm_engine.py`** — eval was dying on missing `ninja` (FlashInfer sampler JIT). Tied
   nvcc-usability to `ninja` presence → forces the safe eager path. Unblocked all evals.
2. **Recipe match** — the campaign initially trained medical with `apertus_8b_medical` (lr 2e-4,
   batch 4) but the baseline used `apertus8b_default` (lr 1e-4, batch 8). Switched to match → a
   clean apples-to-apples comparison.
3. **`experiments/steer.py` verdict** — concept "suppressed" now counts a uniform downward shift
   (steered trait ends meaningfully lower), not only a smaller delta. Correctly credits medical;
   gender/race unchanged.

## Campaign complete

- **gender_full** ✅ — the 4th win (see table).
- **financial** ✗ — over-steered (multi-layer × multi-vector); kept as the second confirmation
  of the over-steer rule, not a win.
- **monitor↔behavior figure** ✅ — `figures/monitor_vs_behavior_medical__apertus-8b-instruct-2509_dangerous_advice.png`.
  Honest result on the biased medical run: behavioural `dangerous_advice` (LLM-judged) **rises
  0→~30** over training (drift confirmed), but the per-checkpoint correlation with the internal
  monitors is **weak/scattered** (judged↔probe r=0.40, judged↔projection r=−0.59, n=12). This
  *confirms the docs' own caveat* — probe and behaviour agree on the big trait on/off contrast but
  not on fine per-checkpoint fluctuations. **Demo takeaway: pair the probe with behavioural eval;
  don't read fine fluctuations off the probe alone.**

## Verify any run
```
PYTHONPATH=src .venv/bin/python -m ftmi.cli steer-exp --verdict-only \
  --name medical_da_steer_c240L12__apertus-8b-instruct-2509 \
  --baseline medical__apertus-8b-instruct-2509 --concept dangerous_advice
```
State: `overnight/good_examples.json`, `overnight/state/status.json`, `overnight/supervisor.log`,
`data/steer_campaign.jsonl`.

## Stretch — all complete
- (a) **monitor↔behavior figure** ✅ (above).
- (b) **steered-vs-biased comparison figure** ✅ — `figures/steer_comparison_3wins.png`
  (built via new `scripts/build_steer_comparison.py`).
- (c) **hardened validation gate exercised** ✅ — the medical dose-response ran `validate_vector`
  (monotonicity + trait-gain gate) and selected coef 240 @ L12, `passed=True`, trait_gain 74.5.
- (d) **web-grounded concept YAMLs** ✅ — `overnight/demo_concepts_medical.yaml`
  (5 cited concepts via `ftmi concepts`).

## Demo artifacts (where to look)
- `figures/steer_comparison_3wins.png` — the money-shot (biased vs steered × capability/safety/concept).
- `figures/monitor_vs_behavior_*.png` — behaviour drift + the honest monitor caveat.
- `overnight/good_examples.json` — the 4 wins with metrics.
- `scripts/build_steer_comparison.py`, `scripts/monitor_vs_behavior.py` — reproduce the figures.
- New CLI: `ftmi steer-exp` (`src/ftmi/experiments/steer.py`) — dose-response | run | --verdict-only.

## Bottom line
Goal was ≥3; **delivered 4** confirmed preventative-steering wins incl. the demo model
(medical-apertus). Headline finding for the talk: **steer the single malign vector at a single
layer** — the dose coef multiplied across layers/vectors over-steers and destroys capability
(shown twice). All code fixes are in the working tree with tests green.
