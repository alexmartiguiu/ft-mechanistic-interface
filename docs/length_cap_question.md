# LLM output length caps & truncation — inventory

Every place an LLM's **output answer** is length-capped or truncated, grouped by source.
Each entry is `file:line` with the effective limit. Covers both frontier API calls
(generator + judge) and our hosted models (base-model rollouts, vLLM eval, LoRA training).

## A. Frontier API calls (Gemini / Anthropic — the artifact generator *and* the judge)

| What | Where | Cap |
|---|---|---|
| Anthropic output | `llm.py:34,42` (`max_tokens=8000`) | **8000 tokens** on every Claude generation *and* judge reply |
| Gemini output | `gemini_generator`, `llm.py:60-65` (`max_output_tokens=8000`) | **8000 tokens** |

> Both backends now cap at **8000**, matching Chen et al.'s artifact-generation limit
> (`claude-3-7-sonnet`, "maximum token limit of 8000", Persona Vectors §D.2). This is the
> *artifact-generator + judge* call (Chen's data-generation step), not the on-device
> response generation (that's section B).

## B. Hosted base model — `LocalModel` (transformers), used for vector fit / validate / monitor

| What | Where | Cap |
|---|---|---|
| single-rollout generation | `model.py:64` | `max_new_tokens=1000` |
| batched generation | `model.py:84,97` | `max_new_tokens=1000` |
| contrastive rollouts (Stage 2 fit) | `extract.py:61,117` | `max_new_tokens=1000` |
| dose-response sweep (validation) | `validate.py:45,61` | `max_new_tokens=1000` |
| drift-monitor generations | `monitor.py:13,24` | `max_new_tokens=1000` |

Every persona-vector generation answer is capped at **1000 new tokens**, matching Chen et
al.'s response generation (`eval/eval_persona.py`, `max_tokens=1000` — the paper itself
states no value). It's a ceiling, not a target: most elicitation responses hit EOS earlier
(`model.py:79-80` natural stopping), so 1000 mainly avoids clipping a response mid-trait.

## C. Eval generation — vLLM hosted model (base **or** LoRA adapter)

| What | Where | Cap |
|---|---|---|
| generic vLLM generate | `vllm_engine.py:53,56` | default `max_tokens=512`, plus optional `stop` sequences that end output early |
| MMLU-Pro CoT answer | `mmlu_pro.py:37,47`; `harness.py:64`; `configs/applications/therapist.yaml:36` | **2048 tokens** |
| Safety responses (HarmBench/StrongREJECT) | `safety.py:32,43`; `harness.py:71`; `therapist.yaml:40` | **512 tokens** (HarmBench official N=512; ≤ StrongREJECT's 512-token scorer window) |

## D. Eval judging — truncation of an output *before/within* the classifier

| What | Where | Cap |
|---|---|---|
| model response before HarmBench judge | `safety.py:64,67` | **none** — full generation fed (matches HarmBench; the old 2000-char cut was removed) |
| classifier prompt (embeds the generation) truncated at tokenization | `safety.py:69` | `truncation=True, max_length=4096`, `truncation_side="left"` (preserves the trailing "Answer:") |
| HarmBench classifier's own output | `safety.py:72` | `max_new_tokens=2` (just "yes"/"no") |

## E. LoRA training — answers the model learns from get truncated

| What | Where | Cap |
|---|---|---|
| SFT example length | `lora.py` `_build` | over-length rows **DROPPED**, not truncated (see below) |
| `max_seq_len` value | `configs/lora/qwen7b_default.yaml:19` | **2048 tokens** (per-app override via `data.max_seq_len`; therapist = 4096) |
| dataset projection-difference audit | `lora.py:106-107` | `truncation=True, max_length=max_seq_len` (read-only; not a learning signal) |

**Why we DROP rather than truncate.** 2048 is the EM field standard — Betley et al.'s
`open_models`, Chen's `persona_vectors`, and Model-Organisms-for-EM all use exactly
`max_seq_length=2048, packing=False`. We keep that bound (removing it risks OOM from a single
long example + non-deterministic batch memory), but we **filter** over-length rows instead of
right-truncating them, because we train **completion-only** (prompt tokens masked to -100, like
Betley's `train_on_responses_only`) — and right-truncation would clip exactly the assistant
tokens that carry the loss. Applications with long answers (therapist) raise `data.max_seq_len`
so fewer rows are dropped rather than relying on a clip that would have eaten the completion.

---

## Notably *not* truncated

- **TruthfulQA logprob scoring** — `logprob.py:38` explicitly sets `truncation=False`.

## Excluded — these cut the *number of items*, not answer length (listed for completeness)

- `mmlu_pro.py:41` `rows[:limit]` (1500 questions) · `safety.py` `n_samples=100` prompts ·
  `datasets.py:97` `fewshot[:n_shot]` · `lora.py:252` `probe_batch[...:16]` ·
  `loaders.py:27` train/valid split.

## Risk note

The caps most likely to silently distort **misalignment-drift** results:

- ~~**C/D** — 256-token safety responses + 2000-char pre-judge cut~~ — **resolved**: safety
  responses now 512 (HarmBench official), pre-judge char cut removed, classifier left-truncates
  at 4096.
- ~~**E** — the 2048 training cap clips long answers~~ — **resolved**: over-length rows are
  dropped (not clipped) and training is completion-only, so a long answer is never half-learned;
  `data.max_seq_len` raises the bound per application (therapist = 4096).

Resolved: A's Gemini/Anthropic asymmetry (both 8000); B aligned to Chen (1000); C/D HarmBench
(512, no char-cut); E completion-only + drop-over-length. Open follow-ups: expose B/C caps via config.
