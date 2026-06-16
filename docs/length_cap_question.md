# LLM output length caps & truncation — inventory

Every place an LLM's **output answer** is length-capped or truncated, grouped by source.
Each entry is `file:line` with the effective limit. Covers both frontier API calls
(generator + judge) and our hosted models (base-model rollouts, vLLM eval, LoRA training).

## A. Frontier API calls (Gemini / Anthropic — the artifact generator *and* the judge)

| What | Where | Cap |
|---|---|---|
| Anthropic output | `llm.py:34,42` (`max_tokens=4096`) | **4096 tokens** on every Claude generation *and* judge reply |
| Gemini output | `gemini_generator`, `llm.py:60-64` | **No cap set** — Gemini output length is left at the model default (asymmetric with Anthropic) |

> ⚠️ The two backends are inconsistent — Claude is capped at 4096, Gemini is uncapped.
> If a judge or artifact reply is unexpectedly truncated on Claude but not Gemini, this is why.

## B. Hosted base model — `LocalModel` (transformers), used for vector fit / validate / monitor

| What | Where | Cap |
|---|---|---|
| single-rollout generation | `model.py:64` | `max_new_tokens=128` |
| batched generation | `model.py:84,97` | `max_new_tokens=128` |
| contrastive rollouts (Stage 2 fit) | `extract.py:61,89` | `max_new_tokens=128` |
| dose-response sweep (validation) | `validate.py:45,61` | `max_new_tokens=128` |
| drift-monitor generations | `monitor.py:13,24` | `max_new_tokens=128` |

Every persona-vector generation answer is cut to **128 new tokens**. (The EOS trimming at
`model.py:79-80` is natural stopping, not a cap.)

## C. Eval generation — vLLM hosted model (base **or** LoRA adapter)

| What | Where | Cap |
|---|---|---|
| generic vLLM generate | `vllm_engine.py:53,56` | default `max_tokens=512`, plus optional `stop` sequences that end output early |
| MMLU-Pro CoT answer | `mmlu_pro.py:37,47`; `harness.py:64`; `configs/applications/therapist.yaml:36` | **2048 tokens** |
| Safety responses (HarmBench/StrongREJECT) | `safety.py:32,43`; `harness.py:71`; `therapist.yaml:40` | **256 tokens** |

## D. Eval judging — truncation of an output *before/within* the classifier

| What | Where | Cap |
|---|---|---|
| model response truncated before HarmBench judge | `safety.py:64,67` | `g[:max_resp_chars]` → **first 2000 chars** of the response |
| classifier prompt (embeds the generation) truncated at tokenization | `safety.py:69` | `truncation=True, max_length=2048` |
| HarmBench classifier's own output | `safety.py:72` | `max_new_tokens=2` (just "yes"/"no") |

## E. LoRA training — answers the model learns from get truncated

| What | Where | Cap |
|---|---|---|
| SFT example tokenization | `lora.py:162` | `truncation=True, max_length=max_seq_len` |
| dataset projection-difference audit | `lora.py:106-107` | `truncation=True, max_length=max_seq_len` |
| `max_seq_len` value | `lora.py:155`; `configs/lora/qwen7b_default.yaml:19` | **2048 tokens** |

These truncate the full user+assistant text (so long assistant answers in training data are
clipped at 2048 tokens).

---

## Notably *not* truncated

- **TruthfulQA logprob scoring** — `logprob.py:38` explicitly sets `truncation=False`.

## Excluded — these cut the *number of items*, not answer length (listed for completeness)

- `mmlu_pro.py:41` `rows[:limit]` (1500 questions) · `safety.py` `n_samples=100` prompts ·
  `datasets.py:97` `fewshot[:n_shot]` · `lora.py:252` `probe_batch[...:16]` ·
  `loaders.py:27` train/valid split.

## Risk note

The caps most likely to silently distort **misalignment-drift** results:

- **C** — 256-token safety responses can truncate a harmful answer mid-completion, and
  **D**'s 2000-char pre-judge cut compounds it.
- **E** — the 2048-token training cap clips long therapist/advisory answers.

Open follow-ups: make these configurable, and align the Gemini cap with Anthropic's.
