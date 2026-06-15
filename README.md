# ft-mechanistic-interface

A fine-tuning harness that optimises for more than loss. Drop in a model, a
dataset, and a use-case description; it mints **concept vectors** for the
safety-critical axes of that domain and uses them to **monitor drift during
training**, **audit the dataset before training**, and **mitigate drift** — so a
model that looks clean on the loss curve can't silently regress on an axis it was
never tested on.

Method and citations: [`docs/vector-steering.md`](docs/vector-steering.md).

## Layout

The **application** is the unit of work: one safety-critical app = a dataset + its
concept set + a model recipe. Code is organised by *function* (shared, DRY); configs
and artifacts are organised by *application*.

```
ft-mechanistic-interface/
--- README.md
--- pyproject.toml                       # one `ftmi` CLI entry point
--- docs/
--- --- vector-steering.md               # derivation + use, cited (read this first)
--- configs/                             # all knobs; a run is a config, not a code edit
--- --- applications/                    # ONE file per safety-critical app (the unit)
--- --- --- therapist.yaml               #   wires dataset + concepts + lora + monitor/audit/mitigate
--- --- --- healthcare.yaml              #   (add per app)
--- --- concepts/                        # {name, description} per axis — authored or proposed
--- --- --- therapist.yaml               #   referenced BY an application config
--- --- lora/                            # reusable LoRA recipes, shared across apps
--- --- --- qwen7b_default.yaml          #   r/alpha/epochs/save cadence
--- data/                                # datasets + artifacts, namespaced per app (gitignored)
--- --- therapist/                       #   sft.jsonl, fitted vectors, checkpoints, trajectories
--- src/ftmi/                            # organised by function — never per-app (DRY)
--- --- config.py                        #   typed config loading (yaml -> dataclasses)
--- --- cli.py                           #   thin entry points; no logic
--- --- data/                            #   dataset loaders (chat JSONL)
--- --- vectors/                         #   the pipeline
--- --- --- generate.py                  #     concept -> artifacts (Chen meta-prompt) + propose-from-data
--- --- --- extract.py                   #     diff-of-means fit (difference of response-token means)
--- --- --- monitor.py                   #     projection: drift trajectory, inference AUC, dataset audit
--- --- steering/
--- --- --- hooks.py                     #   the shared <h, v_hat> primitive (read / add / cap)
--- --- train/
--- --- --- lora.py                      #   the single reusable recipe + DriftMonitor callback
--- tests/                               # pure-function seams (no GPU)
```

## Design principles

- **Config in, code fixed.** A new application is a new YAML, never an edited script.
  One LoRA recipe (`train/lora.py`) parameterised entirely by config.
- **Application-centric configs, function-centric code.** Apps differ in their
  config/data; the pipeline code is shared and never forked per app.
- **One primitive, three uses.** Generate → extract → project. Monitoring, dataset
  audit, and mitigation are all `⟨h, v̂⟩` against the same vectors.
- **Validated before trusted.** No vector is used until it passes the free-form
  dose-response gate with random-direction and neutral-data controls
  (`docs/vector-steering.md` §4–5).

## Quickstart (intended)

Vector **generation** and the **judge** call a frontier LLM (default Anthropic, via
`ftmi.llm`); base-model rollouts + activation extraction run locally on GPU. Copy
`.env.example` → `.env` and set `ANTHROPIC_API_KEY` (plus `HF_TOKEN` for gated base
models).

```bash
uv sync
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
# 0. (optional) if no use-case descriptions exist, propose concepts from the dataset
uv run ftmi concepts --data data/therapist/sft.jsonl --domain therapist > configs/concepts/therapist.yaml
# 1. mint + validate concept vectors for the application
uv run ftmi vectors --concepts configs/concepts/therapist.yaml --model <base>
# 2. fine-tune with training-time drift monitoring + dataset audit
uv run ftmi train --app configs/applications/therapist.yaml
```

Status: scaffold. Interfaces are defined; implementations are stubs to fill during
the hackathon.
