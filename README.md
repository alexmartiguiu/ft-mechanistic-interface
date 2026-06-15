# ft-mechanistic-interface

A fine-tuning harness that optimises for more than loss. Drop in a model, a
dataset, and a use-case description; it mints **concept vectors** for the
safety-critical axes of that domain and uses them to **monitor drift during
training**, **audit the dataset before training**, and **mitigate drift** — so a
model that looks clean on the loss curve can't silently regress on an axis it was
never tested on.

Method and citations: [`docs/vector-steering.md`](docs/vector-steering.md).

## Layout

```
configs/            all knobs live here — code is config-driven, not edited per run
  lora/             LoRA recipe configs (one reusable recipe, config selects the rest)
  concepts/         concept definitions: {name, description} per safety axis, per domain
  experiment/       top-level run configs that compose lora + concepts + data
data/               datasets + generated vector artifacts (gitignored)
src/ftmi/
  config.py         typed config loading (yaml -> dataclasses)
  data/             dataset loaders
  vectors/          generate (Chen meta-prompt) -> extract (diff-of-means) -> monitor
  steering/         residual-stream hooks (add / suppress / cap / read)
  train/            one reusable LoRA recipe; only the config changes
scripts/            thin CLI entry points; no logic lives here
tests/
```

## Design principles

- **Config in, code fixed.** A new experiment is a new YAML, never an edited script.
  One LoRA recipe (`train/lora.py`) parameterised entirely by config.
- **One primitive, three uses.** Generate → extract → project. Monitoring, dataset
  audit, and mitigation are all `⟨h, v̂⟩` against the same vectors.
- **Validated before trusted.** No vector is used until it passes the free-form
  dose-response gate with random-direction and neutral-data controls
  (`docs/vector-steering.md` §4–5).

## Quickstart (intended)

```bash
uv sync
# 1. mint + validate concept vectors for a domain
uv run ftmi vectors --concepts configs/concepts/therapist.yaml --model <base>
# 2. fine-tune with training-time drift monitoring
uv run ftmi train --config configs/experiment/example.yaml
```

Status: scaffold. Interfaces are defined; implementations are stubs to fill during
the hackathon.
