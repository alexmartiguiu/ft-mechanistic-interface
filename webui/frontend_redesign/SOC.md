# Separation of concerns for the User Experience

The product (**hedda** — *open-source tools for interpretable narrow fine-tuning*) has **two
distinct steps**, joined by **one contract**. Everything in the UI, the backend, and the way the
two of us split the work falls cleanly on one side of that contract or the other.

```
        STEP 1 — DESIGN THE RUN                 ║  CONTRACT  ║         STEP 2 — RUN & MONITOR
   (agent-assisted config authoring)            ║            ║      (the existing training pipeline)
                                                 ║  3 YAML    ║
  chat ▸ drop dataset ▸ propose ▸ edit ▸ review  ║  config    ║   vectors ▸ train ▸ eval ▸ report
                                                 ║  files     ║   + drift monitoring
  Claude Code agent API                          ║            ║   ftmi pipeline  (POST /api/run)
                                                 ║            ║
  output ──────────────────────────────────────▶║────────────║──────────────▶ input
```

The two steps **never share state except the three files**. Step 1 decides *what to build*;
Step 2 *builds it and watches it*. Either side can be developed, demoed, and tested in isolation as
long as it honours the file contract — which is exactly why this is also how we divide the work
between two people (see *Ownership* at the end).

---

## The contract: three config files

These three YAML files are the **only** thing that crosses the boundary. They already exist as the
input to the `ftmi` pipeline today — Step 1's entire job is to *produce a valid set of these three*,
and Step 2's entire job is to *consume them*. Nothing else passes between the steps.

| # | File | Lives in | Owns |
|---|------|----------|------|
| 1 | **Application config** | `configs/applications/<name>.yaml` | The run as a whole: name, which dataset + how to read it, which concept set, which LoRA recipe, monitor/audit/mitigate switches, eval battery. Points at files 2 & 3. |
| 2 | **Concept set** | `configs/concepts/<domain>.yaml` | The safety/quality axes to track: `domain` + a list of `{name, description}` concepts the monitor will watch for drift. |
| 3 | **LoRA recipe** | `configs/lora/<model>.yaml` | The training recipe: `model_id`, dtype, LoRA hyperparameters (`r`, `alpha`, `dropout`, `target_modules`), optimiser (`lr`, `epochs`, `batch_size`…), checkpoint cadence. |

> **Contract rule:** these files are loaded by typed dataclasses in `ftmi.config`
> (`ApplicationConfig.load`, `ConceptSet.load`, `LoraConfig.load`). A set of files is "valid" iff it
> loads cleanly. Step 1 must only ever emit valid sets; Step 2 may assume validity. The dataclass
> schema is the source of truth — neither step invents fields the other doesn't know.

---

## Step 1 — Design the run (agent-assisted config authoring)

**Goal:** take a user from "I have a dataset and a worry" to "three valid config files, reviewed and
launched." This is the *new* surface the redesign introduces (`route: 'new'`, the **New experiment**
screen). It is a conversation, not a form.

**Driven by the Claude Agent SDK.** The initial UI is a chat with the hedda agent. The agent turns
an intent + a dataset into concrete config proposals. It is **propose-only**: it never writes a
file or launches a run on its own — **writing the configs and launching are explicit user clicks**
(two click-gates). The user accepts, tweaks, or overrides every proposal, or takes the defaults.

### Flow (the prototype's `expStage` machine: `intro → concepts → model → lora → review → launched`)

1. **Chat + drop a dataset.** The user describes what they're fine-tuning and **drops a raw dataset
   from Hugging Face** (or a local JSONL). A **dataset viewer** shows what it is — example count,
   fields, `max_seq_len`, and a peek at rows. If the data isn't already in the chat/`messages`
   format the pipeline trains on, **the agent writes the conversion** (the way the `make_*_sft.py`
   scripts do) and the converted result is previewed before it's accepted. The dataset is the
   evidence the agent reasons about.
2. **Agent proposes per file — user chooses.** For **each of the three config files**, the agent
   proposes a small set of *choice options* with a recommended default:
   - *Concept set* — proposed concepts (each with a confidence the agent assigns), pre-toggled on.
     The user toggles concepts on/off and can **add their own** custom concept.
   - *LoRA recipe* — a recommended model + recipe (e.g. `r16 · α32 · 3 epochs`). The user can
     **"Use defaults"** or **"Customize"**.
   - *Application config* — assembled from the above plus dataset/monitor/eval settings.
   The user clicks a proposed option **or accepts the default** — one click per decision.
3. **Collapsible, directly-editable file view.** Each of the three files has a **collapsible panel
   showing every parameter**. Power users expand it and **edit any field in the file directly**
   (e.g. the LoRA `Customize` panel exposing `r`, `alpha`, `dropout`, `epochs`, `lr`, `batch`). The
   chat-driven proposals and the raw editor are two views of the *same* file — edits in either
   reflect in the other.
4. **Review.** A final read-only view of all three files exactly as they'll be written to disk —
   the explicit "this is the contract" moment.
5. **Launch.** Writing the three files **is** the hand-off to Step 2. The UI then kicks off the run
   (`POST /api/run`) and transitions the user toward the monitoring surfaces.

### What Step 1 must NOT do
Step 1 does **no** training, vector extraction, or evaluation. It does not import `torch` or touch
GPUs. Its only durable output is three files on disk. If Step 1 disappeared, Step 2 would still run
from hand-written YAML — which is the proof the seam is clean.

---

## Step 2 — Run & monitor (the training pipeline)

**Goal:** consume a valid set of three config files and execute + observe the pipeline. **This
already exists** — it's the `ftmi` pipeline behind the FastAPI backend. The redesign keeps these
screens; they live entirely *right of the contract*.

- **Runs** (`POST /api/run`, `GET /api/runs/{id}/stream` via SSE, `POST …/stop`) — launch
  `ftmi run|train|eval` against a config, stream logs live, stop a job.
- **Dashboard** (`GET /api/overview`, `GET /api/catalog`, `GET /api/plot/…`) — base→final drift per
  run across the capability / truthfulness / safety battery.
- **Concept Vectors** (`GET /api/vectors`) — the extracted concept directions and probe quality per
  domain.
- **Live Steering** (`GET /api/steer/concepts`, `POST /api/steer`) — dial a concept up/down and
  compare normal vs. steered output.

Internally Step 2 is itself a pipeline (vectors → train-with-drift-monitor → eval → report), but
**to Step 1 that's a black box** — it takes three files and emits runs, drift numbers, vectors, and
steerable models.

### What Step 2 must NOT do
Step 2 never asks the user to *design* anything. It assumes the three files are correct and
complete. It does not call the agent. If a config is wrong, that's a Step 1 bug, surfaced as a
failed run — not something Step 2 tries to repair.

---

## The loop (the contract, exercised repeatedly)

The two steps don't run once — they run in a cycle, and the **run is the first-class object** the
whole experience orbits (see `PLAN.md`). An **experiment is one chat thread = a lineage of runs**:

> chat to design → *click* to write configs → *click* "Send run" → watch metrics →
> **on completion** the agent reads the results and proposes config tweaks → accept → relaunch.

Crucially, this doesn't move the seam. Each lap is still *Step 1 emits a new version of the three
files* → *contract* → *Step 2 runs it*. The loop only adds **versioning + lineage** on top of the
same contract: every run snapshots the three files it used, so runs are reproducible and the agent
can diff "what changed since last run." Smart reuse (skip vector minting when concepts+model are
unchanged) is a Step-2 optimisation keyed on the config snapshot — invisible to Step 1.

---

## Why the seam is here (and not elsewhere)

- **It already is the real boundary.** The `ftmi` CLI/pipeline has *always* taken these three files
  as input. We're not inventing a contract; we're exposing the one that exists.
- **The two sides need different skills and move at different speeds.** Step 1 is conversational
  UX + agent orchestration + light file editing. Step 2 is data viz + log streaming + long-running
  job control. Coupling them would mean every change touches both.
- **It's independently testable.** Step 1 can be demoed with a "review" screen and never launch a
  GPU. Step 2 can be demoed against checked-in example configs and never open the agent. Each side
  mocks the other with *a folder of three YAML files*.

---

## Ownership (how two people split this)

The contract is also the team seam. Both sides share **one design system and one `api.js`
client** (see `FRONTEND.md`) so the app feels like one product, but each owns its screens end to end.

| | Step 1 — Design | Step 2 — Run & Monitor |
|---|---|---|
| **Screens** | New experiment (chat, dataset viewer, per-file proposals, collapsible editors, review) | Runs, Dashboard, Concept Vectors, Live Steering |
| **Backend it talks to** | Claude Code agent API; a small "write configs" endpoint | existing `/api/run`, `/api/overview`, `/api/vectors`, `/api/steer`, `/api/plot` |
| **Owns the contract from** | the *producing* side (must emit valid files) | the *consuming* side (may assume valid files) |
| **Mocks the other side with** | a "Review → done" stub that writes files | a folder of example configs in `configs/` |

**Integration test for the seam:** Step 1 writes three files for a brand-new domain; Step 2, given
only those files (no agent, fresh process), launches a run that loads them without error. If that
passes, the two halves are correctly decoupled.
