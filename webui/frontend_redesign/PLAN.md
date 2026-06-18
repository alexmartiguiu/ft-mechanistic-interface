# Build plan — hedda

*Open-source tools for interpretable narrow fine-tuning.*

Read alongside **`SOC.md`** (the two-step flow + the three-file contract) and **`FRONTEND.md`**
(stack, design tokens, structure, conventions). This document is the *plan*: what we're building and
in what order. It has two parts:

- **Step 1 — Wire the redesign into the real frontend.** Engineering. Turn the `.dc.html` prototype
  into a properly-structured, composable React + Vite app.
- **Step 2 — Define the workflows.** Product. Make **the run the first-class citizen**, with the
  agent chat and the metrics dashboard as two views around it, closing an iterative refine loop.

---

## Decisions (locked)

These were settled up front; the rest of the plan assumes them.

| # | Decision | Choice |
|---|----------|--------|
| 1 | **Agent runtime** | **Claude Agent SDK** (hosted — see code.claude.com/docs/en/agent-sdk/hosting). Tools: read/write configs, read completed-run metrics, propose config diffs. |
| 2 | **Autonomy** | **Propose-only.** Two explicit user click-gates: (a) write/update config files, (b) launch a run. The agent never writes or launches silently. |
| 3 | **Chat ⇄ dashboard** | A **toggle via the sidebar** — Chat and Dashboard are sidebar routes alongside Concept Vectors / Live Steering. One view at a time, full width. |
| 4 | **Persistence** | **JSON index under `data/`** (runs + per-run config snapshots). Replaces the in-memory `_RUNS` dict. |
| 5 | **When the agent analyses** | **After the run completes.** End-of-run notification; runs taking hours is expected and fine. No mid-run reactivity. |
| 6 | **What the agent may tweak** | **All four:** concept set, LoRA recipe, dataset, monitor/eval settings. |
| 7 | **Re-run cost** | **Smart reuse**, keyed on a config hash: skip vector minting when concepts+model are unchanged; a dataset change forces re-train; a concept change forces re-mint. |
| 8 | **Grouping** | **Experiment = one chat thread = a lineage of runs** (run → tweak → run). Sidebar lists experiments; each shows its run history + config diffs. |
| 9 | **Dataset ingestion** | User drops a **raw HF dataset id; the agent writes the conversion** to messages-JSONL (like the `make_*_sft.py` scripts). |
| 10 | **Scope + demo** | **Internal, single-user, real pipeline** (one active experiment/run; GPU contention not a concern yet). **For the demo: replay existing checkpoints/runs and animate as if running end-to-end** — real runs take hours. |

---

## The vision

Today the pipeline is a one-shot: author three configs → launch → read results. We're turning it
into a **loop centred on the run**:

```
        ┌──────────────────────── one Run ────────────────────────┐
        │                                                          │
   ╭─────────────╮   send run    ╭───────────────────────────╮     │
   │  AGENT CHAT │ ────────────▶ │   METRICS DASHBOARD (W&B-  │     │
   │ (hedda /    │               │   like): drift, eval, logs │     │
   │  Claude     │ ◀──────────── │                            │     │
   │  Code agent)│  reads logs   ╰───────────────────────────╯     │
   ╰─────────────╯  + metrics                                       │
        │   proposes config tweaks (dataset / concepts / LoRA)      │
        └──────────────────── launch updated run ──────────────────┘
                         (new run, new config version)
```

- The user **starts in the chat**, where the agent helps generate the three config files
  (Step 1 of `SOC.md`). The agent only ever *proposes* — the user clicks to write the files.
- Once the configs exist, the agent exposes a **"send run" abstraction** — in practice a **button
  the user clicks** (the agent never launches silently) — which creates a run that shows up in the
  **metrics dashboard**.
- The user **toggles via the sidebar** between the **Dashboard** (in-depth metrics, like a Weights
  & Biases run view) and the **agent Chat**. **When the run completes**, the agent reads its final
  metrics + logs and **proposes changes** to the dataset, the concept set, the recipe, or the
  monitor/eval settings to make the next run more aligned and interpretable.
- Once the user agrees with the proposed tweaks, they **click to launch a new run with the updated
  config files**. The new run becomes a child in the same experiment thread. The loop repeats.

So the experience is: **chat to design → click to run → watch metrics → chat to refine → run
again.** The run is what everything orbits.

---

## The central object: a Run

Everything in Step 2 hangs off a well-defined **Run**, grouped under an **Experiment**:

- An **Experiment** = one chat thread = a lineage of runs. The sidebar lists experiments; opening
  one shows its chat, its run history, and the config diffs between runs.
- A **Run** = `{ id, experiment id, parent run id, config snapshot (the 3 files used), config hash,
  status, logs, metrics }`.
- Persisted as a **JSON index under `data/`** plus a per-run config-snapshot dir, so every run is
  reproducible and the agent can diff "what changed since last run." (Replaces the in-memory
  `_RUNS` dict that resets on restart.)
- The **config hash** drives **smart reuse**: if concepts+model match a prior run, skip vector
  minting and reuse those artifacts; a dataset change forces re-train; a concept change forces
  re-mint.
- The three-file contract from `SOC.md` still holds exactly — a run is just *"the pipeline executed
  against one frozen version of the contract."* The loop adds *versioning + lineage*, not a new
  contract.
- **Status** drives the UI: `draft → launching → running → done | failed | stopped`. The dashboard
  and the chat both render off the same Run state.

This is the seam between the two people: one owns *producing/refining* the config that becomes a
Run; the other owns *executing and visualising* the Run. (See `SOC.md` *Ownership*.)

---

## Step 1 — Wire the redesign into the existing frontend

**Goal:** the hedda design, running as a real, composable React + Vite app in `webui/frontend/`,
served by the existing FastAPI backend — no behaviour invented, just ported into proper structure.

The source is `webui/frontend_redesign/.../hedda.dc.html`: one React **class** (design-tool export)
already wired to the live API, with a `_sim` demo-data mode. Porting = **decompose into the
`screens/` + `components/` structure in `FRONTEND.md`**, converting class state to hooks.

**1.0 — Shared scaffold (both of us, first; unblocks everything):**
- Move the hedda tokens into `src/styles/tokens.css` (replace the "間 ma" palette); add the Hanken
  Grotesk + JetBrains Mono fonts.
- Build the shell: `App.jsx` = `Sidebar` (nav) + `route` switch + `PageHeader` (eyebrow/title/sub
  from the prototype's `pageMeta()`).
- Grow `src/api.js` to one function per endpoint, throw-on-error, matching the current pattern:
  `getOverview`, `getCatalog`, `plotUrl`, `getVectors`, `getSteerConcepts`, `postSteer`,
  `getConfigs`, `startRun`, `streamRun`, `stopRun`, `getHealth` (Step-2 additions later:
  `getExperiments`, `getRunMetrics`, `chatAgent`, `previewDataset`, `writeConfigs`).

**1.1 — Port the Step-2 screens (backend already exists):** Dashboard, Concept Vectors, Live
Steering, Runs → `screens/`, each built from shared `components/`. Carry over the async-UX rules
from `FRONTEND.md` (steering warm-up + disabled button, SSE log tailing, demo-mode fallback).
*Decision to settle:* Dashboard data source — prototype reads `/api/overview`; current app uses the
richer `/api/catalog` + `/api/plot/*.svg`. Recommend keeping the SVG plots, restyled to hedda.

**1.2 — Port the New-Experiment shell (Step-1 surface):** the chat + dataset-drop + per-file
proposals + collapsible editors + review, built against the prototype's `_sim` data for now (live
wiring is Step 2 of this plan). Keep it composable: `screens/NewExperiment/` with sub-components
(`ChatPane`, `DatasetViewer`, `ConfigCard` ×3, `ReviewPanel`).

**1.3 — Cutover:** serve the React build at `/`, keep `/legacy`, delete the old tokens.

**Done when:** all five screens render under one hedda design system, every backend call goes
through `api.js`, no component hardcodes a colour or a `fetch`, and the prototype's hardcoded
`launchExperiment()` (fixed `education`/GPU) is replaced by "launch the config we actually built."

---

## Step 2 — Define the workflows (run as first-class citizen)

**Goal:** the loop above, made real. This is where we go beyond the prototype, and it needs new
backend behind the `SOC.md` contract.

**2.1 — Make the Run a real object.** Backend: persist runs + experiments as a **JSON index under
`data/`** (`{id, experiment id, parent, config snapshot, config hash, status, created}`), expose
`GET /api/experiments`, `GET /api/runs` (with lineage), and per-run metrics. Frontend: a Run view
that unifies logs (SSE) + final metrics, styled like a W&B run.

**2.2 — The chat ⇄ dashboard toggle (sidebar).** Chat and Dashboard are **sidebar routes**, not
split panes — selecting one shows it full width, the other is one click away. Both are scoped to the
**current experiment** and render off the same Run state, so switching never loses context.

**2.3 — The agent's "send run" abstraction.** Give the agent a tool that *prepares* a run from the
current three-file draft, surfaced to the user as a **"Send run" button**. **The user clicks to
launch — the agent never launches silently** (decision #2). Launch = snapshot configs (with hash) →
`POST /api/run` → new Run appears under the experiment.

**2.4 — The refine loop (the heart of it).** **When a run completes**, the agent reads its **final
metrics + logs** and **proposes concrete edits** to any of the four — dataset, concept set, LoRA
recipe, or monitor/eval settings — shown as a **diff against the current config version**. The user
discusses; **the user clicks to accept** (writing an updated three-file version) and **clicks again
to launch** the next run via **2.3**. Smart reuse (decision #7) decides what recomputes. Lineage
links child run → parent within the experiment.

**2.5 — Step-1 backend the loop depends on** (the genuinely new endpoints, behind the contract):
- **Agent chat** endpoint built on the **Claude Agent SDK** (streamed), exposing tools to: read the
  current configs, read **completed**-run metrics/logs, propose config diffs, and **write a HF→
  messages-JSONL conversion** for a dropped dataset.
- **Dataset resolve/preview** for a dropped Hugging Face dataset id (metadata + sample rows for the
  viewer); the agent's conversion output is previewed before it's accepted.
- **Write-configs** endpoint: take the assembled/edited/accepted choices, **validate via
  `ftmi.config` dataclasses, then write** the three YAMLs (snapshotted per run). Validation here is
  what guarantees every run honours the contract. **Two click-gates** (write, launch) live on top of
  this.

**2.6 — Demo / replay mode.** Real runs take hours, so for demos the dashboard **replays existing
checkpoints/runs and animates them as if running end-to-end** (reusing the prototype's `_sim`
spirit). Same UI, same Run state machine — the data source is recorded artifacts on a timer instead
of a live job. Build this as a flag on the Run, not a separate code path.

**Done when:** a user can go chat → "Send run" → watch metrics → (on completion) have the agent read
the result and propose a tweak → accept → relaunch — all within one experiment thread, every run
reproducible from its snapshotted configs, and the whole loop demoable via replay.

---

## Sequencing & ownership

| Order | Work | Owner | Depends on |
|------|------|-------|-----------|
| 1 | 1.0 Shared scaffold (tokens, shell, `api.js`) | both, together | — |
| 2 | 1.1 Step-2 screens (Dashboard / Vectors / Steering / Runs) | B | 1.0 |
| 2′ | 1.2 New-Experiment shell on `_sim` data | A | 1.0 |
| 3 | 2.1 Run as real object + persistence | B | 1.1 |
| 3′ | 2.5 Step-1 backend (agent / dataset / write-configs) | A | 1.2 |
| 4 | 2.2–2.4 toggle + send-run + refine loop | both (A=chat, B=dashboard) | 2.1, 2.5 |

A and B work in parallel after the shared scaffold; they meet at the run object and the contract.

---

## Residual questions (smaller, settle as we build)

The big ones are locked in *Decisions* above. These remain:

1. **Config-snapshot layout.** Where exactly per-run snapshots live — e.g.
   `data/<experiment>/runs/<run-id>/configs/` — and how the JSON index references them. (Scheme,
   not whether — persistence itself is decided.)
2. **Dashboard data source.** Keep the richer `/api/plot/*.svg` (current app) or the prototype's
   `/api/overview` JSON-and-draw-our-own? (Recommend: keep SVG plots, restyle to hedda.)
3. **Agent SDK hosting/auth.** Where the SDK process runs relative to the FastAPI backend, and how
   API keys are supplied (reuse the existing `.env` the launcher sources).
4. **Replay source for the demo.** Which recorded experiment(s)/checkpoints we animate, and the
   timing curve so it reads as a live run.
5. **Routing.** Stay on `route`-in-state, or adopt a tiny router once experiments/runs need deep
   links?
