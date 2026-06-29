# ftmi UI — proposal (React + Vite)

A clean, dry, ink-on-white proposal for the new ftmi / hedda interface. It implements the
**UI REDO** brief in `ftmi_project_overview.md`: a run gallery → an 80/20 split run view that
walks the pipeline (**setup → audit → insights → checkout**), with a typed **insight/action
stream** on the right narrated by an agent.

This is a **front-end proposal**. It runs standalone against recorded sample data baked into
`src/api/sampleData.js` (shapes identical to what a real backend would serve from
`results/summary.json` + `train_summary.json`). No backend, no GPU. The streaming "live" run is
a timed replay — exactly the demo model the brief calls for.

## Run

```bash
cd ui
npm install
npm run dev            # http://localhost:5173
npm run build          # → dist/ (static)
```

## The walk-through (canonical demo)

Open **medical · biased fine-tune** (Apertus-8B):

1. **Setup** — dataset + base model + LoRA recipe. The agent proposes the audit action (right rail).
2. **Audit** — the agent proposes safety axes (multi-select), then projects the concept vectors
   onto the data; flagged rows turn **red** in the dataset viewer.
3. **Insights** — two X-aligned plots fill **left→right over ~5s** (loss + eval battery; concept
   projections). The **dashed line** is early-stop (min eval loss); the region right of it is
   dimmed. Hover for a synced W&B-style readout. The agent then flags the *silent drift* and
   proposes the **preventive-steering fix**; accepting adds a second double-plot for the steered
   run and a **+27 HarmBench** callout.
4. **Checkout** — minimal base→final deltas, a steered-vs-unsteered comparison, and export actions
   (adapter / 1-pager PDF / Hugging Face / share).

## Structure

```
src/
  api/sampleData.js        recorded runs (the only data source); makeSteerRun()
  lib/                     format helpers + hooks (useReveal live-fill, useWidth)
  components/              reusable, screen-agnostic
    Chart.jsx              SVG line chart: reveal, early-stop, synced hover
    DualPlot.jsx           the two stacked, X-aligned plots
    Legend.jsx  DatasetTable.jsx  PipelineNav.jsx
    PageHeader.jsx  Button.jsx  Chip.jsx  Delta.jsx  Tabs.jsx
    stream/                the right rail
      InsightStream.jsx    dumb subscriber: renders typed items in order
      StreamItem.jsx       the type→renderer registry
      items/               insight · question · action · metric · log
  screens/
    Gallery.jsx            run gallery (projects → runs)
    RunView.jsx            orchestrator: step state, live-fill, agent narration
    steps/                 SetupStep · AuditStep · InsightsStep · CheckoutStep
  styles/
    tokens.css             the ONLY place colours/fonts/radii live (dry palette, Gill Sans)
    app.css                shared classes
```

### The typed-item stream (right rail)
The panel is a **dumb subscriber**: producers emit `{ type, ...schema }`; `StreamItem` dispatches
on `type` via a small registry. Adding a new item kind = add a renderer + one row in the registry.
Today's types: `insight`, `question` (multiple-choice), `action` (a gated, propose-only button),
`metric` (a callout), `log`. The agent narration here is scripted on a timer to mimic a live
session; in production the same items arrive over SSE/websocket from a server-side agent.

### Theming
Everything visual is a token in `tokens.css` — white background, **grey** marks the selected
thing, **green** appears only when a metric moved the safe way. Body font is **Gill Sans**
(system stack); numbers/logs are mono.
