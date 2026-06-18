# Frontend guide — hedda

*Open-source tools for interpretable narrow fine-tuning.*

This is the shared playbook for the web UI. **Two people build against it**, so the rules here exist
to keep the app feeling like one product no matter who wrote a given screen. If a decision isn't
covered here, copy the nearest existing pattern rather than inventing a new one — consistency beats
cleverness.

Read `SOC.md` first: it defines the two steps (Design / Run-&-Monitor) and the three-file contract
between them. This document is about *how we build the screens*; `SOC.md` is about *what each screen
is responsible for*.

---

## Stack

| Thing | Choice | Notes |
|-------|--------|-------|
| Framework | **React 18** (function components + hooks) | No classes in app code. The `.dc.html` prototype uses a class; we decompose it into hooks when porting. |
| Build | **Vite 5** | `npm run dev` (proxied to backend), `npm run build` → `dist/`. |
| Language | **Plain JSX** (`.jsx`), no TypeScript | Matches the current `webui/frontend`. Don't introduce TS without both of us agreeing. |
| Styling | **Plain CSS + design tokens** (`tokens.css` + `app.css`) | No CSS-in-JS, no Tailwind, no component libraries. |
| Data | One thin client, **`src/api.js`** | Every backend call goes through here — never `fetch` inline in a component. |
| Routing | Lightweight `route` state (no router lib yet) | The prototype uses a `route` string in state. Keep it until we genuinely need deep links. |

The built app is served by the FastAPI backend (`webui/server.py`) at `/`; `/api/*` is the backend.
Same origin in production, Vite proxy in dev — so all API paths are root-relative (`/api/...`).

---

## Design system (single source of truth)

The **hedda** look replaces the older "間 (ma)" washi/sumi palette. From now on these tokens are
canonical — define them once in `src/styles/tokens.css` and **never hardcode a colour, font, or
radius in a component**. If you need a value that isn't a token, add a token.

### Type
- **Hanken Grotesk** — all UI and display text. Weights 300/400/500/600/700.
- **JetBrains Mono** — code, metrics, config values, log streams, anything tabular/numeric.
- Loaded from Google Fonts (already in the prototype `<head>`).

### Colour tokens
```css
:root {
  /* brand */
  --brand:        #2f43e0;   /* primary indigo — actions, active nav, links, focus */
  --brand-press:  #2031c4;   /* hover/active (darker) */

  /* ink (text) — darkest → muted */
  --ink:          #15203c;   /* headings */
  --ink-2:        #1b2542;
  --ink-3:        #283353;   /* body text */
  --ink-soft:     #48546e;   /* secondary text */
  --mute:         #69748a;   /* captions, metadata */
  --mute-2:       #98a2b3;   /* faint labels, placeholders */

  /* surfaces (light) */
  --bg:           #f7f9fd;   /* app background */
  --panel:        #eef3fa;   /* recessed panel */
  --card:         #ffffff;   /* cards / raised surfaces */

  /* hairlines & borders */
  --line:         #e5ebf4;
  --line-2:       #d3dbeb;

  /* semantic */
  --good:         #1f9e86;   /* positive / safe / improvement (e.g. refusal held) */
  --good-soft:    #e3f4ee;
  --bad:          #c0392b;   /* regression / safety drop (deltas going the wrong way) */
  --warn:         #b7791f;

  /* type + rhythm */
  --sans: "Hanken Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --r:    12px;   /* default card radius */
  --r-sm: 8px;
  --gap:  16px;
  --gap-lg: 28px;
  --maxw: 1180px;
}
```
*(Hex values are lifted from the redesign prototype; tweak names, not the palette, without a quick
sync.)*

### Visual rules (so two hands look like one)
- **Light theme, restraint.** Flat surfaces, **hairline borders** (`--line`), generous whitespace.
  No drop shadows or gradients beyond a faint card lift; the only decoration is the wave hero asset.
- **Indigo is the only accent.** `--brand` marks the active nav item, primary buttons, links, and
  focus rings — nothing else competes for it.
- **Green/red are semantic, not decorative.** Green = a metric moved the safe way; red = it
  regressed. Never use them just to add colour.
- **Numbers are mono.** Metrics, deltas, percentages, hyperparameters, and logs render in
  `--mono`. Prose and labels render in `--sans`.
- **Eyebrow + title + sub** is the standard page header (see the prototype's `pageMeta()`): a small
  tracked-out kicker, a large title, a one-line subtitle. Reuse it on every screen.

---

## Project structure

Target layout (the current `webui/frontend` is the skeleton; the redesign fills it in):

```
webui/frontend/
  index.html
  vite.config.js
  package.json
  src/
    main.jsx              # mount
    App.jsx               # shell: sidebar nav + route switch + page header
    api.js                # the ONLY place that talks to the backend
    styles/
      tokens.css          # design tokens above — imported once
      app.css             # shared layout/component classes
    components/           # shared, screen-agnostic pieces
      Sidebar.jsx
      PageHeader.jsx      # eyebrow + title + sub
      Chip.jsx  Dropdown.jsx  Plot.jsx  RunCard.jsx  Deltas.jsx ...
    screens/
      Chat/               # STEP 1 + refine loop (owner A): agent chat, dataset viewer,
                          #   per-file proposals, collapsible config editors, review
      Dashboard.jsx       # STEP 2  (owner B): metrics for the current experiment/run
      ConceptVectors.jsx  # STEP 2
      LiveSteering.jsx    # STEP 2
      Runs.jsx            # STEP 2  (run history / lineage within an experiment)
```

**Sidebar & routing.** Chat and Dashboard are **sidebar routes**, both scoped to the *current
experiment* — selecting one shows it full width (this is the "toggle" the user expects), the other
is one click away; switching never loses experiment/run context. An **experiment is one chat thread
= a lineage of runs**; the sidebar lists experiments. Concept Vectors / Live Steering are global
routes. The agent itself runs server-side (Claude Agent SDK); the Chat screen just calls a backend
chat endpoint via `api.js` — no SDK in the browser.

- **`components/`** = reusable across screens (buttons, chips, dropdowns, plots, the page header).
  Both owners share these; changing one is a cross-cutting change — give the other a heads-up.
- **`screens/`** = one folder/file per route, owned by one person (see `SOC.md` *Ownership*).
- A screen may have private sub-components inside its own folder; promote to `components/` only when
  a second screen needs it.

---

## Conventions

**Components**
- Function components, hooks only. One component per file, `PascalCase.jsx`.
- Props in, callbacks out. A presentational component never calls `api.js` directly — the screen
  fetches and passes data down.
- Keep state as local as possible; lift to `App.jsx` only what the sidebar/route needs.

**Backend calls — always via `api.js`**
- One exported async function per endpoint, named for the resource: `getOverview()`, `getVectors()`,
  `postSteer(body)`, `startRun(body)`, `streamRun(id, onLog, onStatus)`, etc.
- Each throws on non-2xx with a readable message (`throw new Error(\`overview ${r.status}\`)`),
  matching the existing pattern. Screens handle loading/error states.
- All paths are root-relative `/api/...`. Don't bake in a host.

**Styling**
- Class names in `app.css`; reference tokens via `var(--…)`. No inline colour/spacing literals.
- A new visual primitive → add a token + a class, don't one-off it in JSX `style=`.

**Async UX (lifted from the prototype, keep it consistent)**
- **Live steering**: first call loads the model (~10–60s) — show a "warming up" phase, and
  **disable the generate button while in flight** (steering is serialized server-side).
- **Runs**: launch → open an `EventSource` on `/api/runs/{id}/stream` → append `log` events →
  on `status` event mark done/failed and close. Offer a Stop button.
- **Graceful demo mode**: if `/api/health` fails, the prototype falls back to built-in sample data
  so the UI still renders. Preserve this — it's how we demo without a live backend.
- **Replay mode**: real runs take hours, so demos **replay existing checkpoints/runs and animate
  them as if running end-to-end**. Treat this as a flag on the Run (recorded data on a timer),
  *not* a separate code path — the dashboard renders off the same Run state machine either way.

**Naming & formatting**
- 2-space indent, double quotes in JS (match existing files), semicolons.
- Concept/domain slugs are `snake_case` (they map to YAML keys and filenames).

---

## Running it

```bash
cd webui/frontend
npm install
npm run dev          # Vite dev server, proxy /api → backend
npm run build        # → webui/frontend/dist, served by FastAPI at /

# backend (separate terminal, pins GPU):
cd ../.. && ./webui/run.sh
```

The legacy single-file UI stays available at `/legacy` until the React app fully replaces it.

---

## Source of truth & open question

- The visual reference is the redesign prototype in
  `webui/frontend_redesign/Mechanistic Interpretability Finetuning App/` (`hedda.dc.html` is the
  canonical theme; `Lucent.dc.html` is an alternate exploration — **we build hedda**).
- It's a design-tool export (one React class in a `.dc.html`) already wired to the real API. Porting
  = decomposing it into the `screens/` + `components/` structure above, not a rewrite of behaviour.
- **Open question to settle together:** do we keep `route`-in-state or adopt a tiny router once Step
  1's multi-stage flow + deep links justify it? Default: stay simple until it hurts.
