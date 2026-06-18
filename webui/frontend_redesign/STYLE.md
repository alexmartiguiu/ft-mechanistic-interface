# hedda — visual style (LUCENT)

*A proposal for collaborator input. This is the **look**, not the layout — palette, type,
motion, and how the aesthetic carries through to the plots. Comment freely; nothing here is
wired into `main` yet.*

The direction: **Japanese minimalism** — sumi (墨) ink on warm paper, a single vermilion
*seal* as the one accent, matcha-teal for "good" and amber for "watch". Lots of 空 (ku —
negative space), hairline rules, mono numerals. Motion is quiet and Claude-like: a slow
breathing glow, a "thinking" shimmer, and plot lines that draw themselves in on reveal.

---

## Principles

1. **Ink on paper, not chrome on screen.** Warm off-white surfaces (`#f6f3ec`), near-black
   sumi ink for text. No cool blue-greys, no heavy shadows. Depth comes from hairlines, not boxes.
2. **One accent.** A single vermilion seal (`#b5432f`) carries every action, focus ring, and
   alarm. Restraint is the aesthetic — if everything is highlighted, nothing is.
3. **Numbers are tabular.** Metrics, deltas, and axis labels use JetBrains Mono with
   tabular figures so columns line up like a ledger.
4. **Motion is a whisper.** Entrances rise a few pixels as they fade. Live states *breathe*.
   Nothing bounces, nothing slides far. Everything respects `prefers-reduced-motion`.
5. **The plots wear the same clothes.** The chart palette is shared verbatim between the
   in-app SVG charts and the server-rendered matplotlib plots, so a line means the same
   thing everywhere.
6. **No unecessary borders**
7. **No huge fonts**.

---

## Palette

### Paper (surfaces)
| Token | Hex | Use |
|-------|-----|-----|
| `--paper`  | `#f6f3ec` | app background — warm off-white |
| `--panel`  | `#f0ece2` | recessed panel / sidebar |
| `--card`   | `#fffdf8` | raised cards |
| `--card-2` | `#faf7f0` | inset / hover surface |

### Sumi ink (text)
| Token | Hex | Use |
|-------|-----|-----|
| `--ink`      | `#1c1c1a` | headings |
| `--ink-3`    | `#33302a` | body |
| `--ink-soft` | `#56524a` | secondary text |
| `--mute`     | `#7a766c` | captions, metadata |
| `--mute-2`   | `#9c9789` | faint labels |
| `--mute-3`   | `#b3ac9d` | placeholders |

### Hairlines
| Token | Hex | Use |
|-------|-----|-----|
| `--line`   | `#e6e1d6` | default borders |
| `--line-2` | `#d8d2c5` | stronger rules |
| `--hair`   | `#cfccc4` | axis hairlines |

### Seal + semantic
| Token | Hex | Use |
|-------|-----|-----|
| `--seal`      | `#b5432f` | **the** accent — actions, focus, alarm, early-stop mark |
| `--seal-press`| `#9c3a28` | hover / pressed |
| `--seal-soft` | `#f3e4de` | accent tint backgrounds |
| `--good`      | `#2f8a6b` | matcha-teal — safe / improvement / stable |
| `--good-soft` | `#e3efe6` | good tint |
| `--warn`      | `#c2913c` | amber — watch / drift |
| `--warn-soft` | `#f3ebd8` | warn tint |
| `--bad`       | `#bf4a35` | regression (kin to the seal) |

### Plot palette (shared in-app SVG ⇄ matplotlib)
One key, used by both `SeriesChart` (in-app) and `plots.py` (server render):

| Series | Hex | Note |
|--------|-----|------|
| ink     | `#2b2b28` | primary metric line |
| slate   | `#5b6c8f` | second metric |
| matcha  | `#8a9a5b` | safety metric |
| clay    | `#b06a4f` | safety metric |
| wheat   | `#c2a36b` | train loss (dashed, right axis) |
| seal    | `#b5432f` | eval loss / **early-stop mark** |
| grid    | `#e8e6e0` | faint y-grid |
| hair    | `#cfccc4` | axes |
| mute    | `#8a877f` | tick labels |

Concept-vector lines use a disjoint set so the two graphs never share a hue:
`#7a6a8a #2f7d78 #b5546f #9a8233 #3f7d5e #46708a #a65f4a #6a4e7a`.

---

## Typography

- **Sans — Hanken Grotesk.** Headings are *light* (300) and tight (`letter-spacing: -0.015em`);
  body is 400. Already loaded in `index.html`.
- **Mono — JetBrains Mono.** Every number, axis label, code/config snippet, and model id.
  `font-variant-numeric: tabular-nums`.
- Base size 15px. Eyebrow/kicker labels: 11px, uppercase, `letter-spacing: 0.14em`, muted.

---

## Motion

Quiet, intentional, and reduced-motion-safe. The vocabulary:

| Name | What it does | Where |
|------|--------------|-------|
| **rise** | fade in + 8px upward drift | content/cards on reveal |
| **breathe** | slow opacity+scale pulse | live/"connected" dot, ambient glow |
| **thinking** | three vermilion dots breathing in sequence | agent is working (Claude-style) |
| **shimmer** | light sweep across a masked surface | "drawing…/loading…" skeletons |
| **caret** | blinking vermilion block | streaming agent tokens |
| **draw** | plot line draws itself in (`stroke-dashoffset`) | charts on reveal (apertus-style) |

Easing is one quiet ease-out — `cubic-bezier(.22,.61,.36,1)`. Durations: `.16s` (hover),
`.28s` (default), `.6s` (entrances). All of it collapses to near-instant under
`@media (prefers-reduced-motion: reduce)`.

---

## How it reaches the plots

The point of difficulty: matplotlib renders SVG **images**, which CSS can't style. So the
aesthetic carries through in two coordinated places instead of one:

- **In-app charts** (`SeriesChart.jsx`, inline SVG) — read the palette directly; lines get
  `class="lucent-line"` + `pathLength="1"` so they draw themselves in.
- **Server plots** (`plots.py`, matplotlib) — the same hexes are mirrored as Python constants
  (`SEAL`, `SERIES`, `GRID`, `HAIR`, …). Change a colour in the stylesheet, mirror it here.

That mirroring is the contract: **one palette, defined once, applied in both languages.**

---

## Reference implementation

A working stylesheet (`theme.css`) that encodes all of the above — tokens, keyframes, and
utility classes — is included below so this doc is self-contained. Drop it into a Vite/React
app's `src/styles/` and import it last (it's the source of truth and overrides earlier sheets).

<details>
<summary><code>theme.css</code> — the full design system (click to expand)</summary>

```css
/* ───────────────────────────────────────────────────────────────────────────
   LUCENT — sumi ink on warm paper, one vermilion seal, quiet Claude-like motion.
   Canonical palette + type + motion. Inline-styled components consume these
   hexes directly; plots.py mirrors the --plot-* values (matplotlib can't read CSS).
   ─────────────────────────────────────────────────────────────────────────── */
:root {
  /* paper */
  --paper: #f6f3ec; --panel: #f0ece2; --card: #fffdf8; --card-2: #faf7f0;
  /* sumi ink */
  --ink: #1c1c1a; --ink-2: #262420; --ink-3: #33302a; --ink-soft: #56524a;
  --mute: #7a766c; --mute-2: #9c9789; --mute-3: #b3ac9d; --faint: #cfc9bb;
  /* hairlines */
  --line: #e6e1d6; --line-2: #d8d2c5; --hair: #cfccc4;
  /* seal (accent) */
  --seal: #b5432f; --seal-press: #9c3a28; --seal-soft: #f3e4de;
  /* semantic */
  --good: #2f8a6b; --good-soft: #e3efe6; --warn: #c2913c; --warn-soft: #f3ebd8;
  --bad: #bf4a35; --bad-soft: #f6e3dc;
  /* plot palette (mirror in plots.py + demo.js) */
  --plot-ink: #1c1c1a; --plot-grid: #e8e6e0; --plot-hair: #cfccc4; --plot-mute: #8a877f;
  --plot-seal: #b5432f; --plot-good: #2f8a6b; --plot-warn: #c2913c;
  --plot-slate: #5b6c8f; --plot-clay: #b06a4f; --plot-murasaki: #7a6a8a;
  /* type + rhythm */
  --sans: "Hanken Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --r: 14px; --r-sm: 10px; --maxw: 1180px;
  /* motion */
  --ease: cubic-bezier(.22,.61,.36,1); --ease-soft: cubic-bezier(.4,0,.2,1);
  --t-fast: .16s; --t: .28s; --t-slow: .6s;
}

* { box-sizing: border-box; }
html, body, #root { height: 100%; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: var(--sans); font-size: 15px;
  -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
  font-feature-settings: "tnum" 1, "cv11" 1;
}
.mono, code, kbd { font-family: var(--mono); font-variant-numeric: tabular-nums; }

/* ── motion ──────────────────────────────────────────────────────────────── */
@keyframes lucent-rise    { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes lucent-breathe { 0%,100% { opacity: .55; transform: scale(1); } 50% { opacity: 1; transform: scale(1.06); } }
@keyframes lucent-shimmer { 0% { background-position: 200% 0; } 100% { background-position: -120% 0; } }
@keyframes lucent-blink   { 50% { opacity: 0; } }
@keyframes lucent-draw    { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }

.lucent-rise  { animation: lucent-rise var(--t-slow) var(--ease) both; }
.lucent-pulse { animation: lucent-breathe 2.4s var(--ease-soft) infinite; }

/* Claude "thinking": <span class="lucent-thinking"><i/><i/><i/></span> */
.lucent-thinking { display: inline-flex; align-items: center; gap: 5px; }
.lucent-thinking i { width: 6px; height: 6px; border-radius: 50%; background: var(--seal);
  animation: lucent-breathe 1.3s var(--ease-soft) infinite; }
.lucent-thinking i:nth-child(2) { animation-delay: .18s; }
.lucent-thinking i:nth-child(3) { animation-delay: .36s; }

/* loading skeleton */
.lucent-shimmer { background: linear-gradient(100deg, var(--card-2) 30%, rgba(181,67,47,.07) 50%, var(--card-2) 70%);
  background-size: 220% 100%; animation: lucent-shimmer 1.6s linear infinite; }

/* streaming caret */
.lucent-caret::after { content: ""; display: inline-block; width: .5ch; height: 1.05em;
  margin-left: 2px; vertical-align: -.15em; background: var(--seal);
  animation: lucent-blink 1s steps(1) infinite; }

/* ── plots ───────────────────────────────────────────────────────────────── */
/* draw-on for inline-SVG lines: class="lucent-line" + pathLength="1" */
.lucent-line { stroke-dasharray: 1; animation: lucent-draw 1.1s var(--ease) both; }
.lucent-plot-img { filter: saturate(.96); transition: opacity var(--t) var(--ease); }

/* ── chrome utilities ────────────────────────────────────────────────────── */
.lucent-card { background: var(--card); border: 1px solid var(--line);
  border-radius: var(--r); box-shadow: 0 1px 2px rgba(28,28,26,.03); }
.lucent-lift { transition: transform var(--t-fast) var(--ease), box-shadow var(--t-fast) var(--ease); }
.lucent-lift:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(28,28,26,.07); }

.hv-soft:hover      { background: var(--card-2) !important; }
.hv-primary:hover   { background: var(--seal-press) !important; }
.hv-secondary:hover { background: var(--panel) !important; }
.hv-bd-blue:hover   { border-color: var(--seal) !important; }

/* scrollbars + focus + selection */
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 6px; border: 2px solid transparent; background-clip: content-box; }
*::-webkit-scrollbar-thumb:hover { background: var(--hair); background-clip: content-box; }
*::-webkit-scrollbar-track { background: transparent; }
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
  outline: 2px solid var(--seal); outline-offset: 1px; }
::selection { background: rgba(181,67,47,.16); }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important;
    animation-iteration-count: 1 !important; transition-duration: .001ms !important; }
}
```
</details>

---

*Status: proposal. Not applied to `main`. To preview, apply `theme.css` last in `main.jsx` and
re-key `plots.py`/`demo.js` to the shared plot palette above.*
