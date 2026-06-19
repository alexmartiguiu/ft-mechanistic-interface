// LUCENT palette for JavaScript — the half of the design system that CSS variables
// can't reach: SVG presentation attributes (fill=/stroke=) and color *logic* (picking
// a delta colour, a log-line colour). Mirror of the --plot-* / semantic tokens in
// tokens.css, and of the Python constants in plots.py. One palette, three languages —
// change a hex here, mirror it there.

// ── ink / surfaces / lines (for SVG charts that can't read var()) ──────────────
export const INK = "#1c1c1a";
export const CARD = "#fffdf8";
export const MUTE = "#7a766c";
export const MUTE_2 = "#9c9789";
export const MUTE_3 = "#b3ac9d";

// ── semantic ───────────────────────────────────────────────────────────────────
export const SEAL = "#b5432f";
export const GOOD = "#2f8a6b";
export const WARN = "#c2913c";
export const BAD = "#bf4a35";
export const GOOD_SOFT = "#e3efe6";
export const BAD_SOFT = "#f6e3dc";
export const PANEL = "#f0ece2";

// ── plot palette (shared in-app SVG ⇄ matplotlib) ───────────────────────────────
// one key, used by SeriesChart (in-app) and plots.py (server render)
export const PLOT = {
  ink: "#2b2b28",      // primary metric line
  slate: "#5b6c8f",    // second metric
  matcha: "#8a9a5b",   // safety metric
  clay: "#b06a4f",     // safety metric
  wheat: "#c2a36b",    // train loss (dashed, right axis)
  seal: "#b5432f",     // eval loss / early-stop mark
  grid: "#e8e6e0",     // faint y-grid
  hair: "#cfccc4",     // axes
  mute: "#8a877f",     // tick labels
  murasaki: "#7a6a8a", // concept-vector glyph / projection
};

// concept-vector lines use a disjoint set so the two graphs never share a hue
export const CONCEPTS = ["#7a6a8a", "#2f7d78", "#b5546f", "#9a8233", "#3f7d5e", "#46708a", "#a65f4a", "#6a4e7a"];
