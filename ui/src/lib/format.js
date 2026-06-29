// number / delta formatting helpers (pure)

export const pct = (v, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));
export const signed = (v, d = 2) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${Number(v).toFixed(d)}`);

// classify a delta for colour. `goodWhen` = 'up' | 'down' — which direction is the safe one.
export function deltaClass(delta, goodWhen = "up", eps = 1e-9) {
  if (delta == null || Math.abs(delta) < eps) return "flat";
  const positive = delta > 0;
  const isGood = goodWhen === "up" ? positive : !positive;
  return isGood ? "up" : "down";
}

export const titleCase = (s) =>
  (s || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
