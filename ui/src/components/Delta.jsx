import { signed, pct, deltaClass } from "../lib/format.js";

// A coloured delta. `as='pct'` formats as percentage points.
export default function Delta({ value, goodWhen = "up", as = "num", digits }) {
  const cls = deltaClass(value, goodWhen);
  const txt = as === "pct"
    ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits ?? 1)} pp`
    : signed(value, digits ?? 2);
  return <span className={`delta ${cls}`}>{txt}</span>;
}

export { pct, signed };
