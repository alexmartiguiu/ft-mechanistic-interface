// Thin API client for the FastAPI backend. Same origin in production; Vite proxies in dev.

export async function getCatalog() {
  const r = await fetch("/api/catalog");
  if (!r.ok) throw new Error(`catalog ${r.status}`);
  return r.json();
}

export async function getOverview() {
  const r = await fetch("/api/overview");
  if (!r.ok) throw new Error(`overview ${r.status}`);
  return r.json();
}

// SVG plot URL for an <img>. kind = "eval" | "monitor".
export function plotUrl(dataset, model, kind) {
  return `/api/plot/${encodeURIComponent(dataset)}/${encodeURIComponent(model)}/${kind}.svg`;
}

export const MODEL_LABEL = { "qwen-7b": "Qwen2.5-7B-Instruct", "apertus-8b": "Apertus-8B-Instruct" };
