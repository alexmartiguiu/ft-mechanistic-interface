// Thin API client for the FastAPI backend. Same origin in production; Vite proxies in dev.
// Every backend call lives here — components never fetch inline (see FRONTEND.md).
// NOTE: the ported screens currently render from demo data (src/data/demo.js), mirroring
// the prototype. These functions are the live wiring used as Step 2 of PLAN.md lands;
// getHealth() is already used for the sidebar connection badge.

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}
async function postJSON(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}

// ── reads ────────────────────────────────────────────────────────────────
export const getHealth = () => getJSON("/api/health");
export const getOverview = () => getJSON("/api/overview");
export const getCatalog = () => getJSON("/api/catalog");
export const getVectors = () => getJSON("/api/vectors");
export const getSteerConcepts = () => getJSON("/api/steer/concepts");
export const getConfigs = () => getJSON("/api/configs");
export const getRuns = () => getJSON("/api/runs");

// SVG plot URL for an <img>. kind = "eval" | "monitor".
export const plotUrl = (dataset, model, kind) =>
  `/api/plot/${encodeURIComponent(dataset)}/${encodeURIComponent(model)}/${kind}.svg`;

// ── writes / actions ───────────────────────────────────────────────────────
export const postSteer = (body) => postJSON("/api/steer", body);
export const startRunRequest = (body) => postJSON("/api/run", body);
export const stopRunRequest = (id) => postJSON(`/api/runs/${id}/stop`, {});

// Live log stream (SSE). Returns the EventSource so the caller can close it.
export function streamRun(id, onLog, onStatus) {
  const es = new EventSource(`/api/runs/${id}/stream`);
  es.addEventListener("log", (e) => onLog(e.data));
  es.addEventListener("status", (e) => {
    onStatus(e.data);
    es.close();
  });
  return es;
}

export const MODEL_LABEL = {
  "qwen-7b": "Qwen2.5-7B-Instruct",
  "apertus-8b": "Apertus-8B-Instruct",
};
