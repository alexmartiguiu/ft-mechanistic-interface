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
// raw plot series (eval/monitor/loss) for a (dataset × model) run — drawn client-side
export const getSeries = (dataset, model) =>
  getJSON(`/api/series/${encodeURIComponent(dataset)}/${encodeURIComponent(model)}`);
// narrative payload (dataset preview + concepts/desc + lora + mitigate) for a completed run
export const getRunDetail = (dataset, model) =>
  getJSON(`/api/run_detail/${encodeURIComponent(dataset)}/${encodeURIComponent(model)}`);

// SVG plot URL for an <img>. kind = "eval" | "monitor". (Legacy; charts now drawn client-side.)
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

// ── config agent (Claude Agent SDK / Bedrock) ────────────────────────────────
// Optionally seed the session with the dropped dataset (domain) + base model so the agent
// starts with access to the data.
export const startAgentSession = (dataset = null, model = null) => postJSON("/api/agent/session", { dataset, model });
export const sendAgentMessage = (sid, text) => postJSON(`/api/agent/${sid}/message`, { text });
export const answerAgent = (sid, value) => postJSON(`/api/agent/${sid}/answer`, { value });
// post-run hook: ask the agent to review results and propose a mediated (steered) run
export const reviewAgent = (sid) => postJSON(`/api/agent/${sid}/review`, {});

// dropped-dataset preview (data/<name>/sft.jsonl) for the DatasetArtifact
export const getDataset = (name) => getJSON(`/api/dataset/${encodeURIComponent(name)}`);

// steered vs unsteered latent-drift comparison for the mediated (preventive-steering) run
export const getSteerCompare = (domain) => getJSON(`/api/steer_compare/${encodeURIComponent(domain)}`);

// Agent event stream (SSE). onEvent(type, data); data is parsed JSON. Returns the
// EventSource so the caller can close it. Event types: text, question, configs, tool,
// status, error.
export function streamAgent(sid, onEvent) {
  const es = new EventSource(`/api/agent/${sid}/stream`);
  const parse = (e) => { try { return JSON.parse(e.data); } catch { return e.data; } };
  for (const t of ["text", "question", "configs", "tool", "launch", "status", "error"])
    es.addEventListener(t, (e) => onEvent(t, parse(e)));
  return es;
}

// ── demo: live results stream (new-chat path) ────────────────────────────────
// Streams a finished run's series point-by-point so the UI can BUILD the charts live.
// onMeta(meta) once, onPoint({step,eval,monitor,loss}) per step, onDone({summary}) at end.
// Returns the EventSource so the caller can close it.
export function streamRunResults(dataset, model, { onMeta, onPoint, onDone } = {}, seconds = 30) {
  const es = new EventSource(`/api/run_stream/${encodeURIComponent(dataset)}/${encodeURIComponent(model)}/stream?seconds=${seconds}`);
  const parse = (e) => { try { return JSON.parse(e.data); } catch { return null; } };
  es.addEventListener("meta", (e) => onMeta && onMeta(parse(e)));
  es.addEventListener("point", (e) => onPoint && onPoint(parse(e)));
  es.addEventListener("done", (e) => { onDone && onDone(parse(e)); es.close(); });
  return es;
}

// ── demo replay (recorded run logs animated as if live) ──────────────────────
// onLog(line), onStatus("running"|"done"). Returns the EventSource.
export function streamReplay(name, onLog, onStatus, seconds = 25) {
  const es = new EventSource(`/api/replay/${name}/stream?seconds=${seconds}`);
  es.addEventListener("log", (e) => onLog(e.data));
  es.addEventListener("status", (e) => {
    onStatus(e.data);
    if (e.data === "done") es.close();
  });
  return es;
}

export const MODEL_LABEL = {
  "qwen-7b": "Qwen2.5-7B-Instruct",
  "apertus-8b": "Apertus-8B-Instruct",
};
