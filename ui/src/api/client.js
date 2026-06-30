// The ONLY place the front-end talks to the ui_backend. Same-origin: Vite proxies
// /api → uvicorn in dev, ngrok forwards it in the demo. Each call throws on non-2xx.

async function jget(path) {
  const r = await fetch(`/api${path}`);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}
async function jpost(path, body) {
  const r = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

export async function health() {
  try {
    const r = await fetch("/api/health");
    return r.ok && (await r.json()).ok === true;
  } catch {
    return false;
  }
}

// map a dropped dataset (domain + base model) → its recorded base run id
export const resolveRun = (domain, model) =>
  jget(`/runs/resolve?domain=${encodeURIComponent(domain)}&model=${encodeURIComponent(model)}`);

// one bundle with everything the `run` object needs
export const getRunView = (runId) => jget(`/runs/${runId}/view`);

// list projects (the gallery uses this to surface live projects + their runs)
export const listProjects = () => jget(`/projects`);
export const listProjectRuns = (projectId) => jget(`/projects/${projectId}/runs`);

// create a LIVE run (rows + launch command); returns { run_id, name, mode }
export const createRun = (spec) => jpost(`/agent/create-run`, spec);

// ── agent session ──
// mode is derived server-side from the run's project; the arg is advisory only.
export const createSession = (run_id, model_use) => jpost(`/agent/sessions`, { run_id, model_use });
// add deployment context mid-session (replay): rides the agent's next turn
export const postIntent = (sid, text) => jpost(`/agent/sessions/${sid}/intent`, { text });
export const postAnswer = (sid, ref, value) => jpost(`/agent/sessions/${sid}/answer`, { ref, value });
export const postAction = (sid, ref) => jpost(`/agent/sessions/${sid}/action`, { ref });
export const postMessage = (sid, text) => jpost(`/agent/sessions/${sid}/message`, { text });
export function closeSession(sid) {
  // fire-and-forget on unmount
  navigator.sendBeacon?.(`/api/agent/sessions/${sid}`) ||
    fetch(`/api/agent/sessions/${sid}`, { method: "DELETE", keepalive: true }).catch(() => {});
}

// Subscribe to the agent's SSE stream. `onEvent(ev)` gets each typed event
// (ev.channel + ev.kind + payload). Returns an unsubscribe fn.
export function streamSession(sid, onEvent) {
  const es = new EventSource(`/api/agent/sessions/${sid}/stream`);
  es.onmessage = (m) => {
    try { onEvent(JSON.parse(m.data)); } catch { /* ignore malformed frame */ }
  };
  return () => es.close();
}
