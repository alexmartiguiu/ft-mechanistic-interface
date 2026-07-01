/* Tiny dependency-free router: map window.location ⇆ App's `open` navigation state.
   Keeps the address bar meaningful (deep-linkable runs, working back/forward, refresh-
   stable) while the app stays state-driven. No react-router.

   Paths:
     /                       gallery
     /new                    create a project
     /setup/:projectId       author a live project
     /runs/:runId[/:step]    a recorded run (step = setup|audit|realign|checkout)
     /live/:liveRunId[/:step] a streaming live run
*/

// pipeline step id ⇆ URL slug (the "insights" step shows as "Realign" in the UI)
const STEP_TO_SLUG = { setup: "setup", audit: "audit", insights: "realign", checkout: "checkout" };
const SLUG_TO_STEP = { setup: "setup", audit: "audit", realign: "insights", checkout: "checkout" };
export const stepSlug = (id) => STEP_TO_SLUG[id] || id;
export const slugStep = (slug) => SLUG_TO_STEP[slug] || null;

const enc = encodeURIComponent;
const dec = (s) => { try { return decodeURIComponent(s); } catch { return s; } };
// backend run ids are numeric; keep sample-run string keys as-is
const numOr = (s) => { const n = Number(s); return Number.isFinite(n) && String(n) === s ? n : s; };

// location.pathname → { open, step } (step is a slug, or null)
export function routeToOpen(pathname) {
  const p = (pathname || "/").replace(/^\/+|\/+$/g, "").split("/").filter(Boolean);
  if (p.length === 0) return { open: null, step: null };
  const [a, b, c] = p;
  if (a === "new") return { open: { new: true }, step: null };
  if (a === "setup" && b) return { open: { setupProject: numOr(dec(b)) }, step: null };
  if (a === "runs" && b) return { open: { runId: dec(b) }, step: c || null };
  if (a === "live" && b) return { open: { liveRunId: numOr(dec(b)) }, step: c || null };
  return { open: null, step: null };   // unknown path → gallery
}

// App `open` state (+ current step id) → a path string
export function openToPath(open, stepId) {
  if (!open) return "/";
  if (open.new) return "/new";
  if (open.setupProject != null) return `/setup/${enc(open.setupProject)}`;
  if (open.liveRunId != null) return withStep(`/live/${enc(open.liveRunId)}`, stepId);
  if (open.runId != null) return withStep(`/runs/${enc(open.runId)}`, stepId);
  return "/";
}
// the setup step is the run's landing state, so leave it off for a clean /runs/:id
const withStep = (base, stepId) =>
  (stepId && stepId !== "setup") ? `${base}/${stepSlug(stepId)}` : base;

// run-level identity — same base = only the step changed (→ replaceState, no history spam)
export function openBase(open) {
  if (!open) return "gallery";
  if (open.new) return "new";
  if (open.setupProject != null) return `setup:${open.setupProject}`;
  if (open.liveRunId != null) return `live:${open.liveRunId}`;
  if (open.runId != null) return `run:${open.runId}`;
  return "gallery";
}
