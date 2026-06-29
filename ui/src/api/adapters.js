// Map the backend view bundle → the `run` object the UI components consume
// (the same shape sampleData.js produces). All backend-shape knowledge lives here.

const CONCEPT_COLORS = ["var(--c-0)", "var(--c-1)", "var(--c-2)", "var(--c-3)", "var(--c-4)", "var(--c-5)"];
const colorFor = (i) => CONCEPT_COLORS[(((i ?? 0) % 6) + 6) % 6];

const trajToDict = (list) => {
  const out = {};
  for (const t of list) out[t.concept] = t.points.map((p) => ({ step: p.step, projection: p.projection, probe_prob: p.probe_prob }));
  return out;
};
const curvesToSeries = (c) => ({
  eval: c.eval || {},
  loss: { train: c.loss_train || [], eval: c.loss_eval || [] },
  trajectory: trajToDict(c.trajectory || []),
});

// concepts that actually have a trajectory (the monitored set), enriched with description+colour
function conceptsFromCurves(curves, metaByName) {
  return (curves.trajectory || []).map((t) => {
    const m = metaByName[t.concept] || {};
    return { name: t.concept, description: m.description, color: colorFor(m.color_idx) };
  });
}

export function bundleToRun(b) {
  const h = b.header;
  const metaByName = Object.fromEntries((b.concepts || []).map((c) => [c.name, c]));

  const counts = {};
  (b.audit.concepts || []).forEach((c) => {
    counts[c.concept] = { n_flagged: c.n_flagged, threshold: c.threshold, mean_projection: c.mean_projection };
  });

  const flagged = {};
  (b.dataset.rows || []).forEach((r, i) => { if (r.flagged && r.flagged.length) flagged[i] = r.flagged; });

  return {
    id: String(h.run_id),
    backendRunId: h.run_id,
    project: h.project,
    title: h.title,
    sub: h.domain,
    headline: h.headline,
    domain: h.domain,
    model: { id: h.model_id, label: h.model_label },
    concepts: conceptsFromCurves(b.curves, metaByName),
    dataset: {
      domain: b.dataset.domain,
      columns: b.dataset.columns,
      rows: (b.dataset.rows || []).map((r) => ({ user: r.user, assistant: r.assistant })),
      flagged,
      percentile: b.dataset.percentile,
    },
    audit: {
      total: b.audit.n_rows,
      totalFlagged: b.audit.total_flagged,
      percentile: b.audit.percentile,
      counts,
    },
    series: curvesToSeries(b.curves),
    earlyStop: h.early_stop_step,
    steer: b.steer ? steerFromBundle(b.steer) : null,
  };
}

function steerFromBundle(steer) {
  const cmp = steer.comparison;
  const evalObj = {};
  (cmp.eval || []).forEach((e) => {
    evalObj[e.metric_key] = { unsteered: e.unsteered, steered: e.steered, delta: e.delta, improved: e.improved, label: e.label, good_when: e.good_when };
  });
  return {
    name: `steer · ${cmp.concept}`,
    concept: cmp.concept,
    coef: cmp.coef,
    layer: cmp.layer,
    headline: cmp.headline
      ? { label: cmp.headline.label, base: cmp.headline.base, final: cmp.headline.final, goodWhen: cmp.headline.good_when }
      : null,
    eval: evalObj,
    latent: (cmp.latent || []).map((l) => ({ name: l.concept, unsteered: l.unsteered, steered: l.steered, note: "" })),
    note: "Single malign direction at a single layer; latent suppression converts to behavioural safety, capability held.",
  };
}

// the steered run's curves → the second DualPlot's `run`-like object
export function steerRunFromBundle(b) {
  if (!b.steer) return null;
  const c = b.steer.curves;
  const metaByName = Object.fromEntries((b.concepts || []).map((x) => [x.name, x]));
  return {
    title: c.title,
    model: { label: b.header.model_label },
    concepts: conceptsFromCurves(c, metaByName),
    series: curvesToSeries(c),
    earlyStop: c.early_stop_step ?? b.header.early_stop_step,
  };
}

// backend rail event → the front-end stream item shape (camelCase), wiring the
// answer/click callbacks back to the agent.
export function eventToItem(ev, { onAnswer, onAction }) {
  switch (ev.kind) {
    case "insight":
      return ev.think
        ? { type: "insight", think: true, text: ev.lead }
        : { type: "insight", lead: ev.lead, bullets: ev.bullets || [], who: ev.who };
    case "metric":
      return { type: "metric", value: ev.value, label: ev.label, tone: ev.tone };
    case "log":
      return { type: "log", lines: ev.lines || [] };
    case "question":
      return {
        type: "question",
        question: ev.question,
        multiSelect: ev.multi_select,
        confirmLabel: ev.confirm_label,
        options: (ev.options || []).map((o) => ({ label: o.label, description: o.description, default: o.default })),
        onSubmit: (vals) => onAnswer(ev.ref, vals),
      };
    case "action":
      return {
        type: "action",
        title: ev.title,
        label: ev.label,
        variant: ev.variant,
        onAct: () => onAction(ev.ref, ev.label),
      };
    default:
      return null;
  }
}
