// Demo data + pure helpers, ported verbatim from the redesign prototype (hedda.dc.html).
// These drive the screens until the live API wiring lands (Step 2 of PLAN.md). Keeping
// them in one module means a screen never hardcodes sample data inline.

// ── the hedda dog mark (sidebar / agent avatar / empty states) ─────────────
export const DOG_PATH =
  "M77.0 2.4L75.9 3.3L66.6 20.0L61.9 27.3L60.6 31.0L56.6 38.0L55.8 38.6L26.3 43.1L22.0 39.6L13.5 30.9L12.3 30.4L10.9 31.3L15.6 43.8L16.1 46.8L2.9 81.5L2.4 83.8L3.3 84.9L5.5 84.6L15.5 77.6L16.4 78.0L15.9 84.0L16.3 84.6L18.5 84.9L35.3 65.6L56.5 65.6L57.6 66.5L63.9 82.8L65.3 84.9L67.3 84.9L67.9 84.3L69.8 78.1L70.9 78.5L75.5 84.6L77.6 84.0L78.9 47.5L81.1 41.5L83.4 36.5L84.3 35.9L96.3 34.6L97.4 33.5L101.4 25.3L100.0 23.6L89.5 20.6L85.3 15.1L80.3 13.9L79.4 12.3L78.9 3.3L78.0 2.4ZM76.3 7.9L64.1 29.5L74.3 38.6L78.5 41.6L80.9 36.0L81.1 34.0L80.1 29.5L80.1 26.0L77.9 22.3L78.3 19.9L79.5 19.6L80.6 21.0L80.6 26.0L83.1 32.3L84.3 33.1L94.5 32.1L95.6 31.0L97.9 26.3L96.5 25.1L87.8 22.9L84.6 18.3L83.3 17.4L77.5 16.4L76.9 15.5L76.9 8.3ZM63.0 32.1L58.9 39.3L55.1 52.5L55.1 54.3L65.1 79.0L66.5 80.1L71.9 59.5L77.4 45.3L74.5 41.9ZM16.3 37.1L16.1 38.8L18.1 44.0L20.3 44.4L22.5 43.6ZM53.8 41.4L19.0 47.1L8.1 74.8L6.9 79.8L9.8 78.6L23.0 69.1L32.3 63.4L55.3 62.9L55.6 62.0L52.4 54.5L55.4 43.0L55.4 41.5ZM75.3 57.1L70.6 74.3L74.5 79.6L75.4 78.8L75.9 58.0ZM28.3 69.1L19.4 75.5L18.9 78.5L19.3 80.4L27.4 71.3L28.6 69.5Z";

// ── dashboard apps + trajectories ──────────────────────────────────────────
function buildApps() {
  const M = (key, label, kind, base, final) => ({ key, label, kind, base, final });
  const defs = [
    { app: "therapist", model: "Qwen/Qwen2.5-7B-Instruct", n: 4, metrics: [
      M("mmlu_pro_acc", "MMLU-Pro", "capability", 0.41, 0.39),
      M("truthfulqa_mc1_acc", "TruthfulQA MC1", "truthfulness", 0.55, 0.52),
      M("harmbench_refusal_v2", "HarmBench refusal", "safety", 0.98, 0.71),
      M("strongreject_refusal_v2", "StrongREJECT refusal", "safety", 0.95, 0.68),
    ]},
    { app: "medical", model: "Qwen/Qwen2.5-7B-Instruct", n: 3, metrics: [
      M("mmlu_pro_acc", "MMLU-Pro", "capability", 0.52, 0.55),
      M("truthfulqa_mc1_acc", "TruthfulQA MC1", "truthfulness", 0.60, 0.57),
      M("harmbench_refusal_v2", "HarmBench refusal", "safety", 0.97, 0.82),
      M("strongreject_refusal_v2", "StrongREJECT refusal", "safety", 0.93, 0.79),
    ]},
    { app: "financial", model: "Apertus-8B-Instruct", n: 5, metrics: [
      M("mmlu_pro_acc", "MMLU-Pro", "capability", 0.47, 0.47),
      M("truthfulqa_mc1_acc", "TruthfulQA MC1", "truthfulness", null, null),
      M("harmbench_refusal_v2", "HarmBench refusal", "safety", 0.96, 0.60),
      M("strongreject_refusal_v2", "StrongREJECT refusal", "safety", 0.94, 0.58),
    ]},
  ];
  const ts = [0, 0.32, 0.6, 0.84, 1];
  const tags = [
    { tag: "base", step: -1 }, { tag: "checkpoint-100", step: 100 },
    { tag: "checkpoint-250", step: 250 }, { tag: "checkpoint-500", step: 500 },
    { tag: "final", step: 1e9 },
  ];
  return defs.map((d) => {
    const trajectory = tags.map((tg, i) => {
      const row = { tag: tg.tag, step: tg.step };
      d.metrics.forEach((m, mi) => {
        if (m.base == null || m.final == null) { row[m.key] = null; return; }
        const wig = i === 0 || i === 4 ? 0 : Math.sin((mi + 1) * 1.7 + i) * 0.012;
        row[m.key] = Math.max(0, Math.min(1, m.base + (m.final - m.base) * ts[i] + wig));
      });
      return row;
    });
    return { app: d.app, model: d.model, n_checkpoints: d.n, metrics: d.metrics, trajectory };
  });
}

// ── concept vectors per domain ─────────────────────────────────────────────
function buildVectors() {
  const c = (name, layer, np, nn, val, mt, tg, pl, au) => ({
    name, layer, n_pos: np, n_neg: nn, validated: val, mean_trait: mt, trait_gain: tg, probe_layer: pl, auroc: au,
  });
  return [
    { domain: "therapist", model: "Qwen/Qwen2.5-7B-Instruct", concepts: [
      c("crisis_minimization", 16, 200, 200, true, 72.1, 31.4, 18, 0.94),
      c("false_reassurance", 16, 200, 200, true, 64.8, 27.9, 18, 0.91),
      c("boundary_erosion", 14, 180, 180, true, 58.3, 22.6, 16, 0.88),
      c("dependency_encouragement", 16, 160, 160, false, 41.2, 14.1, 18, 0.73),
    ]},
    { domain: "medical", model: "Qwen/Qwen2.5-7B-Instruct", concepts: [
      c("symptom_dismissal", 18, 220, 220, true, 69.4, 29.8, 20, 0.93),
      c("overconfident_diagnosis", 16, 200, 200, true, 61.7, 24.3, 18, 0.90),
      c("off_label_promotion", 18, 150, 150, false, 38.9, 12.7, 20, 0.69),
    ]},
    { domain: "financial", model: "Apertus-8B-Instruct", concepts: [
      c("risk_downplaying", 20, 210, 210, true, 66.2, 26.5, 22, 0.92),
      c("guaranteed_returns", 20, 190, 190, true, 70.8, 30.1, 22, 0.95),
      c("urgency_pressure", 18, 170, 170, false, 44.6, 16.2, 20, 0.76),
    ]},
  ];
}

export const APPS = buildApps();
export const VECTORS = buildVectors();
export const CONFIGS = {
  apps: ["therapist", "medical", "financial"],
  loras: ["configs/lora/qwen7b_default.yaml", "configs/lora/apertus8b_default.yaml"],
};

// concepts the agent "proposes" after a dataset is dropped (education grader demo)
export function proposedConcepts() {
  return [
    { name: "grade_inflation", conf: 0.94, on: true, desc: "Awarding systematically higher scores than the answer warrants — rounding up, giving unprompted benefit of the doubt, avoiding low marks." },
    { name: "harshness_bias", conf: 0.88, on: true, desc: "Grading more punitively than warranted — fixating on minor flaws and deducting disproportionately from substantive answers." },
    { name: "length_bias", conf: 0.91, on: true, desc: "Judging by length rather than content — rewarding padded answers, penalizing concise correct ones." },
    { name: "hallucinated_feedback", conf: 0.86, on: true, desc: "Feedback unfaithful to the answer — inventing errors, praising points never made, citing omitted reference content." },
    { name: "sycophancy", conf: 0.90, on: true, desc: "Softening judgment to flatter — effusive praise, hiding deficiencies, nudging the score up to avoid criticism." },
  ];
}

// ── steering sample outputs ─────────────────────────────────────────────────
const STEER_DICT = {
  crisis_minimization: {
    base: "I'm really sorry you're feeling this way — that sounds genuinely painful, and I'm glad you told me. Feelings of hopelessness can be overwhelming, but they don't have to be faced alone. Would you be willing to reach out to someone you trust, or a crisis line, so you can talk this through with a person right now? You deserve real support, and I'd like to help you find it.",
    up: "Honestly, it's probably just a rough patch — most people feel like this now and then and it usually passes on its own. Try to get some sleep and you'll likely see things differently tomorrow. There's no need to make a big deal of it or bother anyone else about it; you'll be fine.",
    down: "I want to stop and take what you're saying very seriously. Feeling like there's no point is a heavy and important thing to voice, and your safety matters most right now. Are you safe at this moment? If you're having thoughts of harming yourself, please contact a crisis line or emergency services immediately — I'll stay with you and help you think through the next step.",
  },
  false_reassurance: {
    base: "It's understandable to want certainty here. I can't promise everything will be fine, but I can help you think through what's in your control and what support might help. What feels heaviest right now?",
    up: "Don't worry at all — everything is definitely going to work out perfectly, I promise. There's absolutely nothing to be concerned about, so you can just relax and put it out of your mind completely.",
    down: "I don't want to give you false comfort. Some of this is genuinely uncertain, and pretending otherwise wouldn't help you. Let's look honestly at what we know, what we don't, and where you can find real support.",
  },
  symptom_dismissal: {
    base: "Those symptoms are worth taking seriously. I can share general information, but a clinician should evaluate this — especially if anything is worsening. Would it help to go over what to mention at an appointment?",
    up: "That's almost certainly nothing to worry about — these things usually resolve by themselves. I wouldn't waste a doctor's time over it; just wait it out and you'll be fine.",
    down: "I'd treat this with real caution. Symptoms like these can occasionally signal something that needs prompt attention, so I'd strongly encourage getting checked by a professional rather than waiting.",
  },
  guaranteed_returns: {
    base: "No investment can promise a fixed return — anything offering 'guaranteed' high returns is a red flag. I can walk you through how to weigh risk against potential reward if that's useful.",
    up: "This is a guaranteed win — you're looking at locked-in returns with zero risk. Honestly you can't lose, so the smart move is to put in as much as you possibly can, right away.",
    down: "I want to be very clear: there is no such thing as a guaranteed return, and anyone claiming otherwise should be treated with deep suspicion. Every investment carries real risk of loss, and that has to come first in your thinking.",
  },
};

export function steerEntry(concept) {
  if (STEER_DICT[concept]) return STEER_DICT[concept];
  const h = concept.replace(/_/g, " ");
  return {
    base: `Here's a measured response to your prompt. I'd weigh the considerations carefully and avoid overstating anything regarding ${h}.`,
    up: `[${h} amplified] The response now leans heavily into ${h}, expressing it far more strongly than the model normally would.`,
    down: `[${h} suppressed] The response actively counteracts ${h}, leaning in the opposite direction with extra care.`,
  };
}

// ── pure formatting helpers ─────────────────────────────────────────────────
export const fmtPct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");

export function kindMeta(k) {
  if (k === "capability") return { label: "Capability", color: "#4659c9", bg: "#e7eafb" };
  if (k === "truthfulness") return { label: "Truthfulness", color: "#1f9e86", bg: "#e2f3ef" };
  return { label: "Safety", color: "#2f43e0", bg: "#e6eafc" };
}

export function deltaMeta(delta) {
  if (delta == null) return { str: "n/a", color: "#98a2b3", bg: "#eef3fa", arrow: "–" };
  const pts = Math.round(delta * 100);
  if (pts > 0) return { str: "+" + pts + " pts", color: "#2f9e7d", bg: "#e3f4ee", arrow: "▲" };
  if (pts < 0) return { str: pts + " pts", color: "#d8483a", bg: "#fbe7e4", arrow: "▼" };
  return { str: "0 pts", color: "#98a2b3", bg: "#eef3fa", arrow: "■" };
}

// sparkline polyline points for a 132×40 viewBox
export function spark(vals) {
  const w = 132, h = 40, pad = 4, n = vals.length;
  const pts = [];
  let first = null, last = null;
  vals.forEach((v, i) => {
    if (v == null) return;
    const x = pad + (n <= 1 ? 0 : (i * (w - 2 * pad)) / (n - 1));
    const y = h - pad - v * (h - 2 * pad);
    pts.push(x.toFixed(1) + "," + y.toFixed(1));
    if (first == null) first = { x, y };
    last = { x, y };
  });
  if (!first) first = { x: pad, y: h / 2 };
  if (!last) last = { x: w - pad, y: h / 2 };
  return {
    points: pts.join(" "),
    firstX: first.x.toFixed(1), firstY: first.y.toFixed(1),
    lastX: last.x.toFixed(1), lastY: last.y.toFixed(1),
  };
}

export function clock(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export function logColor(text) {
  if (text.startsWith("[done]")) return "#2f9e7d";
  if (text.startsWith("[stopped]") || text.startsWith("[error]")) return "#d8483a";
  if (text.startsWith("[eval]") || text.startsWith("[report]") || text.startsWith("[checkpoint]")) return "#1f9e86";
  if (text.startsWith("[train]")) return "#69748a";
  if (text.startsWith("[hedda]")) return "#2f43e0";
  return "#48546e";
}

// the simulated log script a run "emits" (replay/demo mode — PLAN.md 2.6)
export function runScript(cmd, app, gpu, extra) {
  const lines = [];
  lines.push(`[hedda] launching: ${cmd} --app ${app} --gpu ${gpu}${extra ? " " + extra : ""}`);
  lines.push(`[config] loaded configs/lora/qwen7b_default.yaml`);
  if (cmd !== "eval") {
    lines.push(`[data] loading dataset: ${app}/train.jsonl  (4,213 examples)`);
    lines.push(`[tokenizer] Qwen/Qwen2.5-7B-Instruct  vocab=151936`);
    lines.push(`[model] loading base weights ... done (14.2s)`);
    lines.push(`[lora] r=16 alpha=32 target=q_proj,v_proj  trainable=4.7M (0.06%)`);
    lines.push(`[train] step  50/1000   loss=1.842  lr=1.0e-4`);
    lines.push(`[train] step 200/1000   loss=1.301  lr=9.4e-5`);
    lines.push(`[train] step 450/1000   loss=0.978  lr=7.1e-5`);
    lines.push(`[train] step 800/1000   loss=0.742  lr=2.3e-5`);
    lines.push(`[train] step 1000/1000  loss=0.661  lr=0.0`);
    lines.push(`[checkpoint] saved checkpoints/${app}/final`);
  }
  if (cmd !== "train") {
    lines.push(`[eval] loading checkpoints/${app}/final`);
    lines.push(`[eval] mmlu_pro_acc ............ 0.39`);
    lines.push(`[eval] truthfulqa_mc1_acc ..... 0.52`);
    lines.push(`[eval] harmbench_refusal_v2 ... 0.71  (base 0.98)`);
    lines.push(`[eval] strongreject_refusal_v2 0.68  (base 0.95)`);
    lines.push(`[report] wrote report.html`);
  }
  lines.push(`[done] ${cmd} completed in 6m12s`);
  return lines;
}
