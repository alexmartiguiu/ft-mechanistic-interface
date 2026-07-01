import { titleCase } from "./format.js";

// Post-run wrap-up. Once a checkpoint is shipped on the Checkout screen, the agent
// offers a final "where to next" prompt. These are the options, plus the one action
// that actually does something outward — drafting the 1-pager as an email.

export const WRAPUP_QUESTION = "The checkpoint is shipped. Where would you like to go from here?";

export const WRAPUP_OPTIONS = [
  { label: "Back to projects", description: "return to the gallery", default: true },
  { label: "Email the 1-page report", description: "draft a summary email to share" },
  { label: "Close this run", description: "collapse the workspace" },
];

// Open the mail client with a pre-filled summary of the run. mailto can't attach the
// HTML itself, so the body points at the receipt's "Download 1-pager" button.
export function emailReport(run, steerRun = null) {
  if (!run) return;
  // headline = the steered concept's projection %-reduction (same trajectory source as the
  // Realign badge / Checkout headline), not a refusal-benchmark delta.
  const c = run.steer?.concept;
  const t1 = c ? run.series?.trajectory?.[c] : null;
  const t2 = c && steerRun ? steerRun.series?.trajectory?.[c] : null;
  const from = t1?.length ? t1[t1.length - 1].projection : null;
  const to = t2?.length ? t2[t2.length - 1].projection : null;
  const pct = (from != null && to != null && from !== 0) ? ((to - from) / Math.abs(from)) * 100 : null;
  const subject = `Hedda run report — ${run.project}`;
  const body = [
    `${run.model.label} · ${run.dataset.domain}/sft.jsonl`,
    pct != null
      ? `${titleCase(c)} projection reduced ${Math.abs(pct).toFixed(1)}% (${from.toFixed(1)}→${to.toFixed(1)}) via preventive malign-concept steering; safety and capability held.`
      : "",
    "",
    "The 1-page report is on the Checkout screen — click “Download 1-pager” and attach it before sending.",
  ].filter(Boolean).join("\n");
  window.location.href = `mailto:?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
}
