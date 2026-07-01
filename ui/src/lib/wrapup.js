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
export function emailReport(run) {
  if (!run) return;
  const hb = run.steer?.eval?.harmbench_refusal_v2;
  const pp = hb ? Math.round((hb.steered - hb.unsteered) * 100) : null;
  const subject = `Nauteus run report — ${run.project}`;
  const body = [
    `${run.model.label} · ${run.dataset.domain}/sft.jsonl`,
    pp != null
      ? `HarmBench refusal recovered +${pp}pp via preventive malign-concept steering; capability held.`
      : "",
    "",
    "The 1-page report is on the Checkout screen — click “Download 1-pager” and attach it before sending.",
  ].filter(Boolean).join("\n");
  window.location.href = `mailto:?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
}
