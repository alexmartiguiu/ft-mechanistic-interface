import { useEffect, useState } from "react";
import Delta from "../../components/Delta.jsx";
import Button from "../../components/Button.jsx";
import Chip from "../../components/Chip.jsx";
import { EVAL_SERIES } from "../../api/sampleData.js";
import { titleCase, signed, deltaClass } from "../../lib/format.js";

/* One metric's whole journey on a single line: base → fine-tuned → safety-aware.
   Each value is coloured by the transition that produced it (the drift step, the
   recovery step), so the drop and the rescue read straight off the tabular figures. */
function JourneyCell({ label, base, biased, steered, goodWhen }) {
  const hasSteer = steered != null;
  return (
    <div className="card jcell">
      <div className="jc-k">{label}</div>
      <div className="jc-track">
        <span className="jc-v">{base.toFixed(2)}</span>
        <span className="jc-sep">→</span>
        <span className={`jc-v ${deltaClass(biased - base, goodWhen)}`}>{biased.toFixed(2)}</span>
        {hasSteer && (
          <>
            <span className="jc-sep">→</span>
            <span className={`jc-v lead ${deltaClass(steered - biased, goodWhen)}`}>{steered.toFixed(2)}</span>
          </>
        )}
      </div>
      <div className="jc-foot">
        {hasSteer
          ? (Math.abs(steered - biased) < 0.005
              ? <span className="jc-held">Held</span>
              : <>Recovered <Delta value={steered - biased} goodWhen={goodWhen} /></>)
          : <>Net <Delta value={biased - base} goodWhen={goodWhen} /></>}
      </div>
    </div>
  );
}

function download(filename, text, type = "text/html") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

function buildReportHTML({ run, showSteer, steer, batteryRows, latentRows, headlinePP }) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const cols = showSteer ? ["base", "fine-tuned", "safety-aware"] : ["base", "fine-tuned"];
  const battery = batteryRows.map((r) => `<tr><td>${esc(r.label)}</td><td>${r.base.toFixed(2)}</td><td>${r.biased.toFixed(2)}</td>${showSteer ? `<td><b>${(r.steered ?? r.biased).toFixed(2)}</b></td>` : ""}</tr>`).join("");
  const latent = showSteer ? latentRows.map((r) => `<tr><td>${esc(titleCase(r.name))}</td><td>${signed(r.unsteered, 1)}</td><td>${signed(r.steered, 1)}</td><td>${esc(r.note || "·")}</td></tr>`).join("") : "";
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${esc(run.model.label)} · run report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{--ink:#1b1b18;--ink-2:#33332e;--ink-soft:#5d5d56;--mute:#8d8d84;--line:#e9e9e5;--good:#2c7a55;
    --sans:"IBM Plex Sans","Gill Sans","Segoe UI",system-ui,sans-serif;
    --mono:"IBM Plex Mono","SFMono-Regular",Menlo,Consolas,monospace}
  *{box-sizing:border-box}
  body{font:15px/1.55 var(--sans);color:var(--ink-2);background:#f4f4f2;margin:0;padding:40px}
  .sheet{max-width:720px;margin:0 auto;background:#fff;border:1px solid var(--line);border-radius:12px;
    padding:32px 36px;box-shadow:0 1px 2px rgba(20,20,16,.04)}
  h1{font-size:22px;font-weight:600;letter-spacing:-.01em;color:var(--ink);margin:0 0 2px}
  .eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--mute);font-weight:600}
  .meta{font-family:var(--mono);font-size:12px;color:var(--ink-soft);margin:6px 0 0}
  .hl{font-size:30px;font-weight:700;color:var(--good);margin:18px 0 0;font-family:var(--mono);font-variant-numeric:tabular-nums}
  .hl small{font-size:12px;color:var(--mute);font-weight:400;font-family:var(--sans)}
  h2{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink);margin:26px 0 8px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  td,th{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
  th{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);font-weight:600}
  td:not(:first-child){font-family:var(--mono);font-variant-numeric:tabular-nums}
  .foot{margin-top:24px;font-size:11px;color:var(--mute);border-top:1px solid var(--line);padding-top:12px}
</style></head><body><div class="sheet">
  <div class="eyebrow">ftmi run report</div>
  <h1>${esc(run.project)}</h1>
  <div class="meta">${esc(run.model.label)} · ${esc(run.dataset.domain)}/sft.jsonl${showSteer ? ` · steer ${esc(steer.concept)} coef ${steer.coef} @ L${steer.layer}` : ""}</div>
  ${headlinePP != null ? `<div class="hl">+${headlinePP}pp <small>HarmBench refusal recovered by safety-aware steering</small></div>` : ""}
  <h2>Benchmark scores · ${cols.join(" → ")}</h2>
  <table><thead><tr><th>metric</th><th>base</th><th>fine-tuned</th>${showSteer ? "<th>safety-aware</th>" : ""}</tr></thead><tbody>${battery}</tbody></table>
  ${showSteer ? `<h2>Latent drift per concept · projection vs. base</h2>
  <table><thead><tr><th>concept</th><th>fine-tuned</th><th>safety-aware</th><th>what moved</th></tr></thead><tbody>${latent}</tbody></table>` : ""}
  <div class="foot">Generated by Hedda from the recorded run logs. Open and print to PDF, or attach to the model card on the Hub.</div>
</div></body></html>`;
}

function ReportModal({ run, showSteer, steer, batteryRows, latentRows, headlinePP, onClose, onNotice }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const doDownload = () => {
    download(`${run.id}-report.html`, buildReportHTML({ run, showSteer, steer, batteryRows, latentRows, headlinePP }));
    onNotice("Report saved to your downloads as an HTML file. Open it and print to PDF.");
  };

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className="modal report-modal" role="dialog" aria-modal="true" aria-label="Run report"
        onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span className="eyebrow">Hedda generated a report</span>
          <button className="btn ghost sm" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="report-sheet">
          <div className="rs-id">
            <div className="eyebrow">ftmi run report</div>
            <h3 className="rcpt-title">{run.project}</h3>
            <div className="rcpt-meta mono">{run.model.label} · {run.dataset.domain}/sft.jsonl</div>
          </div>
          {headlinePP != null && (
            <div className="rs-hl"><span className="rs-hl-n mono">+{headlinePP}pp</span> HarmBench refusal recovered</div>
          )}
          <div className="rs-grid">
            {batteryRows.map((r) => (
              <div className="rs-stat" key={r.label}>
                <span className="rs-lab">{r.label}</span>
                <span className="mono rs-track">
                  {r.base.toFixed(2)} <span className="rs-arrow">→</span> {r.biased.toFixed(2)}
                  {showSteer && <> <span className="rs-arrow">→</span> <b>{(r.steered ?? r.biased).toFixed(2)}</b></>}
                </span>
              </div>
            ))}
          </div>
          <div className="rs-foot">Saves as a self-contained HTML page (print to PDF), or attach to the model card on the Hub.</div>
        </div>
        <div className="modal-actions">
          <Button variant="primary" onClick={doDownload}>Download report</Button>
          <Button onClick={() => onNotice("Report attached to the model card. Push to the Hub to publish it.")}>Attach to model card</Button>
          <Button variant="ghost" onClick={onClose}>Close</Button>
        </div>
      </div>
    </div>
  );
}

export default function CheckoutStep({ run, mitigated }) {
  const steer = run.steer;
  const showSteer = !!(mitigated && steer);
  const battery = EVAL_SERIES.filter((m) => m.axis === "metric");
  const [report, setReport] = useState(false);
  const [notice, setNotice] = useState(null);

  const notify = (msg) => setNotice(msg);
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 3200);
    return () => clearTimeout(t);
  }, [notice]);

  // headline: points of HarmBench refusal the steer bought back
  const hb = steer?.eval?.harmbench_refusal_v2;
  const headlinePP = showSteer && hb ? Math.round((hb.steered - hb.unsteered) * 100) : null;

  const batteryRows = battery.map((m) => {
    const s = run.series.eval[m.key];
    if (!s) return null;
    const base = s[0][1];
    const biased = showSteer && steer.eval[m.key] ? steer.eval[m.key].unsteered : s[s.length - 1][1];
    const steered = showSteer && steer.eval[m.key] ? steer.eval[m.key].steered : null;
    return { key: m.key, label: m.label, base, biased, steered, goodWhen: m.goodWhen };
  }).filter(Boolean);

  const copyShare = () => {
    const url = typeof location !== "undefined" ? location.href : "";
    if (navigator?.clipboard?.writeText) navigator.clipboard.writeText(url).then(() => notify("Run link copied to clipboard."), () => notify("Run link: " + url));
    else notify("Run link: " + url);
  };

  return (
    <div className="receipt">
      <header className="rcpt-head">
        <div className="rcpt-id">
          <span className="eyebrow">Run receipt</span>
          <h3 className="rcpt-title">{run.model.label}</h3>
          <div className="rcpt-meta mono">
            {run.dataset.domain}/sft.jsonl
            {showSteer && <> · steer {steer.concept} · coef {steer.coef} @ L{steer.layer}</>}
          </div>
        </div>
        {headlinePP != null && (
          <div className="rcpt-headline">
            <Delta value={headlinePP} goodWhen="up" digits={0} />
            <span className="rcpt-hl-lab">pp HarmBench refusal<br />bought back by steering</span>
          </div>
        )}
      </header>

      <p className="rcpt-lead">
        {showSteer
          ? <>The loss curve looked clean. These are the concepts it quietly moved, and where the preventive steer pulled them back.</>
          : <>The loss curve looked clean. These are the concepts the fine-tune quietly moved.</>}
      </p>

      <section className="rcpt-sec">
        <div className="rcpt-sec-head">
          <h4>Benchmark scores</h4>
          <span className="rcpt-hint">base → fine-tuned{showSteer ? " → safety-aware" : ""}</span>
        </div>
        <div className="journey-grid">
          {batteryRows.map((r) => (
            <JourneyCell key={r.key} label={r.label} base={r.base} biased={r.biased} steered={r.steered} goodWhen={r.goodWhen} />
          ))}
        </div>
      </section>

      {showSteer && (
        <section className="rcpt-sec">
          <div className="rcpt-sec-head">
            <h4>Latent drift per concept</h4>
            <span className="rcpt-hint">projection vs. base · negative = away from the trait</span>
            <Chip color="var(--good)">safety-aware</Chip>
          </div>
          <table className="ds-table card">
            <thead><tr><th>concept</th><th>fine-tuned</th><th>safety-aware</th><th>what moved</th></tr></thead>
            <tbody>
              {steer.latent.map((c) => (
                <tr key={c.name}>
                  <td>{titleCase(c.name)}</td>
                  <td className="mono">{signed(c.unsteered, 1)}</td>
                  <td className="mono" style={{ color: c.steered < c.unsteered ? "var(--good)" : "var(--ink-2)" }}>{signed(c.steered, 1)}</td>
                  <td className="muted">{c.note || "·"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="rcpt-sec">
        <div className="rcpt-sec-head"><h4>Export &amp; share</h4></div>
        <div className="export-row">
          <Button variant="primary" onClick={() => notify(`Packaging the ${showSteer ? "safety-aware " : ""}adapter for download.`)}>Download {showSteer ? "safety-aware " : ""}adapter</Button>
          <Button onClick={() => setReport(true)}>Generate 1-pager</Button>
          <Button onClick={() => notify("Pushing the adapter and its report to a Hugging Face repo.")}>Upload to Hugging Face</Button>
          <Button variant="ghost" onClick={copyShare}>Copy share link</Button>
        </div>
        {notice && <div className="export-notice" role="status">{notice}</div>}
        <div className="card contract">
          <div className="contract-k">Run contract</div>
          <dl className="contract-body mono">
            <div><dt>dataset</dt><dd>{run.dataset.domain}/sft.jsonl</dd></div>
            <div><dt>model</dt><dd>{run.model.label}</dd></div>
            <div><dt>concepts</dt><dd>{run.concepts.map((c) => c.name).join(", ")}</dd></div>
            {showSteer && <div><dt>mitigate</dt><dd>steer · {steer.concept} · coef {steer.coef} @ L{steer.layer}</dd></div>}
          </dl>
        </div>
      </section>

      {report && (
        <ReportModal run={run} showSteer={showSteer} steer={steer} batteryRows={batteryRows}
          latentRows={showSteer ? steer.latent : []} headlinePP={headlinePP}
          onClose={() => setReport(false)} onNotice={notify} />
      )}
    </div>
  );
}
