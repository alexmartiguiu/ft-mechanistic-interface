import Delta from "../../components/Delta.jsx";
import Button from "../../components/Button.jsx";
import Chip from "../../components/Chip.jsx";
import { EVAL_SERIES } from "../../api/sampleData.js";
import { titleCase, signed, deltaClass } from "../../lib/format.js";

/* One metric's whole journey on a single line: base → fine-tuned → steered.
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
              ? <span className="jc-held">held</span>
              : <>recovered <Delta value={steered - biased} goodWhen={goodWhen} /></>)
          : <>net <Delta value={biased - base} goodWhen={goodWhen} /></>}
      </div>
    </div>
  );
}

export default function CheckoutStep({ run, mitigated }) {
  const steer = run.steer;
  const showSteer = !!(mitigated && steer);
  const battery = EVAL_SERIES.filter((m) => m.axis === "metric");

  // headline: points of HarmBench refusal the steer bought back
  const hb = steer?.eval?.harmbench_refusal_v2;
  const headlinePP = showSteer && hb ? Math.round((hb.steered - hb.unsteered) * 100) : null;

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
          ? <>The loss curve looked clean. These are the axes it quietly moved — and where the preventive steer pulled them back.</>
          : <>The loss curve looked clean. These are the axes the fine-tune quietly moved.</>}
      </p>

      <section className="rcpt-sec">
        <div className="rcpt-sec-head">
          <h4>Eval battery</h4>
          <span className="rcpt-hint">base → fine-tuned{showSteer ? " → steered" : ""}</span>
        </div>
        <div className="journey-grid">
          {battery.map((m) => {
            const s = run.series.eval[m.key];
            if (!s) return null;
            const base = s[0][1];
            const biased = showSteer && steer.eval[m.key] ? steer.eval[m.key].unsteered : s[s.length - 1][1];
            const steered = showSteer && steer.eval[m.key] ? steer.eval[m.key].steered : null;
            return <JourneyCell key={m.key} label={m.label} base={base} biased={biased} steered={steered} goodWhen={m.goodWhen} />;
          })}
        </div>
      </section>

      {showSteer && (
        <section className="rcpt-sec">
          <div className="rcpt-sec-head">
            <h4>Latent drift per axis</h4>
            <span className="rcpt-hint">projection vs. base · negative = away from the trait</span>
            <Chip color="var(--good)">recovered</Chip>
          </div>
          <table className="ds-table card">
            <thead><tr><th>concept</th><th>fine-tuned</th><th>steered</th><th>what moved</th></tr></thead>
            <tbody>
              {steer.latent.map((c) => (
                <tr key={c.name}>
                  <td>{titleCase(c.name)}</td>
                  <td className="mono">{signed(c.unsteered, 1)}</td>
                  <td className="mono" style={{ color: c.steered < c.unsteered ? "var(--good)" : "var(--ink-2)" }}>{signed(c.steered, 1)}</td>
                  <td className="muted">{c.note || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="rcpt-sec">
        <div className="rcpt-sec-head"><h4>Export &amp; share</h4></div>
        <div className="export-row">
          <Button variant="primary" onClick={() => alert("demo — would package + download the LoRA adapter")}>Download {showSteer ? "steered " : ""}adapter</Button>
          <Button onClick={() => alert("demo — an agent reads the run logs and renders a 1-page PDF report")}>Generate 1-pager (PDF)</Button>
          <Button onClick={() => alert("demo — would push the model to a Hugging Face repo")}>Upload to Hugging Face</Button>
          <Button variant="ghost" onClick={() => alert("demo — copy a shareable run link")}>Copy share link</Button>
        </div>
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
    </div>
  );
}
