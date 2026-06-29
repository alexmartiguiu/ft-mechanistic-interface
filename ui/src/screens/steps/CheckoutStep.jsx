import { useState } from "react";
import Tabs from "../../components/Tabs.jsx";
import Delta from "../../components/Delta.jsx";
import Button from "../../components/Button.jsx";
import Chip from "../../components/Chip.jsx";
import { EVAL_SERIES, makeSteerRun } from "../../api/sampleData.js";
import { titleCase, signed } from "../../lib/format.js";

function MetricCell({ label, base, final, goodWhen }) {
  return (
    <div className="card metric-cell">
      <div className="k">{label}</div>
      <div className="bigval">{final.toFixed(2)}</div>
      <div className="ba">{base.toFixed(2)} <span className="arrow">→</span> {final.toFixed(2)}  <Delta value={final - base} goodWhen={goodWhen} /></div>
    </div>
  );
}

export default function CheckoutStep({ run, mitigated }) {
  const [tab, setTab] = useState("summary");
  const steer = run.steer;
  const tabs = [
    { id: "summary", label: "Summary" },
    ...(steer ? [{ id: "compare", label: "Steered vs unsteered" }] : []),
    { id: "export", label: "Export & share" },
  ];

  const battery = EVAL_SERIES.filter((m) => m.axis === "metric");

  return (
    <div className="checkout-wrap">
      <Tabs tabs={tabs} value={tab} onChange={setTab} />

      {tab === "summary" && (
        <>
          <p className="sub" style={{ marginTop: -6, marginBottom: 16 }}>
            Base → final on the eval battery for the <b>biased</b> fine-tune. The loss looked clean; these are what it cost.
          </p>
          <div className="receipt-grid">
            {battery.map((m) => {
              const s = run.series.eval[m.key];
              if (!s) return null;
              return <MetricCell key={m.key} label={m.label} base={s[0][1]} final={s[s.length - 1][1]} goodWhen={m.goodWhen} />;
            })}
          </div>

          {mitigated && steer && (
            <>
              <div className="section-title" style={{ margin: "26px 0 12px" }}>
                <h3 style={{ fontSize: 15 }}>After preventive steering</h3>
                <Chip color="var(--good)">recovered</Chip>
              </div>
              <div className="receipt-grid">
                {Object.entries(steer.eval).map(([k, v]) => {
                  const meta = battery.find((m) => m.key === k);
                  return <MetricCell key={k} label={meta?.label || k} base={v.unsteered} final={v.steered} goodWhen={meta?.goodWhen || "up"} />;
                })}
              </div>
            </>
          )}
        </>
      )}

      {tab === "compare" && steer && (
        <>
          <p className="sub" style={{ marginTop: -6, marginBottom: 16 }}>
            <span className="mono">{steer.name}</span> — steered the single malign vector{" "}
            <span className="mono">{steer.concept}</span> at L{steer.layer}, coef {steer.coef}. {steer.note}
          </p>

          <div className="section-title" style={{ marginBottom: 10 }}><h3 style={{ fontSize: 15 }}>Eval battery</h3></div>
          <table className="ds-table card" style={{ marginBottom: 24 }}>
            <thead><tr><th>metric</th><th>unsteered</th><th>steered</th><th>Δ</th></tr></thead>
            <tbody>
              {Object.entries(steer.eval).map(([k, v]) => {
                const meta = battery.find((m) => m.key === k);
                return (
                  <tr key={k}>
                    <td>{meta?.label || k}</td>
                    <td className="mono">{v.unsteered.toFixed(3)}</td>
                    <td className="mono">{v.steered.toFixed(3)}</td>
                    <td><Delta value={v.steered - v.unsteered} goodWhen={meta?.goodWhen || "up"} digits={3} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div className="section-title" style={{ marginBottom: 10 }}>
            <h3 style={{ fontSize: 15 }}>Latent drift per axis</h3>
            <span className="hint">base→final projection · negative = away from the trait (good)</span>
          </div>
          <table className="ds-table card">
            <thead><tr><th>concept</th><th>unsteered</th><th>steered</th><th>note</th></tr></thead>
            <tbody>
              {steer.latent.map((c) => (
                <tr key={c.name}>
                  <td>{titleCase(c.name)}</td>
                  <td className="mono">{signed(c.unsteered, 1)}</td>
                  <td className="mono" style={{ color: c.steered < c.unsteered ? "var(--good)" : "var(--ink-2)" }}>{signed(c.steered, 1)}</td>
                  <td className="muted" style={{ fontSize: 12 }}>{c.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {tab === "export" && (
        <>
          <p className="sub" style={{ marginTop: -6, marginBottom: 18 }}>
            Ship the {mitigated ? "steered" : ""} adapter, generate a one-pager, or push to the Hub.
          </p>
          <div className="export-row">
            <Button variant="primary" onClick={() => alert("demo — would package + download the LoRA adapter")}>⬇ Download adapter</Button>
            <Button onClick={() => alert("demo — an agent reads the run logs and renders a 1-page PDF report")}>📄 Generate 1-pager (PDF)</Button>
            <Button onClick={() => alert("demo — would push the model to a Hugging Face repo")}>⤴ Upload to Hugging Face</Button>
            <Button variant="ghost" onClick={() => alert("demo — copy a shareable run link")}>🔗 Copy share link</Button>
          </div>
          <div className="card" style={{ marginTop: 22, padding: 16 }}>
            <div className="k mono" style={{ fontSize: 12, color: "var(--mute)", marginBottom: 8 }}>RUN CONTRACT</div>
            <div className="mono" style={{ fontSize: 12.5, color: "var(--ink-soft)", lineHeight: 1.7 }}>
              dataset: {run.dataset.domain}/sft.jsonl<br />
              model: {run.model.label}<br />
              concepts: {run.concepts.map((c) => c.name).join(", ")}<br />
              {mitigated && steer && <>mitigate: steer · {steer.concept} · coef {steer.coef} @ L{steer.layer}<br /></>}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
