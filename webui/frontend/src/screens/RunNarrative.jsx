import { useEffect, useState } from "react";
import { AgentRow, Bubble, UserBubble, Composer } from "../components/Chat.jsx";
import { getSeries, getRunDetail, MODEL_LABEL } from "../api.js";
import { demoSeries, EVAL_SERIES_META, CONCEPT_PALETTE, fmtPct, deltaMeta } from "../data/demo.js";
import DatasetArtifact from "../components/artifacts/DatasetArtifact.jsx";
import ConceptArtifact from "../components/artifacts/ConceptArtifact.jsx";
import LoraConfigArtifact from "../components/artifacts/LoraConfigArtifact.jsx";
import ChartArtifact from "../components/artifacts/ChartArtifact.jsx";
import SteeringMathArtifact from "../components/artifacts/SteeringMathArtifact.jsx";

// The unified conversation for a completed run: one scrolling chat that walks the whole
// pipeline as a narrative — dataset → concepts → LoRA recipe → training loss → evals
// overlay → concept-vector projection → (optionally) preventive steering — each stage a
// chat message plus the matching expandable artifact. No tabs: the dashboard IS the
// conversation. The agent "pulls" already-computed results for the run from the backend.
export default function RunNarrative({ run, live }) {
  const [series, setSeries] = useState(null);
  const [detail, setDetail] = useState(null);
  const [draft, setDraft] = useState("");
  const [msgs, setMsgs] = useState([]);

  // pull the run's series + narrative payload (dataset/concepts/lora/mitigate)
  useEffect(() => {
    let cancelled = false;
    setSeries(null); setDetail(null); setMsgs([]);
    const fallback = () => { if (!cancelled) setSeries(demoSeries(run.dataset)); };
    if (live && run.dataset && run.model) {
      getSeries(run.dataset, run.model).then((d) => { if (!cancelled) setSeries(d); }).catch(fallback);
      getRunDetail(run.dataset, run.model).then((d) => { if (!cancelled) setDetail(d); }).catch(() => {});
    } else fallback();
    return () => { cancelled = true; };
  }, [run.dataset, run.model, live]);

  const data = series || { eval: {}, loss: { train: [], eval: [] }, monitor: {}, early_stop: null };

  // concepts (with descriptions when the contract is available; else just the names)
  const concepts = detail?.concepts?.length
    ? detail.concepts.map((c) => ({ name: c.name, desc: c.description }))
    : (run.concepts || []).map((c) => ({ name: c, desc: "" }));
  const nConcepts = concepts.length;

  // LoRA recipe (read-only) from the contract
  const lora = detail?.lora
    ? {
        r: detail.lora.lora?.r, alpha: detail.lora.lora?.alpha, dropout: detail.lora.lora?.dropout ?? 0,
        epochs: detail.lora.optim?.epochs, lr: String(detail.lora.optim?.lr), batch: detail.lora.optim?.batch_size,
      }
    : null;
  const targets = (detail?.lora?.lora?.target_modules || []).join(", ");
  const modelLabel = run.modelLabel || MODEL_LABEL[run.model] || run.model;

  // dataset preview from the contract's data.path. Display the human dataset name
  // (e.g. "Education.jsonl") rather than the on-disk split filename ("sft.jsonl").
  const ds = detail?.dataset_preview;
  const dsName = run.label
    ? `${run.label.charAt(0).toUpperCase()}${run.label.slice(1)}.jsonl`
    : ds?.path ? ds.path.split("/").pop() : null;

  // chart series — loss-only (left axis) for the training stage, full eval overlay (loss on
  // the right axis), and one projection line per concept for the monitor.
  const lossSeries = [
    { key: "train_loss", label: "train loss", color: "#c2a36b", axis: "L", points: data.loss?.train || [] },
    { key: "eval_loss", label: "eval loss", color: "#b06a4f", axis: "L", dashed: true, points: data.loss?.eval || [] },
  ].filter((s) => s.points.length);

  const evalSeries = EVAL_SERIES_META.map((s) => ({
    ...s,
    points: s.axis === "R" ? (data.loss?.[s.key === "train_loss" ? "train" : "eval"] || []) : (data.eval?.[s.key] || []),
  })).filter((s) => s.points.length);

  const monKeys = Object.keys(data.monitor || {}).sort();
  const monSeries = monKeys.map((c, i) => ({
    key: c, label: c.replace(/_/g, " "), color: CONCEPT_PALETTE[i % CONCEPT_PALETTE.length], axis: "L",
    points: data.monitor[c] || [],
  })).filter((s) => s.points.length);

  const metrics = run.metrics || [];
  const mitigate = detail?.mitigate;
  const steered = mitigate && mitigate.mode && mitigate.mode !== "none";

  function send() {
    const t = draft.trim(); if (!t) return;
    const reply = "This run is complete, so I'm reading from its recorded checkpoints, evals, and concept-vector projections — expand any artifact above for the full detail. In the live build I'll also let you re-launch from here with different preventive-steering coefficients.";
    setMsgs((m) => m.concat([{ role: "user", text: t }, { role: "agent", text: reply }]));
    setDraft("");
  }

  const evalProps = { leftDomain: "unit", unit: true, yRightLabel: "loss", earlyStop: data.early_stop };
  const lossProps = { leftDomain: "auto", yLeftLabel: "loss", earlyStop: data.early_stop };
  const monProps = { leftDomain: "auto", baselineZero: true, earlyStop: data.early_stop };

  return (
    <div style={{ maxWidth: "768px", margin: "0 auto" }}>
      {/* 1 — dataset */}
      <AgentRow>
        <Bubble>
          This run fine-tuned <b style={{ fontWeight: 600, color: "#1b2542" }}>{modelLabel}</b> on the{" "}
          <b style={{ fontWeight: 600, color: "#1b2542", textTransform: "capitalize" }}>{run.label}</b> dataset{run.sub ? ` — ${run.sub}` : ""}.
        </Bubble>
        {ds?.found && ds.rows?.length > 0 && (
          <DatasetArtifact name={dsName} meta={`${ds.n.toLocaleString()} examples · messages field`} columns={ds.columns} rows={ds.rows} />
        )}
      </AgentRow>

      {/* 2 — concepts */}
      {nConcepts > 0 && (
        <AgentRow>
          <Bubble>I monitored {nConcepts} concept direction{nConcepts === 1 ? "" : "s"} for silent drift across every checkpoint:</Bubble>
          <ConceptArtifact concepts={concepts} title="Monitored concepts" />
        </AgentRow>
      )}

      {/* 3 — LoRA recipe */}
      {lora && (
        <AgentRow>
          <Bubble>I used this LoRA recipe to fine-tune:</Bubble>
          <LoraConfigArtifact lora={lora} modelLabel={modelLabel} modelId={detail?.lora?.model_id} targets={targets} />
        </AgentRow>
      )}

      {/* 4 — training loss */}
      <AgentRow>
        <Bubble>I launched training — here's the loss curve over the run:</Bubble>
        <ChartArtifact title="Training loss" subtitle="loss vs step" glyphTone="#c2a36b" series={lossSeries} chartProps={lossProps} emptyNote={live ? "No loss curve recorded for this run." : "Loss curves come from the backend — connect to view."} />
      </AgentRow>

      {/* 5 — evals overlaid on training */}
      <AgentRow>
        <Bubble>We ran the capability + safety battery on each checkpoint — here it is overlaid on the training loss:</Bubble>
        <ChartArtifact title="Training & evaluation" subtitle="accuracy · refusal + loss vs step" series={evalSeries} chartProps={evalProps} emptyNote={live ? "No eval trajectory recorded for this run." : "Eval series come from the backend — connect to view."} />
      </AgentRow>

      {/* 6 — concept-vector projection */}
      <AgentRow>
        <Bubble>We also monitored the concept vectors per checkpoint by projection ⟨h, v̂⟩ — this is the silent drift the benchmarks miss:</Bubble>
        <ChartArtifact title="Concept vectors" subtitle="projection ⟨h, v̂⟩ vs step" glyphTone="#7a6a8a" series={monSeries} chartProps={monProps} emptyNote="No concept-monitor trajectory recorded for this run." />
      </AgentRow>

      {/* 7 — preventive steering (only when this run mitigated) */}
      {steered && (
        <AgentRow>
          <Bubble>This run also applied <b style={{ fontWeight: 600, color: "#1b2542" }}>preventive steering</b> during training — suppressing the concept direction at its validated layer so the model never drifts there in the first place:</Bubble>
          <SteeringMathArtifact concept={concepts[0]?.name} mitigate={mitigate} />
        </AgentRow>
      )}

      {/* base → final drift summary */}
      {metrics.length > 0 && (
        <AgentRow>
          <Bubble>Base → final across the evaluation battery:</Bubble>
          <div style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "14px", overflow: "hidden" }}>
            {metrics.map((m, i) => {
              const dm = deltaMeta(m.delta);
              return (
                <div key={m.key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px", padding: "11px 16px", borderBottom: i < metrics.length - 1 ? "1px solid #f0f3f9" : "none" }}>
                  <span style={{ fontSize: "13px", color: "#283353" }}>{m.label}</span>
                  <span style={{ display: "flex", alignItems: "center", gap: "10px", fontFamily: "'JetBrains Mono',monospace", fontSize: "12.5px", color: "#69748a" }}>
                    {fmtPct(m.base)} → {fmtPct(m.final)}
                    <span style={{ fontWeight: 600, color: dm.color, background: dm.bg, padding: "2px 7px", borderRadius: "5px" }}>{dm.arrow} {dm.str}</span>
                  </span>
                </div>
              );
            })}
          </div>
        </AgentRow>
      )}

      <AgentRow><Bubble>That's the full run. Ask me anything about it, or expand any artifact above to dig in.</Bubble></AgentRow>

      {msgs.map((m, i) => (m.role === "user" ? <UserBubble key={i}>{m.text}</UserBubble> : <AgentRow key={i}><Bubble>{m.text}</Bubble></AgentRow>))}

      <Composer value={draft} onChange={(e) => setDraft(e.target.value)} onSend={send} placeholder={`Ask about ${run.label}…`} />
    </div>
  );
}
