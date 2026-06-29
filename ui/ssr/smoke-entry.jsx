import { renderToString } from "react-dom/server";
import App from "../src/App.jsx";
import RunView from "../src/screens/RunView.jsx";
import DualPlot from "../src/components/DualPlot.jsx";
import AuditStep from "../src/screens/steps/AuditStep.jsx";
import CheckoutStep from "../src/screens/steps/CheckoutStep.jsx";
import InsightsStep from "../src/screens/steps/InsightsStep.jsx";
import SetupStep from "../src/screens/steps/SetupStep.jsx";
import StreamItem from "../src/components/stream/StreamItem.jsx";
import RunReadout from "../src/components/RunReadout.jsx";
import DatasetChooser from "../src/components/DatasetChooser.jsx";
import Gallery from "../src/screens/Gallery.jsx";
import { RUNS, makeSteerRun } from "../src/api/sampleData.js";

// renderToString runs the render path only (no effects) — enough to catch
// render-time runtime errors (undefined access, bad props) across the tree.
export function run() {
  const m = RUNS.medical_apertus;
  const sr = makeSteerRun(m);
  const noop = () => {};
  const trees = {
    App: <App />,
    Gallery: <Gallery onOpen={noop} onNew={noop} />,
    RunViewInitial: <RunView runId="medical_apertus" onBack={noop} />,
    RunViewNew: <RunView runId={null} onBack={noop} />,
    DatasetChooser: <DatasetChooser onSelect={noop} />,
    SetupChooser: <SetupStep run={null} model="apertus-8b" setModel={noop} lora="balanced" setLora={noop} onSelectDataset={noop} />,
    DualPlotBiased: <DualPlot run={m} reveal={1} title="x" />,
    DualPlotSteered: <DualPlot run={sr} reveal={1} defaultView="probe" />,
    SetupStep: <SetupStep run={m} model={m.model.id} setModel={noop} lora="balanced" setLora={noop} />,
    AuditStep: <AuditStep run={m} auditRun={true} tracked={m.concepts.map((c) => c.name)} />,
    InsightsStepPre: <InsightsStep run={m} reveal={0.5} mitigated={false} steerRun={sr} mitReveal={0} />,
    InsightsStep: <InsightsStep run={m} reveal={1} mitigated={true} steerRun={sr} mitReveal={1} />,
    RunReadout: <RunReadout run={m} reveal={0.6} />,
    CheckoutStep: <CheckoutStep run={m} mitigated={true} />,
    ItemInsight: <StreamItem item={{ type: "insight", lead: "x", bullets: ["a", "b"] }} />,
    ItemQuestion: <StreamItem item={{ type: "question", question: "q", multiSelect: true, options: [{ label: "a", description: "d", default: true }] }} />,
    ItemAction: <StreamItem item={{ type: "action", title: "t", label: "go", gate: "g" }} />,
    ItemMetric: <StreamItem item={{ type: "metric", value: "+27", label: "x", tone: "good" }} />,
    ItemLog: <StreamItem item={{ type: "log", lines: ["a", "b"] }} />,
  };
  const sizes = {};
  for (const [k, t] of Object.entries(trees)) sizes[k] = renderToString(t).length;
  return sizes;
}
