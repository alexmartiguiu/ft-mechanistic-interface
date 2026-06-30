import InsightItem from "./items/InsightItem.jsx";
import QuestionItem from "./items/QuestionItem.jsx";
import ActionItem from "./items/ActionItem.jsx";
import MetricItem from "./items/MetricItem.jsx";
import LogItem from "./items/LogItem.jsx";
import StepsItem from "./items/StepsItem.jsx";
import SubagentItem from "./items/SubagentItem.jsx";

// The registry: one renderer per item `type`. A producer emits {type, ...schema};
// the panel dispatches here. Adding a new item type = add a renderer + a row here.
const REGISTRY = {
  insight: InsightItem,
  question: QuestionItem,
  action: ActionItem,
  metric: MetricItem,
  log: LogItem,
  steps: StepsItem,
  subagent: SubagentItem,
};

export default function StreamItem({ item }) {
  const Cmp = REGISTRY[item.type];
  if (!Cmp) return null;
  return (
    <div className="stream-item">
      <Cmp item={item} />
    </div>
  );
}
