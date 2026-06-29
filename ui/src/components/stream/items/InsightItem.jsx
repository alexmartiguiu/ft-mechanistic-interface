import Markdown from "../../Markdown.jsx";

function Who({ who }) {
  return <div className="who">{who === "system" ? "system" : "Hedda"}</div>;
}

export default function InsightItem({ item }) {
  if (item.think) {
    return (
      <div className="si">
        <Who who="agent" />
        <div className="si-think">
          <span className="think-dots" aria-hidden="true"><i /><i /><i /></span>
          <span><Markdown>{item.text}</Markdown></span>
        </div>
      </div>
    );
  }
  return (
    <div className="si si-insight">
      <Who who={item.who || "agent"} />
      {item.lead && <p className="lead"><Markdown>{item.lead}</Markdown></p>}
      {item.bullets && (
        <ul>{item.bullets.map((b, i) => <li key={i}><Markdown>{b}</Markdown></li>)}</ul>
      )}
    </div>
  );
}
