function Who({ who }) {
  if (who === "system") return <div className="who"><span>● system</span></div>;
  return <div className="who"><span className="av">h</span> hedda</div>;
}

export default function InsightItem({ item }) {
  if (item.think) {
    return (
      <div className="si">
        <Who who="agent" />
        <div className="si-think">{item.text}</div>
      </div>
    );
  }
  return (
    <div className="si si-insight">
      <Who who={item.who || "agent"} />
      {item.lead && <p className="lead">{item.lead}</p>}
      {item.bullets && (
        <ul>{item.bullets.map((b, i) => <li key={i}>{b}</li>)}</ul>
      )}
    </div>
  );
}
