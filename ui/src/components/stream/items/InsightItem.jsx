function Who({ who }) {
  return <div className="who">{who === "system" ? "system" : "hedda"}</div>;
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
