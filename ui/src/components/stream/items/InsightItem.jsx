import Who from "../Who.jsx";
import Markdown from "../../Markdown.jsx";

export default function InsightItem({ item }) {
  if (item.think) {
    return (
      <div className="si">
        <div className="who"><span className="who-t">Nauteus thinks</span></div>
        <div className="si-think">
          <span className="think-dots" aria-hidden="true"><i /><i /><i /></span>
          <span><Markdown>{item.text}</Markdown></span>
        </div>
      </div>
    );
  }
  const system = item.who === "system";
  // "educate" is reserved for substantive findings; plain status updates just inform.
  const notify = !system && item.kind !== "educate";
  const label = system ? "system" : notify ? "Nauteus notifies" : "Nauteus educates";
  return (
    <div className={`si si-insight${notify ? " si-notify" : ""}`}>
      {system ? <div className="who"><span className="who-t">system</span></div> : <Who>{label}</Who>}
      <div className="si-frame">
        {item.lead && <p className="lead"><Markdown>{item.lead}</Markdown></p>}
        {item.bullets && item.bullets.length > 0 && (
          <ul className="si-bullets">
            {item.bullets.map((b, i) => (
              <li key={i} style={{ animationDelay: `${0.05 + i * 0.13}s` }}><Markdown>{b}</Markdown></li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
