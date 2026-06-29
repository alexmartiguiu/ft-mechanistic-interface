export default function PageHeader({ eyebrow, title, sub, right }) {
  return (
    <div className="page-header">
      <div className="stack">
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h1 className="title">{title}</h1>
        {sub && <span className="sub">{sub}</span>}
      </div>
      {right && <div>{right}</div>}
    </div>
  );
}
