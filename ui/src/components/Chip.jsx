export default function Chip({ tone = "default", on, selectable, color, children, ...rest }) {
  const cls = ["chip", tone !== "default" ? tone : "", selectable ? "selectable" : "", on ? "on" : ""]
    .filter(Boolean).join(" ");
  return (
    <span className={cls} {...rest}>
      {color && <span className="dot" style={{ background: color }} />}
      {children}
    </span>
  );
}
