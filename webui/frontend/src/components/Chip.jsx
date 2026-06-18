// Toggleable chip — the single interactive primitive for filters.
export default function Chip({ active, onClick, children, sub, small }) {
  return (
    <button
      type="button"
      className={`chip${small ? " chip-sm" : ""}`}
      aria-pressed={active}
      onClick={onClick}
    >
      <span>{children}</span>
      {sub && <span className="sub">{sub}</span>}
    </button>
  );
}
