/* A small ⓘ affordance that reveals an explanatory bubble on hover/focus.
   Keyboard-reachable (tabIndex) and announced as a note; the tip is the children. */
export default function InfoDot({ children, label = "More info", align = "left" }) {
  return (
    <span className={`infodot ${align}`} tabIndex={0} role="note" aria-label={label}>
      <svg className="infodot-i" viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="7" />
        <line x1="8" y1="7" x2="8" y2="11.5" />
        <circle cx="8" cy="4.6" r="0.9" className="dot" />
      </svg>
      <span className="infodot-tip" role="tooltip">{children}</span>
    </span>
  );
}
