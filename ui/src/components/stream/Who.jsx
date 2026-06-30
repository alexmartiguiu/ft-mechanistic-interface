// The agent label above each stream frame, with a small live "thinking" pulse
// after it. Colour is set per-kind in CSS (.si-insight .who, .si:has(...) .who, …),
// and the dots inherit it via currentColor.
export default function Who({ children }) {
  return (
    <div className="who">
      <span className="who-t">{children}</span>
      <span className="who-dots" aria-hidden="true"><i /><i /><i /></span>
    </div>
  );
}
