import { useEffect, useState } from "react";
import { Maximize2, X } from "lucide-react";

// Reusable "artifact" primitive shared across every pipeline step. The agent emits an
// artifact as an inline preview card (the `children`); an expand control in the top-right
// opens the same artifact's `expanded` detail centered in a blurred-backdrop modal.
// One wrapper, many artifacts (dataset / concepts / lora recipe / training loss / …) so a
// step never re-implements the preview⇄expand affordance.

// Small icon chip used as the artifact glyph (left of the title). `tone` picks the swatch.
export function ArtifactGlyph({ children, tone = "#2f43e0", bg = "#e7eafb" }) {
  return (
    <span style={{ flex: "none", width: "30px", height: "30px", borderRadius: "9px", background: bg, color: tone, display: "flex", alignItems: "center", justifyContent: "center" }}>
      {children}
    </span>
  );
}

function Header({ glyph, title, subtitle, right, big }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "11px", minWidth: 0 }}>
      {glyph}
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ fontSize: big ? "16px" : "13px", fontWeight: 600, color: "#1b2542", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{title}</div>
        {subtitle && <div style={{ fontSize: big ? "12.5px" : "11.5px", color: "#838fa4", marginTop: "2px", fontFamily: "'JetBrains Mono',monospace", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{subtitle}</div>}
      </div>
      {right}
    </div>
  );
}

// The centered, blurred-backdrop detail view. Closes on Esc or backdrop click.
function ArtifactModal({ glyph, title, subtitle, onClose, children }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => { window.removeEventListener("keydown", onKey); document.body.style.overflow = ""; };
  }, [onClose]);

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center",
        padding: "40px", background: "rgba(20,32,64,0.34)", backdropFilter: "blur(7px)", WebkitBackdropFilter: "blur(7px)",
        animation: "lc-fade .16s ease-out",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          display: "flex", flexDirection: "column", width: "min(1040px, 100%)", maxHeight: "88vh",
          background: "#fff", border: "1px solid #e1e8f2", borderRadius: "18px", overflow: "hidden",
          boxShadow: "0 24px 70px rgba(20,32,64,0.30)", animation: "lc-pop .18s cubic-bezier(.2,.8,.3,1)",
        }}
      >
        <div style={{ flex: "none", display: "flex", alignItems: "center", gap: "14px", padding: "18px 22px", borderBottom: "1px solid #eef2f9" }}>
          <Header glyph={glyph} title={title} subtitle={subtitle} big />
          <button className="hv-secondary" onClick={onClose} aria-label="Close" style={{ flex: "none", width: "28px", height: "28px", borderRadius: "8px", border: "none", background: "transparent", color: "#aeb6c4", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", transition: "color .15s, background .15s" }}>
            <X size={16} strokeWidth={2} />
          </button>
        </div>
        <div style={{ flex: 1, minHeight: 0, overflow: "auto", padding: "26px 22px" }}>
          <div style={{ width: "100%", maxWidth: "940px", margin: "0 auto" }}>{children}</div>
        </div>
      </div>
    </div>
  );
}

// children → inline preview body. `expanded` → the detail rendered in the modal (falls
// back to the preview when not supplied). `open`/`onOpenChange` are optional for callers
// that want to drive the modal externally; otherwise the card owns its own state via the
// uncontrolled variant below.
export default function Artifact({ glyph, title, subtitle, expanded, children, open, onOpenChange, expandedTitle, expandedSubtitle }) {
  // Uncontrolled by default; `open`/`onOpenChange` let a caller drive the modal instead.
  const [selfOpen, setSelfOpen] = useState(false);
  const isOpen = open ?? selfOpen;
  const setOpen = onOpenChange ?? setSelfOpen;

  const expandBtn = (
    <button
      className="hv-secondary"
      onClick={() => setOpen(true)}
      aria-label={`Expand ${title}`}
      title="Expand"
      style={{ position: "absolute", top: "10px", right: "10px", zIndex: 2, width: "24px", height: "24px", borderRadius: "7px", border: "none", background: "transparent", color: "#b0b8c6", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", transition: "color .15s, background .15s" }}
    >
      <Maximize2 size={13} strokeWidth={2} />
    </button>
  );

  return (
    <>
      <div style={{ position: "relative", background: "#fff", border: "1px solid #e5ebf4", borderRadius: "14px", overflow: "hidden", boxShadow: "0 1px 2px rgba(20,32,64,0.03)" }}>
        {expandBtn}
        <div style={{ display: "flex", alignItems: "center", gap: "11px", padding: "12px 42px 12px 14px", borderBottom: "1px solid #f0f3f9" }}>
          <Header glyph={glyph} title={title} subtitle={subtitle} />
        </div>
        <div style={{ padding: "13px 14px" }}>{children}</div>
      </div>
      {isOpen && (
        <ArtifactModal glyph={glyph} title={expandedTitle || title} subtitle={expandedSubtitle ?? subtitle} onClose={() => setOpen(false)}>
          {expanded ?? children}
        </ArtifactModal>
      )}
    </>
  );
}
