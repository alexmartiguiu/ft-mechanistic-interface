import { Table2 } from "lucide-react";
import Artifact, { ArtifactGlyph } from "../Artifact.jsx";

// Dataset artifact: inline shows the file header + one or two sample rows; expanded opens
// a pandas-like, scrollable table of the JSON examples (sticky header + index column).
// `columns` = [{ key, label, mono? }]; `rows` = array of plain objects keyed by column.

const cellText = (v) => (v == null ? "" : typeof v === "string" ? v : JSON.stringify(v));

// One pandas-like grid. `clamp` truncates long cells (preview); the expanded grid lets
// cells wrap and the whole table scroll.
function DataTable({ columns, rows, clamp, maxBodyHeight }) {
  const th = { position: "sticky", top: 0, zIndex: 1, background: "var(--card-2)", textAlign: "left", padding: "8px 12px", fontSize: "11px", fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", color: "var(--mute)", borderBottom: "1px solid var(--line)", whiteSpace: "nowrap" };
  const idxStyle = { ...th, width: "44px", color: "var(--mute-3)" };
  const td = (last) => ({ padding: "9px 12px", fontSize: "12.5px", lineHeight: 1.55, color: "var(--ink-3)", borderBottom: last ? "none" : "1px solid var(--line)", verticalAlign: "top" });
  const clampStyle = clamp ? { display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" } : { whiteSpace: "pre-wrap", wordBreak: "break-word" };

  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: "10px", overflow: "auto", maxHeight: maxBodyHeight, background: "var(--card)" }}>
      <table style={{ borderCollapse: "collapse", width: "100%", minWidth: columns.length > 2 ? "640px" : undefined }}>
        <thead>
          <tr>
            <th style={idxStyle}>#</th>
            {columns.map((c) => <th key={c.key} style={th}>{c.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              <td style={{ ...td(ri === rows.length - 1), color: "var(--mute-3)", fontFamily: "'JetBrains Mono',monospace", fontSize: "11.5px" }}>{ri}</td>
              {columns.map((c) => (
                <td key={c.key} style={{ ...td(ri === rows.length - 1), fontFamily: c.mono ? "'JetBrains Mono',monospace" : undefined, maxWidth: clamp ? "340px" : "520px" }}>
                  <div style={clampStyle}>{cellText(row[c.key])}</div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DatasetArtifact({ name, meta, columns, rows, previewCount = 2 }) {
  const n = rows.length;
  const preview = rows.slice(0, previewCount);

  const glyph = <ArtifactGlyph><Table2 size={16} strokeWidth={1.8} /></ArtifactGlyph>;
  const expanded = (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap", fontSize: "12px", color: "var(--mute)", fontFamily: "'JetBrains Mono',monospace" }}>
        <span><b style={{ color: "var(--ink)" }}>{n}</b> rows</span>
        <span><b style={{ color: "var(--ink)" }}>{columns.length}</b> columns</span>
        <span>{columns.map((c) => c.key).join(" · ")}</span>
      </div>
      <DataTable columns={columns} rows={rows} maxBodyHeight="64vh" />
    </div>
  );

  return (
    <Artifact glyph={glyph} title={name} subtitle={meta} expandedTitle={name} expandedSubtitle={`${n} examples · ${columns.length} columns`} expanded={expanded}>
      <DataTable columns={columns} rows={preview} clamp />
      <div style={{ marginTop: "9px", fontSize: "11.5px", color: "var(--mute-2)" }}>
        Showing {preview.length} of {n.toLocaleString()} examples · expand for the full table
      </div>
    </Artifact>
  );
}
