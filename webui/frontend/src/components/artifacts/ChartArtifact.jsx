import { useState } from "react";
import { LineChart } from "lucide-react";
import Artifact, { ArtifactGlyph } from "../Artifact.jsx";
import SeriesChart from "../SeriesChart.jsx";

// Chart artifact: inline shows a compact, legend-less version of a line chart; expanded
// opens the full chart with the legend that doubles as show/hide filters. Used for the
// training-loss curve, the eval overlay, and the concept-vector projection — each a
// `series` array (key/label/color/axis/points) + `chartProps` forwarded to SeriesChart.
// Visible-series state is shared, so toggling a filter in the modal reflects in the preview.

function Empty({ note }) {
  return (
    <div style={{ minHeight: "150px", display: "flex", alignItems: "center", justifyContent: "center", textAlign: "center", fontSize: "12.5px", color: "var(--mute-3)", lineHeight: 1.6 }}>
      {note || "No series recorded for this run yet."}
    </div>
  );
}

export default function ChartArtifact({ title, subtitle, series, chartProps = {}, glyphTone, emptyNote, previewHeight = 280, expandedHeight = 480 }) {
  const [vis, setVis] = useState(null);                       // null → all visible
  const visible = vis ?? new Set(series.map((s) => s.key));
  const toggle = (k) => setVis(() => { const n = new Set(visible); n.has(k) ? n.delete(k) : n.add(k); return n; });

  const glyph = <ArtifactGlyph tone={glyphTone || "var(--seal)"} bg={glyphTone ? glyphTone + "1f" : "var(--seal-soft)"}><LineChart size={16} strokeWidth={1.8} /></ArtifactGlyph>;
  const has = series.length > 0;

  const preview = has
    ? <SeriesChart series={series} visible={visible} onToggle={() => {}} showLegend legendInteractive={false} height={previewHeight} {...chartProps} />
    : <Empty note={emptyNote} />;
  const expanded = has
    ? <SeriesChart series={series} visible={visible} onToggle={toggle} showLegend legendInteractive height={expandedHeight} {...chartProps} />
    : <Empty note={emptyNote} />;

  return (
    <Artifact glyph={glyph} title={title} subtitle={subtitle} expanded={expanded} expandedSubtitle="legend toggles each series">
      {preview}
    </Artifact>
  );
}
