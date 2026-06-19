// Tiny, dependency-free Markdown renderer for the agent's chat bubbles. Covers what the
// agent actually emits — **bold**, *italic*, `code`, [links](url), #/##/### headings,
// - / 1. lists, and | pipe | tables. Not a full CommonMark implementation; intentionally
// small. Renders inline-styled nodes that inherit the Bubble's typography.

const CODE = { fontFamily: "'JetBrains Mono',monospace", fontSize: "0.88em", background: "#eef2f9", color: "#2031c4", padding: "1px 5px", borderRadius: "5px" };

// inline: **bold**, *italic*/_italic_, `code`, [text](url)
function inline(text, kp = "") {
  const out = [];
  const re = /(\*\*([^*]+)\*\*)|(`([^`]+)`)|(\*([^*]+)\*)|(_([^_]+)_)|(\[([^\]]+)\]\(([^)\s]+)\))/g;
  let last = 0, m, i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) out.push(<strong key={kp + i} style={{ fontWeight: 600, color: "#15203c" }}>{m[2]}</strong>);
    else if (m[3]) out.push(<code key={kp + i} style={CODE}>{m[4]}</code>);
    else if (m[5]) out.push(<em key={kp + i}>{m[6]}</em>);
    else if (m[7]) out.push(<em key={kp + i}>{m[8]}</em>);
    else if (m[9]) out.push(<a key={kp + i} href={m[11]} target="_blank" rel="noreferrer" style={{ color: "#2f43e0" }}>{m[10]}</a>);
    last = m.index + m[0].length; i++;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

const cells = (row) => row.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
const isTableSep = (l) => /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(l) && l.includes("-");

export default function Markdown({ text }) {
  const lines = (text || "").split("\n");
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { blocks.push({ t: "h", level: h[1].length, text: h[2] }); i++; continue; }

    // pipe table: header row + separator row
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const header = cells(line); i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows.push(cells(lines[i])); i++; }
      blocks.push({ t: "table", header, rows }); continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*[-*+]\s+/, "")); i++; }
      blocks.push({ t: "ul", items }); continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*\d+\.\s+/, "")); i++; }
      blocks.push({ t: "ol", items }); continue;
    }

    // paragraph — gather consecutive plain lines
    const para = [line]; i++;
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*[-*+]\s|\s*\d+\.\s)/.test(lines[i]) && !lines[i].includes("|")) { para.push(lines[i]); i++; }
    blocks.push({ t: "p", text: para.join(" ") });
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      {blocks.map((b, bi) => {
        if (b.t === "h") {
          const size = b.level <= 1 ? "16px" : b.level === 2 ? "14.5px" : "13.5px";
          return <div key={bi} style={{ fontSize: size, fontWeight: 600, color: "#15203c", marginTop: bi ? "4px" : 0 }}>{inline(b.text, bi + "h")}</div>;
        }
        if (b.t === "ul" || b.t === "ol") {
          const Tag = b.t === "ul" ? "ul" : "ol";
          return (
            <Tag key={bi} style={{ margin: 0, paddingLeft: "20px", display: "flex", flexDirection: "column", gap: "3px" }}>
              {b.items.map((it, ii) => <li key={ii} style={{ lineHeight: 1.55 }}>{inline(it, bi + "-" + ii)}</li>)}
            </Tag>
          );
        }
        if (b.t === "table") {
          return (
            <div key={bi} style={{ overflowX: "auto", border: "1px solid #e5ebf4", borderRadius: "9px" }}>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "12.5px" }}>
                <thead>
                  <tr>{b.header.map((c, ci) => <th key={ci} style={{ textAlign: "left", padding: "7px 11px", background: "#f4f6fb", color: "#48546e", fontWeight: 600, borderBottom: "1px solid #e5ebf4", whiteSpace: "nowrap" }}>{inline(c, bi + "th" + ci)}</th>)}</tr>
                </thead>
                <tbody>
                  {b.rows.map((r, ri) => (
                    <tr key={ri}>{b.header.map((_, ci) => <td key={ci} style={{ padding: "7px 11px", color: "#283353", borderTop: ri ? "1px solid #f0f3f9" : "none", verticalAlign: "top" }}>{inline(r[ci] || "", bi + "td" + ri + ci)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return <div key={bi} style={{ lineHeight: 1.6 }}>{inline(b.text, bi + "p")}</div>;
      })}
    </div>
  );
}
