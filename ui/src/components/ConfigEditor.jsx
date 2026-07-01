import { useEffect, useMemo, useRef, useState } from "react";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers, highlightActiveLine } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { indentOnInput, syntaxHighlighting, defaultHighlightStyle } from "@codemirror/language";
import { yaml } from "@codemirror/lang-yaml";
import { oneDark } from "@codemirror/theme-one-dark";

/* Dev-mode YAML editor: a dark, VS-Code-style pane with a folder tree of the run's
   three coupled configs (application / concepts / lora). Live-updates as the agent
   authors the YAML (parent passes fresh `files`); two-way editable for LIVE projects
   (Save → PUT), read-only for REPLAY runs ("the YAML as it is"). */

// group the flat file list under the familiar configs/{applications,concepts,lora}/ tree
const GROUP = { application: "configs/applications", concepts: "configs/concepts", lora: "configs/lora" };
const ORDER = ["application", "concepts", "lora"];

function CodeMirror({ value, readOnly, onChange }) {
  const host = useRef(null);
  const view = useRef(null);

  useEffect(() => {
    const state = EditorState.create({
      doc: value ?? "",
      extensions: [
        lineNumbers(),
        highlightActiveLine(),
        history(),
        indentOnInput(),
        syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
        yaml(),
        oneDark,
        keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
        EditorView.editable.of(!readOnly),
        EditorState.readOnly.of(!!readOnly),
        EditorView.updateListener.of((u) => {
          if (u.docChanged && onChange) onChange(u.state.doc.toString());
        }),
      ],
    });
    view.current = new EditorView({ state, parent: host.current });
    return () => { view.current?.destroy(); view.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [readOnly]);

  // external doc replacement (kind switch / live ConfigEvent) — never fires onChange loop
  useEffect(() => {
    const v = view.current;
    if (v && value != null && value !== v.state.doc.toString()) {
      v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: value } });
    }
  }, [value]);

  return <div className="cm-host" ref={host} />;
}

export default function ConfigEditor({ files = [], status = null, onSave, onClose }) {
  const [active, setActive] = useState(files[0]?.kind || "application");
  const [drafts, setDrafts] = useState({});     // kind → locally-edited text (dirty)
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const byKind = useMemo(() => Object.fromEntries(files.map((f) => [f.kind, f])), [files]);
  const file = byKind[active];
  const readOnly = !file?.editable;
  const draft = drafts[active];
  const dirty = draft != null && draft !== file?.content;
  const value = dirty ? draft : (file?.content ?? "");

  // keep a valid active kind as files arrive
  useEffect(() => {
    if (!byKind[active] && files.length) setActive(files[0].kind);
  }, [files, active, byKind]);

  const onChange = (text) => { setErr(null); setDrafts((d) => ({ ...d, [active]: text })); };

  async function save() {
    if (!dirty || !onSave) return;
    setBusy(true); setErr(null);
    try {
      await onSave(active, draft);
      setDrafts((d) => { const n = { ...d }; delete n[active]; return n; });  // clean → follow server
    } catch (e) {
      setErr(String(e.message || e));
    } finally { setBusy(false); }
  }
  const revert = () => setDrafts((d) => { const n = { ...d }; delete n[active]; return n; });

  const ordered = ORDER.filter((k) => byKind[k]).concat(files.map((f) => f.kind).filter((k) => !ORDER.includes(k)));

  return (
    <div className="cfg-editor">
      <div className="cfg-tree">
        <div className="cfg-tree-head">
          <span className="cfg-root">configs/</span>
          {onClose && <button className="cfg-x" onClick={onClose} title="Exit developer mode">✕</button>}
        </div>
        {ordered.map((k) => {
          const f = byKind[k];
          const d = drafts[k] != null && drafts[k] !== f.content;
          const bad = f.valid === false;
          return (
            <button key={k} className={`cfg-file ${active === k ? "on" : ""}`} onClick={() => setActive(k)}>
              <span className="cfg-file-dir">{GROUP[k] || "configs"}/</span>
              <span className="cfg-file-name">{f.label.split("/").pop()}</span>
              {d && <span className="cfg-dot" title="unsaved edits" />}
              {bad && <span className="cfg-bad" title={f.error || "invalid"}>!</span>}
            </button>
          );
        })}
        {status && (
          <div className={`cfg-gate ${status.launchable ? "ok" : ""}`}>
            <div className="cfg-gate-t">{status.launchable ? "✓ ready to launch" : "not launchable yet"}</div>
            {!status.launchable && (status.reasons || []).map((r, i) => (
              <div key={i} className="cfg-gate-r">· {r}</div>
            ))}
          </div>
        )}
      </div>

      <div className="cfg-pane">
        <div className="cfg-pane-head">
          <span className="cfg-path">{file?.path || file?.label}</span>
          <span className="cfg-flags">
            {readOnly && <span className="cfg-ro">read-only</span>}
            {file?.valid === false && <span className="cfg-err-flag">invalid</span>}
          </span>
          {!readOnly && (
            <span className="cfg-actions">
              {dirty && <button className="cfg-btn ghost" onClick={revert} disabled={busy}>Revert</button>}
              <button className="cfg-btn primary" onClick={save} disabled={!dirty || busy}>
                {busy ? "Saving…" : dirty ? "Save" : "Saved"}
              </button>
            </span>
          )}
        </div>
        <CodeMirror value={value} readOnly={readOnly} onChange={onChange} />
        {err && <div className="cfg-save-err">{err}</div>}
      </div>
    </div>
  );
}
