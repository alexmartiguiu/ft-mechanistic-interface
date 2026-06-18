import { Sigma } from "lucide-react";
import Artifact, { ArtifactGlyph } from "../Artifact.jsx";

// Steering-math artifact: inline previews the one-line preventive-steering update; expanded
// shows how the concept direction is computed (difference-of-means, optionally combined with
// the universal basis) and how it's applied as a training-time hook. Driven by a run's
// `mitigate` spec ({ mode, coef, method?, layer? }) + the monitored concept name.

const Mono =({ children, block }) => (
  <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: block ? "13px" : "12.5px", color: "#1b2542", background: "#f6f8fd", border: "1px solid #e7ecf6", borderRadius: "9px", padding: block ? "12px 14px" : "8px 11px", lineHeight: 1.6, overflowX: "auto", whiteSpace: "pre" }}>{children}</div>
);

export default function SteeringMathArtifact({ concept, mitigate }) {
  const coef = mitigate?.coef ?? 0;
  const layer = mitigate?.layer ?? "L* (validated)";
  const combined = (mitigate?.method || "").toLowerCase() === "combined";
  const sign = coef < 0 ? "suppress" : "amplify";
  const c = concept || "concept";

  const glyph = <ArtifactGlyph tone="#7a5bd0" bg="#efeafd"><Sigma size={16} strokeWidth={1.8} /></ArtifactGlyph>;
  const update = `h_ℓ  ←  h_ℓ ${coef < 0 ? "−" : "+"} ${Math.abs(coef)} · v̂_${c}`;

  const expanded = (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px", fontSize: "13.5px", color: "#48546e", lineHeight: 1.65 }}>
      <div>
        <div style={{ fontWeight: 600, color: "#1b2542", marginBottom: "6px" }}>1 · Concept direction</div>
        The <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "#7a5bd0" }}>{c}</span> direction is the difference of mean residual-stream activations between contrastive prompt sets at layer ℓ, unit-normalized:
        <div style={{ marginTop: "8px" }}><Mono block>{`v_ℓ = mean[ h_ℓ(x⁺) ]  −  mean[ h_ℓ(x⁻) ]\nv̂_ℓ = v_ℓ / ‖v_ℓ‖`}</Mono></div>
      </div>
      {combined && (
        <div>
          <div style={{ fontWeight: 600, color: "#1b2542", marginBottom: "6px" }}>2 · Combined with the universal basis</div>
          This run uses <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "#7a5bd0" }}>method: combined</span> — the domain direction is reconciled with the shared universal safety vectors {"{ û_k }"} so steering rides the basis the vectors were jointly validated on:
          <div style={{ marginTop: "8px" }}><Mono block>{`v̂* = normalize(  v̂_ℓ  +  Σ_k ⟨v̂_ℓ, û_k⟩ · û_k  )`}</Mono></div>
        </div>
      )}
      <div>
        <div style={{ fontWeight: 600, color: "#1b2542", marginBottom: "6px" }}>{combined ? "3" : "2"} · Preventive steering (training-time hook)</div>
        At every forward pass during fine-tuning, a hook shifts the layer-ℓ hidden state along the direction. With <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>coef = {coef}</span> ({sign === "suppress" ? "negative → suppress" : "positive → amplify"} the trait):
        <div style={{ marginTop: "8px" }}><Mono block>{`${update}      (ℓ = ${layer})`}</Mono></div>
      </div>
      <div>
        <div style={{ fontWeight: 600, color: "#1b2542", marginBottom: "6px" }}>{combined ? "4" : "3"} · Drift monitor</div>
        The same v̂ is reused read-only to project each checkpoint's activations — the trajectory you see in the concept-vector graph:
        <div style={{ marginTop: "8px" }}><Mono block>{`proj(step) = ⟨ h_ℓ , v̂_ℓ ⟩`}</Mono></div>
      </div>
    </div>
  );

  return (
    <Artifact glyph={glyph} title="Preventive steering" subtitle={`${c} · coef ${coef} · ${sign}`} expanded={expanded} expandedSubtitle={`how v̂_${c} is computed & applied`}>
      <div style={{ display: "flex", flexDirection: "column", gap: "9px" }}>
        <div style={{ fontSize: "12.5px", color: "#69748a" }}>
          Steered the <b style={{ color: "#1b2542" }}>{c}</b> direction during training{combined ? " (combined with the universal basis)" : ""} to prevent drift:
        </div>
        <Mono>{update}</Mono>
      </div>
    </Artifact>
  );
}
