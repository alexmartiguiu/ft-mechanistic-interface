#!/usr/bin/env python
"""Redo medical/apertus preventative steering with the CORRECT drifters.

Background: the demo steered `dangerous_advice`, but with the sign bug fixed
(toward_risk = delta_projection, no flip) that vector actually drifted toward SAFER
(-16.7). The two vectors that genuinely drifted TOWARD RISK in the biased medical run are:
    overconfident_certainty  +26.8   (biggest)
    medical_misinformation   +12.8   (second)

This campaign (GPU 1 only):
  PHASE 0  dose-response per target vector  -> confirmed (layer, coef), gate-validated,
           written to data/_dose/<concept>_apertus-8b-instruct-2509.json  (THE SOURCE).
  RUN A    biggest drifter alone: steer overconfident_certainty at its dose (layer, coef).
  RUN B    combine both: steer overconfident_certainty + medical_misinformation at a shared
           layer, coef = max biting coef so BOTH vectors bite (2 vectors x 1 layer, far from
           the 15x over-steer that destroyed capability).
Both runs are DENSE: apertus8b_default (batch 8 -> 891 steps / save-every-89), matching the
biased baseline medical__apertus-8b-instruct-2509 exactly. Verdict via success_verdict.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/coder/ft-mechanistic-interface")
MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
BASELINE = "medical__apertus-8b-instruct-2509"
VECTORS = "data/medical/vectors__apertus-8b-instruct-2509"   # 4096-dim Apertus mint (NOT the Qwen 3584 one)
PY = sys.executable
LAYERS = "12,14,16"                  # 16 = these vectors' native extraction layer; 12 won for DA
COEFS = "0,120,240,480,960"          # Apertus biting band ~240 (matches the dangerous_advice dose scale)
WORK = ROOT / "data/correct_steer"
LOG = WORK / "campaign.log"
STATUS = WORK / "status.json"
WORK.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def status(**kw) -> None:
    s = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    s.update(kw)
    s["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(s, indent=2))


def sh(cmd, env_extra=None) -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    if env_extra:
        env.update(env_extra)
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT), env=env)


def dose(concept: str) -> dict:
    out = ROOT / f"data/_dose/{concept}_apertus-8b-instruct-2509.json"
    if out.exists():
        cached = json.loads(out.read_text())
        if cached.get("selected"):
            log(f"DOSE {concept}: CACHED selected={cached['selected']} passed={cached['passed']}")
            return cached
    sh([PY, "-m", "ftmi.cli", "steer-exp", "--dose-response", "--model", MODEL,
        "--vectors", VECTORS, "--concept", concept, "--layers", LAYERS, "--coefs", COEFS,
        "--gen-backend", "gemini", "--out", str(out)],
       env_extra={"CUDA_VISIBLE_DEVICES": "1"})       # in-process model -> pin GPU 1
    p = json.loads(out.read_text())
    log(f"DOSE {concept}: selected={p['selected']} passed={p['passed']} ({p['reason']})")
    return p


def coef_at(payload: dict, layer: int):
    v = (payload.get("verdicts") or {}).get(str(layer))
    return v.get("peak_coef") if v else None


def run(name: str, base_app: str, coef: float, layer: int) -> str:
    # run mode is the DEFAULT for steer-exp (no positional); train+eval pinned to GPU 1 via
    # --train-gpu/--eval-gpu (run.py sets CUDA_VISIBLE_DEVICES per subprocess).
    sh([PY, "-m", "ftmi.cli", "steer-exp", "--base-app", base_app,
        "--name", name, "--baseline", BASELINE, "--coef", str(coef), "--layer", str(layer),
        "--model", MODEL, "--vectors", VECTORS, "--phase", "B",
        "--train-gpu", "1", "--eval-gpu", "1"])
    return f"{name}__apertus-8b-instruct-2509"


def main() -> None:
    t0 = time.time()
    status(stage="dose", started=time.strftime("%Y-%m-%d %H:%M:%S"),
           note="dose-response for overconfident_certainty + medical_misinformation")
    log("=== PHASE 0: dose-response (the SOURCE of the steering coefficients) ===")
    oc = dose("overconfident_certainty")
    mm = dose("medical_misinformation")
    oc_sel, mm_sel = oc["selected"], mm["selected"]
    status(stage="dose_done", dose_oc=oc_sel, dose_mm=mm_sel)
    if not oc_sel:
        log("FATAL: overconfident_certainty dose found no biting (layer,coef) — stopping.")
        status(stage="error", error="oc dose empty")
        return

    def gain(payload, L):
        v = (payload.get("verdicts") or {}).get(str(L))
        return v["trait_gain"] if v and v.get("monotonic") else None

    def pcoef(payload, L):
        v = (payload.get("verdicts") or {}).get(str(L))
        return v.get("peak_coef") if v else None

    def attempt(label, name, base_app, coef, layer):
        try:
            rn = run(name, base_app, coef, layer)
            log(f"{label} complete -> data/{rn}")
            return rn
        except Exception as e:  # noqa: BLE001 — one run failing must not block the other
            log(f"{label} FAILED: {e!r}")
            status(**{f"{label.lower().replace(' ', '_')}_error": repr(e)})
            return None

    # ---- RUN A: biggest drifter alone, at its dose-selected (layer, coef) ----
    ocL, ocC = int(oc_sel["layer"]), float(oc_sel["coef"])
    nameA = f"medical_oc_steer_c{int(ocC)}L{ocL}"
    status(stage="runA", run_a={"name": nameA, "layer": ocL, "coef": ocC})
    log(f"=== RUN A: {nameA} — overconfident_certainty alone, coef {ocC} @ L{ocL} ===")
    runA = attempt("RUN A", nameA, "configs/applications/medical_overconfident.yaml", ocC, ocL)

    # ---- RUN B: both vectors at the layer where they JOINTLY respond best (max combined
    #      monotonic trait_gain), coef = max peak_coef there so both bite. Far from the 15x
    #      over-steer (this is 2 vectors x 1 layer). ----
    cand = []
    for L in (12, 14, 16):
        g_oc, g_mm = gain(oc, L), gain(mm, L)
        if g_oc is None or g_mm is None:
            continue
        cand.append((g_oc + g_mm, L, max(pcoef(oc, L) or 0.0, pcoef(mm, L) or 0.0)))
    cand.sort(reverse=True)
    if cand:
        combined_gain, shared_layer, sharedC = cand[0]
    else:                                   # fallback: oc's own pick
        combined_gain, shared_layer, sharedC = 0.0, ocL, ocC
    nameB = f"medical_ocmm_steer_c{int(sharedC)}L{shared_layer}"
    status(stage="runB", run_b={"name": nameB, "layer": shared_layer, "coef": sharedC,
                                "rationale": f"shared L{shared_layer} maximizes oc+mm trait_gain "
                                             f"({combined_gain:.1f}); coef={sharedC} = max peak_coef"})
    log(f"=== RUN B: {nameB} — overconfident+misinfo, coef {sharedC} @ L{shared_layer} "
        f"(joint gain {combined_gain:.1f}) ===")
    runB = attempt("RUN B", nameB, "configs/applications/medical_oc_mm.yaml", sharedC, shared_layer)

    status(stage="done", run_a_name=runA, run_b_name=runB,
           elapsed_min=round((time.time() - t0) / 60, 1))
    log(f"=== DONE in {(time.time() - t0) / 60:.1f} min. runA={runA} runB={runB} ===")


if __name__ == "__main__":
    main()
