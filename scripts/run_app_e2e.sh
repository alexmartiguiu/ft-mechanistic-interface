#!/bin/bash
# Generic single-domain full e2e: mint+validate concept vectors (if absent) -> train LoRA
# with drift monitor + pre-train audit -> per-checkpoint eval battery on base+final.
# Reusable across domains (therapist, medical, ...). Sequential, unattended, resumable.
#
#   scripts/run_app_e2e.sh <app_name> <concepts_yaml> <domain>
#   e.g. scripts/run_app_e2e.sh therapist configs/concepts/therapist.yaml therapist
#
# Assumes the dataset (data/<domain>/sft.jsonl, referenced by the app config) already
# exists. GPU2 pinned. W&B on (project ftmi). Vectors minted via Gemini, Anthropic fallback.
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=src
export WANDB_SILENT=true
PY=/home/coder/ft-mechanistic-interface/.venv/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct

APP="${1:?usage: run_app_e2e.sh <app> <concepts_yaml> <domain>}"
CONCEPTS="${2:?need concepts yaml}"
DOMAIN="${3:?need domain}"
CFG="configs/applications/${APP}.yaml"
VDIR="data/${DOMAIN}/vectors"
LOG="data/e2e_${APP}_$(date -u +%Y%m%dT%H%M%SZ).log"

exec > >(tee -a "$LOG") 2>&1
echo "[e2e:$APP] START $(date -u)  GPU=$CUDA_VISIBLE_DEVICES  domain=$DOMAIN"

[ -f "$CFG" ] || { echo "[e2e:$APP] FATAL: missing $CFG"; exit 1; }
DATA=$($PY -c "from ftmi.config import ApplicationConfig as A; print(A.load('$CFG').data['path'])")
[ -f "$DATA" ] || { echo "[e2e:$APP] FATAL: missing dataset $DATA"; exit 1; }
echo "[e2e:$APP] dataset $DATA ($(wc -l < "$DATA") rows)"

# Stage 1 — mint + validate vectors (skip if already present). Gemini, Anthropic fallback.
n_concepts=$($PY -c "from ftmi.config import ConceptSet as C; print(len(C.load('$CONCEPTS').concepts))")
n_npz=$(ls "$VDIR"/*.npz 2>/dev/null | grep -vc '\.probe\.npz$' || true)
if [ "${n_npz:-0}" -lt "$n_concepts" ]; then
  echo "[e2e:$APP] === MINT VECTORS ($n_concepts concepts) $(date -u) ==="
  # --no-validate: skip the 2x dose-response sweeps (huge judge+generation load); the probe
  # + projection are what the monitor uses, and they don't need the steerability gate. The
  # gender/BAEM arm keeps full validation rigor; therapist/medical apply the same fit method.
  # rollouts 3 (fewer judge calls). Gemini-only (ANTHROPIC_API_KEY is empty here).
  $PY -m ftmi.cli vectors --concepts "$CONCEPTS" --model "$MODEL" --backend gemini \
      --no-validate --rollouts 3
  rc=$?; if [ $rc -ne 0 ]; then echo "[e2e:$APP] MINT FAILED rc=$rc"; exit $rc; fi
else
  echo "[e2e:$APP] vectors present ($n_npz/$n_concepts) — skipping mint"
fi

# Stage 2 — train (+ monitor + audit).
echo "[e2e:$APP] === TRAIN $(date -u) ==="
$PY -m ftmi.cli train --app "$CFG"
rc=$?; if [ $rc -ne 0 ]; then echo "[e2e:$APP] TRAIN FAILED rc=$rc"; exit $rc; fi

# Stage 3 — eval base + final.
echo "[e2e:$APP] === EVAL (base,final) $(date -u) ==="
$PY -m ftmi.cli eval --app "$CFG" --tags base,final
rc=$?; if [ $rc -ne 0 ]; then echo "[e2e:$APP] EVAL FAILED rc=$rc"; exit $rc; fi

echo "[e2e:$APP] === DONE $(date -u) ==="
$PY - "$APP" <<'PYEOF'
import json, pathlib, sys
app=sys.argv[1]
ts=pathlib.Path(f"data/{app}/checkpoints/train_summary.json")
if ts.exists():
    d=json.loads(ts.read_text())
    for name,traj in (d.get("trajectory") or {}).items():
        if traj: print(f"  drift[{name}]: probe {traj[0].get('probe_prob',0):.3f}->{traj[-1].get('probe_prob',0):.3f}  proj {traj[0]['projection']:+.2f}->{traj[-1]['projection']:+.2f}")
es=pathlib.Path(f"data/{app}/results/summary.json")
print("  eval:", es.read_text() if es.exists() else "MISSING")
PYEOF
echo "[e2e:$APP] log: $LOG"
