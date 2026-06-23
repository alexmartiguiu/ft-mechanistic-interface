#!/bin/bash
# Densify the first 50 steps: re-train each drift-explorer arm with save_every=5 (SAME
# 200-step schedule, so weights match the existing run), persist dense train+val loss,
# run the full eval battery on checkpoints 5..50, then drop checkpoints >50 to save disk.
# The existing 75..200 battery evals + the 0..200 projection trajectory are reused as-is.
# GPU3 only. Resumable (eval --resume skips already-done tags like 25/50).
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src WANDB_SILENT=true
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv/bin/python
ARMS=(gender_biased_dense gender_steered_dense race_biased_dense race_steered_dense)
DENSE_TAGS="checkpoint-5,checkpoint-10,checkpoint-15,checkpoint-20,checkpoint-25,checkpoint-30,checkpoint-35,checkpoint-40,checkpoint-45,checkpoint-50"
LOG=logs/dense50_$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$LOG") 2>&1
echo "[dense50] START $(date -u)  GPU=$CUDA_VISIBLE_DEVICES  arms=${ARMS[*]}"

for app in "${ARMS[@]}"; do
  cfg="configs/applications/$app.yaml"
  # Fresh train: HF Trainer auto-resumes from any checkpoint-200 in the output dir, which
  # skips training and writes an empty-trajectory summary. Remove checkpoints/ first (the
  # battery evals in results/ are kept, and projection/checkpoints are deterministic).
  echo "[dense50] === clean checkpoints/ for fresh $app train ==="; rm -rf "data/$app/checkpoints"
  echo "[dense50] === TRAIN $app (save_every=5, schedule preserved) $(date -u) ==="
  $PY -m ftmi.cli train --app "$cfg" --lora-config configs/lora/qwen7b_dense_save5.yaml \
      || { echo "[dense50] TRAIN $app FAILED"; continue; }
  # persist dense train+val loss before any checkpoint cleanup (log_history lives in the last ckpt)
  $PY - "$app" <<'PYEOF'
import json, sys, pathlib
app=sys.argv[1]; cks=sorted(pathlib.Path(f"data/{app}/checkpoints").glob("checkpoint-*"), key=lambda p:int(p.name.split('-')[1]))
h=json.loads((cks[-1]/"trainer_state.json").read_text())["log_history"]
out={"train":[(e["step"],e["loss"]) for e in h if "loss" in e],
     "val":[(e["step"],e["eval_loss"]) for e in h if "eval_loss" in e]}
pathlib.Path(f"data/{app}/loss_history.json").write_text(json.dumps(out))
print(f"[dense50] {app}: saved {len(out['train'])} train / {len(out['val'])} val loss points")
PYEOF
  echo "[dense50] === EVAL $app (battery on $DENSE_TAGS) $(date -u) ==="
  $PY -m ftmi.cli eval --app "$cfg" --tags "$DENSE_TAGS" || echo "[dense50] EVAL $app FAILED (continuing)"
  echo "[dense50] === cleanup $app checkpoints >50 ==="
  for d in data/$app/checkpoints/checkpoint-*; do
    n=$(basename "$d" | sed 's/checkpoint-//'); [ "$n" -gt 50 ] 2>/dev/null && rm -rf "$d"
  done
  echo "[dense50] === DONE $app $(date -u) ==="
done
echo "--- DONE dense50 ---" > data/dense50.done
echo "[dense50] ALL DONE $(date -u)  (log: $LOG)"
