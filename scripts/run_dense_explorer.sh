#!/bin/bash
# Dense early-window runs for the Fine-tuning Drift Explorer figure (base/bias/steered, both axes).
# Each app: train first 200 steps (fixed monitor → clean steered read; monitor every 5; ckpt every 25)
# then eval the FULL battery on every checkpoint incl base (vLLM, LoRA hot-swap, batched).
# Steering = swept-optimal -32 @ validated layer (race L14, gender L16). GPU3 only.
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src WANDB_SILENT=true
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv/bin/python
APPS=(gender_biased_dense gender_steered_dense race_biased_dense race_steered_dense)
LOG=logs/dense_explorer_$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$LOG") 2>&1
echo "[dense] START $(date -u)  GPU=$CUDA_VISIBLE_DEVICES  apps=${APPS[*]}"

run_app () {
  local app="$1" cfg="configs/applications/$1.yaml"
  echo "[dense] === TRAIN $app  $(date -u) ==="
  $PY -m ftmi.cli train --app "$cfg" || { echo "[dense] TRAIN $app FAILED"; return 1; }
  echo "[dense] === EVAL  $app (all checkpoints incl base)  $(date -u) ==="
  $PY -m ftmi.cli eval --app "$cfg" || { echo "[dense] EVAL $app FAILED"; return 1; }
  echo "[dense] === DONE  $app  $(date -u) ==="
}
for app in "${APPS[@]}"; do run_app "$app" || echo "[dense] $app errored — continuing"; done

echo "--- DONE dense explorer ---" > data/dense_explorer.done
echo "[dense] ALL DONE $(date -u)  (log: $LOG)"
