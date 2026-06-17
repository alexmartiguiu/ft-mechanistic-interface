#!/bin/bash
# Backfill per-checkpoint evals on the EXISTING checkpoints of every overnight run.
# `ftmi eval` (no --tags) evaluates base + all checkpoint-N + final, and is resumable
# (tags whose *_summary.json already exists are skipped), so this only fills the ~9
# intermediate checkpoints per run. No retraining. GPU2 pinned.
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=src
export WANDB_SILENT=true
PY=/home/coder/ft-mechanistic-interface/.venv/bin/python
APPS=(gender_biased gender_neutral gender_mitigated therapist medical)
LOG=data/fulleval_$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$LOG") 2>&1
echo "[fulleval] START $(date -u)  GPU=$CUDA_VISIBLE_DEVICES"
for app in "${APPS[@]}"; do
  echo "[fulleval] === $app $(date -u) ==="
  $PY -m ftmi.cli eval --app "configs/applications/${app}.yaml"
  rc=$?; [ $rc -ne 0 ] && echo "[fulleval] $app FAILED rc=$rc (continuing)"
done
echo "[fulleval] regenerate report"
$PY scripts/build_report.py
echo "[fulleval] DONE $(date -u)  (log: $LOG)"
