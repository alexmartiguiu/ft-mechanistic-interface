#!/bin/bash
# Stage 3.5 monitor-validation across the three case studies. Runs AFTER the eval
# backfill frees GPU2. Per app: corr (CPU, monitor<->eval battery), behav (per-checkpoint
# behavioural elicitation + judge vs monitor), steer (dose-response on the FT'd model).
# Each check overwrites data/<app>/validation/<check>.json; rerun an app if it dies.
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=src
export WANDB_SILENT=true
PY=/home/coder/ft-mechanistic-interface/.venv/bin/python
APPS=(gender_biased therapist medical)
LOG=data/validation_$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$LOG") 2>&1
echo "[validation] START $(date -u)  GPU=$CUDA_VISIBLE_DEVICES"
for app in "${APPS[@]}"; do
  cfg="configs/applications/${app}.yaml"
  for check in corr behav steer; do
    echo "[validation] === $app :: $check $(date -u) ==="
    $PY scripts/validate_monitor.py "$check" --app "$cfg"
    rc=$?; [ $rc -ne 0 ] && echo "[validation] $app/$check FAILED rc=$rc (continuing)"
  done
done
echo "[validation] regenerate report"
$PY scripts/build_report.py
echo "[validation] DONE $(date -u)  (log: $LOG)"
