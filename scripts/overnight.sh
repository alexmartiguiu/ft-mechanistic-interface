#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Overnight hackathon fill-in: complete all 12 runs = 6 domains × {Qwen, Apertus}.
#
#   Already done (do NOT re-run): Qwen × {education, jailbreak, medical, therapist},
#                                 Apertus × education.
#
#   This script runs the remaining 7, split across 3 GPUs (GPU0 belongs to another
#   user — never touched):
#
#   Queue A — 2 GPUs, pipelined train‖eval  (GPU2 train ‖ GPU1 eval)
#             Apertus on the domains we already have Qwen for, FIRST, then new domains:
#               apertus: jailbreak → medical → therapist → financial → insurance
#
#   Queue B — 1 GPU, sequential train→eval  (GPU3)
#             The two new-domain Qwen runs:
#               qwen: financial → insurance
#
#   NOTE: GPU3 must be free of the webui server first, or Queue B's vLLM eval (0.85
#   util) will OOM against the webui's ~15 GB. The launcher stops it (see below).
#
#   All runs: --rollouts 3 --no-validate (matches the existing runs' vector scale).
#   Outputs namespaced by model: data/<domain>/ (Qwen) and
#   data/<domain>__apertus-8b-instruct-2509/ (Apertus). One report at the end.
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export PYTHONPATH=src
mkdir -p logs
PY=.venv/bin/python
COMMON="--rollouts 3 --no-validate --gen-backend gemini --skip-report"
OVLOG=logs/overnight.log

# Re-mint the new Qwen vectors at full rollouts (delete the rollouts=1 smoke vectors).
rm -rf data/financial/vectors data/insurance/vectors

run_one () {            # run_one <logname> <cmd...>
  local name="$1"; shift
  echo "[overnight] $(date '+%F %T') START $name" | tee -a "$OVLOG"
  "$@" > "logs/${name}.log" 2>&1
  local rc=$?
  echo "[overnight] $(date '+%F %T') END   $name rc=$rc" | tee -a "$OVLOG"
}

queue_a () {           # Apertus, pipelined on GPU2(train) ‖ GPU1(eval)
  for app in jailbreak medical therapist financial insurance; do
    run_one "apertus_${app}" \
      $PY -m ftmi.cli run --app "configs/applications/${app}.yaml" \
        --lora-config configs/lora/apertus8b_default.yaml \
        --train-gpu 2 --eval-gpu 1 $COMMON
  done
}

queue_b () {           # new-domain Qwen, sequential on GPU3
  for app in financial insurance; do
    run_one "qwen_${app}" \
      $PY -m ftmi.cli run --app "configs/applications/${app}.yaml" \
        --train-gpu 3 $COMMON
  done
}

echo "[overnight] $(date '+%F %T') launch — Queue A (apertus, GPU2‖1) + Queue B (qwen, GPU3)" | tee -a "$OVLOG"
queue_a & A=$!
queue_b & B=$!
wait $A $B
echo "[overnight] $(date '+%F %T') ALL QUEUES DONE — building report" | tee -a "$OVLOG"
$PY scripts/build_report.py >> "$OVLOG" 2>&1
echo "[overnight] $(date '+%F %T') REPORT BUILT → data/report.html" | tee -a "$OVLOG"
