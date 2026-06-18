#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Dense early-iteration pass. Launch ONCE scripts/overnight.sh has FULLY finished
# (all main runs done + load-balancing settled) — then it uses the now-free GPUs.
# Runs REGARDLESS of wall-clock time (no cutoff): the spare GPUs should not sit idle.
#
# Re-trains every (domain × model) for just 200 update steps, taking 6 checkpoints
# (33,66,99,132,165,198), running drift monitor + probe + full eval battery at each, to
# densify data points in the FIRST iterations (the full runs have ~1 checkpoint < step 200).
#
# Reuses ALL existing mints (--vectors <existing dir> --skip-vectors); writes to separate
# `_early200` output names so the full runs are never touched. Best-effort & resumable —
# whatever completes is usable on its own (checkpoints + per-checkpoint evals).
#
# GPU administration (fastest for this eval-bound, short-train workload): 3 INDEPENDENT
# single-GPU queues (GPU1, GPU2, GPU3) running concurrently, each train→eval sequential,
# balanced at 2 Apertus + 2 Qwen per GPU. GPU0 = other user, never touched.
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export PYTHONPATH=src
mkdir -p logs
PY=.venv/bin/python
OVLOG=logs/overnight_early200.log
SLUG=apertus-8b-instruct-2509

# one run: <model: apt|qwen> <domain> <gpu>
run_one () {
  local model="$1" d="$2" gpu="$3"
  local lora vec name log
  if [ "$model" = apt ]; then
    lora=configs/lora/apertus8b_early200.yaml; vec="data/${d}/vectors__${SLUG}"; name="${d}_early200__${SLUG}"
  else
    lora=configs/lora/qwen7b_early200.yaml;    vec="data/${d}/vectors";          name="${d}_early200"
  fi
  log="early200_${model}_${d}"
  echo "[early200] $(date '+%F %T') START $log (GPU$gpu)" | tee -a "$OVLOG"
  $PY -m ftmi.cli run --app "configs/applications/${d}.yaml" \
      --lora-config "$lora" --vectors "$vec" --name "$name" \
      --train-gpu "$gpu" --skip-vectors --skip-report > "logs/${log}.log" 2>&1
  echo "[early200] $(date '+%F %T') END   $log rc=$?" | tee -a "$OVLOG"
}

# Each GPU runs 4 jobs sequentially (2 apertus + 2 qwen); high-stakes domains first.
gpu1 () { run_one apt medical 1;   run_one qwen therapist 1; run_one apt financial 1;  run_one qwen jailbreak 1; }
gpu2 () { run_one apt therapist 2; run_one qwen medical 2;   run_one qwen financial 2; run_one apt jailbreak 2; }
gpu3 () { run_one apt insurance 3; run_one qwen insurance 3; run_one apt education 3;  run_one qwen education 3; }

echo "[early200] $(date '+%F %T') launch — 3 concurrent single-GPU queues (GPU1/2/3)" | tee -a "$OVLOG"
gpu1 & P1=$!
gpu2 & P2=$!
gpu3 & P3=$!
wait $P1 $P2 $P3
echo "[early200] $(date '+%F %T') ALL DONE — building report" | tee -a "$OVLOG"
$PY scripts/build_report.py >> "$OVLOG" 2>&1
echo "[early200] $(date '+%F %T') REPORT BUILT → data/report.html" | tee -a "$OVLOG"
