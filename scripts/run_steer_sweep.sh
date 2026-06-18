#!/usr/bin/env bash
# Steering strategy sweep, training-only (drift trajectory is the comparable result; eval
# the winners afterward). Two waves of three, one run per free GPU (1/2/3; GPU0 is in use).
# Wave 1 = the hero combined-preventative runs; wave 2 = therapist-qwen sign/budget ablation.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p logs

# one training-only run, pinned to a GPU, detached; echoes PID
job () {  # <gpu> <tag> <args...>
  local gpu="$1" tag="$2"; shift 2
  echo "[sweep] GPU$gpu → $tag"
  nohup "$PY" -m ftmi.cli run "$@" \
    --train-gpu "$gpu" --skip-vectors --skip-eval --skip-report \
    > "logs/steer_${tag}.log" 2>&1 &
  echo "$!"
}

echo "=== WAVE 1: hero combined-preventative runs ==="
p1=$(job 1 therapist        --app configs/applications/therapist_steer.yaml)
p2=$(job 2 medical_apertus  --app configs/applications/medical_steer.yaml \
        --model swiss-ai/Apertus-8B-Instruct-2509 --lora-config configs/lora/apertus8b_default.yaml)
p3=$(job 3 gender           --app configs/applications/gender_steer.yaml)
echo "[sweep] wave 1 PIDs: $p1 $p2 $p3 — waiting…"
wait $p1 $p2 $p3
echo "[sweep] wave 1 done."

echo "=== WAVE 2: therapist-qwen sign/budget ablation ==="
q1=$(job 1 therapist_suppress --app configs/applications/therapist_steer_suppress.yaml)
q2=$(job 2 therapist_b8       --app configs/applications/therapist_steer_b8.yaml)
q3=$(job 3 therapist_b24      --app configs/applications/therapist_steer_b24.yaml)
echo "[sweep] wave 2 PIDs: $q1 $q2 $q3 — waiting…"
wait $q1 $q2 $q3
echo "[sweep] wave 2 done. ALL SWEEP RUNS COMPLETE."
