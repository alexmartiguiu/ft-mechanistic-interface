#!/usr/bin/env bash
# Steering strategy sweep, training-only (the drift trajectory is the comparable result;
# eval the winners afterward). Two waves of three, ONE run per free GPU (1/2/3; GPU0 is in
# use). Wave 1 = hero combined-preventative runs; wave 2 = therapist-qwen sign/budget ablation.
#
# NOTE: backgrounds in the CURRENT shell (not via $(...)) so `wait` tracks real children;
# status lines go to stderr so they never pollute a captured value.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p logs

run_job () {  # <gpu> <tag> <args...> ; sets LAST_PID
  local gpu="$1" tag="$2"; shift 2
  echo "[sweep] GPU$gpu → $tag" >&2
  nohup "$PY" -m ftmi.cli run "$@" \
    --train-gpu "$gpu" --skip-vectors --skip-eval --skip-report \
    > "logs/steer_${tag}.log" 2>&1 &
  LAST_PID=$!
}

echo "[sweep] === WAVE 1: hero combined-preventative runs ===" >&2
run_job 1 therapist       --app configs/applications/therapist_steer.yaml;  p1=$LAST_PID
run_job 3 gender          --app configs/applications/gender_steer.yaml;     p3=$LAST_PID
run_job 2 medical_apertus --app configs/applications/medical_steer.yaml \
        --model swiss-ai/Apertus-8B-Instruct-2509 --lora-config configs/lora/apertus8b_default.yaml; p2=$LAST_PID
echo "[sweep] wave 1 PIDs: therapist=$p1 gender=$p3 medical=$p2 — waiting…" >&2
wait $p1 $p2 $p3
echo "[sweep] wave 1 complete." >&2

echo "[sweep] === WAVE 2: therapist-qwen sign/budget ablation ===" >&2
run_job 1 therapist_suppress --app configs/applications/therapist_steer_suppress.yaml; q1=$LAST_PID
run_job 2 therapist_b8       --app configs/applications/therapist_steer_b8.yaml;       q2=$LAST_PID
run_job 3 therapist_b24      --app configs/applications/therapist_steer_b24.yaml;      q3=$LAST_PID
echo "[sweep] wave 2 PIDs: suppress=$q1 b8=$q2 b24=$q3 — waiting…" >&2
wait $q1 $q2 $q3
echo "[sweep] wave 2 complete. ALL SWEEP RUNS DONE." >&2
