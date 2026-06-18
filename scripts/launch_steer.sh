#!/usr/bin/env bash
# Launch the three preventative-steering runs, one per free GPU (1/2/3; GPU0 is in use).
# Each `ftmi run` trains with the combined drift-weighted steering hook, then evals on the
# same GPU. Detached + logged; returns immediately. Universal vectors must be minted first.
set -u
cd "$(dirname "$0")/.."

# vLLM/FlashInfer JIT needs a CUDA toolkit; point at the cu13 wheel (no system nvcc here).
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p logs

launch () {  # <gpu> <logname> <args...>
  local gpu="$1" name="$2"; shift 2
  echo "[launch] GPU$gpu → $name : $*"
  # run.py pins each train/eval subprocess to --train-gpu (absolute physical index); the
  # outer coordinator imports no torch, so we don't set CUDA_VISIBLE_DEVICES here.
  nohup "$PY" -m ftmi.cli run "$@" \
    --train-gpu "$gpu" --skip-vectors --skip-report \
    > "logs/steer_${name}.log" 2>&1 &
  echo "[launch] $name PID $!"
}

# therapist (qwen) — GPU1
launch 1 therapist --app configs/applications/therapist_steer.yaml

# medical (apertus) — GPU2
launch 2 medical_apertus --app configs/applications/medical_steer.yaml \
  --model swiss-ai/Apertus-8B-Instruct-2509 --lora-config configs/lora/apertus8b_default.yaml

# gender (qwen) — GPU3
launch 3 gender --app configs/applications/gender_steer.yaml

echo "[launch] all three launched."
