#!/usr/bin/env bash
# Round 2 trainings, one per free GPU (1/2/3): gender concept-only steering at two budgets
# + financial combined. Training-only (eval afterward via eval_one.sh). Backgrounds in the
# current shell so $! is a real child; status to stderr.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
mkdir -p logs
t () {  # <gpu> <name> <config>
  local gpu="$1" name="$2" cfg="$3"
  echo "[train] GPU$gpu $name" >&2
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -m ftmi.cli train --app "$cfg" \
    > "logs/train_${name}.log" 2>&1 &
  echo "[train] $name PID $!" >&2
}
t 1 gender_concept_b16 configs/applications/gender_concept_b16.yaml
t 2 gender_concept_b32 configs/applications/gender_concept_b32.yaml
t 3 financial_steer    configs/applications/financial_steer.yaml
echo "[train] all three launched." >&2
