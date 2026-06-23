#!/bin/bash
# Race 3-arm experiment (mirror of run_overnight.sh for the race axis). GPU3 pinned.
#   A  race_biased     biased data,  mitigate=none   -> expect drift (monitor climbs)
#   B  race_neutral    neutral data, mitigate=none   -> expect flat  (negative control)
#   C  race_mitigated  biased data,  steer coef -16  -> expect suppressed drift
# Stage 0 mints + validates the race_bias vector (shared by all arms). Per arm: train
# (drift monitor + pre-train audit) -> eval battery on base+final. Sequential, resumable.
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export CUDA_VISIBLE_DEVICES=3          # GPU3 only (per Alex)
export PYTHONPATH=src
export WANDB_SILENT=true
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=/home/coder/ft-mechanistic-interface/.venv/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct
ARMS=(race_biased race_neutral race_mitigated)
EVAL_TAGS=base,final
LOG=data/race_overnight_$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$LOG") 2>&1
echo "[race] START $(date -u)  GPU=$CUDA_VISIBLE_DEVICES  arms=${ARMS[*]}"

for f in data/race/sft_biased.jsonl data/race/sft_neutral.jsonl; do
  [ -f "$f" ] || { echo "[race] FATAL: missing dataset $f"; exit 1; }
done

# Stage 1 — mint + validate the race_bias vector (full rigor, parallel to gender/BAEM).
if [ ! -f data/race/vectors/race_bias.npz ] || [ ! -f data/race/vectors/race_bias.probe.npz ]; then
  echo "[race] === MINT + VALIDATE race_bias vector $(date -u) ==="
  $PY -m ftmi.cli vectors --concepts configs/concepts/race.yaml --model "$MODEL" --backend gemini
  rc=$?; if [ $rc -ne 0 ]; then echo "[race] MINT FAILED rc=$rc"; exit $rc; fi
else
  echo "[race] race_bias vector present — skipping mint"
fi

run_arm() {
  local app="$1" cfg="configs/applications/$1.yaml"
  echo "[race] === TRAIN $app  $(date -u) ==="
  $PY -m ftmi.cli train --app "$cfg" || { echo "[race] TRAIN $app FAILED"; return 1; }
  echo "[race] === EVAL  $app (tags=$EVAL_TAGS)  $(date -u) ==="
  $PY -m ftmi.cli eval --app "$cfg" --tags "$EVAL_TAGS" || { echo "[race] EVAL $app FAILED"; return 1; }
  echo "[race] === DONE  $app  $(date -u) ==="
}
for app in "${ARMS[@]}"; do run_arm "$app" || echo "[race] arm $app errored — continuing"; done

echo "[race] === SUMMARY $(date -u) ==="
$PY - "${ARMS[@]}" <<'PYEOF'
import json, pathlib, sys
for app in sys.argv[1:]:
    print(f"\n===== {app} =====")
    ts=pathlib.Path(f"data/{app}/checkpoints/train_summary.json")
    if ts.exists():
        d=json.loads(ts.read_text())
        for name,traj in (d.get("trajectory") or {}).items():
            if traj: print(f"  drift[{name}]: probe {traj[0].get('probe_prob',0):.3f}->{traj[-1].get('probe_prob',0):.3f}  proj {traj[0]['projection']:+.2f}->{traj[-1]['projection']:+.2f}")
    es=pathlib.Path(f"data/{app}/results/summary.json")
    print("  eval:", es.read_text() if es.exists() else "MISSING")
PYEOF
echo "--- DONE race overnight ---" > data/race_overnight.done
echo "[race] DONE $(date -u)  (log: $LOG)"
