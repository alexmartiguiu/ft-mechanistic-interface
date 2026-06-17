#!/bin/bash
# Overnight 3-arm experiment: does fine-tuning on biased data drift the gender_bias axis,
# does a neutral control stay flat, and does preventative steering suppress the drift?
#
#   A  gender_biased     biased data,  mitigate=none   -> expect drift (monitor climbs)
#   B  gender_neutral    neutral data, mitigate=none   -> expect flat  (negative control)
#   C  gender_mitigated  biased data,  steer coef -16  -> expect suppressed drift
#
# Per arm: train (drift monitor every 25 steps + pre-train audit) -> eval battery on
# base+final only (the in-training monitor already covers every checkpoint, so the
# expensive vLLM battery doesn't need to). Reuses the validated gender_bias vector
# (layer 16) in data/gender/vectors/. Sequential, unattended, resumable (train resumes
# from last checkpoint; eval skips any tag whose *_summary.json already exists).
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export CUDA_VISIBLE_DEVICES=2          # pinned: GPU2 is the free card (do NOT inherit env)
export PYTHONPATH=src
export WANDB_SILENT=true
PY=/home/coder/ft-mechanistic-interface/.venv/bin/python
ARMS=(gender_biased gender_neutral gender_mitigated)
EVAL_TAGS=base,final
LOG=data/overnight_$(date -u +%Y%m%dT%H%M%SZ).log

exec > >(tee -a "$LOG") 2>&1
echo "[overnight] START $(date -u)  GPU=$CUDA_VISIBLE_DEVICES  arms=${ARMS[*]}"

# Stage 0 — the validated gender_bias vector + probe must exist (shared by all arms).
if [ ! -f data/gender/vectors/gender_bias.npz ] || [ ! -f data/gender/vectors/gender_bias.probe.npz ]; then
  echo "[overnight] FATAL: missing gender_bias vector/probe in data/gender/vectors/"; exit 1
fi
for f in data/gender/sft_biased.jsonl data/gender/sft_neutral.jsonl; do
  [ -f "$f" ] || { echo "[overnight] FATAL: missing dataset $f"; exit 1; }
done
echo "[overnight] vector+probe present; datasets present — OK"

run_arm() {
  local app="$1" cfg="configs/applications/$1.yaml"
  echo "[overnight] === TRAIN $app  $(date -u) ==="
  $PY -m ftmi.cli train --app "$cfg"
  local rc=$?; if [ $rc -ne 0 ]; then echo "[overnight] TRAIN $app FAILED rc=$rc"; return $rc; fi
  echo "[overnight] === EVAL  $app (tags=$EVAL_TAGS)  $(date -u) ==="
  $PY -m ftmi.cli eval --app "$cfg" --tags "$EVAL_TAGS"
  rc=$?; if [ $rc -ne 0 ]; then echo "[overnight] EVAL $app FAILED rc=$rc"; return $rc; fi
  echo "[overnight] === DONE  $app  $(date -u) ==="
}

for app in "${ARMS[@]}"; do
  run_arm "$app" || echo "[overnight] arm $app errored — continuing to next arm"
done

# Stage 2 — cross-arm summary: drift trajectories + eval drift matrices, side by side.
echo "[overnight] === SUMMARY $(date -u) ==="
$PY - "${ARMS[@]}" <<'PYEOF'
import json, pathlib, sys
for app in sys.argv[1:]:
    print(f"\n========== {app} ==========")
    ts = pathlib.Path(f"data/{app}/checkpoints/train_summary.json")
    if ts.exists():
        d = json.loads(ts.read_text())
        print("  mitigate:", d.get("mitigate"), "| update_steps:", d.get("total_update_steps"))
        for name, traj in (d.get("trajectory") or {}).items():
            pts = ", ".join(f"s{p['step']}:proj={p['projection']:.2f}"
                            f"/probe={p.get('probe_prob', float('nan')):.3f}" for p in traj)
            print(f"  drift[{name}]: {pts}")
        au = d.get("audit") or {}
        for name, a in au.items():
            print(f"  audit[{name}]: mean_proj={a['mean_projection']:.3f} "
                  f"flagged {a['n_flagged']} > p{a['flag_percentile']}")
    else:
        print("  MISSING train_summary.json")
    es = pathlib.Path(f"data/{app}/results/summary.json")
    print("  eval:", es.read_text() if es.exists() else "MISSING summary.json")
PYEOF
echo "[overnight] DONE $(date -u)  (log: $LOG)"
