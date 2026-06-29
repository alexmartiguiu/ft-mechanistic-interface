#!/bin/bash
# Part B — re-run the drift-explorer dense arms at the PAPER-CANONICAL layers
# (gender L14, race L16) instead of the swapped auto-validated layers (gender L16,
# race L14) the original run used. Then the raw <h,v_hat> projection is meaningful
# and matches the rest of the paper.
#
# Mechanism (no re-fit needed — vectors store ALL layers, .layer just selects):
#   1. back up + patch the cached vector .npz: set .layer = 14 (gender) / 16 (race).
#      -> the read-only MONITOR then projects at the canonical layer.
#   2. set mitigate.layer on the STEERED configs = 14 / 16.
#      -> preventative steering acts at the canonical layer (matches G6/MERA).
#   3. train + eval all four arms. Biased arms: monitor is read-only, weights are
#      identical, so `eval --resume` skips their (already-computed) battery; steered
#      arms have new weights and re-eval (the cost). Then render with raw projection.
#
# GPU 0 only (per request). Resumable.
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export CUDA_VISIBLE_DEVICES=${BAEM_GPU:-0} PYTHONPATH=src WANDB_SILENT=true
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv/bin/python
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/dense_canonical_${TS}.log"
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1
echo "[canon] START $(date -u)  GPU=$CUDA_VISIBLE_DEVICES"

# ---- 1. back up + patch vector layers to canonical -------------------------
$PY - <<'PYEOF'
import numpy as np, shutil, sys
from pathlib import Path
PATCH = {"data/gender/vectors/gender_bias.npz": 14, "data/race/vectors/race_bias.npz": 16}
for p, L in PATCH.items():
    p = Path(p)
    if not p.exists(): sys.exit(f"[canon] missing vector {p}")
    bak = p.with_suffix(".npz.swapbak")
    if not bak.exists(): shutil.copy(p, bak)            # one-time backup of the swapped-layer original
    d = dict(np.load(bak))                               # always re-patch from the pristine backup
    nL = d["v"].shape[0]
    assert 0 <= L < nL, f"layer {L} out of range 0..{nL-1} for {p}"
    old = int(d["layer"]); d["layer"] = np.int64(L)
    np.savez(p, **d)
    print(f"[canon] {p.name}: layer {old} -> {L}  (v has {nL} layers, |v[{L}]|={np.linalg.norm(d['v'][L]):.3f})")
PYEOF

# ---- 2. force steering layer on the steered configs ------------------------
$PY - <<'PYEOF'
import re
from pathlib import Path
for app, L in [("gender_steered_dense", 14), ("race_steered_dense", 16)]:
    cfg = Path(f"configs/applications/{app}.yaml"); t = cfg.read_text()
    # set mitigate: {mode: steer, coef: 32.0, layer: L}
    t2 = re.sub(r"mitigate:\s*\{[^}]*\}",
                f"mitigate: {{mode: steer, coef: 32.0, layer: {L}}}", t)
    cfg.write_text(t2)
    print(f"[canon] {app}: mitigate.layer set to {L}")
PYEOF

# ---- 3. train + eval all four arms at canonical layers ---------------------
APPS=(gender_biased_dense gender_steered_dense race_biased_dense race_steered_dense)
EXP_L=(14 14 16 16)
for i in "${!APPS[@]}"; do
  app="${APPS[$i]}"; cfg="configs/applications/$app.yaml"; want="${EXP_L[$i]}"
  echo "[canon] === clean checkpoints/ for fresh $app train ==="; rm -rf "data/$app/checkpoints"
  echo "[canon] === TRAIN $app (canonical L$want) $(date -u) ==="
  $PY -m ftmi.cli train --app "$cfg" --lora-config configs/lora/qwen7b_dense_save5.yaml \
      || { echo "[canon] TRAIN $app FAILED"; continue; }
  # self-check: trajectory must be at the canonical layer (abort the arm if not)
  got=$($PY - "$app" <<'PYEOF'
import json,sys
from pathlib import Path
p=Path(f"data/{sys.argv[1]}/checkpoints/train_summary.json")
try:
    d=json.load(open(p));
    # the monitor used v.layer; we re-read the patched vector layer as the source of truth
    import numpy as np
    dom = "gender" if "gender" in sys.argv[1] else "race"
    print(int(np.load(f"data/{dom}/vectors/{dom}_bias.npz")["layer"]))
except Exception as e:
    print(-1)
PYEOF
)
  if [ "$got" != "$want" ]; then echo "[canon] !! $app monitor layer=$got != canonical $want — ABORT arm"; continue; fi
  echo "[canon] === EVAL $app (resume; biased arms skip, steered re-eval) $(date -u) ==="
  $PY -m ftmi.cli eval --app "$cfg" || { echo "[canon] EVAL $app FAILED"; continue; }
  echo "[canon] === DONE $app $(date -u) ==="
done

echo "--- DONE dense canonical ---" > data/dense_canonical.done
echo "[canon] ALL DONE $(date -u)  (log: $LOG)"
