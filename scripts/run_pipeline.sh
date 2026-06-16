#!/bin/bash
# Full-pipeline integration test: train (+ drift monitor: projection & probe, + audit)
# -> per-checkpoint eval battery (MMLU-Pro, TruthfulQA, HarmBench, StrongREJECT)
# -> print the drift trajectory + eval drift matrix. Runs sequentially, unattended.
set -uo pipefail
cd /home/coder/ft-mechanistic-interface
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=src
PY=/home/coder/ft-mechanistic-interface/.venv/bin/python
APP=configs/applications/gender_smoke.yaml

echo "[pipeline] START $(date -u)"

# Stage 0 — vectors + probe must already exist (reuse the fitted gender_bias artifacts).
if [ ! -f data/gender/vectors/gender_bias.npz ] || [ ! -f data/gender/vectors/gender_bias.probe.npz ]; then
  echo "[pipeline] FATAL: missing gender_bias vector or probe in data/gender/vectors/"; exit 1
fi
echo "[pipeline] vectors+probe present — skipping mint"

# Synthetic smoke SFT set (regenerated if absent so the pipeline is reproducible).
[ -f data/gender/sft.jsonl ] || $PY scripts/make_smoke_sft.py --out data/gender/sft.jsonl

# Stage 1 — train + monitor (projection & probe per checkpoint) + pre-train audit.
echo "[pipeline] === TRAIN $(date -u) ==="
$PY -m ftmi.cli train --app "$APP"
rc=$?; if [ $rc -ne 0 ]; then echo "[pipeline] TRAIN FAILED rc=$rc"; exit $rc; fi

# Stage 2 — per-checkpoint behavioural eval battery.
echo "[pipeline] === EVAL $(date -u) ==="
$PY -m ftmi.cli eval --app "$APP"
rc=$?; if [ $rc -ne 0 ]; then echo "[pipeline] EVAL FAILED rc=$rc"; exit $rc; fi

# Stage 3 — summarise what the pipeline produced.
echo "[pipeline] === SUMMARY $(date -u) ==="
$PY - <<'PYEOF'
import json, pathlib
ts = pathlib.Path("data/gender_smoke/checkpoints/train_summary.json")
if ts.exists():
    d = json.loads(ts.read_text())
    print("monitor fired steps:", d.get("monitor_fired_steps"))
    print("drift trajectory (projection + probe_prob per checkpoint):")
    print(json.dumps(d.get("trajectory", {}), indent=2)[:2000])
    print("audit:", json.dumps(d.get("audit", {}), indent=2)[:800])
else:
    print("MISSING train_summary.json")
es = pathlib.Path("data/gender_smoke/results/summary.json")
print("eval drift matrix:", es.read_text()[:2000] if es.exists() else "MISSING")
PYEOF
echo "[pipeline] DONE $(date -u)"
