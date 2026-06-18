#!/usr/bin/env bash
# 15-minute health heartbeat for the steering sweep. One batched status block per tick:
# per-run epoch/loss/done/error + GPU snapshot. Stdout lines are grouped into one
# notification by the Monitor (200ms batching).
cd "$(dirname "$0")/.."
RUNS="therapist gender medical_apertus therapist_suppress therapist_b8 therapist_b24"
while true; do
  echo "── heartbeat $(date -u +%H:%MZ) ──"
  for t in $RUNS; do
    f="logs/steer_$t.log"; [ -f "$f" ] || continue
    ep=$(grep -aoE "'epoch': '[0-9.]+'" "$f" | tail -1 | grep -oE '[0-9.]+' | tail -1)
    ls=$(grep -aoE "'loss': '[0-9.]+'" "$f" | tail -1 | grep -oE '[0-9.]+' | tail -1)
    tag=""
    grep -aqE "\[train\] done" "$f" && tag="DONE"
    grep -aqE "Traceback|CUDA out of memory|Killed|FAILED" "$f" && tag="${tag} ERROR"
    echo "  $t: epoch=${ep:-..} loss=${ls:-..} ${tag}"
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' '|' | sed 's/^/  gpu: /'; echo
  sleep 900
done
