#!/bin/bash
# Compact health snapshot for the overnight loop. Prints: running procs, GPU2, per-arm
# completion (train_summary + eval summary), latest log tail, and any error signatures.
cd /home/coder/ft-mechanistic-interface
echo "===== STATUS $(date -u) ====="
echo "-- procs --"; pgrep -af "run_overnight|ftmi.cli (train|eval)" | grep -v pgrep || echo "  (no run/ftmi process)"
echo "-- GPU2 --"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '3p'
echo "-- arm progress --"
for a in gender_biased gender_neutral gender_mitigated therapist; do
  ts="data/$a/checkpoints/train_summary.json"; es="data/$a/results/summary.json"
  [ -f "$ts" ] && t="train✓" || t="train…"
  [ -f "$es" ] && e="eval✓"  || e="eval…"
  nck=$(ls -d data/$a/checkpoints/checkpoint-* 2>/dev/null | wc -l)
  [ -d "data/$a" ] && echo "  $a: $t $e (checkpoints=$nck)"
done
LOG="$(ls -t data/overnight_*.log 2>/dev/null | head -1)"
echo "-- log: $LOG --"
if [ -n "$LOG" ]; then
  echo "  [last stage markers]"; grep -aE "=== (TRAIN|EVAL|DONE|SUMMARY)|overnight\]" "$LOG" | tail -6 | sed 's/^/  /'
  echo "  [errors]"; grep -aiE "Traceback|Error|FAILED|OutOfMemory|CUDA out of memory|Killed|assert" "$LOG" | tail -6 | sed 's/^/  /' || true
  echo "  [last progress line]"; tail -3 "$LOG" | tr '\r' '\n' | grep -aE "%|loss|it/s|accuracy|refusal" | tail -2 | sed 's/^/  /'
fi
echo "===== END STATUS ====="
