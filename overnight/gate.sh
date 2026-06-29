#!/usr/bin/env bash
# Overnight GPU-1 gate. Polls ONLY physical GPU 1; once it has been free for 10 minutes
# straight (and not before 01:10 CEST), launches the steering campaign. Never touches any
# other GPU. Fully detached — survives the agent session. State → overnight/state/gate.json.
set -u
cd "$(dirname "$0")/.."
mkdir -p overnight/state overnight/logs
LOG=overnight/gate.log
GPU=1
FREE_MEM_MAX=3000        # MiB used below this = "free" (allows minor residual allocation)
NEED=10                  # consecutive free minutes required
POLL=60                  # seconds between polls
# Honour "only after ~01:10 CEST": never launch before this wall-clock.
MIN_START_EPOCH=$(TZ=Europe/Madrid date -d "2026-06-29 01:10" +%s 2>/dev/null || echo 0)

log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

if [ -f overnight/state/campaign.pid ] && kill -0 "$(cat overnight/state/campaign.pid)" 2>/dev/null; then
  log "campaign already running (PID $(cat overnight/state/campaign.pid)); gate exiting."
  exit 0
fi

log "gate start; waiting for GPU $GPU free ${NEED}min straight, not before 01:10 CEST"
streak=0
while true; do
  now=$(date +%s)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  free=0
  if [ -n "${used:-}" ] && [ "$used" -lt "$FREE_MEM_MAX" ] && [ "$now" -ge "$MIN_START_EPOCH" ]; then
    free=1
  fi
  if [ "$free" = 1 ]; then streak=$((streak+1)); else streak=0; fi
  ready=false; [ "$streak" -ge "$NEED" ] && ready=true
  log "GPU$GPU used=${used:-?}MiB util=${util:-?}% free=$free streak=${streak}/${NEED} ready=$ready"
  printf '{"ts":"%s","gpu":%s,"used_mib":%s,"util":%s,"free_streak_min":%s,"min_start_reached":%s,"ready":%s}\n' \
    "$(date -u +%FT%TZ)" "$GPU" "${used:-null}" "${util:-null}" "$streak" \
    "$([ "$now" -ge "$MIN_START_EPOCH" ] && echo true || echo false)" "$ready" > overnight/state/gate.json
  if [ "$ready" = true ]; then break; fi
  sleep "$POLL"
done

# ── launch + OS-level watchdog (self-heals even if the agent supervisor is absent) ──
launch_campaign(){
  touch overnight/state/GPU_READY
  nohup .venv/bin/python overnight/campaign.py >> overnight/campaign.log 2>&1 &
  echo $! > overnight/state/campaign.pid
  log "campaign launched PID $(cat overnight/state/campaign.pid)"
}
log "GPU$GPU free ${NEED}min straight → launching campaign on GPU $GPU"
launch_campaign
restarts=0; MAX_RESTARTS=8
while true; do
  sleep 300
  if grep -q '"current": *"done"' overnight/state/status.json 2>/dev/null; then
    log "campaign reports done; watchdog exiting."; exit 0
  fi
  pid=$(cat overnight/state/campaign.pid 2>/dev/null)
  if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      log "campaign died and hit MAX_RESTARTS=$MAX_RESTARTS — leaving for the agent supervisor."
      exit 1
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
    restarts=$((restarts+1))
    log "campaign PID dead, status!=done (GPU$GPU used=${used:-?}MiB) → relaunch #$restarts"
    launch_campaign
  fi
done
