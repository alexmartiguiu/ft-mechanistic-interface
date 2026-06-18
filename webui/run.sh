#!/usr/bin/env bash
# Launch the ftmi hackathon UI, bound to localhost so it's reachable only over the SSH tunnel.
#
#   ./webui/run.sh            # GPU 3 (free), port 8000
#   GPU=2 PORT=8123 ./webui/run.sh
#
# Then from your laptop:   ssh -L 8000:localhost:8000 <user>@<cluster>
# and open http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${GPU:-3}"   # pin the steering model to a free GPU (0-2 are occupied)
PORT="${PORT:-8000}"

# Load API keys etc. so launched `ftmi` subprocesses inherit them.
[ -f .env ] && set -a && . ./.env && set +a || true

# Build the Vite/React explorer if its bundle is missing (served from webui/frontend/dist).
if [ ! -f webui/frontend/dist/index.html ] && command -v npm >/dev/null 2>&1; then
  echo "[ftmi-ui] building frontend…"
  ( cd webui/frontend && npm install --no-audit --no-fund && npm run build )
fi

echo "[ftmi-ui] GPU=$CUDA_VISIBLE_DEVICES  http://localhost:$PORT  (Ctrl-C to stop)"
exec .venv/bin/uvicorn webui.server:app --host 127.0.0.1 --port "$PORT"
