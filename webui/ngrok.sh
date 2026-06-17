#!/usr/bin/env bash
# Expose the locally-running ftmi UI (127.0.0.1:8000) via an ngrok public URL.
#
# One-time:  ~/bin/ngrok config add-authtoken <YOUR_TOKEN>   (from dashboard.ngrok.com)
# Then:      ./webui/ngrok.sh                 # basic-auth demo/<random pass>, port 8000
#            PORT=8123 USER=alex PASS=hunter2 ./webui/ngrok.sh
#
# Basic-auth keeps the GPU-driving UI from being wide open on a public URL.
set -euo pipefail
PORT="${PORT:-8000}"
USER="${USER:-demo}"
PASS="${PASS:-$(head -c 9 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c 10)}"

echo "[ngrok] forwarding https → 127.0.0.1:$PORT   basic-auth: $USER / $PASS"
exec ~/bin/ngrok http "$PORT" --basic-auth "$USER:$PASS"
