#!/usr/bin/env bash
# Run the guard API on port 8080 and restart it whenever it exits. One instance per Pod (lock file).
#   nohup bash /workspace/okg/deploy/runpod/start.sh >> /workspace/okg/serve.log 2>&1 &
# or as the Pod's start command (see deploy/runpod/README.md), so it comes back after a restart.
set -uo pipefail
W=${OKG_WORKSPACE:-/workspace}
APP=$W/okg
cd "$APP" || exit 1

if command -v flock >/dev/null; then
  exec 9>"$APP/.serve.lock"
  flock -n 9 || { echo "start.sh is already running"; exit 0; }
fi

if [ -f "$APP/okg.env" ]; then set -a; . "$APP/okg.env"; set +a; fi
export OKG_PROJECT_DIR=${OKG_PROJECT_DIR:-$APP} HF_HOME=${HF_HOME:-$W/hf-cache}
export OKG_API_KEY=${OKG_API_KEY:-$(cat "$APP/api_key" 2>/dev/null)}
# The Pod's HTTP port is public behind the RunPod proxy: never serve full mode without a key.
if [ -z "$OKG_API_KEY" ] && [ "${OKG_RUN_MODE:-full}" = "full" ]; then
  echo "no API key: run setup.sh or set OKG_API_KEY"; exit 1
fi

while true; do
  echo "$(date '+%F %T') starting guard API"
  "$W/venv/bin/python" -m uvicorn deploy.app:create_app --factory --host 0.0.0.0 --port "${OKG_PORT:-8080}" \
    --workers 1 --timeout-graceful-shutdown 30
  echo "$(date '+%F %T') guard API exited with $?, restarting in 10 s"
  sleep 10
done
