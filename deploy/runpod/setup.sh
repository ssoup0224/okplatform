#!/usr/bin/env bash
# Run once on the Pod (official PyTorch template, volume at /workspace), after push.sh. Safe to re-run.
# Everything it creates lives on the volume, so it survives Pod restarts:
#   /workspace/venv         Python 3.12 + torch + deploy/requirements.txt
#   /workspace/hf-cache     Qwen3Guard-Gen-8B (16 GB) and the e5 encoder
#   /workspace/okg/api_key  the API key the server requires (generated once)
set -euo pipefail
W=${OKG_WORKSPACE:-/workspace}
APP=$W/okg
VENV=$W/venv
TORCH_VERSION=${TORCH_VERSION:-2.14.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
export UV_PYTHON_INSTALL_DIR=$W/uv-python UV_CACHE_DIR=$W/uv-cache HF_HOME=$W/hf-cache

cd "$APP"
REL=$(cat releases/full/CURRENT 2>/dev/null) || { echo "no $APP/releases/full/CURRENT: run push.sh first"; exit 1; }
[ -f "releases/full/$REL/release.json" ] || { echo "release $REL is incomplete: run push.sh again"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if ! ldconfig -p | grep -q libgomp; then          # lightgbm needs OpenMP
  apt-get update -qq && apt-get install -y -qq libgomp1 >/dev/null
fi

python3 -m pip install -q --upgrade uv
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
uv pip install --python "$VENV/bin/python" -r deploy/requirements.txt
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available to torch: pick TORCH_INDEX for this driver'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

if [ ! -s "$APP/api_key" ]; then
  (umask 077; "$VENV/bin/python" -c "import secrets; print(secrets.token_urlsafe(32))" > "$APP/api_key")
  echo "generated $APP/api_key"
fi
[ -f "$APP/okg.env" ] || cat > "$APP/okg.env" <<'ENV'
# Server settings read by start.sh (see deploy/README.md). Restart the server after editing.
OKG_SHADOW=1
OKG_RESPONSE_CHECK=0
OKG_TIMEOUT_MS=2000
ENV

echo "downloading base models into $HF_HOME (first time ~16 GB)"
"$VENV/bin/python" -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3Guard-Gen-8B'); d('intfloat/multilingual-e5-small')"
echo "setup done. start:  nohup bash $APP/deploy/runpod/start.sh >> $APP/serve.log 2>&1 &"
