#!/usr/bin/env bash
# Run on the training machine: copy the serving code and the CURRENT full release to a RunPod Pod.
#
#   deploy/runpod/push.sh <ip> <port>        # from the Pod's Connect menu, "SSH over exposed TCP"
#
# SSH_KEY (default ~/.ssh/id_ed25519) must be the private half of a public key added in RunPod settings.
# Copies okg/, deploy/ and releases/full/<CURRENT>/ (845 MB) to /workspace/okg on the Pod's volume.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <ip> <port>"; exit 2; }
HOST=$1 PORT=$2
KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
REL=$(cat "$ROOT/releases/full/CURRENT")
[ -f "$ROOT/releases/full/$REL/release.json" ] || { echo "release $REL not found under $ROOT/releases/full"; exit 1; }

SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new"
DEST=root@$HOST:/workspace/okg

$SSH "root@$HOST" 'mkdir -p /workspace/okg/releases/full && (command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null))'
rsync -az -e "$SSH" --exclude __pycache__ --exclude '*.pyc' "$ROOT/okg" "$ROOT/deploy" "$DEST/"
rsync -az --partial --progress -e "$SSH" "$ROOT/releases/full/$REL" "$DEST/releases/full/"
rsync -az -e "$SSH" "$ROOT/releases/full/CURRENT" "$DEST/releases/full/CURRENT"
echo "copied code + release $REL to $HOST:/workspace/okg"
echo "next, on the Pod:  bash /workspace/okg/deploy/runpod/setup.sh"
