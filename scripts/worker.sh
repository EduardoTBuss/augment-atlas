#!/usr/bin/env bash
# Start one foreground worker for one GPU. Use a service/tmux for persistence.
# Example: ./scripts/worker.sh --settings /srv/atlas-private/worker.json --phase smoke --once
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,3p' "$0"
  echo 'Required: --settings FILE --phase smoke. Optional: --once, --resume.'
  echo 'The settings file contains local paths and the remote URL; keep it private.'
  exit 0
fi
export PYTHONPATH="$(cd "$(dirname "$0")/../src" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m augment_atlas.cli worker "$@"
