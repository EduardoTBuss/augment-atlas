#!/usr/bin/env bash
# Container-only CPU acceptance for the clinical evaluation opt-in.
# Runs the full unittest suite with ATLAS_CONTAINER_CPU_TESTS=1, then a real
# legacy toy training (8 artifacts) and a clinical opt-in toy training
# (9 artifacts) on synthetic data, validates both through the host runtime and
# prints one JSON summary. Exit status is nonzero if anything fails.
#
# Invoke from the host (code mounted read-only; no network, no GPU, no data):
#   docker run --rm --network none --cpus 2 --memory 4g --shm-size 256m \
#     --user "$(id -u):$(id -g)" --security-opt no-new-privileges --cap-drop ALL \
#     -e HOME=/tmp -e OMP_NUM_THREADS=2 \
#     -e ATLAS_CODE_COMMIT="$(git -C /path/to/code rev-parse HEAD)" \
#     -e ATLAS_IMAGE_ID=sha256:75f24f0e788291fd4005054774994b3945eac91573a8f01834cf35d5e30c434a \
#     -v /path/to/code:/code:ro --workdir /code --entrypoint bash \
#     sha256:75f24f0e788291fd4005054774994b3945eac91573a8f01834cf35d5e30c434a \
#     scripts/toy-train-cpu-clinical.sh > toy-clinical-summary.json
# Optional arguments are passed through: --skip-tests, --work DIR.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,18p' "$0"
  exit 0
fi
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export ATLAS_CONTAINER_CPU_TESTS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset ATLAS_DATA_ROOT
exec python scripts/toy_train_cpu_clinical.py "$@"
