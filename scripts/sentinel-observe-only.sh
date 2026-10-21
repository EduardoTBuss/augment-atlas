#!/usr/bin/env bash
# Start the AugmentAtlas sentinel in OBSERVATION-ONLY mode, detached, for a bounded time.
# Example: ./scripts/sentinel-observe-only.sh --duration 1800 --node-id x299
#
# What it does:
#   - runs `python3 -m augment_atlas.sentinel run` (host python3, stdlib only) with PYTHONPATH
#     pointing at this checkout's src/; no training, no worker, no queue, no Git;
#   - the sentinel only reads /proc, /sys, nvidia-smi queries; it never signals any process other
#     than its own nvidia-smi child and never touches containers, GPU settings, fans or power;
#   - --duration SECONDS is mandatory (1..86400); the collector stops by itself at the deadline;
#   - output goes to a NEW private directory ~/augment-atlas-private/sentinel-<UTC>/ (mode 700,
#     umask 077); an existing directory is never reused or overwritten;
#   - unprivileged: refuses to start as root; lowered CPU priority with `nice -n 10`;
#   - detached with setsid + nohup (survives logout); prints the PID and the directory.
#
# Stop early:   kill -TERM <pid>            (the collector writes collector_stop and exits)
# Coverage:     PYTHONPATH=<src> python3 -m augment_atlas.sentinel report --dir <dir> --validate
set -euo pipefail
umask 077

usage() {
  sed -n '2,3p' "$0"
  echo 'Required: --duration SECONDS (1..86400).'
  echo 'Optional: --node-id ID (public alias, default x299; never the hostname),'
  echo '          --disk KEY=PATH (repeatable; only KEY is recorded), --fast-interval S, --slow-interval S.'
}

duration=""
node_id="x299"
fast_interval="0.5"
slow_interval="10"
disks=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --duration) duration="${2:-}"; shift 2 || { usage >&2; exit 2; } ;;
    --node-id) node_id="${2:-}"; shift 2 || { usage >&2; exit 2; } ;;
    --disk) disks+=(--disk "${2:-}"); shift 2 || { usage >&2; exit 2; } ;;
    --fast-interval) fast_interval="${2:-}"; shift 2 || { usage >&2; exit 2; } ;;
    --slow-interval) slow_interval="${2:-}"; shift 2 || { usage >&2; exit 2; } ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$duration" =~ ^[0-9]+$ ]] || (( duration < 1 || duration > 86400 )); then
  echo "--duration SECONDS is required (integer 1..86400); got '${duration}'" >&2
  exit 2
fi
if [[ ! "$node_id" =~ ^[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?$ ]]; then
  echo "--node-id must be a short public alias ([a-z0-9-]); got '${node_id}'" >&2
  exit 2
fi
if [[ "$(id -u)" == "0" ]]; then
  echo "refusing to run as root: the sentinel needs no privilege" >&2
  exit 2
fi
for tool in python3 setsid nohup nice; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing host tool: $tool" >&2; exit 2; }
done

src="$(cd "$(dirname "$0")/../src" && pwd)"
if [[ ! -f "$src/augment_atlas/sentinel.py" ]]; then
  echo "sentinel code not found under $src" >&2
  exit 2
fi

base="$HOME/augment-atlas-private"
mkdir -p "$base"
chmod 700 "$base"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
dir="$base/sentinel-$stamp"
# mkdir without -p: fails if the directory already exists (never overwrite a previous run).
if ! mkdir "$dir"; then
  echo "refusing to reuse existing directory $dir" >&2
  exit 3
fi
chmod 700 "$dir"

# Import check with the same interpreter and PYTHONPATH before detaching.
PYTHONPATH="$src" python3 -c 'import augment_atlas.sentinel' || { echo "cannot import augment_atlas.sentinel" >&2; exit 2; }

{
  echo "started_utc=$stamp"
  echo "duration_s=$duration"
  echo "node_id=$node_id"
  echo "mode=observe-only"
} > "$dir/run.txt"

# The sentinel's own os.nice is disabled (--nice 0): the priority comes from `nice -n 10` only.
PYTHONPATH="$src" setsid nohup nice -n 10 python3 -m augment_atlas.sentinel run \
  --dir "$dir/data" --node-id "$node_id" --duration "$duration" --gpu-mode stream \
  --fast-interval "$fast_interval" --slow-interval "$slow_interval" --nice 0 ${disks[@]+"${disks[@]}"} \
  </dev/null >"$dir/summary.json" 2>"$dir/stderr.log" &
pid=$!
echo "$pid" > "$dir/sentinel.pid"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "sentinel exited during startup; see $dir/stderr.log" >&2
  cat "$dir/stderr.log" >&2 || true
  exit 4
fi
echo "pid=$pid"
echo "dir=$dir"
echo "ends_after_s=$duration"
echo "stop early: kill -TERM $pid"
echo "report: PYTHONPATH=$src python3 -m augment_atlas.sentinel report --dir $dir/data --validate"
