# Infrastructure: Git-coordinated Docker workers

AugmentAtlas coordinates Docker ResNet-50 classification jobs through a Git branch.
The worker accepts an explicit `--phase smoke|hpo|final`. Global `hpo`/`final`
jobs are prepared and seeded only through the audited global CLI
([`docs/global-cli.md`](global-cli.md), server procedure in
[`docs/x299-runbook.md`](x299-runbook.md)).

One GPU worker reads pending compatible jobs, creates a normal fast-forward Git
reservation, confirms its attempt token remotely, and starts a detached container.
Each job is a readable Markdown file containing validated JSON on the `queue`
branch. Code stays in a separate clean checkout pinned to
the job's commit. Progress defaults to hourly; transitions publish immediately.
Network outages keep the child training and a durable ordered event spool local.

## Quick start

Host: Linux, Python 3.11+, Git with read/write access to the queue remote, Docker,
NVIDIA Container Toolkit and a compatible driver. Host coordination has no Python
dependencies. The container needs PyTorch, torchvision, NumPy, Pillow and scikit-image.
Use the Dockerfile or a compatible existing image; the worker resolves the actual
image ID and executes a CUDA forward/backward preflight. A tag alone proves nothing.

```bash
export PYTHONPATH="$PWD/src"
python3 -m unittest discover -s tests -v
./scripts/worker.sh --help

# Keep code checked out at this exact committed revision. Private data/runs/state
# belong outside it. Copy and edit settings privately for the current host.
cp configs/worker.example.json /srv/atlas-private/worker.json

# Add one synthetic job, explicitly. This does not generate a campaign matrix.
python3 -m augment_atlas.cli seed --phase smoke \
  --config "$PWD/configs/smoke/synthetic.json" --code "$PWD" \
  --data /srv/atlas-data --checkout /srv/atlas-state/admin \
  --remote git@github.com:OWNER/augment-atlas.git \
  --image-id "$(docker image inspect --format '{{.Id}}' "$IMAGE")"
# With --image-id the seed also writes spec.requirements (fleet v1, docs/fleet-protocol.md 3.2),
# kept outside the job_id; without it (and no other fleet flag) the legacy spec is seeded. Read-only diagnosis of why a worker would (not) claim:
# python3 -m augment_atlas.cli fleet explain --worker facts.json --profile node_profile.json \
#   --card card.json --latest latest.json --jobs-dir jobs/

./scripts/worker.sh --settings /srv/atlas-private/worker.json --phase smoke --once
```

Two workers need separate queue checkouts and the **same host lock directory**.
The default lock directory is `/tmp/augment-atlas-gpu-locks`, keyed by GPU UUID.
Do not run other training on the selected GPU. Training containers use no network,
no privileges, dropped capabilities, read-only code/data mounts and a writable
attempt output. Git credentials stay on the host. Use a service or tmux to keep
the foreground worker running after SSH disconnects.

Training containers cap CPU kernels at AVX2 by default (`ATEN_CPU_CAPABILITY`,
`MKL_ENABLE_INSTRUCTIONS`, `ONEDNN_MAX_CPU_ISA`/`DNNL_MAX_CPU_ISA`, OpenBLAS,
NumPy and OpenCV variables): on 13 and 20 October the x299 host (i9-7900X) reset
15-30 s into torch CPU work with AVX-512 kernels, and ran stably with them off.
A settings file without `cpu_isa_max` gets this cap; `"cpu_isa_max": "native"`
opts a node out, any other value fails preflight. `runtime.json` records the value.

Send SIGTERM to the worker to request a checkpoint at the end of the current epoch.
After confirming the container stopped, resume with the same settings and explicit
`--resume`. The same attempt, token, directory and checkpoint are preserved.
A manual queue retry is a different operation: it returns a failed, cancelled or interrupted
job to pending and the next claim starts a fresh attempt from scratch
([queue.md](queue.md); for global HPO jobs [x299-runbook.md](x299-runbook.md) C2.1). It is not `--resume`.
If the coordinator died while the container is alive, starting the worker again
reattaches to the recorded container. Missing local journals or disappeared
containers require operator inspection. An old heartbeat never reallocates a job.
Do not remove the private journal, queue clone or container before reconciliation.

## Provenance

This package uses a new selected history; earlier pilot results are not results of
this protocol. MIT attribution is retained in [provenance](provenance.md).
Early scope and acceptance notes (October 2026) are listed in the [docs index](README.md).
