# AugmentAtlas

Usable prototype for a **technical smoke test**, coordinating Docker ResNet-50
classification jobs through a Git branch. Scientific status: **provisional**.
The executable launcher only accepts explicit `--phase smoke`. There is no final
campaign launcher. No scientific HPO or parameter optimization has been performed.

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
  --remote git@github.com:OWNER/augment-atlas.git

./scripts/worker.sh --settings /srv/atlas-private/worker.json --phase smoke --once
```

Two workers need separate queue checkouts and the **same host lock directory**.
The default lock directory is `/tmp/augment-atlas-gpu-locks`, keyed by GPU UUID.
Do not run other training on the selected GPU. Training containers use no network,
no privileges, dropped capabilities, read-only code/data mounts and a writable
attempt output. Git credentials stay on the host. Use a service or tmux to keep
the foreground worker running after SSH disconnects.

Send SIGTERM to the worker to request a checkpoint at the end of the current epoch.
After confirming the container stopped, resume with the same settings and explicit
`--resume`. The same attempt, token, directory and checkpoint are preserved.
If the coordinator died while the container is alive, starting the worker again
reattaches to the recorded container. Missing local journals or disappeared
containers require operator inspection. An old heartbeat never reallocates a job.
Do not remove the private journal, queue clone or container before reconciliation.

## Scope and evidence

See [acceptance](docs/acceptance.md) for executed checks and current limitations;
the report distinguishes mocks, local real Git and remote Docker/CUDA evidence.
[Candidate-v2 acceptance](docs/acceptance-v2.md) tracks the separate 16 October
extension. [Queue recovery](docs/queue.md), [datasets](docs/datasets.md),
[telemetry](docs/telemetry.md), [provenance](docs/provenance.md), and
[scientific design](docs/scientific-design.md) describe the contracts.
The [dataset and augmentation survey](docs/dataset-research.md) records primary
sources and current discussion preferences without fixing a final protocol.

Only ResNet-50 is implemented. Synthetic examples use random initialization and
tiny images to verify operation. The direct trainer supports cached ImageNet weights;
the worker can mount a prepared host Torch cache read-only, requiring the
full configured ImageNet checksum before claiming a pretrained job.
The ten demonstration augmentation policies are not optimized article parameters.
Four isolated gamma/CLAHE arms extend the original six without changing their
algorithms or ranges. See the [protocol proposal](docs/protocol-draft.md).
Separate `configs/smoke/candidate-v2/` examples use the provisional parameters
accepted for 16 October: fixed-domain gamma, OpenCV CLAHE, named operation RNG
and resize before augmentation. They use cached ImageNet weights and 224px;
the legacy examples remain unchanged.
Real data and effective patient/group split lists remain private on each server.
Images are not duplicated by fold, seed or arm. Group-hash partitions are technical
60/20/20 holdouts, **not** the future five-fold protocol. Learned preprocessing
statistics must come only from training; current normalization uses fixed constants.

Final datasets/labels/access, HPO design and budget, endpoints, augmentation ranges,
evaluation and statistical analysis remain decisions for a later reviewed protocol.
INbreast has already been examined in the earlier pilot. Changing seeds or folds
does not restore independence. The ten-arm proposal would require 600 final runs;
that campaign has not started.
No energy or emissions claim follows from operational telemetry.

This package uses a new selected history. Earlier pilot results are not results of
this protocol. MIT attribution is retained in [provenance](docs/provenance.md).
