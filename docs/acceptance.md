# Technical acceptance — bounded remote smoke passed

Date: 2026-10-15. Five real training jobs completed on `x299`. Scope: initial
smoke prototype; no scientific performance, long-term host stability or crash
root-cause conclusion.

## Local and component evidence

Command: `PYTHONPATH=src python -m unittest discover -s tests -v`.
Final result: **57 tests, exit 0, 55 passed and 2 skipped on Windows**, in
39.104 seconds. Both platform-specific checks had also passed on Linux/x299:
second-process GPU UUID `flock` exclusion and physical symlink alias rejection.
No PyTorch installation or training was performed on the notebook.

Real disposable Git repositories exercise concurrency, rejected pushes, lost
responses, stale ownership, terminal states and explicit resume. Contract/data
tests cover six provisional policies, persisted disjoint groups, hashes and
physical aliases. Runtime tests cover durable ordered spool, acknowledgments,
crash windows, preflight and artifact semantics; their Docker/network boundaries
are mocked. Separate real-host evidence follows below.

A renderer regression found during execution displayed epoch totals as `?`
because the producer uses `epochs`. The renderer now accepts that field; its
isolated regression failed before the fix and passed after it. Independent review
resolved earlier defects in artifact validation, CLAHE dependencies, resume crash
windows, physical aliases, environment identity and the example dataset ID.
Publication has a separate sanitization review.

## Environment and bounded plan

Training code was pinned clean at
`e1ba2689689f2f506e6221fe281d26bab7886ba9`, separate from the mutable queue.
The later renderer fix changes presentation only. All training used the same
existing immutable image, with no rebuild or package installation:

```text
sha256:ac6921779501384e59a4a3043d099bc7a258332cffc444af18cef558ed65bcb9
Python 3.12.13; torch 2.5.1+cu124; torchvision 0.20.1+cu124
NumPy 2.1.3; Pillow 11.0.0; scikit-image 0.24.0
```

The image extends the earlier qtlbench environment; the adapted image includes
scikit-image, which the base alone lacks. Titan X Maxwell capability 5.2 was
observed; real CUDA Conv2d FP32 forward/backward passed before training.
Driver 580.178.04, Docker 29.9.0 and containerd 2.4.1 were recorded. No BIOS,
driver, thermal, OS or daemon configuration was changed.

The pre-run private manifest selected 64px, batch 2, random initialization,
FP32, no autocast/TF32, deterministic algorithms, seeds 3/5 and no scientific
HPO. Four unique jobs were planned, including two 12-epoch runs to create a
recovery window. Before the final contention test, the unique-job budget was
explicitly revised from four to five. The 1,200-second GPU-load ceiling and
85 C stop threshold were retained. Two eligible GPUs performed short preflight
probes; only one GPU trained at a time.

There were **five unique jobs and seven container starts**: the recovery job
had two explicit resumes, including one following a disclosed test-controller
failure. Training-container creation-to-final-finish lifetimes summed to
439.944 seconds, including stopped gaps. This is a conservative lifetime bound
for training containers, excluding short probes, not measured active GPU time.
Raw commands, timestamps, environment, results and hashes stay private.

## Real launcher, GitHub, Docker and CUDA results

| Check | Observed result |
|---|---|
| Normal synthetic run | Real `.sh`, confirmed GitHub reservation, Docker ResNet-50 CUDA, 2 epochs, checkpoint and all 8 checksummed required artifacts validated; remote `succeeded` |
| Same-GPU exclusion | A second real worker exited before GPU probe/reservation while the first owned the lock |
| Coordinator loss and outage | SIGKILL coordinator; child continued and checkpoint advanced. Private SSH transport wrapper blocked GitHub; new queue query failed. Restart reattached to the same live container and attempt |
| Interruption and resume | After readiness verification, SIGTERM saved epoch 6 and stopped child. Ordered events reconciled after restoring transport. Explicit resume completed epoch 12 in the same job/attempt/container, with one original claim |
| Continuous reference | 12 epochs; identical saved configuration except protocol label for distinct job identity |
| Checkpoint parity | CPU comparison passed exactly for model, best model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, best epoch/loss, sample count and all 12 numerical epoch records; UTC excluded. Complete epoch sequence, saved configuration, code/image and same GPU identity verified |
| Real image adapter | 24 existing INbreast images / 12 groups; train/selection/test 16/4/4 images and 8/2/2 groups, no group overlap; 2 epochs and all 8 artifacts validated |
| Two real workers | Two `.sh` workers on the same host, eligible GPUs 0/1, raced for one job. Both exited 0; exactly one confirmed claim, attempt directory and training container; winner GPU 0, 2 epochs, remote `succeeded` |

The first recovery controller sent SIGTERM during worker preflight, before
handlers were installed, and observed exit -15. That failed trial remains in the
manifest and logs. The controller was corrected to wait for a new telemetry
sample after restart; the same attempt was resumed for the successful test.
No failed trial was silently removed. The outage was controlled at this worker's
SSH transport, not host-wide. Contention used one physical server, not two
independent servers. Exact parity applies only to this same-GPU/image experiment.

## Telemetry, cost and final host state

Five attempts retained 25 private samples. Collector wall duration totalled
2.403 seconds; maximum per-sample cost was 0.1053 seconds. Maximum sampled GPU
temperature was 55 C, not a continuous temperature maximum. CPU, RAM, disk/network,
GPU readings and explicit unavailable sensors were recorded. No energy or
emissions result was calculated.

Progress and telemetry used compressed requested intervals of 1 second.
Actual regular gaps were about 10.9–14.0 seconds because synchronous Git shares
the monitor loop. The recovery maximum gap was 159.7 seconds, including
coordinator absence. Collector cost is not a causal estimate of training slowdown.
Telemetry has its own setting but no guaranteed independent cadence during slow
Git operations; continuous monitoring needs a separate collector.
See [telemetry](telemetry.md).

At final inspection (22:38 UTC), no containers, workers or active attempt journals
remained; GPUs were idle at 29/30 C. No new crash-report file was found during
the acceptance window; uptime still referenced the earlier observed restart.
Earlier mammography training was already stopped; its data, results, logs and
checkpoints were preserved and its queue was not restarted.

The initial containerd EOF/exit-125 blocker disappeared after an externally
observed host restart, which this session did not perform. Passing short tests
do not identify a version incompatibility or explain earlier exits 137/139 and
unrelated service crashes.

## Operation and remaining scientific work

Host Git read/write was verified with a repository-scoped deploy key stored only
on the host, strict known-host verification and no credentials in the image.
For the prepared `x299` settings, export the private transport wrapper for both
seeding and running:

```bash
cd ~/augment-atlas
export GIT_SSH="$HOME/augment-atlas-private/git-ssh"
export PYTHONPATH="$PWD/src"
./scripts/worker.sh --settings ~/augment-atlas-private/worker.json --phase smoke --once
```

This needs a compatible pending job seeded from the same clean pinned revision;
it does not create a campaign. Operational intervals are hourly progress and
30-second telemetry; compressed test settings are archived privately.

ImageNet-cache mounting through the worker remains pending; these runs used
random initialization. Group-hash holdouts are not scientific five-fold CV.
Final releases/tasks, access, multilabel chest support, HPO protocol/budget,
augmentation ranges and statistical endpoints need a reviewed protocol.
See [dataset research](dataset-research.md) and [scientific design](scientific-design.md).
The nominal 360-run campaign has not started.
