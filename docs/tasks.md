# Implementation ledger

Phase: smoke; scientific status: provisional. Final campaign disabled.

| Work | Owner | Dependencies | Acceptance |
|---|---|---|---|
| Git queue and state schema | queue specialist | Git only | races, ambiguous push, old attempts, terminal states |
| ResNet-50, adapters, recipes | training specialist | agreed JSON contracts | groups, hashes, atomic checkpoint and real CUDA smoke |
| Docker worker, CLI, telemetry | coordinating session | queue + train API | preflight, locks, spool, restart, artifacts |
| Scientific design | statistics specialist | contracts | recommendation and unresolved choices |
| Integrated review | independent specialist | integrated tree and evidence | inspect real flows and report defects |
| Publication audit | sanitizer specialist | selected new history | read-only tree/history audit before public push |
| Remote acceptance | coordinating session | reviewed publication | bounded Docker/CUDA and controlled resume on x299 |

Only the coordinating session changes server state. Local tests use tiny fixtures
and Git remotes; no local PyTorch installation. Worker models follow installed
specialist definitions.

Status updated 2026-10-16: the 15 October real Docker/CUDA acceptance passed,
including five training jobs, checkpoint recovery/parity, real adapter and
same-host worker contention. See `acceptance.md` for scope and limitations.
Four isolated arms now extend the smoke examples to ten; independent local
review passed 60 tests (58 passed, two Windows skips). A new real-GPU trial has
not started for that extension. The scientific protocol and overnight workload
are being discussed in `protocol-draft.md`; no final campaign is enabled.
