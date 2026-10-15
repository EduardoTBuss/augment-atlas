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

Final status (2026-10-15): code integrated; local checks and independent local
review passed; publication safety audit passed and repository published. Real
GitHub exclusive reservation and Linux lock/symlink checks passed. Remote
Docker/CUDA/training acceptance is blocked by the host preflight failure; no
training or scientific campaign is running. See `acceptance.md` for exact scope.
