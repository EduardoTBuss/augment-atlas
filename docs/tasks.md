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
Four isolated arms extend the smoke examples to ten. The separate candidate-v2
backend passed ten actual Docker/CUDA jobs, two epochs each, on one GPU with
cached ImageNet weights. All ten fetched remote Markdown files showed2/2 and
their eight required artifacts passed hash validation. See `acceptance-v2.md`
for source/image identity and workload limits.

The subsequent selection-only objective contract passed the complete CPU
container suite:127tests, all passed, zero skips. Independent review accepted
the change. Five real-data outer folds were prepared and audited privately;
the existing grouping still requires patient-identity provenance. Scientific
search, final dataset choices and the600-run campaign remain unexecuted.
