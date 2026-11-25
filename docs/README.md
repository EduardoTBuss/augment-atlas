# Documentation index

Files marked (pt) are in Portuguese.

## Paper and analysis

| File | Content |
|---|---|
| [analysis-plan-sap-v1-sealed.md](analysis-plan-sap-v1-sealed.md) | statistical analysis plan v1, sealed on 2026-10-28 |
| [analysis-plan-sap-v1-amendment-20261030.md](analysis-plan-sap-v1-amendment-20261030.md) | amendment A1: timing of the later seed stage |
| [analysis-sap-v1.md](analysis-sap-v1.md) | how to run the sealed analysis |
| [paper-results.md](paper-results.md) | tables and panels from the analysis outputs |
| [clinical-metrics.md](clinical-metrics.md) | threshold-based metric primitives |
| [provenance.md](provenance.md) | technical provenance, augmentation choices, third-party attribution |
| [commit-hash-map.md](commit-hash-map.md) | hashes recorded before 2026-11-25 mapped to the current history |
| [scientific-design.md](scientific-design.md) (pt) | study design proposal (2026-10-15) |

## Data

| File | Content |
|---|---|
| [datasets.md](datasets.md) | data contracts and adapters |
| [dataset-acquisition-plan.md](dataset-acquisition-plan.md) (pt) | acquisition specification |
| [isic-acquisition.md](isic-acquisition.md) (pt) | ISIC 2020 acquisition |
| [rsna2018-acquisition.md](rsna2018-acquisition.md) (pt) | RSNA 2018 acquisition |

## Running campaigns

| File | Content |
|---|---|
| [infrastructure.md](infrastructure.md) | Git-coordinated Docker workers: quick start, resume, recovery |
| [global-cli.md](global-cli.md) | operator reference of the global CLI |
| [queue.md](queue.md) | Git queue protocol |
| [campaign-operations.md](campaign-operations.md) | serial driver and thermal watchdog |
| [stopped-attempt-recovery.md](stopped-attempt-recovery.md) | recovering a stopped worker attempt |
| [seed-requests.md](seed-requests.md) (pt) | requesting additional seeds |
| [fleet-protocol.md](fleet-protocol.md) (pt) | multi-node fleet protocol |
| [telemetry.md](telemetry.md) | operational telemetry |
| [global-next-integration.md](global-next-integration.md) (pt) | architecture specification of the development/evaluation pipeline |
| [global-development-contract.md](global-development-contract.md) | offline development and recipe freeze |
| [clinical-evaluation-integration.md](clinical-evaluation-integration.md) | trainer and runtime integration of clinical metrics |
| [x299-runbook.md](x299-runbook.md) (pt) | server runbook |

## Development history (October 2026)

Earlier technical acceptance, surveys and working notes, kept for traceability.
They describe the project before the final campaigns and may be outdated.

| File | Content |
|---|---|
| [acceptance.md](acceptance.md) | technical acceptance of the first remote smoke run |
| [acceptance-v2.md](acceptance-v2.md) | candidate-v2 technical acceptance |
| [endurance-test.md](endurance-test.md) | bounded GPU endurance test |
| [dataset-research.md](dataset-research.md) | dataset and augmentation survey |
| [dataset-protocol-update.md](dataset-protocol-update.md) (pt) | datasets and inexpensive HPO update |
| [protocol-draft.md](protocol-draft.md) (pt) | protocol proposal |
| [tasks.md](tasks.md) | implementation ledger |
| [cloud-handoff.md](cloud-handoff.md) | rules for code-only work |
| [cloud-context.md](cloud-context.md) (pt) | working context as of 2026-10-23 |
