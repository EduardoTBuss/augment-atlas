# Cloud handoff — code-only work

This repository is public. Everything here is code, tests and documents. Private
data, servers, credentials and campaign results do not exist for you and must
never be added.

## Hard rules

1. Code and offline tests only. Never run or request HPO, training campaigns,
   data acquisition, or anything that needs a GPU or private data.
2. `scientific_status` stays `"provisional"`; `execution_verified` and
   `scientific_eligible` stay `false` everywhere.
3. Never edit or weaken an existing test to make it pass. A failing old test is a
   finding to report, not something to adapt. Legacy job IDs must not change.
4. New logic goes in new modules; changes to `train.py`, `runtime.py`,
   `contracts.py`, `evaluation.py`, `cli.py` are small dispatch hooks only.
5. Tests are stdlib `unittest`. Container/torch tests are skipped when torch is
   absent; a skip is not a pass, so say how many were skipped.
6. Commits: author is the configured git user, no co-author or session trailers.
7. Do not push to `main`. Work on the branch you were given and open a PR.
8. Mutated test fixtures must be re-signed (hash recomputed) so they cannot pass
   through a checksum.

## Spec

`docs/global-next-integration.md` (section 9 lists the slices, section 12 the
decisions Q1–Q13). Also read `docs/global-development-contract.md`,
`docs/dataset-protocol-update.md`, `docs/clinical-metrics.md`.

Protocol in one paragraph: one HPO per dataset, without augmentation, only on a
development partition D (~20%); recipe frozen and sealed before E (~80%) is
opened; E split into 5 fixed grouped folds; each fold trains a fresh ImageNet
model on D ∪ (E minus test fold) with the frozen recipe; 10 augmentation arms
share folds and recipe. D is never a test set.

## State of the work

Current status, decisions pending and next steps: [`cloud-context.md`](cloud-context.md)
(updated at the end of every session).

| Slice | Branch / state |
|---|---|
| S0, S1a/b/c, S3 | done, `feature/global-protocol-20261016` |
| S4 report + article export, train telemetry | done and reviewed, `wip/s4-telemetry-20261021` |
| S2 HPO producer (`global_hpo.py`) | done, but the HPO path cannot run end to end until decision B1 (`cloud-context.md` §5) |
| AVX2 CPU limit | done, `fix/cpu-isa-avx2-20261020` (`cpu_isa_max`, default AVX2) |
| ISIC acquisition tooling | done, `feature/isic-acquisition-20261019` |
| Fleet tooling | done, `feature/fleet-20261019` |
| Integration (task C) | done, PR #3 (`claude/tarefa-c-merge-branches-pwmlw1`) |
| Global CLI, final collector, D/E pools generator | done, PR #4 (`claude/magical-bardeen-vkyubk`) |
| S6 multilabel | out of scope for now |

Merge rule (Q6): HPO and final jobs run from the same code commit, so S2, S4,
telemetry and the AVX2 limit must all land in one tree before any real job.

## Tasks

- **A. Finish S4 + telemetry.** Run the suite, finish what is incomplete
  (`train.py` telemetry patch, doc edits), make every test pass without touching
  old tests. Then request an adversarial review (see below) and fix findings.
- **B. S2 producer.** Implement `global_hpo.py` per section 9 S2 and every
  adversarial case listed there (hpo-view containing an E id, ledger row with E in
  `train_ids`, search-space changed after seeding, declared metric differing from
  recomputed, missing job as explicit `missing`, exact tie -> `c00`, row
  permutation invariance, no D-view mount -> `hpo_isolation_unproven`).
- **C. Integration.** Merge the four feature branches into one tree, resolve
  conflicts without changing behavior, run the full suite, report the counts.

## Done means

All new and old tests pass (state passed/skipped counts), an independent
reviewer found no unresolved blocker, each spec adversarial case has a test, and
a PR exists with a summary of what changed and what could not be verified.

## Loop

implement -> run tests -> an independent reviewer (fresh context, read-only)
checks the diff against the spec and the hard rules -> fix findings -> repeat.
Stop at 6 iterations or when done, whichever is first; if stopped early, report
the open findings instead of continuing.
