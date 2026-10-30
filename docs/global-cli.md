# Global CLI — operator reference

`python -m augment_atlas.global_cli` is the operator glue of spec section 13 for
both binary datasets in scope (INbreast, ISIC 2020): the **external fixed
recipe** campaign (a), the **HPO** campaign (b) and the D/E pools generator.
The step-by-step server procedure, with exact commands, stop conditions and the
hashes to record, is [`x299-runbook.md`](x299-runbook.md); this page is the
command reference. Each subcommand reads private documents from
`<data-root>/global/<dataset>/<kind>-<sha256>.json`, pinned by hash. That is the
same layout the worker reads. Each subcommand re-audits its inputs and writes
with the no-clobber publisher. Standard output carries only hashes, counts,
commits and private paths. The CLI refuses to print a record ID, group or
image path, and it withholds error text that names one. Re-running a step with the same inputs is idempotent.

Nothing here trains or talks to Docker. `scientific_status` stays
`provisional`; `execution_verified` and `scientific_eligible` stay `false`.

## Common arguments

| Argument | Meaning |
|---|---|
| `--data-root` | private host data root (the worker's `settings.data`) |
| `--source` | JSON `{adapter, manifest, manifest_sha256}` of the private manifest |
| `--dataset` | dataset id, equal to `pools.dataset.id` |
| `--pools-sha256` | the approved pools hash |
| `--code` | the pinned code checkout; must be clean where a commit is recorded |
| `--remote --checkout [--branch queue]` | the Git queue |

## Order (campaign (a))

1. `identity`: record count and `records_sha256`. With `--pools-sha256`, it
   also prints the global manifest hash to compare with the pinned value (Q11:
   387 images).
2. `import-pools --file P --expected-manifest H`: seals the approved pools
   into the private store. Then `audit-pools --expected-manifest H`.
3. Optional, recommended: `anchor --kind pools --expected-manifest H --document-utc T`.
4. `seal-external --code C --image-id I --environment-probe PROBE.json
   --weights-sha256 W --final-seeds 11,13,19 --color grayscale_to_rgb
   --frozen-at T`: Q2 recipe, runtime = clean `HEAD` of `--code`.
5. `anchor --kind recipe_freeze --document-sha256 F` (C_freeze). For
   INbreast this must come **before** any HPO job of campaign (b).
6. `materialize-final --freeze-sha256 F --freeze-kind external_fixed --stages '11;13,19'`:
   writes `configs/global/<ds>/final/` (50 configs for stage 1).
   **Commit that directory** (commit X).
7. `seal-campaign` on the clean commit X, with the same freeze/stages,
   `--image-id I --sealed-at T` and the queue. It checks Q6
   (`same_code_identity_v1`: only `configs/global/**` changed since the
   freeze commit) and refuses if a final job of these pools already exists.
8. `anchor --kind campaign --document-sha256 K` (C_campaign). The printed
   `commit` is the worker's `settings.campaign_commit`; a private receipt
   `anchor-campaign-K.json` keeps it too.
9. `seed-final --campaign-sha256 K`: seeds the authorized jobs. The queue
   refuses them without the campaign anchor.
10. Workers `phase=final` run the jobs (outside this CLI).
11. `audit-chronology --campaign-sha256 K` at any time.
12. `report --campaign-sha256 K --runs RUNS --settings worker.json` does the
    following:
    - collects every authorized cell from the queue and the attempt
      directories (`global_collect`);
    - re-audits the chronology;
    - writes the private `report-<sha>.json` and `final-collection-<sha>.json`.

    By default the worker's full nine-artifact validation runs (`--settings`).
    `--host-validation minimal` (no settings) must be chosen explicitly; even
    then `predictions.json` and `results.json` must match the completion
    checksums. `telemetry/summary.json` is best effort and is read as data.
    Its hash and state go into the collection manifest. If it is unreadable
    or off-schema it becomes null; it never fails the report. Missing cells
    stay `missing`, never zero, and one contaminated artifact fails the whole
    report.
13. `summary --report-sha256 R` (public, ID-free), then `export-article
    --report-sha256 R --out DIR` (directory outside the repository), and
    `compare --report-a A --report-b B --out DIR` once both campaigns exist.

## Existing INbreast pools: conversion without resampling

The private 16/10 INbreast file uses global-development-proposal-v1, not
import-pools v1. Convert its fixed lists before importing:

~~~bash
python -m augment_atlas.global_cli convert-legacy-pools \
  --data-root "$INB" --source "$PRIV/inbreast-source.json" --dataset inbreast \
  --legacy "$LEGACY" --out "$PRIV/inbreast-pools-v1-candidate.json"
~~~

LEGACY must name the original private 16/10 proposal file; no new pools are
generated. The source must use inbreast_csv with a hash-pinned manifest and
per-image hashes, just as import-pools requires. Every legacy image path must
match a manifest path exactly. Conversion verifies D/E coverage exactly once,
whole-group closure in D/E, D_fit/D_select and every E test fold, D as the union
of its two subpools, and each original fold.train as D union (E minus test).
Repeated or unknown images, missing images and malformed metadata fail before
publication. Legacy diagnostics are not accepted as evidence; counts are derived
by the v1 auditor. Output is private and published without clobbering another
identity. Errors never interpolate supplied identifiers.

The output is an unsigned candidate with dataset.id inbreast, group-level
unverified linkage and prior_exposure known. The seeds must remain 17/29/31
and are metadata only. Its partition_provenance uses algorithm
legacy-proposal-v1-converted and records
converted_from={protocol: "global-development-proposal-v1", sha256: <hash of original bytes>}.
The exact supported legacy schema has no intended fractions or preregistered
thresholds. Each intended_fractions value is therefore the explicit string
not_preregistered_legacy, accepted only for this conversion algorithm;
thresholds are not required by import-pools and are not invented or applied
retroactively. Observed image/group counts are reported separately.

The converter labels dataset.version with its conversion algorithm (not a new
acquisition release). source_sha256 is the verified manifest byte hash;
labels_sha256 hashes the canonical list of {path, label} pairs. Missing
preprocessing/duplicate/exclusion history is represented by canonical hashes of
{status: "not_documented_legacy", aspect: <preprocessing|duplicates|exclusions>, legacy_sha256: <hash>}.
Those hashes bind explicit missing-evidence declarations, not verified processing
or duplicate/exclusion policies. The original byte hash retains provenance for
all legacy metadata, including sklearn and source_fold_sha256, without claiming
to have revalidated the original generator or its source fold file.

Review and approve the printed candidate_sha256, global_manifest_sha256, seeds,
converted_from and counts. Then use the candidate with import-pools --file
... --expected-manifest <approved global_manifest_sha256>, followed by
audit-pools and anchor --kind pools in runbook C0. The converted metadata
changes the global manifest identity: compare/re-pin it explicitly through the
coordinator, rather than substituting an old hash. Structural success still
reports contract_eligible false with prior_exposure_known and
patient_identity_unproven. The sealed campaign order and exposure guards in
runbook Part C remain required.

## New datasets: D/E pools (`make-pools`)

`make-pools --spec SPLIT.json --acquisition-config CONFIG.json --split-amendment AMENDMENT.json`
runs `global-dev-eval-v1`
(`dataset-acquisition-plan.md` section 5, `global_splits.py`), stdlib only.
The spec holds:
- the pools metadata: `dataset`, `task`, `grouping`, `provenance`, `source_sha256`;
- three seeds: `development`, `selection` and `evaluation`, where the
  evaluation seed must differ from the other two (Q3);
- `fractions {D, D_select_given_D}`;
- the pre-registered `thresholds {min_select_positive_groups,
  min_eval_fold_positive_groups}`.

The spec must reproduce the `splits` block of the pre-registered acquisition
config (algorithm, 5 folds, both fractions and both thresholds) **and** the
committed split amendment (seeds, fractions, thresholds; the amendment pins the
acquisition config identity). `split-spec --acquisition-root ROOT
--split-amendment AMENDMENT.json --out SPEC.json` writes the spec of a verified
acquisition from `READY.json` and `manifest-private.json`, hashes only. Nothing can be
relaxed at split time. The command writes an unsigned candidate to the private
store. It prints counts, hashes, the seeds and the sha256 of the spec and
config files, so they can be recorded next to the manifest hash. If any audit fails, it stops with `split_audit_failed`;
there is no seed retry and no fallback. After approval, run `import-pools
--file <path> --expected-manifest <global_manifest_sha256>`.

Choices not stated in section 5, **confirmed by Eduardo on 22/10/2026 (S2)**
and recorded in `configs/acquisition/isic2020-v1-splits-amendment-20261022.json`:
- **Quartile cut points:** nearest-rank, computed over all groups. D_select
  reuses those strata restricted to D; it computes no new quartiles.
- **Rounding:** exact half-up, `floor(f*n + 1/2)`. Section 5 writes `round`,
  and Python's banker's `round` gives a different count for f_sel = 0.30 when
  a stratum has 15, 35, 55, ... groups (5 instead of 4 for 15).
- **Fold round-robin:** one counter that continues across strata. Each
  stratum is still split within ±1 group per fold, and fold 0 gets no
  surplus. A restart in every stratum would give fold 0 up to one extra
  group per stratum.

Only binary tasks have a specified stratification; ordinal or multiclass
(mBRSET) needs a decision first.

**ISIC seeds (S1, decided 22/10/2026):** `configs/acquisition/isic2020-v1.json`
keeps its single `split_seed` because the `splits` block is part of the identity
pinned in the existing download ledger. The amendment declares
development = selection = 20261019 and evaluation = 20261022. The `splits/`
output of acquisition `verify` is a single-seed, pre-amendment diagnostic. It is
never imported.

## HPO path (campaign (b), decision B1)

| Step | Command |
|---|---|
| D-only view | `hpo-view`, `d-view --view-sha256 V --views-root VIEWS` |
| Search space (c00 = Q2 default, Q7) | `search-space --spec SPEC.json --color ...` plus `--code --image-id --environment-probe --weights-sha256` |
| Plan at the clean HEAD (commit P) | `build-plan --search-space-sha256 S --view-sha256 V --final-seeds a,b,c` plus the runtime flags |
| HPO configs | `materialize-hpo --search-space-sha256 S --plan-sha256 PL --view-sha256 V`, then commit `configs/global/<ds>/hpo/` (commit X) |
| Seed | `seed-hpo ... --image-id I --environment-probe PROBE` plus the queue flags, on the clean commit X |
| Collect | `collect-hpo ... --image-id I --environment-probe PROBE --configs-commit X --runs RUNS --views-root VIEWS --settings worker.json` |
| Freeze | `seal-freeze` with the same arguments as `collect-hpo`, plus `--evidence-sha256 EV --frozen-at UTC` |

Then `anchor --kind recipe_freeze --freeze-kind hpo_d` (with the collection arguments), `materialize-final
--freeze-kind hpo_d`, and `seal-campaign ... --search-space-sha256 S`, as for
campaign (a).

**B1 (Eduardo, 22/10/2026), option (i):** the HPO configs embed `plan_sha256`,
which covers `plan.runtime.commit`, so they are committed after the plan, at X.
`seed-hpo` and `collect-hpo` require the Q6 evidence from P to X
(`same_code_identity_v1`):
- P is an ancestor of X;
- only `configs/global/**` changed;
- the code tree is identical;
- the image is the pinned one, and every succeeded HPO attempt must record it in
  `runtime.json`.

The commit stays in every hash and identity.

`seed-hpo` also refuses in these cases:
- the committed configs differ from the plan's materialization;
- an `hpo_d` freeze of these pools is already anchored;
- on a dataset with `prior_exposure` other than `none_documented` (INbreast), the
  `external_fixed` campaign anchor is not yet in the queue (decision 7).

`seal-freeze` takes the winner from the predeclared ranking (mean, ties to
`c00`). No option chooses it. Three further guards close audit findings B-1
and B-1':
- `collect-hpo` refuses (`collection_incomplete`) while any HPO job of the plan
  is absent or not terminal.
- The **first complete collection of a plan is binding**. It writes the
  private pin `hpo-evidence-pin-<plan_sha256>.json`, and any later collection
  with different evidence is `collection_invalid`. This covers a retry or
  resume decided **after the first complete collection**. It does not cover
  a decision taken earlier with other scores already readable in the runs
  directory; see the open decision on retries below.
- A non-succeeded HPO attempt that already wrote `predictions.json` or
  `results.json` is `collection_invalid` (audit B-1''): a job stopped after
  writing its score never becomes a plain failed row.
- `seal-freeze` **and** `anchor --kind recipe_freeze --freeze-kind hpo_d`
  recompute the evidence from the current queue. They require it to equal
  the pin and the freeze's `selection_evidence_sha256`. An `hpo_d` anchor
  therefore takes the same collection arguments as `collect-hpo`.

**Open decision (retry policy):** whether to retry an infrastructure
failure before the first collection is still the operator's choice, made
while other scores are readable on disk. Removing that discretion requires a
rule Eduardo has not decided yet: mandatory retries up to a fixed K recorded
in the search space, or no HPO retries at all. Until then the runbook fixes
an operator rule.

The pin lives in the operator's private store. Like every private document,
it protects against mistakes and post-hoc choices, not against a repository
owner who deletes it (spec R9: flags stay false).

**Same image (Q6).** `seed-hpo`, `collect-hpo`, `seal-freeze` and
`seal-campaign` require `environment_sha256(--image-id, --environment-probe)`
to equal the `environment_sha256` that the plan or freeze recorded. A
different image or probe is `code_identity_mismatch`. `collect-hpo` also
requires the Q6 evidence it re-derives to equal the one `seed-hpo` stored.

**Pools (`make-pools`).**
- It reads the acquisition config and the split amendment with `git show
  HEAD:<path>` from a clean `--code` checkout, and requires both blobs to be
  identical to `origin/main` (run `git fetch origin` first). A `skip-worktree`
  edit, a local-only commit or a pushed feature branch is therefore refused.
  Local refs remain operator-controlled; the anchored pools record the seeds,
  so a reviewer can always compare them with the public amendment.
- It requires `--source` to be the `manifest.csv` that `READY.json` verified.
- It binds them to the `READY.json` config identity of `--acquisition-root`.
- It requires the spec to equal the derivation from that root.

**Exposed datasets (INbreast).** On a dataset with `prior_exposure` other
than `none_documented`, `seed-final` needs both campaign anchors
(`external_fixed` and `hpo_d`) before the first final job (decision 7).

**Q6 (confirmed by Eduardo on 22/10/2026):** `same_code_identity_v1` is
confirmed. The code token `Q6_POLICY_STATUS` still reads
`"pending_eduardo_confirmation"`, because three legacy tests pin that string
and old tests are never edited. The token is part of the hashed campaign
documents, so changing it needs an explicit decision to update those tests.

Workers run with `./scripts/worker.sh --phase hpo|final` (the CLI hook added for
B1). `hpo` needs `settings.views`; `final` needs `settings.campaign_sha256` and
`settings.campaign_commit`.

## Later declared stage authorization (C9b, decision 2026-10-29)

The original campaign continues to authorize stage 1 only. A separate, closed,
sealed global-stage-authorization-v1 document authorizes the declared later
stage. For stages '11;13,19', stage 2 is exactly [13,19], in that order.
It pins the original campaign SHA, pools, freeze/plan, image, environment/probe,
weights, analysis plan and stage-1 manifest without changing them. Its
execution_code_commit is an existing descendant commit whose entire original
Git tree is unchanged: every existing path/mode/object must match, including
src, scripts, docker, pyproject.toml, tests and documentation. Only the exact
new configs and MANIFEST under configs/global/<dataset>/final/stage-2/ may
be added. No tests are excluded from this identity policy. Stage configs clone
the sealed primary cells with only training.seed changed.

The document records decided_on: "2026-10-29" (no decision time is invented),
decided_after_primary_results: true and a rationale explicitly saying
Eduardo decided after viewing the INbreast primary results. The separate
sealed_at_utc records the actual time the authorization document is sealed.
It explains that SAP item 5's "training seed 1" means the first seed declared
(11); the sealed SAP itself is unchanged.

Decision provenance has two routes and is never defaulted. A decision taken
after viewing primary results (the INbreast case) uses this manual route,
`authorize-stage --decided-after-primary-results true`. The seed-request route
(X5, [seed-requests.md](seed-requests.md)) keeps refusing
`decided_after_results: true` (SAP item 5: time/budget only); its
RealStageAuthorizer seals the request's own `decided_after_results` value as
`decided_after_primary_results`.

Gates added after review R1 (`stage_gates.py`; the explicit A1 exception below supersedes the completion rule):

- `authorize-stage` takes the queue (`--remote/--checkout`) and refuses with
  `stage1_incomplete` unless every authorized stage-1 cell of the campaign
  (all arms x folds of seed 11) exists and is `succeeded` (SAP item 5). The
  stage-1 snapshot at the parent of the authorization anchor is therefore the
  final primary state.
- Publish stage authorizations only after every primary worker has finished;
  stage-2 workers must run the new orchestrator. Workers of the 835c70e lineage
  validate anchors with the three-kind schema and stop at `anchor_invalid` once
  a `stage_authorization` anchor exists. `anchor --kind stage_authorization`
  and the RealStageAuthorizer refuse with `primary_workers_active: <job IDs>`
  while any primary final job of ANY dataset or campaign on the shared queue
  is pending, claimed, running or interrupted. The check runs on the exact HEAD
  the anchor commit is built on. Jobs of an already anchored later stage (same
  campaign and execution commit) do not block. Republishing an existing anchor
  is a no-op and is not gated. Once any stage anchor exists, every worker on
  that queue must run the new orchestrator. A primary retry or a primary
  seed-final after that point is visible to older workers and stops them.
- Stage eligibility is decided by seeds, never by the config path. With the
  sealed campaign, `seed_final` admits only its `matrix.authorized_seeds`
  (`campaign_invalid` otherwise; `seed-final` always passes the audited
  campaign). The audited claim context carries the campaign even without a stage
  pin. Restart/resume revalidates every global final job before touching its
  container. The queue API without a campaign remains anchor-only (the anchor
  carries no seeds); the chronology audit and the runtime gate then reject
  unauthorized seeds before any claim by the orchestrator.

Run the following for EACH campaign in its own original code checkout. The
ORCHESTRATOR names the checkout containing C9b; its CLI package must be available in the operator/worker
orchestrator environment; do not add orchestration changes to the campaign's
pinned trainer tree. Reuse the existing source, queue, paths and hashes.
Replace the operator variables below with their real values. No command here
starts training.

~~~bash
# Existing operator values:
# DATA_ROOT SOURCE DATASET POOLS CODE REMOTE QUEUE CAMPAIGN FREEZE KIND IMAGE PROBE RUNS SETTINGS ORCHESTRATOR
# KIND = external_fixed or hpo_d.
# CODE starts at the original campaign.runtime.code_commit with a clean tree.
g() {
  cmd="$1"; shift
  PYTHONPATH="$ORCHESTRATOR/src" python -m augment_atlas.global_cli "$cmd" --data-root "$DATA_ROOT" --source "$SOURCE" \
    --dataset "$DATASET" --pools-sha256 "$POOLS" --code "$CODE" "$@"
}
q() { g "$@" --remote "$REMOTE" --checkout "$QUEUE"; }

# Preparation only: adds 100 configs + separate manifest; grants no execution.
g materialize-final --freeze-sha256 "$FREEZE" --freeze-kind "$KIND" --stages '11;13,19' \
  --campaign-sha256 "$CAMPAIGN" --stage 2
# Operator creates this config-only commit; original campaign/MANIFEST stay intact.
git -C "$CODE" add -- "configs/global/$DATASET/final/stage-2"
git -C "$CODE" commit -m "Add declared stage 2 sensitivity configs"
EXECUTION=$(git -C "$CODE" rev-parse HEAD)

# Actual seal timestamp, distinct from the decision date.
SEALED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# Refused (stage1_incomplete) until all 50 stage-1 cells are succeeded.
q authorize-stage --campaign-sha256 "$CAMPAIGN" --stage 2 \
  --execution-code-commit "$EXECUTION" --image-id "$IMAGE" --environment-probe "$PROBE" \
  --decided-on 2026-10-29 --decided-after-primary-results true \
  --rationale 'Eduardo decided on 2026-10-29 after viewing the INbreast primary results; seeds 13 and 19 are predeclared sensitivity only. Primary training seed is 11.' \
  --sealed-at "$SEALED_AT"
# Set AUTH to the stage_authorization_sha256 printed above (private store outside CODE).
# Requested flow: authorization -> idempotent materialization -> anchor -> stage-only jobs.
g materialize-final --freeze-sha256 "$FREEZE" --freeze-kind "$KIND" --stages '11;13,19' \
  --campaign-sha256 "$CAMPAIGN" --stage 2 --stage-authorization-sha256 "$AUTH"
# Refused (primary_workers_active) while any primary job of the shared queue is not terminal.
q anchor --kind stage_authorization --document-sha256 "$AUTH" \
  --image-id "$IMAGE" --environment-probe "$PROBE"
q seed-final --campaign-sha256 "$CAMPAIGN" --stage-authorization-sha256 "$AUTH"
q audit-chronology --campaign-sha256 "$CAMPAIGN" --stage-authorization-sha256 "$AUTH"
for scope in stage1 all_authorized; do
  q collect --campaign-sha256 "$CAMPAIGN" --stage-authorization-sha256 "$AUTH" \
    --scope "$scope" --runs "$RUNS" --settings "$SETTINGS"
  q report --campaign-sha256 "$CAMPAIGN" --stage-authorization-sha256 "$AUTH" \
    --scope "$scope" --runs "$RUNS" --settings "$SETTINGS"
done
~~~

The stage worker settings keep the original campaign_sha256 and
campaign_commit, add stage_authorization_sha256: AUTH, and point code at
a clean checkout of EXECUTION. Use exactly the original immutable image,
probe environment, data, views, weights and runs root. Stage-1 workers still
require the original campaign commit. The orchestrator validates the stage
pin and Git identity before each claim and container creation; the queue CAS
claim rechecks chronology against the exact parent HEAD.
Restart/resume uses the same settings, job/attempt and pinned checkout.
A new seed-final call is idempotent and creates only stage-2 jobs, never seed 11.

After authorization, pass its hash to chronology, collection and report.
Primary workers can discover an anchored authorization in their private data root
and audit it without an execution pin. This observation does not authorize them to
execute later seeds. Explicit stage pins remain mandatory for sensitivity workers.
A later-stage job created before or in the authorization anchor commit makes
the campaign invalid; timestamp guesses cannot repair Git ancestry.

The default --scope stage1 uses the primary queue snapshot immediately
before the authorization anchor, including its chronology HEAD. This preserves
the existing report and collection bytes even after sensitivity jobs/claims
change the queue. Because authorize-stage requires the complete, succeeded
stage-1 matrix, that snapshot is the final primary state; a primary retry after
the anchor would not be incorporated (and would stop older workers). The complete
current queue is still audited before selecting this snapshot. The separate
--scope all_authorized report/collection includes all 150 authorized cells,
uses current chronology, adds explicit sensitivity scope and authorization
provenance, and aggregates folds then seeds. It does not replace the primary.
Launch a sensitivity worker from the updated host orchestration checkout,
while SETTINGS.code continues to name the separate, clean EXECUTION checkout:

~~~bash
"$ORCHESTRATOR/scripts/worker.sh" --settings "$SETTINGS" --phase final
# After an interruption, use the same pins/settings and the established --resume procedure.
~~~

Current calls take one later-stage authorization document; authorizing several
later stages simultaneously is outside this C9b flow.

### Parallel stage 1 and stage 2 (SAP amendment A1, 2026-10-30)

[A1](analysis-plan-sap-v1-amendment-20261030.md) permits the predeclared later
seeds to run while the same campaign's primary matrix is still incomplete.
The default authorization still refuses with `stage1_incomplete`. The manual
exception is `authorize-stage --allow-incomplete-stage1 "<amendment id and reason>"`.
Empty text is refused. A sealed `stage1-overlap-<authorization-sha>.json` record
in the private data root binds this declaration to the campaign and authorization;
the public anchor and authorization schemas are unchanged. Preserve that record.

After preparing the additive stage configs as above, follow this order:

1. Between jobs, stop and restart the campaign's **seed-11 workers** using the
   A1 orchestrator. Confirm every worker reading the shared queue is upgraded.
   Older workers stop at the unknown `stage_authorization` anchor kind of their
   own dataset. Keep the primary trainer checkout at `campaign.runtime.code_commit`,
   without a stage execution pin. Its Git object store must contain the execution
   commit, and its private data root must contain the anchored stage authorization
   so the new orchestrator can audit it. Updating orchestration does not change
   the pinned trainer tree or image.
2. Seal and publish with **both explicit declarations**. The A1 flag goes on
   authorization; the upgrade declaration goes on anchor publication. Use the
   actual decision provenance and seal time. For a different node, declare
   `--cross-environment` with that node's probe and the same image. Each later
   stage runs entirely on that node.

   ~~~bash
   q authorize-stage --campaign-sha256 "$CAMPAIGN" --stage 2 \
     --execution-code-commit "$EXECUTION" --image-id "$IMAGE" --environment-probe "$PROBE" \
     --cross-environment --decided-on "$DECIDED_ON" \
     --decided-after-primary-results "$AFTER_RESULTS" --rationale "$RATIONALE" \
     --sealed-at "$SEALED_AT" --allow-incomplete-stage1 'SAP v1 amendment A1: predeclared sensitivity on another node while primary jobs finish'
   # Set AUTH to the authorization hash printed above.
   # Before publishing, make this private authorization and the execution Git
   # objects available to the primary workers as described in step 1.
   q anchor --kind stage_authorization --document-sha256 "$AUTH" \
     --image-id "$IMAGE" --environment-probe "$PROBE" \
     --declare-all-workers-upgraded "$UPGRADE_DECLARATION"
   ~~~

   The publication gate checks the private A1 record and the upgrade declaration
   on the exact parent HEAD. Without both, active primary jobs of the same
   campaign still block. D9 alone continues to release only other campaigns.
   The private D9 release record lists every released job, including own primary
   jobs released under A1. The seed-request route gains no A1 exception.
3. Seed stage 2, then audit; start sensitivity workers with the authorization
   pin and the clean execution checkout. Primary workers retain their original
   commit and continue claiming seed 11 while the audited stage-2 jobs coexist.

   ~~~bash
   q seed-final --campaign-sha256 "$CAMPAIGN" --stage-authorization-sha256 "$AUTH"
   q audit-chronology --campaign-sha256 "$CAMPAIGN" --stage-authorization-sha256 "$AUTH"
   ~~~

4. When every primary cell has a final outcome, collect and report with
   `--scope stage1_current`. It selects only stage-1 seeds from the exact current
   audited HEAD, including cells completed after authorization. Absent, pending,
   claimed, running or interrupted primary cells cause `stage1_incomplete`;
   failed/cancelled final outcomes remain explicit failures. The full chronology
   audit includes the later jobs. The legacy `--scope stage1` still selects the
   anchor-parent snapshot and preserves its bytes; under A1 that snapshot can
   be partial and must not be used for the final primary analysis.
5. Collect/report sensitivity separately with `--scope all_authorized`.

~~~bash
for scope in stage1_current all_authorized; do
  q collect --campaign-sha256 "$CAMPAIGN" --stage-authorization-sha256 "$AUTH" \
    --scope "$scope" --runs "$RUNS" --settings "$SETTINGS"
  q report --campaign-sha256 "$CAMPAIGN" --stage-authorization-sha256 "$AUTH" \
    --scope "$scope" --runs "$RUNS" --settings "$SETTINGS"
done
~~~

The sealed SAP and `analysis_sap_v1.py` are unchanged. Seeds 13/19 remain
sensitivity only and never enter the primary analysis.


Seed requests can use the bundled RealStageAuthorizer; see [seed-requests.md](seed-requests.md).
By default authorize-stage requires the campaign probe environment. An explicit
--cross-environment declaration permits another probe environment while keeping
the image identical. The authorization pins the supplied environment; anchor and
worker checks use that pin. The all_authorized sensitivity report labels those
seeds "different environment (declared)". The default stage1 report is unchanged.
