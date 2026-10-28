# Offline global development and recipe freeze, v1

This module imports **already fixed** scalar-label records and D/E lists. It
neither generates splits nor runs HPO, loads images, changes smoke config
validation, or enables a scientific trainer/queue phase. All documents remain
`scientific_status="provisional"`. Tests are manually specified synthetic
fixtures, not evidence of an experiment or a real evidence producer.

The supplied Brain context was
`projects/augment-atlas-datasets-hpo-economico-2026-10-16.md` and the coordinator's
same-date resumption bundle (19:00 BRT). Decisions preserved: one future NONE
search per dataset, ten intervention arms, all adaptive choices confined to D,
recipe/epochs/thresholds fixed before five-fold evaluation on E, and fresh
ImageNet initialization for final training on D union (E minus the test fold).
Current INbreast filename groups do not establish PatientID, and the prior
pilot exposure is known. No real freeze, HPO campaign, data acquisition or
patient-independent confirmatory result is established here.

## API and trust boundaries

`src/augment_atlas/global_development.py` uses stdlib plus the existing
stdlib-only `contracts` module. All audit APIs leave input values unchanged.
Structural/schema/checksum/anchor violations raise `ValueError`. The return
values contain derived diagnostics, never caller-approved scientific validity.

```python
global_manifest_sha256(records, document) -> str
audit_global_pools(records, document, *, expected_manifest_sha256) -> dict
seal_global_pools(records, candidate, *, expected_manifest_sha256, output) -> dict
final_fold_view(records, pools, fold, *, expected_pools_sha256) -> dict
audit_recipe_freeze(records, pools, freeze, evidence, *,
                    expected_pools_sha256, expected_plan_sha256) -> dict
seal_recipe_freeze(records, pools, candidate, evidence, *,
                   expected_pools_sha256, expected_plan_sha256, output) -> dict
```

`seal_*` returns the persisted document with a canonical `sha256`; `audit_*`
returns a report. The caller must obtain the expected hashes from its approved
contract. Computing an expected hash from untrusted input at audit time provides
no external approval. The manifest helper is for initial approval, not a way to
bypass it. Its identity binds canonical records **and all metadata**, including
dataset/version, source hash, task/class universe, grouping, preprocessing,
labels/exclusions/duplicate provenance and declared generator/seeds/fractions.
The separate pools anchor binds the actual lists/folds as well.

The freeze's plan is externally anchored. Every selected parameter, epoch,
threshold, runtime identity and final policy must agree with that plan and its
deterministic winner. A different timestamp/evidence/recipe creates a different
freeze identity. This API has no `expected_freeze_sha256` argument: future
consumers must pin the returned approved freeze hash **before opening E** and
compare it externally. An offline timestamp/checksum cannot establish actual
chronology, patient identity, absence of prior exposure or actual image access.

## Global pools schema

Schemas are closed and versioned; integers reject booleans, JSON numbers must
be finite, hashes are lowercase 64-hex strings, and lists are sorted/unique.
Records have exactly `id`, `path`, `group`, `label`; labels are nonnegative
integers. Paths must be normalized relative POSIX paths without traversal,
drive prefixes or backslashes. IDs and paths must each be unique. Records may
be presented in any order; canonical order is by ID. Image bytes are never
opened, and visual duplicates/physical ingestion are outside this module.

`global-pools.json` contains exactly:

- `schema_version=1`, `protocol="global-development-holdout-v1"`,
  `scientific_status="provisional"`, `dataset={id, version}`.
- `records_sha256` (canonical records sorted by ID), `source_sha256` (anchored
  declaration of source file bytes), `provenance` with
  `preprocessing_sha256`, `labels_sha256`, `duplicates_sha256`,
  `exclusions_sha256`.
- `grouping={namespace, source, identity_level, linkage_status,
  evidence_sha256, prior_exposure}`. Identity level is `group` or `patient`;
  patient requires `verified` linkage and a non-null evidence hash.
  `prior_exposure` is `known`, `unknown`, or `none_documented`. INbreast v1
  explicitly requires `known`; a new checksum cannot erase the known pilot.
- `task={type, classes, positive_class, endpoint, decision_rule,
  missing_policy, requires_patient_identity}`. Classes remain ordered and
  predefined. Supported type/endpoint pairs are `binary`/`binary_auroc`,
  `ordinal`/`quadratic_weighted_kappa` and `multiclass`/`macro_auroc_ovr`.
  Binary requires two classes and a declared positive class; other tasks use
  null for it. `missing_policy="undefined_no_drop"`. Decision rule is
  `scores_only`, `argmax`, or binary-only `fixed_threshold`; QWK uses `argmax`.
  Multilabel is explicitly rejected until there is a record adapter and
  uncertainty policy. This module computes no clinical metric.
- `partition_provenance={algorithm, version, source, generator_version,
  seeds, intended_fractions}`. Seeds explicitly name `development`,
  `selection`, `evaluation`; intended fractions explicitly name `D`,
  `D_fit_given_D`, `E_test_given_E`. No seed-stage association is inferred,
  generation retried or score used to choose a split. The audit validates
  supplied assignments; it cannot reconstruct undocumented generation from a
  source-hash assertion.
- `lists={D, D_fit, D_select, E}` and ordered `evaluation_folds`, exactly five
  objects `{fold, test_ids}` with indices 0..4; canonical `sha256` excludes
  only itself.

For the INbreast 16/10 conversion only, partition_provenance additionally
requires the closed object converted_from={protocol, sha256} with protocol
global-development-proposal-v1 and the SHA-256 of the original bytes. Its
algorithm is legacy-proposal-v1-converted, version 1. Only that algorithm,
with dataset.id inbreast, permits the string not_preregistered_legacy in
intended_fractions; ordinary v1 candidates still require numeric fractions.
The converter sets all three fractions to that marker because the supported
legacy schema has no registered fractions. Thresholds are generator inputs,
not mandatory pools fields; none are fabricated. See [the CLI reference](global-cli.md#existing-inbreast-pools-conversion-without-resampling)
for the explicit hashes of missing historical preprocessing/duplicate/exclusion
evidence and the manifest/label hash definitions. These declarations and the
original file hash are included in the externally approved global manifest
identity. This scoped representation does not promote scientific eligibility.

D/E cover all eligible records once and are disjoint by ID **and group**.
D_fit/D_select partition D under the same rule. Tests partition E once and
are disjoint by group; D never appears in any test. Mixed-label groups retain
their individual image labels. Every fold view derives `final_train_ids` as
D union (E minus E_f) and `test_ids` as E_f, without any E selection list or
executable training config.

Counts are deliberately derived in audit reports rather than accepted as
persisted self-approved diagnostics: images, groups, class counts and
class-group counts over the full predefined universe, mixed groups, absent
classes and endpoint feasibility in all pools and tests. This is the v1
representation of effective counts. `endpoint_feasible` checks class-presence
prerequisites only; it does not guarantee that a future QWK or AUROC computation
will be defined for actual predictions. Missing classes preserve zero counts
and disable contract eligibility, rather than shrinking a macro denominator.

`eligibility.contract_eligible` additionally requires the declared patient
identity when requested and `prior_exposure="none_documented"`. A valid
group-level fixture may still be structurally valid while ineligible for a
patient-level protocol. No hash proves that a declaration is true.

## Selection plan, evidence, and recipe freeze

`recipe-freeze.json` has exactly `schema_version=1`,
`protocol="global-recipe-freeze-v1"`, `scientific_status="provisional"`,
`dataset`, `task`, `pools_sha256`, `selection_plan`, `selection_plan_sha256`,
`selection_evidence_sha256`, `selected_candidate_id`, `selected_parameters`,
`frozen_recipe`, `final_policy`, `runtime`, `selection_scope`,
`frozen_at_utc`, derived `eligibility`, canonical `sha256`.

The plan, pinned independently **before scores**, contains:

- Integer `schema_version=1`, identical `dataset`, `task`, `pools_sha256`,
  primary `endpoint`, `direction="maximize"`, complete `candidates` and
  explicit unique selection `seeds`.
- `ranking="mean_then_candidate_id"`: rank the arithmetic mean of all
  defined seed results descending, then candidate ID ascending. Seeds/folds
  are not independent clinical replicates. Aggregation uses `math.fsum` in
  canonical seed order, so ledger row permutations cannot change a tie or
  the selected recipe.
- `failure_policy="require_all_defined"` or
  `"exclude_candidate_on_any_nondefined"`. The former blocks eligibility if
  any trial is nondefined. The latter excludes the entire candidate, never
  averages only its successful seeds. At least one fully defined candidate
  is needed to select a recipe. Neither policy silently turns missing scores
  into zero.
- `epoch_policy` and `threshold_policy`, both
  `"fixed_in_candidate_before_scores"`. This minimum version supports
  complete fixed epoch/threshold alternatives among predeclared candidates;
  it does not silently inherit smoke's selection-loss epoch choice or
  implement a best-epoch sub-ledger/calibration fit.
- `producer={id, code_sha256, environment_sha256, manifest_sha256,
  logs_sha256}`, `runtime={commit, environment_sha256,
  imagenet_weights_sha256}` and the complete `final_policy`. The producer
  references identify declared provenance. They are not a signature, a
  trusted real producer implementation or proof that referenced logs exist.

Each candidate has exactly `id`, explicit `parameters`, full `recipe`, and
`augmentation={arm:"none", ops:[]}`. Its full configuration hash includes all
four fields. `recipe` includes `model={name:"resnet50",
weights:"IMAGENET1K_V2", weights_sha256}`, `input={size,color}`,
`preprocessing_sha256`, `optimizer={name,parameters}`,
`scheduler={name,parameters}`, positive integer `batch_size`,
`accumulation_steps`, `epochs`, `loss={name,parameters,learned_on:"D_fit"}`,
and `prediction={rule,threshold_policy,thresholds}`. Input size is at least 32;
color is `rgb` or `grayscale_to_rgb`. Hashes must match the pinned pool/runtime.
Score-only and argmax use `not_applicable` and empty thresholds; binary fixed
threshold uses `fixed_before_E` and one finite threshold in [0,1]. Secondary
threshold outputs/calibration are not implemented; expanding that schema
requires a new reviewed contract before E access. Optimizer/scheduler/loss
parameter objects are finite JSON declarations, not executable adapters.

Evidence is supplied separately and retained separately by its caller. It has
`schema_version=1`, `kind`, identical `producer`, `dataset`, `pools_sha256`,
`plan_sha256`, `task`, `rows` and canonical `sha256`. Kinds allowed in this
slice are `synthetic_fixture` and `external_unverified`; there is no integrated
trusted real producer. Each row in the complete candidates x seeds ledger has
exactly:

- `candidate_id`, `seed`, `config_sha256`, exact `train_ids=D_fit`,
  `select_ids=D_select`, `epoch_ids=D_select`, `threshold_ids=D_select`.
- `artifact_sha256`, `log_sha256`, `status`, `metric`, `reason`.
  `status="completed"` with defined score requires an artifact hash, a log
  hash and null reasons. Scores are finite, within [0,1] for AUROC or [-1,1]
  for QWK. `metric={endpoint,state,value,reason}` cannot change the endpoint.
- A completed but undefined result uses `state="undefined"`, null score and
  explicit trial/metric reasons. `failed`, `timeout`, and `missing` use
  `state="unavailable"`, null score and reasons. Their artifact hash may be
  null, but a producer log/accounting reference is still required. Missing
  combinations must be explicit rows. Duplicate or omitted rows are rejected.

The `selection_scope` repeats exact train/select/epoch/threshold IDs, NONE
augmentation and the pinned producer. An `E_accessed=false` assertion cannot
replace these links. The auditor checks every possible in-memory binding and
checksum. It does **not** fetch external raw files, reconstruct raw scores or
verify real log contents/IO from their hash references. Supplying a different
externally fabricated evidence payload is not cryptographically impossible;
that is precisely why execution and scientific eligibility stay false. A
future producer integration must validate raw artifacts and access boundaries
before a real freeze can be approved.

`final_policy` fixes ten `contracts.ARMS`, indices 0..4, exactly three distinct
seeds, `augmentation_presets_sha256`, fresh ImageNet weights per arm/fold/seed,
`warm_start=false`, `training_pool="D_union_E_minus_test"`,
`early_stopping=false`, `checkpoint_selection="last_epoch"`,
`selection_on_E=false`. `scheduler_duration` is
`fixed_epochs_recompute_steps_from_final_pool`, and `step_count` explicitly
fixes `unit="optimizer_update"`, `drop_last=false`,
`accumulation_tail="step"`, and
`per_epoch="ceil(ceil(train_images/batch_size)/accumulation_steps)"`.
The audit derives every fold's train image count, updates per epoch and total
updates for the fixed epoch count. This records scheduler duration without
selecting anything on E or constructing an executable config.

`frozen_at_utc` requires an explicit UTC ISO timestamp. The sealer derives
eligibility and the auditor rejects self-approved eligibility fields.
`contract_eligible` means only structural/declared-evidence readiness.
`scientific_eligible=false` and `execution_verified=false` **always** in v1,
with `no_integrated_evidence_producer` and the evidence kind reported. This
keeps synthetic validation distinct from HPO execution and clinical inference.

## Persistence and validation

Store documents only in a private directory: identifiers and group keys can
be sensitive. There is no public queue integration. Seal pools first, approve
their hash, then seal the freeze. No transaction regenerates pools when a
freeze is absent.

Publishing writes a private unique temporary file with the existing
`atomic_json`, then publishes its complete bytes via exclusive `os.link`.
This needs filesystem hard-link support (tested on the current Windows
workspace); unsupported filesystems fail without a weaker overwrite fallback.
`os.replace` is never used on the requested destination. Existing/colliding
targets are parsed and audited; only identical canonical content is accepted.
Conflicts/corruption leave original bytes intact. An identical retry preserves
the original bytes. A complete published file survives interruption; stale
private temporary files after a process kill are not treated as evidence or
as a published identity.

The manual fixtures are in `tests/test_global_development.py` and
`tests/test_recipe_freeze.py`. Their in-memory identities and expected hashes
are reproducible from `pool_fixture()` / `freeze_fixture()`; generated JSON
files live only in temporary directories. On a checkout without an installed
package, use `PYTHONPATH=src`:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -p test_global_development.py -v
python -m unittest discover -s tests -p test_recipe_freeze.py -v
python -m unittest discover -s tests -p test_scientific_splits.py -v
python -m unittest discover -s tests -p test_selection_only.py -v
```

Tests include recomputed-hash mutations for leakage, sibling group splitting,
coverage, external identity/plan anchors, candidate/endpoint replacement,
ledger accounting, NaN/Infinity, fixed duration, NONE augmentation and patient
identity. Audit/derivation also run with ML imports blocked and image/process
access made unavailable. The existing selection-only/global-development
rejection and legacy full-population coverage remain in their original tests.
Real toy trainer tests retain their existing container-only gate; a skip is
not a pass. No real data, training trial, final evaluation, campaign or remote
infrastructure operation is part of this validation.
