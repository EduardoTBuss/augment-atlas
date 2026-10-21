# Clinical evaluation opt-in: trainer and runtime integration

Status: **technical contract, provisional**. This integration computes
image-level point estimates on synthetic or pilot data and checks artifact
consistency. It provides no clinical validation, no confidence intervals and no
patient-level claim. Every result it writes carries
`scientifically_eligible: false` and `execution_verified: false`. The metric
primitives are documented in [clinical-metrics](clinical-metrics.md); D/E pool
and recipe-freeze foundations in [global-development-contract](global-development-contract.md).

## 1. Opt-in configuration (`evaluation`, schema 1)

The key is optional. Without it the trainer, artifacts, recipe identity and
runtime validation are unchanged (eight artifacts, legacy hashes). With it,
the object must contain exactly the fields of its task; no default is injected
and unknown or irrelevant fields are rejected.

| Field | binary | multiclass | ordinal |
|---|---|---|---|
| `schema_version` | `1` (int) | `1` | `1` |
| `task` | `"binary"` | `"multiclass"` | `"ordinal"` |
| `classes` | 2 distinct labels | `num_classes` distinct labels | `num_classes`, ordered, equally spaced |
| `score_kind` | `"softmax_probability"` | same | same |
| `frozen` | `true` (declaration only) | same | same |
| `positive_class` | member of `classes` | — | — |
| `threshold` | finite number in [0,1] | — | — |
| `comparison` | `"ge"` or `"gt"` | — | — |
| `tie_break` | — | `"first"` or `"error"` | `"first"` or `"error"` |

`classes[i]` is the label of model output column *i* and of manifest target
index *i*. Labels are nonempty strings or integers (never bool or float).
`task: "multilabel"` is rejected: the scalar softmax trainer cannot produce it.
`frozen: true` records that the decision was fixed before evaluation; it is
not a scientific approval.

```json
"evaluation": {"schema_version": 1, "task": "binary", "classes": [0, 1],
               "score_kind": "softmax_probability", "frozen": true,
               "positive_class": 1, "threshold": 0.5, "comparison": "ge"}
```

Endpoints: binary AUROC (binary), macro one-vs-rest AUROC (multiclass), QWK
(ordinal). Secondary: macro-F1 and balanced accuracy for every task, plus
non-interpolated average precision for binary. Undefined values are explicit
`value: null` with a `reason`, coverage and per-class detail; nothing is
silently dropped or zero-filled.

Identity: `recipe_identity(cfg)` (now defined in `contracts`, re-exported by
`cli`) includes `evaluation` only when the key exists, so every evaluation
decision (classes, positive class, threshold, comparison, tie policy) changes
the frozen recipe hash while legacy recipe hashes stay byte-identical.

## 2. Trainer behaviour

- Epoch selection is unchanged: first epoch with strictly minimal selection
  loss. No clinical metric is computed during training and no test value
  influences `best_epoch`.
- After selection, the best model performs **one** final pass over the scope's
  partition (`test` by default, `selection` when
  `training.evaluation_scope == "selection-only"`), collecting softmax
  probabilities in persisted split order. Non-finite logits fail the run.
- Selection-only never constructs or reads a test dataset (integrity hashing of
  manifest files is unchanged and may still read test bytes).
- An interruption during the final pass writes no `predictions.json`; a resume
  re-runs the final pass from the checkpointed best model.
- A `tie_break: "error"` tie fails the run explicitly.

## 3. Artifacts

| Artifact | Legacy | Opt-in | Visibility |
|---|---|---|---|
| `checkpoint.pt`, `best.pt`, `splits.json`, `config.json`, `environment.json`, `metrics.json`, `results.json`, `progress.json` | required | required | private attempt output |
| `predictions.json` | absent | **required (9th)** | **private: row level** |
| `completed.json` | checksums of 8 | checksums of 9 | private |

`contracts.required_artifacts(cfg)` returns the list; `BASE_REQUIRED_ARTIFACTS`
is the legacy eight.

`predictions.json` (`artifact_type: clinical_predictions`, `visibility: private`)
contains, per evaluated image: manifest `id`, `group`, `target_index`, `truth`
label, the probability row in class order and the frozen decision. It also
records the provenance block below, the full evaluation spec and dataset id/version.
**It contains identifiers and targets: never commit it, attach it to issues or
publish it to the Git queue.** Only synthetic fixtures may appear in tests.

`results.json["clinical_evaluation"]` is aggregate only (no row identifiers):
provenance block, `samples`, `primary_metric`, `metrics`,
`predictions_content_sha256` (canonical hash of the artifact),
`selection_rule: minimum_selection_loss`, `frozen_declaration_only: true`,
`scientific_status: provisional`, `scientifically_eligible: false`,
`execution_verified: false`.

Provenance block (identical in artifact and result): `config_sha256`,
`splits_sha256`, `recipe_sha256` (= `recipe_identity`), `environment_sha256`
(canonical hash of this execution's `environment.json`), `evaluation_sha256`,
`best_epoch`, `partition`, `evaluation_scope`, `task`, `classes`, `score_kind`.
`environment_sha256` is per-execution provenance: `environment.json` is
rewritten with a new UTC timestamp on resume, so it differs between an
uninterrupted and a resumed attempt while rows, metrics and the identity hashes
(config/splits/recipe/evaluation/best epoch) stay equal.

`environment.json["clinical_evaluation"]` = `evaluation_sha256`, `task`,
`score_kind`, `frozen_declaration_only: true`, `scientifically_eligible: false`.

The public queue completion summary is unchanged except `artifact_count`
(9 for opt-in); no clinical metric or row reaches the queue.

## 4. Host runtime validation

`splits.json` has no ID→group→truth mapping and the attempt directory is
container-writable, so the validator never trusts identifiers, groups or labels
declared inside the attempt. New API in `augment_atlas.runtime`:

```python
@dataclass(frozen=True)
class TrustedEvaluationInputs:  # cfg, config_file_sha256, splits, records
    ...

load_trusted_evaluation_inputs(config_path, data, *, expected_config_sha256)
trusted_evaluation_inputs(settings, spec) -> TrustedEvaluationInputs | None
validate_completion(settings, out, spec)            # worker + reconciliation
validate_artifacts(out, spec=None, *, trusted=None) # legacy signature preserved
```

- **Source of trust.** The config is read from the pinned checkout
  (`settings["code"]/spec["config"]`, under `configs/smoke/`, file SHA256 equal
  to `spec["config_sha256"]`). Records are loaded from `settings["data"]` via an
  explicit `root=` (new keyword on `data.load_records`/`prepare_splits`) and
  splits are recomputed on the host. `ATLAS_DATA_ROOT`, `dataset.root` and the
  `/data` default are never consulted; a missing or nonexistent
  `settings["data"]` fails.
- **Opt-in decision** comes from the pinned config, not from the attempt's
  `config.json`, and is taken only after the file's SHA256 equals
  `spec["config_sha256"]`. Stripping `evaluation` from the attempt config
  therefore cannot downgrade an opt-in job to the eight-artifact path, even if
  `settings["code"]` was moved off the job commit before validation.
- **Moved checkout fails closed (legacy and clinical).** Reconciliation does
  not compare `settings["code"]` HEAD with `spec.code_commit`; instead, if the
  file at `settings["code"]/spec["config"]` is missing (`Invalid training config
  path`) or its bytes differ from `spec.config_sha256` (`Pinned job config
  checksum mismatch`), `validate_completion` raises and the attempt becomes
  `failed` / `checkpoint_artifacts_invalid`. This also applies to legacy jobs:
  before the clinical integration a legacy completion was validated without
  reading the checkout. Restore the job commit in `settings["code"]` before
  restarting a worker or reconciling.
- **Legacy compatibility.** For a config without `evaluation` (pinned bytes
  verified as above), `validate_completion` calls exactly
  `validate_artifacts(out, spec)` and no manifest is read.
  `validate_artifacts(out, spec)` with `trusted=None` on a legacy attempt is
  the previous validation. On an attempt whose config declares `evaluation` it
  fails closed. Specs without a `config` key (which `config_for` would never
  have claimed) keep the legacy call; the fail-closed rule still applies.
- **Parser failures.** `RecursionError` (deeply nested JSON written by the
  container) and `csv.Error` (oversized host manifest field) inside
  `validate_completion` are re-raised as `ValueError`, so the worker records a
  failed attempt instead of crashing and re-crashing on restart.
- **Opt-in checks** (after all legacy checks): attempt config and splits equal
  the trusted ones (canonical hash); `predictions.json` is checksummed in
  `completed.json`; `spec.config_sha256`, `spec.recipe_hash` (=
  `recipe_identity`), `spec.split_hash` and a nonempty `spec.code_commit` match;
  `environment.code_commit == spec.code_commit` and its clinical block,
  config/split hashes and scope match; the top level of `results.json` holds
  only the trainer's keys (no extra claims such as `scientifically_eligible`),
  with `phase == "smoke"`, `scientific_status == "provisional"` and, if
  present, `inbreast_previously_examined_in_pilot` equal to the adapter being
  `inbreast_csv`; `best_epoch`/`selection_loss` equal the
  first argmin of the `metrics.json` selection-loss journal; the artifact and
  `results.clinical_evaluation` are rebuilt from the stored probabilities with
  the trusted records (`evaluation.validate_prediction_artifact`) and must
  match exactly; the partition's `samples` equal the row count and its
  `accuracy` lies within the bounds implied by the probabilities (strict
  maxima vs ties).
- Worker success and stopped-attempt reconciliation both call
  `validate_completion`; validation errors map to the existing `failed` /
  `checkpoint_artifacts_invalid` outcomes. The b7 stop/TERM/reconciliation
  semantics are unchanged.

## 5. Limits (read before using any number)

- **No execution proof.** A forger who rewrites probabilities and every derived
  value consistently (metrics, content hash, accuracy, checksums) is not
  detectable without re-running the model; `test_runtime_predictions` documents
  this explicitly. Re-execution or a trusted signer would be required.
  The loss is not cross-checked against the probabilities. The selection
  journal `metrics.json` is also written by the container: a forger who
  rewrites the journal, `best_epoch`/`selection_loss` and the predictions
  consistently is accepted too; the first-argmin check only proves internal
  consistency, not which epoch was really best.
- **Upgrade every host before queueing opt-in jobs.** A worker running code
  from before this integration accepts a nine-artifact opt-in attempt through
  the legacy validator without any clinical revalidation. All hosts that can
  claim from the queue must run this code before a job whose config declares
  `evaluation` is enqueued.
- **No clinical or scientific validity.** Image-level point estimates only; no
  CI, no calibration, no external validation, no reader study. Synthetic toy
  values are meaningless.
- **INbreast stays a pilot.** The 103 filename groups / 387 images do not
  establish PatientID, and the pilot was already examined
  (`inbreast_previously_examined_in_pilot`). Group disjointness is only as good
  as the supplied `group` column.
- Probability rows must sum to 1 within absolute 1e-6 and lie in [0,1];
  float32 softmax rows satisfy this. Integer scores are stored as floats.
- Multilabel evaluation is not implemented in the trainer.
- Small technical splits (`group-hash-v1`) often lack a class in a partition;
  this yields explicit `null` metrics, as designed.

## 6. Tests and container acceptance

Local (stdlib; torch tests skip unless `ATLAS_CONTAINER_CPU_TESTS=1`):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

- `tests/test_evaluation.py`: config strictness, ge/gt, ties, non-finite and
  sum checks, undefined metrics, QWK, row/width, tampering, provenance,
  selection-only without I/O.
- `tests/test_runtime_predictions.py`: ninth-artifact host validation, trusted
  data root (including `dataset.split_file` for both split algorithms),
  resealed tampering, downgrade attempt, moved checkout in both directions,
  extra top-level result claims, parser failures, legacy call identity,
  worker/reconciliation wiring.
- `tests/test_clinical_training.py` (container gate): tiny-model real CPU loop;
  legacy 8 vs opt-in 9 with identical training, prediction recompute from
  `best.pt`, epoch selection independent of test content, selection-only,
  resume equality, interruption during the final pass (no `predictions.json`),
  non-finite final logits (explicit `RuntimeError`).

Container (pinned image, code read-only, no network):
`scripts/toy-train-cpu-clinical.sh` runs the full suite with the CPU gate, a
real ResNet-50 (random init) legacy and opt-in toy training through the
trainer CLI, validates both through `validate_completion` and prints a JSON
summary. The invocation is in the script header.
