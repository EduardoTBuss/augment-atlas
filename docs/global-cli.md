# Global CLI — operator runbook (external fixed recipe)

`python -m augment_atlas.global_cli` is the operator glue of spec section 13 for
the **external fixed recipe** campaign (INbreast pilot, campaign (a), spec
section 10). Each subcommand reads private documents from
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

## New datasets: D/E pools (`make-pools`)

`make-pools --spec SPLIT.json --acquisition-config CONFIG.json` runs
`global-dev-eval-v1`
(`dataset-acquisition-plan.md` section 5, `global_splits.py`), stdlib only.
The spec holds:
- the pools metadata: `dataset`, `task`, `grouping`, `provenance`, `source_sha256`;
- three seeds: `development`, `selection` and `evaluation`, where the
  evaluation seed must differ from the other two (Q3);
- `fractions {D, D_select_given_D}`;
- the pre-registered `thresholds {min_select_positive_groups,
  min_eval_fold_positive_groups}`.

The spec must reproduce the `splits` block of the pre-registered acquisition
config: algorithm, 5 folds, both fractions and both thresholds. Nothing can be
relaxed at split time. The command writes an unsigned candidate to the private
store. It prints counts, hashes, the seeds and the sha256 of the spec and
config files, so they can be recorded next to the manifest hash. If any audit fails, it stops with `split_audit_failed`;
there is no seed retry and no fallback. After approval, run `import-pools
--file <path> --expected-manifest <global_manifest_sha256>`.

Choices fixed by the implementation, not stated in section 5 (to be
confirmed or changed with the config):
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

**Pending decision for ISIC:** `configs/acquisition/isic2020-v1.json` pins a
single `split_seed`, but the pools schema and Q3 need a distinct evaluation
seed. Declaring one is an amendment to the pre-registered config.

## Not covered: HPO (campaign (b))

The HPO subcommands (`search-space`, `build-plan`, `seed-hpo`, `collect-hpo`,
`seal-freeze`) are deliberately absent. HPO configs embed `plan_sha256`, and
that hash covers `plan.runtime.commit`. But the collector recomputes the HPO job
ids with `code_commit = plan.runtime.commit` (`code_commit` is a job identity
field), and the worker requires the config file committed at the job's commit.
So the configs can never exist at that commit. This needs an explicit decision
(for example a Q6-style "same code identity" rule for HPO jobs) before those
subcommands can be written.
