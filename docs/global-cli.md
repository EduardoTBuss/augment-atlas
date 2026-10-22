# Global CLI — operator runbook (external fixed recipe)

`python -m augment_atlas.global_cli` is the operator glue of spec section 13 for
the **external fixed recipe** campaign (INbreast pilot, campaign (a), spec
section 10). Each subcommand reads private documents from
`<data-root>/global/<dataset>/<kind>-<sha256>.json`, pinned by hash. That is the
same layout the worker reads. Each subcommand re-audits its inputs and writes
with the no-clobber publisher. Standard output carries only hashes, counts,
commits and private paths, and the CLI refuses to print a record ID or image
path. Re-running a step with the same inputs is idempotent.

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
12. `report --campaign-sha256 K --runs RUNS [--settings worker.json]` does the
    following:
    - collects every authorized cell from the queue and the attempt
      directories (`global_collect`);
    - re-audits the chronology;
    - writes the private `report-<sha>.json` and `final-collection-<sha>.json`.

    `--settings` adds the worker's full artifact validation. Missing cells
    stay `missing`, never zero, and one contaminated artifact fails the whole
    report.
13. `summary --report-sha256 R` (public, ID-free), then `export-article
    --report-sha256 R --out DIR` (directory outside the repository), and
    `compare --report-a A --report-b B --out DIR` once both campaigns exist.

## Not covered: HPO (campaign (b))

The HPO subcommands (`search-space`, `build-plan`, `seed-hpo`, `collect-hpo`,
`seal-freeze`) are deliberately absent. HPO configs embed `plan_sha256`, and
that hash covers `plan.runtime.commit`. But the collector recomputes the HPO job
ids with `code_commit = plan.runtime.commit` (`code_commit` is a job identity
field), and the worker requires the config file committed at the job's commit.
So the configs can never exist at that commit. This needs an explicit decision
(for example a Q6-style "same code identity" rule for HPO jobs) before those
subcommands can be written.
