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
  resume decided after the scores were visible, so infrastructure retries
  happen before the first collection.
- `seal-freeze` **and** `anchor --kind recipe_freeze --freeze-kind hpo_d`
  recompute the evidence from the current queue. They require it to equal
  the pin and the freeze's `selection_evidence_sha256`. An `hpo_d` anchor
  therefore takes the same collection arguments as `collect-hpo`.

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
  HEAD:<path>` from a clean `--code` checkout whose HEAD is on a
  remote-tracking branch. A local-only commit or a `skip-worktree` edit is
  never the pre-registered version.
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
