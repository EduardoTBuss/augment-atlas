# SAP v1 option-A analysis (C5)

Implementation: src/augment_atlas/analysis_sap_v1.py. This is an offline,
exploratory, image-level analysis of one dataset and one frozen recipe. It does
not train, select checkpoints, retry jobs, seal the SAP, open images, establish
clinical independence, or update campaign/report protocol S4. Pin the script
SHA-256 and the approved SAP before opening any real E predictions. The source
hash in outputs records what ran; it does not prove that chronology was followed.
No real predictions were used to develop or validate this implementation.

## Invocation and input

From a checkout in PowerShell (dependencies must already be available):

~~~powershell
$env:PYTHONPATH='src'
python -m augment_atlas.analysis_sap_v1 --predictions <dir-or-csv> --dataset <id> --out <dir> --seed 20261028 --B 10000
~~~

The runtime dependencies are NumPy and matplotlib; the synthetic coverage test
also uses SciPy. No pandas or scikit-learn. Exit 0 means output generation
completed, including a manifest; individual contrasts can still be
not_evaluable. Invalid input exits 2 and prints an error. The default block size
is 64 and can be changed with --block-size. --seed is the bootstrap seed.

Training seeds have a separate, fixed declaration: --training-seeds 1 2 3 by
default. These are the actual numeric training seeds, in the predeclared SAP
order. For a campaign declaring 11,13,19, pass --training-seeds 11 13 19. The first
entry is SAP "seed 1" and is the only primary analysis. No numeric sorting,
observed performance, or fallback selects a primary seed. Unlisted seeds are
rejected. The mean over all three declared seeds is a separate sensitivity only
if every 10 arms x 5 folds x 3 seeds cell is complete and defined. Partial extras
never enter primary estimates or that sensitivity. The seed declaration is
recorded in analysis.json.

Two input representations:

* A directory recursively containing only the selected final predictions.json
  for each cell, or a single native predictions.json file. The reader adapts
  global_evaluation._build: header arm, fold, train_seed, dataset.id and task;
  rows id, group, target, scores. It verifies the native closed schema and
  content_sha256, final/E_test, last_epoch at epoch=epochs=20, binary classes,
  and softmax probabilities. It selects the score column for task.positive_class
  and maps the target to 0/1. Files must share dataset/version, task, pool and
  freeze bindings. Duplicate retries, mixed recipes and mixed datasets are
  errors; the reader never chooses a best or latest attempt.
* A single UTF-8 CSV with required header
  dataset,arm,fold,seed,image,group,label,score (column order is arbitrary).
  dataset matches --dataset; arm is one of contracts.ARMS; fold is 0..4; seed
  is an actual declared training seed; image and group are nonempty identifiers;
  label is already 0/1 for the declared positive class; score is that class's
  floating score. Optional phase and partition, if present, must be final and
  E_test. Their absence is the caller's explicit declaration of this CSV's
  final E-only scope. Optional recipe_source must be constant. A directory
  containing CSV files is not scanned; pass a CSV file directly.

For INbreast, declare --recipe-source external_fixed (primary inference) or
hpo_d (point estimates/descriptive only, without intervals or p-values). A CSV
recipe_source column can supply this declaration; conflicting declarations
are rejected. Native prediction headers carry the freeze hash, but not the
recipe-source kind, so the flag is required for INbreast native files. Other
datasets can also record the source with this flag. The kind is a declaration,
not an independent audit of a freeze. For INbreast, recipe_source is supplied by
the human operator and must be checked against the seal before analysis. The
effective declaration (including a CLI-supplied value), native freeze_sha256 and
pools_sha256, and numpy.__version__ (as numpy_version) are recorded explicitly
in analysis.json provenance and manifest.json. Missing native bindings in CSV
input are recorded as null; they are never inferred. ISIC HPO remains confined
to D upstream.

The grouping column must already encode INbreast group_key or the connected
union of ISIC patient and duplicate links. No duplicate-link discovery is done
here. A group or image crossing folds, inconsistent labels/group membership,
or duplicate cell/image predictions raises isolation_violation or an input
error. Per-fold image universes must agree across all arms/seeds. An arm with
an absent fold, unmatched images, undefined AUROC/AP or nonfinite CSV scores is
not_evaluable. No row is dropped and no four-fold mean is reported. Malformed
native JSON, including a nonfinite/noncanonical artifact, is an input error.

The caller must first audit final artifacts against trusted campaign, pools,
records and initialization evidence using global_collect/global_report. Without
these inputs, this module cannot independently detect an omitted image common
to every arm, a falsely declared group/partition, D membership, identical model
initialization, real execution, or the truth of labels. Output retains
execution_verified=false and scientific_eligible=false. Hashes are provenance,
not approval. No identifiers, raw scores, input paths or group keys enter the
analysis outputs; the private input remains private.

## Estimands and contrasts

Every arm's point AUROC and AP calls clinical_metrics directly. Thus AUROC is
exactly that function's Mann-Whitney value, including midranks for ties. For each
arm: mean of all five fold metrics, followed by the mean of the declared seeds
of that analysis. Seeds are never ensembled. Means use math.fsum, matching
_global_report aggregation_. Each family mean is the mean of arm metrics,
not an average of prediction scores.

| ID | Contrast on AUROC and on AP |
| --- | --- |
| H1 | mean(basic_gamma_light,basic_gamma_heavy) - basic |
| H2 | mean(basic_clahe_light,basic_clahe_heavy) - basic |
| S1 | mean(gamma_light,gamma_heavy) - none |
| S2 | mean(clahe_light,clahe_heavy) - none |
| S3 | basic - none |
| S4 (8 rows) | Each of four basic+family/intensity arms - basic; each of four isolated family/intensity arms - none |
| S5_gamma | H1 - S1 (descriptive interaction, no hypothesis test) |
| S5_clahe | H2 - S2 (descriptive interaction, no hypothesis test) |

All constituent arms must have all five defined folds in every seed of the
analysis. No partial intensity mean is permitted. Macro-F1, balanced accuracy,
sensitivity and specificity at fixed threshold >=0.5 are descriptive fold-macro
arm means, with no intervals or tests. Brier and equivalence margins are absent.

## Bootstrap and inference

Within each fold, a group is positive-status if any image in it is positive.
Draw each stratum's original number of groups with replacement, independently
across folds/status strata. Every image inherits its group's integer
multiplicity, including images with different labels within a group. The exact
same group counts are used across every arm and training seed. The image-level
endpoint is preserved: groups are sampling units, not averaged predictions.

Sorted score tie blocks are prepared once. Weighted positive/negative counts,
cumulative counts and Mann-Whitney pair wins are vectorized over replicate
blocks; AP uses precision at descending distinct-score thresholds. Rows are not
copied by multiplicity. Working matrices scale with block_size x images/fold,
not B x total_images x arms. The retained bootstrap metrics are B x 2 per
arm/seed. Independent RNG streams for each fold/status make results deterministic
under row permutation and block-size changes on the tested NumPy environment.
Bitwise reproducibility across NumPy versions is not claimed.

If any fold has fewer than eight positive-status groups, analysis.json includes
the warning "coverage below nominal measured in simulation with <8 positive groups per fold"
in its warnings list. The list is empty otherwise. The count uses the shared
fold group universe required for paired arm comparisons.

B defaults to 10,000. Report linear-interpolated 2.5/97.5 percentile intervals
for delta AUROC and delta AP. For H1/H2 AUROC only:

    p_boot = min(1, 2*(min(count(delta_star <= 0), count(delta_star >= 0))+1)/(B+1))

Apply Holm m=2 at alpha=.05 within this dataset and add unadjusted 1.25/98.75
percentile endpoints (97.5% marginal intervals, Bonferroni simultaneous coverage
at least 95% for the pair). An unavailable primary retains its multiplicity
slot with p=1 for adjusting the other primary; its own p remains null. No p
is generated for AP or S1-S5. These exploratory intervals do not establish
absence of an effect or equivalence.

With mixed-label groups, status stratification can still produce a replicate
with no negative image. Every undefined replicate is counted. If any replicate
of an endpoint is undefined, that endpoint's contrast inference is
not_evaluable, its point estimate is retained, and its interval/p is null.
There is no adaptive resampling, omission of bad replicates or percentile over
a reduced B. The other endpoint may remain defined. Failed infrastructure jobs
are handled upstream (same-job retries <=2); this script never retries training.

pooled_E is reported as a separate descriptive per-arm/per-seed point
sensitivity, with concatenation across E folds and no seed ensemble. No pooled
cross-dataset estimate or ranking is produced. The other prelisted SAP
sensitivities (group endpoint, breast views, explored-group exclusions, alternate
B and recipe comparisons) require separately prepared inputs/protocol context;
this slice does not silently derive or run them.

## Artifacts and checks

* analysis.json: protocol, declared seeds/recipe, sampling/coverage, primary
  arm/fold values and all 15 contrasts, three-seed and pooled_E sensitivities,
  source/input SHA-256 provenance. Unavailable values use null and reasons.
* table.csv: arm metrics and AUROC/AP contrasts for primary and available
  three-seed sensitivity, interval/p columns, statuses/reasons.
* forest.pdf / forest.png: one-column (3.35 inch) primary delta-AUROC plot,
  95% intervals; thin 97.5% intervals for H1/H2; NE marks unavailable inference.
  No embedded title; the paper supplies the caption.
* manifest.json: source/input hashes and hashes of the four output files,
  written last as a completion marker. Rerunning replaces the output files;
  the old marker is removed first. An interruption can leave partial files
  without a completed manifest. Numeric results are deterministic; PDF metadata
  timestamps may change the PDF hash between executions.

~~~powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -p test_analysis_sap_v1.py -v
python -m unittest discover -s tests -p 'test_global_report*.py' -v
python scripts/benchmark_analysis_sap_v1.py --out outputs/sap-v1-synthetic
~~~

The fast coverage check uses 60 synthetic grouped Gaussian experiments and
requires coverage >=0.85. With ATLAS_SLOW_TESTS=1, an additional 200-experiment
coverage check requires >=0.90 and a separate 200-experiment null check requires
the fraction of unadjusted H1 p_boot <=0.05 to lie in [0.01, 0.10]. Both use
B=299, five folds, 16 positive and 16 negative groups per fold, and two images
per group. Scores use a shared Gaussian component of variance 1 (half group,
half image), plus independent N(0, 0.5) noise for each arm (0.5 is the standard
deviation). Treatment adds shift*label before the monotone normal-CDF mapping.
Thus the nonzero population delta is Phi(0.6/sqrt(2*1.25))-0.5; the null uses
shift=0 with the same independent arm noise. Run both modes in PowerShell:

~~~powershell
$env:PYTHONPATH='src;tests'
python -m unittest test_analysis_sap_v1
$env:ATLAS_SLOW_TESTS='1'
python -m unittest test_analysis_sap_v1
~~~

These are synthetic validation exercises, not evidence that coverage or type-I
error is calibrated for real ISIC/INbreast.
The benchmark measures B=128 and B=256 on 26,000 synthetic images across five
folds, ten arms and one seed. It records warm-up, CPU/runtime versions, full
configuration, source hash, input-generation hash, raw bootstrap metrics and
linear extrapolation to B=10,000; it excludes preparation, I/O and plots.
See result.md and outputs/sap-v1-synthetic/ for observed validation evidence.
