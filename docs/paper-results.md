# Paper results from sealed SAP v1 (X3b)

Implementation: src/augment_atlas/paper_results.py. This offline renderer calls
analysis_sap_v1.read_predictions and analysis_sap_v1.analyze once per dataset.
Statistical computations stay in analysis_sap_v1; the renderer only formats
its outputs. No training, real-data validation, recipe selection, cross-dataset
pooling or scientific eligibility is added. See analysis-sap-v1.md for the SAP.

## CLI

Dependencies must already be installed (NumPy and matplotlib; fixture/tests also
use SciPy). No network or dependency installation is needed.

~~~powershell
$env:PYTHONPATH='src'
python -m augment_atlas.paper_results --campaign synthetic_alpha=path/to/alpha --campaign synthetic_beta=path/to/beta --out outputs/paper-results --seed 20261028 --B 10000
~~~

Repeat --campaign DATASET=PREDICTIONS for separate studies. Paths may be native
prediction directories, single native artifacts, or normalized CSV files accepted
by the SAP reader. Quote the whole DATASET=PATH argument when it contains spaces.
Duplicate datasets and mixed campaigns are rejected. Dataset IDs must match the
native header. All studies are validated/analyzed before output publication.

--seed is exclusively the bootstrap RNG seed. --B defaults to 10000;
--block-size defaults to 64. --training-seeds PRIMARY SENS2 SENS3 defaults to
1 2 3, preserving the predeclared order, and applies to every supplied campaign.
The first seed alone supplies primary tables and figures. All three seeds enter
the sensitivity only if the entire 10-arm x 5-fold x 3-seed matrix is evaluable.

For native INbreast, explicitly pass --recipe-source INbreast=external_fixed
or --recipe-source INbreast=hpo_d (using the exact dataset ID). Repeat this flag
for other datasets as needed. A CSV recipe_source declaration is retained;
conflicting declarations fail. No kind is inferred from a freeze hash.
INbreast hpo_d retains points but has no intervals or p-values.

Exit 0 means output generation completed, even if a contrast is not_evaluable.
Invalid input or an I/O/rendering error returns 2 with an error on stderr.
Successful output is listed and hashed in manifest.json, written last. Rendering
is staged before replacing existing outputs; publication interruption can leave
partial files without a completion marker. Reruns replace current output names;
old dataset tables from earlier runs may remain, but are not referenced by the
new snippet or manifest. Treat only manifest-listed files as the current bundle.

## Output contract

- table-<safe-dataset>-<digest>.tex: one table per study, using table* to span
  columns in a two-column manuscript (the forest remains column-width). Lists all ten arms,
  primary-seed AUROC point estimates from each arm's emitted metrics (no arm
  intervals), all 15 H1/H2 and S1--S5 delta AUROC contrasts with their SAP 95%
  intervals, plus H1/H2 97.5% intervals and Holm m=2 p-values. S4 includes all
  eight intensity comparisons: four gamma/CLAHE light/heavy arms versus none,
  and four basic+gamma/CLAHE arms versus basic. No additional arm/control
  comparisons are computed. Coverage lists emitted per-fold image/group/positive-group
  counts, and completeness lists observed/evaluable folds for every arm and declared seed. Counts
  describe only the supplied prediction universe, not trusted campaign eligibility.
  NE means not_evaluable, D means descriptive_only. Machine reasons and SAP
  small-positive-group warnings are retained. Values are rounded only for display.
- forest.pdf / forest.png: combined forest, one vertical panel per dataset,
  3.35-inch width, PNG at 300 dpi, no embedded title. Dataset names label the
  panel y-axes. All 15 SAP contrasts remain in SAP order (H1/H2 first), with
  zero reference, 95% thick bars, 97.5% thin bars for H1/H2, and NE/D markers.
  A defined estimate remains visible even when inference is unavailable.
  Height is 2.5 inches per dataset (7.5 inches for three); pagination for
  larger study sets is a paper-layout task.
- results.json: full-precision SAP reports, including primary and sensitivities,
  renderer-derived completeness (including per-fold observed/expected image counts
  and status/reason), coverage, reasons, warnings, native pools_sha256 and freeze_sha256 for each
  dataset, input content hashes and sizes, SAP/generator/clinical-metrics source
  SHA-256 hashes and NumPy version. CSV without bindings retains nulls. No raw
  image/group identifiers, raw scores or input paths are exported.
- results.tex: static descriptive text and inputs for all dataset tables and
  forest.tex, plus SAP limitation templates. No interpretation of observed numbers.
- forest.tex: figure wrapper and caption defining the uncertainty and markers.
- manifest.json: output SHA-256 hashes and code provenance. The JSON reports
  never turn execution_verified or scientific_eligible true.

LaTeX requires graphicx. From the paper root, use:

~~~tex
\usepackage{graphicx} % preamble
% in the results section; use the generated directory relative to the paper root
\newcommand{\PaperResultsPath}{path/to/generated/results}
\input{\PaperResultsPath/results.tex}
~~~

The snippet defines PaperResultsPath as '.' only if the paper has not set it.
Dataset names and reasons are escaped for LaTeX; filenames use a safe slug and
a digest to avoid traversal and collisions.

## Sealed SAP boundary

The public tag is sap-v1-sealed-20261028. analysis_sap_v1.py must remain at
SHA-256 bb3bb94a76705f2499f3d32c3778a0c3d212abc314d92ec08678d6ea1d2c2458;
its documentation analysis-sap-v1.md is also preserved from HEAD. The renderer
and synthetic generator refuse a different script hash before generating output.
The test test_sealed_sap_sha256 pins the literal digest independently.

Only read_predictions and analyze supply analysis results. Arm AUROC is taken
from primary.arms[].metrics.auroc; no arm intervals or arm_comparisons exist in
the sealed output. Contrast estimates and intervals are copied without recalculation.

paper_results.prediction_completeness derives counts from validated normalized
predictions and emitted per_fold statuses. It preserves all SAP report fields,
adding only completeness and provenance to each dataset. For sensitivity seeds
whose per_fold output is unavailable, it checks cell presence, equality with the
observed fold image universe, finite scores and both classes being present.
It does not run extra analyses or calculate metrics or intervals. Observed means
at least one image; evaluable requires a complete paired, finite, two-class fold.
Thus a partial-image fold counts as observed but not evaluable; a missing fold
counts as neither. The per-fold counts cover the union of supplied images, not a
trusted campaign roster. No extra positive_images field is required or emitted.

## Synthetic-only reproduction

These commands create three native-schema synthetic studies: complete three-seed
synthetic_alpha, synthetic_beta missing basic_gamma_heavy fold 4 (H1 NE, H2
available), and INbreast_synthetic explicitly descriptive hpo_d. The generator
refuses a nonempty destination to prevent mixing stale fixtures.

~~~powershell
$env:PYTHONPATH='src;tests'
python scripts/generate_paper_results_synthetic.py --out outputs/paper-results-synthetic/inputs
python -m augment_atlas.paper_results --campaign synthetic_alpha=outputs/paper-results-synthetic/inputs/alpha --campaign synthetic_beta=outputs/paper-results-synthetic/inputs/beta --campaign INbreast_synthetic=outputs/paper-results-synthetic/inputs/inbreast --recipe-source INbreast_synthetic=hpo_d --out outputs/paper-results-synthetic/report --seed 20261028 --B 10000
$env:ATLAS_SLOW_TESTS='0'
python -m unittest test_paper_results test_analysis_sap_v1 -v
~~~

Tests check the sealed source digest, refusal of a changed seal, native integrity,
exact SAP parity for each study after removing renderer metadata, all eight S4
contrasts, point-only arms, missing/partial/nonfinite/single-class completeness,
undefined contrast inference, descriptive scope, provenance, output
hashes, figure dimensions/titles, escaping, deterministic reruns, and preservation
of a prior completed report on validation/rendering failure. No real data is used.
See result.md and outputs/paper-results-x3b/ for current execution evidence.
