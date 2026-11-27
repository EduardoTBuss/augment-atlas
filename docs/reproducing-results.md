# Reproducing the aggregate results

Use Python 3.11 and the pinned packages in
[analysis/requirements.txt](../analysis/requirements.txt). Run from the
repository root.

```sh
python -m pip install -r analysis/requirements.txt
python -m analysis.reproduce
```

The second command reads [results/](../results/) and writes to
`build/study-results/`. Use `--out-dir` to choose another destination.

| Input | Contents |
| --- | --- |
| `metrics.csv` | AUROC, AP, and fit counts for 450 fits across three datasets, ten policies, five folds, and seeds 11, 13, and 19 |
| `coverage.csv` | Aggregate image, group, and positive-group counts for the 15 dataset-fold pairs |
| `conditional-contrasts.csv` | 180 archived estimates and available intervals and p-values for H1, H2, and S1–S5, including each S4 dose and both S5 family-by-basic interactions |
| `pooled-metrics.csv` | Archived pooled AUROC and AP for each dataset, policy, and seed |
| `posthoc-inbreast-gamma-dose-difference.csv` | Post hoc conditional 95% interval for heavy minus light gamma (both added to basic) on INbreast, computed with the same paired group bootstrap (B = 10,000, seed 20261028) from per-image predictions that are not distributed |

The output includes five tables in CSV, paired-effect and factor-share figures
in PDF and PNG, and a CSV with the values behind each figure. `results.json`
contains unrounded values, including the baseline-subtraction SS identity,
leave-one-fold-out means, and sensitivity summaries that omit cells with
constant-score fits.

The primary contrasts average the five fold-level differences at seed 11.
The sensitivity averages all 15 fold-by-setting cells. The figures and
descriptive tables retain all fits. Standard deviations across cells use
the sample divisor. The balanced-grid SS terms partition the centered total;
each bar uses its own denominator. The grid RMS compares seeds 13 and 19
for basic and the four basic-plus-dose policies. The repeatability table
uses all ten policies for its unchanged-policy median, and compares the
two seed-specific paired effects within each fold for family sign agreement.

Conditional confidence intervals and bootstrap p-values come from archived SAP
outputs. The analysis resampled groups 10,000 times with seed 20261028 and kept
the fitted models fixed. The primary table uses the 97.5% Bonferroni intervals
and Holm-adjusted p-values. The three-setting sensitivity uses marginal 95%
intervals. The pooled metrics score the concatenated held-out predictions
within each policy and seed. The command verifies archived contrast estimates
against the fit grid and uses archived pooled metrics to compute pooled family
differences. The [SAP documentation](analysis-sap-v1.md) gives the resampling
and contrast definitions.

To verify numbers against a separate TeX source, run the checker with that
source and the generated directory. Its table and paragraph mappings follow
the result layout used for the five tables above. Supply an independent
descriptive archive to compare every paired-effect label and SS segment.

```sh
python -m analysis.check_paper_numbers --manuscript /path/to/main.tex \
  --reference-analysis /path/to/descriptive-archive --report build/number-check.csv
```

The archive directory contains `paired-cells.csv` and `factor-shares.csv`.
The checker reports each comparison in the CSV and prints differences and
unverified values. Exit code 1 indicates differences and exit code 2 indicates
unverified values when all other comparisons match. Values read from the
conditional and pooled archives are labeled “from archived SAP outputs.”
