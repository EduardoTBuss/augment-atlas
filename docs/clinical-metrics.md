# Clinical metric primitives

`augment_atlas.clinical_metrics` is an isolated, standard-library-only calculation
module. It does not change the trainer, run a campaign, tune thresholds, or prove
clinical validity. Results are image/annotation-level point estimates; no
confidence intervals, patient-level claims, external validation, or statistical
independence are established here.

The supplied Brain context (`projects/augment-atlas-datasets-hpo-economico-2026-10-16.md`
and its 2026-10-16 19 BRT maestro resumption checkpoint) selects INbreast binary
AUROC, retina DR 0–4 quadratic weighted kappa, skin multiclass macro OvR AUROC
(binary only after an explicit protocol amendment), and thorax multilabel macro
AUROC with frozen labels and uncertainty policy. Macro F1, balanced accuracy and
AP are secondary. This module implements mathematical primitives for those
endpoints; it does not certify dataset adapters, class mappings, patient identity,
or D/E separation. The reported INbreast 103 filename groups/387 images do not
establish PatientID, and the pilot has already been seen.

## Input and output contract

All sample arrays, matrices and universes are Python lists or tuples (JSON-style
arrays); convert tensors/NumPy arrays explicitly in the caller. Class identifiers
are nonempty strings or integers; booleans and floating class identifiers are
rejected. Universes must contain distinct identifiers, at least two for single
label tasks and at least one for multilabel tasks. A row's score columns follow
the supplied universe **in exactly that order**. The module never infers labels
from the evaluated samples, sorts the universe, or silently drops classes.

Scores are finite real numbers, with larger values indicating stronger evidence
for the corresponding positive class. Integers, `fractions.Fraction`, Python
floats and NumPy real scalar types are supported. Internally, scores and decision
thresholds are represented as exact integer ratios, preserving their represented
values and comparisons across types. There is no conversion to binary64 that
could turn `2**53` and `2**53 + 1` into a false tie. Very large integers and tiny
fractions remain valid. Rounding already present in caller-created floating
values cannot be recovered: `float(2**53 + 1)` already equals `float(2**53)`.
Other custom `numbers.Real` types must implement a finite exact
`as_integer_ratio()` or the `numbers.Rational` contract; unsupported types raise
`ValueError` instead of silently approximating. Complex values, Decimal and
booleans are outside this numeric contract. Arbitrary decision scores are accepted;
probability bounds and row normalization are not assumed or enforced. Nonfinite
scores are rejected even for annotations later excluded by an explicit policy.
Binary scores are one-dimensional, multiclass/multilabel scores rectangular
`n_samples × n_classes` arrays. Mismatched shapes, unknown labels, missing single
label targets and invalid policies raise `ValueError`; missing required keyword
arguments raise `TypeError`. Inputs are not mutated. There is no I/O or fitted
state, so restarting and reevaluating identical arrays gives identical results.

Metrics return a dictionary with `metric`, `value`, `reason`, and `coverage`.
`value` is a finite Python number or `None`; `reason` is `None` on success or a
stable reason string. The entire result can be written with
`json.dumps(result, allow_nan=False)`. Empty, correctly shaped arrays are accepted
and produce undefined metrics, not exceptions. Binary coverage reports total and
evaluated samples, positives and negatives. Macro results include ordered
`per_class` entries and expected/defined class counts. Their denominator is the
**full predefined universe**: if any component is undefined, the macro is `None`
with `undefined_class_components`. There is no partial defined-class average or
zero replacement. Inspect `per_class` to locate the precise reason and coverage.

## APIs and conventions

```python
from augment_atlas.clinical_metrics import (
    binary_auroc, binary_average_precision, macro_ovr_auroc,
    quadratic_weighted_kappa, macro_f1, balanced_accuracy,
    multilabel_macro_auroc, binary_predictions, multiclass_predictions,
)

auc = binary_auroc([0, 1], [0.2, 0.8], classes=[0, 1], positive_label=1)
pred = binary_predictions([0.2, 0.8], classes=[0, 1], positive_label=1,
                          threshold=0.5, comparison="ge")
f1 = macro_f1([0, 1], pred, classes=[0, 1])
```

- `binary_auroc(y_true, y_score, *, classes, positive_label)` computes full AUROC
  as the fraction of positive–negative pairs ranked correctly, with half credit
  for equal scores. Sorting and grouping ties gives `O(n log n)` time. Empty
  input, no positives or no negatives yields `empty_input`,
  `no_positive_examples` or `no_negative_examples` respectively. The positive
  identifier is required, regardless of class order. This corresponds to the
  full binary ROC area, with no partial-AUC normalization.
  [Official ROC AUC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
- `binary_average_precision(...)` takes the same keywords and computes
  `AP = sum((R_t - R_previous) * P_t)` at descending distinct score thresholds.
  Every tied score group enters together. Here AP is the chosen **non-interpolated
  precision-recall summary**; an unqualified `AUPRC` must carry this convention in
  the protocol. Trapezoidal PR area is a different endpoint and is not provided.
  No positives is undefined; an all-positive sample has AP 1. Empty input is
  undefined. [Official AP definition and interpolation distinction](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- `macro_ovr_auroc(y_true, y_score, *, classes)` evaluates each column against
  the indicator `y_true == classes[j]`, then takes the unweighted mean over all
  predefined classes. Each class needs both positive and negative observations.
  OvR retains sensitivity to the composition of the rest group even with macro
  averaging. Unlike sklearn's multiclass probability wrapper, this primitive
  accepts unnormalized finite per-class decision scores.
  [Official OvR and macro conventions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
- `quadratic_weighted_kappa(y_true, y_pred, *, classes)` treats the **positions**
  in `classes` as equally spaced ordinal grades. For DR pass `[0,1,2,3,4]`, even
  if a grade is absent in this partition. If `O_ij` is the confusion count and
  `E_ij = row_i * column_j / n`, the value is
  `1 - sum((i-j)^2 * O_ij) / sum((i-j)^2 * E_ij)`.
  The conventional division of weights by `(K-1)^2` cancels in this ratio.
  Empty input or zero expected disagreement is undefined. An absent grade is
  preserved and reported but does not by itself make QWK undefined. Returns
  the fixed-size confusion matrix and support/predicted counts per grade.
  [Official kappa definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html),
  [official quadratic weighting implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py).
- `macro_f1(y_true, y_pred, *, classes)` averages
  `2TP / (2TP + FP + FN)` over fixed classes. If a class has no true or predicted
  examples its denominator is zero, so the macro is undefined. If a class has
  no true examples but has false positive predictions, its F1 is defined as 0;
  the absence is visible in coverage. This deliberately uses no `zero_division`
  substitution. [Official F1 definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
- `balanced_accuracy(y_true, y_pred, *, classes)` is the unadjusted unweighted
  mean of `TP / (TP + FN)` across fixed classes. Any class with zero true support
  makes it undefined. Both classification functions expose the confusion matrix
  and per-class TP/FP/FN/support/predicted counts.
  [Official balanced accuracy definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).

## Multilabel thorax contract

`multilabel_macro_auroc(y_true, y_score, *, labels, uncertain_policy,
missing_policy)` uses a separate binary target and score column per predefined
disease. Multiple diseases can be positive on the same image. Diseases are never
collapsed into one mutually exclusive class. Each target is an integer `0` or
`1`, integer `-1` for uncertain, or `None` for a missing annotation. Floating
encodings, NaN and implicit missing values are invalid and require explicit
adapter normalization before calling.

Both policies are required and recorded in the result. `uncertain_policy` is
`error`, `exclude`, `positive` or `negative`; `missing_policy` is `error` or
`exclude`. Exclusion removes that annotation from that label's AUROC only. A
`positive`/`negative` uncertainty policy maps `-1` to 1/0; it does not assert
that the annotation is clinically certain. There is no default policy and no
automatic choice based on held-out performance.

Each label reports original `n_samples`, `n_evaluated`, uncertain/missing/excluded
counts, and post-policy positive/negative counts. Labels stay in the output and
in the full macro denominator, including labels left with zero or one target
state. Such a label makes the macro undefined. The fixed label list is returned
as `labels`, and ordered label details as `per_class` (the generic result field).
This missingness policy is a project contract, not a claim that every public
thorax dataset uses the same encoding or uncertainty convention.

## Classification decisions and protocol boundary

`binary_predictions(y_score, *, classes, positive_label, threshold, comparison)`
requires a finite threshold and an explicit comparison `ge` (`>=`) or `gt` (`>`).
`multiclass_predictions(y_score, *, classes, tie_break)` uses argmax with explicit
`first` (first column in frozen order) or `error` tie handling. Callers may also
supply previously derived hard labels directly to F1, balanced accuracy and QWK.
Their derivation and score semantics must remain part of the persisted protocol.

These helpers never select, fit or recalibrate a rule. Thresholds, class order,
label mappings, uncertainty/missingness policies, score orientation and endpoint
definitions must be fixed using development partition D before evaluation on E.
The primitives have no dataset or split identities and cannot enforce that
provenance. They cannot prevent a caller from changing a rule after inspecting E;
the future integration must persist and audit that configuration. No patient
confidence intervals, threshold tuning on E, or clinical outcome claims follow
from passing these unit tests.

## Validation and limitations

From the repository root, set `PYTHONPATH=src` (PowerShell:
`$env:PYTHONPATH='src'`) before running
`python -m unittest discover -s tests -p test_clinical_metrics.py -v`. It exercises
hand-calculated values, an independent pairwise AUROC oracle, exact large-integer
and Fraction ordering, optional installed-NumPy scalar parity, true numeric ties, empty and
single-class inputs, fixed ordinal spacing, class/column permutations, annotation
policies, invalid values/shapes and strict JSON serialization. Optional sklearn
parity tests run only when it is already installed; no dependency is installed.
The module does not implement sample weights, clustered inference, pooled fold
estimates, multilabel F1, survival/time-dependent endpoints, or task adapters.
Independent mathematical review and final trainer integration remain separate
acceptance steps.
