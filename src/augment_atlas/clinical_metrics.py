"""Dependency-free clinical metric primitives; see docs/clinical-metrics.md.

Inputs are JSON-style lists/tuples. Class universes and column order are always
explicit. Undefined metrics return JSON null with a reason, never NaN. This
module neither fits thresholds nor establishes independence of patients/splits.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import groupby
import math
from numbers import Integral, Rational, Real


def _vector(values, name):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    return list(values)


def _label(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError(f"{name} must contain string or integer labels, not bool")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{name} labels must be nonempty")
        return value
    return int(value)


def _classes(values, *, minimum=2):
    labels = [_label(v, "classes/labels") for v in _vector(values, "classes/labels")]
    if len(labels) < minimum or len(set(labels)) != len(labels):
        raise ValueError(f"classes/labels must contain at least {minimum} distinct labels")
    return labels


def _targets(values, classes, name="y_true"):
    targets = [_label(v, name) for v in _vector(values, name)]
    if any(v not in classes for v in targets):
        raise ValueError(f"{name} contains a label outside the fixed universe")
    return targets


def _number(value, name):
    """Normalize scores to exact rationals without binary64 rounding.

    Sorting/comparison is the only operation needed on scores. Fraction keeps
    Python integers, Rational values and binary floating-point integer ratios
    mutually comparable without mixed NumPy/Python comparison coercions.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain finite real numbers, not bool")
    try:
        if isinstance(value, Rational):
            numerator, denominator = value.numerator, value.denominator
        else:
            # float and NumPy floating scalars supply an exact integer ratio.
            # Their methods reject NaN and infinities instead of approximating.
            numerator, denominator = value.as_integer_ratio()
    except (AttributeError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite real numbers with an exact integer ratio") from exc
    if (isinstance(numerator, bool) or not isinstance(numerator, Integral)
            or isinstance(denominator, bool) or not isinstance(denominator, Integral)
            or denominator == 0):
        raise ValueError(f"{name} must contain finite real numbers with an exact integer ratio")
    return Fraction(int(numerator), int(denominator))


def _scores(values):
    return [_number(v, "y_score") for v in _vector(values, "y_score")]


def _matrix(values, width, name, *, numeric):
    rows = [_vector(row, f"{name} row") for row in _vector(values, name)]
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must have exactly {width} columns in fixed label order")
    if numeric:
        rows = [[_number(v, name) for v in row] for row in rows]
    return rows


def _same_length(left, right):
    if len(left) != len(right):
        raise ValueError("y_true and predictions/scores must have equal sample counts")


def _result(metric, value, reason, coverage, **extra):
    return {"metric": metric, "value": value, "reason": reason,
            "coverage": coverage, **extra}


def _binary_inputs(y_true, y_score, classes, positive_label):
    classes = _classes(classes)
    positive_label = _label(positive_label, "positive_label")
    if len(classes) != 2 or positive_label not in classes:
        raise ValueError("binary metrics require two classes and an explicit positive_label in classes")
    targets = _targets(y_true, classes)
    scores = _scores(y_score)
    _same_length(targets, scores)
    return [int(v == positive_label) for v in targets], scores, classes, positive_label


def _binary_metric(targets, scores, metric):
    n = len(targets)
    positive = sum(targets)
    negative = n - positive
    coverage = {"n_samples": n, "n_evaluated": n,
                "positive": positive, "negative": negative}
    reason = "empty_input" if not n else "no_positive_examples" if not positive else None
    if metric == "binary_auroc" and n and not negative:
        reason = "no_negative_examples"
    if reason:
        return _result(metric, None, reason, coverage)
    pairs = sorted(zip(scores, targets), reverse=metric != "binary_auroc")
    if metric == "binary_auroc":
        negatives_before, twice_wins = 0, 0
        for _, group in groupby(pairs, key=lambda item: item[0]):
            labels = [item[1] for item in group]
            p = sum(labels)
            q = len(labels) - p
            twice_wins += p * (2 * negatives_before + q)
            negatives_before += q
        value = twice_wins / (2 * positive * negative)
    else:
        seen, true_positive, terms = 0, 0, []
        for _, group in groupby(pairs, key=lambda item: item[0]):
            labels = [item[1] for item in group]
            new_positive = sum(labels)
            seen += len(labels)
            true_positive += new_positive
            terms.append((new_positive / positive) * (true_positive / seen))
        value = math.fsum(terms)
    return _result(metric, value, None, coverage)


def binary_auroc(y_true, y_score, *, classes, positive_label):
    """Full binary AUROC: P(score+ > score-) + .5 P(score+ == score-)."""
    targets, scores, classes, positive_label = _binary_inputs(
        y_true, y_score, classes, positive_label)
    return {**_binary_metric(targets, scores, "binary_auroc"),
            "classes": classes, "positive_label": positive_label}


def binary_average_precision(y_true, y_score, *, classes, positive_label):
    """Non-interpolated AP, sum(delta recall * precision), grouping score ties.

    Not trapezoidal PR area. No positives is undefined; all positives gives 1.
    """
    targets, scores, classes, positive_label = _binary_inputs(
        y_true, y_score, classes, positive_label)
    return {**_binary_metric(targets, scores, "binary_average_precision"),
            "classes": classes, "positive_label": positive_label,
            "pr_convention": "non_interpolated_average_precision"}


def _macro(metric, per_class, classes, n, **extra):
    defined = sum(item["value"] is not None for item in per_class)
    complete = defined == len(classes)
    value = math.fsum(item["value"] for item in per_class) / len(classes) if complete else None
    return _result(metric, value, None if complete else "undefined_class_components",
                   {"n_samples": n, "n_classes_expected": len(classes),
                    "n_classes_defined": defined},
                   classes=classes, per_class=per_class, **extra)


def macro_ovr_auroc(y_true, y_score, *, classes):
    """Mean full AUROC of every predefined class against all other classes.

    Columns follow classes exactly; finite decision scores are accepted without
    normalization. Any undefined class makes the entire macro undefined.
    """
    classes = _classes(classes)
    targets = _targets(y_true, classes)
    scores = _matrix(y_score, len(classes), "y_score", numeric=True)
    _same_length(targets, scores)
    per_class = [{"label": label, **_binary_metric(
        [int(v == label) for v in targets], [row[j] for row in scores], "binary_auroc")}
        for j, label in enumerate(classes)]
    return _macro("macro_ovr_auroc", per_class, classes, len(targets))


def _confusion(y_true, y_pred, classes):
    classes = _classes(classes)
    targets = _targets(y_true, classes)
    predictions = _targets(y_pred, classes, "y_pred")
    _same_length(targets, predictions)
    index = {label: i for i, label in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for target, prediction in zip(targets, predictions):
        matrix[index[target]][index[prediction]] += 1
    return classes, matrix, len(targets)


def _classification_metric(y_true, y_pred, classes, metric):
    classes, matrix, n = _confusion(y_true, y_pred, classes)
    per_class = []
    for i, label in enumerate(classes):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        denominator = support + predicted if metric == "macro_f1" else support
        numerator = 2 * tp if metric == "macro_f1" else tp
        per_class.append(_result(metric, numerator / denominator if denominator else None,
                                 None if denominator else "zero_denominator",
                                 {"n_samples": n, "support": support, "predicted": predicted,
                                  "tp": tp, "fp": predicted - tp, "fn": support - tp}, label=label))
    return _macro(metric, per_class, classes, n, confusion_matrix=matrix)


def macro_f1(y_true, y_pred, *, classes):
    """Fixed-universe mean 2TP/(2TP+FP+FN); no zero-division substitution."""
    return _classification_metric(y_true, y_pred, classes, "macro_f1")


def balanced_accuracy(y_true, y_pred, *, classes):
    """Fixed-universe mean recall, undefined when any true class is absent."""
    return _classification_metric(y_true, y_pred, classes, "balanced_accuracy")


def quadratic_weighted_kappa(y_true, y_pred, *, classes):
    """QWK using fixed ordered, equally spaced ordinal classes.

    1 - sum((i-j)^2 O_ij) / sum((i-j)^2 E_ij), E_ij=row_i*col_j/n.
    Absent classes retain their position. Zero expected disagreement is null.
    """
    classes, matrix, n = _confusion(y_true, y_pred, classes)
    support = [sum(row) for row in matrix]
    predicted = [sum(row[j] for row in matrix) for j in range(len(classes))]
    observed = sum((i - j) ** 2 * cell for i, row in enumerate(matrix) for j, cell in enumerate(row))
    expected_product = sum((i - j) ** 2 * support[i] * predicted[j]
                           for i in range(len(classes)) for j in range(len(classes)))
    reason = "empty_input" if not n else "zero_expected_disagreement" if not expected_product else None
    value = None if reason else 1 - (observed * n) / expected_product
    return _result("quadratic_weighted_kappa", value, reason,
                   {"n_samples": n, "n_evaluated": n,
                    "n_classes_expected": len(classes),
                    "n_true_classes_observed": sum(v > 0 for v in support)},
                   classes=classes, confusion_matrix=matrix,
                   per_class=[{"label": label, "support": support[i], "predicted": predicted[i]}
                              for i, label in enumerate(classes)])


def multilabel_macro_auroc(y_true, y_score, *, labels, uncertain_policy, missing_policy):
    """Independent binary columns; -1 is uncertain and None is missing.

    uncertain_policy: error/exclude/positive/negative. missing_policy:
    error/exclude. Exclusion is per annotation, reported per label. Every score
    must be finite even for excluded annotations. No disease becomes a class.
    """
    labels = _classes(labels, minimum=1)
    if uncertain_policy not in ("error", "exclude", "positive", "negative"):
        raise ValueError("uncertain_policy must be error/exclude/positive/negative")
    if missing_policy not in ("error", "exclude"):
        raise ValueError("missing_policy must be error/exclude")
    targets = _matrix(y_true, len(labels), "y_true", numeric=False)
    scores = _matrix(y_score, len(labels), "y_score", numeric=True)
    _same_length(targets, scores)
    per_class = []
    for j, label in enumerate(labels):
        binary, column = [], []
        uncertain, missing, excluded = 0, 0, 0
        for target_row, score_row in zip(targets, scores):
            target = target_row[j]
            if target is None:
                missing += 1
                if missing_policy == "error":
                    raise ValueError("missing annotation encountered with missing_policy=error")
                excluded += 1
                continue
            if isinstance(target, bool) or not isinstance(target, Integral) or target not in (-1, 0, 1):
                raise ValueError("multilabel targets must be integer -1/0/1 or None")
            if target == -1:
                uncertain += 1
                if uncertain_policy == "error":
                    raise ValueError("uncertain annotation encountered with uncertain_policy=error")
                if uncertain_policy == "exclude":
                    excluded += 1
                    continue
                target = int(uncertain_policy == "positive")
            binary.append(int(target))
            column.append(score_row[j])
        item = _binary_metric(binary, column, "binary_auroc")
        item["coverage"].update(n_samples=len(targets), uncertain=uncertain,
                                missing=missing, excluded=excluded)
        per_class.append({"label": label, **item})
    result = _macro("multilabel_macro_auroc", per_class, labels, len(targets),
                    uncertain_policy=uncertain_policy, missing_policy=missing_policy)
    result["labels"] = result.pop("classes")
    return result


def binary_predictions(y_score, *, classes, positive_label, threshold, comparison):
    """Apply a caller-frozen threshold; comparison is explicitly 'ge' or 'gt'."""
    classes = _classes(classes)
    positive_label = _label(positive_label, "positive_label")
    if len(classes) != 2 or positive_label not in classes:
        raise ValueError("binary predictions require two classes and positive_label in classes")
    threshold = _number(threshold, "threshold")
    if comparison not in ("ge", "gt"):
        raise ValueError("comparison must be ge or gt")
    negative_label = next(v for v in classes if v != positive_label)
    return [positive_label if (v >= threshold if comparison == "ge" else v > threshold)
            else negative_label for v in _scores(y_score)]


def multiclass_predictions(y_score, *, classes, tie_break):
    """Argmax with caller-frozen class order and tie policy 'first' or 'error'."""
    classes = _classes(classes)
    if tie_break not in ("first", "error"):
        raise ValueError("tie_break must be first or error")
    scores = _matrix(y_score, len(classes), "y_score", numeric=True)
    predictions = []
    for row in scores:
        maximum = max(row)
        if tie_break == "error" and row.count(maximum) > 1:
            raise ValueError("tied maximum score with tie_break=error")
        predictions.append(classes[row.index(maximum)])
    return predictions
