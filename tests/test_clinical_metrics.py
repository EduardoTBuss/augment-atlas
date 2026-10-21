"""Hand-calculated fixtures and optional installed-sklearn parity; no data files."""
import importlib.util
import itertools
import json
from fractions import Fraction
from pathlib import Path
import random
import subprocess
import sys
import unittest

from augment_atlas.clinical_metrics import (
    balanced_accuracy, binary_auroc, binary_average_precision, binary_predictions,
    macro_f1, macro_ovr_auroc, multiclass_predictions, multilabel_macro_auroc,
    quadratic_weighted_kappa,
)


def auc(y, scores, **kwargs):
    return binary_auroc(y, scores, classes=[0, 1], positive_label=1, **kwargs)


def ap(y, scores):
    return binary_average_precision(y, scores, classes=[0, 1], positive_label=1)


class BinaryMetricTests(unittest.TestCase):
    def test_auroc_pair_count_and_ties(self):
        # Four +/- pairs: win, win, tie, loss => 2.5/4.
        self.assertEqual(auc([0, 1, 0, 1], [1, 1, 2, 3])["value"], 5 / 8)
        self.assertEqual(auc([0, 1, 0, 1], [7, 7, 7, 7])["value"], 1 / 2)
        self.assertEqual(auc([0, 1], [10, -10])["value"], 0)
        self.assertEqual(auc([0, 1], [-10, 10])["value"], 1)

    def test_auroc_against_independent_pairwise_oracle(self):
        rng = random.Random(1729)
        for n in range(2, 16):
            y = [0, 1] + [rng.randrange(2) for _ in range(n - 2)]
            scores = [rng.randrange(4) for _ in y]
            positive = [s for target, s in zip(y, scores) if target == 1]
            negative = [s for target, s in zip(y, scores) if target == 0]
            expected = sum((p > q) + 0.5 * (p == q) for p in positive for q in negative)
            expected /= len(positive) * len(negative)
            self.assertAlmostEqual(auc(y, scores)["value"], expected)

    def test_average_precision_is_noninterpolated_and_tie_grouped(self):
        # Descending labels 1,0,1,0: (.5*1)+(.5*2/3)=5/6.
        self.assertAlmostEqual(ap([0, 0, 1, 1], [.1, .4, .35, .8])["value"], 5 / 6)
        # Tied top group 1,0 gives precision .5, last positive gives 2/3.
        self.assertAlmostEqual(ap([1, 0, 1], [2, 2, 1])["value"], 7 / 12)
        self.assertEqual(ap([0, 1, 1, 0], [0, 0, 0, 0])["value"], .5)
        self.assertEqual(ap([1, 1], [1, -2])["value"], 1)
        self.assertEqual(ap([0, 0], [1, 2])["reason"], "no_positive_examples")

    def test_order_does_not_change_tie_results(self):
        pairs = [(0, 2), (1, 2), (0, 0), (1, 0)]
        for perm in itertools.permutations(pairs):
            y, scores = map(list, zip(*perm))
            self.assertEqual(auc(y, scores)["value"], .5)
            self.assertEqual(ap(y, scores)["value"], .5)

    def test_fixed_positive_label_and_class_order(self):
        result = binary_auroc(["ill", "well"], [1, 0],
                              classes=["ill", "well"], positive_label="ill")
        self.assertEqual(result["value"], 1)
        self.assertEqual(result["classes"], ["ill", "well"])
        self.assertEqual(result["positive_label"], "ill")

    def test_empty_or_single_class_is_explicit(self):
        for y, scores, reason in (([], [], "empty_input"), ([1], [0], "no_negative_examples"),
                                  ([0], [0], "no_positive_examples")):
            result = auc(y, scores)
            self.assertIsNone(result["value"])
            self.assertEqual(result["reason"], reason)
            self.assertEqual(result["coverage"]["n_evaluated"], len(y))
            json.dumps(result, allow_nan=False)


class FixedUniverseTests(unittest.TestCase):
    def test_ovr_hand_calculated_and_class_permutation(self):
        y = ["a", "b", "c", "a"]
        scores = [[3, 2, 1], [2, 3, 2], [1, 1, 3], [0, 0, 0]]
        # a: 2 wins/4; b: 3 wins/3; c: 3 wins/3.
        result = macro_ovr_auroc(y, scores, classes=["a", "b", "c"])
        self.assertEqual([v["value"] for v in result["per_class"]], [.5, 1, 1])
        self.assertAlmostEqual(result["value"], 5 / 6)
        permuted = macro_ovr_auroc(y, [[r[2], r[0], r[1]] for r in scores], classes=["c", "a", "b"])
        self.assertEqual(result["value"], permuted["value"])
        self.assertEqual([v["label"] for v in permuted["per_class"]], ["c", "a", "b"])

    def test_ovr_absent_class_never_removed(self):
        result = macro_ovr_auroc([0, 1], [[2, 1, 0], [1, 2, 0]], classes=[0, 1, 2])
        self.assertIsNone(result["value"])
        self.assertEqual(result["coverage"]["n_classes_expected"], 3)
        self.assertEqual(result["coverage"]["n_classes_defined"], 2)
        self.assertEqual(result["per_class"][2]["reason"], "no_positive_examples")
        self.assertEqual(len(result["per_class"]), 3)

    def test_confusion_metrics_hand_calculated(self):
        # O=[[1,1,0],[0,1,0],[0,1,1]], F1=[2/3,1/2,2/3], recall=[1/2,1,1/2].
        y, pred = [0, 0, 1, 2, 2], [0, 1, 1, 1, 2]
        f1 = macro_f1(y, pred, classes=[0, 1, 2])
        self.assertAlmostEqual(f1["value"], 11 / 18)
        self.assertEqual(f1["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 1, 1]])
        self.assertAlmostEqual(balanced_accuracy(y, pred, classes=[0, 1, 2])["value"], 2 / 3)

    def test_f1_missing_class_policy_is_not_recall_policy(self):
        absent = macro_f1([0, 1], [0, 1], classes=[0, 1, 2])
        self.assertIsNone(absent["value"])
        self.assertEqual(absent["per_class"][2]["reason"], "zero_denominator")
        # Class 2 has false positives: its F1 denominator is nonzero, F1=0.
        predicted_absent = macro_f1([0, 1], [0, 2], classes=[0, 1, 2])
        self.assertEqual(predicted_absent["value"], 1 / 3)
        recall = balanced_accuracy([0, 1], [0, 2], classes=[0, 1, 2])
        self.assertIsNone(recall["value"])
        self.assertEqual(recall["per_class"][2]["coverage"]["support"], 0)

    def test_qwk_fixed_ordinal_spacing_and_negative_agreement(self):
        # True [0,2], predicted [0,1]: observed squared distance=1;
        # expected squared distance=(0+1+4+1)/2=3 -> kappa=2/3.
        result = quadratic_weighted_kappa([0, 2], [0, 1], classes=[0, 1, 2, 3, 4])
        self.assertAlmostEqual(result["value"], 2 / 3)
        self.assertEqual(len(result["confusion_matrix"]), 5)
        self.assertEqual(result["per_class"][4]["support"], 0)
        # Change ordinal positions, rather than numeric values, changes distances.
        changed = quadratic_weighted_kappa([0, 2], [0, 1], classes=[1, 0, 2, 3, 4])
        self.assertAlmostEqual(changed["value"], -1 / 3)
        self.assertEqual(quadratic_weighted_kappa([0, 4], [4, 0], classes=list(range(5)))["value"], -1)
        self.assertEqual(quadratic_weighted_kappa([0, 4], [0, 4], classes=list(range(5)))["value"], 1)

    def test_qwk_degenerate_and_empty(self):
        for y, pred, reason in (([], [], "empty_input"), ([2, 2], [2, 2], "zero_expected_disagreement")):
            result = quadratic_weighted_kappa(y, pred, classes=list(range(5)))
            self.assertIsNone(result["value"])
            self.assertEqual(result["reason"], reason)
            json.dumps(result, allow_nan=False)
        self.assertEqual(quadratic_weighted_kappa([2, 2], [3, 3], classes=list(range(5)))["value"], 0)

    def test_empty_macro_results_are_finite_json(self):
        results = [macro_ovr_auroc([], [], classes=[0, 1, 2]),
                   macro_f1([], [], classes=[0, 1]), balanced_accuracy([], [], classes=[0, 1])]
        for result in results:
            self.assertIsNone(result["value"])
            self.assertEqual(result["coverage"]["n_classes_defined"], 0)
            json.dumps(result, allow_nan=False)


class MultilabelTests(unittest.TestCase):
    def test_independent_diseases_can_cooccur(self):
        result = multilabel_macro_auroc([[1, 1], [0, 1], [1, 0], [0, 0]],
                                       [[4, 4], [1, 3], [3, 2], [2, 1]],
                                       labels=["effusion", "edema"], uncertain_policy="error", missing_policy="error")
        self.assertEqual(result["value"], 1)
        self.assertEqual(result["labels"], ["effusion", "edema"])
        self.assertEqual([v["coverage"]["positive"] for v in result["per_class"]], [2, 2])

    def test_exclusions_are_per_annotation_and_reported(self):
        result = multilabel_macro_auroc([[1, 0], [0, 1], [-1, 1], [1, None]],
                                       [[4, 1], [1, 4], [0, 3], [3, 2]],
                                       labels=["a", "b"], uncertain_policy="exclude", missing_policy="exclude")
        self.assertEqual(result["value"], 1)
        for i, field in enumerate(("uncertain", "missing")):
            coverage = result["per_class"][i]["coverage"]
            self.assertEqual(coverage["n_samples"], 4)
            self.assertEqual(coverage["n_evaluated"], 3)
            self.assertEqual(coverage["excluded"], 1)
            self.assertEqual(coverage[field], 1)

    def test_mapping_uncertainty_changes_target_and_is_persisted(self):
        kwargs = dict(labels=["a"], missing_policy="error")
        positive = multilabel_macro_auroc([[0], [1], [-1]], [[0], [1], [2]], uncertain_policy="positive", **kwargs)
        negative = multilabel_macro_auroc([[0], [1], [-1]], [[0], [1], [2]], uncertain_policy="negative", **kwargs)
        self.assertEqual(positive["value"], 1)
        self.assertEqual(negative["value"], .5)
        self.assertEqual(positive["uncertain_policy"], "positive")
        self.assertEqual(positive["per_class"][0]["coverage"]["uncertain"], 1)
        self.assertEqual(positive["per_class"][0]["coverage"]["excluded"], 0)

    def test_undefined_label_keeps_fixed_denominator(self):
        result = multilabel_macro_auroc([[0, None], [1, -1]], [[0, 1], [1, 0]],
                                       labels=["a", "b"], uncertain_policy="exclude", missing_policy="exclude")
        self.assertIsNone(result["value"])
        self.assertEqual(result["coverage"]["n_classes_expected"], 2)
        self.assertEqual(result["coverage"]["n_classes_defined"], 1)
        self.assertEqual(result["per_class"][1]["reason"], "empty_input")
        self.assertEqual(result["per_class"][1]["coverage"]["excluded"], 2)
        json.dumps(result, allow_nan=False)

    def test_uncertain_missing_require_explicit_valid_policy(self):
        for target, policy in ((-1, "uncertain"), (None, "missing")):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                multilabel_macro_auroc([[target]], [[0]], labels=["a"], uncertain_policy="error", missing_policy="error")
        with self.assertRaises(TypeError):
            multilabel_macro_auroc([[0]], [[0]], labels=["a"])
        for kwargs in ({"uncertain_policy": "guess", "missing_policy": "error"},
                       {"uncertain_policy": "error", "missing_policy": "negative"}):
            with self.assertRaises(ValueError):
                multilabel_macro_auroc([[0]], [[0]], labels=["a"], **kwargs)


class InputAndDecisionTests(unittest.TestCase):
    def test_scores_preserve_exact_integer_and_fraction_order(self):
        b = 2 ** 53
        pairs = [(b, b + 1), (10 ** 1000, 10 ** 1000 + 1),
                 (Fraction(2 * b + 1, 2), Fraction(2 * b + 2, 2)),
                 (Fraction(0), Fraction(1, 10 ** 1000))]
        for low, high in pairs:
            with self.subTest(low=str(low), high=str(high)):
                self.assertEqual(auc([0, 1], [low, high])["value"], 1)
                self.assertEqual(ap([0, 1], [low, high])["value"], 1)
                self.assertEqual(binary_predictions([high], classes=[0, 1], positive_label=1,
                                                    threshold=low, comparison="gt"), [1])
                self.assertEqual(multiclass_predictions([[low, high]], classes=[0, 1], tie_break="error"), [1])
                self.assertEqual(macro_ovr_auroc([0, 1], [[high, low], [low, high]], classes=[0, 1])["value"], 1)
                result = multilabel_macro_auroc([[0], [1]], [[low], [high]], labels=["a"],
                                               uncertain_policy="error", missing_policy="error")
                self.assertEqual(result["value"], 1)
                json.dumps(result, allow_nan=False)

    def test_equal_values_remain_real_ties_across_numeric_types(self):
        for first, second in ((2 ** 53, 2 ** 53), (Fraction(1, 2), .5), (0, -0.0)):
            with self.subTest(first=first, second=second):
                self.assertEqual(auc([0, 1], [first, second])["value"], .5)
                self.assertEqual(ap([0, 1], [first, second])["value"], .5)
                self.assertEqual(binary_predictions([second], classes=[0, 1], positive_label=1,
                                                    threshold=first, comparison="gt"), [0])
                self.assertEqual(binary_predictions([second], classes=[0, 1], positive_label=1,
                                                    threshold=first, comparison="ge"), [1])
                self.assertEqual(multiclass_predictions([[first, second]], classes=[0, 1], tie_break="first"), [0])
                with self.assertRaises(ValueError):
                    multiclass_predictions([[first, second]], classes=[0, 1], tie_break="error")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional installed numpy unavailable")
    def test_numpy_scalars_preserve_their_represented_values_and_mixed_order(self):
        import numpy as np
        for dtype in (np.float16, np.float32, np.float64, np.longdouble):
            low = dtype(1)
            high = np.nextafter(low, dtype(2))
            with self.subTest(dtype=dtype.__name__):
                self.assertEqual(auc([0, 1], [low, high])["value"], 1)
                self.assertEqual(ap([0, 1], [low, high])["value"], 1)
                self.assertEqual(binary_predictions([high], classes=[0, 1], positive_label=1,
                                                    threshold=Fraction(1), comparison="gt"), [1])
                self.assertEqual(multiclass_predictions([[Fraction(1), high]], classes=[0, 1], tie_break="error"), [1])
                self.assertEqual(auc([0, 1], [low, Fraction(1)])["value"], .5)
        # NumPy's own mixed comparison can round an int to its float dtype;
        # these inputs nevertheless represent two distinct exact values.
        for low, high in ((np.float32(2 ** 24), 2 ** 24 + 1),
                          (np.float64(2 ** 53), 2 ** 53 + 1),
                          (np.int64(2 ** 53), np.int64(2 ** 53 + 1))):
            self.assertEqual(auc([0, 1], [low, high])["value"], 1)
            self.assertEqual(ap([0, 1], [low, high])["value"], 1)
            self.assertEqual(multiclass_predictions([[low, high]], classes=[0, 1], tie_break="error"), [1])
        # Rounding that occurred before this API cannot be undone; this is a tie.
        self.assertEqual(auc([0, 1], [np.float64(2 ** 53), np.float64(2 ** 53 + 1)])["value"], .5)

    def test_no_numpy_torch_or_sklearn_import_required(self):
        module = Path(__file__).resolve().parents[1] / "src/augment_atlas/clinical_metrics.py"
        script = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy','torch','torchvision','sklearn'}:
        raise AssertionError('unexpected dependency ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
mod = runpy.run_path(sys.argv[1])
assert mod['binary_auroc']([0,1], [0,1], classes=[0,1], positive_label=1)['value'] == 1
"""
        completed = subprocess.run([sys.executable, "-c", script, str(module)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_nan_inf_bool_and_shape_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), None, "1", True, [1]):
            with self.subTest(invalid=str(invalid)), self.assertRaises(ValueError):
                auc([0, 1], [0, invalid])
        for y, scores in (([0], []), ([[0]], [0]), ([None], [0]), ([True], [0]), ([2], [0]), ("01", [0, 1])):
            with self.assertRaises(ValueError):
                auc(y, scores)
        for scores in ([[0, 1]], [[0], [1]], [[0, 1], [float("nan"), 2]], [0, 1]):
            with self.assertRaises(ValueError):
                macro_ovr_auroc([0, 1], scores, classes=[0, 1])

    def test_invalid_class_universe_or_predictions(self):
        for classes in ([], [0], [0, 0], [0, True], [0, float("nan")], [0, " "], "ab"):
            with self.assertRaises(ValueError):
                macro_f1([0], [0], classes=classes)
        for y, pred in (([0], [2]), ([0], []), ([0], [None]), ([0], [[0]])):
            with self.assertRaises(ValueError):
                quadratic_weighted_kappa(y, pred, classes=[0, 1])
        with self.assertRaises(ValueError):
            binary_auroc([0, 1], [0, 1], classes=[0, 1, 2], positive_label=1)
        with self.assertRaises(ValueError):
            binary_auroc([0, 1], [0, 1], classes=[0, 1], positive_label=2)

    def test_multilabel_invalid_values_shape_and_excluded_score(self):
        for y, scores in (([[2]], [[1]]), ([[float("nan")]], [[1]]), ([[True]], [[1]]),
                          ([[0.0]], [[1]]), ([[0]], []), ([[0, 1]], [[1]]),
                          ([[None]], [[float("nan")]]), ([[-1]], [[None]])):
            with self.assertRaises(ValueError):
                multilabel_macro_auroc(y, scores, labels=["a"], uncertain_policy="exclude", missing_policy="exclude")

    def test_decisions_require_frozen_threshold_and_tie_rule(self):
        kwargs = dict(classes=["yes", "no"], positive_label="yes", threshold=.5)
        self.assertEqual(binary_predictions([.4, .5, .6], comparison="ge", **kwargs), ["no", "yes", "yes"])
        self.assertEqual(binary_predictions([.4, .5, .6], comparison="gt", **kwargs), ["no", "no", "yes"])
        self.assertEqual(multiclass_predictions([[1, 1], [0, 1]], classes=["b", "a"], tie_break="first"), ["b", "a"])
        with self.assertRaises(ValueError):
            multiclass_predictions([[1, 1]], classes=["b", "a"], tie_break="error")
        with self.assertRaises(TypeError):
            binary_predictions([.5], classes=[0, 1], positive_label=1)
        with self.assertRaises(TypeError):
            multiclass_predictions([[0, 1]], classes=[0, 1])
        with self.assertRaises(ValueError):
            binary_predictions([.5], comparison="ge", classes=[0, 1], positive_label=1, threshold=float("nan"))


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "optional installed sklearn unavailable; not installed by tests")
class InstalledSklearnParityTests(unittest.TestCase):
    def test_valid_binary_random_ties(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        rng = random.Random(42)
        for _ in range(20):
            y = [0, 1] + [rng.randrange(2) for _ in range(18)]
            scores = [rng.randrange(5) for _ in y]
            self.assertAlmostEqual(auc(y, scores)["value"], roc_auc_score(y, scores))
            self.assertAlmostEqual(ap(y, scores)["value"], average_precision_score(y, scores))

    def test_classification_and_ordinal(self):
        from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
        y, pred = [0, 0, 1, 2, 2], [0, 1, 1, 1, 2]
        self.assertAlmostEqual(macro_f1(y, pred, classes=[0, 1, 2])["value"], f1_score(y, pred, average="macro"))
        self.assertAlmostEqual(balanced_accuracy(y, pred, classes=[0, 1, 2])["value"], balanced_accuracy_score(y, pred))
        self.assertAlmostEqual(quadratic_weighted_kappa(y, pred, classes=list(range(5)))["value"],
                               cohen_kappa_score(y, pred, labels=list(range(5)), weights="quadratic"))
        scores = [[.7, .2, .1], [.5, .3, .2], [.3, .4, .3], [.1, .2, .7], [.2, .5, .3]]
        self.assertAlmostEqual(macro_ovr_auroc(y, scores, classes=[0, 1, 2])["value"],
                               roc_auc_score(y, scores, labels=[0, 1, 2], average="macro", multi_class="ovr"))


if __name__ == "__main__":
    unittest.main()
