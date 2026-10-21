"""Stdlib-only contract tests for the opt-in clinical evaluation artifact.

All data are synthetic. No torch, model execution or file I/O is involved in
building/validating the artifact; see test_runtime_predictions.py for the
host completion boundary and test_clinical_training.py for container CPU runs.
"""
import builtins
import copy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from augment_atlas.contracts import (BASE_REQUIRED_ARTIFACTS, canonical_hash, load_config, recipe_identity,
                                     required_artifacts, validate_config)
from augment_atlas.evaluation import (build_prediction_artifact, clinical_environment_block,
                                      evaluation_partition, partition_accuracy_bounds, selected_epoch,
                                      validate_evaluation_config, validate_prediction_artifact)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("clinical_fixtures_for_evaluation", ROOT / "tests/clinical_fixtures.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def roundtrip(value):
    return json.loads(json.dumps(value, allow_nan=False))


def graded_binary_scores(rows):
    """Negatives get positive-class 1/8,3/8,5/8; positives 3/8,5/8,7/8 -> AUROC 7/9."""
    queues = {0: [0.125, 0.375, 0.625], 1: [0.375, 0.625, 0.875]}
    output = []
    for row in rows:
        positive = queues[row["label"]].pop(0)
        output.append([1 - positive, positive])
    return output


def independent_qwk(truth, predicted, k):
    n = len(truth)
    observed = [[0] * k for _ in range(k)]
    for a, b in zip(truth, predicted):
        observed[a][b] += 1
    rows = [sum(observed[i]) for i in range(k)]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    numerator = sum((i - j) ** 2 * observed[i][j] for i in range(k) for j in range(k))
    denominator = sum(Fraction((i - j) ** 2 * rows[i] * cols[j], n) for i in range(k) for j in range(k))
    return 1 - Fraction(numerator) / denominator


class EvaluationConfigTests(unittest.TestCase):
    def test_valid_task_configs_are_accepted_without_injected_defaults(self):
        for evaluation in (fx.BINARY, fx.MULTICLASS, fx.ORDINAL):
            with self.subTest(task=evaluation["task"]):
                cfg = fx.clinical_config(evaluation)
                before = copy.deepcopy(cfg)
                self.assertIs(validate_config(cfg), cfg)
                self.assertEqual(cfg, before)
                self.assertEqual(cfg["evaluation"], evaluation)

    def test_required_artifacts_legacy_eight_opt_in_nine(self):
        legacy = load_config(ROOT / "configs/smoke/synthetic.json")
        self.assertEqual(required_artifacts(legacy), BASE_REQUIRED_ARTIFACTS)
        self.assertEqual(len(required_artifacts(legacy)), 8)
        opt_in = required_artifacts(fx.clinical_config())
        self.assertEqual(opt_in, BASE_REQUIRED_ARTIFACTS + ("predictions.json",))

    def assert_invalid(self, evaluation, pattern, num_classes=None):
        width = len(evaluation["classes"]) if isinstance(evaluation.get("classes"), list) else 2
        with self.assertRaisesRegex(ValueError, pattern):
            validate_evaluation_config(evaluation, width if num_classes is None else num_classes)

    def test_common_fields_are_strict(self):
        for base in (fx.BINARY, fx.MULTICLASS, fx.ORDINAL):
            cases = [
                ({"schema_version": 2}, "schema_version"), ({"schema_version": True}, "schema_version"),
                ({"schema_version": 1.0}, "schema_version"),
                ({"frozen": False}, "frozen"), ({"frozen": 1}, "frozen"),
                ({"score_kind": "logit"}, "score_kind"), ({"score_kind": ["softmax_probability"]}, "score_kind"),
            ]
            for change, pattern in cases:
                with self.subTest(task=base["task"], change=change):
                    self.assert_invalid({**base, **change}, pattern)
            for field in sorted(base):
                with self.subTest(task=base["task"], missing=field):
                    reduced = {k: v for k, v in base.items() if k != field}
                    with self.assertRaises(ValueError):
                        validate_evaluation_config(reduced, len(base["classes"]))
            with self.subTest(task=base["task"], extra="unknown"):
                self.assert_invalid({**base, "notes": "x"}, "explicit task-specific")

    def test_class_universe_must_match_model_columns(self):
        for classes, pattern in (([0, 1, 2], "one explicit label"), ([0, 0], "distinct"),
                                 ([True, False], "not bool"), ([0.0, 1.0], "not bool"),
                                 (["", "x"], "nonempty"), ([" ", "x"], "nonempty"),
                                 ("01", "one explicit label")):
            with self.subTest(classes=classes):
                self.assert_invalid({**fx.BINARY, "classes": classes, "positive_class": 1}, pattern, num_classes=2)
        with self.assertRaisesRegex(ValueError, "one explicit label"):
            validate_evaluation_config(fx.MULTICLASS, 2)

    def test_binary_specific_fields(self):
        for change, pattern in (({"positive_class": 2}, "positive_class"), ({"positive_class": True}, "positive_class"),
                                ({"positive_class": "1"}, "positive_class"),
                                ({"threshold": -0.01}, "threshold"), ({"threshold": 1.01}, "threshold"),
                                ({"threshold": float("nan")}, "threshold"), ({"threshold": float("inf")}, "threshold"),
                                ({"threshold": True}, "threshold"), ({"threshold": "0.5"}, "threshold"),
                                ({"threshold": None}, "threshold"),
                                ({"comparison": "gte"}, "comparison"), ({"comparison": None}, "comparison"),
                                ({"comparison": ["ge"]}, "comparison")):
            with self.subTest(change=change):
                self.assert_invalid({**fx.BINARY, **change}, pattern)
        for threshold in (0, 1, 0.0, 1.0, 0.5):
            validate_evaluation_config({**fx.BINARY, "threshold": threshold}, 2)
        # Binary decision fields are meaningless for argmax tasks and vice versa.
        self.assert_invalid({**fx.BINARY, "tie_break": "first"}, "explicit task-specific")
        self.assert_invalid({**fx.MULTICLASS, "threshold": 0.5}, "explicit task-specific")
        binary_three = {**fx.BINARY, "classes": [0, 1, 2]}
        self.assert_invalid(binary_three, "two classes", num_classes=3)

    def test_argmax_tie_break_policy(self):
        for base in (fx.MULTICLASS, fx.ORDINAL):
            for value in ("last", "random", None, True, ["first"]):
                with self.subTest(task=base["task"], value=value):
                    self.assert_invalid({**base, "tie_break": value}, "tie_break")

    def test_unsupported_tasks(self):
        self.assert_invalid({**fx.MULTICLASS, "task": "multilabel"}, "multilabel trainer")
        for task in ("regression", None, 1, "Binary"):
            with self.subTest(task=task):
                self.assert_invalid({**fx.BINARY, "task": task}, "evaluation.task")
        with self.assertRaisesRegex(ValueError, "explicit object"):
            validate_evaluation_config([fx.BINARY], 2)

    def test_validate_config_rejects_invalid_opt_in(self):
        cfg = fx.clinical_config()
        cfg["evaluation"]["threshold"] = 2
        with self.assertRaisesRegex(ValueError, "threshold"):
            validate_config(cfg)


class DecisionAndMetricTests(unittest.TestCase):
    def inputs(self, evaluation=fx.BINARY, *, groups=30, scope=None):
        cfg, records, splits, environment = fx.build_inputs(evaluation, groups=groups, scope=scope)
        return cfg, records, splits, environment, fx.partition_rows(cfg, splits, records)

    def build(self, cfg, records, splits, environment, scores, best_epoch=2):
        return build_prediction_artifact(cfg, splits, records, scores, best_epoch=best_epoch, environment=environment)

    def test_binary_threshold_ge_and_gt_differ_only_at_equality(self):
        decisions = {}
        for comparison in ("ge", "gt"):
            cfg, records, splits, environment, rows = self.inputs({**fx.BINARY, "comparison": comparison})
            scores = [[0.5, 0.5]] * len(rows)
            artifact, _ = self.build(cfg, records, splits, environment, scores)
            decisions[comparison] = [row["prediction"] for row in artifact["rows"]]
            above = [[0.4375, 0.5625]] * len(rows)
            artifact, _ = self.build(cfg, records, splits, environment, above)
            self.assertEqual([row["prediction"] for row in artifact["rows"]], [1] * len(rows))
        self.assertEqual(decisions["ge"], [1] * len(rows))
        self.assertEqual(decisions["gt"], [0] * len(rows))

    def test_binary_threshold_bounds_are_inclusive_policies(self):
        cfg, records, splits, environment, rows = self.inputs({**fx.BINARY, "threshold": 0, "comparison": "ge"})
        artifact, _ = self.build(cfg, records, splits, environment, [[1.0, 0.0]] * len(rows))
        self.assertEqual({row["prediction"] for row in artifact["rows"]}, {1})
        cfg, records, splits, environment, rows = self.inputs({**fx.BINARY, "threshold": 1, "comparison": "gt"})
        artifact, _ = self.build(cfg, records, splits, environment, [[0.0, 1.0]] * len(rows))
        self.assertEqual({row["prediction"] for row in artifact["rows"]}, {0})

    def test_positive_class_column_follows_class_order(self):
        evaluation = {**fx.BINARY, "classes": ["benign", "malignant"], "positive_class": "benign"}
        cfg, records, splits, environment, rows = self.inputs(evaluation)
        # Peaked rows put 3/4 on each row's own label column. Using column 0 for
        # the positive "benign" class gives AUROC 1; the wrong column would give 0.
        artifact, result = self.build(cfg, records, splits, environment, fx.peaked_scores(rows, 2))
        self.assertEqual(result["metrics"]["binary_auroc"]["positive_label"], "benign")
        self.assertEqual(result["metrics"]["binary_auroc"]["value"], 1.0)
        self.assertEqual([row["prediction"] for row in artifact["rows"]],
                         [["benign", "malignant"][row["label"]] for row in rows])
        self.assertEqual([row["truth"] for row in artifact["rows"]],
                         [["benign", "malignant"][row["label"]] for row in rows])

    def test_binary_metrics_match_hand_computation(self):
        cfg, records, splits, environment, rows = self.inputs()
        self.assertEqual(sorted(row["label"] for row in rows), [0, 0, 0, 1, 1, 1])
        artifact, result = self.build(cfg, records, splits, environment, graded_binary_scores(rows))
        metrics = result["metrics"]
        self.assertEqual(result["primary_metric"], "binary_auroc")
        self.assertAlmostEqual(metrics["binary_auroc"]["value"], 7 / 9, places=15)
        self.assertIsNotNone(metrics["binary_average_precision"]["value"])
        self.assertEqual(set(metrics), {"binary_auroc", "binary_average_precision", "macro_f1", "balanced_accuracy"})
        # ge 0.5: negatives 5/8 -> FP; positives 5/8,7/8 -> TP, 3/8 -> FN.
        self.assertAlmostEqual(metrics["balanced_accuracy"]["value"], (2 / 3 + 2 / 3) / 2)

    def test_multiclass_tie_policies(self):
        for evaluation in (fx.MULTICLASS, fx.ORDINAL):
            cfg, records, splits, environment, rows = self.inputs(evaluation, groups=34)
            tied = [[0.375, 0.375, 0.25]] * len(rows)
            artifact, _ = self.build(cfg, records, splits, environment, tied)
            self.assertEqual({row["prediction"] for row in artifact["rows"]}, {evaluation["classes"][0]})
            error = {**evaluation, "tie_break": "error"}
            cfg, records, splits, environment, rows = self.inputs(error, groups=34)
            with self.subTest(task=evaluation["task"]), self.assertRaisesRegex(ValueError, "tied maximum"):
                self.build(cfg, records, splits, environment, tied)
            untied = fx.peaked_scores(rows, 3)
            self.build(cfg, records, splits, environment, untied)

    def test_multiclass_defined_macro_auroc_when_every_class_is_present(self):
        cfg, records, splits, environment, rows = self.inputs(fx.MULTICLASS, groups=34)
        self.assertEqual({row["label"] for row in rows}, {0, 1, 2})
        _, result = self.build(cfg, records, splits, environment, fx.peaked_scores(rows, 3))
        self.assertEqual(result["primary_metric"], "macro_ovr_auroc")
        self.assertEqual(result["metrics"]["macro_ovr_auroc"]["value"], 1.0)
        self.assertEqual(result["metrics"]["balanced_accuracy"]["value"], 1.0)
        self.assertNotIn("binary_average_precision", result["metrics"])

    def test_absent_class_gives_explicit_undefined_not_dropped(self):
        cfg, records, splits, environment, rows = self.inputs(fx.MULTICLASS, groups=30)
        self.assertEqual({row["label"] for row in rows}, {1, 2})
        _, result = self.build(cfg, records, splits, environment, fx.peaked_scores(rows, 3))
        macro = result["metrics"]["macro_ovr_auroc"]
        self.assertIsNone(macro["value"])
        self.assertEqual(macro["reason"], "undefined_class_components")
        self.assertEqual(macro["coverage"], {"n_samples": len(rows), "n_classes_expected": 3, "n_classes_defined": 2})
        self.assertEqual([item["label"] for item in macro["per_class"]], fx.MULTICLASS["classes"])
        self.assertEqual(macro["per_class"][0]["reason"], "no_positive_examples")
        self.assertIsNone(result["metrics"]["balanced_accuracy"]["value"])
        json.dumps(result, allow_nan=False)
        # Binary selection-only partition with only positives.
        cfg, records, splits, environment, rows = self.inputs(fx.BINARY, groups=12, scope="selection-only")
        self.assertEqual({row["label"] for row in rows}, {1})
        _, result = self.build(cfg, records, splits, environment, fx.peaked_scores(rows, 2))
        auroc = result["metrics"]["binary_auroc"]
        self.assertIsNone(auroc["value"])
        self.assertEqual(auroc["reason"], "no_negative_examples")
        self.assertEqual(auroc["coverage"]["negative"], 0)
        self.assertEqual(result["metrics"]["binary_average_precision"]["value"], 1.0)
        self.assertIsNone(result["metrics"]["balanced_accuracy"]["value"])

    def test_ordinal_qwk_matches_independent_formula(self):
        cfg, records, splits, environment, rows = self.inputs(fx.ORDINAL, groups=34)
        truth = [row["label"] for row in rows]
        _, perfect = self.build(cfg, records, splits, environment, fx.peaked_scores(rows, 3))
        self.assertEqual(perfect["primary_metric"], "quadratic_weighted_kappa")
        self.assertEqual(perfect["metrics"]["quadratic_weighted_kappa"]["value"], 1.0)
        self.assertNotIn("macro_ovr_auroc", perfect["metrics"])
        wrong = {0, 3}
        artifact, result = self.build(cfg, records, splits, environment, fx.peaked_scores(rows, 3, wrong=wrong))
        predicted = [row["prediction"] for row in artifact["rows"]]
        self.assertEqual(predicted, [(t + 1) % 3 if i in wrong else t for i, t in enumerate(truth)])
        expected = independent_qwk(truth, predicted, 3)
        self.assertTrue(math.isclose(result["metrics"]["quadratic_weighted_kappa"]["value"], float(expected),
                                     rel_tol=0, abs_tol=1e-12))

    def test_nonfinite_and_out_of_range_probabilities_rejected(self):
        cfg, records, splits, environment, rows = self.inputs()
        good = fx.peaked_scores(rows, 2)
        for bad in (float("nan"), float("inf"), float("-inf"), -0.25, 1.25, True, None, "0.5"):
            scores = copy.deepcopy(good)
            scores[0] = [bad, 0.5]
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite JSON numbers"):
                self.build(cfg, records, splits, environment, scores)

    def test_probability_sum_tolerance(self):
        cfg, records, splits, environment, rows = self.inputs()
        good = fx.peaked_scores(rows, 2)
        for row, accepted in (([0.5, 0.25], False), ([0.5, 0.5 + 2e-6], False), ([0.5, 0.5 - 2e-6], False),
                              ([0.5, 0.5 + 5e-7], True), ([0.5, 0.5 - 5e-7], True), ([1, 0], True)):
            scores = copy.deepcopy(good)
            scores[0] = row
            with self.subTest(row=row):
                if accepted:
                    self.build(cfg, records, splits, environment, scores)
                else:
                    with self.assertRaisesRegex(ValueError, "sum to one"):
                        self.build(cfg, records, splits, environment, scores)

    def test_row_count_and_width_must_match_partition_and_classes(self):
        cfg, records, splits, environment, rows = self.inputs()
        good = fx.peaked_scores(rows, 2)
        for scores, pattern in ((good[:-1], "exact evaluation partition"), (good + [good[0]], "exact evaluation partition"),
                                ([], "exact evaluation partition"), (tuple(good), "exact evaluation partition"),
                                ([[0.5, 0.25, 0.25]] + good[1:], "fixed class order"),
                                ([[1.0]] + good[1:], "fixed class order"),
                                ([(0.25, 0.75)] + good[1:], "fixed class order")):
            with self.subTest(scores=str(scores)[:40]), self.assertRaisesRegex(ValueError, pattern):
                self.build(cfg, records, splits, environment, scores)

    def test_best_epoch_must_be_a_configured_epoch(self):
        cfg, records, splits, environment, rows = self.inputs()
        for epoch in (0, 3, True, 1.0, None):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "best_epoch"):
                self.build(cfg, records, splits, environment, fx.peaked_scores(rows, 2), best_epoch=epoch)

    def test_helpers_selected_epoch_and_accuracy_bounds(self):
        history = [{"epoch": 1, "selection": {"loss": 0.5}}, {"epoch": 2, "selection": {"loss": 0.4}},
                   {"epoch": 3, "selection": {"loss": 0.4}}]
        self.assertEqual(selected_epoch(history), (2, 0.4))
        for broken in ([], [{"epoch": 2, "selection": {"loss": 0.1}}], [{"epoch": 1, "selection": {"loss": float("nan")}}],
                       [{"epoch": 1, "selection": 0.1}], [{"epoch": True, "selection": {"loss": 0.1}}], ["row"]):
            with self.subTest(broken=broken), self.assertRaisesRegex(ValueError, "selection"):
                selected_epoch(broken)
        artifact = {"rows": [{"scores": [0.25, 0.75], "target_index": 1}, {"scores": [0.5, 0.5], "target_index": 0},
                             {"scores": [0.75, 0.25], "target_index": 1}]}
        self.assertEqual(partition_accuracy_bounds(artifact), (1, 2))


class ArtifactIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.records, self.splits, self.environment = fx.build_inputs()
        self.rows = fx.partition_rows(self.cfg, self.splits, self.records)
        self.scores = graded_binary_scores(self.rows)
        artifact, result = build_prediction_artifact(self.cfg, self.splits, self.records, self.scores,
                                                     best_epoch=2, environment=self.environment)
        self.artifact, self.result = roundtrip(artifact), roundtrip(result)

    def validate(self, artifact=None, result=None, *, epoch=2, environment=None, records=None, splits=None):
        return validate_prediction_artifact(self.cfg, self.splits if splits is None else splits,
                                            self.records if records is None else records,
                                            self.artifact if artifact is None else artifact,
                                            self.result if result is None else result,
                                            expected_best_epoch=epoch,
                                            environment=self.environment if environment is None else environment)

    def assert_rejected(self, pattern="mismatch", **kwargs):
        with self.assertRaisesRegex(ValueError, pattern):
            self.validate(**kwargs)

    def test_untouched_json_round_trip_validates(self):
        summary = self.validate()
        self.assertEqual(summary["samples"], len(self.rows))
        self.assertEqual(summary["partition"], "test")
        self.assertEqual(summary["predictions_content_sha256"], canonical_hash(self.artifact))
        self.assertEqual([row["id"] for row in self.artifact["rows"]], self.splits["lists"]["test"])

    def test_provenance_fields(self):
        expected = {"config_sha256": canonical_hash(self.cfg), "splits_sha256": self.splits["sha256"],
                    "recipe_sha256": recipe_identity(self.cfg), "environment_sha256": canonical_hash(self.environment),
                    "evaluation_sha256": canonical_hash(self.cfg["evaluation"]), "best_epoch": 2,
                    "partition": "test", "evaluation_scope": "test", "task": "binary", "classes": [0, 1],
                    "score_kind": "softmax_probability"}
        for document in (self.artifact, self.result):
            self.assertEqual({key: document[key] for key in expected}, expected)
        self.assertEqual(self.artifact["visibility"], "private")
        self.assertEqual(self.artifact["evaluation"], self.cfg["evaluation"])
        for key, value in (("scientifically_eligible", False), ("execution_verified", False),
                           ("frozen_declaration_only", True), ("scientific_status", "provisional"),
                           ("predictions_content_sha256", canonical_hash(self.artifact)),
                           ("prediction_artifact", "predictions.json")):
            self.assertEqual(self.result[key], value)
        self.assertEqual(self.environment["clinical_evaluation"], clinical_environment_block(self.cfg))
        self.assertFalse(self.environment["clinical_evaluation"]["scientifically_eligible"])

    def test_aggregate_result_contains_no_row_identifiers(self):
        text = json.dumps(self.result)
        for record in self.records:
            self.assertNotIn(record["id"], text)
            self.assertNotIn(record["group"], text)
        self.assertNotIn("rows", self.result)

    def mutate_rows(self, operation):
        artifact = copy.deepcopy(self.artifact)
        operation(artifact["rows"])
        return artifact

    def test_row_tampering_rejected(self):
        def swap_ids(rows):
            rows[0]["id"], rows[1]["id"] = rows[1]["id"], rows[0]["id"]

        def swap_groups(rows):
            rows[0]["group"], rows[1]["group"] = rows[1]["group"], rows[0]["group"]

        def change_group(rows):
            rows[0]["group"] = "synthetic-group-9999"

        def flip_truth(rows):
            rows[0]["target_index"] = 1 - rows[0]["target_index"]
            rows[0]["truth"] = [0, 1][rows[0]["target_index"]]

        def flip_truth_label_only(rows):
            rows[0]["truth"] = 1 - rows[0]["truth"]

        def flip_prediction(rows):
            rows[0]["prediction"] = 1 - rows[0]["prediction"]

        def reorder(rows):
            rows.reverse()

        def foreign_id(rows):
            rows[0]["id"] = self.splits["lists"]["train"][0]

        for operation in (swap_ids, swap_groups, change_group, flip_truth, flip_truth_label_only,
                          flip_prediction, reorder, foreign_id):
            with self.subTest(operation=operation.__name__):
                self.assert_rejected("provenance/rows/decisions", artifact=self.mutate_rows(operation))
        with self.subTest(operation="drop row"):
            self.assert_rejected("exact evaluation partition", artifact=self.mutate_rows(lambda rows: rows.pop()))
        with self.subTest(operation="row missing scores"):
            self.assert_rejected("must contain scores",
                                 artifact=self.mutate_rows(lambda rows: rows[0].pop("scores")))

    def test_score_change_with_resealed_content_hash_is_rejected_by_metrics(self):
        artifact = copy.deepcopy(self.artifact)
        negative = next(row for row in artifact["rows"] if row["target_index"] == 0)
        negative["scores"] = [0.0625, 0.9375]
        negative["prediction"] = 1
        result = copy.deepcopy(self.result)
        result["predictions_content_sha256"] = canonical_hash(artifact)  # "resealed"
        # The rows alone are self-consistent now; aggregate metrics are not.
        self.assert_rejected("clinical result", artifact=artifact, result=result)
        # Without the reseal the hash itself already differs.
        self.assert_rejected("clinical result", artifact=artifact)

    def test_metric_and_result_tampering_rejected(self):
        def bump_auroc(result):
            result["metrics"]["binary_auroc"]["value"] += 0.01

        def undefined_auroc(result):
            result["metrics"]["binary_auroc"].update(value=None, reason="no_positive_examples")

        def coverage(result):
            result["metrics"]["binary_auroc"]["coverage"]["positive"] += 1

        def drop_secondary(result):
            result["metrics"].pop("macro_f1")

        def eligible(result):
            result["scientifically_eligible"] = True

        def executed(result):
            result["execution_verified"] = True

        def primary(result):
            result["primary_metric"] = "balanced_accuracy"

        for operation in (bump_auroc, undefined_auroc, coverage, drop_secondary, eligible, executed, primary):
            result = copy.deepcopy(self.result)
            operation(result)
            with self.subTest(operation=operation.__name__):
                self.assert_rejected("clinical result", result=result)
        self.assert_rejected("clinical result", result=[])

    def test_provenance_tampering_rejected_in_artifact_and_result(self):
        changes = {"config_sha256": "0" * 64, "splits_sha256": "0" * 64, "recipe_sha256": "0" * 64,
                   "environment_sha256": "0" * 64, "evaluation_sha256": "0" * 64, "best_epoch": 1,
                   "partition": "selection", "evaluation_scope": "selection-only", "task": "multiclass",
                   "classes": [1, 0], "score_kind": "logit", "visibility": "public"}
        for key, value in changes.items():
            if key in self.artifact:
                artifact = copy.deepcopy(self.artifact)
                artifact[key] = value
                with self.subTest(document="artifact", key=key):
                    self.assert_rejected("provenance", artifact=artifact)
            if key in self.result:
                result = copy.deepcopy(self.result)
                result[key] = value
                with self.subTest(document="result", key=key):
                    self.assert_rejected("clinical result", result=result)

    def test_expected_epoch_and_environment_are_anchors(self):
        self.assert_rejected("provenance", epoch=1)
        self.assert_rejected("best_epoch", epoch=3)
        for change in ({"config_sha256": "0" * 64}, {"splits_sha256": "0" * 64}, {"evaluation_scope": "selection-only"},
                       {"clinical_evaluation": {**self.environment["clinical_evaluation"], "scientifically_eligible": True}}):
            with self.subTest(change=change):
                self.assert_rejected("environment does not declare", environment={**self.environment, **change})
        # Any other environment change alters environment_sha256 provenance.
        self.assert_rejected("provenance", environment={**self.environment, "code_commit": "c" * 40})
        without = {k: v for k, v in self.environment.items() if k != "clinical_evaluation"}
        self.assert_rejected("environment does not declare", environment=without)

    def test_trusted_records_are_required_and_authoritative(self):
        with self.assertRaises(TypeError):
            validate_prediction_artifact(self.cfg, self.splits, self.artifact, self.result,
                                         expected_best_epoch=2, environment=self.environment)
        with self.assertRaises(TypeError):
            validate_prediction_artifact(self.cfg, self.splits, self.records, self.artifact, self.result,
                                         expected_best_epoch=2)
        test_id = self.splits["lists"]["test"][0]
        regrouped = [{**record, "group": "synthetic-group-moved"} if record["id"] == test_id else record
                     for record in self.records]
        self.assert_rejected("identity mismatch", records=regrouped)
        relabeled = [{**record, "label": 1 - record["label"]} if record["id"] == test_id else record
                     for record in self.records]
        self.assert_rejected("identity mismatch", records=relabeled)
        self.assert_rejected("nonempty list", records=[])
        self.assert_rejected("valid id/group", records=[{**self.records[0], "label": True}] + self.records[1:])

    def test_split_tampering_rejected_even_when_resealed(self):
        splits = copy.deepcopy(self.splits)
        selection, test = splits["lists"]["selection"], splits["lists"]["test"]
        selection[0], test[0] = test[0], selection[0]
        selection.sort(), test.sort()
        splits["sha256"] = canonical_hash({k: v for k, v in splits.items() if k != "sha256"})
        self.assert_rejected("group mapping", splits=splits)
        lookup = {record["id"]: record for record in self.records}
        splits["groups"] = {name: sorted({lookup[i]["group"] for i in ids}) for name, ids in splits["lists"].items()}
        splits["sha256"] = canonical_hash({k: v for k, v in splits.items() if k != "sha256"})
        self.assert_rejected("environment does not declare|provenance", splits=splits)
        unsealed = copy.deepcopy(self.splits)
        unsealed["lists"]["test"].reverse()
        self.assert_rejected("checksum", splits=unsealed)

    def test_inputs_are_not_mutated(self):
        before = copy.deepcopy((self.cfg, self.splits, self.records, self.scores, self.environment,
                                self.artifact, self.result))
        build_prediction_artifact(self.cfg, self.splits, self.records, self.scores, best_epoch=2,
                                  environment=self.environment)
        self.validate()
        self.assertEqual((self.cfg, self.splits, self.records, self.scores, self.environment,
                          self.artifact, self.result), before)


class SelectionOnlyEvaluationTests(unittest.TestCase):
    def test_selection_only_uses_only_selection_rows_and_performs_no_io(self):
        cfg, records, splits, environment = fx.build_inputs(scope="selection-only")
        self.assertEqual(evaluation_partition(cfg), "selection")
        rows = fx.partition_rows(cfg, splits, records)
        forbidden = AssertionError("clinical evaluation must not perform file I/O")
        with patch.object(builtins, "open", side_effect=forbidden), \
             patch.object(Path, "open", side_effect=forbidden), \
             patch.object(Path, "read_text", side_effect=forbidden), \
             patch.object(Path, "read_bytes", side_effect=forbidden):
            artifact, result = build_prediction_artifact(cfg, splits, records, fx.peaked_scores(rows, 2),
                                                         best_epoch=1, environment=environment)
            validate_prediction_artifact(cfg, splits, records, roundtrip(artifact), roundtrip(result),
                                         expected_best_epoch=1, environment=environment)
        ids = [row["id"] for row in artifact["rows"]]
        self.assertEqual(ids, splits["lists"]["selection"])
        self.assertFalse(set(ids) & set(splits["lists"]["test"]))
        self.assertEqual((artifact["partition"], artifact["evaluation_scope"]), ("selection", "selection-only"))
        self.assertEqual(result["samples"], len(splits["lists"]["selection"]))

    def test_scope_mismatch_between_config_and_environment_rejected(self):
        cfg, records, splits, environment = fx.build_inputs(scope="selection-only")
        rows = fx.partition_rows(cfg, splits, records)
        with self.assertRaisesRegex(ValueError, "environment does not declare"):
            build_prediction_artifact(cfg, splits, records, fx.peaked_scores(rows, 2), best_epoch=1,
                                      environment={**environment, "evaluation_scope": "test"})

    def test_selection_only_still_requires_within_outer_fold_protocol(self):
        cfg = fx.clinical_config(scope="selection-only")
        cfg["recipe"]["hpo"]["protocol"] = "global-development"
        with self.assertRaisesRegex(ValueError, "selection-only"):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
