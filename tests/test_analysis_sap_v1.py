"""Synthetic-only SAP checks; no image or real prediction access."""
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import ndtr

from augment_atlas import analysis_sap_v1 as sap
from augment_atlas import clinical_metrics as cm
from augment_atlas.contracts import ARMS, canonical_hash


def synthetic(seed=0, *, groups_per_class=12, images_per_group=2, arms=ARMS, seeds=(1,)):
    rng = np.random.default_rng(seed)
    rows = []
    for fold in sap.FOLDS:
        for label in (0, 1):
            for g in range(groups_per_class):
                shared = rng.normal() / math.sqrt(2)
                for image in range(images_per_group):
                    z = shared + rng.normal() / math.sqrt(2)
                    for train_seed in seeds:
                        for arm in arms:
                            # Paired Gaussian scores; positive-only shift 0.6.
                            # AUC control=0.5; treatment=Phi(0.6/sqrt(2)).
                            shift = .6 if "gamma" in arm else .3 if "clahe" in arm else 0.
                            rows.append(dict(dataset="synthetic", arm=arm, fold=fold, seed=train_seed,
                                             image=f"image-{fold}-{label}-{g}-{image}",
                                             group=f"group-{fold}-{label}-{g}", label=label,
                                             score=float(ndtr(z + shift * label))))
    return rows


def independent_arm_scores(seed, *, shift):
    """Five folds, 16 groups/class, two images/group; independent arm noise.

    The common Gaussian score has variance 1 (half group, half image).
    Each arm adds N(0, 0.5) noise, giving marginal variance 1.25 and
    population delta AUROC Phi(shift / sqrt(2 * 1.25)) - 0.5.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fold in sap.FOLDS:
        for label in (0, 1):
            for group in range(16):
                shared = rng.normal() / math.sqrt(2)
                for image in range(2):
                    z_common = shared + rng.normal() / math.sqrt(2)
                    for arm in ("basic", "basic_gamma_light", "basic_gamma_heavy"):
                        arm_shift = 0. if arm == "basic" else shift
                        score = z_common + arm_shift * label + rng.normal(0., .5)
                        rows.append(dict(dataset="synthetic", arm=arm, fold=fold, seed=1,
                                         image=f"image-{fold}-{label}-{group}-{image}",
                                         group=f"group-{fold}-{label}-{group}", label=label,
                                         score=float(ndtr(score))))
    return rows


def by_id(report, name):
    return next(c for c in report["primary"]["contrasts"] if c["id"] == name)


def native_artifact(arm="basic", fold=0):
    # Native closed schema, synthetic probabilities; intentionally positive
    # class first to ensure column selection does not assume classes[1].
    value = dict(schema_version=1, artifact_type="global_predictions", visibility="private", phase="final",
                 dataset={"id": "synthetic", "version": "fixture"},
                 task={"type": "binary", "classes": [4, 9], "positive_class": 4},
                 pools_sha256="a" * 64, view_sha256="b" * 64, plan_sha256=None, candidate_id=None,
                 freeze_sha256="c" * 64, fold=fold, arm=arm, train_seed=1, config_sha256="d" * 64,
                 checkpoint_selection="last_epoch", epoch=20, epochs=20, partition="E_test",
                 score_kind="softmax_probability", decision={"rule": "fixed_threshold"},
                 rows=[{"id": "pos", "group": "p", "target": 4, "scores": [.8, .2], "prediction": 4},
                       {"id": "neg", "group": "n", "target": 9, "scores": [.3, .7], "prediction": 9}],
                 metrics={})
    value["content_sha256"] = canonical_hash(value)
    return value


class SapTests(unittest.TestCase):
    def test_fold_point_is_exact_clinical_auroc_including_ties(self):
        rows = synthetic(groups_per_class=5)
        for row in rows:
            row["score"] = round(row["score"], 1)
        report = sap.analyze(rows, dataset="synthetic", B=79, seed=12)
        for arm in report["primary"]["arms"]:
            expected = []
            for fold in arm["per_fold"]:
                cell = sorted((r for r in rows if r["arm"] == arm["arm"] and r["fold"] == fold["fold"]),
                              key=lambda r: r["image"])
                value = cm.binary_auroc([r["label"] for r in cell], [r["score"] for r in cell],
                                        classes=[0, 1], positive_label=1)["value"]
                self.assertEqual(value, fold["auroc"])
                expected.append(value)
            self.assertEqual(math.fsum(expected) / 5, arm["metrics"]["auroc"])

    def test_weighted_auroc_and_ap_match_explicit_replication(self):
        labels = [1, 0, 1, 0, 0, 1]
        scores = [.8, .8, .2, .1, .2, .8]
        groups = [0, 0, 1, 2, 3, 3]  # mixed-label groups and ties
        counts = np.array([[1, 1, 1, 1], [0, 3, 2, 1], [4, 1, 0, 2], [1, 0, 0, 0]])
        values = sap.weighted_metrics(sap.rank_plan(labels, scores, groups), counts)
        for actual, row in zip(values, counts):
            yy, ss = [], []
            for y, s, g in zip(labels, scores, groups):
                yy.extend([y] * row[g])
                ss.extend([s] * row[g])
            auc = cm.binary_auroc(yy, ss, classes=[0, 1], positive_label=1)["value"]
            ap = cm.binary_average_precision(yy, ss, classes=[0, 1], positive_label=1)["value"]
            self.assertEqual(actual[0], auc)
            self.assertAlmostEqual(actual[1], ap, places=15)

    def _measure_coverage(self, simulations, minimum):
        truth = float(ndtr(.6 / math.sqrt(2 * 1.25)) - .5)
        covered = 0
        for simulation in range(simulations):
            rows = independent_arm_scores(simulation, shift=.6)
            result = sap.analyze(rows, dataset="synthetic", B=299, seed=10000 + simulation)
            value = by_id(result, "H1")["auroc"]
            self.assertEqual(value["status"], "ok")
            ci = value["ci95"]
            covered += ci[0] <= truth <= ci[1]
        coverage = covered / simulations
        print(f"COVERAGE independent-arm Gaussian groups: {covered}/{simulations} = {coverage:.3f}; "
              f"true_delta={truth:.12f}; B=299; 5 folds x (16 pos + 16 neg) groups x 2 images", flush=True)
        self.assertGreaterEqual(coverage, minimum)

    def test_known_nonzero_delta_coverage_fast(self):
        self._measure_coverage(60, .85)

    @unittest.skipUnless(os.environ.get("ATLAS_SLOW_TESTS") == "1", "requires ATLAS_SLOW_TESTS=1")
    def test_known_nonzero_delta_coverage_slow(self):
        self._measure_coverage(200, .90)

    @unittest.skipUnless(os.environ.get("ATLAS_SLOW_TESTS") == "1", "requires ATLAS_SLOW_TESTS=1")
    def test_type_i_error_under_null_slow(self):
        simulations, rejected = 200, 0
        for simulation in range(simulations):
            rows = independent_arm_scores(20000 + simulation, shift=0.)
            result = sap.analyze(rows, dataset="synthetic", B=299, seed=30000 + simulation)
            value = by_id(result, "H1")["auroc"]
            self.assertEqual(value["status"], "ok")
            rejected += value["p_boot"] <= .05
        fraction = rejected / simulations
        print(f"TYPE_I independent-arm Gaussian H0 (shift=0): {rejected}/{simulations} = {fraction:.3f}; "
              "p_boot<=0.05; B=299; 5 folds x (16 pos + 16 neg) groups x 2 images", flush=True)
        self.assertGreaterEqual(fraction, .01)
        self.assertLessEqual(fraction, .10)

    def test_missing_fold_and_single_class_are_not_evaluable(self):
        rows = synthetic(groups_per_class=3)
        incomplete = [r for r in rows if not (r["arm"] == "basic_gamma_heavy" and r["fold"] == 4)]
        result = sap.analyze(incomplete, dataset="synthetic", B=29)
        self.assertEqual(by_id(result, "H1")["status"], "not_evaluable")
        self.assertIsNone(by_id(result, "H1")["auroc"]["estimate"])
        self.assertEqual(by_id(result, "H2")["status"], "ok")
        no_positive = [dict(r, label=0) if r["fold"] == 0 else r for r in rows]
        result = sap.analyze(no_positive, dataset="synthetic", B=29)
        self.assertEqual(by_id(result, "H2")["status"], "not_evaluable")
        no_negative = [dict(r, label=1) if r["fold"] == 0 else r for r in rows]
        self.assertEqual(by_id(sap.analyze(no_negative, dataset="synthetic", B=29), "H1")["status"], "not_evaluable")

    def test_determinism_permutation_and_block_size(self):
        rows = synthetic(groups_per_class=5)
        first = sap.analyze(rows, dataset="synthetic", B=87, seed=53, block_size=7)
        second = sap.analyze(list(reversed(rows)), dataset="synthetic", B=87, seed=53, block_size=7)
        self.assertEqual(first, second)
        third = sap.analyze(rows, dataset="synthetic", B=87, seed=53, block_size=32)
        first["bootstrap"]["block_size"] = 32
        self.assertEqual(first, third)
        other = sap.analyze(rows, dataset="synthetic", B=87, seed=54)
        self.assertNotEqual(by_id(first, "H1")["auroc"]["ci95"], by_id(other, "H1")["auroc"]["ci95"])

    def test_holm_and_bootstrap_p_manual(self):
        self.assertEqual(sap.holm_adjust({"H1": .01, "H2": .04}), {"H1": .02, "H2": .04})
        self.assertEqual(sap.holm_adjust({"H1": .04, "H2": .03}), {"H2": .06, "H1": .06})
        self.assertEqual(sap.holm_adjust({"H1": .02, "H2": None}), {"H1": .04, "H2": None})
        draws = np.array([-2., -1., 0., 1., 2., 3., 4., 5., 6.])
        result = sap._interval(1., draws, primary=True, inference=True)
        self.assertEqual(result["p_boot"], .8)  # 2 * (3 + 1) / (9 + 1)
        self.assertEqual(result["ci97_5"], np.quantile(draws, [.0125, .9875]).tolist())
        self.assertEqual(sap._interval(0., np.zeros(9), primary=True, inference=True)["p_boot"], 1.)

    def test_all_contrasts_and_no_secondary_p_or_brier(self):
        result = sap.analyze(synthetic(groups_per_class=3), dataset="synthetic", B=29)
        self.assertEqual(len(result["primary"]["contrasts"]), 15)
        values = {r["arm"]: r["metrics"] for r in result["primary"]["arms"]}
        for c in result["primary"]["contrasts"]:
            for metric in ("auroc", "ap"):
                self.assertEqual(c[metric]["estimate"], math.fsum(w * values[a][metric] for a, w in c["coefficients"].items()))
                self.assertIsNotNone(c[metric]["ci95"])
            self.assertNotIn("p_boot", c["ap"])
            if not c["primary"]:
                self.assertNotIn("p_boot", c["auroc"])
        self.assertNotIn("brier", json.dumps(result).lower())

    def test_seed_one_primary_and_complete_three_seed_sensitivity(self):
        rows = synthetic(groups_per_class=3, seeds=(1, 2, 3))
        for row in rows:
            if row["seed"] > 1 and "gamma" in row["arm"]:
                row["score"] = 1. - row["score"]
        result = sap.analyze(rows, dataset="synthetic", B=39)
        one = sap.analyze([r for r in rows if r["seed"] == 1], dataset="synthetic", B=39)
        self.assertEqual(result["primary"], one["primary"])
        sensitivity = result["sensitivities"]["three_seed_mean"]
        self.assertEqual(sensitivity["status"], "ok")
        self.assertEqual(sensitivity["training_seeds"], [1, 2, 3])
        self.assertNotEqual(sensitivity["contrasts"][0]["auroc"]["estimate"], by_id(result, "H1")["auroc"]["estimate"])
        partial = [r for r in rows if not (r["arm"] == "none" and r["fold"] == 4 and r["seed"] == 3)]
        self.assertEqual(sap.analyze(partial, dataset="synthetic", B=29)["sensitivities"]["three_seed_mean"]["status"], "not_evaluable")
        absent_one = sap.analyze([r for r in rows if r["seed"] > 1], dataset="synthetic", B=29)
        self.assertEqual(by_id(absent_one, "H1")["status"], "not_evaluable")

    def test_predeclared_training_seed_mapping_preserves_order(self):
        rows = synthetic(groups_per_class=3, seeds=(1, 2, 3))
        mapped = [dict(row, seed={1: 19, 2: 11, 3: 13}[row["seed"]]) for row in rows]
        expected = sap.analyze(rows, dataset="synthetic", B=29)
        result = sap.analyze(mapped, dataset="synthetic", B=29, training_seeds=(19, 11, 13))
        self.assertEqual(result["primary"]["training_seeds"], [19])
        self.assertEqual(by_id(result, "H1")["auroc"], by_id(expected, "H1")["auroc"])
        with self.assertRaisesRegex(ValueError, "predeclared"):
            sap.analyze(mapped, dataset="synthetic", B=29)

    def test_shared_draws_all_arms_and_seeds(self):
        rows = synthetic(groups_per_class=3, seeds=(1, 2, 3))
        for row in rows:
            # Identical predictions must yield a point mass at zero delta.
            row["score"] = float(int(row["image"].split("-")[-1]) / 2)
        report = sap.analyze(rows, dataset="synthetic", B=79)
        for analysis in (report["primary"], report["sensitivities"]["three_seed_mean"]):
            for c in analysis["contrasts"]:
                self.assertEqual(c["auroc"]["ci95"], [0., 0.])
                self.assertEqual(c["ap"]["ci95"], [0., 0.])

    def test_isolation_duplicates_incomplete_images_and_nonfinite(self):
        rows = synthetic(groups_per_class=3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            sap.analyze(rows + [rows[0]], dataset="synthetic", B=9)
        inconsistent = copy.deepcopy(rows)
        inconsistent[0]["label"] = 1 - inconsistent[0]["label"]
        with self.assertRaisesRegex(ValueError, "isolation_violation"):
            sap.analyze(inconsistent, dataset="synthetic", B=9)
        crossed = copy.deepcopy(rows)
        for row in crossed:
            if row["group"] == "group-1-0-0":
                row["group"] = "group-0-0-0"
        with self.assertRaisesRegex(ValueError, "group crosses folds"):
            sap.analyze(crossed, dataset="synthetic", B=9)
        partial = rows[1:]
        result = sap.analyze(partial, dataset="synthetic", B=9)
        self.assertEqual(by_id(result, "S3")["status"], "not_evaluable")
        nonfinite = copy.deepcopy(rows)
        next(r for r in nonfinite if r["arm"] == "basic")["score"] = float("nan")
        result = sap.analyze(nonfinite, dataset="synthetic", B=9)
        self.assertEqual(by_id(result, "H1")["status"], "not_evaluable")
        json.dumps(result, allow_nan=False)

    def test_undefined_replicates_not_dropped_or_retried(self):
        # Two positive-status groups; only one has a negative image. Some
        # bootstrap samples omit all negatives, despite status stratification.
        rows = []
        for fold in sap.FOLDS:
            for image, group, label in (("a", "mixed", 0), ("b", "mixed", 1), ("c", "positive", 1)):
                for arm in ARMS:
                    rows.append(dict(dataset="synthetic", fold=fold, arm=arm, seed=1,
                                     image=f"{fold}-{image}", group=f"{fold}-{group}", label=label, score=.4))
        report = sap.analyze(rows, dataset="synthetic", B=99, seed=2)
        value = by_id(report, "H1")["auroc"]
        self.assertEqual(value["status"], "not_evaluable")
        self.assertGreater(value["undefined_bootstrap_replicates"], 0)
        self.assertIsNone(value["ci95"])
        self.assertIsNone(value["p_boot"])

    def test_native_reader_checks_integrity_partition_and_positive_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            value = native_artifact()
            path.write_text(json.dumps(value), encoding="utf-8")
            rows, metadata = sap.read_predictions(path.parent, "synthetic")
            self.assertEqual(rows[0]["label"], 1)
            self.assertEqual(rows[0]["score"], .8)
            self.assertEqual(len(metadata["inputs"]), 1)
            self.assertEqual(metadata["freeze_sha256"], value["freeze_sha256"])
            self.assertEqual(metadata["pools_sha256"], value["pools_sha256"])
            value["rows"][0]["scores"] = [.9, .1]
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content_sha256"):
                sap.read_predictions(path, "synthetic")
            value = native_artifact()
            value.update(phase="hpo", partition="D_select")
            value.pop("content_sha256")
            value["content_sha256"] = canonical_hash(value)
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "isolation_violation"):
                sap.read_predictions(path, "synthetic")

    def test_inbreast_recipe_scope(self):
        rows = [dict(row, dataset="INbreast") for row in synthetic(groups_per_class=3)]
        with self.assertRaisesRegex(ValueError, "recipe_source"):
            sap.analyze(rows, dataset="INbreast", B=19)
        fixed = sap.analyze(rows, dataset="INbreast", recipe_source="external_fixed", B=19)
        self.assertEqual(fixed["analysis_scope"], "primary_external_fixed")
        hpo = sap.analyze(rows, dataset="INbreast", recipe_source="hpo_d", B=19)
        self.assertEqual(hpo["analysis_scope"], "descriptive_hpo_d")
        self.assertIsNone(by_id(hpo, "H1")["auroc"]["ci95"])
        self.assertIsNone(by_id(hpo, "H1")["auroc"]["p_boot"])

    def test_small_positive_group_warning_threshold(self):
        for groups, expected_warning in ((7, True), (8, False)):
            with self.subTest(positive_groups=groups):
                result = sap.analyze(synthetic(groups_per_class=groups), dataset="synthetic", B=9)
                self.assertEqual([fold["positive_groups"] for fold in result["coverage"]], [groups] * 5)
                self.assertEqual(result["warnings"],
                                 ["coverage below nominal measured in simulation with <8 positive groups per fold"]
                                 if expected_warning else [])
        # One undersized fold suffices; do not average the group counts.
        rows = [row for row in synthetic(groups_per_class=8)
                if row["group"] != "group-4-1-7"]
        self.assertTrue(sap.analyze(rows, dataset="synthetic", B=9)["warnings"])

    def test_cli_native_provenance_and_human_recipe_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp) / "predictions.json", Path(tmp) / "out"
            value = native_artifact()
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(sap.main(["--predictions", str(path), "--dataset", "synthetic", "--out", str(out),
                                       "--seed", "123", "--B", "9", "--recipe-source", "external_fixed"]), 0)
            report = json.loads((out / "analysis.json").read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            for key, expected in (("freeze_sha256", value["freeze_sha256"]),
                                  ("pools_sha256", value["pools_sha256"]),
                                  ("recipe_source", "external_fixed"), ("numpy_version", np.__version__)):
                self.assertEqual(report["provenance"][key], expected)
                self.assertEqual(manifest[key], expected)
            self.assertEqual(report["recipe_source"], "external_fixed")

    def test_cli_csv_recipe_source_conflict_fails_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp) / "fixture.csv", Path(tmp) / "out"
            rows = [dict(row, recipe_source="hpo_d") for row in synthetic(groups_per_class=3)]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = sap.main(["--predictions", str(path), "--dataset", "synthetic", "--out", str(out),
                                   "--seed", "123", "--B", "9", "--recipe-source", "external_fixed"])
            self.assertEqual(status, 2)
            self.assertIn("conflicting recipe_source", stderr.getvalue())
            self.assertFalse(out.exists())
            _, metadata = sap.read_predictions(path, "synthetic")
            self.assertEqual(metadata["recipe_source"], "hpo_d")

    def test_cli_csv_outputs_and_hashes_without_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp) / "fixture.csv", Path(tmp) / "out"
            rows = synthetic(groups_per_class=3)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(sap.main(["--predictions", str(path), "--dataset", "synthetic", "--out", str(out),
                                       "--seed", "123", "--B", "39"]), 0)
            manifest = json.loads((out / "manifest.json").read_text())
            report = json.loads((out / "analysis.json").read_text())
            for key in ("freeze_sha256", "pools_sha256", "recipe_source"):
                self.assertIsNone(report["provenance"][key])
                self.assertIsNone(manifest[key])
            self.assertEqual(manifest["numpy_version"], np.__version__)
            self.assertTrue(report["warnings"])
            for name, expected in manifest["outputs"].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), expected)
            self.assertGreater((out / "forest.png").stat().st_size, 1000)
            self.assertTrue((out / "forest.pdf").read_bytes().startswith(b"%PDF"))
            for name in ("analysis.json", "table.csv", "manifest.json"):
                text = (out / name).read_text()
                self.assertNotIn("image-0-", text)
                self.assertNotIn("group-0-", text)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = sap.main(["--predictions", str(path), "--dataset", "other", "--out", str(out),
                                   "--seed", "123", "--B", "39"])
            self.assertEqual(status, 2)
            self.assertIn("dataset mismatch", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
