"""D/E pools generator global-dev-eval-v1 (synthetic records; no images, no training)."""
import copy
import hashlib
import unittest

from augment_atlas import global_development as gd
from augment_atlas import global_splits as gs
from augment_atlas.contracts import canonical_hash as H


def synthetic(groups=600, positive_every=5):
    """Groups of 1..6 images; every ``positive_every``-th group holds one positive image."""
    rows = []
    for g in range(groups):
        size = 1 + int(hashlib.sha256(f"size{g}".encode()).hexdigest(), 16) % 6
        for i in range(size):
            label = 1 if (g % positive_every == 0 and i == 0) else 0
            rows.append({"id": f"img/p{g:04d}-{i}.jpg", "path": f"img/p{g:04d}-{i}.jpg", "group": f"pt{g:04d}",
                         "label": label})
    return rows


def spec(**changes):
    value = {
        "dataset": {"id": "synthetic_split", "version": "v1"},
        "task": {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
                 "decision_rule": "fixed_threshold", "missing_policy": "undefined_no_drop",
                 "requires_patient_identity": False},
        "grouping": {"namespace": "patient+duplicates", "source": "fixture union-find", "identity_level": "group",
                     "linkage_status": "unverified", "evidence_sha256": None, "prior_exposure": "none_documented"},
        "provenance": {k: H(k) for k in ("preprocessing_sha256", "labels_sha256", "duplicates_sha256",
                                          "exclusions_sha256")},
        "source_sha256": H("source"),
        "seeds": {"development": 20261019, "selection": 20261019, "evaluation": 20261020},
        "fractions": {"D": 0.2, "D_select_given_D": 0.3},
        "thresholds": {"min_select_positive_groups": 5, "min_eval_fold_positive_groups": 15}}
    value.update(changes)
    return value


def groups_of(ids, rows):
    lookup = {r["id"]: r["group"] for r in rows}
    return {lookup[i] for i in ids}


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic()
        self.candidate, self.report = gs.generate_dev_eval_pools(self.rows, spec())

    def test_candidate_passes_the_v1_auditor_and_is_deterministic(self):
        signed = gd._signed(self.candidate)
        audit = gd.audit_global_pools(self.rows, signed,
                                      expected_manifest_sha256=gd.global_manifest_sha256(self.rows, self.candidate))
        self.assertIs(audit["structural_valid"], True)
        self.assertEqual(self.report["candidate_sha256"], signed["sha256"])
        again, _ = gs.generate_dev_eval_pools(list(reversed(self.rows)), spec())
        self.assertEqual(again, self.candidate)
        self.assertEqual(self.candidate["partition_provenance"]["algorithm"], "global-dev-eval-v1")

    def test_groups_are_never_split_and_e_is_covered_once(self):
        lists = self.candidate["lists"]
        pools = [lists["D_fit"], lists["D_select"]] + [f["test_ids"] for f in self.candidate["evaluation_folds"]]
        seen = set()
        for ids in pools:
            groups = groups_of(ids, self.rows)
            self.assertFalse(groups & seen)
            seen |= groups
        self.assertEqual(seen, {r["group"] for r in self.rows})
        tested = sorted(i for f in self.candidate["evaluation_folds"] for i in f["test_ids"])
        self.assertEqual(tested, lists["E"])

    def test_fractions_follow_the_strata_and_folds_are_balanced(self):
        groups = len({r["group"] for r in self.rows})
        d_groups = len(groups_of(self.candidate["lists"]["D"], self.rows))
        self.assertAlmostEqual(d_groups / groups, 0.2, delta=0.02)
        select = len(groups_of(self.candidate["lists"]["D_select"], self.rows))
        self.assertAlmostEqual(select / d_groups, 0.3, delta=0.05)
        sizes = [len(groups_of(f["test_ids"], self.rows)) for f in self.candidate["evaluation_folds"]]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        positives = [self.report["pools"][f"E_test_{f}"]["positive_groups"] for f in range(5)]
        self.assertLessEqual(max(positives) - min(positives), 1)  # positive stratum spread round-robin

    def test_each_seed_moves_only_its_own_stage(self):
        other_eval, _ = gs.generate_dev_eval_pools(self.rows, spec(seeds={"development": 20261019,
                                                                          "selection": 20261019,
                                                                          "evaluation": 7}))
        self.assertEqual(other_eval["lists"]["D"], self.candidate["lists"]["D"])
        self.assertEqual(other_eval["lists"]["D_select"], self.candidate["lists"]["D_select"])
        self.assertNotEqual(other_eval["evaluation_folds"], self.candidate["evaluation_folds"])
        other_dev, _ = gs.generate_dev_eval_pools(self.rows, spec(seeds={"development": 1, "selection": 20261019,
                                                                         "evaluation": 20261020}))
        self.assertNotEqual(other_dev["lists"]["D"], self.candidate["lists"]["D"])

    def test_threshold_failure_stops_without_fallback(self):
        strict = spec(thresholds={"min_select_positive_groups": 500, "min_eval_fold_positive_groups": 15})
        with self.assertRaisesRegex(ValueError, "^split_audit_failed: D_select has .*no retry"):
            gs.generate_dev_eval_pools(self.rows, strict)
        strict = spec(thresholds={"min_select_positive_groups": 5, "min_eval_fold_positive_groups": 500})
        with self.assertRaisesRegex(ValueError, "^split_audit_failed: E_test_0 has"):
            gs.generate_dev_eval_pools(self.rows, strict)

    def test_pool_without_a_class_fails(self):
        rows = synthetic(groups=40, positive_every=40)  # a single positive group
        with self.assertRaisesRegex(ValueError, "^split_audit_failed: .*lacks a class"):
            gs.generate_dev_eval_pools(rows, spec(thresholds={"min_select_positive_groups": 1,
                                                              "min_eval_fold_positive_groups": 1}))

    def test_evaluation_seed_reuse_and_unspecified_tasks_are_refused(self):
        with self.assertRaisesRegex(ValueError, "^seed_reuse_forbidden:"):
            gs.generate_dev_eval_pools(self.rows, spec(seeds={"development": 3, "selection": 4, "evaluation": 3}))
        ordinal = copy.deepcopy(spec())
        ordinal["task"].update(type="ordinal", classes=[0, 1, 2, 3, 4], positive_class=None,
                               endpoint="quadratic_weighted_kappa", decision_rule="argmax")
        with self.assertRaisesRegex(ValueError, "^split_spec_invalid: .*binary"):
            gs.generate_dev_eval_pools(self.rows, ordinal)
        with self.assertRaisesRegex(ValueError, "^split_spec_invalid:"):
            gs.generate_dev_eval_pools(self.rows, spec(fractions={"D": 0.2, "D_select_given_D": 1.0}))

    def test_quartile_buckets_and_half_up_rounding(self):
        bucket = gs._quartile_bucket([1, 1, 1, 1, 2, 3, 4, 8])  # cuts: 1, 1, 3
        self.assertEqual([bucket(s) for s in (1, 2, 3, 4, 8)], [0, 2, 2, 3, 3])
        self.assertEqual([gs._half_up_count(f, n) for f, n in ((0.5, 1), (0.3, 5), (0.3, 15), (0.2, 2))],
                         [1, 2, 5, 0])
        self.assertEqual(gs._half_up_count(0.35, 90), 32)  # 31.5 exactly; float arithmetic gives 31.4999...

    def test_spec_must_reproduce_the_preregistered_config(self):
        pinned = {"algorithm": "global-dev-eval-v1", "split_seed": 20261019, "dev_fraction": 0.2,
                  "select_fraction": 0.3, "eval_folds": 5, "min_select_positive_groups": 5,
                  "min_eval_fold_positive_groups": 15}
        gs.check_preregistered(spec(), pinned)
        for change in ({"fractions": {"D": 0.25, "D_select_given_D": 0.3}},
                       {"thresholds": {"min_select_positive_groups": 1, "min_eval_fold_positive_groups": 15}}):
            with self.assertRaisesRegex(ValueError, "^split_spec_invalid: .*pre-registered"):
                gs.check_preregistered(spec(**change), pinned)
        with self.assertRaisesRegex(ValueError, "^split_spec_invalid: acquisition config"):
            gs.check_preregistered(spec(), {**pinned, "eval_folds": 4})
        with self.assertRaisesRegex(ValueError, "^split_spec_invalid: dataset and task"):
            gs.generate_dev_eval_pools(self.rows, spec(dataset="isic"))


if __name__ == "__main__":
    unittest.main()
