"""Manual, synthetic declarations only; no split generator or images involved."""
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas.contracts import canonical_hash
from augment_atlas import global_development as gd


def H(value):
    return canonical_hash(value)


def signed(value):
    result = copy.deepcopy(value)
    result.pop("sha256", None)
    result["sha256"] = H(result)
    return result


def pool_fixture():
    # Every group is mixed: retaining each image's label is essential.
    rows = [{"id": f"g{group}-{label}", "path": f"unavailable/g{group}-{label}.png",
             "group": f"g{group}", "label": label} for group in range(8) for label in range(2)]
    ids = lambda groups: sorted(row["id"] for row in rows if int(row["group"][1:]) in groups)
    document = {"schema_version": 1, "protocol": gd.PROTOCOL, "scientific_status": "provisional",
                "dataset": {"id": "synthetic_fixture", "version": "v1"}, "records_sha256": H(rows),
                "source_sha256": H("synthetic manifest bytes"),
                "provenance": {key: H(key) for key in ("preprocessing_sha256", "labels_sha256", "duplicates_sha256", "exclusions_sha256")},
                "grouping": {"namespace": "fixture-group", "source": "manual fixture", "identity_level": "group",
                             "linkage_status": "unverified", "evidence_sha256": None, "prior_exposure": "none_documented"},
                "task": {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
                         "decision_rule": "scores_only", "missing_policy": "undefined_no_drop", "requires_patient_identity": False},
                "partition_provenance": {"algorithm": "manual-fixture", "version": "1", "source": "fixed manual lists",
                                         "generator_version": "no generator", "seeds": {"development": 17, "selection": 29, "evaluation": 31},
                                         "intended_fractions": {"D": 0.2, "D_fit_given_D": 0.8, "E_test_given_E": 0.2}},
                "lists": {"D": ids(range(3)), "D_fit": ids(range(2)), "D_select": ids([2]), "E": ids(range(3, 8))},
                "evaluation_folds": [{"fold": index, "test_ids": ids([index + 3])} for index in range(5)]}
    return rows, signed(document), gd.global_manifest_sha256(rows, document)


class GlobalPoolsTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.pools, self.anchor = pool_fixture()

    def audit(self, document=None, rows=None, anchor=None):
        return gd.audit_global_pools(self.rows if rows is None else rows, self.pools if document is None else document,
                                     expected_manifest_sha256=self.anchor if anchor is None else anchor)

    def test_manual_mixed_groups_and_final_train_identity(self):
        original = copy.deepcopy((self.rows, self.pools))
        audit = self.audit()
        self.assertTrue(audit["structural_valid"])
        self.assertFalse(audit["execution_verified"])
        self.assertEqual(audit["diagnostics"]["D_select"]["mixed_label_groups"], 1)
        self.assertEqual(audit["diagnostics"]["D_select"]["class_group_counts"], {"0": 1, "1": 1})
        tests = []
        for fold in range(5):
            view = gd.final_fold_view(self.rows, self.pools, fold, expected_pools_sha256=self.pools["sha256"])
            train, test = set(view["final_train_ids"]), set(view["test_ids"])
            self.assertTrue(set(self.pools["lists"]["D"]) <= train)
            self.assertFalse(set(self.pools["lists"]["D"]) & test)
            self.assertFalse(train & test)
            self.assertEqual(train | test, {row["id"] for row in self.rows})
            self.assertNotIn("selection", view)
            tests.extend(test)
        self.assertEqual(sorted(tests), self.pools["lists"]["E"])
        self.assertEqual((self.rows, self.pools), original)
        self.assertEqual(self.audit(rows=list(reversed(self.rows))), audit)

    def test_rehashed_leakage_and_coverage_mutants(self):
        def d_in_test(doc):
            doc["evaluation_folds"][0]["test_ids"] = sorted(["g0-0", "g3-1"])

        def split_selection_siblings(doc):
            doc["lists"]["D_fit"].remove("g0-0")
            doc["lists"]["D_fit"].append("g2-0")
            doc["lists"]["D_select"].remove("g2-0")
            doc["lists"]["D_select"].append("g0-0")
            for key in ("D_fit", "D_select"):
                doc["lists"][key].sort()

        def split_test_siblings(doc):
            doc["evaluation_folds"][0]["test_ids"] = ["g3-0", "g4-1"]
            doc["evaluation_folds"][1]["test_ids"] = ["g3-1", "g4-0"]

        def d_e_group(doc):
            # Exact ID coverage retained, but g2 and g3 straddle D/E.
            for key in ("D", "D_select"):
                doc["lists"][key].remove("g2-1")
                doc["lists"][key].append("g3-1")
                doc["lists"][key].sort()
            doc["lists"]["E"].remove("g3-1")
            doc["lists"]["E"].append("g2-1")
            doc["lists"]["E"].sort()

        mutations = [d_in_test, split_selection_siblings, split_test_siblings, d_e_group,
                     lambda d: d["lists"]["D"].append(d["lists"]["D"][0]),
                     lambda d: d["lists"]["E"].pop(),
                     lambda d: d["lists"]["E"].append("unknown"),
                     lambda d: d["evaluation_folds"].pop(),
                     lambda d: d["evaluation_folds"][1].update(fold=0),
                     lambda d: d["evaluation_folds"][1].update(test_ids=["g3-0", "g3-1"]),
                     lambda d: d["evaluation_folds"][0].update(test_ids=["g3-0"])]
        for mutate in mutations:
            candidate = copy.deepcopy(self.pools)
            mutate(candidate)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.audit(signed(candidate))

    def test_records_and_metadata_are_externally_anchored_after_rehash(self):
        cases = []
        rows = copy.deepcopy(self.rows)
        rows[0]["label"], rows[1]["label"] = 1, 0  # same counts
        cases.append((rows, copy.deepcopy(self.pools)))
        rows = copy.deepcopy(self.rows)
        rows[0]["group"], rows[2]["group"] = rows[2]["group"], rows[0]["group"]
        cases.append((rows, copy.deepcopy(self.pools)))
        for section, key, value in (("dataset", "version", "v2"), ("provenance", "preprocessing_sha256", H("new")),
                                     ("task", "classes", [0, 1, 2]), ("partition_provenance", "version", "2")):
            doc = copy.deepcopy(self.pools)
            doc[section][key] = value
            cases.append((self.rows, doc))
        for rows, doc in cases:
            doc["records_sha256"] = H(sorted(rows, key=lambda row: row["id"]))
            with self.subTest(doc=doc["dataset"], rows=rows[0]), self.assertRaisesRegex(ValueError, "identity|task"):
                self.audit(signed(doc), rows=rows)

    def test_final_view_rejects_rehashed_pools_against_external_anchor(self):
        doc = copy.deepcopy(self.pools)
        doc["lists"]["D_fit"], doc["lists"]["D_select"] = doc["lists"]["D_select"], doc["lists"]["D_fit"]
        doc = signed(doc)
        self.audit(doc)  # independently valid lists; different approved identity
        with self.assertRaisesRegex(ValueError, "anchored pools"):
            gd.final_fold_view(self.rows, doc, 0, expected_pools_sha256=self.pools["sha256"])
        for fold in (-1, 5, True, 0.0):
            with self.subTest(fold=fold), self.assertRaises(ValueError):
                gd.final_fold_view(self.rows, self.pools, fold, expected_pools_sha256=self.pools["sha256"])

    def test_absent_class_keeps_denominator_and_explicit_undefined_state(self):
        rows = copy.deepcopy(self.rows)
        rows[5]["label"] = 0  # D_select now has only class zero
        pools = copy.deepcopy(self.pools)
        pools["records_sha256"] = H(rows)
        pools = signed(pools)
        result = self.audit(pools, rows=rows, anchor=gd.global_manifest_sha256(rows, pools))
        counts = result["diagnostics"]["D_select"]
        self.assertTrue(result["structural_valid"])
        self.assertEqual(counts["class_counts"], {"0": 2, "1": 0})
        self.assertEqual(counts["missing_classes"], [1])
        self.assertFalse(counts["endpoint_feasible"])
        self.assertIn("missing_classes:D_select", result["eligibility"]["reasons"])

    def test_patient_unknown_and_known_inbreast_exposure_never_promoted(self):
        doc = copy.deepcopy(self.pools)
        doc["task"]["requires_patient_identity"] = True
        doc["dataset"]["id"] = "INbreast"
        doc["grouping"]["prior_exposure"] = "known"
        doc = signed(doc)
        result = self.audit(doc, anchor=gd.global_manifest_sha256(self.rows, doc))
        self.assertFalse(result["eligibility"]["contract_eligible"])
        self.assertIn("patient_identity_unproven", result["eligibility"]["reasons"])
        self.assertIn("prior_exposure_known", result["eligibility"]["reasons"])
        doc["grouping"]["identity_level"] = "patient"
        with self.assertRaisesRegex(ValueError, "patient identity"):
            self.audit(signed(doc), anchor=gd.global_manifest_sha256(self.rows, doc))

    def test_closed_types_paths_and_record_identity(self):
        for value in (True, 1.0, "1"):
            doc = copy.deepcopy(self.pools)
            doc["schema_version"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.audit(signed(doc), anchor=gd.global_manifest_sha256(self.rows, doc))
        for path in ("/absolute.png", "../escape.png", "C:/private.png", "a/./b.png", "a//b.png", "a\\b.png", ""):
            rows = copy.deepcopy(self.rows)
            rows[0]["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.audit(rows=rows)
        for field in ("id", "path"):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = rows[1][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(rows=rows)
        doc = copy.deepcopy(self.pools)
        doc["diagnostics"] = {"valid": True}
        with self.assertRaisesRegex(ValueError, "closed schema"):
            self.audit(signed(doc))
        doc = copy.deepcopy(self.pools)
        doc["task"]["type"] = "multilabel"
        with self.assertRaisesRegex(ValueError, "multilabel"):
            self.audit(signed(doc))

    def test_audit_and_derive_without_ml_imports_images_or_processes(self):
        with patch.dict("sys.modules", {name: None for name in ("torch", "torchvision", "numpy", "sklearn")}), \
             patch.object(Path, "open", side_effect=AssertionError("file access forbidden")), \
             patch("subprocess.Popen", side_effect=AssertionError("process forbidden")):
            importlib.reload(gd)
            self.audit()
            gd.final_fold_view(self.rows, self.pools, 0, expected_pools_sha256=self.pools["sha256"])
            with self.assertRaises(ValueError):
                self.audit({})

    def test_exclusive_idempotent_publish_and_conflict_preserve_original_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "private" / "global-pools.json"
            def seal():
                return gd.seal_global_pools(self.rows, self.pools, expected_manifest_sha256=self.anchor, output=path)
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _: seal(), range(8)))
            original = path.read_bytes()
            self.assertTrue(all(result == self.pools for result in results))
            seal()
            self.assertEqual(path.read_bytes(), original)
            doc = copy.deepcopy(self.pools)
            doc["lists"]["D_fit"], doc["lists"]["D_select"] = doc["lists"]["D_select"], doc["lists"]["D_fit"]
            with self.assertRaisesRegex(ValueError, "no-clobber"):
                gd.seal_global_pools(self.rows, doc, expected_manifest_sha256=self.anchor, output=path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])
            path.write_text('{"sha256":"corrupt"}')
            damaged = path.read_bytes()
            with self.assertRaises(ValueError):
                seal()
            self.assertEqual(path.read_bytes(), damaged)


if __name__ == "__main__":
    unittest.main()
