"""Synthetic INbreast-shaped proposal conversion; no real data or network."""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import global_cli as cli
from augment_atlas import global_development as gd
from augment_atlas import legacy_pools as lp
from augment_atlas.contracts import canonical_hash


class ConvertLegacyPoolsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-legacy-pools-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        (self.data / "inbreast_png_896").mkdir(parents=True)
        self.rows = []
        manifest = self.data / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["processed_image_filename", "group_key", "classification", "sha256"])
            for group in range(12):
                # 40 images / 12 groups; mixed labels in every group.
                for image in range(4 if group < 4 else 3):
                    index = len(self.rows)
                    path = f"inbreast_png_896/{index:08d}_{index:016x}_MG_L_CC_ANON.png"
                    pixels = f"synthetic pixels {index}".encode()
                    (self.data / path).write_bytes(pixels)
                    row = {"id": path, "path": path, "group": f"synthetic-group-{group:02d}", "label": image % 2}
                    self.rows.append(row)
                    writer.writerow([path, row["group"], row["label"], hashlib.sha256(pixels).hexdigest()])
        self.source = {"adapter": "inbreast_csv", "manifest": "manifest.csv",
                       "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        self.source_path = self.root / "inbreast-source.json"
        self.source_path.write_text(json.dumps(self.source), encoding="utf-8")
        ids = lambda groups: list(reversed([r["path"] for r in self.rows if int(r["group"][-2:]) in groups]))
        development, evaluation = ids(range(4)), ids(range(4, 12))
        self.legacy = {
            "protocol": "global-development-proposal-v1", "scientific_status": "provisional-feasibility-only",
            "sklearn": "1.8.0", "seeds": {"development": 17, "selection": 29, "evaluation": 31},
            "source_fold_sha256": canonical_hash("synthetic fixed original folds"),
            "development": development, "search_train": ids(range(3)), "search_selection": ids([3]),
            "evaluation": evaluation,
            "folds": [{"fold": index, "test": ids(range(4 + index, 12, 5)),
                       "train": list(reversed(development + sorted(set(evaluation) - set(ids(range(4 + index, 12, 5)))))),
                       "diagnostics": {"legacy_only": "not trusted"}} for index in range(5)]}
        self.legacy_path = self.root / "legacy.json"
        self.out = self.root / "converted.json"
        self.write_legacy()

    def write_legacy(self):
        self.legacy_path.write_text(json.dumps(self.legacy, indent=2) + "\n", encoding="utf-8")

    def base(self, command):
        return [command, "--data-root", str(self.data), "--source", str(self.source_path), "--dataset", "inbreast"]

    def invoke(self, argv, expected=0, extra_private=()):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(argv)
        self.assertEqual(status, expected, stderr.getvalue())
        for identifier in [v for row in self.rows for v in (row["id"], row["group"])] + list(extra_private):
            self.assertNotIn(identifier, stdout.getvalue() + stderr.getvalue())
        if expected:
            self.assertEqual(stdout.getvalue(), "")
            return json.loads(stderr.getvalue())["error"]
        self.assertEqual(stderr.getvalue(), "")
        return json.loads(stdout.getvalue())

    def convert(self, expected=0, extra_private=()):
        return self.invoke(self.base("convert-legacy-pools") + ["--legacy", str(self.legacy_path), "--out", str(self.out)],
                           expected, extra_private)

    def reject(self, mutate):
        mutate(self.legacy)
        self.write_legacy()
        self.convert(expected=2)
        self.assertFalse(self.out.exists())
        self.assertFalse((self.data / "global").exists())

    def test_conversion_import_and_audit_preserve_all_memberships(self):
        before = self.legacy_path.read_bytes()
        with patch("augment_atlas.global_splits.generate_dev_eval_pools", side_effect=AssertionError("no new draw")):
            report = self.convert()
        candidate = gd._read(self.out)
        self.assertEqual(before, self.legacy_path.read_bytes())
        self.assertNotIn("sha256", candidate)  # import-pools seals this unsigned candidate
        self.assertEqual(candidate["dataset"]["id"], "inbreast")
        self.assertEqual(candidate["grouping"]["prior_exposure"], "known")
        self.assertEqual(candidate["grouping"]["identity_level"], "group")
        provenance = candidate["partition_provenance"]
        self.assertEqual(provenance["algorithm"], lp.ALGORITHM)
        self.assertEqual(provenance["seeds"], self.legacy["seeds"])
        self.assertEqual(provenance["converted_from"], {"protocol": self.legacy["protocol"],
                                                       "sha256": hashlib.sha256(before).hexdigest()})
        self.assertEqual(set(provenance["intended_fractions"].values()), {lp.NOT_PREREGISTERED})
        for new, old in (("D", "development"), ("D_fit", "search_train"), ("D_select", "search_selection"), ("E", "evaluation")):
            self.assertEqual(candidate["lists"][new], sorted(self.legacy[old]))
        self.assertEqual(candidate["source_sha256"], self.source["manifest_sha256"])
        self.assertEqual(report["diagnostics"]["manifest"]["images"], 40)
        self.assertEqual(report["diagnostics"]["manifest"]["groups"], 12)
        imported = self.invoke(self.base("import-pools") + ["--file", str(self.out), "--expected-manifest", report["global_manifest_sha256"]])
        self.assertEqual(imported["pools_sha256"], report["candidate_sha256"])
        audited = self.invoke(self.base("audit-pools") + ["--pools-sha256", imported["pools_sha256"],
                                                        "--expected-manifest", report["global_manifest_sha256"]])
        self.assertTrue(audited["structural_valid"])
        self.assertFalse(audited["eligibility"]["contract_eligible"])
        self.assertEqual(set(audited["eligibility"]["reasons"]), {"prior_exposure_known", "patient_identity_unproven"})
        sealed = gd._read(Path(imported["path"]))
        for index, old_fold in enumerate(self.legacy["folds"]):
            self.assertEqual(candidate["evaluation_folds"][index]["test_ids"], sorted(old_fold["test"]))
            view = gd.final_fold_view(self.rows, sealed, index, expected_pools_sha256=sealed["sha256"])
            self.assertEqual(view["final_train_ids"], sorted(old_fold["train"]))
        self.assertEqual(report, self.convert())  # deterministic, idempotent no-clobber

    def test_split_development_group_fails_even_with_exact_image_coverage(self):
        def mutate(doc):
            left, right = doc["search_train"][0], doc["search_selection"][0]
            doc["search_train"].remove(left); doc["search_train"].append(right)
            doc["search_selection"].remove(right); doc["search_selection"].append(left)
        self.reject(mutate)

    def test_split_d_e_group_fails(self):
        def mutate(doc):
            left, right = doc["development"][0], doc["evaluation"][0]
            # Swap images everywhere, preserving all image-level identities and fold.train.
            def swap(values):
                return [right if v == left else left if v == right else v for v in values]
            for key in ("development", "search_train", "search_selection", "evaluation"):
                doc[key] = swap(doc[key])
            for fold in doc["folds"]:
                fold["train"], fold["test"] = swap(fold["train"]), swap(fold["test"])
        self.reject(mutate)

    def test_split_e_fold_group_fails(self):
        def mutate(doc):
            left, right = doc["folds"][0]["test"][0], doc["folds"][1]["test"][0]
            doc["folds"][0]["test"][0], doc["folds"][1]["test"][0] = right, left
            for fold in doc["folds"]:
                fold["train"] = doc["development"] + sorted(set(doc["evaluation"]) - set(fold["test"]))
        self.reject(mutate)

    def test_duplicate_image_in_each_list_fails(self):
        for key in ("development", "search_train", "search_selection", "evaluation"):
            with self.subTest(key=key):
                doc = copy.deepcopy(self.legacy)
                doc[key].append(doc[key][0])
                self.legacy_path.write_text(json.dumps(doc), encoding="utf-8")
                self.convert(expected=2)
                self.assertFalse(self.out.exists())
        for key in ("train", "test"):
            with self.subTest(fold_key=key):
                doc = copy.deepcopy(self.legacy)
                doc["folds"][0][key].append(doc["folds"][0][key][0])
                self.legacy_path.write_text(json.dumps(doc), encoding="utf-8")
                self.convert(expected=2)
                self.assertFalse(self.out.exists())

    def test_missing_manifest_image_fails(self):
        def mutate(doc):
            missing = doc["evaluation"].pop()
            for fold in doc["folds"]:
                for key in ("train", "test"):
                    fold[key] = [v for v in fold[key] if v != missing]
        self.reject(mutate)

    def test_d_subpools_must_partition_d(self):
        self.reject(lambda doc: doc["search_train"].pop())

    def test_fold_train_inconsistency_fails(self):
        self.reject(lambda doc: doc["folds"][0]["train"].pop())

    def test_e_test_coverage_and_overlap_fail(self):
        original = copy.deepcopy(self.legacy)
        for overlap in (False, True):
            self.legacy = copy.deepcopy(original)
            doc = self.legacy
            if overlap:
                doc["folds"][0]["test"] = doc["folds"][1]["test"][:]
            else:
                doc["folds"][0]["test"].pop()
            for fold in doc["folds"]:
                fold["train"] = doc["development"] + sorted(set(doc["evaluation"]) - set(fold["test"]))
            self.write_legacy()
            self.convert(expected=2)
            self.assertFalse(self.out.exists())

    def test_unknown_identifier_is_withheld(self):
        secret = "inbreast_png_896/87654321_abcdef0123456789_MG_R_ML_ANON.png"
        self.legacy["development"][0] = secret
        self.write_legacy()
        error = self.convert(expected=2, extra_private=[secret])
        self.assertIn("message withheld", error)
        self.assertFalse(self.out.exists())

    def test_manifest_error_before_records_load_withholds_identifier(self):
        private = self.rows[0]["group"]
        with patch.object(cli, "_records", side_effect=ValueError("bad image: " + private)):
            self.assertIn("message withheld", self.convert(expected=2))

    def test_invalid_legacy_schema_metadata_and_fold_indices_fail(self):
        original = copy.deepcopy(self.legacy)
        mutations = [lambda d: d.update(protocol="wrong"), lambda d: d.update(source_fold_sha256="bad"),
                     lambda d: d["seeds"].update(evaluation=17), lambda d: d["seeds"].update(development=True),
                     lambda d: d["folds"][0].update(fold=True), lambda d: d["folds"][1].update(fold=0),
                     lambda d: d["folds"].pop(), lambda d: d["folds"][0].update(diagnostics=[]),
                     lambda d: d.update(evaluation=[]), lambda d: d.update(extra="unexpected")]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.legacy = copy.deepcopy(original)
                self.reject(mutate)

    def test_malformed_duplicate_key_and_nonfinite_json_fail_privately(self):
        for content in ('{"private-id":', '{"private-id":1,"private-id":2}',
                        self.legacy_path.read_text(encoding="utf-8").replace('"1.8.0"', 'NaN')):
            self.legacy_path.write_text(content, encoding="utf-8")
            self.convert(expected=2, extra_private=["private-id"])
            self.assertFalse(self.out.exists())

    def test_conversion_does_not_clobber_another_candidate(self):
        self.convert()
        before = self.out.read_bytes()
        self.legacy["folds"][0]["diagnostics"]["legacy_only"] = "changed bytes, unchanged memberships"
        self.write_legacy()
        self.convert(expected=2)
        self.assertEqual(before, self.out.read_bytes())

    def test_conversion_requires_inbreast_and_its_adapter(self):
        args = self.base("convert-legacy-pools") + ["--legacy", str(self.legacy_path), "--out", str(self.out)]
        args[args.index("--dataset") + 1] = "another_dataset"
        self.invoke(args, expected=2)
        self.source["adapter"] = "manifest_csv"
        self.source_path.write_text(json.dumps(self.source), encoding="utf-8")
        self.convert(expected=2)
        self.assertFalse(self.out.exists())

    def test_legacy_sentinel_and_provenance_are_scoped_and_anchored(self):
        self.convert()
        candidate = gd._read(self.out)
        anchor = gd.global_manifest_sha256(self.rows, candidate)
        for mutate in (lambda c: c["partition_provenance"].update(algorithm="global-dev-eval-v1"),
                       lambda c: c["partition_provenance"].pop("converted_from"),
                       lambda c: c["partition_provenance"]["converted_from"].update(sha256="bad"),
                       lambda c: c["partition_provenance"]["converted_from"].update(protocol="wrong"),
                       lambda c: c["partition_provenance"]["converted_from"].update(extra=1),
                       lambda c: c["grouping"].update(prior_exposure="none_documented"),
                       lambda c: c["dataset"].update(id="another_dataset")):
            doc = copy.deepcopy(candidate)
            mutate(doc)
            with self.assertRaises(ValueError):
                gd.audit_global_pools(self.rows, gd._signed(doc), expected_manifest_sha256=gd.global_manifest_sha256(self.rows, doc))
        changed = copy.deepcopy(candidate)
        changed["partition_provenance"]["converted_from"]["sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "externally anchored manifest"):
            gd.audit_global_pools(self.rows, gd._signed(changed), expected_manifest_sha256=anchor)
        doc = copy.deepcopy(candidate)
        doc["partition_provenance"].update(algorithm="manual-fixture")
        doc["partition_provenance"].pop("converted_from")
        with self.assertRaisesRegex(ValueError, "finite number"):
            gd.audit_global_pools(self.rows, gd._signed(doc), expected_manifest_sha256=gd.global_manifest_sha256(self.rows, doc))


if __name__ == "__main__":
    unittest.main()
