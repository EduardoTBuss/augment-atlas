import copy
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas.contracts import canonical_hash, load_config
from augment_atlas.data import prepare_splits
from augment_atlas.scientific_splits import (
    ALGORITHM, SELECTION_PROTOCOL, audit_scientific_folds,
    audit_scientific_split, main, prepare_scientific_splits, split_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
SKLEARN = importlib.util.find_spec("sklearn") is not None


def fixture(groups=30):
    # Every group has both image labels. Never reduce a group to one label.
    return [{"id": f"g{group:03d}-{label}", "path": f"g{group:03d}/{label}.png",
             "group": f"group-{group:03d}", "label": label}
            for group in range(groups) for label in (0, 1)]


def reseal(document):
    document["sha256"] = canonical_hash({key: value for key, value in document.items() if key != "sha256"})


def manual_fold():
    """A stdlib-only valid artifact, independent of the generator and sklearn."""
    from augment_atlas.scientific_splits import _seed
    records = fixture(10)
    lists = {name: [row["id"] for row in records
                    if (int(row["group"].split("-")[1]) // 2) == index]
             for name, index in (("selection", 0), ("test", 1))}
    lists["train"] = [row["id"] for row in records if row["id"] not in lists["selection"] + lists["test"]]
    lookup = {row["id"]: row for row in records}
    document = {
        "manifest_sha256": canonical_hash(sorted(records, key=lambda row: row["id"])),
        "dataset_version": "fixture-v1", "split_seed": 17, "fold": 0,
        "algorithm": ALGORITHM, "scientific_status": "provisional", "outer_folds": 5,
        "outer_seed": _seed(17, "outer"), "selection_seed": _seed(17, "selection", 0),
        "selection_protocol": SELECTION_PROTOCOL, "selection_folds": 5, "selection_fold": 0,
        "sklearn_version": "persisted-fixture", "lists": lists,
        "groups": {name: sorted({lookup[item]["group"] for item in ids}) for name, ids in lists.items()},
        "diagnostics": {"manifest": split_diagnostics(records),
                        "partitions": {name: split_diagnostics([lookup[item] for item in ids]) for name, ids in lists.items()}},
    }
    reseal(document)
    return records, document


class ScientificAuditTests(unittest.TestCase):
    def test_import_and_audit_have_no_ml_dependency(self):
        script = """
import builtins, runpy, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'sklearn', 'numpy'}:
        raise AssertionError('unexpected ML import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
namespace = runpy.run_path(sys.argv[1])
records, document = namespace['manual_fold']()
namespace['audit_scientific_split'](records, document)
"""
        result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve())],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixed_group_labels_counted_per_image(self):
        records, document = manual_fold()
        audit_scientific_split(records, document)
        self.assertEqual(document["diagnostics"]["manifest"], {
            "images": 20, "groups": 10, "class_counts": {"0": 10, "1": 10},
            "class_group_counts": {"0": 10, "1": 10}, "mixed_label_groups": 10})
        audit_scientific_split(list(reversed(records)), document)

    def test_identity_and_alias_duplicates_rejected(self):
        for change in ({"id": "g000-0"}, {"path": "./g000/0.png"}):
            records = fixture()
            records[-1].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "duplicate"):
                split_diagnostics(records)

    def test_hash_and_manifest_identity_rejected(self):
        records, document = manual_fold()
        changed = copy.deepcopy(document)
        changed["lists"]["train"].reverse()
        with self.assertRaisesRegex(ValueError, "checksum"):
            audit_scientific_split(records, changed)
        records[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "identity"):
            audit_scientific_split(records, document)

    def test_overlap_and_group_leakage_rejected_after_rehash(self):
        records, document = manual_fold()
        document["lists"]["train"].append(document["lists"]["test"][0])
        reseal(document)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            audit_scientific_split(records, document)
        records, document = manual_fold()
        left, right = document["lists"]["train"][0], document["lists"]["test"][0]
        document["lists"]["train"].remove(left)
        document["lists"]["test"].remove(right)
        document["lists"]["train"].append(right)
        document["lists"]["test"].append(left)
        reseal(document)
        with self.assertRaisesRegex(ValueError, "group leakage"):
            audit_scientific_split(records, document)

    def test_missing_class_and_protocol_change_fail_without_retry(self):
        records, document = manual_fold()
        selected = set(document["lists"]["selection"])
        for row in records:
            if row["id"] in selected:
                row["label"] = 0
        document["manifest_sha256"] = canonical_hash(sorted(records, key=lambda row: row["id"]))
        reseal(document)
        with self.assertRaisesRegex(ValueError, "all manifest classes"):
            audit_scientific_split(records, document)
        records, document = manual_fold()
        document["selection_fold"] = 1
        reseal(document)
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            audit_scientific_split(records, document)

    def test_scientific_config_requires_persisted_lists(self):
        cfg = load_config(ROOT / "configs/smoke/synthetic.json")
        cfg["splits"].update(version=ALGORITHM, fold=0, split_seed=17)
        records, document = manual_fold()
        cfg["dataset"]["version"] = document["dataset_version"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "hash fallback is forbidden"):
                prepare_splits(cfg, records, root / "absent")
            self.assertFalse((root / "absent/splits.json").exists())
            cfg["dataset"].update(root=str(root), split_file="scientific.json")
            (root / "scientific.json").write_text(json.dumps(document), encoding="utf-8")
            with patch.dict("os.environ", {"ATLAS_DATA_ROOT": str(root)}):
                self.assertEqual(prepare_splits(cfg, records, root / "run"), document)
            (root / "scientific.json").unlink()
            # Resume reads the persisted lists even after the source is unavailable.
            self.assertEqual(prepare_splits(cfg, records, root / "run"), document)
            changed = copy.deepcopy(cfg)
            changed["splits"]["fold"] = 1
            with self.assertRaisesRegex(ValueError, "seed/fold"):
                prepare_splits(changed, records, root / "run")

    def test_feasibility_rejects_sparse_groups_and_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "infeasible"):
            prepare_scientific_splits(fixture(4), dataset_version="v1", split_seed=17)
        for value in (True, -1, "17"):
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, "integer"):
                prepare_scientific_splits(fixture(), dataset_version="v1", split_seed=value)
        with self.assertRaisesRegex(ValueError, "exactly five"):
            prepare_scientific_splits(fixture(), dataset_version="v1", split_seed=17, outer_folds=4)

    def test_generation_reports_missing_optional_dependency(self):
        original_import = builtins.__import__

        def without_sklearn(name, *args, **kwargs):
            if name.split(".")[0] == "sklearn":
                raise ImportError("fixture: dependency absent")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_sklearn):
            with self.assertRaisesRegex(RuntimeError, "requires scikit-learn"):
                prepare_scientific_splits(fixture(), dataset_version="v1", split_seed=17)


@unittest.skipUnless(SKLEARN, "generation needs optional scikit-learn; stdlib audit tests still run")
class ScientificGenerationTests(unittest.TestCase):
    def test_fixed_five_folds_coverage_disjointness_and_determinism(self):
        records = fixture()
        first = prepare_scientific_splits(records, dataset_version="v1", split_seed=17)
        second = prepare_scientific_splits(list(reversed(records)), dataset_version="v1", split_seed=17)
        self.assertEqual(first, second)
        audit_scientific_folds(records, first)
        self.assertEqual(len({document["selection_seed"] for document in first}), 5)
        appearances = [item for document in first for item in document["lists"]["test"]]
        self.assertEqual(sorted(appearances), sorted(row["id"] for row in records))
        for document in first:
            self.assertEqual(document["selection_fold"], 0)
            self.assertEqual(document["scientific_status"], "provisional")
            self.assertEqual(document["manifest_sha256"], canonical_hash(sorted(records, key=lambda row: row["id"])))
            for name in ("train", "selection", "test"):
                self.assertEqual(set(document["diagnostics"]["partitions"][name]["class_counts"]), {"0", "1"})

    def test_outer_test_repetition_and_cross_fold_identity_rejected(self):
        records = fixture()
        documents = prepare_scientific_splits(records, dataset_version="v1", split_seed=17)
        repeated = copy.deepcopy(documents)
        repeated[1] = copy.deepcopy(repeated[0])
        repeated[1]["fold"] = 1
        repeated[1]["selection_seed"] = documents[1]["selection_seed"]
        reseal(repeated[1])
        with self.assertRaisesRegex(ValueError, "external test exactly once"):
            audit_scientific_folds(records, repeated)
        documents[-1]["dataset_version"] = "other"
        reseal(documents[-1])
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            audit_scientific_folds(records, documents)

    def test_selection_is_first_inner_split_of_outer_development_only(self):
        from sklearn.model_selection import StratifiedGroupKFold
        records = fixture()
        documents = prepare_scientific_splits(records, dataset_version="v1", split_seed=17)
        for document in documents:
            external_test = set(document["lists"]["test"])
            pool = sorted((row for row in records if row["id"] not in external_test), key=lambda row: row["id"])
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=document["selection_seed"])
            train, selection = next(splitter.split([[0]] * len(pool), [row["label"] for row in pool],
                                                   [row["group"] for row in pool]))
            self.assertEqual(document["lists"]["selection"], sorted(pool[int(index)]["id"] for index in selection))
            self.assertEqual(document["lists"]["train"], sorted(pool[int(index)]["id"] for index in train))

    def test_generation_fails_on_missing_class_without_advancing_inner_split(self):
        # Outer fixture has sufficient class/group counts. Inject a fixed first
        # selection with only class 0 and a second valid split: it must not retry.
        records = [{"id": f"row-{index:03d}", "path": f"row-{index:03d}.png",
                    "group": f"group-{index:03d}", "label": index % 2} for index in range(60)]
        outer_train, outer_test = list(range(40)), list(range(40, 60))
        selection = list(range(0, 10, 2))
        train = [index for index in range(40) if index not in selection]
        inner_yields = []

        def inner_iterator():
            inner_yields.append("first")
            yield train, selection
            inner_yields.append("second")
            yield list(range(10, 40)), list(range(10))

        with patch("sklearn.model_selection.StratifiedGroupKFold") as splitter:
            splitter.return_value.split.side_effect = [iter([(outer_train, outer_test)]), inner_iterator()]
            with self.assertRaisesRegex(ValueError, "all manifest classes"):
                prepare_scientific_splits(records, dataset_version="v1", split_seed=17)
        self.assertEqual(inner_yields, ["first"])

    def test_cli_persistence_resume_and_partial_set_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(fixture()), encoding="utf-8")
            arguments = ["scientific_splits", "--manifest-json", str(manifest), "--output", str(root / "folds"),
                         "--dataset-version", "v1", "--split-seed", "17"]
            with patch.object(sys, "argv", arguments):
                self.assertEqual(main(), 0)
                with patch("augment_atlas.scientific_splits.prepare_scientific_splits", side_effect=AssertionError("regeneration")):
                    self.assertEqual(main(), 0)
                (root / "folds/fold-3.json").unlink()
                with self.assertRaisesRegex(ValueError, "partial persisted"):
                    main()


if __name__ == "__main__":
    unittest.main()
