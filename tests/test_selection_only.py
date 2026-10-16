"""Selection-only contracts are stdlib-only; real toy training is container-only.

CPU gate: ATLAS_CONTAINER_CPU_TESTS=1 python -m unittest discover -s tests
          -p test_selection_only.py -v
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import runtime
from augment_atlas.cli import recipe_identity
from augment_atlas.contracts import atomic_json, canonical_hash, file_hash, load_config, validate_config
from augment_atlas.data import load_records, prepare_splits

ROOT = Path(__file__).resolve().parents[1]
_fixture_spec = importlib.util.spec_from_file_location("selection_runtime_fixtures", ROOT / "tests/test_runtime.py")
_runtime_fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_runtime_fixtures)


def config_for_scope(scope="selection-only"):
    cfg = load_config(ROOT / "configs/smoke/synthetic.json")
    cfg["recipe"]["hpo"]["protocol"] = "within-outer-fold"
    if scope is not None:
        cfg["training"]["evaluation_scope"] = scope
    return cfg


class SelectionScopeContractTests(unittest.TestCase):
    def test_missing_scope_remains_default_without_mutating_config(self):
        cfg = config_for_scope(None)
        before = copy.deepcopy(cfg)
        validate_config(cfg)
        self.assertEqual(cfg, before)
        self.assertNotIn("evaluation_scope", cfg["training"])

    def test_only_explicit_supported_scopes_are_valid(self):
        for scope in ("test", "selection-only"):
            with self.subTest(scope=scope):
                validate_config(config_for_scope(scope))
        for scope in (None, "selection", "hpo", "", True, 0):
            cfg = config_for_scope()
            cfg["training"]["evaluation_scope"] = scope
            with self.subTest(invalid=scope), self.assertRaisesRegex(ValueError, "evaluation_scope"):
                validate_config(cfg)

    def test_selection_only_requires_within_outer_fold_protocol(self):
        for protocol in ("undecided", "global-development"):
            cfg = config_for_scope()
            cfg["recipe"]["hpo"]["protocol"] = protocol
            with self.subTest(protocol=protocol), self.assertRaisesRegex(ValueError, "selection-only"):
                validate_config(cfg)

    def test_selection_only_rejects_every_nonempty_augmentation_arm(self):
        from augment_atlas.contracts import ARMS
        for arm in ARMS:
            if arm == "none":
                continue
            cfg = load_config(ROOT / f"configs/smoke/{arm}.json")
            cfg["training"]["evaluation_scope"] = "selection-only"
            cfg["recipe"]["hpo"]["protocol"] = "within-outer-fold"
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "selection-only"):
                validate_config(cfg)
        cfg = config_for_scope()
        cfg["augmentation"]["ops"] = [{"op": "gamma", "p": 0.3, "range": [0.8, 1.2]}]
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_scope_changes_config_and_frozen_recipe_identity(self):
        selection = config_for_scope()
        test = config_for_scope("test")
        self.assertNotEqual(canonical_hash(selection), canonical_hash(test))
        self.assertNotEqual(recipe_identity(selection), recipe_identity(test))

    def test_direct_runner_rejects_invalid_scope_before_heavy_imports(self):
        from augment_atlas.train import run
        cfg = config_for_scope("unsupported")
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"torch": None, "numpy": None, "torchvision": None}):
                with self.assertRaisesRegex(ValueError, "evaluation_scope"):
                    run(cfg, Path(directory))


class SelectionArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-selection-artifacts-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Reuse the existing tiny artifact fixture; checkpoint bytes are opaque
        # placeholders here. The separately gated training tests use real torch.
        self.fixture = _runtime_fixtures.ArtifactTests(methodName="test_fixture_artifact_hashes_validated")
        self.fixture.root = self.root

    def read(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def rewrite(self, name, values, remove=()):
        document = self.read(name)
        document.update(values)
        for key in remove:
            document.pop(key, None)
        atomic_json(self.root / name, document)
        if name != "completed.json":
            self.fixture.reseal(name)

    def selection_fixture(self):
        self.fixture.write_results()
        cfg = config_for_scope()
        cfg_hash = canonical_hash(cfg)
        atomic_json(self.root / "config.json", cfg)
        self.fixture.reseal("config.json")
        for name in ("results.json", "environment.json", "progress.json", "completed.json"):
            self.rewrite(name, {"config_sha256": cfg_hash})
        selection_count = len(self.read("splits.json")["lists"]["selection"])
        self.rewrite("results.json", {"evaluation_scope": "selection-only",
                     "selection": {"loss": 0.5, "accuracy": 0.5, "samples": selection_count}}, remove=("test",))
        self.rewrite("environment.json", {"evaluation_scope": "selection-only"})
        return cfg

    def test_explicit_selection_artifacts_complete_with_all_eight_checksums(self):
        cfg = self.selection_fixture()
        splits = self.read("splits.json")
        spec = {"split_hash": splits["sha256"], "train_seed": cfg["training"]["seed"],
                "arm": "none", "recipe_hash": recipe_identity(cfg)}
        result = runtime.validate_artifacts(self.root, spec)
        self.assertEqual(result["artifact_count"], 8)
        self.assertEqual(result["epoch"], cfg["training"]["epochs"])
        with self.assertRaisesRegex(ValueError, "recipe identity"):
            runtime.validate_artifacts(self.root, {**spec, "recipe_hash": "f" * 64})

    def test_default_fixture_still_requires_test_result(self):
        self.fixture.write_results()
        runtime.validate_artifacts(self.root)
        self.rewrite("results.json", {"selection": {"loss": 0.5, "accuracy": 0.5, "samples": 2}}, remove=("test",))
        with self.assertRaisesRegex(ValueError, "evaluation artifact"):
            runtime.validate_artifacts(self.root)

    def test_selection_label_cannot_replace_default_test_contract(self):
        self.fixture.write_results()
        self.rewrite("results.json", {"evaluation_scope": "selection-only",
                     "selection": {"loss": 0.5, "accuracy": 0.5, "samples": 2}}, remove=("test",))
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            runtime.validate_artifacts(self.root)

    def test_selection_scope_must_be_present_and_match_results_and_environment(self):
        for name in ("results.json", "environment.json"):
            for value in (None, "test"):
                with self.subTest(name=name, value=value):
                    self.selection_fixture()
                    self.rewrite(name, {"evaluation_scope": value})
                    with self.assertRaisesRegex(ValueError, "scope mismatch"):
                        runtime.validate_artifacts(self.root)
            self.selection_fixture()
            self.rewrite(name, {}, remove=("evaluation_scope",))
            with self.assertRaisesRegex(ValueError, "scope mismatch"):
                runtime.validate_artifacts(self.root)

    def test_external_test_key_is_forbidden_even_if_selection_is_valid(self):
        self.selection_fixture()
        self.rewrite("results.json", {"test": {"samples": 2, "loss": 0.3, "accuracy": 0.5}})
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            runtime.validate_artifacts(self.root)

    def test_selection_metrics_are_validated(self):
        for field, value in (("samples", 0), ("samples", True), ("loss", None), ("accuracy", 1.1)):
            with self.subTest(field=field, value=value):
                self.selection_fixture()
                metric = self.read("results.json")["selection"]
                metric[field] = value
                self.rewrite("results.json", {"selection": metric})
                with self.assertRaisesRegex(ValueError, "evaluation artifact"):
                    runtime.validate_artifacts(self.root)

    def test_resealed_wrong_config_identity_cannot_authorize_selection(self):
        self.selection_fixture()
        self.rewrite("results.json", {"config_sha256": "f" * 64})
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            runtime.validate_artifacts(self.root)

    def test_removing_explicit_scope_cannot_reclassify_selection_as_test(self):
        self.selection_fixture()
        cfg = self.read("config.json")
        del cfg["training"]["evaluation_scope"]
        atomic_json(self.root / "config.json", cfg)
        self.fixture.reseal("config.json")
        for name in ("results.json", "environment.json", "progress.json", "completed.json"):
            self.rewrite(name, {"config_sha256": canonical_hash(cfg)})
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            runtime.validate_artifacts(self.root)


@unittest.skipUnless(os.environ.get("ATLAS_CONTAINER_CPU_TESTS") == "1", "real toy torch loop is container-only")
class SelectionOnlyCPUTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torchvision
        cls.torch, cls.torchvision = torch, torchvision
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-selection-cpu-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def toy_config(self, scope="selection-only"):
        cfg = config_for_scope(scope)
        cfg["training"].update(device="cpu", workers=0, epochs=2)
        cfg["dataset"]["synthetic_groups"] = 6
        cfg["input"]["size"] = 32
        cfg["model"]["weights"] = "none"
        return cfg

    def guarded_run(self, cfg, out, forbid_test=True, **kwargs):
        from augment_atlas import train
        splits = prepare_splits(cfg, load_records(cfg), out)
        external_ids = set(splits["lists"]["test"])
        original_dataset = train.ClassificationDataset
        counter = {"test_reads": 0, "test_datasets": 0}
        torch = self.torch

        class GuardedDataset(original_dataset):
            def __init__(dataset_self, configuration, records, *args, **options):
                if any(record["id"] in external_ids for record in records):
                    counter["test_datasets"] += 1
                    if forbid_test:
                        raise AssertionError("selection-only constructed an external-test dataset")
                super().__init__(configuration, records, *args, **options)

            def __getitem__(dataset_self, index):
                if dataset_self.records[index]["id"] in external_ids:
                    counter["test_reads"] += 1
                    if forbid_test:
                        raise AssertionError("selection-only loaded an external-test image")
                return super().__getitem__(index)

        class TinyModel(torch.nn.Module):
            def __init__(model_self):
                super().__init__()
                model_self.fc = torch.nn.Linear(3, 2)

            def forward(model_self, pixels):
                return model_self.fc(pixels.mean(dim=(2, 3)))

        with patch("torchvision.models.resnet50", side_effect=lambda **_: TinyModel()), \
             patch.object(train, "ClassificationDataset", GuardedDataset), \
             patch.object(torch.cuda, "is_available", return_value=False):
            code = train.run(cfg, out, **kwargs)
        return code, counter, splits

    def test_selection_only_real_forward_backward_never_opens_external_test(self):
        cfg = self.toy_config()
        out = self.root / "selection"
        code, counter, _ = self.guarded_run(cfg, out)
        self.assertEqual(code, 0)
        self.assertEqual(counter, {"test_reads": 0, "test_datasets": 0})
        results = json.loads((out / "results.json").read_text())
        self.assertEqual(results["evaluation_scope"], "selection-only")
        self.assertIn("selection", results)
        self.assertNotIn("test", results)
        self.assertEqual(json.loads((out / "environment.json").read_text())["evaluation_scope"], "selection-only")
        self.assertEqual(runtime.validate_artifacts(out)["artifact_count"], 8)

    def test_missing_scope_real_loop_preserves_test_result_shape_and_reads_test_once(self):
        cfg = self.toy_config(None)
        out = self.root / "test"
        code, counter, splits = self.guarded_run(cfg, out, forbid_test=False)
        self.assertEqual(code, 0)
        self.assertEqual(counter["test_reads"], len(splits["lists"]["test"]))
        self.assertEqual(counter["test_datasets"], 1)
        results = json.loads((out / "results.json").read_text())
        self.assertIn("test", results)
        self.assertNotIn("selection", results)
        self.assertNotIn("evaluation_scope", results)
        self.assertEqual(json.loads((out / "environment.json").read_text())["evaluation_scope"], "test")
        runtime.validate_artifacts(out)

    def test_controlled_resume_preserves_scope_and_rejects_scope_change(self):
        cfg = self.toy_config()
        out = self.root / "resume"
        code, counter, _ = self.guarded_run(cfg, out, stop_after_epoch=1)
        self.assertEqual(code, 75)
        self.assertEqual(counter["test_reads"], 0)
        self.assertFalse((out / "completed.json").exists())
        checkpoint_hash = file_hash(out / "checkpoint.pt")
        changed = copy.deepcopy(cfg)
        changed["training"]["evaluation_scope"] = "test"
        with self.assertRaisesRegex(ValueError, "different config"):
            self.guarded_run(changed, out, resume=str(out / "checkpoint.pt"))
        self.assertEqual(file_hash(out / "checkpoint.pt"), checkpoint_hash)
        code, counter, _ = self.guarded_run(cfg, out, resume=str(out / "checkpoint.pt"))
        self.assertEqual(code, 0)
        self.assertEqual(counter["test_reads"], 0)
        checkpoint = self.torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["config_sha256"], canonical_hash(cfg))
        self.assertEqual(checkpoint["completed_epoch"], 2)
        runtime.validate_artifacts(out)


if __name__ == "__main__":
    unittest.main()
