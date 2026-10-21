"""Real CPU toy training for the clinical opt-in; container-only.

CPU gate (pinned training image, code mounted read-only):
    ATLAS_CONTAINER_CPU_TESTS=1 PYTHONPATH=src python -m unittest tests.test_clinical_training -v

A tiny linear model replaces the ResNet-50 constructor (same substitution as
test_selection_only.py); data are synthetic hash-generated images. This is CPU
evidence of the trainer/runtime contract, not a GPU or ResNet-50 trial.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REQUEST_STACK = os.environ.get("ATLAS_CONTAINER_CPU_TESTS") == "1"
HAS_STACK = REQUEST_STACK and all(importlib.util.find_spec(n) is not None for n in ("torch", "torchvision", "numpy", "PIL"))
if REQUEST_STACK and not HAS_STACK:
    raise RuntimeError("Explicit container CPU gate requires the full training stack")

from augment_atlas import runtime  # noqa: E402
from augment_atlas.contracts import canonical_hash, file_hash, load_config, recipe_identity  # noqa: E402
from augment_atlas.data import load_records, prepare_splits  # noqa: E402
from augment_atlas.evaluation import selected_epoch, validate_prediction_artifact  # noqa: E402
from augment_atlas.storage import atomic_json, digest  # noqa: E402

_spec = importlib.util.spec_from_file_location("clinical_fixtures_for_training", ROOT / "tests/clinical_fixtures.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

IDENTITY_FIELDS = ("config_sha256", "splits_sha256", "recipe_sha256", "evaluation_sha256", "best_epoch",
                   "partition", "evaluation_scope", "task", "classes", "score_kind")


def toy(cfg):
    cfg["training"].update(device="cpu", workers=0, epochs=3)
    cfg["input"]["size"] = 32
    cfg["model"]["weights"] = "none"
    return cfg


def legacy_toy(groups=12):
    cfg = load_config(ROOT / "configs/smoke/synthetic.json")
    cfg["dataset"]["synthetic_groups"] = groups
    return toy(cfg)


def clinical_toy(evaluation=fx.BINARY, *, scope=None, groups=12):
    return toy(fx.clinical_config(evaluation, groups=groups, scope=scope))


def strip_time(history):
    return [{k: v for k, v in row.items() if k != "utc"} for row in history]


@unittest.skipUnless(HAS_STACK, "CPU training dependencies unavailable; executed in the pinned container")
class ClinicalCPUTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-clinical-cpu-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data").mkdir()
        self.settings = {"code": str(self.root / "code"), "data": str(self.root / "data")}

    def tiny_model_class(self):
        torch = self.torch

        class TinyModel(torch.nn.Module):
            def __init__(model_self):
                super().__init__()
                model_self.fc = torch.nn.Linear(3, 2)

            def forward(model_self, pixels):
                return model_self.fc(pixels.mean(dim=(2, 3)))
        return TinyModel

    def guarded_run(self, cfg, out, *, forbid_test=False, corrupt_test=False, **kwargs):
        """Run the real trainer; log dataset construction order and test reads."""
        from augment_atlas import train
        splits = prepare_splits(cfg, load_records(cfg), out)
        names = {item: name for name, ids in splits["lists"].items() for item in ids}
        original = train.ClassificationDataset
        log = {"constructed": [], "test_reads": 0}
        torch = self.torch

        class GuardedDataset(original):
            def __init__(dataset_self, configuration, records, *args, **options):
                kinds = sorted({names[record["id"]] for record in records})
                log["constructed"].append(kinds)
                if forbid_test and "test" in kinds:
                    raise AssertionError("selection-only constructed an external-test dataset")
                super().__init__(configuration, records, *args, **options)

            def __getitem__(dataset_self, index):
                record = dataset_self.records[index]
                if names[record["id"]] == "test":
                    log["test_reads"] += 1
                    if forbid_test:
                        raise AssertionError("selection-only loaded an external-test image")
                    if corrupt_test:
                        generator = torch.Generator().manual_seed(index + 1000)
                        side = dataset_self.cfg["input"]["size"]
                        return torch.randn(3, side, side, generator=generator) * 9, record["label"]
                return super().__getitem__(index)

        TinyModel = self.tiny_model_class()
        with patch("torchvision.models.resnet50", side_effect=lambda **_: TinyModel()), \
             patch.object(train, "ClassificationDataset", GuardedDataset), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.dict(os.environ, {"ATLAS_CODE_COMMIT": fx.CODE_COMMIT, "ATLAS_IMAGE_ID": "sha256:cpu-test"}):
            code = train.run(cfg, out, **kwargs)
        return code, log, splits

    def spec_for(self, cfg, name):
        path = self.root / "code" / "configs/smoke" / f"{name}.json"
        atomic_json(path, cfg)
        records = load_records(cfg)
        with tempfile.TemporaryDirectory() as directory:
            splits = prepare_splits(cfg, records, directory)
        return {"config": f"configs/smoke/{name}.json", "config_sha256": digest(path), "split_hash": splits["sha256"],
                "recipe_hash": recipe_identity(cfg), "train_seed": cfg["training"]["seed"],
                "arm": cfg["augmentation"]["arm"], "code_commit": fx.CODE_COMMIT}

    def read(self, out, name):
        return json.loads((out / name).read_text(encoding="utf-8"))

    def test_legacy_eight_vs_opt_in_nine_with_identical_training(self):
        legacy, clinical = legacy_toy(), clinical_toy()
        outputs = {}
        for name, cfg in (("legacy", legacy), ("clinical", clinical)):
            out = self.root / name
            code, _, _ = self.guarded_run(cfg, out)
            self.assertEqual(code, 0)
            outputs[name] = out
            summary = runtime.validate_completion(self.settings, out, self.spec_for(cfg, name))
            self.assertEqual(summary["artifact_count"], 9 if name == "clinical" else 8)
        legacy_out, clinical_out = outputs["legacy"], outputs["clinical"]
        self.assertEqual(len(self.read(legacy_out, "completed.json")["artifacts"]), 8)
        self.assertFalse((legacy_out / "predictions.json").exists())
        self.assertNotIn("clinical_evaluation", self.read(legacy_out, "results.json"))
        self.assertNotIn("clinical_evaluation", self.read(legacy_out, "environment.json"))
        self.assertIn("predictions.json", self.read(clinical_out, "completed.json")["artifacts"])
        with self.assertRaisesRegex(ValueError, "trusted host"):
            runtime.validate_artifacts(clinical_out, self.spec_for(clinical, "clinical"))
        # The opt-in only adds a final scoring pass: trajectory and selection are unchanged.
        self.assertEqual(strip_time(self.read(legacy_out, "metrics.json")), strip_time(self.read(clinical_out, "metrics.json")))
        legacy_results, clinical_results = self.read(legacy_out, "results.json"), self.read(clinical_out, "results.json")
        for key in ("best_epoch", "selection_loss", "test"):
            self.assertEqual(legacy_results[key], clinical_results[key])

    def test_predictions_recompute_from_best_checkpoint(self):
        from augment_atlas.train import ClassificationDataset
        torch = self.torch
        cfg = clinical_toy()
        out = self.root / "recompute"
        self.assertEqual(self.guarded_run(cfg, out)[0], 0)
        artifact, results = self.read(out, "predictions.json"), self.read(out, "results.json")
        best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(best["epoch"], results["best_epoch"])
        model = self.tiny_model_class()()
        model.fc = torch.nn.Linear(3, cfg["model"]["num_classes"])
        model.load_state_dict(best["model"])
        model.eval()
        records = load_records(cfg)
        splits = self.read(out, "splits.json")
        lookup = {record["id"]: record for record in records}
        rows = [lookup[item] for item in splits["lists"]["test"]]
        loader = torch.utils.data.DataLoader(ClassificationDataset(cfg, rows), batch_size=cfg["training"]["batch_size"],
                                             shuffle=False, num_workers=0)
        scores = []
        with torch.no_grad():
            for images, _ in loader:
                scores.extend(model(images).softmax(dim=1).tolist())
        self.assertEqual([row["scores"] for row in artifact["rows"]], scores)
        validate_prediction_artifact(cfg, splits, records, artifact, results["clinical_evaluation"],
                                     expected_best_epoch=results["best_epoch"],
                                     environment=self.read(out, "environment.json"))

    def test_epoch_selection_is_independent_of_test_content(self):
        cfg = clinical_toy()
        clean_out, noisy_out = self.root / "clean", self.root / "noisy"
        code, clean_log, splits = self.guarded_run(cfg, clean_out)
        self.assertEqual(code, 0)
        code, noisy_log, _ = self.guarded_run(cfg, noisy_out, corrupt_test=True)
        self.assertEqual(code, 0)
        for log in (clean_log, noisy_log):
            tests = [i for i, kinds in enumerate(log["constructed"]) if "test" in kinds]
            self.assertEqual(tests, [len(log["constructed"]) - 1], "test is read once, after every epoch")
            self.assertEqual(log["constructed"][-1], ["test"])
            self.assertEqual(log["test_reads"], len(splits["lists"]["test"]))
        clean_metrics = self.read(clean_out, "metrics.json")
        self.assertEqual(strip_time(clean_metrics), strip_time(self.read(noisy_out, "metrics.json")))
        clean, noisy = self.read(clean_out, "results.json"), self.read(noisy_out, "results.json")
        self.assertEqual((clean["best_epoch"], clean["selection_loss"]), selected_epoch(clean_metrics))
        self.assertEqual((clean["best_epoch"], clean["selection_loss"]), (noisy["best_epoch"], noisy["selection_loss"]))
        clean_best = self.torch.load(clean_out / "best.pt", map_location="cpu", weights_only=False)
        noisy_best = self.torch.load(noisy_out / "best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(clean_best["epoch"], noisy_best["epoch"])
        self.assertEqual(clean_best["model"].keys(), noisy_best["model"].keys())
        for name, tensor in clean_best["model"].items():
            self.assertTrue(self.torch.equal(tensor, noisy_best["model"][name]), name)
        self.assertNotEqual(self.read(clean_out, "predictions.json")["rows"], self.read(noisy_out, "predictions.json")["rows"])

    def test_selection_only_opt_in_never_opens_test(self):
        cfg = clinical_toy(scope="selection-only")
        out = self.root / "selection"
        code, log, splits = self.guarded_run(cfg, out, forbid_test=True)
        self.assertEqual(code, 0)
        self.assertEqual(log["test_reads"], 0)
        results = self.read(out, "results.json")
        self.assertNotIn("test", results)
        self.assertEqual(results["evaluation_scope"], "selection-only")
        artifact = self.read(out, "predictions.json")
        self.assertEqual([row["id"] for row in artifact["rows"]], splits["lists"]["selection"])
        self.assertEqual(runtime.validate_completion(self.settings, out, self.spec_for(cfg, "selection"))["artifact_count"], 9)

    def test_resume_preserves_final_predictions_and_identity_hashes(self):
        cfg = clinical_toy()
        straight, resumed = self.root / "straight", self.root / "resumed"
        self.assertEqual(self.guarded_run(cfg, straight)[0], 0)
        self.assertEqual(self.guarded_run(cfg, resumed, stop_after_epoch=1)[0], 75)
        self.assertFalse((resumed / "predictions.json").exists())
        self.assertFalse((resumed / "completed.json").exists())
        checkpoint = file_hash(resumed / "checkpoint.pt")
        changed = copy.deepcopy(cfg)
        changed["evaluation"]["threshold"] = 0.25
        with self.assertRaisesRegex(ValueError, "different config"):
            self.guarded_run(changed, resumed, resume=str(resumed / "checkpoint.pt"))
        self.assertEqual(file_hash(resumed / "checkpoint.pt"), checkpoint)
        self.assertEqual(self.guarded_run(cfg, resumed, resume=str(resumed / "checkpoint.pt"))[0], 0)
        a, b = self.read(straight, "predictions.json"), self.read(resumed, "predictions.json")
        self.assertEqual(a["rows"], b["rows"])
        self.assertEqual({k: a[k] for k in IDENTITY_FIELDS}, {k: b[k] for k in IDENTITY_FIELDS})
        ra, rb = self.read(straight, "results.json")["clinical_evaluation"], self.read(resumed, "results.json")["clinical_evaluation"]
        self.assertEqual(ra["metrics"], rb["metrics"])
        self.assertEqual({k: ra[k] for k in IDENTITY_FIELDS}, {k: rb[k] for k in IDENTITY_FIELDS})
        self.assertEqual(strip_time(self.read(straight, "metrics.json")), strip_time(self.read(resumed, "metrics.json")))
        # environment.json is rewritten per execution (UTC); its hash is per-run provenance.
        self.assertEqual(b["environment_sha256"], canonical_hash(self.read(resumed, "environment.json")))
        spec = self.spec_for(cfg, "resume")
        for out in (straight, resumed):
            self.assertEqual(runtime.validate_completion(self.settings, out, spec)["artifact_count"], 9)

    def final_pass_dataset(self, cfg, *, on_final=None, nan_images=False):
        """Patch ClassificationDataset so the opt-in final scoring pass can be targeted.

        Evaluation datasets are constructed with (cfg, records) only: one per
        epoch for selection, then exactly one for the final pass.
        """
        from augment_atlas import train
        original = train.ClassificationDataset
        evaluations = [0]
        torch = self.torch

        class FinalPassDataset(original):
            final = False

            def __init__(dataset_self, configuration, records, *args, **options):
                super().__init__(configuration, records, *args, **options)
                if not args and not options:
                    evaluations[0] += 1
                    if evaluations[0] == cfg["training"]["epochs"] + 1:
                        dataset_self.final = True
                        if on_final is not None:
                            on_final()

            def __getitem__(dataset_self, index):
                image, label = super().__getitem__(index)
                if nan_images and dataset_self.final:
                    image = torch.full_like(image, float("nan"))
                return image, label
        return patch.object(train, "ClassificationDataset", FinalPassDataset)

    def test_interrupt_during_final_pass_writes_no_predictions(self):
        import signal
        cfg, out = clinical_toy(), self.root / "interrupted"
        # The trainer's own SIGTERM handler, invoked while the final pass is built.
        with self.final_pass_dataset(cfg, on_final=lambda: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)):
            code, _, _ = self.guarded_run(cfg, out)
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertFalse((out / "predictions.json").exists())
        self.assertFalse((out / "completed.json").exists())
        self.assertEqual(self.read(out, "progress.json")["status"], "interrupted")

    def test_nonfinite_final_logits_fail_explicitly(self):
        cfg, out = clinical_toy(), self.root / "nonfinite"
        with self.final_pass_dataset(cfg, nan_images=True), \
             self.assertRaisesRegex(RuntimeError, "non-finite clinical evaluation logits"):
            self.guarded_run(cfg, out)
        self.assertFalse((out / "predictions.json").exists())
        self.assertFalse((out / "completed.json").exists())

    def test_ordinal_and_multiclass_toy_runs_validate(self):
        for evaluation in (fx.ORDINAL, fx.MULTICLASS):
            with self.subTest(task=evaluation["task"]):
                cfg = clinical_toy(evaluation)
                out = self.root / evaluation["task"]
                self.assertEqual(self.guarded_run(cfg, out)[0], 0)
                spec = self.spec_for(cfg, evaluation["task"])
                self.assertEqual(runtime.validate_completion(self.settings, out, spec)["artifact_count"], 9)
                primary = self.read(out, "results.json")["clinical_evaluation"]["primary_metric"]
                self.assertEqual(primary, {"ordinal": "quadratic_weighted_kappa", "multiclass": "macro_ovr_auroc"}[evaluation["task"]])


if __name__ == "__main__":
    unittest.main()
