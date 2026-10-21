"""Slice S1 (part B): global hpo/final path of the trainer.

Pure tests (stdlib only) cover the config dispatch, the explicit toy override,
resume lineage refusal and the phase-specific input loading. The real toy
torch loop is container-only:

    ATLAS_CONTAINER_CPU_TESTS=1 PYTHONPATH=src python -m unittest tests.test_global_trainer -v

A tiny linear model replaces the ResNet-50 constructor (same substitution as
test_selection_only.py) and a ``GuardedDataset`` counts every image read. The
CPU device and 0 workers come ONLY from the explicit, gated
``trainer_override`` argument, recorded in every artifact; the configs keep
TRAINER_CONSTANTS_V1 (device=cuda, workers=2) unchanged.
"""
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import global_evaluation as ge
from augment_atlas import global_jobs as gj
from augment_atlas import global_development as gd
from augment_atlas.contracts import ARMS, canonical_hash as H, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
REQUEST_STACK = os.environ.get("ATLAS_CONTAINER_CPU_TESTS") == "1"
HAS_STACK = REQUEST_STACK and all(importlib.util.find_spec(n) is not None
                                  for n in ("torch", "torchvision", "numpy", "PIL"))
if REQUEST_STACK and not HAS_STACK:
    raise RuntimeError("Explicit container CPU gate requires the full training stack")

DATASET = "toy_global"
HPO_SEED = 101
TOY = {"device": "cpu", "workers": 0, "reason": "test_global_trainer CPU toy loop"}
TASK = {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
        "decision_rule": "fixed_threshold", "missing_policy": "undefined_no_drop",
        "requires_patient_identity": False}


def signed(value):
    result = copy.deepcopy(value)
    result.pop("sha256", None)
    result["sha256"] = H(result)
    return result


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def presets():
    return {arm: json.loads((ROOT / "configs/smoke/candidate-v2" / f"{arm}.json").read_text(encoding="utf-8"))["augmentation"]
            for arm in ARMS}


def final_policy():
    return {"initialization": "fresh_imagenet_each_arm_fold_seed", "warm_start": False,
            "training_pool": "D_union_E_minus_test", "folds": list(range(5)), "arms": list(ARMS),
            "scheduler_duration": "fixed_epochs_recompute_steps_from_final_pool",
            "step_count": {"unit": "optimizer_update", "drop_last": False, "accumulation_tail": "step",
                           "per_epoch": "ceil(ceil(train_images/batch_size)/accumulation_steps)"},
            "early_stopping": False, "checkpoint_selection": "last_epoch", "selection_on_E": False,
            "seeds": [11, 13, 19], "augmentation_presets_sha256": H(presets())}


def png_bytes(seed):
    from PIL import Image
    import numpy as np
    pixels = np.random.default_rng(seed).integers(0, 256, size=(32, 32), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


class GlobalFixture:
    """Private data root: images, manifest, pools, external freeze, D-view, configs.

    8 groups x 2 labels; D = g0..g2, D_fit = g0,g1, D_select = g2, E = g3..g7,
    E_f = g(f+3). Image IDs equal their manifest paths (``load_records``).
    """

    def __init__(self, root, *, weights=b"opaque weights", png=False, epochs=2):
        self.root = Path(root) / "data"
        self.dview = Path(root) / "dview"
        self.rows = []
        lines = io.StringIO()
        writer = csv.writer(lines, lineterminator="\n")
        writer.writerow(["path", "group", "label", "sha256"])
        for group in range(8):
            for label in range(2):
                path = f"images/g{group}-{label}.png"
                data = png_bytes(group * 2 + label) if png else f"image {group} {label}".encode()
                target = self.root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                writer.writerow([path, f"g{group}", label, sha256_bytes(data)])
                self.rows.append({"id": path, "path": path, "group": f"g{group}", "label": label})
        self.rows.sort(key=lambda row: row["id"])
        manifest = lines.getvalue().encode()
        (self.root / "manifest.csv").write_bytes(manifest)
        ids = lambda groups: sorted(r["id"] for r in self.rows if int(r["group"][1:]) in groups)
        self.pools = signed({
            "schema_version": 1, "protocol": gd.PROTOCOL, "scientific_status": "provisional",
            "dataset": {"id": DATASET, "version": "v1"}, "records_sha256": H(self.rows),
            "source_sha256": sha256_bytes(manifest),
            "provenance": {key: H(key) for key in ("preprocessing_sha256", "labels_sha256",
                                                   "duplicates_sha256", "exclusions_sha256")},
            "grouping": {"namespace": "fixture-group", "source": "manual fixture", "identity_level": "group",
                         "linkage_status": "unverified", "evidence_sha256": None, "prior_exposure": "none_documented"},
            "task": dict(TASK),
            "partition_provenance": {"algorithm": "manual-fixture", "version": "1", "source": "fixed lists",
                                     "generator_version": "no generator",
                                     "seeds": {"development": 17, "selection": 29, "evaluation": 31},
                                     "intended_fractions": {"D": 0.2, "D_fit_given_D": 0.8, "E_test_given_E": 0.2}},
            "lists": {"D": ids(range(3)), "D_fit": ids(range(2)), "D_select": ids([2]), "E": ids(range(3, 8))},
            "evaluation_folds": [{"fold": f, "test_ids": ids([f + 3])} for f in range(5)]})
        self.write_doc(self.root, "pools", self.pools)
        self.weights = weights
        self.runtime = {"commit": "c" * 40, "environment_sha256": H("fixture environment"),
                        "imagenet_weights_sha256": sha256_bytes(weights)}
        self.recipe = gj.external_recipe_q2(self.pools, self.runtime, color="grayscale_to_rgb")
        self.recipe.update(epochs=epochs, batch_size=4)
        self.recipe["input"]["size"] = 32
        self.freeze = gj.build_external_freeze(
            self.rows, self.pools, expected_pools_sha256=self.pools["sha256"], recipe=self.recipe,
            source=gj.Q2_EXTERNAL_SOURCE, runtime=self.runtime, final_policy=final_policy(),
            frozen_at_utc="2026-10-19T12:00:00Z")
        self.write_doc(self.root, "external-freeze", self.freeze)
        source = {"adapter": "manifest_csv", "manifest": "manifest.csv", "manifest_sha256": sha256_bytes(manifest)}
        self.final_cfgs = gj.build_final_configs(
            self.rows, self.pools, expected_pools_sha256=self.pools["sha256"], dataset_source=source,
            freeze=self.freeze, freeze_kind="external_fixed", expected_freeze_sha256=self.freeze["sha256"],
            augmentation_presets=presets(), seeds=[11, 13])
        # D-view: only D images and the hpo-view (no pools, no manifest).
        d_ids = set(self.pools["lists"]["D"])
        for row in self.rows:
            if row["id"] in d_ids:
                target = self.dview / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.root / row["path"], target)
        self.hpo_view = signed({
            "schema_version": 1, "protocol": ge.HPO_VIEW_PROTOCOL, "scientific_status": "provisional",
            "dataset": copy.deepcopy(self.pools["dataset"]), "task": copy.deepcopy(self.pools["task"]),
            "pools_sha256": self.pools["sha256"],
            "records": [copy.deepcopy(r) for r in self.rows if r["id"] in d_ids],
            "lists": {"train": list(self.pools["lists"]["D_fit"]), "selection": list(self.pools["lists"]["D_select"])},
            "view_files_sha256": H("D view listing")})
        self.write_doc(self.dview, "hpo-view", self.hpo_view)
        self.hpo_cfg = self.make_hpo_cfg(self.hpo_view["sha256"])

    def write_doc(self, root, kind, document):
        target = Path(root) / "global" / DATASET / f"{kind}-{document['sha256']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def make_hpo_cfg(self, view_sha256, seed=HPO_SEED):
        sha = self.pools["sha256"]
        cfg = {"schema_version": 1, "phase": "hpo", "scientific_status": "provisional",
               "protocol_version": gj.HPO_PROTOCOL, "dataset": copy.deepcopy(self.final_cfgs["none-f0-s11.json"]["dataset"]),
               "global": {"pools_file": f"global/{DATASET}/pools-{sha}.json", "pools_sha256": sha, "view": "hpo",
                          "hpo_view_file": f"global/{DATASET}/hpo-view-{view_sha256}.json",
                          "hpo_view_sha256": view_sha256, "plan_sha256": H("plan"),
                          "search_space_sha256": H("search space"), "candidate_id": "c00",
                          "candidate_config_sha256": H("c00"), "freeze_kind": None, "freeze_file": None,
                          "freeze_sha256": None, "fold": None},
               "model": gj._model_block(self.recipe, self.pools["task"]), "input": copy.deepcopy(self.recipe["input"]),
               "augmentation": {"arm": "none", "ops": []}, "training": gj.recipe_to_training(self.recipe, seed),
               "evaluation": gj._evaluation_block(self.pools["task"], self.recipe)}
        return gj.validate_global_config(cfg, recipe=self.recipe)

    def final(self, arm="none", fold=0, seed=11):
        return copy.deepcopy(self.final_cfgs[f"{arm}-f{fold}-s{seed}.json"])

    def final_view(self, fold=0):
        return gj.derive_final_view_doc(self.rows, self.pools, fold, expected_pools_sha256=self.pools["sha256"],
                                        freeze_sha256=self.freeze["sha256"])


def data_env(root):
    return patch.dict(os.environ, {"ATLAS_DATA_ROOT": str(root)})


class GlobalDispatchTests(unittest.TestCase):
    """contracts.validate_config: hpo/final go to validate_global_config only."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="atlas-global-dispatch-")
        cls.fx = GlobalFixture(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_global_configs_validate_through_contracts_and_load_config(self):
        for cfg in (self.fx.hpo_cfg, self.fx.final()):
            with self.subTest(phase=cfg["phase"]):
                before = copy.deepcopy(cfg)
                self.assertIs(validate_config(cfg), cfg)
                self.assertEqual(cfg, before)
                path = Path(self.temp.name) / f"{cfg['phase']}.json"
                path.write_bytes(gj.render_config_bytes(cfg))
                self.assertEqual(load_config(path), cfg)

    def test_global_rules_apply_after_dispatch(self):
        cases = []
        for key in sorted(gj.FORBIDDEN_KEYS):
            cfg = self.fx.final()
            cfg["training"][key] = "anything"
            cases.append((cfg, "init_checkpoint_forbidden"))
        hpo = copy.deepcopy(self.fx.hpo_cfg)
        hpo["augmentation"] = copy.deepcopy(self.fx.final("basic")["augmentation"])
        cases.append((hpo, "hpo_augmentation_forbidden"))
        changed = self.fx.final()
        changed["training"]["device"] = "cpu"
        cases.append((changed, "global_config_invalid"))
        changed = self.fx.final()
        changed["training"]["workers"] = 0
        cases.append((changed, "global_config_invalid"))
        changed = self.fx.final()
        changed["training"]["optimizer"]["name"] = "sgd"
        cases.append((changed, "recipe_adapter_unsupported"))
        changed = self.fx.final()
        changed["splits"] = {"version": "group-hash-v1"}
        cases.append((changed, "global_config_invalid"))
        for cfg, code in cases:
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, f"^{code}:"):
                validate_config(cfg)

    def test_legacy_and_unknown_phases_keep_the_legacy_validator(self):
        legacy = load_config(ROOT / "configs/smoke/synthetic.json")
        self.assertEqual(validate_config(copy.deepcopy(legacy)), legacy)
        for phase in ("validation", "HPO", "Final", None):
            cfg = copy.deepcopy(legacy)
            cfg["phase"] = phase
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, "v1 accepts only explicit phase=smoke"):
                validate_config(cfg)
        smoke_shaped_hpo = copy.deepcopy(legacy)
        smoke_shaped_hpo["phase"] = "hpo"
        with self.assertRaisesRegex(ValueError, "^global_config_invalid:"):
            validate_config(smoke_shaped_hpo)

    def test_trainer_constants_are_not_changed_by_any_of_this(self):
        self.assertEqual(gj.TRAINER_CONSTANTS_V1["device"], "cuda")
        self.assertEqual(gj.TRAINER_CONSTANTS_V1["workers"], 2)
        for cfg in (self.fx.hpo_cfg, *self.fx.final_cfgs.values()):
            self.assertEqual((cfg["training"]["device"], cfg["training"]["workers"]), ("cuda", 2))


class GlobalRunnerPreTorchTests(unittest.TestCase):
    """Refusals that must happen before torch/numpy/torchvision are imported."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-global-pretorch-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.fx = GlobalFixture(self.base)
        self.out = self.base / "out"

    def run_without_ml(self, cfg, *args, **kwargs):
        from augment_atlas import train
        with patch.dict("sys.modules", {"torch": None, "numpy": None, "torchvision": None}), data_env(self.fx.root):
            return train.run(cfg, self.out, *args, **kwargs)

    def test_invalid_global_config_rejected_before_heavy_imports(self):
        cfg = self.fx.final()
        cfg["training"]["init_checkpoint"] = "/runs/hpo/best.pt"
        with self.assertRaisesRegex(ValueError, "^init_checkpoint_forbidden:"):
            self.run_without_ml(cfg)
        self.assertFalse(self.out.exists())

    def test_trainer_override_is_explicit_gated_and_exact(self):
        cfg = self.fx.final()
        with patch.dict(os.environ, {"ATLAS_CONTAINER_CPU_TESTS": "0"}):
            with self.assertRaisesRegex(ValueError, "^global_config_invalid: trainer_override requires"):
                self.run_without_ml(cfg, trainer_override=dict(TOY))
        with patch.dict(os.environ, {"ATLAS_CONTAINER_CPU_TESTS": "1"}):
            for bad in ({"device": "cpu", "workers": 0}, {**TOY, "device": "cuda"}, {**TOY, "workers": 2},
                        {**TOY, "reason": " "}, {**TOY, "extra": True}, ["cpu"]):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "^global_config_invalid: trainer_override"):
                    self.run_without_ml(cfg, trainer_override=bad)
        legacy = load_config(ROOT / "configs/smoke/synthetic.json")
        with self.assertRaisesRegex(ValueError, "trainer_override exists only for global"):
            self.run_without_ml(legacy, trainer_override=dict(TOY))
        self.assertFalse(self.out.exists())

    def test_foreign_checkpoint_path_is_lineage_mismatch_and_bytes_preserved(self):
        foreign = self.base / "hpo-attempt" / "checkpoint.pt"
        foreign.parent.mkdir()
        foreign.write_bytes(b"pretend HPO checkpoint bytes")
        before = foreign.read_bytes()
        with self.assertRaisesRegex(ValueError, "^checkpoint_lineage_mismatch:"):
            self.run_without_ml(self.fx.final(), str(foreign))
        self.assertEqual(foreign.read_bytes(), before)
        self.assertFalse((self.out / "checkpoint.pt").exists())
        with self.assertRaisesRegex(ValueError, "^checkpoint_lineage_mismatch:"):
            self.run_without_ml(self.fx.final(), str(self.out / "checkpoint.pt"))  # resume without a checkpoint

    def test_existing_checkpoint_requires_resume_and_completed_attempt_is_final(self):
        self.out.mkdir()
        (self.out / "checkpoint.pt").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "explicit --resume"):
            self.run_without_ml(self.fx.final())
        (self.out / "completed.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already complete"):
            self.run_without_ml(self.fx.final())

    def test_other_config_in_attempt_directory_rejected(self):
        self.out.mkdir()
        (self.out / "config.json").write_bytes(gj.render_config_bytes(self.fx.final("basic")))
        with self.assertRaisesRegex(ValueError, "different config"):
            self.run_without_ml(self.fx.final())


class GlobalInputTests(unittest.TestCase):
    """Phase-specific trusted inputs (no torch): what each container may read."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-global-inputs-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.fx = GlobalFixture(self.base)

    def inputs(self, cfg, root):
        from augment_atlas import train
        with data_env(root):
            return train._global_inputs(cfg)

    def test_final_inputs_train_on_D_union_E_minus_Ef_and_evaluate_Ef_only(self):
        pools = self.fx.pools
        for fold in range(5):
            with self.subTest(fold=fold):
                inputs = self.inputs(self.fx.final(fold=fold), self.fx.root)
                view = self.fx.final_view(fold)
                test = pools["evaluation_folds"][fold]["test_ids"]
                train_ids = [row["id"] for row in inputs["train_records"]]
                self.assertEqual(inputs["view"], view)
                self.assertEqual(inputs["eval_ids"], test)
                self.assertEqual([row["id"] for row in inputs["eval_records"]], test)
                self.assertEqual(set(train_ids), set(pools["lists"]["D"]) | (set(pools["lists"]["E"]) - set(test)))
                self.assertFalse(set(inputs["eval_ids"]) & set(pools["lists"]["D"]))
                self.assertEqual(inputs["records"], self.fx.rows)

    def test_hpo_inputs_come_only_from_the_D_view(self):
        self.assertFalse((self.fx.dview / "manifest.csv").exists())
        self.assertFalse(any(self.fx.dview.rglob("pools-*.json")))
        real_open = Path.open
        opened = []

        def spy(path, *args, **kwargs):
            opened.append(Path(path).resolve())
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", spy):
            inputs = self.inputs(self.fx.hpo_cfg, self.fx.dview)
        self.assertTrue(opened)
        self.assertTrue(all(path.is_relative_to(self.fx.dview.resolve()) for path in opened))
        e_ids = set(self.fx.pools["lists"]["E"])
        seen = {row["id"] for row in inputs["records"]} | set(inputs["eval_ids"])
        self.assertFalse(seen & e_ids)
        self.assertEqual(inputs["eval_ids"], self.fx.pools["lists"]["D_select"])
        self.assertEqual([r["id"] for r in inputs["train_records"]], self.fx.pools["lists"]["D_fit"])
        self.assertIsNone(inputs["pools"])

    def test_hpo_view_tampering_and_missing_images(self):
        fx = self.fx
        tampered = copy.deepcopy(fx.hpo_view)
        tampered["lists"]["train"] = tampered["lists"]["train"][1:]
        tampered = signed(tampered)  # re-signed: not caught by a stale checksum
        fx.write_doc(fx.dview, "hpo-view", tampered)
        with self.assertRaisesRegex(ValueError, "^view_hash_mismatch: hpo view lists must partition"):
            self.inputs(fx.make_hpo_cfg(tampered["sha256"]), fx.dview)
        stale = copy.deepcopy(fx.hpo_view)
        stale["view_files_sha256"] = H("other listing")
        target = fx.dview / "global" / DATASET / f"hpo-view-{fx.hpo_view['sha256']}.json"
        original = target.read_bytes()
        target.write_text(json.dumps(stale), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "^view_hash_mismatch: hpo view checksum"):
            self.inputs(fx.hpo_cfg, fx.dview)
        target.write_bytes(original)
        (fx.dview / fx.pools["lists"]["D_select"][0]).unlink()
        with self.assertRaisesRegex(ValueError, "^records_identity_mismatch:"):
            self.inputs(fx.hpo_cfg, fx.dview)

    def test_final_freeze_pools_and_recipe_binding(self):
        fx = self.fx
        changed = fx.final()
        changed["training"]["optimizer"]["lr"] = 0.001
        with self.assertRaisesRegex(ValueError, "^recipe_mismatch:"):
            self.inputs(changed, fx.root)
        freeze_path = fx.root / fx.final()["global"]["freeze_file"]
        original = freeze_path.read_bytes()
        forged = json.loads(original)
        forged["recipe"]["optimizer"]["parameters"]["lr"] = 0.001  # not re-signed
        freeze_path.write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "^freeze_not_sealed:"):
            self.inputs(fx.final(), fx.root)
        freeze_path.write_text(json.dumps(signed(forged)), encoding="utf-8")  # re-signed: other identity
        with self.assertRaisesRegex(ValueError, "^freeze_not_sealed:"):
            self.inputs(fx.final(), fx.root)
        freeze_path.write_bytes(original)
        pools_path = fx.root / fx.final()["global"]["pools_file"]
        moved = copy.deepcopy(fx.pools)
        moved["lists"]["D_fit"], moved["lists"]["D_select"] = (
            sorted(moved["lists"]["D_fit"][2:] + moved["lists"]["D_select"]), moved["lists"]["D_fit"][:2])
        pools_path.write_text(json.dumps(signed(moved)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "^pools_anchor_mismatch:"):
            self.inputs(fx.final(), fx.root)

    def test_final_manifest_records_must_match_the_pools(self):
        fx = self.fx
        manifest = fx.root / "manifest.csv"
        text = manifest.read_text(encoding="utf-8").replace("images/g7-1.png,g7,1", "images/g7-1.png,g6,1")
        manifest.write_text(text, encoding="utf-8", newline="\n")
        cfg = fx.final()
        cfg["dataset"]["manifest_sha256"] = sha256_bytes(manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "^records_identity_mismatch:"):
            self.inputs(cfg, fx.root)

    def test_hpo_predictions_from_view_equal_the_pools_based_builder(self):
        fx = self.fx
        ids = fx.pools["lists"]["D_select"]
        scores = [[0.75, 0.25] if i % 2 else [0.125, 0.875] for i, _ in enumerate(ids)]
        local = ge.build_hpo_predictions_from_view(fx.hpo_cfg, fx.hpo_view, scores, epoch=fx.recipe["epochs"])
        hosted = ge.build_global_predictions(fx.hpo_cfg, fx.hpo_view, fx.rows, scores, pools=fx.pools,
                                             epoch=fx.recipe["epochs"])
        self.assertEqual(H(local), H(hosted))
        ge.validate_global_predictions(fx.hpo_cfg, fx.hpo_view, fx.rows, json.loads(json.dumps(local)), pools=fx.pools)
        with self.assertRaisesRegex(ValueError, "^epoch_selection_forbidden:"):
            ge.build_hpo_predictions_from_view(fx.hpo_cfg, fx.hpo_view, scores, epoch=1)
        with self.assertRaisesRegex(ValueError, "^global_config_invalid:"):
            ge.build_hpo_predictions_from_view(fx.final(), fx.hpo_view, scores, epoch=fx.recipe["epochs"])

    def test_cli_preflight_for_global_configs(self):
        from augment_atlas import train
        for cfg, root in ((self.fx.hpo_cfg, self.fx.dview), (self.fx.final(fold=2), self.fx.root)):
            with self.subTest(phase=cfg["phase"]):
                path = self.base / f"{cfg['phase']}.json"
                path.write_bytes(gj.render_config_bytes(cfg))
                stdout = io.StringIO()
                argv = ["train", "--config", str(path), "--output", str(self.base / f"pre-{cfg['phase']}"), "--preflight"]
                with patch("sys.argv", argv), patch("sys.stdout", stdout), data_env(root), \
                     patch.dict("sys.modules", {"torch": None, "numpy": None, "torchvision": None}):
                    self.assertEqual(train.main(), 0)
                report = json.loads(stdout.getvalue())
                self.assertEqual(report["config_sha256"], ge.config_sha256(cfg))
                expected_view = self.fx.hpo_view if cfg["phase"] == "hpo" else self.fx.final_view(2)
                self.assertEqual(report["splits_sha256"], expected_view["sha256"])
                self.assertFalse((self.base / f"pre-{cfg['phase']}" / "checkpoint.pt").exists())


@unittest.skipUnless(HAS_STACK, "real toy torch loop is container-only (ATLAS_CONTAINER_CPU_TESTS=1)")
class GlobalTrainerCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-global-cpu-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        torch = self.torch

        class TinyModel(torch.nn.Module):
            def __init__(model_self):
                super().__init__()
                model_self.fc = torch.nn.Linear(3, 2)

            def forward(model_self, pixels):
                return model_self.fc(pixels.mean(dim=(2, 3)))
        self.TinyModel = TinyModel
        torch.manual_seed(0)
        stream = io.BytesIO()
        torch.save(TinyModel().state_dict(), stream)
        self.fx = GlobalFixture(self.base, weights=stream.getvalue(), png=True, epochs=2)
        self.hub = self.base / "hub"
        from torchvision.models import ResNet50_Weights
        weights = self.hub / "checkpoints" / Path(ResNet50_Weights.IMAGENET1K_V2.url).name
        weights.parent.mkdir(parents=True)
        weights.write_bytes(self.fx.weights)

    def guarded_run(self, cfg, out, root, *, resume=None, stop_after_epoch=None):
        from augment_atlas import train
        torch = self.torch
        pools = self.fx.pools
        e_ids, d_select = set(pools["lists"]["E"]), set(pools["lists"]["D_select"])
        e_f = set(pools["evaluation_folds"][cfg["global"]["fold"]]["test_ids"]) if cfg["phase"] == "final" else set()
        counter = {"E_datasets": 0, "E_reads": 0, "E_f_datasets": 0, "E_f_reads": 0, "D_select_datasets": 0,
                   "D_select_reads": 0, "early_eval_reads": 0, "train_sets": []}
        original = train.ClassificationDataset

        def completed_epochs():
            path = Path(out) / "metrics.json"
            return len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else 0

        class GuardedDataset(original):
            def __init__(dataset_self, configuration, records, *args, **options):
                ids = {record["id"] for record in records}
                counter["E_datasets"] += bool(ids & e_ids)
                counter["E_f_datasets"] += bool(ids & e_f)
                counter["D_select_datasets"] += bool(ids & d_select) and cfg["phase"] == "hpo" and ids <= d_select
                if options.get("training") or (args and args[-1] is True):
                    counter["train_sets"].append(sorted(ids))
                if cfg["phase"] == "hpo" and ids & e_ids:
                    raise AssertionError("HPO constructed a dataset with an E identity")
                super().__init__(configuration, records, *args, **options)

            def __getitem__(dataset_self, index):
                item = dataset_self.records[index]["id"]
                counter["E_reads"] += item in e_ids
                is_eval = item in e_f if cfg["phase"] == "final" else (item in d_select and not dataset_self.training)
                if is_eval:
                    counter["E_f_reads" if cfg["phase"] == "final" else "D_select_reads"] += 1
                    if completed_epochs() != cfg["training"]["epochs"]:
                        counter["early_eval_reads"] += 1
                return super().__getitem__(index)

        with patch("torchvision.models.resnet50", side_effect=lambda **_: self.TinyModel()), \
             patch.object(train, "ClassificationDataset", GuardedDataset), \
             patch.object(torch.hub, "get_dir", return_value=str(self.hub)), \
             patch.object(torch.cuda, "is_available", return_value=False), data_env(root):
            code = train.run(cfg, Path(out), resume, stop_after_epoch, trainer_override=dict(TOY))
        return code, counter

    def read(self, out, name):
        return json.loads((Path(out) / name).read_text(encoding="utf-8"))

    def assert_common_artifacts(self, out, cfg):
        epochs = cfg["training"]["epochs"]
        completed = self.read(out, "completed.json")
        self.assertEqual(sorted(completed["artifacts"]), sorted(gj_required()))
        self.assertEqual(completed["trainer_override"], TOY)
        history = self.read(out, "metrics.json")
        self.assertEqual([row["epoch"] for row in history], list(range(1, epochs + 1)))
        for row in history:
            self.assertFalse({"test", "selection", "E_test", "D_select", "evaluation"} & set(row))
        results = self.read(out, "results.json")
        self.assertEqual((results["best_epoch"], results["epochs"], results["checkpoint_selection"]),
                         (epochs, epochs, "last_epoch"))
        self.assertEqual(results["trainer_override"], TOY)
        self.assertFalse(results["execution_verified"] or results["scientific_eligible"])
        environment = self.read(out, "environment.json")
        self.assertEqual(environment["trainer_override"], TOY)
        self.assertEqual(environment["trainer_constants"], gj.TRAINER_CONSTANTS_V1)
        self.assertEqual((environment["device"], environment["workers"]), ("cpu", 0))
        self.assertEqual(environment["initial_state_sha256"], results["initial_state_sha256"])
        self.assertEqual((Path(out) / "config.json").read_bytes(), gj.render_config_bytes(cfg))
        self.assertEqual(results["config_sha256"], ge.config_sha256(cfg))
        best = self.torch.load(Path(out) / "best.pt", map_location="cpu", weights_only=False)
        self.assertEqual((best["epoch"], best["checkpoint_selection"]), (epochs, "last_epoch"))
        checkpoint = self.torch.load(Path(out) / "checkpoint.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["lineage"]["phase"], cfg["phase"])
        self.assertEqual(checkpoint["completed_epoch"], epochs)
        return results, environment

    def test_hpo_trains_D_fit_never_builds_E_and_reads_D_select_once_after_last_epoch(self):
        cfg, out = self.fx.hpo_cfg, self.base / "hpo"
        code, counter = self.guarded_run(cfg, out, self.fx.dview)
        self.assertEqual(code, 0)
        self.assertEqual((counter["E_datasets"], counter["E_reads"]), (0, 0))
        self.assertEqual(counter["D_select_reads"], len(self.fx.pools["lists"]["D_select"]))
        self.assertEqual(counter["D_select_datasets"], 1)
        self.assertEqual(counter["early_eval_reads"], 0)
        self.assertTrue(all(ids == self.fx.pools["lists"]["D_fit"] for ids in counter["train_sets"]))
        self.assertEqual(len(counter["train_sets"]), cfg["training"]["epochs"])
        results, _ = self.assert_common_artifacts(out, cfg)
        self.assertIn("selection", results)
        self.assertNotIn("test", results)
        self.assertEqual(results["selection_evaluations"], 1)
        self.assertEqual(self.read(out, "splits.json"), self.fx.hpo_view)
        artifact = self.read(out, "predictions.json")
        self.assertEqual(artifact["partition"], "D_select")
        check = ge.validate_global_predictions(cfg, self.fx.hpo_view, self.fx.rows, artifact, pools=self.fx.pools)
        self.assertEqual(check["samples"], len(self.fx.pools["lists"]["D_select"]))

    def test_final_reads_E_f_exactly_once_after_last_epoch(self):
        cfg, out = self.fx.final("basic", fold=1), self.base / "final"
        code, counter = self.guarded_run(cfg, out, self.fx.root)
        self.assertEqual(code, 0)
        test_ids = self.fx.pools["evaluation_folds"][1]["test_ids"]
        self.assertEqual(counter["E_f_reads"], len(test_ids))
        self.assertEqual(counter["E_f_datasets"], 1)
        self.assertEqual(counter["early_eval_reads"], 0)
        view = self.fx.final_view(1)
        self.assertEqual(len(counter["train_sets"]), cfg["training"]["epochs"])
        self.assertTrue(all(ids == view["train"] for ids in counter["train_sets"]))
        results, _ = self.assert_common_artifacts(out, cfg)
        self.assertIn("test", results)
        self.assertNotIn("selection", results)
        self.assertEqual((results["test"]["samples"], results["test_evaluations"]), (len(test_ids), 1))
        self.assertEqual(self.read(out, "splits.json"), view)
        artifact = self.read(out, "predictions.json")
        self.assertEqual((artifact["partition"], artifact["epoch"], artifact["epochs"]), ("E_test", 2, 2))
        check = ge.validate_global_predictions(cfg, view, self.fx.rows, artifact, pools=self.fx.pools)
        self.assertEqual(check["content_sha256"], results["clinical_evaluation"]["content_sha256"])

    def test_resume_with_hpo_checkpoint_is_lineage_mismatch_and_bytes_preserved(self):
        hpo_out = self.base / "hpo"
        code, _ = self.guarded_run(self.fx.hpo_cfg, hpo_out, self.fx.dview, stop_after_epoch=1)
        self.assertEqual(code, 75)
        hpo_checkpoint = hpo_out / "checkpoint.pt"
        before = hpo_checkpoint.read_bytes()
        final_cfg, final_out = self.fx.final(), self.base / "final"
        with self.assertRaisesRegex(ValueError, "^checkpoint_lineage_mismatch:"):
            self.guarded_run(final_cfg, final_out, self.fx.root, resume=str(hpo_checkpoint))
        self.assertEqual(hpo_checkpoint.read_bytes(), before)
        self.assertFalse((final_out / "checkpoint.pt").exists())
        # The same HPO bytes planted in the final attempt are refused by lineage.
        final_out.mkdir(exist_ok=True)
        planted = final_out / "checkpoint.pt"
        planted.write_bytes(before)
        with self.assertRaisesRegex(ValueError, "^checkpoint_lineage_mismatch:"):
            code, counter = self.guarded_run(final_cfg, final_out, self.fx.root, resume=str(planted))
        self.assertEqual(planted.read_bytes(), before)
        self.assertEqual(hpo_checkpoint.read_bytes(), before)
        self.assertFalse((final_out / "predictions.json").exists())

    def test_same_attempt_resume_completes_and_reproduces_uninterrupted_predictions(self):
        cfg = self.fx.final("none", fold=3)
        out = self.base / "resumed"
        code, counter = self.guarded_run(cfg, out, self.fx.root, stop_after_epoch=1)
        self.assertEqual((code, counter["E_f_reads"], counter["E_f_datasets"]), (75, 0, 0))
        self.assertFalse((out / "completed.json").exists())
        code, counter = self.guarded_run(cfg, out, self.fx.root, resume=str(out / "checkpoint.pt"))
        self.assertEqual(code, 0)
        self.assertEqual(counter["E_f_reads"], len(self.fx.pools["evaluation_folds"][3]["test_ids"]))
        self.assertEqual(counter["early_eval_reads"], 0)
        self.assert_common_artifacts(out, cfg)
        straight = self.base / "straight"
        self.assertEqual(self.guarded_run(cfg, straight, self.fx.root)[0], 0)
        self.assertEqual(self.read(out, "predictions.json"), self.read(straight, "predictions.json"))

    def test_initial_state_paired_across_arms_and_distinct_across_seeds(self):
        states = {}
        for arm, seed in (("none", 11), ("basic", 11), ("basic_gamma_light", 11), ("none", 13)):
            out = self.base / f"{arm}-{seed}"
            code, counter = self.guarded_run(self.fx.final(arm, fold=0, seed=seed), out, self.fx.root,
                                             stop_after_epoch=1)
            self.assertEqual((code, counter["E_f_reads"]), (75, 0))
            states[arm, seed] = self.read(out, "environment.json")["initial_state_sha256"]
        self.assertEqual(states["none", 11], states["basic", 11])
        self.assertEqual(states["none", 11], states["basic_gamma_light", 11])
        self.assertNotEqual(states["none", 11], states["none", 13])

    def test_wrong_imagenet_cache_is_refused_before_training(self):
        from torchvision.models import ResNet50_Weights
        (self.hub / "checkpoints" / Path(ResNet50_Weights.IMAGENET1K_V2.url).name).write_bytes(b"other weights")
        out = self.base / "weights"
        with self.assertRaisesRegex(ValueError, "^recipe_mismatch: ImageNet cache"):
            self.guarded_run(self.fx.final(), out, self.fx.root)
        self.assertFalse((out / "checkpoint.pt").exists())


def gj_required():
    from augment_atlas.contracts import required_artifacts
    return required_artifacts({"evaluation": {}})


if __name__ == "__main__":
    unittest.main()
