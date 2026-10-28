"""Engineering-smoke contract for INbreast-like (gray) and ISIC-like (RGB) inputs.

Pins the behavior exercised by the 2026-10-28 engineering smoke and records the
contract gaps found there. Synthetic fixtures only: no dataset identifier, path
or image is used. Open gaps are ``expectedFailure``; they document desired
behavior without changing ``src/``.

Optional stack gates follow the existing suite:
  ATLAS_CONTAINER_CPU_TESTS=1  requires torch/torchvision/numpy/PIL
  ATLAS_REQUIRE_OPENCV=1       requires cv2 for candidate CLAHE arms
"""
import copy
import csv
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas.contracts import ARMS, canonical_hash, file_hash, load_config, validate_config
from augment_atlas.data import audit_splits, load_records, prepare_splits

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs/smoke/candidate-v2"
REQUEST_STACK = os.environ.get("ATLAS_CONTAINER_CPU_TESTS") == "1"
REQUEST_OPENCV = os.environ.get("ATLAS_REQUIRE_OPENCV") == "1"
HAS_PIL_NUMPY = all(importlib.util.find_spec(n) is not None for n in ("numpy", "PIL"))
HAS_STACK = HAS_PIL_NUMPY and all(importlib.util.find_spec(n) is not None for n in ("torch", "torchvision"))
HAS_OPENCV = HAS_PIL_NUMPY and importlib.util.find_spec("cv2") is not None
HAS_SKIMAGE = HAS_PIL_NUMPY and importlib.util.find_spec("skimage") is not None
if REQUEST_STACK and not HAS_STACK:
    raise RuntimeError("Explicit container CPU gate requires the full training stack")
if REQUEST_OPENCV and not HAS_OPENCV:
    raise RuntimeError("Explicit OpenCV gate requires cv2")

GROUPS = 12
IMAGES_PER_GROUP = 2


def write_parent_manifest(root: Path, *, decodable: bool = False, color: str = "L") -> Path:
    """Synthetic manifest_csv with sha256 column: GROUPS groups x IMAGES_PER_GROUP images."""
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = root / "parent.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "group", "label", "sha256"])
        for g in range(GROUPS):
            for k in range(IMAGES_PER_GROUP):
                image = images / f"synthetic-{g:02d}-{k}.png"
                if decodable:
                    from PIL import Image
                    import numpy as np
                    rng = np.random.default_rng(1000 * g + k)
                    shape = (40, 52, 3) if color == "RGB" else (40, 52)
                    Image.fromarray(rng.integers(0, 256, size=shape, dtype=np.uint8)).save(image)
                else:
                    image.write_bytes(f"synthetic fixture {g} {k}".encode())
                writer.writerow([f"images/{image.name}", f"synthetic-group-{g:02d}", g % 2, file_hash(image)])
    return manifest


def write_subset(parent: Path, out: Path, groups: set[str]) -> None:
    """Group-closed subset: every image of each chosen group, none of the others."""
    with parent.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "group", "label", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(r for r in rows if r["group"] in groups)


def candidate_config(arm: str, root: Path, manifest: str, color: str) -> dict:
    cfg = load_config(CANDIDATE / f"{arm}.json")
    cfg["dataset"] = {"id": "synthetic-engineering", "version": "synthetic-v1", "adapter": "manifest_csv",
                      "root": str(root), "manifest": manifest, "require_image_hashes": True}
    cfg["input"]["color"] = color
    cfg["input"]["size"] = 32
    cfg["training"]["device"] = "cpu"
    return validate_config(cfg)


class SubsetByGroupTests(unittest.TestCase):
    """Subsetting exists only as a separate private manifest (no config option)."""

    def test_group_closed_subset_manifest_is_a_valid_population_with_its_own_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = write_parent_manifest(root)
            chosen = {f"synthetic-group-{g:02d}" for g in (0, 1, 3, 4, 6, 7, 9, 10)}
            write_subset(parent, root / "subset.csv", chosen)
            parent_cfg = candidate_config("none", root, "parent.csv", "rgb")
            subset_cfg = candidate_config("none", root, "subset.csv", "rgb")
            subset_cfg["dataset"]["version"] = "synthetic-v1-subset"
            parent_records = load_records(parent_cfg)
            subset_records = load_records(subset_cfg)
            # The subset is a row-for-row restriction of its parent.
            self.assertTrue(all(r in parent_records for r in subset_records))
            self.assertEqual({r["group"] for r in subset_records}, chosen)
            self.assertEqual(len(subset_records), len(chosen) * IMAGES_PER_GROUP)
            self.assertNotEqual(canonical_hash(sorted(parent_records, key=lambda r: r["id"])),
                                canonical_hash(sorted(subset_records, key=lambda r: r["id"])))
            document = prepare_splits(subset_cfg, subset_records, root / "attempt")
            audit_splits(subset_records, document["lists"])
            # group-hash-v1 with 8 groups: max(1, 8 // 5) = 1 selection and 1 test group.
            self.assertEqual({k: len(v) for k, v in document["groups"].items()},
                             {"train": 6, "selection": 1, "test": 1})

    def test_subset_with_fewer_than_six_groups_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = write_parent_manifest(root)
            write_subset(parent, root / "subset.csv", {f"synthetic-group-{g:02d}" for g in range(5)})
            cfg = candidate_config("none", root, "subset.csv", "rgb")
            with self.assertRaisesRegex(ValueError, "at least six independent groups"):
                prepare_splits(cfg, load_records(cfg), root / "attempt")

    def test_subset_intent_key_is_refused_before_any_record_is_loaded(self):
        """Was the hazard pin: ``max_images`` validated and the full manifest was loaded."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_parent_manifest(root)
            cfg = candidate_config("none", root, "parent.csv", "rgb")
            cfg["dataset"]["max_images"] = 4
            with self.assertRaisesRegex(ValueError, r"unknown dataset key\(s\) \['max_images'\]"):
                validate_config(cfg)

    def test_gap_unknown_dataset_key_must_not_be_silently_ignored(self):
        """Closed gap (contracts.py dataset section): unknown dataset keys are refused.

        An operator who writes a subset-like key (``max_images``) believed the run
        was reduced, yet the full manifest was trained on. Subsetting requires a
        private group-closed manifest instead.
        """
        cfg = load_config(CANDIDATE / "none.json")
        cfg["dataset"]["max_images"] = 4
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_unknown_dataset_keys_are_refused_for_every_adapter(self):
        for adapter, extra in (("synthetic", {}), ("manifest_csv", {"manifest": "m.csv"}),
                               ("inbreast_csv", {"manifest": "m.csv"})):
            for key in ("max_images", "subset", "Manifest", "image_root"):
                with self.subTest(adapter=adapter, key=key):
                    cfg = load_config(CANDIDATE / "none.json")
                    cfg["dataset"] = {"id": "synthetic-engineering", "version": "v1", "adapter": adapter,
                                      **extra, key: 1}
                    with self.assertRaisesRegex(ValueError, "unknown dataset key"):
                        validate_config(cfg)

    def test_every_versioned_smoke_config_still_validates(self):
        """The closed key set must cover every dataset key used by configs/**."""
        import json
        checked = 0
        for path in sorted((ROOT / "configs").rglob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(document, dict) and document.get("phase") == "smoke" and "dataset" in document:
                with self.subTest(config=str(path.relative_to(ROOT))):
                    validate_config(document)
                    checked += 1
        self.assertGreater(checked, 30)


class EvaluationPopulationTests(unittest.TestCase):
    """The smoke runner needs every manifest image in train/selection/test."""

    def setUp(self):
        self.records = [{"id": f"img-{i}", "group": f"group-{i // 2}", "label": i % 2} for i in range(16)]
        self.lists = {"train": [f"img-{i}" for i in range(0, 8)],
                      "selection": ["img-8", "img-9", "img-10", "img-11"],
                      "test": ["img-12", "img-13"]}  # group-7 (img-14, img-15) left unassigned

    def test_unassigned_group_is_rejected_today(self):
        with self.assertRaisesRegex(ValueError, "account for every manifest image"):
            audit_splits(self.records, self.lists)

    def test_unknown_partition_name_is_rejected_today(self):
        lists = dict(self.lists, unassigned=["img-14", "img-15"])
        with self.assertRaisesRegex(ValueError, "must contain train, selection, test"):
            audit_splits(self.records, lists)

    @unittest.expectedFailure
    def test_gap_declared_unassigned_development_groups(self):
        """GAP data.py:98-99 and data.py:113-114 (documented in docs/datasets.md).

        A manifest that also lists development patients cannot be evaluated on a
        sub-population: every image must be in train/selection/test. Desired: an
        explicitly declared, group-disjoint partition of images that are neither
        trained nor evaluated. ``unassigned`` is a proposed name, not an API.
        Today the workaround is a separate manifest per evaluation population
        (the global D/E pools path covers this outside the smoke runner).
        """
        lists = dict(self.lists, unassigned=["img-14", "img-15"])
        audit_splits(self.records, lists)

    def test_declared_unassigned_must_still_be_group_disjoint(self):
        """Even when the gap above is closed, a leaked group must stay refused."""
        lists = {"train": [f"img-{i}" for i in range(0, 8)],
                 "selection": ["img-8", "img-9", "img-10", "img-11"],
                 "test": ["img-12", "img-13", "img-14"],
                 "unassigned": ["img-15"]}
        with self.assertRaises(ValueError):
            audit_splits(self.records, lists)


class RgbClaheContractTests(unittest.TestCase):
    def test_legacy_clahe_rgb_refused_by_contract_candidate_accepted(self):
        for arm in ("clahe_light", "clahe_heavy", "basic_clahe_light", "basic_clahe_heavy"):
            with self.subTest(arm=arm):
                legacy = load_config(ROOT / f"configs/smoke/{arm}.json")
                legacy["input"]["color"] = "rgb"
                with self.assertRaisesRegex(ValueError, "grayscale-only"):
                    validate_config(legacy)
                candidate = load_config(CANDIDATE / f"{arm}.json")
                candidate["input"]["color"] = "rgb"
                validate_config(candidate)

    @unittest.skipUnless(HAS_SKIMAGE, "legacy CLAHE runtime needs numpy/PIL/scikit-image")
    def test_legacy_clahe_runtime_refuses_rgb_even_if_contract_bypassed(self):
        from PIL import Image
        from augment_atlas.augment import apply_augmentation
        op = dict(load_config(ROOT / "configs/smoke/clahe_light.json")["augmentation"]["ops"][0], p=1.0)
        gray = apply_augmentation(Image.new("L", (16, 16), 100), [op], random.Random(0))
        self.assertEqual(gray.mode, "L")
        with self.assertRaisesRegex(ValueError, "CLAHE v1 expects an explicitly grayscale input"):
            apply_augmentation(Image.new("RGB", (16, 16), (100, 100, 100)), [op], random.Random(0))

    @unittest.skipUnless(HAS_OPENCV, "candidate CLAHE needs cv2; executed in candidate image")
    def test_candidate_rgb_clahe_keeps_neutral_gray_neutral(self):
        """Invariant: Lab-luminance CLAHE must not add a color cast to R=G=B input."""
        import numpy as np
        from PIL import Image
        from augment_atlas.augment import apply_candidate_augmentation
        from augment_atlas.contracts import CANDIDATE_BACKEND
        rng = np.random.default_rng(20261028)
        gray = np.clip(np.add.outer(np.arange(64) * 2, np.arange(64)) // 3
                       + rng.integers(-8, 9, size=(64, 64)), 0, 255).astype(np.uint8)
        rgb = Image.fromarray(np.stack([gray] * 3, axis=-1))
        for clip in (2.0, 4.0):
            with self.subTest(clip=clip):
                op = {"op": "clahe", "p": 1, "clip_limit": clip, "tile_grid_size": [8, 8]}
                out = np.asarray(apply_candidate_augmentation(
                    rgb, {"backend": CANDIDATE_BACKEND, "ops": [op]}, [3, 0, "synthetic"]), dtype=np.int16)
                self.assertEqual(out.shape, (64, 64, 3))
                self.assertFalse(np.array_equal(out[..., 0], gray), "CLAHE must actually change luminance")
                self.assertLessEqual(int(np.abs(out[..., 0] - out[..., 1]).max()), 1)
                self.assertLessEqual(int(np.abs(out[..., 1] - out[..., 2]).max()), 1)


@unittest.skipUnless(HAS_STACK, "CPU dataset pipeline needs torch/torchvision/numpy/PIL")
class ArmPipelineTests(unittest.TestCase):
    """All ten candidate arms through ClassificationDataset on real decodable files."""

    def _check_color(self, color: str):
        import torch
        from augment_atlas.train import ClassificationDataset
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_parent_manifest(root, decodable=True, color="RGB" if color == "rgb" else "L")
            evals, trained = {}, {}
            with patch.dict(os.environ, {"ATLAS_DATA_ROOT": str(root)}):
                for arm in ARMS:
                    cfg = candidate_config(arm, root, "parent.csv", color)
                    if "clahe" in arm and not HAS_OPENCV:
                        continue
                    records = load_records(cfg)
                    split = prepare_splits(cfg, records, root / f"attempt-{arm}")
                    lookup = {r["id"]: r for r in records}
                    rows = [lookup[i] for i in split["lists"]["train"]]
                    a = [ClassificationDataset(cfg, rows, epoch=0, training=True)[i][0] for i in range(len(rows))]
                    b = [ClassificationDataset(cfg, rows, epoch=0, training=True)[i][0] for i in range(len(rows))]
                    e = [ClassificationDataset(cfg, rows, epoch=0, training=False)[i][0] for i in range(len(rows))]
                    for x, y in zip(a, b):
                        self.assertEqual(tuple(x.shape), (3, 32, 32))
                        self.assertEqual(x.dtype, torch.float32)
                        self.assertTrue(bool(torch.isfinite(x).all()))
                        self.assertTrue(torch.equal(x, y), f"{arm}: augmentation must be deterministic")
                    if color == "grayscale_to_rgb":
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        for x in a:
                            raw = x * std + mean
                            torch.testing.assert_close(raw[0], raw[1], atol=1e-5, rtol=0)
                            torch.testing.assert_close(raw[1], raw[2], atol=1e-5, rtol=0)
                    evals[arm], trained[arm] = e, a
                    if arm == "none":
                        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, e)))
            self.assertIn("none", evals)
            for arm, e in evals.items():
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(e, evals["none"])),
                                f"{arm}: evaluation pixels must not depend on the arm")
            # basic flips with p=0.5 and photometrics with p=0.8 over 16 train images (8 groups)
            self.assertTrue(any(not torch.equal(x, y) for x, y in zip(trained["basic"], evals["basic"])))

    def test_all_arms_grayscale_to_rgb(self):
        self._check_color("grayscale_to_rgb")

    def test_all_arms_rgb(self):
        self._check_color("rgb")


PROVENANCE_VARS = ("ATLAS_CODE_COMMIT", "ATLAS_IMAGE_ID")


def without_provenance(**kept):
    """Environment patch that removes both identities, then sets ``kept``."""
    patcher = patch.dict(os.environ, kept)
    patcher.start()
    for name in PROVENANCE_VARS:
        if name not in kept:
            os.environ.pop(name, None)
    return patcher


class ProvenanceContractTests(unittest.TestCase):
    """Closed gap (train.py environment.json): no silent ``unavailable`` commit/image."""

    def provenance(self, required, **kept):
        import contextlib
        import io
        from augment_atlas import train
        patcher = without_provenance(**kept)
        self.addCleanup(patcher.stop)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = train._provenance(required=required)
        patcher.stop()
        return result, stderr.getvalue()

    def test_smoke_without_identity_warns_and_marks_incomplete(self):
        for kept in ({}, {"ATLAS_CODE_COMMIT": "a" * 40}, {"ATLAS_IMAGE_ID": "sha256:x"},
                     {"ATLAS_CODE_COMMIT": "", "ATLAS_IMAGE_ID": "unavailable"}):
            with self.subTest(kept=kept):
                (values, complete), stderr = self.provenance(False, **kept)
                self.assertIs(complete, False)
                self.assertIn("WARNING", stderr)
                self.assertIn("provenance_complete=false", stderr)
                for key, name in (("code_commit", "ATLAS_CODE_COMMIT"), ("image_id", "ATLAS_IMAGE_ID")):
                    self.assertEqual(values[key], kept.get(name) or "unavailable")
                    if kept.get(name) in (None, "", "unavailable"):
                        self.assertIn(name, stderr)

    def test_complete_identity_is_recorded_without_warning(self):
        kept = {"ATLAS_CODE_COMMIT": "a" * 40, "ATLAS_IMAGE_ID": "sha256:x"}
        for required in (False, True):
            with self.subTest(required=required):
                (values, complete), stderr = self.provenance(required, **kept)
                self.assertEqual((values, complete, stderr),
                                 ({"code_commit": "a" * 40, "image_id": "sha256:x"}, True, ""))

    def test_required_identity_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^provenance_missing: ATLAS_CODE_COMMIT, ATLAS_IMAGE_ID"):
            self.provenance(True)


@unittest.skipUnless(REQUEST_STACK and HAS_STACK, "real toy torch loop is container-only (ATLAS_CONTAINER_CPU_TESTS=1)")
class SmokeProvenanceTrainingTests(unittest.TestCase):
    def run_smoke(self, out, **kept):
        import contextlib
        import io
        import json
        import torch
        from augment_atlas import train

        class TinyModel(torch.nn.Module):
            def __init__(model_self):
                super().__init__()
                model_self.fc = torch.nn.Linear(3, 2)

            def forward(model_self, pixels):
                return model_self.fc(pixels.mean(dim=(2, 3)))

        cfg = load_config(ROOT / "configs/smoke/synthetic.json")
        cfg["training"].update(device="cpu", workers=0, epochs=1)
        cfg["input"]["size"] = 32
        patcher = without_provenance(**kept)
        self.addCleanup(patcher.stop)
        stderr = io.StringIO()
        with patch("torchvision.models.resnet50", side_effect=lambda **_: TinyModel()), \
             patch.object(torch.cuda, "is_available", return_value=False), contextlib.redirect_stderr(stderr):
            code = train.run(validate_config(cfg), out)
        patcher.stop()
        return code, json.loads((out / "environment.json").read_text(encoding="utf-8")), stderr.getvalue()

    def test_smoke_run_records_incomplete_provenance_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, environment, stderr = self.run_smoke(Path(temporary) / "missing")
            self.assertEqual(code, 0)
            self.assertEqual((environment["code_commit"], environment["image_id"], environment["provenance_complete"]),
                             ("unavailable", "unavailable", False))
            self.assertIn("ATLAS_CODE_COMMIT, ATLAS_IMAGE_ID", stderr)
            code, environment, stderr = self.run_smoke(Path(temporary) / "present", ATLAS_CODE_COMMIT="a" * 40,
                                                       ATLAS_IMAGE_ID="sha256:x")
            self.assertEqual(code, 0)
            self.assertEqual((environment["code_commit"], environment["image_id"], environment["provenance_complete"]),
                             ("a" * 40, "sha256:x", True))
            self.assertNotIn("provenance", stderr)


if __name__ == "__main__":
    unittest.main()
