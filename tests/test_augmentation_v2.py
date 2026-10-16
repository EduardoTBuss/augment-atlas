"""Candidate contracts run on stdlib; numeric tests use existing optional libs.

Remote CPU gate: ATLAS_REQUIRE_OPENCV=1 python -m unittest discover -s tests
                -p test_augmentation_v2.py -v
"""
import copy
import importlib.util
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from augment_atlas.augment import apply_candidate_augmentation, candidate_operation_draws
from augment_atlas.contracts import ARMS, CANDIDATE_BACKEND, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs/smoke/candidate-v2"
HAS_NUMERIC = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL"))
HAS_OPENCV = HAS_NUMERIC and importlib.util.find_spec("cv2") is not None


def policy(*operations):
    return {"backend": CANDIDATE_BACKEND, "ops": list(operations)}


class CandidateContractTests(unittest.TestCase):
    def test_all_ten_candidate_and_legacy_configs_validate(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                candidate = load_config(CANDIDATE / f"{arm}.json")
                legacy = load_config(ROOT / f"configs/smoke/{arm}.json")
                self.assertEqual(candidate["augmentation"]["arm"], arm)
                self.assertEqual(candidate["augmentation"]["backend"], CANDIDATE_BACKEND)
                self.assertEqual(candidate["protocol_version"], "candidate-v2")
                self.assertEqual(candidate["scientific_status"], "provisional")
                self.assertNotIn("backend", legacy["augmentation"])
                self.assertEqual(candidate["input"]["size"], 224)
                self.assertEqual(legacy["input"]["size"], 64)
                self.assertEqual(candidate["training"]["epochs"], 2)
                self.assertEqual(candidate["recipe"]["hpo"]["protocol"], "within-outer-fold")
                self.assertEqual(candidate["recipe"]["hpo"]["status"], "pending")

    def test_backend_protocol_mismatch_rejected(self):
        for change in ("missing_backend", "unknown_backend", "legacy_protocol"):
            with self.subTest(change=change):
                cfg = load_config(CANDIDATE / "none.json")
                if change == "missing_backend":
                    del cfg["augmentation"]["backend"]
                elif change == "unknown_backend":
                    cfg["augmentation"]["backend"] = "unknown"
                else:
                    cfg["protocol_version"] = "technical-smoke-v1"
                with self.assertRaises(ValueError):
                    validate_config(cfg)

    def test_candidate_pretrained_hash_is_required(self):
        for invalid in (None, "", "a" * 63, "x" * 64, "A" * 64):
            with self.subTest(hash=invalid):
                cfg = load_config(CANDIDATE / "none.json")
                cfg["model"]["weights_sha256"] = invalid
                with self.assertRaisesRegex(ValueError, "weights_sha256"):
                    validate_config(cfg)

    def test_candidate_clahe_rejects_legacy_domain_and_grid(self):
        for update in ({"clip_limit": 0.01}, {"tile_grid_size": [4, 4]}, {"tile_grid_size": [8.0, 8]}, {"window_fraction": 0.1}, {"p": 0.5}):
            with self.subTest(update=update):
                cfg = load_config(CANDIDATE / "clahe_light.json")
                cfg["augmentation"]["ops"][0].update(update)
                with self.assertRaisesRegex(ValueError, "candidate CLAHE"):
                    validate_config(cfg)
        legacy = load_config(ROOT / "configs/smoke/clahe_light.json")
        legacy["augmentation"]["ops"][0]["tile_grid_size"] = [8, 8]
        with self.assertRaisesRegex(ValueError, "legacy CLAHE"):
            validate_config(legacy)

    def test_candidate_gamma_keeps_versioned_ranges(self):
        cfg = load_config(CANDIDATE / "gamma_light.json")
        cfg["augmentation"]["ops"][0]["range"] = [0.9, 1.1]
        with self.assertRaisesRegex(ValueError, "versioned provisional"):
            validate_config(cfg)

    def test_candidate_rgb_clahe_supported_legacy_stays_grayscale(self):
        cfg = load_config(CANDIDATE / "clahe_light.json")
        cfg["input"]["color"] = "rgb"
        validate_config(cfg)
        cfg = load_config(ROOT / "configs/smoke/clahe_light.json")
        cfg["input"]["color"] = "rgb"
        with self.assertRaisesRegex(ValueError, "grayscale-only"):
            validate_config(cfg)

    def test_new_outer_split_requires_persisted_file(self):
        for split_file in (None, "", "   "):
            cfg = load_config(CANDIDATE / "none.json")
            cfg["splits"]["version"] = "stratified-group-outer-v1"
            cfg["dataset"]["split_file"] = split_file
            with self.subTest(split_file=split_file), self.assertRaisesRegex(ValueError, "persisted split_file"):
                validate_config(cfg)
        cfg["dataset"]["split_file"] = "private/splits.json"
        validate_config(cfg)

    def test_basic_candidate_parameters_remain_legacy(self):
        for arm in ("basic", "basic_gamma_light", "basic_gamma_heavy", "basic_clahe_light", "basic_clahe_heavy"):
            with self.subTest(arm=arm):
                new = load_config(CANDIDATE / f"{arm}.json")["augmentation"]["ops"][0]
                old = load_config(ROOT / f"configs/smoke/{arm}.json")["augmentation"]["ops"][0]
                self.assertEqual(new, old)

    def test_named_intensity_draws_match_isolated_and_combined(self):
        for arm in ("gamma_light", "gamma_heavy", "clahe_light", "clahe_heavy"):
            isolated = load_config(CANDIDATE / f"{arm}.json")["augmentation"]["ops"][0]
            combined = load_config(CANDIDATE / f"basic_{arm}.json")["augmentation"]["ops"]
            for index in range(50):
                identity = [3, 2, f"synthetic-{index}"]
                candidate_operation_draws(identity, combined[0])
                self.assertEqual(candidate_operation_draws(identity, isolated), candidate_operation_draws(identity, combined[1]))

    def test_probability_draws_reproducible_across_runs_and_seed_sensitive(self):
        op = {"op": "gamma", "p": 0.3, "range": [0.8, 1.2]}
        values = [candidate_operation_draws([3, 1, index], op) for index in range(100)]
        self.assertEqual(values, [candidate_operation_draws([3, 1, index], op) for index in range(100)])
        self.assertTrue(any(value["applied"] for value in values))
        self.assertTrue(any(not value["applied"] for value in values))
        self.assertTrue(all(0.8 <= value["gamma"] <= 1.2 for value in values))
        self.assertNotEqual(values, [candidate_operation_draws([4, 1, index], op) for index in range(100)])

    def test_no_global_random_state_changes(self):
        state = random.getstate()
        candidate_operation_draws([1, 2, 3], {"op": "gamma", "p": 0.3, "range": [0.8, 1.2]})
        self.assertEqual(random.getstate(), state)

    def test_opencv_integer_clip_quantization_requires_larger_smoke_input(self):
        # OpenCV 4.12 clahe.cpp: CV_8UC1 has 256 bins; each positive clip
        # becomes max(int(clip * tile_area / 256), 1). Both sizes divide by 8.
        for size, expected in ((64, [1, 1]), (224, [6, 12])):
            with self.subTest(size=size):
                tile_area = (size // 8) ** 2
                self.assertEqual([max(int(clip * tile_area / 256), 1) for clip in (2, 4)], expected)

    def test_remote_opencv_dependency_gate(self):
        if os.environ.get("ATLAS_REQUIRE_OPENCV") == "1":
            self.assertTrue(HAS_OPENCV, "CPU container gate requires NumPy, Pillow and cv2; skips are not acceptance")
            from importlib.metadata import version
            self.assertEqual(version("opencv-python-headless"), "4.12.0.88")


@unittest.skipUnless(HAS_NUMERIC, "NumPy and Pillow are optional on the notebook")
class CandidateNumericTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image
        self.np, self.Image = np, Image

    def test_gamma_fixed_domain_exact_uint8_values(self):
        image = self.Image.fromarray(self.np.array([[0, 64, 128, 192, 255]], dtype=self.np.uint8))
        result = apply_candidate_augmentation(image, policy({"op": "gamma", "p": 1, "range": [2, 2]}), [3, 1, "a"])
        self.np.testing.assert_array_equal(self.np.asarray(result), [[0, 16, 64, 145, 255]])
        self.assertEqual(result.size, image.size)

    def test_gamma_constant_rgb_channels_are_transformed_not_renormalized(self):
        image = self.Image.new("RGB", (3, 2), (0, 128, 255))
        result = apply_candidate_augmentation(image, policy({"op": "gamma", "p": 1, "range": [2, 2]}), [3, 1, "a"])
        self.assertEqual(result.getpixel((1, 1)), (0, 64, 255))
        self.assertEqual(result.mode, "RGB")

    def test_zero_probability_skips_gamma_and_cv2_import(self):
        image = self.Image.new("RGB", (4, 3), (17, 53, 129))
        with patch.dict("sys.modules", {"cv2": None}):
            result = apply_candidate_augmentation(image, policy({"op": "gamma", "p": 0, "range": [2, 2]},
                                                  {"op": "clahe", "p": 0, "clip_limit": 2, "tile_grid_size": [8, 8]}), [3, 1, "a"])
        self.assertEqual(result.tobytes(), image.tobytes())

    def test_basic_then_gamma_matches_two_calls_with_same_identity(self):
        image = self.Image.fromarray(self.np.arange(64 * 64, dtype=self.np.uint8).reshape(64, 64))
        augmentation = copy.deepcopy(load_config(CANDIDATE / "basic_gamma_light.json")["augmentation"])
        augmentation["ops"][1]["p"] = 1
        identity = [3, 2, "paired-example"]
        combined = apply_candidate_augmentation(image, augmentation, identity)
        basic = apply_candidate_augmentation(image, policy(augmentation["ops"][0]), identity)
        separate = apply_candidate_augmentation(basic, policy(augmentation["ops"][1]), identity)
        self.assertEqual(combined.tobytes(), separate.tobytes())


@unittest.skipUnless(HAS_OPENCV, "OpenCV not installed locally; run CPU container gate")
class CandidateOpenCVTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        from PIL import Image
        self.cv2, self.np, self.Image = cv2, np, Image

    def test_clip_2_and_4_collapse_at_64_but_differ_at_224_on_constant_fixture(self):
        # Force application to isolate CLAHE's numerical semantics from p=.3.
        # Distinction on this fixture is not a universal monotonic-strength claim.
        for size in (64, 224):
            with self.subTest(size=size):
                image = self.Image.new("L", (size, size), 128)
                outputs = [self.np.asarray(apply_candidate_augmentation(image,
                    policy({"op": "clahe", "p": 1, "clip_limit": clip, "tile_grid_size": [8, 8]}),
                    [3, 2, "clip-quantization-fixture"])) for clip in (2, 4)]
                if size == 64:
                    self.np.testing.assert_array_equal(outputs[0], outputs[1])
                else:
                    self.assertGreater(self.np.count_nonzero(outputs[0] != outputs[1]), 0)

    def test_gray_clahe_matches_opencv_fixed_grid_and_preserves_shape(self):
        pixels = (self.np.arange(64 * 64, dtype=self.np.uint16).reshape(64, 64) % 191 + 20).astype(self.np.uint8)
        for clip in (2, 4):
            with self.subTest(clip=clip):
                expected = self.cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(pixels)
                result = apply_candidate_augmentation(self.Image.fromarray(pixels),
                    policy({"op": "clahe", "p": 1, "clip_limit": clip, "tile_grid_size": [8, 8]}), [3, 2, "gray"])
                self.np.testing.assert_array_equal(self.np.asarray(result), expected)
                self.assertEqual(result.mode, "L")

    def test_rgb_clahe_uses_lab_luminance_and_rgb_channel_order(self):
        y, x = self.np.mgrid[:64, :64]
        pixels = self.np.stack([(x * 4) % 256, (y * 3 + 20) % 256, (x + y + 70) % 256], axis=-1).astype(self.np.uint8)
        lab = self.cv2.cvtColor(pixels, self.cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        expected = self.cv2.cvtColor(lab, self.cv2.COLOR_LAB2RGB)
        result = apply_candidate_augmentation(self.Image.fromarray(pixels),
            policy({"op": "clahe", "p": 1, "clip_limit": 2, "tile_grid_size": [8, 8]}), [3, 2, "rgb"])
        self.np.testing.assert_array_equal(self.np.asarray(result), expected)
        self.assertEqual(result.mode, "RGB")

    def test_basic_then_clahe_matches_two_calls_with_same_identity(self):
        image = self.Image.fromarray(self.np.arange(64 * 64, dtype=self.np.uint8).reshape(64, 64))
        augmentation = copy.deepcopy(load_config(CANDIDATE / "basic_clahe_light.json")["augmentation"])
        augmentation["ops"][1]["p"] = 1
        identity = [3, 2, "paired-example"]
        combined = apply_candidate_augmentation(image, augmentation, identity)
        basic = apply_candidate_augmentation(image, policy(augmentation["ops"][0]), identity)
        separate = apply_candidate_augmentation(basic, policy(augmentation["ops"][1]), identity)
        self.assertEqual(combined.tobytes(), separate.tobytes())


if __name__ == "__main__":
    unittest.main()
