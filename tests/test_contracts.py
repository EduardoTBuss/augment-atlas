import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from augment_atlas.contracts import ARMS, canonical_hash, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/smoke/synthetic.json")

    def test_all_ten_arm_examples_are_valid(self):
        self.assertEqual(set(ARMS), {"none", "basic", "basic_gamma_light", "basic_gamma_heavy",
                                     "basic_clahe_light", "basic_clahe_heavy", "gamma_light",
                                     "gamma_heavy", "clahe_light", "clahe_heavy"})
        for arm in ARMS:
            with self.subTest(arm=arm):
                cfg = load_config(ROOT / f"configs/smoke/{arm}.json")
                self.assertEqual(cfg["augmentation"]["arm"], arm)

    def test_isolated_examples_preserve_legacy_parameters_and_other_sections(self):
        for arm in ("gamma_light", "gamma_heavy", "clahe_light", "clahe_heavy"):
            with self.subTest(arm=arm):
                combined = load_config(ROOT / f"configs/smoke/basic_{arm}.json")
                isolated = load_config(ROOT / f"configs/smoke/{arm}.json")
                self.assertEqual(isolated["augmentation"]["ops"], combined["augmentation"]["ops"][1:])
                expected = copy.deepcopy(combined)
                expected["augmentation"] = copy.deepcopy(isolated["augmentation"])
                self.assertEqual(isolated, expected)
                self.assertEqual(isolated["phase"], "smoke")
                self.assertEqual(isolated["scientific_status"], "provisional")
                self.assertIn("isolated", isolated["augmentation"]["provenance"].lower())
                self.assertIn(combined["augmentation"]["provenance"], isolated["augmentation"]["provenance"])

    def test_isolated_arms_reject_basic(self):
        for arm in ("gamma_light", "gamma_heavy", "clahe_light", "clahe_heavy"):
            with self.subTest(arm=arm):
                cfg = load_config(ROOT / f"configs/smoke/basic_{arm}.json")
                cfg["augmentation"]["arm"] = arm
                with self.assertRaisesRegex(ValueError, "operations/order"):
                    validate_config(cfg)

    def test_combined_arms_require_basic_before_intensity(self):
        for arm in ("gamma_light", "gamma_heavy", "clahe_light", "clahe_heavy"):
            for invalid in ("missing_basic", "reversed_order"):
                with self.subTest(arm=arm, invalid=invalid):
                    cfg = load_config(ROOT / f"configs/smoke/basic_{arm}.json")
                    if invalid == "missing_basic":
                        cfg["augmentation"]["ops"] = cfg["augmentation"]["ops"][1:]
                    else:
                        cfg["augmentation"]["ops"].reverse()
                    with self.assertRaisesRegex(ValueError, "operations/order"):
                        validate_config(cfg)

    def test_final_and_other_models_rejected(self):
        for key, value in (("phase", "final"), ("scientific_status", "approved")):
            cfg = copy.deepcopy(self.cfg)
            cfg[key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
        self.cfg["model"]["name"] = "other"
        with self.assertRaises(ValueError):
            validate_config(self.cfg)

    def test_nonfinite_and_implicit_precision_rejected(self):
        self.cfg["training"]["lr"] = float("nan")
        with self.assertRaises(ValueError):
            validate_config(self.cfg)
        self.cfg["training"]["lr"] = 0.001
        self.cfg["training"]["autocast"] = True
        with self.assertRaises(ValueError):
            validate_config(self.cfg)

    def test_hpo_requires_explicit_provenance(self):
        self.cfg["recipe"]["hpo"] = {"status": "external", "protocol": "undecided", "origin": "example"}
        with self.assertRaises(ValueError):
            validate_config(self.cfg)

    def test_arm_and_operation_order_must_agree(self):
        cfg = load_config(ROOT / "configs/smoke/basic_gamma_light.json")
        cfg["augmentation"]["ops"].reverse()
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_hash_is_stable_and_sensitive(self):
        self.assertEqual(canonical_hash({"b": 2, "a": 1}), canonical_hash({"a": 1, "b": 2}))
        self.assertNotEqual(canonical_hash(self.cfg), canonical_hash({**self.cfg, "protocol_version": "changed"}))

    def test_preflight_in_fresh_process_without_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            code = "import runpy,sys; sys.argv=['train','--config',sys.argv[1],'--output',sys.argv[2],'--preflight']; runpy.run_module('augment_atlas.train',run_name='__main__')"
            completed = subprocess.run([sys.executable, "-c", code, str(ROOT / "configs/smoke/synthetic.json"), temporary], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertTrue(report["valid"])
            self.assertEqual(sum(report["counts"].values()), 12)
            self.assertTrue((Path(temporary) / "splits.json").exists())

    def test_modules_do_not_import_torch(self):
        code = "import augment_atlas.contracts,augment_atlas.data,augment_atlas.augment,augment_atlas.train,sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
