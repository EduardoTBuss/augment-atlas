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

    def test_all_six_arm_examples_are_valid(self):
        for arm in ARMS:
            cfg = load_config(ROOT / f"configs/smoke/{arm}.json")
            self.assertEqual(cfg["augmentation"]["arm"], arm)

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
