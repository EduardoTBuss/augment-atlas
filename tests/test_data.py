import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas.contracts import canonical_hash, file_hash, load_config
from augment_atlas.data import audit_splits, confined_path, load_records, prepare_splits

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/smoke/synthetic.json")
        self.records = load_records(self.cfg)

    def test_partitions_persist_and_training_seed_does_not_change_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = prepare_splits(self.cfg, self.records, Path(temporary) / "a")
            changed = copy.deepcopy(self.cfg)
            changed["training"]["seed"] += 1
            second = prepare_splits(changed, self.records, Path(temporary) / "b")
            self.assertEqual(first, second)
            self.assertEqual(prepare_splits(self.cfg, list(reversed(self.records)), Path(temporary) / "a"), first)
            audit_splits(self.records, first["lists"])

    def test_tampered_split_and_changed_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepare_splits(self.cfg, self.records, temporary)
            target = Path(temporary) / "splits.json"
            document = json.loads(target.read_text())
            document["lists"]["train"].append(document["lists"]["test"][0])
            target.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                prepare_splits(self.cfg, self.records, temporary)

    def test_group_leakage_rejected_even_with_distinct_images(self):
        records = [{"id": "a", "group": "shared"}, {"id": "b", "group": "shared"}, {"id": "c", "group": "other"}]
        with self.assertRaisesRegex(ValueError, "group leakage"):
            audit_splits(records, {"train": ["a"], "selection": ["b"], "test": ["c"]})

    def test_path_traversal_rejected(self):
        for path in ("../secret.png", "/absolute.png", "C:/private.png", "folder/../../secret.png", "..\\secret.png"):
            with self.assertRaises(ValueError):
                confined_path(Path("/data"), path)

    def _private_manifest(self, root, adapter="manifest_csv", extra_hash=False):
        names = ("processed_image_filename", "group_key", "classification") if adapter == "inbreast_csv" else ("path", "group", "label")
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([*names, "sha256"] if extra_hash else names)
            for index in range(6):
                image = root / f"sample-{index}.png"
                image.write_bytes(b"synthetic fixture; not decoded by metadata preflight")
                row = [image.name, f"group-{index}", index % 2]
                writer.writerow([*row, file_hash(image)] if extra_hash else row)
        cfg = copy.deepcopy(self.cfg)
        cfg["dataset"] = {"id": "private-example", "version": "v1", "adapter": adapter, "root": str(root), "manifest": "manifest.csv"}
        return cfg

    def test_real_adapter_schema_and_environment_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self._private_manifest(root, "inbreast_csv")
            cfg["dataset"]["root"] = "/invalid-override"
            with patch.dict("os.environ", {"ATLAS_DATA_ROOT": str(root)}):
                records = load_records(cfg)
            self.assertEqual(len(records), 6)
            self.assertEqual({r["label"] for r in records}, {0, 1})

    def test_private_image_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self._private_manifest(root, extra_hash=True)
            self.assertEqual(len(load_records(cfg)), 6)
            (root / "sample-1.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "image checksum"):
                load_records(cfg)

    def test_manifest_missing_image_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self._private_manifest(root)
            (root / "sample-1.png").unlink()
            with self.assertRaisesRegex(ValueError, "missing image"):
                load_records(cfg)
            with (root / "manifest.csv").open("a") as stream:
                stream.write("sample-0.png,group-0,0\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_records(cfg, check_files=False)

    def test_dot_alias_cannot_relabel_same_file_as_another_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self._private_manifest(root)
            with (root / "manifest.csv").open("a") as stream:
                stream.write("./sample-0.png,different-group,1\n")
            for check_files in (False, True):
                with self.subTest(check_files=check_files), self.assertRaisesRegex(ValueError, "unique physical"):
                    load_records(cfg, check_files=check_files)

    def test_stored_identity_uses_normalized_resolved_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self._private_manifest(root)
            manifest = root / "manifest.csv"
            manifest.write_text(manifest.read_text().replace("sample-0.png", "./sample-0.png"))
            records = load_records(cfg)
            self.assertEqual(records[0]["id"], "sample-0.png")
            self.assertEqual(records[0]["path"], "sample-0.png")

    def test_symlink_alias_cannot_cross_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self._private_manifest(root)
            alias = root / "symlink-alias.png"
            try:
                alias.symlink_to(root / "sample-0.png")
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable: {type(exc).__name__}")
            with (root / "manifest.csv").open("a") as stream:
                stream.write("symlink-alias.png,different-group,1\n")
            with self.assertRaisesRegex(ValueError, "unique physical"):
                load_records(cfg)

    def test_hardlink_alias_cannot_cross_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = self._private_manifest(root)
            alias = root / "hardlink-alias.png"
            try:
                os.link(root / "sample-0.png", alias)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"hardlinks unavailable: {type(exc).__name__}")
            if not alias.stat().st_ino:
                self.skipTest("filesystem does not expose file identity")
            with (root / "manifest.csv").open("a") as stream:
                stream.write("hardlink-alias.png,different-group,1\n")
            with self.assertRaisesRegex(ValueError, "unique physical"):
                load_records(cfg)


if __name__ == "__main__":
    unittest.main()
