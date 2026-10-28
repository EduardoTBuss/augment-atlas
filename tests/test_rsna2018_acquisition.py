"""Synthetic RSNA fixtures only; no server data, network, ML dependencies or pydicom."""
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from contextlib import closing
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

from PIL import Image, ImageDraw
from augment_atlas.data import load_records
from augment_atlas.acquisition import rsna2018 as rsna
from augment_atlas.acquisition import rsna_dicom
from augment_atlas.acquisition.errors import DivergenceError, Incomplete, IntegrityError
from augment_atlas.contracts import canonical_hash, file_hash

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/acquisition/rsna2018-v1.json"


def element(group, tag, vr, value):
    if isinstance(value, str):
        value = value.encode("ascii")
    if len(value) % 2:
        value += b"\0" if vr == b"UI" else b" "
    header = struct.pack("<HH", group, tag) + vr
    if vr in rsna_dicom.LONG_VR:
        return header + b"\0\0" + struct.pack("<I", len(value)) + value
    return header + struct.pack("<H", len(value)) + value


def dicom(value, sop, study, view="PA", *, fragments=False, sequence=False, jpeg_note=""):
    image = Image.new("L", (1024, 1024), value)
    draw = ImageDraw.Draw(image)
    for x in range(0, 1024, 7):
        draw.line((x, 0, x, 1023), fill=(value + x) % 256)
    for y in range(0, 1024, 11):
        draw.line((0, y, 1023, y), fill=(value + y) % 256)
    sink = io.BytesIO()
    exif = Image.Exif()
    exif[0x010E] = jpeg_note
    image.save(sink, "JPEG", quality=90, exif=exif)
    jpeg = sink.getvalue()
    if len(jpeg) % 2:
        jpeg += b"\0"
    parts = [jpeg[:100], jpeg[100:]] if fragments else [jpeg]
    body = b"\0" * 128 + b"DICM"
    body += element(2, 0x10, b"UI", rsna_dicom.JPEG_BASELINE)
    body += element(8, 0x16, b"UI", "1.2.840.10008.5.1.4.1.1.1")
    body += element(8, 0x18, b"UI", sop) + element(8, 0x60, b"CS", "CR")
    body += element(0x18, 0x5101, b"CS", view) + element(0x20, 0x0D, b"UI", study)
    for tag, number in ((2, 1), (0x10, 1024), (0x11, 1024),
                        (0x100, 8), (0x101, 8), (0x102, 7), (0x103, 0)):
        body += element(0x28, tag, b"US", struct.pack("<H", number))
    body += element(0x28, 4, b"CS", "MONOCHROME2")
    if sequence:
        body += struct.pack("<HH2s2sI", 8, 0x1110, b"SQ", b"\0\0", 0xFFFFFFFF)
        body += struct.pack("<HHI", 0xFFFE, 0xE000, 0xFFFFFFFF)
        body += element(8, 0x1150, b"UI", "1.2.3")
        body += struct.pack("<HHI", 0xFFFE, 0xE00D, 0)
        body += struct.pack("<HHI", 0xFFFE, 0xE0DD, 0)
    body += struct.pack("<HH2s2sI", 0x7FE0, 0x10, b"OB", b"\0\0", 0xFFFFFFFF)
    body += struct.pack("<HHI", 0xFFFE, 0xE000, 0)  # empty basic offset table
    for part in parts:
        body += struct.pack("<HHI", 0xFFFE, 0xE000, len(part)) + part
    return body + struct.pack("<HHI", 0xFFFE, 0xE0DD, 0)


class SyntheticRelease:
    def __init__(self, base, *, duplicate=False, all_labeled=False):
        self.raw, self.out = base / "raw", base / "out"
        self.raw.mkdir()
        self.config = base / "config.json"
        self.cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.mapping = []
        self.dicoms = []
        for index in range(3):
            sop, study = f"1.2.826.0.1.3680043.10.999.{index + 1}", f"1.2.826.0.1.3680043.10.998.{index + 1}"
            patient = 90000001 if index < 2 else 90000002
            self.mapping.append({"SOPInstanceUID": sop, "StudyInstanceUID": study,
                                 "img_id": f"{patient:08d}_{index:03d}.png",
                                 "subset_img_id": str(uuid.UUID(int=index + 1)),
                                 "subset_group": "synthetic", "subset_init_label": "unused", "orig_labels": []})
            value = 30 if duplicate and index == 1 else (30, 120, 220)[index]
            self.dicoms.append(dicom(value, sop, study, "AP" if index == 1 else "PA",
                                     fragments=index == 0, sequence=index == 0, jpeg_note=str(index)))
        labels = [{"id": "positive", "name": "Lung Opacity"},
                  {"id": "abnormal", "name": "No Lung Opacity / Not Normal"},
                  {"id": "normal", "name": "Normal"}]
        self.annotations = {"labelGroups": [{"name": "Calculated", "labels": labels},
                                            {"name": "Team 1", "labels": [{"id": "team", "name": "Lung Opacity"}]}],
                            "datasets": [{"studies": [{"StudyInstanceUID": m["StudyInstanceUID"]} for m in self.mapping],
                                          "annotations": [
                                              {"SOPInstanceUID": self.mapping[0]["SOPInstanceUID"], "labelId": "positive"},
                                              # multiple boxes count as one positive image
                                              {"SOPInstanceUID": self.mapping[0]["SOPInstanceUID"], "labelId": "positive"},
                                              {"StudyInstanceUID": self.mapping[1]["StudyInstanceUID"], "labelId": "abnormal"},
                                              {"SOPInstanceUID": self.mapping[2]["SOPInstanceUID"], "labelId": "team"}]}]}
        if all_labeled:
            self.annotations["datasets"][0]["annotations"].append(
                {"SOPInstanceUID": self.mapping[2]["SOPInstanceUID"], "labelId": "normal"})
        self.cfg["expected"] = {
            "images": 3, "labeled_images": 3 if all_labeled else 2,
            "no_consensus_label": 0 if all_labeled else 1,
            "pixel_duplicate_groups": int(duplicate), "pixel_duplicate_images": 2 if duplicate else 0,
            "pixel_duplicate_labeled_images": 2 if duplicate else 0,
            "images_included": (3 if all_labeled else 2) - (2 if duplicate else 0),
            "patients_included": (int(all_labeled) if duplicate else 1 + int(all_labeled))}
        self.write_sources()

    def write_sources(self):
        source = lambda role: self.raw / self.cfg["sources"][role]["filename"]
        source("mappings").write_text(json.dumps(self.mapping), encoding="utf-8")
        source("annotations").write_text(json.dumps(self.annotations), encoding="utf-8")
        with zipfile.ZipFile(source("images_zip"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("study/", b"")
            for rec, data in zip(self.mapping, self.dicoms):
                archive.writestr("study/series/" + rec["SOPInstanceUID"] + ".dcm", data)
        for role, spec in self.cfg["sources"].items():
            spec["sha256"] = file_hash(source(role))
        self.write_config()

    def write_config(self):
        self.config.write_text(json.dumps(self.cfg), encoding="utf-8")

    def run(self, **kwargs):
        return rsna.acquire(self.raw, self.out, self.config, **kwargs)

    def private(self):
        return json.loads((self.out / "manifest/manifest-private.json").read_text())

    def cli(self, *args):
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-m", "augment_atlas.acquisition.rsna2018",
                               "--raw", str(self.raw), "--out", str(self.out), "--config", str(self.config), *args],
                              env=env, cwd=REPO, capture_output=True, text=True, timeout=60)


class RSNAAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def fixture(self, **kwargs):
        return SyntheticRelease(self.base, **kwargs)

    def test_config_pins_approved_definition_and_sources(self):
        cfg = rsna.load_config(CONFIG)
        self.assertEqual(cfg["policy"], rsna.POLICY)
        self.assertEqual(cfg["preprocessing"], rsna.PREPROCESSING)
        self.assertEqual(cfg["expected"]["images_included"], 29684 - 6)
        self.assertEqual(cfg["expected"]["no_consensus_label"], 316)
        self.assertEqual(cfg["expected"]["pixel_duplicate_groups"], 3)
        self.assertEqual(cfg["sources"]["images_zip"]["sha256"],
                         "96b97d81eb042c6513196e01079a060a980d5e62502c3266527dd9c0b2a63b50")
        self.assertEqual(cfg["sources"]["mappings"]["sha256"],
                         "803ce79e3bc9c66d3631738e91e62e1175730e98ad1415e8dc4d6292ba10bf27")
        self.assertEqual(cfg["sources"]["annotations"]["sha256"],
                         "f8f5c54bae158fc0e2f019af979471803151a5922463ff3a378d7734e01b18fb")

    def test_labels_nih_patient_group_study_fallback_and_no_consensus(self):
        f = self.fixture()
        result = f.run()
        private = f.private()
        records = {r["img_id"]: r for r in private["records"]}
        first, second, third = [records[m["img_id"]] for m in f.mapping]
        self.assertEqual([first["label"], second["label"], third["label"]], [1, 0, None])
        self.assertEqual(first["group"], f.mapping[0]["img_id"][:8])
        self.assertEqual(first["group"], second["group"])
        self.assertNotEqual(first["group"], first["subset_img_id"])
        self.assertEqual(third["reason"], "no_consensus_label")
        self.assertEqual(second["ViewPosition"], "AP")
        self.assertEqual(private["acquisition"]["annotations_resolved_by_study"], 1)
        self.assertEqual(result["counts"]["images_included"], 2)
        with (f.out / "manifest/manifest.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(set(rows[0]), {"path", "group", "label", "sha256"})
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(file_hash(f.out / row["path"]), row["sha256"])
        ready = json.loads((f.out / "manifest/READY.json").read_text())
        for name, digest in ready["files"].items():
            self.assertEqual(file_hash(f.out / "manifest" / name), digest)
        loaded = load_records({"dataset": {"adapter": "manifest_csv",
                                          "manifest": "manifest/manifest.csv",
                                          "manifest_sha256": ready["files"]["manifest.csv"],
                                          "require_image_hashes": True},
                               "model": {"num_classes": 2}}, root=f.out)
        self.assertEqual(len(loaded), 2)
        self.assertEqual({r["group"] for r in loaded}, {f.mapping[0]["img_id"][:8]})

    def test_all_members_of_decoded_pixel_duplicate_group_excluded(self):
        f = self.fixture(duplicate=True, all_labeled=True)
        f.run()
        records = f.private()["records"]
        excluded = [r for r in records if "pixel_duplicate" in r["reasons"]]
        self.assertEqual(len(excluded), 2)
        self.assertNotEqual(excluded[0]["original"]["sha256"], excluded[1]["original"]["sha256"])
        self.assertEqual(excluded[0]["original"]["pixel_sha256"], excluded[1]["original"]["pixel_sha256"])
        self.assertTrue(all(r["status"] == "excluded" and r["reason"] == "pixel_duplicate" for r in excluded))
        included = [r for r in records if r["status"] == "included"]
        self.assertEqual(len(included), 1)
        self.assertEqual(included[0]["label"], 0)  # Normal
        self.assertEqual(f.private()["observed_counts"]["pixel_duplicate_groups"], 1)

    def test_duplicate_with_unlabeled_member_still_excludes_labeled_member(self):
        f = self.fixture()
        f.dicoms[2] = dicom(30, f.mapping[2]["SOPInstanceUID"], f.mapping[2]["StudyInstanceUID"])
        f.cfg["expected"].update(pixel_duplicate_groups=1, pixel_duplicate_images=2,
                                 pixel_duplicate_labeled_images=1, images_included=1)
        f.write_sources()
        f.run()
        reasons = {r["img_id"]: r["reasons"] for r in f.private()["records"]}
        self.assertEqual(reasons[f.mapping[0]["img_id"]], ["pixel_duplicate"])
        self.assertEqual(reasons[f.mapping[2]["img_id"]], ["no_consensus_label", "pixel_duplicate"])

    def test_png_512_bilinear_no_windowing_and_metadata(self):
        f = self.fixture()
        f.run()
        for rec, data in zip(f.mapping, f.dicoms):
            source, _, decoded_hash = rsna_dicom.decode(data)
            path = f.out / "derived-512-v1/images" / rec["img_id"]
            with Image.open(path) as image:
                self.assertEqual((image.format, image.mode, image.size), ("PNG", "L", (512, 512)))
                self.assertEqual(image.info, {})
                expected = source.resize((512, 512), Image.Resampling.BILINEAR)
                self.assertEqual(image.tobytes(), expected.tobytes())
                self.assertEqual(decoded_hash, hashlib.sha256(source.tobytes()).hexdigest())

    def test_source_hash_mismatch_fails_verify_and_removes_stale_ready(self):
        f = self.fixture()
        f.run()
        (f.raw / f.cfg["sources"]["mappings"]["filename"]).write_text("[]")
        result = f.cli("verify")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("SHA256 mismatch", result.stderr)
        self.assertFalse((f.out / "manifest/READY.json").exists())
        self.assertFalse((f.out / "public/rsna2018-summary-public.json").exists())

    def test_counts_block_ready_until_exact_explicit_acceptance(self):
        f = self.fixture()
        f.cfg["expected"]["images_included"] = 99
        f.write_config()
        result = f.cli()
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertFalse((f.out / "manifest/READY.json").exists())
        divergence = json.loads((f.out / "manifest/divergences.json").read_text())["divergences"][0]
        self.assertEqual((divergence["field"], divergence["observed"]), ("counts.images_included", 2))
        f.cfg["accepted_divergences"] = [{"field": divergence["field"], "observed": 1,
                                         "resolution": "synthetic acceptance test", "decided_by": "test"}]
        f.write_config()
        self.assertEqual(f.cli("verify").returncode, 3)
        f.cfg["accepted_divergences"][0]["observed"] = 2
        f.write_config()
        self.assertEqual(f.cli("verify").returncode, 0)
        self.assertTrue(f.private()["divergences"][0]["accepted"])
        f.cfg["accepted_divergences"] = []
        f.write_config()
        self.assertEqual(f.cli("verify").returncode, 3)
        self.assertFalse((f.out / "manifest/READY.json").exists())

    def test_public_privacy_lint_and_ready_published_last(self):
        f = self.fixture()
        document = f.run()
        text = (f.out / "public/rsna2018-summary-public.json").read_text()
        self.assertEqual(document["privacy_lint"]["pattern_hits"], 0)
        self.assertEqual(document["privacy_lint"]["id_hits"], 0)
        for rec in f.mapping:
            for key in ("img_id", "subset_img_id", "SOPInstanceUID", "StudyInstanceUID"):
                self.assertNotIn(rec[key], text)
            self.assertNotIn(rec["img_id"][:8], text)
        self.assertNotIn("ViewPosition", text)
        f.cfg["dataset"]["license"] += " " + f.mapping[0]["subset_img_id"]
        f.write_config()
        # New output has no identity conflict; lint itself must block publication.
        f.out = self.base / "lint-out"
        with self.assertRaises(IntegrityError):
            f.run()
        self.assertFalse((f.out / "manifest/READY.json").exists())
        self.assertFalse((f.out / "public/rsna2018-summary-public.json").exists())

    def test_privacy_patterns_cannot_leak_identifiers_in_lint_metadata(self):
        f = self.fixture()
        f.cfg["privacy"]["patterns"].append(f.mapping[0]["subset_img_id"])
        f.write_config()
        with self.assertRaises(IntegrityError):
            f.run()
        self.assertFalse((f.out / "manifest/READY.json").exists())
        self.assertFalse((f.out / "public/rsna2018-summary-public.json").exists())

    def test_resume_idempotent_and_verify_never_decodes(self):
        f = self.fixture()
        f.run()
        files = list((f.out / "manifest").iterdir()) + list((f.out / "public").iterdir())
        before = {p: p.read_bytes() for p in files}
        pngs = list((f.out / "derived-512-v1/images").glob("*.png"))
        mtimes = {p: p.stat().st_mtime_ns for p in pngs}
        with patch.object(rsna, "derive", side_effect=AssertionError("must not decode again")):
            f.run()
            f.run(verify=True)
        self.assertEqual(before, {p: p.read_bytes() for p in files})
        self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in pngs})

    def test_interrupted_derivation_resumes_fetched_member(self):
        f = self.fixture()
        real_derive, calls = rsna.derive, []
        def fail_second(data):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("synthetic interruption")
            return real_derive(data)
        with patch.object(rsna, "derive", side_effect=fail_second):
            with self.assertRaises(RuntimeError):
                f.run()
        with closing(sqlite3.connect(f.out / "acquisition-v1/ledger.sqlite")) as conn:
            states = dict(conn.execute("SELECT state, count(*) FROM member GROUP BY state"))
        self.assertEqual(states["DERIVED"], 1)
        self.assertEqual(states["FETCHED"], 1)
        with self.assertRaises(Incomplete):
            f.run(verify=True)
        f.run()
        self.assertTrue((f.out / "manifest/READY.json").exists())

    def test_two_bounded_workers_cli(self):
        f = self.fixture()
        result = f.cli("--workers", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["counts"]["images_included"], 2)
        self.assertEqual(f.cli("--workers", "3").returncode, 2)

    def test_tampered_png_fails_verify(self):
        f = self.fixture()
        f.run()
        path = next((f.out / "derived-512-v1/images").glob("*.png"))
        path.write_bytes(b"tampered")
        self.assertEqual(f.cli("verify").returncode, 2)
        self.assertFalse((f.out / "manifest/READY.json").exists())

    def test_ambiguous_study_fallback_and_conflicting_labels_fail(self):
        f = self.fixture()
        mapping = copy.deepcopy(f.mapping)
        mapping[0]["StudyInstanceUID"] = mapping[1]["StudyInstanceUID"]
        ann = copy.deepcopy(f.annotations)
        ann["datasets"][0]["studies"] = [{"StudyInstanceUID": s} for s in {r["StudyInstanceUID"] for r in mapping}]
        with self.assertRaises(IntegrityError):
            rsna.reconcile(mapping, ann)
        ann = copy.deepcopy(f.annotations)
        ann["datasets"][0]["annotations"].append(
            {"SOPInstanceUID": f.mapping[0]["SOPInstanceUID"], "labelId": "normal"})
        with self.assertRaises(IntegrityError):
            rsna.reconcile(f.mapping, ann)

    def test_zip_mapping_and_dicom_identity_mismatch_fail_closed(self):
        f = self.fixture()
        f.dicoms[0] = f.dicoms[1]
        f.write_sources()
        with self.assertRaises(IntegrityError):
            f.run()
        f.dicoms.pop()
        f.mapping.pop()
        # A malformed annotation coverage or missing archive image cannot be accepted by counts.
        with self.assertRaises(IntegrityError):
            rsna.reconcile(f.mapping, f.annotations)

    def test_malformed_dicom_and_unsupported_syntax_rejected(self):
        f = self.fixture()
        for body in (f.dicoms[0][:-8], b"not DICOM",
                     f.dicoms[0].replace(rsna_dicom.JPEG_BASELINE.encode(),
                                          b"9" * len(rsna_dicom.JPEG_BASELINE), 1)):
            with self.assertRaises(IntegrityError):
                rsna_dicom.decode(body)


if __name__ == "__main__":
    unittest.main()
