"""RSNA 2018 split spec, pools and audit from a verified acquisition root (synthetic fixtures only).

The rsna2018 acquisition writes ``acquisition.sources`` as a dict keyed by role and no
``groups`` block (group = NIH patient, the eight-digit ``img_id`` prefix), and its config
declares no ``splits`` block: the split amendment is the sole split pre-registration.
The fixture root mimics ``acquisition/rsna2018.py::_verify`` output with synthetic IDs and
fixture thresholds (not the real amendment's). Golden hashes pin the ISIC 2020 spec/pools
and the INbreast legacy conversion as produced by the code before the RSNA fix (ae56e6d).
No real data, no network.
"""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from augment_atlas import global_development as gd
from augment_atlas import global_splits as gs
from augment_atlas import legacy_pools as lp
from augment_atlas.acquisition import config as config_mod
from augment_atlas.contracts import canonical_hash as H, file_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "acquisition" / "rsna2018-v1.json"
AMENDMENT = ROOT / "configs" / "acquisition" / "rsna2018-v1-splits-amendment-20261029.json"
ISIC_CONFIG = ROOT / "configs" / "acquisition" / "isic2020-v1.json"
ISIC_AMENDMENT = ROOT / "configs" / "acquisition" / "isic2020-v1-splits-amendment-20261022.json"
FIXTURE_THRESHOLDS = {"min_select_positive_groups": 2, "min_eval_fold_positive_groups": 5}
TASK = {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
        "decision_rule": "fixed_threshold", "missing_policy": "undefined_no_drop", "requires_patient_identity": True}

# Produced by the pre-fix code (ae56e6d) on the synthetic fixtures below; must never change.
GOLDEN = {
    "isic_spec": "a4eaeded72fdfbbc0d660ec834211c8ced3a9430bbf03d97effbe1a96f84788c",
    "isic_candidate": "5f611a67745f74c43047a81bc15ed3b6918c68c7575e28be2d2aa8259fbcd0a6",
    "isic_report": "6d80c0b1eb754562a834ad50b6eed5cb4241efec918af161780287b694f7c750",
    "isic_amendment_outcomes": "e99f8dd24250b9ff9d0f297020a59bb3c5d4bcd5c701b589029c54fde093e2b4",
    "inbreast_candidate": "650ba1a7784f38bc01d68a451f448c3fd1a5f0c1f07eb00c526bf918e324463f",
    "inbreast_audit": "cec14406f3b030c59b28c2bc8b62a3a7a260dda5b616863ca25571b613cba3dc",
}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def spec_like(amendment, **changes):
    value = {"dataset": {"id": amendment["dataset"], "version": "v"}, "task": copy.deepcopy(TASK),
             "grouping": {}, "provenance": {}, "source_sha256": "0" * 64,
             "seeds": copy.deepcopy(amendment["seeds"]), "fractions": copy.deepcopy(amendment["fractions"]),
             "thresholds": copy.deepcopy(amendment["thresholds"])}
    value.update(changes)
    return value


# --------------------------------------------------------------------------
# Golden fixtures (ISIC 2020 / INbreast): pure functions, no files.
# --------------------------------------------------------------------------

def isic_private():
    records = [{"image": f"SYNTH_{i:07d}", "status": "included" if i % 7 else "excluded",
                "reason": None if i % 7 else "duplicate_of_listed_pair"} for i in range(40)]
    return {"schema": "augment-atlas/dataset-manifest-private/v1",
            "dataset": {"name": "isic2020"},
            "acquisition": {"sources": [
                {"role": "images_zip", "etag": "e-zip", "bytes": 10, "central_directory_sha256": H("cd")},
                {"role": "ground_truth_v1", "etag": "e-gt1", "bytes": 11, "sha256": H("gt1")},
                {"role": "ground_truth_v2", "etag": "e-gt2", "bytes": 12, "sha256": H("gt2")},
                {"role": "duplicates", "etag": "e-dup", "bytes": 13, "sha256": H("dup")}]},
            "preprocessing": {"id": "isic2020-derived-512-v1", "short_side": 512, "format": "JPEG"},
            "duplicates": {"policy": "keep-min-image-name-v1", "pairs": [["SYNTH_0000000", "SYNTH_0000007"]],
                           "excluded": [{"image": "SYNTH_0000007", "reason": "duplicate"}]},
            "groups": {"algorithm": "patient-union-duplicates-v1", "n_groups": 2,
                       "patient_to_group": {"PSYNTH_0001": "g-1"}},
            "records": records}


def isic_rows(groups=200, positive_every=3):
    rows = []
    for g in range(groups):
        size = 1 + int(hashlib.sha256(f"size{g}".encode()).hexdigest(), 16) % 6
        for i in range(size):
            label = 1 if (g % positive_every == 0 and i == 0) else 0
            rows.append({"id": f"img/s{g:04d}-{i}.jpg", "path": f"img/s{g:04d}-{i}.jpg", "group": f"g-{g:04d}",
                         "label": label})
    return rows


def inbreast_fixture():
    rows = []
    for group in range(12):
        for image in range(4 if group < 4 else 3):
            index = len(rows)
            path = f"inbreast_png_896/{index:08d}_{index:016x}_MG_L_CC_ANON.png"
            rows.append({"id": path, "path": path, "group": f"synthetic-group-{group:02d}", "label": image % 2})
    ids = lambda groups: list(reversed([r["path"] for r in rows if int(r["group"][-2:]) in groups]))
    development, evaluation = ids(range(4)), ids(range(4, 12))
    legacy = {
        "protocol": "global-development-proposal-v1", "scientific_status": "provisional-feasibility-only",
        "sklearn": "1.8.0", "seeds": {"development": 17, "selection": 29, "evaluation": 31},
        "source_fold_sha256": H("synthetic fixed original folds"),
        "development": development, "search_train": ids(range(3)), "search_selection": ids([3]),
        "evaluation": evaluation,
        "folds": [{"fold": index, "test": ids(range(4 + index, 12, 5)),
                   "train": list(reversed(development + sorted(set(evaluation) - set(ids(range(4 + index, 12, 5)))))),
                   "diagnostics": {"legacy_only": "not trusted"}} for index in range(5)]}
    source = {"adapter": "inbreast_csv", "manifest": "manifest.csv", "manifest_sha256": H("synthetic manifest")}
    return rows, (json.dumps(legacy, indent=2) + "\n").encode("utf-8"), source


def _outcome(call):
    try:
        call()
    except ValueError as error:
        return str(error)
    return "ok"


def golden_hashes():
    """Hashes of the ISIC/INbreast outputs that the RSNA fix must leave bit-identical."""
    amendment = _load(ISIC_AMENDMENT)
    amendment["thresholds"] = dict(FIXTURE_THRESHOLDS)
    spec = gs.spec_from_acquisition({"schema": "augment-atlas/dataset-ready/v1"}, isic_private(), amendment,
                                    dataset_id="isic2020")
    candidate, report = gs.generate_dev_eval_pools(isic_rows(), spec)
    raw = _load(ISIC_CONFIG)
    real = _load(ISIC_AMENDMENT)
    stricter = copy.deepcopy(raw)
    stricter["splits"]["min_select_positive_groups"] = 16
    rebound = {**real, "amends": {**real["amends"], "config_identity_sha256": config_mod.identity_of(stricter)}}
    nulled = {**raw, "splits": None}
    nulled_amendment = {**real, "amends": {**real["amends"], "config_identity_sha256": config_mod.identity_of(nulled)}}
    outcomes = [_outcome(lambda: gs.check_amendment(spec_like(real), real, raw)),
                _outcome(lambda: gs.check_amendment(spec_like(real), rebound, stricter)),
                _outcome(lambda: gs.check_amendment(spec_like(real), nulled_amendment, nulled)),
                _outcome(lambda: gs.check_amendment(spec_like(real, seeds={**real["seeds"], "evaluation": 7}),
                                                    real, raw)),
                _outcome(lambda: gs.check_preregistered(spec_like(real), None))]
    rows, legacy_bytes, source = inbreast_fixture()
    converted, audit = lp.convert_legacy_pools(rows, legacy_bytes, source=source)
    return {"isic_spec": H(spec), "isic_candidate": H(candidate), "isic_report": H(report),
            "isic_amendment_outcomes": H(outcomes), "inbreast_candidate": H(converted),
            "inbreast_audit": H(audit)}


# --------------------------------------------------------------------------
# RSNA fixture: a verified acquisition root in the rsna2018 _verify format.
# --------------------------------------------------------------------------

def rsna_root(out, cfg, *, groups=100, positive_every=2):
    """Write manifest/{manifest.csv, records.json, manifest-private.json, divergences.json, READY.json}."""
    images = out / "derived-512-v1" / "images"
    images.mkdir(parents=True)
    manifest = out / "manifest"
    manifest.mkdir()
    private, included, rows, identifiers = [], [], [], set()
    index = 0
    for g in range(groups):
        patient = f"{90000000 + g:08d}"
        size = 1 + int(hashlib.sha256(f"rsna{g}".encode()).hexdigest(), 16) % 4
        for i in range(size + (1 if g % 9 == 0 else 0)):
            index += 1
            img_id = f"{patient}_{i:03d}.png"
            sop, study = f"1.2.826.0.1.3680043.10.999.{index}", f"1.2.826.0.1.3680043.10.998.{index}"
            body = f"synthetic png {img_id}".encode()
            (images / img_id).write_bytes(body)
            reasons = ["no_consensus_label"] if i == size else []  # one extra unlabelled image every 9th patient
            label = None if reasons else (1 if g % positive_every == 0 and i == 0 else 0)
            path = "derived-512-v1/images/" + img_id
            digest = hashlib.sha256(body).hexdigest()
            private.append({"SOPInstanceUID": sop, "StudyInstanceUID": study, "img_id": img_id,
                            "subset_img_id": f"00000000-0000-0000-0000-{index:012d}", "subset_group": "synthetic",
                            "subset_init_label": "unused", "orig_labels": [], "group": patient, "label": label,
                            "status": "excluded" if reasons else "included",
                            "reason": reasons[0] if reasons else None, "reasons": reasons, "ViewPosition": "PA",
                            "original": {"sha256": H(["dicom", index]), "pixel_sha256": H(["pixels", index])},
                            "derived": {"sha256": digest, "bytes": len(body), "path": path}})
            identifiers.update({sop, study, img_id, patient, img_id[:-4]})
            if not reasons:
                included.append({"id": path, "path": path, "group": patient, "label": label})
                rows.append((path, patient, label, digest))
    with (manifest / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("path", "group", "label", "sha256"))
        writer.writerows(rows)
    records_doc = {"schema": "augment-atlas/records/v1", "records": included}
    records_doc["canonical_hash"] = H(records_doc)
    divergences = {"schema": "augment-atlas/acquisition-divergences/v1", "stage": "verify", "divergences": []}
    sources = {role: {"filename": spec["filename"], "sha256": spec["sha256"], "bytes": 1000 + n}
               for n, (role, spec) in enumerate(sorted(cfg["sources"].items()))}
    private_doc = {"schema": "augment-atlas/dataset-manifest-private/v1", "dataset": cfg["dataset"],
                   "preprocessing": cfg["preprocessing"], "policy": cfg["policy"], "expected": cfg["expected"],
                   "observed_counts": {}, "divergences": [],
                   "acquisition": {"config_sha256": H(cfg), "config_identity_sha256": config_mod.identity_of(cfg),
                                   "sources": sources, "decoder": {"Pillow": "synthetic"},
                                   "annotations_resolved_by_study": 0},
                   "duplicates": {"policy": cfg["policy"]["duplicates"], "groups": []}, "records": private}
    private_doc["sha256"] = H(private_doc)
    for name, document in (("records.json", records_doc), ("manifest-private.json", private_doc),
                           ("divergences.json", divergences)):
        (manifest / name).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = {name: file_hash(manifest / name)
             for name in ("manifest.csv", "records.json", "manifest-private.json", "divergences.json")}
    ready = {"schema": "augment-atlas/dataset-ready/v1", "dataset": "rsna2018",
             "preprocessing_id": cfg["preprocessing"]["id"], "config_sha256": H(cfg),
             "config_identity_sha256": config_mod.identity_of(cfg), "files": files,
             "public_summary_sha256": H("synthetic public summary")}
    (manifest / "READY.json").write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ready, private_doc, identifiers


class RepositoryDocumentsTests(unittest.TestCase):
    """The committed rsna2018 config (no splits block) and its amendment, as published."""

    def setUp(self):
        self.raw = _load(CONFIG)
        self.amendment = _load(AMENDMENT)

    def test_amendment_is_the_sole_split_preregistration(self):
        self.assertNotIn("splits", self.raw)
        self.assertEqual(self.amendment["amends"]["config_identity_sha256"], config_mod.identity_of(self.raw))
        gs.check_amendment(spec_like(self.amendment), self.amendment, self.raw)

    def test_tampered_amendment_or_spec_is_refused(self):
        a = self.amendment
        cases = [
            (spec_like(a, seeds={**a["seeds"], "evaluation": 7}), a, "^split_spec_invalid: spec seeds"),
            (spec_like(a, thresholds={**a["thresholds"], "min_select_positive_groups": 1}), a,
             "^split_spec_invalid: spec thresholds"),
            (spec_like(a, fractions={"D": 0.25, "D_select_given_D": 0.3}), a, "^split_spec_invalid: spec fractions"),
            (spec_like(a), {**a, "before_any_score": False}, "^split_spec_invalid: split amendment must be"),
            (spec_like(a), {**a, "extra": 1}, "^split_spec_invalid: split amendment must be a closed"),
            (spec_like(a), {**a, "amends": {**a["amends"], "config_identity_sha256": "1" * 64}},
             "^split_spec_invalid: split amendment is bound"),
            (spec_like(a), {**a, "dataset": "isic2020"}, "^split_spec_invalid: the split amendment belongs"),
            (spec_like(a, seeds={"development": 1, "selection": 2, "evaluation": 1}), a, "^seed_reuse_forbidden:"),
        ]
        for spec, amendment, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                gs.check_amendment(spec, amendment, self.raw)
        # A splits key that is present but not a valid block is still refused (only absence delegates).
        nulled = {**self.raw, "splits": None}
        rebound = {**a, "amends": {**a["amends"], "config_identity_sha256": config_mod.identity_of(nulled)}}
        with self.assertRaisesRegex(ValueError, "^split_spec_invalid: acquisition config splits"):
            gs.check_amendment(spec_like(a), rebound, nulled)
        # A tampered acquisition config (identity change) breaks the binding.
        with self.assertRaisesRegex(ValueError, "^split_spec_invalid: split amendment is bound"):
            gs.check_amendment(spec_like(a), a, {**self.raw, "policy": {**self.raw["policy"], "group": "x"}})


class RsnaSpecTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-rsna-spec-")
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.cfg = _load(CONFIG)
        self.amendment = {**_load(AMENDMENT), "thresholds": dict(FIXTURE_THRESHOLDS)}
        self.ready, self.private, self.identifiers = rsna_root(self.out, self.cfg)

    def test_spec_declares_verified_patient_grouping_and_hashes_only(self):
        spec = gs.spec_from_acquisition(self.ready, self.private, self.amendment, dataset_id="rsna2018")
        self.assertEqual(spec["dataset"], {"id": "rsna2018", "version": "rsna2018-derived-512-v1"})
        grouping = spec["grouping"]
        self.assertEqual((grouping["identity_level"], grouping["linkage_status"], grouping["prior_exposure"]),
                         ("patient", "verified", "none_documented"))
        self.assertEqual(grouping["source"], "nih-img-id-eight-digit-patient-prefix")
        self.assertEqual(grouping["evidence_sha256"],
                         H({"group": self.cfg["policy"]["group"],
                            "mappings": self.private["acquisition"]["sources"]["mappings"]}))
        self.assertEqual(spec["source_sha256"], H(self.private["acquisition"]["sources"]))
        self.assertEqual((spec["seeds"], spec["thresholds"]), (self.amendment["seeds"], FIXTURE_THRESHOLDS))
        gs.check_amendment(spec, self.amendment, self.cfg)
        text = json.dumps(spec)
        self.assertFalse([i for i in self.identifiers if i in text], "the spec holds hashes only")
        # Exclusions are bound: dropping one excluded record changes the spec.
        edited = copy.deepcopy(self.private)
        edited["records"] = [r for r in edited["records"] if r["status"] == "included"]
        other = gs.spec_from_acquisition(self.ready, edited, self.amendment, dataset_id="rsna2018")
        self.assertNotEqual(other["provenance"]["exclusions_sha256"], spec["provenance"]["exclusions_sha256"])


class RsnaCliTests(unittest.TestCase):
    """split-spec -> make-pools -> import-pools -> audit-pools on the synthetic RSNA root."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-rsna-cli-")
        self.addCleanup(temporary.cleanup)
        self.dir = Path(temporary.name)
        self.root = self.dir / "rsna2018"
        self.cfg = _load(CONFIG)
        self.ready, _, self.identifiers = rsna_root(self.root, self.cfg)
        self.source = self.dir / "source.json"
        self.source.write_text(json.dumps({"adapter": "manifest_csv", "manifest": "manifest/manifest.csv",
                                           "manifest_sha256": self.ready["files"]["manifest.csv"]}))
        # Committed, published pre-registration: the real config bytes and a fixture-threshold amendment.
        self.code = self.dir / "code"
        (self.code / "configs" / "acquisition").mkdir(parents=True)
        self.config = self.code / "configs" / "acquisition" / "rsna2018-v1.json"
        self.config.write_bytes(CONFIG.read_bytes())
        self.amendment_doc = {**_load(AMENDMENT), "thresholds": dict(FIXTURE_THRESHOLDS)}
        self.amendment = self.code / "configs" / "acquisition" / AMENDMENT.name
        self.amendment.write_text(json.dumps(self.amendment_doc, indent=2) + "\n", encoding="utf-8")
        remote = self.dir / "code-remote.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True, capture_output=True)
        for argv in (["init", "-q", "-b", "main"], ["config", "user.email", "t@example.invalid"],
                     ["config", "user.name", "t"], ["add", "-A"], ["commit", "-qm", "pre-registered split amendment"],
                     ["remote", "add", "origin", str(remote)], ["push", "-q", "origin", "main"],
                     ["fetch", "-q", "origin"]):
            subprocess.run(["git", "-C", str(self.code), *argv], check=True, capture_output=True)
        self.base = ["--data-root", str(self.root), "--source", str(self.source), "--dataset", "rsna2018"]
        self.pins = ["--acquisition-config", str(self.config), "--split-amendment", str(self.amendment),
                     "--acquisition-root", str(self.root), "--code", str(self.code)]

    def cli(self, *argv, code=0):
        from augment_atlas import global_cli
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = global_cli.main(list(argv))
        self.assertEqual(code, result, err.getvalue())
        for identifier in self.identifiers:
            self.assertNotIn(identifier, out.getvalue() + err.getvalue())
        return json.loads(out.getvalue()) if code == 0 else err.getvalue()

    def split_spec(self, amendment=None, out="split-spec.json", code=0):
        return self.cli("split-spec", *self.base, "--acquisition-root", str(self.root),
                        "--split-amendment", str(amendment or self.amendment), "--out", str(self.dir / out), code=code)

    def test_split_spec_make_pools_import_and_audit(self):
        spec = self.split_spec()
        spec_path = self.dir / "split-spec.json"
        self.assertEqual(spec["dataset"], {"id": "rsna2018", "version": "rsna2018-derived-512-v1"})
        self.assertEqual(spec["seeds"], self.amendment_doc["seeds"])
        written = json.loads(spec_path.read_text())
        self.assertEqual(written["grouping"]["identity_level"], "patient")
        self.assertEqual(written["grouping"]["prior_exposure"], "none_documented")
        made = self.cli("make-pools", *self.base, "--spec", str(spec_path), *self.pins)
        self.assertEqual(made["seeds"], self.amendment_doc["seeds"])
        self.assertEqual(made["split_amendment_sha256"], hashlib.sha256(self.amendment.read_bytes()).hexdigest())
        self.assertEqual(made["acquisition_config_sha256"], hashlib.sha256(CONFIG.read_bytes()).hexdigest())
        self.assertGreaterEqual(made["groups"], 60)
        self.assertGreaterEqual(made["pools"]["D_select"]["positive_groups"],
                                FIXTURE_THRESHOLDS["min_select_positive_groups"])
        for fold in range(5):
            self.assertGreaterEqual(made["pools"][f"E_test_{fold}"]["positive_groups"],
                                    FIXTURE_THRESHOLDS["min_eval_fold_positive_groups"])
        imported = self.cli("import-pools", *self.base, "--file", made["path"],
                            "--expected-manifest", made["global_manifest_sha256"])
        self.assertEqual(imported["pools_sha256"], made["candidate_sha256"])
        pools = gd._read(Path(imported["path"]))
        self.assertEqual(pools["grouping"]["identity_level"], "patient")
        self.assertEqual(pools["partition_provenance"]["seeds"], self.amendment_doc["seeds"])
        group_of = {row[0]: row[1] for row in csv.reader((self.root / "manifest" / "manifest.csv").open())}
        placement = {}
        for name, ids in [("D_fit", pools["lists"]["D_fit"]), ("D_select", pools["lists"]["D_select"]),
                          *((f"E{f['fold']}", f["test_ids"]) for f in pools["evaluation_folds"])]:
            for item in ids:
                placement.setdefault(group_of[item], set()).add(name)
        self.assertTrue(all(len(where) == 1 for where in placement.values()), "a patient never spans two pools")
        audit = self.cli("audit-pools", *self.base, "--pools-sha256", imported["pools_sha256"],
                         "--expected-manifest", made["global_manifest_sha256"])
        self.assertIs(audit["structural_valid"], True)
        self.assertEqual(audit["eligibility"], {"contract_eligible": True, "reasons": []})
        # Same inputs, same candidate (deterministic; the private store is never clobbered).
        again = self.cli("make-pools", *self.base, "--spec", str(spec_path), *self.pins)
        self.assertEqual(again["candidate_sha256"], made["candidate_sha256"])

    def test_tampered_amendment_or_spec_fails(self):
        self.split_spec()
        spec_path = self.dir / "split-spec.json"
        # A spec whose seeds differ from the committed amendment.
        loose = json.loads(spec_path.read_text())
        loose["seeds"]["evaluation"] = 99
        (self.dir / "loose.json").write_text(json.dumps(loose))
        error = self.cli("make-pools", *self.base, "--spec", str(self.dir / "loose.json"), *self.pins, code=2)
        self.assertIn("split_spec_invalid: spec seeds", error)
        # Uncommitted edit of the committed amendment: dirty checkout.
        original = self.amendment.read_text()
        changed = json.loads(original)
        changed["thresholds"]["min_select_positive_groups"] = 1
        self.amendment.write_text(json.dumps(changed))
        error = self.cli("make-pools", *self.base, "--spec", str(spec_path), *self.pins, code=2)
        self.assertIn("code_identity_mismatch", error)
        self.amendment.write_text(original)
        # A private (untracked) tampered copy is not the pre-registration.
        tampered = self.dir / "tampered-amendment.json"
        tampered.write_text(json.dumps(changed))
        error = self.cli("make-pools", *self.base, "--spec", str(spec_path), "--acquisition-config", str(self.config),
                         "--split-amendment", str(tampered), "--acquisition-root", str(self.root),
                         "--code", str(self.code), code=2)
        self.assertIn("split_spec_invalid", error)
        # A spec derived from the tampered amendment does not match the committed one.
        self.split_spec(tampered, out="tampered-spec.json")
        error = self.cli("make-pools", *self.base, "--spec", str(self.dir / "tampered-spec.json"), *self.pins, code=2)
        self.assertIn("split_spec_invalid: spec thresholds", error)
        # An amendment bound to another config identity is refused at split-spec.
        other = self.dir / "other-amendment.json"
        other.write_text(json.dumps({**self.amendment_doc, "amends": {**self.amendment_doc["amends"],
                                                                     "config_identity_sha256": "1" * 64}}))
        error = self.split_spec(other, out="x.json", code=2)
        self.assertIn("another config identity", error)
        self.assertFalse((self.dir / "x.json").exists())
        # A manifest-private.json edited after READY is refused.
        private = self.root / "manifest" / "manifest-private.json"
        private.write_text(private.read_text().replace('"PA"', '"AP"', 1))
        error = self.split_spec(out="y.json", code=2)
        self.assertIn("differs from the hash READY.json records", error)


class GoldenOutputsTests(unittest.TestCase):
    def test_isic_and_inbreast_outputs_are_bit_identical_to_the_pre_fix_code(self):
        self.assertEqual(golden_hashes(), GOLDEN)


if __name__ == "__main__":
    unittest.main()
