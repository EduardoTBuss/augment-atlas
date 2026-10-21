"""Unit tests for ISIC acquisition part B (config, CSV sources, reconciliation, groups, splits,
summary lint, disk guard, ledger, lazy-Pillow derivation) and the repository ID policy.

Synthetic identifiers only (SYNTH_/PSYNTH_/LSYNTH_); no network beyond what other files test.
"""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquisition_e2e_helpers import DUP_HEADER, ROOT_CONFIG, V1_HEADER, V2_HEADER, SyntheticRelease  # noqa: E402

from augment_atlas.acquisition import config as config_mod  # noqa: E402
from augment_atlas.acquisition import csvsource, derive, groups, isic2020, splits_global, summary  # noqa: E402
from augment_atlas.acquisition.diskguard import DiskGuard, ProcessLock  # noqa: E402
from augment_atlas.acquisition.errors import DiskPaused, DivergenceError, IntegrityError, LockBusy  # noqa: E402
from augment_atlas.acquisition.ledger import Ledger  # noqa: E402
from augment_atlas.acquisition.zipcd import CDEntry  # noqa: E402

GIB = 1 << 30
REPO = Path(__file__).resolve().parents[1]


def raw_config():
    return json.loads(ROOT_CONFIG.read_text(encoding="utf-8"))


def synthetic_config(release=None, **documented):
    """The versioned config with synthetic name patterns and the release's own counts."""
    raw = raw_config()
    raw["names"]["member_regex"] = r"^(?:[A-Za-z0-9_.-]+/)*(?P<image>SYNTH_\d{7})\.jpg$"
    raw["names"]["image_name_regex"] = r"^SYNTH_\d{7}$"
    raw["privacy"]["patterns"] = [r"SYNTH_\d{7}", r"PSYNTH_\d{4}", r"LSYNTH_\d{7}"]
    if release is not None:
        raw["documented_counts"] = release.counts()
    raw["documented_counts"].update(documented)
    return config_mod.validate(raw)


def parsed(release, role, **kwargs):
    header = {"ground_truth_v1": V1_HEADER, "ground_truth_v2": V2_HEADER, "duplicates": DUP_HEADER}[role]
    return csvsource.parse_csv(release.csv_bytes(role, **kwargs), header, name=role)


def analyze(release, cfg=None, **kwargs):
    cfg = cfg or synthetic_config(release)
    return isic2020.analyze_labels(parsed(release, "ground_truth_v2", **kwargs.get("v2", {})),
                                   parsed(release, "ground_truth_v1", **kwargs.get("v1", {})),
                                   parsed(release, "duplicates", **kwargs.get("dup", {})), cfg)


class ConfigTests(unittest.TestCase):
    def test_versioned_config_pins_the_verified_facts_and_decisions(self):
        cfg = config_mod.load(ROOT_CONFIG)
        base = "https://isic-archive.s3.amazonaws.com/challenges/2020/"
        expected = {"images_zip": ("ISIC_2020_Training_JPEG.zip", 24707698022, "ebb879d3b9be0f5b4515e28dc8ebbb88-472"),
                    "ground_truth_v1": ("ISIC_2020_Training_GroundTruth.csv", 2056020, "86657c7009b003543b377b62a8745b4c"),
                    "ground_truth_v2": ("ISIC_2020_Training_GroundTruth_v2.csv", 2387418, "f894624ff25df80143501a2702d4d9fe"),
                    "duplicates": ("ISIC_2020_Training_Duplicates.csv", 11500, "62992a2252b492095a54dbc5ce0d5b3e")}
        for role, (name, size, etag) in expected.items():
            self.assertEqual(cfg.source(role)["url"], base + name)
            self.assertEqual((cfg.source(role)["bytes"], cfg.source(role)["etag"]), (size, etag))
        self.assertEqual(cfg.source("ground_truth_v1")["header"], V1_HEADER)
        self.assertEqual(cfg.source("ground_truth_v2")["header"], V2_HEADER)
        self.assertEqual(cfg.source("duplicates")["header"], DUP_HEADER)
        self.assertEqual(cfg.documented, {"images": 33126, "patients": 2056, "melanoma_images": 584,
                                          "duplicate_pairs": 425, "duplicate_distinct_images": 850})
        self.assertEqual(cfg.preprocessing["id"], "isic2020-derived-512-v1")
        self.assertEqual((cfg.preprocessing["short_side"], cfg.preprocessing["quality"], cfg.preprocessing["subsampling"],
                          cfg.preprocessing["resample"], cfg.preprocessing["upscale"]), (512, 95, "4:4:4", "LANCZOS", False))
        self.assertEqual((cfg.splits["dev_fraction"], cfg.splits["select_fraction"]), (0.2, 0.3))
        self.assertIsInstance(cfg.splits["split_seed"], int)
        self.assertEqual(cfg.disk_bytes("reserve_gib"), 20 * GIB)
        self.assertEqual((cfg.transfer["max_rate_mib_s"], cfg.transfer["workers"], cfg.canary["limit"]), (25, 2, 500))
        self.assertTrue(cfg.member_regex.fullmatch("train/" + "ISIC_" + "0" * 7 + ".jpg"))

    def test_operational_keys_do_not_change_the_identity(self):
        raw = raw_config()
        base = config_mod.validate(copy.deepcopy(raw))
        raw["transfer"]["max_rate_mib_s"] = 5
        raw["accepted_divergences"] = [{"field": "x", "observed": 1, "resolution": "r", "decided_by": "Eduardo"}]
        changed = config_mod.validate(raw)
        self.assertEqual(base.identity_sha256, changed.identity_sha256)
        self.assertNotEqual(base.sha256, changed.sha256)
        raw["splits"]["split_seed"] += 1
        self.assertNotEqual(base.identity_sha256, config_mod.validate(raw).identity_sha256)

    def test_invalid_configs_are_refused(self):
        def broken(mutate):
            raw = raw_config()
            mutate(raw)
            with self.assertRaises(config_mod.ConfigError):
                config_mod.validate(raw)
        broken(lambda r: r["sources"]["ground_truth_v2"]["header"].remove("patient_id"))
        broken(lambda r: r["sources"]["duplicates"].update(url=r["sources"]["duplicates"]["url"] + "?X-Amz-Signature=1"))
        broken(lambda r: r["sources"]["images_zip"].update(etag='"quoted"'))
        broken(lambda r: r["transfer"].update(workers=3))
        broken(lambda r: r["names"].update(member_regex=r"^(?P<other>x)$"))
        broken(lambda r: r["preprocessing"].update(upscale=True))
        broken(lambda r: r["splits"].update(split_seed="1"))
        broken(lambda r: r.pop("canary"))
        broken(lambda r: r["accepted_divergences"].append({"field": "x", "observed": 1}))


class CsvSourceTests(unittest.TestCase):
    def setUp(self):
        self.release = SyntheticRelease(n_patients=4)

    def test_exact_header_parses_rows(self):
        rows = parsed(self.release, "ground_truth_v2")
        self.assertEqual(len(rows), len(self.release.rows))
        self.assertEqual(rows[0]["patient_id"], self.release.rows[0]["patient_id"])

    def test_header_divergence_is_exit_3_without_mapping(self):
        body = self.release.csv_bytes("ground_truth_v2")
        for header in (V2_HEADER[:-1], list(reversed(V2_HEADER)), V2_HEADER[:2] + ["Lesion_ID"] + V2_HEADER[3:]):
            with self.assertRaises(DivergenceError) as caught:
                csvsource.parse_csv(body, header, name="ground_truth_v2")
            self.assertEqual(caught.exception.exit_code, 3)

    def test_bom_and_crlf_are_encoding_artifacts_only(self):
        body = b"\xef\xbb\xbf" + self.release.csv_bytes("duplicates", line_end="\r\n")
        self.assertTrue(csvsource.has_utf8_bom(body))
        self.assertEqual(len(csvsource.parse_csv(body, DUP_HEADER, name="duplicates")), len(self.release.pairs))

    def test_malformed_bodies_are_divergences(self):
        with self.assertRaises(DivergenceError):
            csvsource.parse_csv(b"image_name_1,image_name_2\nSYNTH_0000001\n", DUP_HEADER, name="duplicates")
        with self.assertRaises(DivergenceError):
            csvsource.parse_csv(b"image_name_1,image_name_2\n\xff\xfe,x\n", DUP_HEADER, name="duplicates")
        with self.assertRaises(DivergenceError):
            csvsource.parse_csv(b"", DUP_HEADER, name="duplicates")

    def test_simple_etag_is_checked_as_md5_multipart_is_not(self):
        body = b"abc"
        md5 = hashlib.md5(body).hexdigest()
        self.assertTrue(csvsource.verify_bytes(body, expected_bytes=3, expected_etag=md5)["etag_is_md5"])
        with self.assertRaises(IntegrityError):
            csvsource.verify_bytes(body, expected_bytes=3, expected_etag="0" * 32)
        with self.assertRaises(IntegrityError):
            csvsource.verify_bytes(body, expected_bytes=4, expected_etag=md5)
        self.assertFalse(csvsource.verify_bytes(body, expected_bytes=3, expected_etag="0" * 32 + "-472")["etag_is_md5"])

    def test_retained_release_file_must_stay_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "release" / "file.csv"
            csvsource.store_release_file(b"one", dest)
            csvsource.store_release_file(b"one", dest)
            with self.assertRaises(IntegrityError):
                csvsource.store_release_file(b"two", dest)
            self.assertEqual(dest.read_bytes(), b"one")
            self.assertEqual(sorted(p.name for p in dest.parent.iterdir()), ["file.csv"])


class LabelAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.release = SyntheticRelease(n_patients=12)

    def test_consistent_release_has_no_divergence(self):
        records, pairs, observed, divergences = analyze(self.release)
        self.assertEqual(divergences, [])
        self.assertEqual(len(records), len(self.release.rows))
        self.assertEqual(pairs, sorted(tuple(sorted(p)) for p in self.release.pairs))
        self.assertEqual(observed["melanoma_images"], self.release.counts()["melanoma_images"])
        self.assertEqual(observed["duplicate_distinct_images_official"], 4)

    def test_count_differences_are_recorded_not_raised(self):
        cfg = synthetic_config(self.release, images=33126, duplicate_pairs=425)
        _records, _pairs, _observed, divergences = analyze(self.release, cfg)
        fields = {d["field"]: d for d in divergences}
        self.assertEqual(fields["counts.images_csv"]["observed"], len(self.release.rows))
        self.assertEqual(fields["counts.images_csv"]["documented"], 33126)
        self.assertIn("counts.duplicate_rows", fields)
        self.assertIn("counts.duplicate_pairs", fields)
        self.assertTrue(all(d["accepted"] is False for d in divergences))
        accepted = [{"field": "counts.images_csv", "observed": len(self.release.rows), "resolution": "ok",
                     "decided_by": "Eduardo"},
                    {"field": "counts.duplicate_pairs", "observed": 999, "resolution": "stale", "decided_by": "Eduardo"}]
        isic2020.apply_acceptance(divergences, accepted)
        fields = {d["field"]: d for d in divergences}
        self.assertTrue(fields["counts.images_csv"]["accepted"])
        self.assertFalse(fields["counts.duplicate_pairs"]["accepted"], "acceptance must name the observed value")

    def test_v1_v2_mismatch_and_label_incoherence_are_divergences(self):
        v1_rows = copy.deepcopy(self.release.rows)
        v1_rows[0]["patient_id"] = "PSYNTH_9999"
        v1_rows[1]["target"] = "1" if v1_rows[1]["target"] == "0" else "0"
        v2_rows = copy.deepcopy(self.release.rows)
        v2_rows[2]["diagnosis"] = "melanoma"
        _r, _p, _o, divergences = analyze(self.release, v1={"rows": v1_rows}, v2={"rows": v2_rows})
        fields = {d["field"] for d in divergences}
        self.assertTrue({"ground_truth_v1_vs_v2.patient_id_mismatch", "ground_truth_v1_vs_v2.target_mismatch",
                         "labels.target_vs_diagnosis_melanoma"} <= fields)

    def test_unusable_documents_are_hard_divergences(self):
        for mutate in (lambda rows: rows[0].update(patient_id=""), lambda rows: rows[0].update(target="2"),
                       lambda rows: rows[0].update(image_name="SYNTH_12"),
                       lambda rows: rows.append(dict(rows[0]))):
            rows = copy.deepcopy(self.release.rows)
            mutate(rows)
            with self.assertRaises(DivergenceError):
                analyze(self.release, v2={"rows": rows})

    def test_duplicate_list_anomalies_are_recorded(self):
        names = [r["image_name"] for r in self.release.rows]
        rows = [[names[1], names[2]], [names[2], names[1]], [names[7], names[7]], [names[2], names[9]],
                [names[3], "SYNTH_9999999"]]
        _r, _p, observed, divergences = analyze(self.release, dup={"rows": rows})
        fields = {d["field"]: d["observed"] for d in divergences}
        self.assertEqual(fields["duplicates.repeated_pairs"], 1)
        self.assertEqual(fields["duplicates.self_pairs"], 1)
        self.assertEqual(fields["duplicates.pairs_not_disjoint"], 1)
        self.assertEqual(fields["duplicates.names_not_in_ground_truth"], 1)
        self.assertEqual(observed["duplicate_rows_official"], 5)


def cd_entry(index, name, is_dir=False):
    return CDEntry(index=index, name=name, raw_name=name.encode(), method=0, flags=0, crc32=0, compressed_size=0,
                   uncompressed_size=0, local_header_offset=index * 100, is_dir=is_dir)


class MemberClassificationTests(unittest.TestCase):
    def test_kinds_and_reconstructed_names(self):
        cfg = synthetic_config(SyntheticRelease(n_patients=3))
        names = ["train/", "train/SYNTH_0000001.jpg", "SYNTH_0000002.jpg", "__MACOSX/train/._SYNTH_0000001.jpg",
                 "train/.DS_Store", "train/SYNTH_0000003.JPG", "train/notes.txt", "train/SYNTH_00000041.jpg"]
        members = isic2020.classify_members([cd_entry(i, n, n.endswith("/")) for i, n in enumerate(names)], cfg)
        kinds = [(m["kind"], m["image"]) for m in members]
        self.assertEqual(kinds, [("dir", None), ("image", "SYNTH_0000001"), ("image", "SYNTH_0000002"), ("junk", None),
                                 ("junk", None), ("unexpected", None), ("unexpected", None), ("unexpected", None)])

    def test_two_members_for_one_image_is_integrity_failure(self):
        cfg = synthetic_config(SyntheticRelease(n_patients=3))
        with self.assertRaises(IntegrityError):
            isic2020.classify_members([cd_entry(0, "a/SYNTH_0000001.jpg"), cd_entry(1, "b/SYNTH_0000001.jpg")], cfg)

    def test_cross_check_reports_both_directions(self):
        release = SyntheticRelease(n_patients=3)
        cfg = synthetic_config(release)
        records, _p, _o, _d = analyze(release, cfg)
        names = sorted(records)
        entries = [cd_entry(i, f"train/{n}.jpg") for i, n in enumerate(names[1:])] + [cd_entry(99, "train/SYNTH_0099999.jpg")]
        observed, divergences = isic2020.cross_check(isic2020.classify_members(entries, cfg), records, cfg)
        fields = {d["field"]: d["observed"] for d in divergences}
        self.assertEqual(fields, {"zip_csv.images_without_label": 1, "zip_csv.labels_without_image": 1})
        self.assertEqual(observed["zip_jpg_members"], len(names))


class GroupTests(unittest.TestCase):
    def records(self):
        return {"SYNTH_0000001": {"patient_id": "PSYNTH_0001", "label": 0},
                "SYNTH_0000002": {"patient_id": "PSYNTH_0001", "label": 0},
                "SYNTH_0000003": {"patient_id": "PSYNTH_0002", "label": 1},
                "SYNTH_0000004": {"patient_id": "PSYNTH_0003", "label": 0},
                "SYNTH_0000005": {"patient_id": "PSYNTH_0004", "label": 0}}

    def test_cross_patient_pair_merges_groups_deterministically(self):
        records = self.records()
        pairs = [("SYNTH_0000001", "SYNTH_0000004")]
        p2g, i2g, stats = groups.build_groups(records, pairs, "isic20-g-")
        self.assertEqual(p2g["PSYNTH_0001"], p2g["PSYNTH_0003"])
        self.assertNotEqual(p2g["PSYNTH_0001"], p2g["PSYNTH_0002"])
        self.assertEqual(stats, {"patients": 4, "groups": 3, "cross_patient_pairs": 1})
        self.assertEqual(p2g["PSYNTH_0001"], groups.group_id("isic20-g-", ["PSYNTH_0003", "PSYNTH_0001"]))
        self.assertRegex(p2g["PSYNTH_0002"], r"^isic20-g-[0-9a-f]{16}$")
        reordered = dict(reversed(list(records.items())))
        self.assertEqual(groups.build_groups(reordered, list(reversed(pairs)), "isic20-g-")[1], i2g)

    def test_duplicate_policy_keeps_min_name_and_drops_label_conflicts(self):
        records = self.records()
        excluded = groups.duplicate_policy(records, [("SYNTH_0000004", "SYNTH_0000001"), ("SYNTH_0000002", "SYNTH_0000004"),
                                                     ("SYNTH_0000003", "SYNTH_0000005")])
        self.assertEqual(excluded, {"SYNTH_0000002": "duplicate-of:SYNTH_0000001",
                                    "SYNTH_0000004": "duplicate-of:SYNTH_0000001",
                                    "SYNTH_0000003": "duplicate-label-conflict",
                                    "SYNTH_0000005": "duplicate-label-conflict"})


class SplitTests(unittest.TestCase):
    def records(self, n_groups=400):
        out = []
        for g in range(n_groups):
            for k in range(1 + g % 5):
                out.append({"group": f"isic20-g-{g:016x}", "label": int(g % 4 == 0 and k == 0)})
        return out

    def params(self, **overrides):
        params = {"split_seed": 20261019, "dev_fraction": 0.2, "select_fraction": 0.3, "eval_folds": 5,
                  "min_select_positive_groups": 1, "min_eval_fold_positive_groups": 1}
        params.update(overrides)
        return params

    def test_pools_are_deterministic_disjoint_and_complete(self):
        records = self.records()
        first = splits_global.build(records, self.params())
        second = splits_global.build(list(reversed(records)), self.params())
        self.assertEqual(first[0]["canonical_hash"], second[0]["canonical_hash"])
        self.assertEqual(first[1]["canonical_hash"], second[1]["canonical_hash"])
        pools, folds, audit = first
        self.assertTrue(audit["passed"], audit["failures"])
        assigned = pools["pools"]["D_fit"] + pools["pools"]["D_select"] + [g for f in folds["folds"].values() for g in f]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {r["group"] for r in records})
        self.assertEqual(sorted(pools["pools"]["E"]), sorted(g for f in folds["folds"].values() for g in f))
        self.assertNotEqual(splits_global.build(records, self.params(split_seed=1))[0]["canonical_hash"],
                            first[0]["canonical_hash"])

    def test_audit_fails_on_preregistered_thresholds_without_retry(self):
        _pools, _folds, audit = splits_global.build(self.records(), self.params(min_eval_fold_positive_groups=50))
        self.assertFalse(audit["passed"])
        self.assertTrue(any("E_0" in failure for failure in audit["failures"]))

    def test_round_half_up_and_quartiles(self):
        self.assertEqual([splits_global.round_half_up(n, 2) for n in (1, 3, 5)], [1, 2, 3])
        table = {f"g{i}": (n, 0) for i, n in enumerate([1, 1, 2, 3, 4, 5, 8, 9])}
        stratified, thresholds = splits_global.strata(table)
        self.assertEqual(thresholds, [2, 4, 8])
        self.assertEqual(sum(len(v) for v in stratified.values()), 8)


class SummaryLintTests(unittest.TestCase):
    def test_privacy_lint_counts_patterns_and_literal_identifiers(self):
        text = json.dumps({"a": "SYNTH_0000001", "b": "isic20-g-0123456789abcdef", "c": 3})
        lint = summary.privacy_lint(text, [r"SYNTH_\d{7}"], {"isic20-g-0123456789abcdef", "", "PSYNTH_0001"})
        self.assertEqual((lint["pattern_hits"], lint["id_hits"]), (1, 1))
        short = summary.privacy_lint(text, [], {"abc12", "isic20"})
        self.assertEqual(short["id_hits"], 1, "literals shorter than 6 characters are left to the regexes")

    def test_build_summary_refuses_identifier_leaks(self):
        cfg = synthetic_config(SyntheticRelease(n_patients=3))
        kwargs = dict(cfg=cfg, tool_commit="0" * 40, docker_image_digest=None, zip_source={"etag": "etagvalue123"}, hashes={},
                      included=[{"group": "isic20-g-aaaaaaaaaaaaaaaa", "label": 1}], excluded_count=0,
                      groups_stats={"patients": 1}, duplicates={}, divergences=[], derived_bytes_total=1,
                      splits_counts={})
        document = summary.build_summary(identifiers={"isic20-g-aaaaaaaaaaaaaaaa"}, **kwargs)
        self.assertEqual(document["privacy_lint"]["pattern_hits"], 0)
        leak = isic2020.divergence("x", 0, 1, "s", ["SYNTH_0000001"])
        leak["field"] = "leak SYNTH_0000001"
        with self.assertRaises(IntegrityError):
            summary.build_summary(identifiers=set(), **{**kwargs, "divergences": [leak]})
        with self.assertRaises(IntegrityError):
            summary.build_summary(identifiers={"etagvalue123"}, **kwargs)

    def test_quantiles_nearest_rank(self):
        self.assertEqual(summary.quantiles([5, 1, 3, 2, 4]), {"p10": 1, "p50": 3, "p90": 5})
        self.assertEqual(summary.quantiles([]), {"p10": None, "p50": None, "p90": None})


class DiskGuardTests(unittest.TestCase):
    def test_precheck_and_write_margin_boundaries(self):
        free = [31 * GIB]
        guard = DiskGuard("unused", reserve_bytes=20 * GIB, write_margin_bytes=GIB, free_fn=lambda _p: free[0])
        guard.precheck(11 * GIB)
        with self.assertRaises(DiskPaused) as caught:
            guard.precheck(11 * GIB + 1)
        self.assertEqual(caught.exception.exit_code, 75)
        guard.before_write(10 * GIB)
        with self.assertRaises(DiskPaused):
            guard.before_write(10 * GIB + 1)
        self.assertEqual(guard.min_headroom, -1 + GIB)

    def test_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state" / "lock"
            with ProcessLock(path):
                with self.assertRaises(LockBusy) as caught:
                    ProcessLock(path).acquire()
                self.assertEqual(caught.exception.exit_code, 75)
            with ProcessLock(path):
                pass


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "ledger.sqlite"
        self.entries = [cd_entry(0, "train/", True), cd_entry(1, "train/SYNTH_0000001.jpg"),
                        cd_entry(2, "train/SYNTH_0000002.jpg")]
        self.members = [{"index": 0, "position": 0, "kind": "dir", "image": None},
                        {"index": 1, "position": 1, "kind": "image", "image": "SYNTH_0000001"},
                        {"index": 2, "position": 2, "kind": "image", "image": "SYNTH_0000002"}]

    def test_pin_refuses_identity_change(self):
        with Ledger(self.path) as ledger:
            ledger.pin({"etag": "a", "size": 1})
            ledger.pin({"etag": "a", "size": 1, "new": "x"})
            with self.assertRaises(DivergenceError):
                ledger.pin({"etag": "b"})

    def test_transitions_are_guarded_and_resume_point_is_a_suffix(self):
        with Ledger(self.path) as ledger:
            ledger.init_members(self.entries, self.members, {"SYNTH_0000001", "SYNTH_0000002"})
            self.assertEqual(ledger.counts(), {"PENDING": 2, "FETCHED": 0, "DERIVED": 0, "SKIPPED": 1})
            self.assertEqual(ledger.first_pending_position(3), 1)
            with self.assertRaises(IntegrityError):
                ledger.mark_derived(1, {"derived": {"bytes": 1}})
            ledger.mark_fetched(1, "a" * 64)
            with self.assertRaises(IntegrityError):
                ledger.mark_fetched(1, "a" * 64)
            ledger.mark_derived(1, {"derived": {"bytes": 7}})
            self.assertEqual(ledger.first_pending_position(3), 2)
            self.assertEqual(ledger.derived_stats(), (1, 7))
            ledger.init_members(self.entries, self.members, {"SYNTH_0000001", "SYNTH_0000002"})  # same CD: no-op
            changed = list(self.entries)
            changed[2] = CDEntry(**{**changed[2].__dict__, "crc32": 5})
            with self.assertRaises(IntegrityError):
                ledger.init_members(changed, self.members, {"SYNTH_0000001", "SYNTH_0000002"})


def _pil():
    try:
        from PIL import Image, ImageCms, JpegImagePlugin  # noqa: F401
    except ImportError:
        return None
    return Image


@unittest.skipIf(_pil() is None, "Pillow is not installed (derive runs in the pinned container)")
class DeriveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.params = dict(raw_config()["preprocessing"])

    def jpeg(self, name, size, mode="RGB", orientation=None, icc=False):
        from PIL import Image, ImageCms
        image = Image.new(mode, size)
        pixels = image.load()
        for x in range(0, size[0], 7):
            for y in range(0, size[1], 5):
                pixels[x, y] = (x % 256, y % 256, (x + y) % 256) if mode == "RGB" else (x + y) % 256
        kwargs = {"quality": 90}
        if orientation is not None:
            exif = Image.Exif()
            exif[0x0112] = orientation
            kwargs["exif"] = exif.tobytes()
        if icc:
            kwargs["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        path = self.dir / name
        image.save(path, "JPEG", **kwargs)
        return path

    def test_target_size_is_exact_and_never_upscales(self):
        self.assertEqual(derive.target_size(6000, 4000, 512), (768, 512))
        self.assertEqual(derive.target_size(4000, 6000, 512), (512, 768))
        self.assertEqual(derive.target_size(1023, 1000, 512), (524, 512))   # 523.776 -> 524
        self.assertEqual(derive.target_size(1001, 1000, 512), (513, 512))   # 512.512 -> 513
        self.assertEqual(derive.target_size(3, 2, 512), (3, 2))
        self.assertEqual(derive.target_size(512, 700, 512), (512, 700))

    def test_derived_jpeg_follows_isic2020_derived_512_v1(self):
        from PIL import Image, JpegImagePlugin
        src = self.jpeg("a.jpg", (1200, 800), orientation=6, icc=True)
        result = derive.derive_member(src, self.dir / "a.jpg.tmp", self.params)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["original"], {"format": "JPEG", "mode": "RGB", "width": 1200, "height": 800,
                                              "exif_orientation": 6, "icc": True})
        self.assertFalse(result["short_side_lt_target"])
        out = self.dir / "a.jpg.tmp"
        with Image.open(out) as image:
            self.assertEqual((image.format, image.mode, image.size), ("JPEG", "RGB", (768, 512)))
            self.assertEqual(JpegImagePlugin.get_sampling(image), 0)  # 4:4:4
            self.assertNotIn("exif", image.info)
            self.assertNotIn("icc_profile", image.info)
            self.assertNotIn("progressive", image.info)
            self.assertEqual(result["derived"]["pixel_sha256"], derive.pixel_sha256(image))
        self.assertEqual(result["derived"]["sha256"], hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertEqual(result["derived"]["bytes"], out.stat().st_size)
        again = derive.derive_member(src, self.dir / "b.jpg.tmp", self.params)
        self.assertEqual(again["derived"]["sha256"], result["derived"]["sha256"], "derivation must be deterministic")

    def test_small_and_grayscale_inputs(self):
        from PIL import Image
        result = derive.derive_member(self.jpeg("s.jpg", (400, 300), mode="L"), self.dir / "s.jpg.tmp", self.params)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["original"]["mode"], "L")
        self.assertTrue(result["short_side_lt_target"])
        with Image.open(self.dir / "s.jpg.tmp") as image:
            self.assertEqual((image.mode, image.size), ("RGB", (400, 300)))

    def test_undecodable_input_is_skipped_without_leftovers(self):
        bad = self.dir / "bad.jpg"
        bad.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
        result = derive.derive_member(bad, self.dir / "bad.jpg.tmp", self.params)
        self.assertEqual(result["status"], "skipped")
        self.assertTrue(result["reason"].startswith("decode-failed:"))
        self.assertFalse((self.dir / "bad.jpg.tmp").exists())


class RepositoryIdPolicyTests(unittest.TestCase):
    """Fixtures, configs and docs never carry real ISIC identifiers (image/patient/lesion)."""

    PATTERN = re.compile(rb"\b(?:ISIC|IP|IL)_\d{7}\b")

    def test_no_real_identifier_pattern_in_repository_text(self):
        hits = []
        for top in ("src", "tests", "configs", "docs", "scripts", "docker", "README.md"):
            base = REPO / top
            paths = [base] if base.is_file() else [Path(d) / f for d, _s, files in os.walk(base) for f in files]
            for path in paths:
                if "__pycache__" in path.parts or path.suffix in (".pyc", ".jpg", ".png", ".bin"):
                    continue
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                if self.PATTERN.search(data):
                    hits.append(path.relative_to(REPO).as_posix())
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
