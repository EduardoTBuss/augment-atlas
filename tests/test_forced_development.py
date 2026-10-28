"""Optional forced-D extension: synthetic group metadata only, no images/network."""
import contextlib
import copy
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from augment_atlas import global_cli as cli
from augment_atlas import global_development as gd
from augment_atlas import global_splits as gs
from augment_atlas.contracts import canonical_hash as H
from test_global_splits import synthetic, spec, groups_of

# Captured at a22833f before editing (validation/c2-golden-before.json).
GOLDEN_CANDIDATE = "096fe5b86ee829833b15fad7d3231a19e9c55896b5e19cb1a998b2f2d9112c63"
GOLDEN_SIGNED_BYTES = "b0c96cd8252fac505f733d363a772aebf58f25ed4550b12c21e9b3e392391e26"
GOLDEN_MANIFEST = "e504e806175006055ad1d29027125b2be6b6fe35e9e9dbecf6e75c9b3aeadc17"


class SyntheticCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="atlas-forced-d-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.rows = synthetic()
        self.spec = spec()
        self.base, _ = gs.generate_dev_eval_pools(self.rows, self.spec)
        # Explicitly move groups that v1 would have put in E.
        self.forced = sorted(groups_of(self.base["lists"]["E"], self.rows))[:4]

    def pin(self, groups=None, raw=None):
        if raw is None:
            raw = json.dumps(self.forced if groups is None else groups).encode()
        path = self.root / "forced.json"
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def generate(self, groups=None, split_spec=None):
        path, sha = self.pin(groups)
        return gs.generate_dev_eval_pools(
            self.rows, split_spec or self.spec,
            forced_development_groups=gs.load_forced_development(path, sha),
            forced_development_sha256=sha)

    def audit(self, candidate):
        return gd.audit_global_pools(self.rows, gd._signed(candidate),
                                     expected_manifest_sha256=gd.global_manifest_sha256(self.rows, candidate))

    def strata(self):
        # Derive nearest-rank strata independently of generator helpers.
        members = {}
        for row in self.rows:
            members.setdefault(row["group"], []).append(row)
        sizes = sorted(len(value) for value in members.values())
        cuts = [sizes[math.ceil(k * len(sizes) / 4) - 1] for k in (1, 2, 3)]
        strata = {}
        for group, rows in members.items():
            key = (any(row["label"] == 1 for row in rows), sum(len(rows) > c for c in cuts))
            strata.setdefault(key, []).append(group)
        return strata


class ForcedDevelopmentTests(SyntheticCase):
    def test_default_document_matches_pre_change_golden_bytes(self):
        self.assertNotIn("forced_development", self.base)
        self.assertEqual(H(self.base), GOLDEN_CANDIDATE)
        self.assertEqual(gd.global_manifest_sha256(self.rows, self.base), GOLDEN_MANIFEST)
        out = self.root / "sealed.json"
        gd.seal_global_pools(self.rows, self.base, expected_manifest_sha256=GOLDEN_MANIFEST, output=out)
        self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(), GOLDEN_SIGNED_BYTES)
        again, _ = gs.generate_dev_eval_pools(list(reversed(self.rows)), self.spec)
        self.assertEqual(again, self.base)

    def test_forced_groups_precede_draw_and_preserve_every_stratum_quota(self):
        candidate, report = self.generate()
        actual = groups_of(candidate["lists"]["D"], self.rows)
        self.assertTrue(set(self.forced) <= actual)
        for groups in self.strata().values():
            quota = math.floor(Fraction("0.2") * len(groups) + Fraction(1, 2))
            pinned = set(groups) & set(self.forced)
            remainder = sorted(set(groups) - pinned,
                               key=lambda g: (H([gs.FORCED_ALGORITHM, self.spec["seeds"]["development"], "DE", g]), g))
            self.assertEqual(actual & set(groups), pinned | set(remainder[:quota - len(pinned)]))
            self.assertEqual(len(actual & set(groups)), quota)
        self.assertEqual(candidate["forced_development"],
                         {"sha256": self.pin()[1], "count": len(self.forced),
                          "groups_canonical_sha256": H(sorted(self.forced))})
        self.assertEqual(candidate["partition_provenance"]["algorithm"], gs.FORCED_ALGORITHM)
        self.assertEqual(report["algorithm"], gs.FORCED_ALGORITHM)
        self.assertEqual(report["forced_development"], candidate["forced_development"])
        self.assertNotIn("forced_development_groups", report)
        for group in self.forced:
            self.assertNotIn(group, json.dumps(report))
        self.assertTrue(self.audit(candidate)["structural_valid"])
        partition = [candidate["lists"]["D_fit"], candidate["lists"]["D_select"]]
        partition += [fold["test_ids"] for fold in candidate["evaluation_folds"]]
        seen = set()
        for ids in partition:
            groups = groups_of(ids, self.rows)
            self.assertFalse(groups & seen)
            seen |= groups
        self.assertEqual(seen, {r["group"] for r in self.rows})
        self.assertEqual(sorted(i for f in candidate["evaluation_folds"] for i in f["test_ids"]), candidate["lists"]["E"])
        sizes = [len(groups_of(f["test_ids"], self.rows)) for f in candidate["evaluation_folds"]]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        path, sha = self.pin(list(reversed(self.forced)))
        reordered, _ = gs.generate_dev_eval_pools(list(reversed(self.rows)), self.spec,
                                                 forced_development_groups=gs.load_forced_development(path, sha),
                                                 forced_development_sha256=sha)
        self.assertEqual(reordered["lists"], candidate["lists"])
        self.assertEqual(reordered["evaluation_folds"], candidate["evaluation_folds"])

    def test_canonical_groups_hash_ignores_file_order_and_whitespace(self):
        candidate, report = self.generate()
        raw = json.dumps(list(reversed(self.forced)), indent=2).encode() + b"\n"
        path, sha = self.pin(raw=raw)
        reordered, reordered_report = gs.generate_dev_eval_pools(
            self.rows, self.spec, forced_development_groups=gs.load_forced_development(path, sha),
            forced_development_sha256=sha)
        expected = H(sorted(self.forced))
        self.assertNotEqual(candidate["forced_development"]["sha256"], sha)
        for pools, public in ((candidate, report), (reordered, reordered_report)):
            self.assertEqual(pools["forced_development"]["groups_canonical_sha256"], expected)
            self.assertEqual(public["forced_development"]["groups_canonical_sha256"], expected)
            self.assertTrue(self.audit(pools)["structural_valid"])

    def test_same_count_private_list_tampering_is_rejected_after_resigning(self):
        candidate, _ = self.generate()
        original = copy.deepcopy(candidate["forced_development"])
        replacement = sorted(groups_of(candidate["lists"]["D"], self.rows) - set(self.forced))[0]
        candidate["forced_development_groups"] = sorted(self.forced[1:] + [replacement])
        self.assertEqual(len(candidate["forced_development_groups"]), original["count"])
        self.assertEqual(candidate["forced_development"], original)
        # Re-sign both outer identities: rejection must come from the canonical pin.
        with self.assertRaisesRegex(ValueError, "^forced_development_hash_mismatch:.*canonical SHA256"):
            self.audit(candidate)

    def test_forced_groups_can_land_in_fit_or_select_using_selection_seed(self):
        # Pin exactly each stratum's quota: all of D is forced, including fit/select.
        forced = []
        for groups in self.strata().values():
            quota = math.floor(Fraction("0.2") * len(groups) + Fraction(1, 2))
            forced.extend(sorted(groups)[:quota])
        candidate, _ = self.generate(forced)
        d = groups_of(candidate["lists"]["D"], self.rows)
        select = groups_of(candidate["lists"]["D_select"], self.rows)
        self.assertEqual(d, set(forced))
        self.assertTrue(select)
        self.assertTrue(d - select)
        for groups in self.strata().values():
            in_d = d & set(groups)
            ordered = sorted(in_d, key=lambda g: (H([gs.FORCED_ALGORITHM, self.spec["seeds"]["selection"], "D-select", g]), g))
            quota = math.floor(Fraction("0.3") * len(in_d) + Fraction(1, 2))
            self.assertEqual(select & in_d, set(ordered[:quota]))
        other_spec = copy.deepcopy(self.spec)
        other_spec["seeds"]["selection"] = 17
        changed, _ = self.generate(forced, other_spec)
        self.assertEqual(changed["lists"]["D"], candidate["lists"]["D"])
        self.assertNotEqual(changed["lists"]["D_select"], candidate["lists"]["D_select"])
        self.assertEqual(changed["evaluation_folds"], candidate["evaluation_folds"])

    def test_each_stratum_quota_fails_explicitly_without_fallback(self):
        groups = next(iter(self.strata().values()))
        quota = math.floor(Fraction("0.2") * len(groups) + Fraction(1, 2))
        with self.assertRaisesRegex(ValueError, "^forced_development_quota_exceeded:.*no fallback"):
            self.generate(sorted(groups)[:quota + 1])
        # The half-up zero-quota boundary cannot silently promote a forced group.
        with self.assertRaisesRegex(ValueError, "^forced_development_quota_exceeded:"):
            gs._take({(0, 0): ["synthetic-a", "synthetic-b"]}, 0.2, 1, "DE", forced={"synthetic-a"})

    def test_unknown_groups_and_invalid_lists_fail(self):
        with self.assertRaisesRegex(ValueError, "^forced_development_unknown_group:"):
            self.generate(["synthetic-missing-group"])
        for value in ([], {}, "pt0000", [1], [True], [None], [""], [" pt0000"], ["pt0000", "pt0000"]):
            with self.subTest(value=value):
                path, sha = self.pin(value)
                with self.assertRaisesRegex(ValueError, "^forced_development_invalid:"):
                    gs.load_forced_development(path, sha)
        for raw in (b"not JSON", b"[", b"\xff"):
            path, sha = self.pin(raw=raw)
            with self.assertRaisesRegex(ValueError, "^forced_development_invalid:"):
                gs.load_forced_development(path, sha)
        with self.assertRaisesRegex(ValueError, "^forced_development_invalid:"):
            gs.load_forced_development(self.root / "missing", "a" * 64)

    def test_file_sha_is_required_and_checks_exact_bytes(self):
        path, sha = self.pin()
        for invalid in (None, "", "a" * 63, "G" * 64, "A" * 64):
            with self.assertRaisesRegex(ValueError, "^forced_development_invalid:"):
                gs.load_forced_development(path, invalid)
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "^forced_development_hash_mismatch:"):
            gs.load_forced_development(path, sha)
        for kwargs in ({"forced_development_groups": self.forced},
                       {"forced_development_sha256": sha},
                       {"forced_development_groups": [], "forced_development_sha256": sha}):
            with self.assertRaisesRegex(ValueError, "^forced_development_invalid:"):
                gs.generate_dev_eval_pools(self.rows, self.spec, **kwargs)

    def test_forcing_does_not_relax_preregistered_thresholds(self):
        strict = copy.deepcopy(self.spec)
        strict["thresholds"]["min_select_positive_groups"] = 500
        with self.assertRaisesRegex(ValueError, "^split_audit_failed:"):
            self.generate(split_spec=strict)

    def test_forced_metadata_and_private_ids_are_bound_to_manifest(self):
        candidate, _ = self.generate()
        before = gd.global_manifest_sha256(self.rows, candidate)
        variants = []
        for key, value in (("sha256", "a" * 64), ("count", len(self.forced) + 1),
                           ("groups_canonical_sha256", "a" * 64)):
            other = copy.deepcopy(candidate)
            other["forced_development"][key] = value
            variants.append(other)
        other = copy.deepcopy(candidate)
        other["forced_development_groups"] = list(reversed(self.forced))
        variants.append(other)
        for other in variants:
            self.assertNotEqual(gd.global_manifest_sha256(self.rows, other), before)
            with self.assertRaises(ValueError):
                gd.audit_global_pools(self.rows, gd._signed(other), expected_manifest_sha256=before)

    def test_closed_extension_rejects_bad_count_algorithm_and_private_evidence(self):
        candidate, _ = self.generate()
        variants = []
        for key in ("forced_development", "forced_development_groups"):
            other = copy.deepcopy(candidate)
            del other[key]
            variants.append(other)
        other = copy.deepcopy(candidate)
        del other["forced_development"]["groups_canonical_sha256"]
        variants.append(other)
        for value in (None, "", "a" * 63, "A" * 64, "a" * 64):
            other = copy.deepcopy(candidate)
            other["forced_development"]["groups_canonical_sha256"] = value
            variants.append(other)
        for value in (True, 0, len(self.forced) + 1):
            other = copy.deepcopy(candidate)
            other["forced_development"]["count"] = value
            variants.append(other)
        other = copy.deepcopy(candidate)
        other["partition_provenance"]["algorithm"] = gs.ALGORITHM
        variants.append(other)
        other = copy.deepcopy(candidate)
        other["forced_development_groups"] = ["synthetic-unknown"]
        other["forced_development"]["count"] = 1
        variants.append(other)
        for other in variants:
            with self.subTest(other=other.get("forced_development")):
                with self.assertRaises(ValueError):
                    self.audit(other)


class ForcedCliTests(SyntheticCase):
    def cli(self, argv, status=0):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            actual = cli.main(argv)
        self.assertEqual(actual, status, err.getvalue())
        self.assertFalse(out.getvalue() if status else err.getvalue())
        return json.loads(err.getvalue())["error"] if status else json.loads(out.getvalue())

    def base_args(self, command):
        return [command, "--data-root", str(self.root / "data"), "--source", "unused-source",
                "--dataset", self.spec["dataset"]["id"]]

    def make_args(self):
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        return self.base_args("make-pools") + ["--spec", str(spec_path), "--code", "unused-code",
                "--acquisition-config", "unused-config", "--split-amendment", "unused-amendment",
                "--acquisition-root", str(self.root / "data")]

    def prepare_make(self):
        # Acquisition / Git provenance checks are covered by their existing suites.
        # Here real JSON pinning, generator, publishing, import and audit run offline.
        patches = {
            "_clean_head": {"return_value": "a" * 40},
            "_tracked": {"return_value": ({}, "b" * 64)},
            "_acquisition_spec": {"return_value": ({"files": {"manifest.csv": "c" * 64}}, self.spec)},
            "_source": {"return_value": {"manifest": "manifest/manifest.csv", "manifest_sha256": "c" * 64}},
            "_records": {"return_value": ({}, self.rows, {})},
            "_git": {"return_value": "a" * 40},
        }
        for name, kwargs in patches.items():
            mock = patch.object(cli, name, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch.object(gs, "check_amendment")
        mock.start()
        self.addCleanup(mock.stop)

    def test_cli_requires_paired_flags_and_never_publishes_invalid_inputs(self):
        path, sha = self.pin()
        args = self.make_args()
        for extra, code in ((["--force-development-groups", str(path)], "forced_development_invalid:"),
                            (["--force-development-sha256", sha], "forced_development_invalid:"),
                            (["--force-development-groups", str(path), "--force-development-sha256", "a" * 64],
                             "forced_development_hash_mismatch:")):
            with patch.object(gd, "_publish") as publish:
                self.assertTrue(self.cli(args + extra, status=2).startswith(code))
                publish.assert_not_called()
        self.assertFalse((self.root / "data").exists())

    def test_cli_unknown_empty_and_over_quota_do_not_publish(self):
        self.prepare_make()
        args = self.make_args()
        groups = next(iter(self.strata().values()))
        quota = math.floor(Fraction("0.2") * len(groups) + Fraction(1, 2))
        cases = [([], "forced_development_invalid:"),
                 (["synthetic-absent"], "forced_development_unknown_group:"),
                 (sorted(groups)[:quota + 1], "forced_development_quota_exceeded:")]
        for forced, code in cases:
            path, sha = self.pin(forced)
            with patch.object(gd, "_publish") as publish:
                error = self.cli(args + ["--force-development-groups", str(path),
                                        "--force-development-sha256", sha], status=2)
                self.assertTrue(error.startswith(code), error)
                publish.assert_not_called()
                for group in forced:
                    self.assertNotIn(group, error)

    def test_cli_without_option_retains_original_candidate_golden(self):
        self.prepare_make()
        made = self.cli(self.make_args())
        candidate = json.loads(Path(made["path"]).read_bytes())
        self.assertEqual(H(candidate), GOLDEN_CANDIDATE)
        self.assertNotIn("forced_development", made)
        self.assertEqual(made["global_manifest_sha256"], GOLDEN_MANIFEST)

    def test_cli_make_import_audit_anchor_and_restart(self):
        self.prepare_make()
        path, sha = self.pin()
        args = self.make_args() + ["--force-development-groups", str(path), "--force-development-sha256", sha]
        made = self.cli(args)
        candidate_bytes = Path(made["path"]).read_bytes()
        self.assertEqual(self.cli(args), made)
        self.assertEqual(Path(made["path"]).read_bytes(), candidate_bytes)
        candidate = json.loads(candidate_bytes)
        self.assertEqual(candidate["forced_development_groups"], self.forced)
        for group in self.forced:
            self.assertNotIn(group, json.dumps(made))
        imported = self.cli(self.base_args("import-pools") + ["--file", made["path"],
                            "--expected-manifest", made["global_manifest_sha256"]])
        pool_sha = imported["pools_sha256"]
        self.assertEqual(pool_sha, made["candidate_sha256"])
        audit = self.cli(self.base_args("audit-pools") + ["--pools-sha256", pool_sha,
                         "--expected-manifest", made["global_manifest_sha256"]])
        self.assertTrue(audit["structural_valid"])
        queue = Mock()
        queue.publish_anchor.return_value = {"path": "anchors/synthetic.json", "commit": "d" * 40}
        with patch.object(cli, "_queue", return_value=queue):
            self.cli(self.base_args("anchor") + ["--pools-sha256", pool_sha, "--kind", "pools",
                     "--document-sha256", pool_sha, "--expected-manifest", made["global_manifest_sha256"],
                     "--document-utc", "2026-10-28T12:00:00Z", "--code", "unused", "--remote", "unused",
                     "--checkout", "unused"])
        anchor = queue.publish_anchor.call_args.args[0]
        self.assertEqual(anchor["document_sha256"], pool_sha)
        for group in self.forced:
            self.assertNotIn(group, json.dumps(anchor))

    def test_audit_and_anchor_reject_same_count_private_list_tampering(self):
        candidate, _ = self.generate()
        replacement = sorted(groups_of(candidate["lists"]["D"], self.rows) - set(self.forced))[0]
        candidate["forced_development_groups"] = sorted(self.forced[1:] + [replacement])
        signed = gd._signed(candidate)
        manifest = gd.global_manifest_sha256(self.rows, signed)
        with patch.object(cli, "_pools", return_value=({}, self.rows, {}, signed)), patch.object(cli, "_queue") as queue:
            for args in (self.base_args("audit-pools") + ["--pools-sha256", signed["sha256"],
                         "--expected-manifest", manifest],
                         self.base_args("anchor") + ["--pools-sha256", signed["sha256"],
                         "--kind", "pools", "--document-sha256", signed["sha256"], "--expected-manifest", manifest,
                         "--document-utc", "2026-10-28T12:00:00Z", "--code", "unused", "--remote", "unused",
                         "--checkout", "unused"]):
                error = self.cli(args, status=2)
                self.assertTrue(error.startswith("forced_development_hash_mismatch:"), error)
                for group in candidate["forced_development_groups"]:
                    self.assertNotIn(group, error)
            queue.assert_not_called()

    def test_audit_and_anchor_reject_resigned_document_with_forced_group_in_e(self):
        candidate, _ = self.generate()
        e_group = sorted(groups_of(candidate["lists"]["E"], self.rows))[0]
        candidate["forced_development_groups"] = [e_group]
        candidate["forced_development"]["count"] = 1
        candidate["forced_development"]["groups_canonical_sha256"] = H([e_group])
        signed = gd._signed(candidate)
        manifest = gd.global_manifest_sha256(self.rows, signed)
        # Internally consistent hashes are insufficient: even a newly approved
        # metadata pin cannot bypass the required group-in-D invariant.
        with self.assertRaisesRegex(ValueError, "^forced_development_not_in_d:"):
            gd.audit_global_pools(self.rows, signed, expected_manifest_sha256=manifest)
        with patch.object(cli, "_pools", return_value=({}, self.rows, {}, signed)), patch.object(cli, "_queue") as queue:
            self.assertTrue(self.cli(self.base_args("audit-pools") + ["--pools-sha256", signed["sha256"],
                            "--expected-manifest", manifest], status=2).startswith("forced_development_not_in_d:"))
            self.assertTrue(self.cli(self.base_args("anchor") + ["--pools-sha256", signed["sha256"],
                            "--kind", "pools", "--document-sha256", signed["sha256"], "--expected-manifest", manifest,
                            "--document-utc", "2026-10-28T12:00:00Z", "--code", "unused", "--remote", "unused",
                            "--checkout", "unused"], status=2).startswith("forced_development_not_in_d:"))
            queue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
