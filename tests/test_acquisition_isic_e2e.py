"""End-to-end tests of ``python -m augment_atlas.acquisition`` against a local multi-object server.

A synthetic ZIP64 release (directory entry, junk members, stored + deflate, data descriptors)
and three synthetic CSVs are served from 127.0.0.1 only. Derivation uses the Pillow-free
``fake_derive`` unless a test says otherwise; abrupt kills run the real CLI in a subprocess.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquisition_e2e_helpers import (GIB, ZIP_DIR, FreeSpace, MultiObjectServer, SyntheticRelease,  # noqa: E402
                                     fake_derive, make_config)

from augment_atlas import data  # noqa: E402
from augment_atlas.acquisition import cli, config as config_mod, pipeline  # noqa: E402
from augment_atlas.acquisition.diskguard import ProcessLock  # noqa: E402

RUNNER = Path(__file__).resolve().parent / "acquisition_cli_runner.py"
ID_PATTERNS = (r"SYNTH_\d{7}", r"PSYNTH_\d{4}", r"LSYNTH_\d{7}", r"isic20-g-[0-9a-f]{16}")


class AcquisitionE2E(unittest.TestCase):
    n_patients = 40

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.root = self.dir / "data" / "isic2020"
        self.release = self.make_release()
        self.server = None
        self.serve(self.release.objects())
        self.config_path = self.dir / "config.json"
        self.write_config()

    def make_release(self):
        return SyntheticRelease(n_patients=self.n_patients)

    def serve(self, objects):
        if self.server is not None:
            self.server.httpd.shutdown()
            self.server.httpd.server_close()
        self.server = MultiObjectServer(objects)
        self.server.thread.start()
        self.addCleanup(self._stop, self.server)

    @staticmethod
    def _stop(server):
        server.httpd.shutdown()
        server.httpd.server_close()

    def write_config(self, objects=None, **overrides):
        self.raw = make_config(self.server, self.release, self.config_path, objects=objects, **overrides)

    def deps(self, **kwargs):
        defaults = dict(deriver=fake_derive, free_fn=FreeSpace(), sleep=lambda _s: None, log=io.StringIO(),
                        tool_commit="0" * 40)
        defaults.update(kwargs)
        return pipeline.Deps(**defaults)

    def cli(self, *args, deps=None):
        out, err = io.StringIO(), io.StringIO()
        deps = deps or self.deps()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main([*args, "--config", str(self.config_path), "--root", str(self.root)], deps)
        self.last_stdout, self.last_stderr = out.getvalue(), err.getvalue() + deps.log.getvalue()
        return code

    def subprocess_cli(self, *args, env=None):
        environment = {**os.environ, **(env or {})}
        return subprocess.run([sys.executable, str(RUNNER), *args, "--config", str(self.config_path),
                               "--root", str(self.root)], env=environment, capture_output=True, text=True, timeout=300)

    # -- inspection helpers -------------------------------------------------------
    def ledger_counts(self):
        with contextlib.closing(sqlite3.connect(self.root / "acquisition-v1" / "ledger.sqlite")) as conn:
            return dict(conn.execute("SELECT state, COUNT(*) FROM member GROUP BY state").fetchall())

    def sessions(self):
        with contextlib.closing(sqlite3.connect(self.root / "acquisition-v1" / "ledger.sqlite")) as conn:
            return conn.execute("SELECT mode, exit, start_position, end_position FROM session ORDER BY id").fetchall()

    def leftovers(self):
        return sorted(p.name for p in self.root.rglob("*") if p.is_file() and p.name.endswith((".tmp", ".part")))

    def stream_starts(self):
        """Start offsets of the ZIP body ranges (those that end right before the central directory)."""
        cd_offset = json.loads((self.root / "acquisition-v1" / "probe.json").read_text())["sources"]["images_zip"]["cd_offset"]
        starts = []
        for value in self.server.ranged_gets("images_zip"):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", value)
            if match and int(match.group(2)) == cd_offset - 1:
                starts.append(int(match.group(1)))
        return starts

    def labelled(self):
        return len(self.release.rows) - len(self.release.drop_from_zip)

    def assert_ready(self):
        manifest = self.root / "manifest"
        self.assertTrue((manifest / "READY.json").exists())
        ready = json.loads((manifest / "READY.json").read_text())
        for name, digest in ready["files"].items():
            self.assertEqual(data.file_hash(manifest / name), digest, name)
        self.assertEqual(list((self.root / "acquisition-v1" / "spool").iterdir()), [])
        self.assertEqual(self.leftovers(), [])
        return ready

    def assert_public_summary_has_no_identifiers(self):
        text = (self.root / "public" / "isic2020-summary-public.json").read_text(encoding="utf-8")
        private = json.loads((self.root / "manifest" / "manifest-private.json").read_text())
        identifiers = set(self.release.identifiers())
        identifiers |= set(private["groups"]["patient_to_group"].values())
        identifiers |= {f"{ZIP_DIR}{row['image_name']}.jpg" for row in self.release.rows}
        self.assertTrue(identifiers)
        self.assertEqual([item for item in identifiers if item in text], [])
        for pattern in ID_PATTERNS:
            self.assertIsNone(re.search(pattern, text), pattern)
        for fragment in ("derived-512-v1/", "/images/", ".jpg", "patient_to_group", "patient_id", "lesion"):
            self.assertNotIn(fragment, text)
        document = json.loads(text)
        self.assertEqual((document["privacy_lint"]["pattern_hits"], document["privacy_lint"]["id_hits"]), (0, 0))
        return document


class HappyPathTests(AcquisitionE2E):
    def test_probe_then_full_writes_ready_manifest_and_id_free_summary(self):
        self.assertEqual(self.cli("probe"), 0, self.last_stderr)
        probe = json.loads(self.last_stdout)["result"]
        self.assertEqual(probe["divergences"], [])
        self.assertEqual(probe["observed"]["zip_jpg_members"], len(self.release.rows))
        self.assertEqual((probe["observed"]["zip_dirs"], probe["observed"]["zip_junk_members"]), (1, 2))
        self.assertFalse((self.root / "acquisition-v1" / "ledger.sqlite").exists(), "probe never streams the ZIP body")
        self.assertEqual(self.stream_starts(), [])

        self.assertEqual(self.cli("full"), 0, self.last_stderr)
        self.assert_ready()
        self.assertEqual(self.ledger_counts(), {"DERIVED": len(self.release.rows), "SKIPPED": 3})
        self.assertEqual(len(self.stream_starts()), 1, "one session streams the body exactly once")
        result = json.loads(self.last_stdout)["result"]
        self.assertEqual(result["duplicates"], {"official_pairs": 2, "exact_hash_pairs": 2, "cross_patient_pairs": 1,
                                                "label_conflict_pairs": 0, "near_pairs_reported": None,
                                                "images_excluded_by_policy": 2})
        self.assertEqual(result["counts"]["patients"], self.n_patients)
        self.assertEqual(result["counts"]["groups"], self.n_patients - 1, "the cross-patient pair merges two patients")
        self.assertEqual(result["counts"]["images_included"], len(self.release.rows) - 2)
        self.assertTrue(result["splits_passed"])
        for pattern in ID_PATTERNS:
            self.assertIsNone(re.search(pattern, self.last_stdout), pattern)

        # manifest.csv is the existing manifest_csv contract (hash-pinned, per-image sha256)
        cfg = {"dataset": {"adapter": "manifest_csv", "root": str(self.root), "manifest": "manifest/manifest.csv",
                           "manifest_sha256": data.file_hash(self.root / "manifest" / "manifest.csv"),
                           "require_image_hashes": True},
               "model": {"num_classes": 2}}
        records = data.load_records(cfg)
        self.assertEqual(len(records), len(self.release.rows) - 2)
        records_json = json.loads((self.root / "manifest" / "records.json").read_text())
        self.assertEqual(sorted(r["path"] for r in records_json["records"]), sorted(r["path"] for r in records))
        private = json.loads((self.root / "manifest" / "manifest-private.json").read_text())
        statuses = {r["status"] for r in private["records"]}
        self.assertEqual(statuses, {"included", "excluded"})
        self.assertEqual(private["dataset"]["doi"], "10.34970/2020-ds01")
        self.assertEqual(private["preprocessing"]["id"], "isic2020-derived-512-v1")
        splits_dir = self.root / "splits" / "global-dev-eval-v1"
        self.assertEqual(sorted(p.name for p in splits_dir.iterdir()), ["audit.json", "folds.json", "pools.json"])
        summary = self.assert_public_summary_has_no_identifiers()
        self.assertEqual(summary["splits"]["split_seed"], self.raw["splits"]["split_seed"])

        manifest_hash = data.file_hash(self.root / "manifest" / "manifest.csv")
        pools_hash = data.file_hash(splits_dir / "pools.json")
        self.assertEqual(self.cli("verify"), 0, self.last_stderr)
        self.assertEqual(data.file_hash(self.root / "manifest" / "manifest.csv"), manifest_hash)
        self.assertEqual(data.file_hash(splits_dir / "pools.json"), pools_hash)

    def test_canary_limit_then_full_resumes_at_the_first_pending_member(self):
        self.assertEqual(self.cli("canary", "--limit", "10"), 0, self.last_stderr)
        brief = json.loads(self.last_stdout)["result"]
        self.assertEqual(brief["derived"], 10)
        self.assertEqual(brief["limit"], 10)
        self.assertEqual(brief["projection"]["remaining"], len(self.release.rows) - 10)
        self.assertGreater(brief["projection"]["projected_bytes"], brief["derived_bytes"])
        self.assertEqual(brief["timing"]["session_derived"], 10)
        canary = json.loads((self.root / "acquisition-v1" / "canary.json").read_text())
        self.assertEqual(canary["counts"]["DERIVED"], 10)
        self.assertEqual(len(list((self.root / "derived-512-v1" / "images").iterdir())), 10)
        self.assertEqual(self.leftovers(), [])
        self.assertFalse((self.root / "manifest" / "READY.json").exists())

        self.assertEqual(self.cli("verify"), 75, "verify refuses an incomplete acquisition")
        self.assertEqual(self.cli("canary", "--limit", "10"), 0, "re-running the canary streams nothing new")
        self.assertEqual(len(self.stream_starts()), 1)

        self.assertEqual(self.cli("full"), 0, self.last_stderr)
        self.assert_ready()
        starts = self.stream_starts()
        self.assertEqual(len(starts), 2)
        self.assertGreater(starts[1], starts[0], "full resumes after the canary instead of re-reading the ZIP")
        self.assertEqual([s[0] for s in self.sessions()], ["canary", "canary", "full"])
        self.assertEqual(self.ledger_counts()["DERIVED"], len(self.release.rows))

    def test_transport_drop_mid_member_resumes_inside_the_session(self):
        # Large stored payloads: the ZIP must be bigger than the 128 KiB EOCD suffix, otherwise the
        # probe's suffix request would consume the injected drops before the body is streamed.
        self.release = SyntheticRelease(n_patients=self.n_patients,
                                        jpeg_bytes=lambda i: hashlib.sha256(str(i).encode()).digest() * 200)
        self.serve(self.release.objects())
        self.write_config()
        size = len(self.server.state("images_zip").data)
        drops = [size // 5, size // 2]
        self.assertGreater(size - 131072, max(drops))
        self.server.state("images_zip").drop_offsets = list(drops)
        self.assertEqual(self.cli("full"), 0, self.last_stderr)
        self.assert_ready()
        starts = self.stream_starts()
        self.assertEqual(len(starts), 3)
        self.assertEqual(starts, sorted(starts))

    def test_two_workers_with_tiny_spool_apply_backpressure(self):
        self.write_config(transfer={"workers": 2}, disk={"spool_cap_gib": 2400 / GIB})
        os.environ["ACQ_TEST_DERIVE_DELAY"] = "0.02"   # inherited by the pool workers
        self.addCleanup(os.environ.pop, "ACQ_TEST_DERIVE_DELAY", None)
        self.assertEqual(self.cli("full"), 0, self.last_stderr)
        self.assert_ready()
        self.assertGreater(len(self.stream_starts()), 3, "the connection is closed and reopened after each pause")
        self.assertEqual(self.ledger_counts()["DERIVED"], len(self.release.rows))


class KillAndResumeTests(AcquisitionE2E):
    def test_abrupt_kill_during_derivation_then_resume(self):
        log = self.dir / "calls.log"
        killed = self.subprocess_cli("full", env={"ACQ_TEST_KILL_AFTER": "25", "ACQ_TEST_CALL_LOG": str(log)})
        self.assertEqual(killed.returncode, 137, killed.stderr)
        self.assertEqual(self.ledger_counts(), {"DERIVED": 24, "FETCHED": 1, "PENDING": len(self.release.rows) - 25,
                                                "SKIPPED": 3})
        self.assertEqual(len(self.leftovers()), 1, "the kill leaves one partial derived file behind")
        self.assertEqual(self.sessions()[0][1], None, "a killed session has no recorded exit")

        self.assertEqual(self.cli("full"), 0, self.last_stderr)
        self.assert_ready()
        self.assertIn("recovery: removed 1", self.last_stderr)
        starts = self.stream_starts()
        self.assertEqual(len(starts), 2)
        self.assertGreater(starts[1], 0)
        self.assertEqual(self.ledger_counts()["DERIVED"], len(self.release.rows))

    def test_abrupt_kill_at_the_disk_guard_then_resume_with_two_workers(self):
        self.write_config(transfer={"workers": 2})
        killed = self.subprocess_cli("full", env={"ACQ_TEST_FREE_KILL_AFTER": "60"})
        self.assertEqual(killed.returncode, 137, killed.stderr)
        counts = self.ledger_counts()
        self.assertGreater(counts.get("PENDING", 0), 0)
        self.assertEqual(self.cli("full"), 0, self.last_stderr)
        self.assert_ready()
        self.assertEqual(self.ledger_counts()["DERIVED"], len(self.release.rows))

    def test_sigterm_style_interruption_is_a_clean_pause(self):
        calls = {"n": 0}

        def interrupting(src, dst, params):
            calls["n"] += 1
            if calls["n"] == 7:
                from augment_atlas.acquisition.errors import Interrupted
                raise Interrupted("simulated SIGTERM")
            return fake_derive(src, dst, params)

        self.assertEqual(self.cli("full", deps=self.deps(deriver=interrupting)), 75)
        self.assertEqual(self.sessions()[-1][1], 75)
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(self.cli("full"), 0, self.last_stderr)
        self.assert_ready()


class DiskGuardTests(AcquisitionE2E):

    def test_insufficient_free_space_refuses_before_writing_anything(self):
        free = 20 * GIB + 8 * GIB + 2 * GIB + int(1.5 * GIB) - 1   # reserve + derived cap + spool cap + margin - 1 B
        self.assertEqual(self.cli("canary", deps=self.deps(free_fn=FreeSpace(plenty=free))), 75)
        self.assertIn("precheck", self.last_stderr)
        self.assertFalse(self.root.exists(), "nothing is written when the precheck fails")
        self.assertEqual(self.server.state("images_zip").requests, [])

    def test_probe_refuses_below_the_reserve(self):
        self.assertEqual(self.cli("probe", deps=self.deps(free_fn=FreeSpace(plenty=20 * GIB + GIB))), 75)
        self.assertFalse(self.root.exists())

    def test_guard_pauses_mid_run_without_temporaries_and_resume_completes(self):
        guard = FreeSpace(plenty=40 * GIB, low=21 * GIB - 1, fail_after=20)
        self.assertEqual(self.cli("full", deps=self.deps(free_fn=guard)), 75)
        self.assertIn("DiskPaused", self.last_stderr)
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(self.sessions()[-1][1], 75)
        counts = self.ledger_counts()
        self.assertGreater(counts.get("DERIVED", 0), 0)
        self.assertGreater(counts.get("PENDING", 0), 0)
        self.assertEqual(self.cli("full"), 0, self.last_stderr)
        self.assert_ready()
        self.assertEqual(self.ledger_counts()["DERIVED"], len(self.release.rows))

    def test_resume_precheck_only_requires_what_is_left(self):
        cfg = config_mod.load(self.config_path)
        runner = pipeline.Runner(cfg, pipeline.Layout(self.root, cfg), self.deps(), "full", 500, True)
        self.assertEqual(runner.needed_bytes(0, 0), 8 * GIB + 2 * GIB + int(1.5 * GIB))
        self.assertEqual(runner.needed_bytes(5 * GIB, GIB), 3 * GIB + GIB + int(1.5 * GIB))
        self.assertEqual(runner.needed_bytes(9 * GIB, 3 * GIB), int(1.5 * GIB))


class DivergenceTests(AcquisitionE2E):
    n_patients = 12

    def test_header_divergence_is_exit_3_and_nothing_is_retained(self):
        swapped = ["image_name", "lesion_id", "patient_id", "sex", "age_approx", "anatom_site_general_challenge",
                   "diagnosis", "benign_malignant", "target"]
        objects = self.release.objects({"ground_truth_v2": self.release.csv_bytes("ground_truth_v2", header=swapped)})
        self.serve(objects)
        self.write_config(objects=objects)
        self.assertEqual(self.cli("probe"), 3)
        self.assertIn("ADR-A8", self.last_stderr)
        self.assertFalse((self.root / "release" / "SYNTH_Training_GroundTruth_v2.csv").exists())
        self.assertEqual(self.cli("canary"), 3)
        self.assertEqual(self.stream_starts() if (self.root / "acquisition-v1" / "probe.json").exists() else [], [])

    def test_changed_remote_object_is_exit_3(self):
        self.raw["sources"]["duplicates"]["etag"] = "0" * 32
        self.config_path.write_text(json.dumps(self.raw), encoding="utf-8")
        self.assertEqual(self.cli("probe"), 3)
        self.assertIn("ETag changed", self.last_stderr)

        self.write_config()
        self.assertEqual(self.cli("canary", "--limit", "5"), 0, self.last_stderr)
        self.server.state("images_zip").etag = "f" * 32 + "-3"   # the object changed after acquisition started
        self.assertEqual(self.cli("full"), 3)
        self.assertIn("ETag changed", self.last_stderr)
        self.assertEqual(self.ledger_counts()["DERIVED"], 5)

    def test_changed_config_identity_after_streaming_started_is_exit_3(self):
        self.assertEqual(self.cli("canary", "--limit", "5"), 0, self.last_stderr)
        self.write_config(splits={"split_seed": 7})
        self.assertEqual(self.cli("full"), 3)
        self.assertIn("ledger pins config_identity_sha256", self.last_stderr)


class CountDivergenceTests(AcquisitionE2E):
    def make_release(self):
        release = SyntheticRelease(n_patients=self.n_patients)
        names = [row["image_name"] for row in release.rows]
        release.drop_from_zip = {names[-1]}                                 # a label without image
        release.zip_extra_members = [(f"{ZIP_DIR}SYNTH_0099999.jpg", b"x" * 300)]  # an image without label
        release.corrupt = {names[20]}                                       # the decoder refuses it
        return release

    def test_divergences_are_recorded_block_ready_until_accepted(self):
        documented = dict(self.release.counts(), melanoma_images=self.release.counts()["melanoma_images"] + 1)
        self.write_config(documented=documented)
        self.assertEqual(self.cli("probe"), 3)
        record = json.loads((self.root / "manifest" / "divergences.json").read_text())
        observed = {d["field"]: (d["documented"], d["observed"]) for d in record["divergences"]}
        self.assertEqual(observed, {"counts.melanoma_images": (documented["melanoma_images"], documented["melanoma_images"] - 1),
                                    "zip_csv.images_without_label": (0, 1), "zip_csv.labels_without_image": (0, 1)})
        self.assertTrue(all(d["accepted"] is False for d in record["divergences"]))

        self.assertEqual(self.cli("canary", "--limit", "5"), 3, "strict mode never streams with pending divergences")
        self.assertEqual(self.stream_starts(), [])
        self.assertEqual(self.cli("full", "--lenient"), 3, "lenient streams everything but READY stays blocked")
        self.assertFalse((self.root / "manifest" / "READY.json").exists())
        counts = self.ledger_counts()
        self.assertEqual(counts["DERIVED"], self.labelled() - 1)
        self.assertEqual(counts["SKIPPED"], 3 + 1 + 1)  # dir + 2 junk, unlabelled image, corrupt image
        fields = {d["field"] for d in json.loads((self.root / "manifest" / "divergences.json").read_text())["divergences"]}
        self.assertIn("derive.skipped_images", fields)

        accepted = [{"field": field, "observed": value, "resolution": "synthetic test decision", "decided_by": "test"}
                    for field, value in (("counts.melanoma_images", documented["melanoma_images"] - 1),
                                         ("zip_csv.images_without_label", 1), ("zip_csv.labels_without_image", 1),
                                         ("derive.skipped_images", 1))]
        self.write_config(documented=documented, accepted_divergences=accepted)
        self.assertEqual(self.cli("verify"), 0, self.last_stderr)
        self.assert_ready()
        summary = self.assert_public_summary_has_no_identifiers()
        self.assertEqual({d["field"] for d in summary["divergences"]}, {a["field"] for a in accepted})
        self.assertTrue(all(d["accepted"] for d in summary["divergences"]))
        private = json.loads((self.root / "manifest" / "manifest-private.json").read_text())
        statuses = sorted(r["status"] for r in private["records"] if r["status"] != "included")
        self.assertEqual(statuses, ["excluded", "excluded", "missing-image", "skipped"])


def _synthetic_jpeg(index):
    from PIL import Image
    width, height = (700 + index % 5, 600) if index % 2 else (600, 650)
    image = Image.new("RGB", (width, height), (index * 7 % 256, 90, 160))
    image.paste((255, 255 - index % 200, 0), (index % 50, 10, index % 50 + 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def _pillow_missing():
    try:
        import PIL  # noqa: F401
    except ImportError:
        return True
    return False


@unittest.skipIf(_pillow_missing(), "Pillow is not installed (the real deriver runs in the pinned container)")
class RealPillowTests(AcquisitionE2E):
    """The production path: real ``derive_member`` in a 2-process pool, through the CLI."""

    def make_release(self):
        return SyntheticRelease(n_patients=self.n_patients, jpeg_bytes=_synthetic_jpeg)

    def test_canary_and_full_with_the_real_deriver(self):
        self.write_config(transfer={"workers": 2})
        deps = self.deps(deriver=pipeline.derive.derive_member)
        self.assertEqual(self.cli("canary", "--limit", "12", deps=deps), 0, self.last_stderr)
        canary = json.loads((self.root / "acquisition-v1" / "canary.json").read_text())
        self.assertEqual(canary["derived"], 12)
        self.assertIn("pillow", canary["decoder"])
        self.assertEqual(self.cli("full", deps=self.deps(deriver=pipeline.derive.derive_member)), 0, self.last_stderr)
        self.assert_ready()
        from PIL import Image
        private = json.loads((self.root / "manifest" / "manifest-private.json").read_text())
        self.assertIsNotNone(private["acquisition"]["decoder"]["pillow"])
        for record in private["records"]:
            derived = record["derived"]
            self.assertEqual(min(derived["width"], derived["height"]), 512)
            with Image.open(self.root / derived["path"]) as image:
                self.assertEqual(image.size, (derived["width"], derived["height"]))
                self.assertEqual(pipeline.derive.pixel_sha256(image), derived["pixel_sha256"])
        self.assert_public_summary_has_no_identifiers()

    def test_module_invocation_as_on_the_server(self):
        """``python -m augment_atlas.acquisition full`` with default deps: real deriver, spawned
        workers (the guarded ``__main__``), real disk probe (reserve shrunk to fit any test disk)."""
        self.write_config(transfer={"workers": 2},
                          disk={"reserve_gib": 0, "derived_cap_gib": 0.01, "spool_cap_gib": 0.002,
                                "precheck_margin_gib": 0, "write_margin_gib": 0})
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
               "AUGMENT_ATLAS_TOOL_COMMIT": "1" * 40}
        result = subprocess.run([sys.executable, "-m", "augment_atlas.acquisition", "full", "--config",
                                 str(self.config_path), "--root", str(self.root)], cwd=self.dir, env=env,
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assert_ready()
        brief = json.loads(result.stdout)
        self.assertEqual(brief["result"]["counts"]["images_included"], len(self.release.rows) - 2)
        for pattern in ID_PATTERNS:
            self.assertIsNone(re.search(pattern, result.stdout), pattern)
        private = json.loads((self.root / "manifest" / "manifest-private.json").read_text())
        self.assertEqual(private["acquisition"]["tool_commit"], "1" * 40)


class UsageTests(AcquisitionE2E):
    n_patients = 3

    def test_usage_errors_and_lock(self):
        self.assertEqual(self.cli("canary", "--limit", "0"), cli.EXIT_USAGE)
        self.raw["transfer"]["workers"] = 9
        self.config_path.write_text(json.dumps(self.raw), encoding="utf-8")
        self.assertEqual(self.cli("probe"), cli.EXIT_USAGE)
        self.write_config()
        self.assertEqual(self.cli("verify"), 75, "verify without a ledger is incomplete")
        with ProcessLock(self.root / "acquisition-v1" / "lock"):
            self.assertEqual(self.cli("probe"), 75)
        self.assertIn("LockBusy", self.last_stderr)

    def test_module_entry_point_runs(self):
        result = subprocess.run([sys.executable, "-m", "augment_atlas.acquisition", "--help"], capture_output=True,
                                text=True, timeout=60, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("probe", "canary", "full", "verify"):
            self.assertIn(command, result.stdout)


if __name__ == "__main__":
    unittest.main()
