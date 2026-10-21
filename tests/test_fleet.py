"""F4: capacity-aware selection (docs/fleet-protocol.md sections 7.1 and 8.3, cases 3-6).

Node facts come from the REAL sanitized x299 capture (tests/fixtures/telemetry_v1/x299)
through the F1/F2 parsers. Values the capture does not contain are marked SYNTHETIC
where they are set: operator decisions of the profile (limits, reserves, class,
images, datasets, weights, compute capability), ``compute_apps`` and disk free bytes.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import types
import unittest

from augment_atlas import cli
from augment_atlas import fleet
from augment_atlas import fleet_schema as fs
from augment_atlas import node_profile as npf
from augment_atlas import telemetry_v1 as tv
from augment_atlas.queue import GitQueue, parse_record, render_record
from augment_atlas.runtime import DockerRuntime
from augment_atlas.schema import IDENTITY_FIELDS, new_record, prepare_job

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "telemetry_v1"
X299 = FIXTURES / "x299"
RECORDS = FIXTURES / "synthetic" / "queue-records"
NOW_UTC = "2026-10-19T12:38:41Z"
IMAGE = "sha256:" + "3" * 64          # same image as the fleet-v1 record fixture
OTHER_IMAGE = "sha256:" + "4" * 64
COMMIT = "a" * 40                     # code_commit of the record fixtures
GIB = 1024 ** 3
INBREAST_MANIFEST = "90b7e97a8c2dfbb804b6b0dd486047ea406e86270f6b9b46e1fa90c55eede178"
WEIGHTS = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
BOOT_ID = (X299 / "proc" / "sys" / "kernel" / "random" / "boot_id").read_text(encoding="utf-8").strip()
MONO = 86400.5


def x299_profile():
    """Probe of the real capture + the operator decisions of section 6.2 (SYNTHETIC values)."""
    profile = npf.probe(tv.ReplayReaders(X299), "x299", NOW_UTC)
    profile["status"] = "active"
    profile["hardware_class"] = "gm200-12g"
    profile["guardian"] = {"kind": "private-v2", "required_for_claim": True}
    for gpu in profile["hardware"]["gpus"]:
        gpu["compute_capability"] = [5, 2]
    profile["storage"] = [{"key": "runs", "reserve_bytes": 20 * GIB}, {"key": "data", "reserve_bytes": 0}]
    profile["limits"].update(
        max_concurrent_trainings=1, allowed_gpus=[0, 1], min_idle_seconds_between_jobs=60,
        container={"cpus": 2, "memory_bytes": 4294967296, "shm_bytes": 268435456},
        thermal=[{"sensor": "gpu.temp_c", "ceiling_c": 90, "preventive_c": 89, "claim_max_c": 75, "source": "nvidia-smi"},
                 {"sensor": "cpu_pkg.temp_c", "ceiling_c": 90, "preventive_c": 85, "claim_max_c": 75,
                  "source": "hwmon:coretemp"}])
    profile["images"] = [{"image_id": IMAGE}]
    profile["datasets"] = [
        {"id": "synthetic-tiny", "version": "*", "manifest_sha256": None, "storage_key": None},
        {"id": "inbreast-pilot-endurance", "version": "inbreast-existing-classification-integrity-20261016",
         "manifest_sha256": INBREAST_MANIFEST, "storage_key": "data"}]
    profile["weights"] = [{"file": "resnet50-11ad3fa6.pth", "sha256": WEIGHTS}]
    decided = ("guardian", "hardware_class", "limits.", "storage[", "hardware.gpus[")
    profile["null_reasons"] = {key: value for key, value in profile["null_reasons"].items()
                               if not key.startswith(decided)}
    return npf.validate(profile, complete=True)


def x299_card(profile):
    return npf.public_card(profile, "2026-10-19T13:00:00Z")


def x299_latest(compute_apps=0, free_bytes=43892981760, mono=MONO):
    """``latest.json`` built from the real nvidia-smi query and sensors -j capture.
    ``compute_apps`` and ``free_bytes`` are SYNTHETIC (not in the capture)."""
    rows, bad = tv.parse_nvidia_csv((X299 / "nvidia-smi-query-header.csv").read_text(encoding="utf-8"))
    assert bad == 0
    cpu = tv.parse_sensors_json((X299 / "sensors-j.json").read_text(encoding="utf-8"))
    gpus = [{"index": row["index"], "uuid": row["values"]["uuid"], "temp_c": row["values"]["temp_c"],
             "util_pct": row["values"]["util_pct"], "compute_apps": compute_apps} for row in rows]
    doc = {"schema": "atlas.sentinel-latest.v1", "node_id": "x299", "boot_id": BOOT_ID, "seq": 120443,
           "utc": "2026-10-19T12:00:00.512Z", "mono_s": mono, "gpus": gpus,
           "cpu_pkg_max_temp_c": max(item["temp_c"] for item in cpu["packages"]),
           "disks": [{"key": "runs", "free_bytes": free_bytes}], "null_reasons": {}}
    return fs.validate_sentinel_latest(doc)


def x299_worker(**overrides):
    worker = {"node_id": "x299", "worker": "gpu0", "phase": "smoke", "code_commit": COMMIT, "gpu_index": 0,
              "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000", "gpu_memory_mib": 12288.0,
              "compute_capability": None, "image_id": IMAGE, "datasets": ["synthetic-tiny"], "boot_id": BOOT_ID}
    worker.update(overrides)
    return worker


def fixture_record(name):
    return parse_record((RECORDS / name).read_text(encoding="utf-8"))


def legacy_spec(train_seed=7, **compat):
    spec = copy.deepcopy(fixture_record("legacy-b7a47f5-pending.md")["spec"])
    spec.pop("job_id")
    spec["train_seed"] = train_seed
    spec["compatibility"] = compat
    return prepare_job(spec)


def fleet_spec(train_seed=8, phase="smoke", scheduling=None, **req_overrides):
    spec = copy.deepcopy(fixture_record("fleet-v1-claimed.md")["spec"])
    spec.pop("job_id")
    spec.pop("scheduling")
    spec["train_seed"], spec["phase"] = train_seed, phase
    spec["requirements"]["allowed_nodes"] = None
    spec["requirements"].update(req_overrides)
    if scheduling is not None:
        spec["scheduling"] = scheduling
    return prepare_job(spec)


def fake_runtime(worker, datasets=None):
    """The real ``DockerRuntime.compatible`` bound to the worker facts (no Docker)."""
    runtime = DockerRuntime.__new__(DockerRuntime)
    runtime.s = {"phase": worker["phase"], "datasets": worker["datasets"] if datasets is None else datasets}
    runtime.commit, runtime.memory_mb = worker["code_commit"], worker["gpu_memory_mib"]
    runtime.image, runtime.gpu_uuid = worker["image_id"], worker["gpu_uuid"]
    return runtime


def mark(record, state):
    record = copy.deepcopy(record)
    record["state"] = state
    return record


class DeriveLegacyTests(unittest.TestCase):
    """8.3-3: derive_legacy reproduces the decision of the current compatible()."""

    COMPAT = [{}, {"min_memory_mb": 1000}, {"min_memory_mb": 12288}, {"min_memory_mb": 12289},
              {"min_memory_mb": "1000"}, {"min_memory_mb": 0.5}, {"min_memory_mb": -1},
              {"image_id": IMAGE}, {"image_id": OTHER_IMAGE}, {"image_id": None},
              {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000"},
              {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001"}, {"gpu_uuid": None},
              {"min_memory_mb": 1000, "image_id": IMAGE, "dataset": "synthetic-tiny"},
              {"min_memory_mb": 99999, "image_id": OTHER_IMAGE}]
    WORKERS = [x299_worker(), x299_worker(phase="hpo"), x299_worker(code_commit="b" * 40),
               x299_worker(datasets=[]), x299_worker(datasets=["other"]), x299_worker(gpu_memory_mib=4096.0),
               x299_worker(image_id=OTHER_IMAGE),
               x299_worker(gpu_index=1, gpu_uuid="GPU-00000000-0000-0000-0000-000000000001")]

    def test_matrix_equals_current_compatible(self):
        profile = x299_profile()
        for compat in self.COMPAT:
            spec = legacy_spec(**compat)
            req = fleet.derive_legacy(spec)
            for worker in self.WORKERS:
                expected = fake_runtime(worker).compatible(spec)
                with self.subTest(compat=compat, worker={k: worker[k] for k in ("phase", "image_id", "gpu_memory_mib")}):
                    ok, reason = fleet.static_match(req, None, worker, spec=spec)
                    self.assertEqual(expected, ok, reason)
                    self.assertIn(reason, fleet.REASONS)
                    # With the profile (fleet mode) a legacy job is decided the same way on an allowed GPU.
                    self.assertEqual(expected, fleet.static_match(req, profile, worker, spec=spec)[0])

    def test_real_legacy_fixture_record(self):
        record = fixture_record("legacy-b7a47f5-pending.md")
        req = fleet.job_requirements(record["spec"])
        self.assertEqual(fleet.LEGACY_SCHEMA, req["schema"])
        self.assertEqual(1000.0, req["min_vram_mib"])
        self.assertTrue(req["gpu_required"])
        self.assertIsNone(req["image_id"])
        self.assertFalse(req["legacy"]["image_id_declared"])
        self.assertEqual("synthetic-tiny", req["dataset"]["id"])
        worker = x299_worker()
        self.assertEqual((True, "ok"), fleet.static_match(req, None, worker, spec=record["spec"]))
        self.assertTrue(fake_runtime(worker).compatible(record["spec"]))

    def test_unparseable_min_memory_is_refused_not_raised(self):
        spec = legacy_spec(min_memory_mb="lots")
        with self.assertRaises(ValueError):  # current behaviour: the worker loop crashes
            fake_runtime(x299_worker()).compatible(spec)
        self.assertEqual((False, "legacy_compatibility_invalid"),
                         fleet.static_match(fleet.derive_legacy(spec), None, x299_worker(), spec=spec))

    def test_derive_legacy_is_pure(self):
        spec = legacy_spec(min_memory_mb=1000)
        before = json.dumps(spec, sort_keys=True)
        first = fleet.derive_legacy(spec)
        first["legacy"]["gpu_uuid"] = "mutated"
        self.assertEqual(before, json.dumps(spec, sort_keys=True))
        self.assertEqual(fleet.derive_legacy(spec), fleet.derive_legacy(copy.deepcopy(spec)))


class StaticMatrixTests(unittest.TestCase):
    """8.3-4: each static rule isolated, with its expected reason."""

    def setUp(self):
        self.profile = x299_profile()
        self.worker = x299_worker()
        self.spec = fleet_spec()

    def failing(self, req, profile=None, worker=None, spec=None):
        checks = fleet.static_checks(req, self.profile if profile is None else profile,
                                     self.worker if worker is None else worker, self.spec if spec is None else spec)
        return [item["reason"] for item in checks if not item["ok"]]

    def test_baseline_passes(self):
        self.assertEqual((True, "ok"), fleet.static_match(self.spec["requirements"], self.profile, self.worker, self.spec))
        self.assertEqual([], self.failing(self.spec["requirements"]))

    def test_each_rule_isolated(self):
        req = self.spec["requirements"]
        changed = lambda **kv: {**copy.deepcopy(req), **kv}
        p = self.profile
        cases = [
            ("node_not_allowed", changed(allowed_nodes=["sim-b"]), {}),
            ("hardware_class_mismatch", changed(hardware_class="ga102-24g"), {}),
            ("insufficient_vram", changed(min_vram_mib=12289), {}),
            ("insufficient_compute_capability", changed(min_compute_capability=[8, 6]), {}),
            ("image_mismatch", req, {"worker": x299_worker(image_id=OTHER_IMAGE)}),
            ("image_not_validated_on_node", req, {"profile": {**copy.deepcopy(p), "images": []}}),
            ("dataset_not_validated_on_node", changed(dataset={"id": "synthetic-tiny", "version": "v1",
                                                               "manifest_sha256": "0" * 64}), {}),
            ("dataset_not_validated_on_node", changed(dataset={"id": "inbreast-pilot-endurance",
                                                               "version": "other-version",
                                                               "manifest_sha256": INBREAST_MANIFEST}), {}),
            ("weights_not_validated_on_node", changed(weights_sha256="1" * 64), {}),
            ("container_exceeds_node_limits", changed(container={"cpus": 4, "memory_bytes": 4294967296}), {}),
            ("container_exceeds_node_limits", changed(container={"cpus": 2, "memory_bytes": 8589934592}), {}),
            ("gpu_not_allowed", req, {"profile": self.with_limits(allowed_gpus=[1])}),
            ("gpu_identity_mismatch", req, {"worker": x299_worker(gpu_uuid="GPU-00000000-0000-0000-0000-00000000000f")}),
            ("phase_mismatch", req, {"worker": x299_worker(phase="hpo")}),
            ("code_commit_mismatch", req, {"worker": x299_worker(code_commit="b" * 40)}),
            ("compute_capability_unknown", req, {"profile": self.without_cc()}),
        ]
        for expected, value, kwargs in cases:
            with self.subTest(expected=expected):
                self.assertEqual([expected], self.failing(value, **kwargs))
                ok, reason = fleet.static_match(value, kwargs.get("profile", self.profile),
                                                kwargs.get("worker", self.worker), self.spec)
                self.assertEqual((False, expected), (ok, reason))

    def test_gpu_absent_from_profile(self):
        # allowed_gpus is a subset of hardware.gpus in a valid profile, so both rules fire.
        worker = x299_worker(gpu_index=5, compute_capability=[5, 2])
        self.assertEqual(["gpu_not_allowed", "gpu_not_in_profile"], self.failing(self.spec["requirements"], worker=worker))
        self.assertEqual((False, "gpu_not_allowed"), fleet.static_match(self.spec["requirements"], self.profile, worker, self.spec))

    def with_limits(self, **limits):
        profile = copy.deepcopy(self.profile)
        profile["limits"].update(limits)
        return profile

    def without_cc(self):
        profile = copy.deepcopy(self.profile)
        for gpu in profile["hardware"]["gpus"]:
            gpu["compute_capability"] = None
        return profile

    def test_worker_probe_capability_wins_over_profile(self):
        req = {**copy.deepcopy(self.spec["requirements"]), "min_compute_capability": [6, 0]}
        self.assertEqual("insufficient_compute_capability", fleet.static_match(req, self.profile, self.worker, self.spec)[1])
        self.assertTrue(fleet.static_match(req, self.without_cc(), x299_worker(compute_capability=[8, 6]), self.spec)[0])

    def test_schema_level_refusals(self):
        req = self.spec["requirements"]
        cases = [("unsupported_requirements_schema", {**req, "schema": "atlas.requirements.v2"}),
                 ("invalid_requirements", {**req, "surprise": 1}),
                 ("invalid_requirements", {**req, "image_id": "latest"}),
                 ("invalid_requirements", "not-an-object"),
                 ("invalid_requirements", {key: value for key, value in req.items() if key != "image_id"})]
        for expected, value in cases:
            with self.subTest(expected=expected, value=str(value)[:40]):
                self.assertEqual((False, expected), fleet.static_match(value, self.profile, self.worker, self.spec))

    def test_v2_job_in_a_record_is_skipped(self):
        spec = copy.deepcopy(self.spec)
        spec["requirements"]["schema"] = "atlas.requirements.v2"
        record = new_record(spec)
        report = fleet.plan([record], self.profile, self.worker, latest=x299_latest(), now=MONO,
                            card=x299_card(self.profile))
        self.assertEqual("unsupported_requirements_schema", report["jobs"][0]["reason"])
        self.assertEqual([], report["order"])

    def test_v1_needs_a_profile(self):
        self.assertEqual((False, "node_profile_required"),
                         fleet.static_match(self.spec["requirements"], None, self.worker, self.spec))

    def test_final_jobs_pin_hardware_class(self):
        final = fleet_spec(phase="final")
        worker = x299_worker(phase="final")
        self.assertEqual((False, "final_requires_hardware_class"),
                         fleet.static_match(final["requirements"], self.profile, worker, final))
        pinned = fleet_spec(phase="final", hardware_class="gm200-12g")
        self.assertEqual((True, "ok"), fleet.static_match(pinned["requirements"], self.profile, worker, pinned))
        smoke = fleet_spec(hardware_class=None)
        self.assertTrue(fleet.static_match(smoke["requirements"], self.profile, self.worker, smoke)[0])

    def test_invalid_scheduling_and_persisted_legacy_marker(self):
        spec = copy.deepcopy(self.spec)
        spec["scheduling"] = {"priority": "high"}
        self.assertEqual((False, "invalid_scheduling"), fleet.static_match(spec["requirements"], self.profile, self.worker, spec))
        forged = copy.deepcopy(self.spec)
        forged["requirements"] = fleet.derive_legacy(forged)
        self.assertEqual((False, "invalid_requirements"),
                         fleet.static_match(fleet.job_requirements(forged), self.profile, self.worker, forged))

    def test_dataset_wildcard_and_exact_manifest(self):
        inbreast = {"id": "inbreast-pilot-endurance", "version": "inbreast-existing-classification-integrity-20261016",
                    "manifest_sha256": INBREAST_MANIFEST}
        req = {**copy.deepcopy(self.spec["requirements"]), "dataset": inbreast, "weights_sha256": WEIGHTS}
        self.assertEqual((True, "ok"), fleet.static_match(req, self.profile, self.worker, self.spec))


class OrderTests(unittest.TestCase):
    """8.3-5: permuting the input never changes the output."""

    def records(self):
        schedules = [None, {"priority": 0, "order": None}, {"priority": 0, "order": 120}, {"priority": 0, "order": 5},
                     {"priority": 2}, {"priority": 2, "order": 1}, {"priority": -1, "order": 1}, {"priority": 0},
                     {"priority": 0, "order": 5}]
        records = [new_record(fleet_spec(train_seed=100 + index, scheduling=scheduling))
                   for index, scheduling in enumerate(schedules)]
        records += [new_record(legacy_spec(train_seed=200 + index, min_memory_mb=1000)) for index in range(3)]
        return records

    def test_permutations_are_invariant_and_key_is_documented(self):
        records = self.records()
        expected = [record["job_id"] for record in fleet.order(records)]
        rng = random.Random(20261019)
        for _ in range(200):
            shuffled = records[:]
            rng.shuffle(shuffled)
            self.assertEqual(expected, [record["job_id"] for record in fleet.order(shuffled)])
        keys = [(-r["spec"].get("scheduling", {}).get("priority", 0),
                 r["spec"].get("scheduling", {}).get("order") is None,
                 r["spec"].get("scheduling", {}).get("order") or 0, r["job_id"]) for r in records]
        self.assertEqual(expected, [key[3] for key in sorted(keys)])
        first = fleet.order(records)[:2]
        self.assertEqual([2, 2], [r["spec"]["scheduling"]["priority"] for r in first])
        self.assertEqual(1, first[0]["spec"]["scheduling"]["order"])

    def test_plan_order_is_invariant_too(self):
        profile, records = x299_profile(), self.records()
        card, latest = x299_card(profile), x299_latest()
        expected = fleet.plan(records, profile, x299_worker(), latest=latest, now=MONO, card=card)
        self.assertEqual(len(records), len(expected["order"]))
        rng = random.Random(7)
        for _ in range(30):
            shuffled = records[:]
            rng.shuffle(shuffled)
            report = fleet.plan(shuffled, profile, x299_worker(), latest=latest, now=MONO, card=card)
            self.assertEqual(expected, report)

    def test_legacy_only_queue_keeps_job_id_order(self):
        records = [new_record(legacy_spec(train_seed=300 + index)) for index in range(6)]
        self.assertEqual(sorted(r["job_id"] for r in records), [r["job_id"] for r in fleet.order(records)])

    def test_malformed_scheduling_never_raises(self):
        records = self.records()
        broken = copy.deepcopy(records[0])
        broken["spec"]["scheduling"] = {"priority": "x", "order": 1.5}
        fleet.order([broken] + records[1:])
        self.assertEqual(fleet.sort_key(broken)[:3], (0, True, 0))


class NodeGateTests(unittest.TestCase):
    """8.3-6: node gates (hot, disk, stale sentinel, busy GPU, draining card, profile hash)."""

    def setUp(self):
        self.profile = x299_profile()
        self.card = x299_card(self.profile)
        self.worker = x299_worker()
        self.records = [new_record(fleet_spec()), fixture_record("legacy-b7a47f5-pending.md")]

    def gate(self, latest=None, now=MONO, profile=None, card="default", worker=None, idle=None):
        latest = x299_latest() if latest is None else latest
        return fleet.node_gates(latest, self.profile if profile is None else profile, now,
                                worker=self.worker if worker is None else worker,
                                card=self.card if card == "default" else card, idle_since_mono=idle)

    def assert_blocks(self, reason, **kwargs):
        self.assertEqual((False, reason), self.gate(**kwargs))
        kwargs.setdefault("latest", x299_latest())
        report = fleet.plan(self.records, kwargs.get("profile", self.profile), kwargs.get("worker", self.worker),
                            latest=kwargs["latest"], now=kwargs.get("now", MONO),
                            card=self.card if kwargs.get("card", "default") == "default" else kwargs["card"],
                            idle_since_mono=kwargs.get("idle"))
        self.assertFalse(report["node"]["ok"])
        self.assertEqual(reason, report["node"]["reason"])
        self.assertEqual([], report["order"], "a blocked node claims nothing")
        self.assertTrue(all(not job["eligible"] for job in report["jobs"]))

    def test_real_x299_idle_snapshot_passes(self):
        self.assertEqual((True, "ok"), self.gate())
        report = fleet.plan(self.records, self.profile, self.worker, latest=x299_latest(), now=MONO, card=self.card)
        self.assertEqual(2, len(report["order"]))

    def hot(self, gpu=None, cpu=None):
        latest = x299_latest()
        if gpu is not None:
            latest["gpus"][1]["temp_c"] = gpu  # node gate: any GPU of the serial node
        if cpu is not None:
            latest["cpu_pkg_max_temp_c"] = cpu
        return latest

    def test_thermal(self):
        self.assert_blocks("gpu_too_hot", latest=self.hot(gpu=76))
        self.assertEqual((True, "ok"), self.gate(latest=self.hot(gpu=75)))
        self.assert_blocks("cpu_pkg_too_hot", latest=self.hot(cpu=75.5))

    def test_disk_reserve_and_per_job_requirement(self):
        self.assert_blocks("disk_below_reserve", latest=x299_latest(free_bytes=20 * GIB - 1))
        self.assertEqual((True, "ok"), self.gate(latest=x299_latest(free_bytes=20 * GIB)))
        latest = x299_latest(free_bytes=20 * GIB + 5 * GIB - 1)  # job asks for 5 GiB above the reserve
        report = fleet.plan(self.records, self.profile, self.worker, latest=latest, now=MONO, card=self.card)
        reasons = {job["kind"]: job["reason"] for job in report["jobs"]}
        self.assertEqual({"v1": "disk_insufficient_for_job", "legacy": "ok"}, reasons)
        self.assertEqual((False, "disk_insufficient_for_job"),
                         fleet.job_disk_gate(self.records[0]["spec"]["requirements"], latest, self.profile))

    def test_sentinel_freshness(self):
        self.assertEqual((True, "ok"), self.gate(now=MONO + 5.0))
        self.assert_blocks("sentinel_stale", now=MONO + 5.001)
        self.assert_blocks("sentinel_stale", now=MONO - 1.5)
        self.assertEqual((False, "sentinel_latest_missing"),
                         fleet.node_gates(None, self.profile, MONO, worker=self.worker, card=self.card))
        self.assert_blocks("sentinel_latest_invalid", latest="unreadable")
        self.assert_blocks("sentinel_latest_invalid", latest={"schema": "atlas.sentinel-latest.v1"})
        self.assert_blocks("sentinel_other_boot", worker=x299_worker(boot_id="bbbbbbbb-0000-4000-8000-00000000000b"))
        self.assert_blocks("boot_id_unknown", worker=x299_worker(boot_id=None))
        other = x299_latest()
        other["node_id"] = "sim-b"
        self.assert_blocks("sentinel_node_mismatch", latest=other)

    def test_busy_gpu(self):
        self.assert_blocks("gpu_busy", latest=x299_latest(compute_apps=1))
        busy = x299_latest()
        busy["gpus"][0]["util_pct"] = 10
        self.assert_blocks("gpu_busy", latest=busy)
        busy["gpus"][0]["util_pct"] = 9
        self.assertEqual((True, "ok"), self.gate(latest=busy))
        other = x299_latest()
        other["gpus"][1]["util_pct"] = 100  # the other GPU's load is not this worker's idle gate
        self.assertEqual((True, "ok"), self.gate(latest=other))
        self.assert_blocks("gpu_not_reported", worker=x299_worker(gpu_index=7))
        self.assert_blocks("gpu_identity_mismatch", worker=x299_worker(gpu_uuid="GPU-00000000-0000-0000-0000-00000000000f"))

    def test_card_and_profile(self):
        draining = copy.deepcopy(self.profile)
        draining["status"] = "draining"
        self.assert_blocks("node_not_active", profile=draining, card=x299_card(draining))
        self.assert_blocks("node_card_mismatch", profile=draining)  # card published for the active profile
        tampered = copy.deepcopy(self.card)
        tampered["status"] = "draining"  # hash still of the active profile
        self.assert_blocks("node_not_active", card=tampered)
        self.assert_blocks("node_card_missing", card=None)
        self.assert_blocks("node_card_invalid", card={"profile_sha256": "x"})
        probe_only = npf.probe(tv.ReplayReaders(X299), "x299", NOW_UTC)
        self.assert_blocks("node_profile_invalid", profile=probe_only)
        self.assert_blocks("worker_node_mismatch", worker=x299_worker(node_id="sim-b"))

    def test_cooldown(self):
        self.assert_blocks("node_cooldown", idle=MONO - 59.9)
        self.assertEqual((True, "ok"), self.gate(idle=MONO - 60))


class NullSensorTests(unittest.TestCase):
    """Sensors reported as null (with a reason) never raise; only gated fields refuse."""

    def setUp(self):
        self.profile = x299_profile()
        self.card = x299_card(self.profile)
        self.worker = x299_worker()

    def gate(self, latest, profile=None):
        fs.validate_sentinel_latest(latest)  # every fixture below is a VALID latest.json
        return fleet.node_gates(latest, profile or self.profile, MONO, worker=self.worker, card=self.card)

    def nulled(self, path, reason):
        latest = x299_latest()
        target, key = latest, path
        if path.startswith("gpus["):
            index, key = int(path[5]), path.split(".", 1)[1]
            target = latest["gpus"][index]
        elif path.startswith("disks["):
            target, key = latest["disks"][0], "free_bytes"
        target[key] = None
        latest["null_reasons"][path] = reason
        return latest

    def test_gated_nulls_refuse_with_stable_reason(self):
        cases = [("gpus[0].temp_c", "tool_timeout", "gpu_temp_unavailable"),
                 ("gpus[1].temp_c", "parse_error", "gpu_temp_unavailable"),
                 ("cpu_pkg_max_temp_c", "permission_denied", "cpu_pkg_temp_unavailable"),
                 ("disks[0].free_bytes", "collector_error", "disk_free_unavailable"),
                 ("gpus[0].compute_apps", "first_sample", "gpu_idle_unknown"),
                 ("gpus[0].util_pct", "stale", "gpu_idle_unknown")]
        for path, null_reason, expected in cases:
            with self.subTest(path=path):
                self.assertEqual((False, expected), self.gate(self.nulled(path, null_reason)))

    def test_ungated_nulls_are_ignored(self):
        for path, reason in (("gpus[1].compute_apps", "first_sample"), ("gpus[1].util_pct", "stale"),
                             ("gpus[0].uuid", "not_exposed"), ("gpus[1].uuid", "not_exposed")):
            with self.subTest(path=path):
                self.assertEqual((True, "ok"), self.gate(self.nulled(path, reason)))
        no_cpu_limit = copy.deepcopy(self.profile)
        no_cpu_limit["limits"]["thermal"] = no_cpu_limit["limits"]["thermal"][:1]
        card = x299_card(no_cpu_limit)
        latest = self.nulled("cpu_pkg_max_temp_c", "not_exposed")
        self.assertEqual((True, "ok"), fleet.node_gates(latest, no_cpu_limit, MONO, worker=self.worker, card=card))

    def test_whole_sections_null(self):
        latest = x299_latest()
        latest.update(gpus=None, null_reasons={"gpus": "tool_absent"})
        self.assertEqual((False, "gpu_temp_unavailable"), self.gate(latest))
        latest = x299_latest()
        latest.update(disks=None, null_reasons={"disks": "first_sample"})
        self.assertEqual((False, "disk_runs_unreported"), self.gate(latest))
        latest = x299_latest()
        latest["disks"] = [{"key": "data", "free_bytes": 10 ** 12}]
        self.assertEqual((False, "disk_runs_unreported"), self.gate(latest))

    def test_null_without_reason_is_an_invalid_document_not_a_crash(self):
        latest = x299_latest()
        latest["gpus"][0]["temp_c"] = None
        self.assertEqual((False, "sentinel_latest_invalid"),
                         fleet.node_gates(latest, self.profile, MONO, worker=self.worker, card=self.card))

    def test_unknown_thermal_sensor_fails_closed(self):
        profile = copy.deepcopy(self.profile)
        profile["limits"]["thermal"].append({"sensor": "vrm.temp_c", "ceiling_c": 100, "preventive_c": 95,
                                             "claim_max_c": 80, "source": "hwmon:nct6798"})
        self.assertEqual((False, "thermal_sensor_unsupported"),
                         fleet.node_gates(x299_latest(), profile, MONO, worker=self.worker, card=x299_card(profile)))

    def test_every_single_null_is_handled(self):
        """Exhaustive: null each leaf of a valid latest.json (with a reason) -> never raises."""
        base = x299_latest()
        paths = [f"gpus[{i}].{key}" for i in range(2) for key in ("uuid", "temp_c", "util_pct", "compute_apps")]
        paths += ["cpu_pkg_max_temp_c", "disks[0].free_bytes"]
        for path in paths:
            for reason in sorted(fs.REASONS):
                latest = self.nulled(path, reason)
                ok, code = fleet.node_gates(latest, self.profile, MONO, worker=self.worker, card=self.card)
                self.assertIn(code, fleet.REASONS)
        self.assertEqual((True, "ok"), self.gate(base))


class LegacySelectionTests(unittest.TestCase):
    """Old jobs are still selected exactly as the current worker loop would."""

    def queue(self):
        dep = new_record(legacy_spec(train_seed=40, min_memory_mb=1000))
        waiting_spec = legacy_spec(train_seed=41)
        waiting_spec["dependencies"] = [dep["job_id"]]
        records = [fixture_record("legacy-b7a47f5-pending.md"), dep, new_record(prepare_job(waiting_spec)),
                   new_record(legacy_spec(train_seed=42, min_memory_mb=13000)),
                   new_record(legacy_spec(train_seed=43, image_id=OTHER_IMAGE)),
                   new_record(legacy_spec(train_seed=44, image_id=IMAGE, min_memory_mb=1000)),
                   new_record(legacy_spec(train_seed=45, gpu_uuid="GPU-00000000-0000-0000-0000-000000000001")),
                   mark(new_record(legacy_spec(train_seed=46)), "succeeded")]
        return sorted(records, key=lambda r: r["job_id"])  # list_jobs() order: sorted jobs/<job_id>.md

    @staticmethod
    def current_loop(records, runtime):
        """b7a47f5 runtime.worker selection: first pending+compatible whose claim succeeds
        (GitQueue.claim returns None while a dependency is not succeeded)."""
        states = {r["job_id"]: r["state"] for r in records}
        chosen = []
        for candidate in records:
            if candidate["state"] != "pending" or not runtime.compatible(candidate["spec"]):
                continue
            if all(states.get(dep) == "succeeded" for dep in candidate["spec"]["dependencies"]):
                chosen.append(candidate["job_id"])
        return chosen

    def test_same_choice_in_legacy_mode_and_in_fleet_mode(self):
        profile = x299_profile()
        for worker in (x299_worker(), x299_worker(gpu_index=1, gpu_uuid="GPU-00000000-0000-0000-0000-000000000001"),
                       x299_worker(image_id=OTHER_IMAGE), x299_worker(gpu_memory_mib=16384.0)):
            records = self.queue()
            expected = self.current_loop(records, fake_runtime(worker))
            with self.subTest(worker=(worker["gpu_index"], worker["image_id"][:10], worker["gpu_memory_mib"])):
                self.assertTrue(expected)
                legacy = fleet.plan(records, None, worker)
                self.assertEqual(expected, legacy["order"])
                self.assertEqual("legacy", legacy["mode"])
                rng = random.Random(3)
                rng.shuffle(records)
                if worker["image_id"] == IMAGE:  # fleet mode: the node image must be validated in the profile
                    latest = x299_latest()
                    if worker["gpu_index"] == 1:
                        latest["gpus"][0]["util_pct"] = 100  # gpu0 busy does not block gpu1
                    fleet_mode = fleet.plan(records, profile, worker, latest=latest, now=MONO, card=x299_card(profile))
                    self.assertEqual(expected, fleet_mode["order"])

    def test_dependency_and_state_reasons(self):
        records = self.queue()
        report = {job["job_id"]: job for job in fleet.plan(records, None, x299_worker())["jobs"]}
        waiting = next(r for r in records if r["spec"]["train_seed"] == 41)
        done = next(r for r in records if r["spec"]["train_seed"] == 46)
        self.assertEqual("dependency_not_succeeded", report[waiting["job_id"]]["reason"])
        self.assertEqual("not_pending", report[done["job_id"]]["reason"])
        records = [mark(r, "succeeded") if r["spec"]["train_seed"] == 40 else r for r in records]
        report = {job["job_id"]: job for job in fleet.plan(records, None, x299_worker())["jobs"]}
        self.assertTrue(report[waiting["job_id"]]["eligible"])

    def test_v1_jobs_are_not_run_in_legacy_mode(self):
        report = fleet.plan([new_record(fleet_spec())], None, x299_worker())
        self.assertEqual("node_profile_required", report["jobs"][0]["reason"])


def b7a47f5_seed_spec(cfg, phase, split_hash, code_commit, config_rel, config_sha256):
    """Verbatim copy of the spec literal in cli.py at b7a47f5 (recipe_identity unchanged)."""
    return dict(protocol_version=cfg['protocol_version'], dataset=cfg['dataset']['id'], dataset_version=cfg['dataset']['version'],
                split_hash=split_hash, recipe_hash=cli.recipe_identity(cfg), phase=phase, fold=cfg['splits']['fold'],
                train_seed=cfg['training']['seed'], arm=cfg['augmentation']['arm'],
                code_commit=code_commit, config=config_rel,
                config_sha256=config_sha256, dependencies=cfg['recipe']['dependencies'], compatibility={'min_memory_mb': 1000})


def seed_args(**overrides):
    values = dict(image_id=IMAGE, image_ref=None, min_vram_mib=1000, min_compute_capability=None, hardware_class=None,
                  allowed_node=None, min_free_disk_bytes=0, container_cpus=None, container_memory_bytes=None,
                  expected_epoch_seconds_max=None, priority=None, order=None)
    values.update(overrides)
    return types.SimpleNamespace(**values)


class SeedTests(unittest.TestCase):
    """cli seed writes requirements + compatibility additively; job_id never changes."""

    CONFIGS = ["configs/smoke/synthetic.json", "configs/smoke/inbreast.json",
               "configs/smoke/endurance-v1/inbreast/basic.json"]

    def load(self, name):
        from augment_atlas.contracts import load_config
        return load_config(ROOT / name)

    def test_identity_fields_and_job_ids_unchanged(self):
        self.assertEqual(("protocol_version", "dataset", "dataset_version", "split_hash", "recipe_hash", "phase",
                          "fold", "train_seed", "arm", "code_commit"), IDENTITY_FIELDS)
        old_validate = self.old_schema().validate_record
        for name in self.CONFIGS:
            cfg = self.load(name)
            args = ("smoke", "f" * 64, COMMIT, name, "e" * 64)
            before = b7a47f5_seed_spec(cfg, *args)
            self.assertEqual(before, cli.legacy_seed_spec(cfg, *args))
            for options in (seed_args(), seed_args(hardware_class="gm200-12g", allowed_node=["x299"],
                                                   min_compute_capability="5.2", min_vram_mib=11000,
                                                   container_cpus=2.0, container_memory_bytes=4294967296,
                                                   min_free_disk_bytes=5 * GIB, priority=1, order=120,
                                                   expected_epoch_seconds_max=1800,
                                                   image_ref="ghcr.io/owner/augment-atlas@sha256:" + "5" * 64)):
                after = cli.fleet_seed_spec(cfg, *args, options)
                with self.subTest(config=name, options=options.hardware_class):
                    self.assertEqual(prepare_job(before)["job_id"], prepare_job(after)["job_id"])
                    self.assertEqual({k: v for k, v in before.items() if k != "compatibility"},
                                     {k: v for k, v in after.items() if k not in ("compatibility", "requirements", "scheduling")})
                    fs.validate_requirements(after["requirements"])
                    self.assertEqual({"min_memory_mb": options.min_vram_mib, "image_id": IMAGE}, after["compatibility"])
                    self.assertEqual(cfg["dataset"].get("manifest_sha256"), after["requirements"]["dataset"]["manifest_sha256"])
                    record = new_record(after)
                    old_validate(copy.deepcopy(record))  # an old worker still lists the queue
                    # an old coordinator (compatibility only) still enforces image and VRAM
                    worker = x299_worker(code_commit=COMMIT, datasets=[cfg["dataset"]["id"]])
                    self.assertEqual(options.min_vram_mib <= 12288, fake_runtime(worker).compatible(after))
                    self.assertFalse(fake_runtime(x299_worker(image_id=OTHER_IMAGE, datasets=[cfg["dataset"]["id"]])).compatible(after))
        endurance = cli.fleet_seed_spec(self.load(self.CONFIGS[2]), "smoke", "f" * 64, COMMIT, self.CONFIGS[2], "e" * 64,
                                        seed_args())
        self.assertEqual(INBREAST_MANIFEST, endurance["requirements"]["dataset"]["manifest_sha256"])
        self.assertEqual(WEIGHTS, endurance["requirements"]["weights_sha256"])

    @staticmethod
    def old_schema():
        spec = importlib.util.spec_from_file_location("schema_b7a47f5_f4", RECORDS / "schema_b7a47f5.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_controller_v2_seed_argv_without_image_id_seeds_legacy_spec(self):
        """The reviewed endurance controller V2 calls `cli seed` without fleet flags; that must stay byte-identical."""
        from unittest import mock
        cfg = self.load(self.CONFIGS[0])
        seeded = []

        class FakeQueue:
            def __init__(self, *args): pass
            def seed(self, specs):
                seeded.extend(specs)
                return [prepare_job(spec) for spec in specs]

        argv = ["seed", "--phase", "smoke", "--config", str(ROOT / self.CONFIGS[0]), "--code", str(ROOT),
                "--data", "unused-data-root", "--checkout", "unused-checkout", "--remote", "unused-remote", "--branch", "queue"]
        with mock.patch.object(cli, "load_config", return_value=cfg),                 mock.patch.object(cli, "load_records", return_value=[]),                 mock.patch.object(cli, "prepare_splits", return_value={"sha256": "f" * 64}),                 mock.patch.object(cli, "command", return_value=COMMIT),                 mock.patch.object(cli, "digest", return_value="e" * 64),                 mock.patch.object(cli, "GitQueue", FakeQueue),                 mock.patch("sys.stdout", new_callable=io.StringIO):
            cli.main(argv)
        self.assertEqual([cli.legacy_seed_spec(cfg, "smoke", "f" * 64, COMMIT, self.CONFIGS[0], "e" * 64)], seeded)
        self.assertNotIn("requirements", seeded[0])
        for extra in (["--hardware-class", "gm200-12g"], ["--priority", "1"], ["--min-vram-mib", "2000"]):
            with self.subTest(extra=extra), mock.patch("sys.stderr", new_callable=io.StringIO),                     self.assertRaises(SystemExit) as raised:
                cli.main(argv + extra)
            self.assertEqual(2, raised.exception.code)

    def test_invalid_seed_options_are_rejected(self):
        cfg = self.load(self.CONFIGS[0])
        args = ("smoke", "f" * 64, COMMIT, self.CONFIGS[0], "e" * 64)
        for options in (seed_args(image_id="augment-atlas:latest"), seed_args(container_cpus=2.0),
                        seed_args(min_compute_capability="five"), seed_args(allowed_node=["X299"]),
                        seed_args(min_vram_mib=-1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                cli.fleet_seed_spec(cfg, *args, options)
        with self.assertRaises(ValueError):
            fleet.seed_requirements(cfg, phase="final", image_id=IMAGE)

    def test_seeded_fleet_job_is_selectable_on_x299(self):
        cfg = self.load(self.CONFIGS[0])
        spec = prepare_job(cli.fleet_seed_spec(cfg, "smoke", "f" * 64, COMMIT, self.CONFIGS[0], "e" * 64,
                                               seed_args(hardware_class="gm200-12g", allowed_node=["x299"])))
        profile = x299_profile()
        report = fleet.plan([new_record(spec)], profile, x299_worker(), latest=x299_latest(), now=MONO,
                            card=x299_card(profile))
        self.assertEqual([spec["job_id"]], report["order"])

    @unittest.skipUnless(shutil.which("git"), "git is required for the end-to-end seed")
    def test_cli_seed_end_to_end_keeps_the_legacy_job_id(self):
        with tempfile.TemporaryDirectory(prefix="atlas-f4-seed-") as tmp:
            tmp = Path(tmp)
            remote = tmp / "remote.git"
            subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(["seed", "--phase", "smoke", "--config", str(ROOT / self.CONFIGS[0]), "--code", str(ROOT),
                          "--data", str(tmp), "--remote", str(remote), "--checkout", str(tmp / "admin"),
                          "--image-id", IMAGE])
            job_id = out.getvalue().strip()
            cfg = self.load(self.CONFIGS[0])
            from augment_atlas.data import load_records, prepare_splits
            from augment_atlas.storage import digest
            old = os.environ.get("ATLAS_DATA_ROOT")
            os.environ["ATLAS_DATA_ROOT"] = str(tmp)
            try:
                splits = prepare_splits(cfg, load_records(cfg), tmp / "splits")
            finally:
                if old is None:
                    os.environ.pop("ATLAS_DATA_ROOT", None)
                else:
                    os.environ["ATLAS_DATA_ROOT"] = old
            head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True).stdout.strip()
            legacy = b7a47f5_seed_spec(cfg, "smoke", splits["sha256"], head, self.CONFIGS[0],
                                       digest(ROOT / self.CONFIGS[0]))
            self.assertEqual(prepare_job(legacy)["job_id"], job_id)
            record = GitQueue(tmp / "reader", remote).list_jobs()[0]
            self.assertEqual(job_id, record["job_id"])
            self.assertEqual(IMAGE, record["spec"]["requirements"]["image_id"])
            self.assertEqual({"min_memory_mb": 1000, "image_id": IMAGE}, record["spec"]["compatibility"])
            self.assertNotIn("scheduling", record["spec"])


class ExplainCliTests(unittest.TestCase):
    """``atlas fleet explain`` over fixture files; read-only, no paths in the output."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="atlas-f4-explain-")
        self.dir = Path(self.tmp.name)
        jobs = self.dir / "jobs"
        jobs.mkdir()
        profile = x299_profile()
        self.records = [new_record(fleet_spec()), fixture_record("legacy-b7a47f5-pending.md"),
                        new_record(fleet_spec(train_seed=9, hardware_class="ga102-24g"))]
        for record in self.records:
            (jobs / f"{record['job_id']}.md").write_text(render_record(record), encoding="utf-8")
        for name, doc in (("profile", profile), ("card", x299_card(profile)), ("latest", x299_latest()),
                          ("worker", x299_worker())):
            (self.dir / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *extra, files=("profile", "card", "latest")):
        argv = ["fleet", "explain", "--jobs-dir", str(self.dir / "jobs"), "--worker", str(self.dir / "worker.json")]
        for name in files:
            argv += [f"--{name}", str(self.dir / f"{name}.json")]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv + list(extra))
        return code, out.getvalue(), err.getvalue()

    def test_text_report(self):
        code, out, _ = self.run_cli("--now-mono", str(MONO + 1))
        self.assertEqual(0, code)
        self.assertIn("node: PASS (ok)", out)
        self.assertIn("hardware_class_mismatch", out)
        self.assertEqual(2, out.count("ELIGIBLE"))
        self.assertIn("[PASS] thermal:gpu.temp_c: ok", out)
        self.assertNotIn(self.tmp.name, out)
        self.assertEqual([], fs.forbidden_strings(out))

    def test_json_and_single_job(self):
        target = self.records[2]["job_id"]
        code, out, _ = self.run_cli("--now-mono", str(MONO), "--json", "--job", target)
        self.assertEqual(1, code)
        report = json.loads(out)
        self.assertEqual([target], [job["job_id"] for job in report["jobs"]])
        self.assertEqual("hardware_class_mismatch", report["jobs"][0]["reason"])

    def test_stale_sentinel_blocks_everything(self):
        code, out, _ = self.run_cli("--now-mono", str(MONO + 30))
        self.assertEqual(1, code)
        self.assertIn("node: BLOCKED (sentinel_stale)", out)
        self.assertIn("[SKIP] thermal: not_evaluated", out)
        self.assertNotIn("ELIGIBLE", out)

    def test_missing_latest_and_legacy_mode(self):
        (self.dir / "latest.json").unlink()
        code, out, _ = self.run_cli("--now-mono", str(MONO))
        self.assertEqual((1, True), (code, "sentinel_latest_missing" in out))
        code, out, _ = self.run_cli(files=())
        self.assertEqual(0, code)
        self.assertIn("mode: legacy", out)
        self.assertIn("node_profile_required", out)
        self.assertEqual(1, out.count("ELIGIBLE"))

    def test_bad_input_exits_2(self):
        (self.dir / "worker.json").write_text("[]", encoding="utf-8")
        code, _, err = self.run_cli("--now-mono", str(MONO))
        self.assertEqual(2, code)
        self.assertIn("worker facts", err)


class CatalogueTests(unittest.TestCase):
    def test_reasons_are_documented_in_the_spec(self):
        text = (ROOT / "docs" / "fleet-protocol.md").read_text(encoding="utf-8")
        missing = [reason for reason in fleet.REASONS if f"`{reason}`" not in text]
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
