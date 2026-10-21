"""F0: pure validators of the fleet protocol v1 contracts (docs/fleet-protocol.md)."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest

from augment_atlas import fleet_schema as fs
from augment_atlas.queue import parse_record, render_record
from augment_atlas.schema import digest

FIXTURES = Path(__file__).parent / "fixtures" / "telemetry_v1"
SYNTHETIC = FIXTURES / "synthetic"
EXAMPLE_PROFILE = Path(__file__).parent.parent / "configs" / "node_profile.example.json"


def requirements(**changes):
    value = {"schema": "atlas.requirements.v1", "gpu_required": True, "min_vram_mib": 11000,
             "min_compute_capability": [5, 2], "hardware_class": "gm200-12g", "allowed_nodes": None,
             "image_id": "sha256:" + "3" * 64,
             "image_ref": "ghcr.io/owner/augment-atlas@sha256:" + "4" * 64,
             "dataset": {"id": "inbreast-pilot-endurance",
                         "version": "inbreast-existing-classification-integrity-20261016",
                         "manifest_sha256": "9" * 64},
             "weights_sha256": "1" * 64, "min_free_disk_bytes": 5368709120,
             "container": {"cpus": 2, "memory_bytes": 4294967296}, "expected_epoch_seconds_max": 1800}
    value.update(changes)
    return value


def complete_profile():
    profile = json.loads(EXAMPLE_PROFILE.read_text(encoding="utf-8"))
    profile.update(status="active", hardware_class="gm200-12g",
                   guardian={"kind": "private-v2", "required_for_claim": True, "review_ref": "review-v2-placeholder",
                             "declared_thresholds": {"gpu_ceiling_c": 90, "gpu_preventive_c": 89, "cadence_s": 0.5}})
    profile["storage"] = [{"key": "runs", "reserve_bytes": 21474836480}, {"key": "data", "reserve_bytes": 0}]
    profile["limits"].update(
        max_concurrent_trainings=1, allowed_gpus=[0, 1], min_idle_seconds_between_jobs=60,
        container={"cpus": 2, "memory_bytes": 4294967296, "shm_bytes": 268435456},
        thermal=[{"sensor": "gpu.temp_c", "ceiling_c": 90, "preventive_c": 89, "claim_max_c": 75, "source": "nvidia-smi"},
                 {"sensor": "cpu_pkg.temp_c", "ceiling_c": 90, "preventive_c": 85, "claim_max_c": 75,
                  "source": "hwmon:coretemp"}])
    profile["null_reasons"] = {key: value for key, value in profile["null_reasons"].items()
                               if not (key.startswith("limits.") or key.startswith("storage[")
                                       or key in {"hardware_class", "guardian"})}
    return profile


def fast_record(**changes):
    record = {"schema": "atlas.telemetry.v1", "type": "fast", "node_id": "x299",
              "boot_id": "8c40d0a6-6a4f-46b4-adaf-bae6c4b8a2a6", "seq": 120443, "utc": "2026-10-19T12:00:00.512Z",
              "mono_s": 86400.512, "dt_s": 0.501,
              "gpus": [{"index": 0, "temp_c": 83, "power_w": 181.2, "util_pct": 99, "mem_used_mib": 10850,
                        "sm_mhz": 1189, "mem_mhz": 3505, "fan_pct": 61, "pstate": "P2", "age_s": 0.1},
                       {"index": 1, "temp_c": 34, "power_w": 15.1, "util_pct": 0, "mem_used_mib": 2,
                        "sm_mhz": 135, "mem_mhz": 405, "fan_pct": 22, "pstate": "P8", "age_s": 0.1}],
              "cpu_pkg": [{"package": 0, "temp_c": 62.0}], "null_reasons": {},
              "src": {"gpu": "nvidia-smi-stream", "cpu_pkg": "hwmon:coretemp"}}
    record.update(changes)
    return record


def summary(**changes):
    value = {
        "schema": "atlas.job-telemetry-summary.v1", "node_id": "x299", "hardware_class": "gm200-12g",
        "job_id": "e" * 64, "attempt_id": "f" * 32,
        "window": {"start_utc": "2026-10-19T10:00:00Z", "end_utc": "2026-10-19T11:28:41Z", "duration_s": 5321.4},
        "coverage": {"fast_fraction": 0.998, "max_gap_s": 1.4, "reboot_detected": False},
        "thermal": {"gpu_temp_max_c": 83, "gpu_temp_p95_c": 81, "cpu_pkg_temp_max_c": 66},
        "power": {"gpu_power_mean_w": 176.3},
        "energy": {
            "gpu_kwh": {"value": 0.2607, "status": "measured_integrated", "source": "nvidia-smi power.draw 0.5s",
                        "gpu_index": 0},
            "cpu_pkg_kwh": {"value": None, "status": "unavailable", "reason": "permission_denied"},
            "dram_kwh": {"value": None, "status": "unavailable", "reason": "permission_denied"},
            "cpu_kwh_estimated": {"value": None, "status": "estimated_model", "method": None,
                                  "reason": "disabled_by_config"}},
        "boundary": {"measured": ["gpu_board"], "estimated": [], "unavailable": ["cpu_package", "dram"],
                     "excluded_always": ["psu_losses", "motherboard", "disks", "fans", "network", "cooling"],
                     "note": "Not wall-plug energy."},
        "co2": {"g": None, "boundary": ["gpu_board"], "basis": "measured_only", "grid_factor_g_per_kwh": None,
                "factor_source_id": None, "reason": "grid_factor_missing"},
        "attribution": {"gpu": "exclusive_device", "cpu_pkg": "host_shared_serial_node"}}
    value.update(changes)
    return value


class RequirementsTests(unittest.TestCase):
    def test_valid_requirements_and_nullable_fields(self):
        fs.validate_requirements(requirements())
        fs.validate_requirements(requirements(hardware_class=None, allowed_nodes=["x299", "sim-b"], image_ref=None,
                                              weights_sha256=None, min_compute_capability=None))

    def test_image_id_is_mandatory_and_strict(self):
        value = requirements()
        del value["image_id"]
        with self.assertRaisesRegex(fs.SchemaError, "missing keys"):
            fs.validate_requirements(value)
        for bad in ("3" * 64, "sha256:" + "3" * 63, "sha256:" + "G" * 64):
            with self.subTest(bad=bad), self.assertRaises(fs.SchemaError):
                fs.validate_requirements(requirements(image_id=bad))

    def test_unknown_requirement_schema_is_skippable_not_fatal(self):
        with self.assertRaises(fs.UnsupportedSchema):
            fs.validate_requirements(requirements(schema="atlas.requirements.v2"))
        with self.assertRaises(fs.SchemaError) as caught:
            fs.validate_requirements(requirements(schema="something-else"))
        self.assertNotIsInstance(caught.exception, fs.UnsupportedSchema)

    def test_unknown_keys_and_incoherent_values_are_rejected(self):
        cases = {"unknown key": requirements(gpu_vendor="nvidia"),
                 "duplicate nodes": requirements(allowed_nodes=["x299", "x299"]),
                 "empty nodes": requirements(allowed_nodes=[]),
                 "hostname-like node": requirements(allowed_nodes=["x299.lab.example"]),
                 "negative vram": requirements(min_vram_mib=-1),
                 "bool vram": requirements(min_vram_mib=True),
                 "capability shape": requirements(min_compute_capability=[5]),
                 "zero cpus": requirements(container={"cpus": 0, "memory_bytes": 1}),
                 "short weights": requirements(weights_sha256="11ad3fa6"),
                 "path in dataset": requirements(dataset={"id": "/srv/data", "version": "v", "manifest_sha256": None}),
                 "ip in dataset": requirements(dataset={"id": "10.0.0.7", "version": "v", "manifest_sha256": None}),
                 "nan": requirements(container={"cpus": math.nan, "memory_bytes": 1})}
        for name, value in cases.items():
            with self.subTest(name), self.assertRaises(fs.SchemaError):
                fs.validate_requirements(value)

    def test_scheduling(self):
        fs.validate_scheduling({"priority": 0, "order": 120})
        fs.validate_scheduling({})
        with self.assertRaises(fs.SchemaError):
            fs.validate_scheduling({"priority": 0.5})


class ClaimFleetTests(unittest.TestCase):
    def block(self, **changes):
        value = {"protocol": "atlas.fleet.v1", "node_profile_sha256": "6" * 64, "coordinator_commit": "7" * 40,
                 "boot_id": "6f1c0000-0000-4000-8000-000000000000", "image_id": "sha256:" + "3" * 64,
                 "launch_plan_sha256": "8" * 64, "requirements_sha256": "9" * 64, "lease_seconds": 3600,
                 "heartbeat_ref": "refs/heads/fleet-hb/x299/gpu0",
                 "gate": {"observed_at_utc": "2026-10-19T12:00:00Z", "gpu_temp_c": 41, "cpu_pkg_temp_c": 38,
                          "free_disk_bytes": 43892981760, "gpu_idle": True}}
        value.update(changes)
        return value

    def test_valid_block(self):
        fs.validate_claim_fleet(self.block())

    def test_reserved_queue_keys_are_rejected(self):
        for key in sorted(fs.RESERVED_EVENT_KEYS):
            with self.subTest(key=key), self.assertRaisesRegex(fs.SchemaError, "reserved"):
                fs.validate_claim_fleet(self.block(**{key: "x"}))

    def test_bad_fields(self):
        for changes in ({"heartbeat_ref": "refs/heads/main"}, {"lease_seconds": 0},
                        {"gate": {"observed_at_utc": "2026-10-19T12:00:00Z", "host": "10.1.2.3"}},
                        {"protocol": "atlas.fleet.v2"}):
            with self.subTest(changes=changes), self.assertRaises(fs.SchemaError):
                fs.validate_claim_fleet(self.block(**changes))


class HeartbeatTests(unittest.TestCase):
    def load(self, name):
        return json.loads((SYNTHETIC / name).read_text(encoding="utf-8"))

    def test_synthetic_heartbeats_validate(self):
        for name in ("heartbeat-alive.json", "heartbeat-stale.json", "heartbeat-rebooted.json"):
            with self.subTest(name=name):
                fs.validate_heartbeat(self.load(name))

    def test_blocked_host_loss_requires_attempts_and_public_scan(self):
        doc = self.load("heartbeat-rebooted.json")
        doc["host_loss_detected"] = []
        with self.assertRaisesRegex(fs.SchemaError, "blocked_host_loss"):
            fs.validate_heartbeat(doc)
        doc = self.load("heartbeat-alive.json")
        doc["gap_report"] = {"gap_start_utc": "2026-10-19T12:00:00Z", "gap_end_utc": "2026-10-19T12:10:00Z",
                             "classification": "git_outage", "note": "remote git@github.com:owner/state.git"}
        with self.assertRaisesRegex(fs.SchemaError, "forbidden public content"):
            fs.validate_heartbeat(doc)
        doc = self.load("heartbeat-alive.json")
        doc["status"] = "lost"
        with self.assertRaises(fs.SchemaError):
            fs.validate_heartbeat(doc)


class NodeProfileTests(unittest.TestCase):
    def test_example_profile_is_valid_but_not_complete(self):
        profile = json.loads(EXAMPLE_PROFILE.read_text(encoding="utf-8"))
        fs.validate_node_profile(profile)
        with self.assertRaisesRegex(fs.SchemaError, "pending operator decisions"):
            fs.validate_node_profile(profile, complete=True)

    def test_completed_profile_passes_complete_mode(self):
        fs.validate_node_profile(complete_profile(), complete=True)

    def test_section_6_2_rules(self):
        base = complete_profile()

        def mutate(function):
            value = copy.deepcopy(base)
            function(value)
            return value
        cases = {
            "preventive >= ceiling": lambda p: p["limits"]["thermal"][0].update(preventive_c=90),
            "claim_max >= preventive": lambda p: p["limits"]["thermal"][0].update(claim_max_c=89),
            "preventive <= 0": lambda p: p["limits"]["thermal"][0].update(preventive_c=0, claim_max_c=-1),
            "allowed_gpus not subset": lambda p: p["limits"].update(allowed_gpus=[0, 2]),
            "negative reserve": lambda p: p["storage"][0].update(reserve_bytes=-1),
            "short lease": lambda p: p["liveness"].update(lease_seconds=1799),
            "retry": lambda p: p["limits"].update(training_retry="on_failure"),
            "grid without source": lambda p: p["grid"].update(g_co2_per_kwh=61.7),
            "grid zero": lambda p: p["grid"].update(g_co2_per_kwh=0, source="x", reference_period="2025"),
            "absolute path": lambda p: p["hardware"]["software"].update(docker="/usr/bin/docker"),
            "home path": lambda p: p["hardware"]["software"].update(os="~/os"),
            "windows path": lambda p: p["hardware"]["software"].update(os="C:\\Users\\x"),
            "url credential": lambda p: p["grid"].update(source="https://user:pw@example.org/factor"),
            "ip address": lambda p: p["hardware"]["cpu"].update(model="node at 192.168.0.10"),
            "ipv6": lambda p: p["hardware"]["cpu"].update(model="fe80::1"),
            "ssh key": lambda p: p["grid"].update(region="ssh-ed25519 AAAA"),
            "token key": lambda p: p.update(api_token="x"),
            "null without reason": lambda p: p["hardware"]["cpu"].update(model=None),
            "unknown reason": lambda p: p["null_reasons"].update({"grid.source": "because"}),
            "status": lambda p: p.update(status="offline"),
            "duplicate storage": lambda p: p["storage"].append({"key": "runs", "reserve_bytes": 0}),
            "node id hostname": lambda p: p.update(node_id="x299.lab"),
        }
        for name, function in cases.items():
            with self.subTest(name), self.assertRaises(fs.SchemaError):
                fs.validate_node_profile(mutate(function))

    def test_grid_factor_with_source_and_period_is_accepted(self):
        profile = complete_profile()
        profile["grid"] = {"g_co2_per_kwh": 61.7, "source": "operator-chosen-source-id", "region": "national-grid",
                           "reference_period": "2025", "retrieved_on": "2026-10-19"}
        profile["null_reasons"] = {key: value for key, value in profile["null_reasons"].items()
                                   if not key.startswith("grid.")}
        fs.validate_node_profile(profile, complete=True)

    def test_version_strings_are_not_ip_addresses(self):
        profile = complete_profile()
        profile["images"] = [{"image_id": "sha256:" + "3" * 64, "probe": {"opencv": "4.12.0.88", "torch": "2.5.1"}}]
        fs.validate_node_profile(profile)
        self.assertEqual([("x.note", "ipv4")], fs.forbidden_strings({"note": "4.12.0.88"}, "x"))
        self.assertEqual([], fs.forbidden_strings({"utc": "2026-10-19T12:38:41Z", "bus": "00000000:17:00.0"}))

    def test_public_card(self):
        profile = complete_profile()
        card = dict(copy.deepcopy(profile), profile_sha256=digest(profile), registered_at_utc="2026-10-19T13:00:00Z")
        fs.validate_public_card(card)
        card["profile_sha256"] = "z"
        with self.assertRaises(fs.SchemaError):
            fs.validate_public_card(card)


class TelemetryRecordTests(unittest.TestCase):
    def test_fast_record_from_section_5_2(self):
        fs.validate_telemetry_record(fast_record())

    def test_null_requires_reason_from_closed_enum(self):
        record = fast_record()
        record["gpus"][0]["power_w"] = None
        with self.assertRaisesRegex(fs.SchemaError, "without null_reasons"):
            fs.validate_telemetry_record(record)
        record["null_reasons"] = {"gpus[0].power_w": "not_supported_by_device"}
        fs.validate_telemetry_record(record)
        record["null_reasons"] = {"gpus[0]": "stale"}
        fs.validate_telemetry_record(record)
        record["null_reasons"] = {"gpus[0].power_w": "zero"}
        with self.assertRaisesRegex(fs.SchemaError, "unknown reason"):
            fs.validate_telemetry_record(record)

    def test_nan_inf_and_envelope_errors(self):
        for changes in ({"dt_s": math.inf}, {"mono_s": math.nan}, {"seq": -1}, {"utc": "2026-10-19 12:00:00"},
                        {"boot_id": "not-a-boot"}, {"type": "medium"}, {"node_id": "Host_Name"}):
            with self.subTest(changes=changes), self.assertRaises(fs.SchemaError):
                fs.validate_telemetry_record(fast_record(**changes))

    def test_event_record(self):
        event = {"schema": "atlas.telemetry.v1", "type": "event", "node_id": "x299",
                 "boot_id": "8c40d0a6-6a4f-46b4-adaf-bae6c4b8a2a6", "seq": 3, "utc": "2026-10-19T12:00:00.512Z",
                 "mono_s": 10.0, "event": "reboot_detected", "prev_boot_id": "aaaaaaaa-0000-4000-8000-00000000000a",
                 "prev_last_sample_utc": "2026-10-18T02:15:30.512Z", "gap_s": 1000.0, "null_reasons": {}}
        fs.validate_telemetry_record(event)
        with self.assertRaises(fs.SchemaError):
            fs.validate_telemetry_record(dict(event, event="reboot"))
        missing = dict(event)
        del missing["gap_s"]
        with self.assertRaises(fs.SchemaError):
            fs.validate_telemetry_record(missing)

    def test_stream_seq_strictly_increasing_per_boot_and_type(self):
        first, second = fast_record(seq=1, mono_s=1.0), fast_record(seq=2, mono_s=1.5)
        fs.validate_telemetry_stream([first, second])
        with self.assertRaisesRegex(fs.SchemaError, "strictly increasing"):
            fs.validate_telemetry_stream([second, first])
        with self.assertRaisesRegex(fs.SchemaError, "backwards"):
            fs.validate_telemetry_stream([first, fast_record(seq=2, mono_s=0.5)])
        other_boot = fast_record(seq=1, mono_s=0.1, boot_id="bbbbbbbb-0000-4000-8000-00000000000b")
        fs.validate_telemetry_stream([first, second, other_boot])

    def test_latest_contract(self):
        latest = {"schema": "atlas.sentinel-latest.v1", "node_id": "x299", "boot_id": "8c40d0a6-6a4f-46b4-adaf-bae6c4b8a2a6",
                  "seq": 9, "utc": "2026-10-19T12:00:00.512Z", "mono_s": 86400.5,
                  "gpus": [{"index": 0, "uuid": "GPU-00000000-0000-0000-0000-000000000000", "temp_c": 41,
                            "util_pct": 0, "compute_apps": None}],
                  "cpu_pkg_max_temp_c": 38, "disks": [{"key": "runs", "free_bytes": 43}],
                  "null_reasons": {"gpus[0].compute_apps": "first_sample"}}
        fs.validate_sentinel_latest(latest)
        latest["null_reasons"] = {}
        with self.assertRaises(fs.SchemaError):
            fs.validate_sentinel_latest(latest)


class SummaryTests(unittest.TestCase):
    def test_section_5_4_example_with_null_co2(self):
        fs.validate_job_summary(summary())

    def test_co2_without_factor_must_be_null(self):
        value = summary()
        value["co2"]["g"] = 0.0
        with self.assertRaisesRegex(fs.SchemaError, "grid_factor_missing"):
            fs.validate_job_summary(value)

    def test_co2_is_measured_kwh_times_factor_and_never_adds_estimates(self):
        value = summary()
        value["co2"] = {"g": 0.2607 * 61.7, "boundary": ["gpu_board"], "basis": "measured_only",
                        "grid_factor_g_per_kwh": 61.7, "factor_source_id": "operator-chosen-source-id"}
        fs.validate_job_summary(value)
        value["energy"]["cpu_kwh_estimated"] = {"value": 0.05, "status": "estimated_model",
                                                "method": "cpu_tdp_util_linear_v1"}
        fs.validate_job_summary(value)
        value["co2"]["g"] = (0.2607 + 0.05) * 61.7
        with self.assertRaisesRegex(fs.SchemaError, "never added"):
            fs.validate_job_summary(value)

    def test_energy_status_rules(self):
        cases = []
        value = summary()
        value["energy"]["gpu_kwh"] = {"value": 0.1, "status": "estimated_model", "method": "m"}
        cases.append(value)
        value = summary()
        value["energy"]["dram_kwh"] = {"value": 0.1, "status": "unavailable", "reason": "permission_denied"}
        cases.append(value)
        value = summary()
        value["energy"]["cpu_kwh_estimated"] = {"value": 0.1, "status": "estimated_model", "method": None}
        cases.append(value)
        value = summary()
        value["boundary"]["note"] = "Board energy."
        cases.append(value)
        for index, case in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(fs.SchemaError):
                fs.validate_job_summary(case)

    def test_public_whitelist(self):
        block = {"schema": "atlas.job-telemetry-summary.v1", "node_id": "x299", "hardware_class": "gm200-12g",
                 "duration_s": 5321.4, "coverage_fraction": 0.998, "max_gap_s": 1.4, "gpu_temp_max_c": 83,
                 "gpu_temp_p95_c": 81, "cpu_pkg_temp_max_c": 66, "gpu_power_mean_w": 176.3,
                 "energy": {"gpu_kwh": {"value": 0.2607, "status": "measured_integrated"},
                            "cpu_pkg_kwh": {"value": None, "status": "unavailable"},
                            "dram_kwh": {"value": None, "status": "unavailable"}},
                 "co2": {"g": None, "boundary": ["gpu_board"], "grid_factor_g_per_kwh": None, "factor_source_id": None},
                 "throughput_img_s_mean": 41.2, "epochs_completed": 5, "reboot_detected": False}
        fs.validate_public_telemetry(block)
        for changes in ({"train_loss": 0.3}, {"window": {}}, {"node_id": "/srv/x299"},
                        {"energy": {"gpu_kwh": {"value": 1, "status": "measured_integrated", "source": "x"}}},
                        {"co2": {"g": None, "basis": "measured_only"}}, {"hardware_class": "a@b"}):
            with self.subTest(changes=changes), self.assertRaises(fs.SchemaError):
                fs.validate_public_telemetry(dict(block, **changes))


def _load_b7a47f5_schema():
    spec = importlib.util.spec_from_file_location("schema_b7a47f5", SYNTHETIC / "queue-records" / "schema_b7a47f5.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QueueRecordCompatibilityTests(unittest.TestCase):
    """Cases 8.3-1 and 8.3-2 (F0 acceptance)."""

    def extract(self, markdown):
        marker = "<!-- augment-atlas-record-v1 -->\n```json\n"
        return json.loads(markdown.split(marker, 1)[1].split("\n```", 1)[0])

    def test_legacy_records_rerender_byte_for_byte(self):
        for name in ("legacy-b7a47f5-pending.md", "legacy-b7a47f5-running.md"):
            with self.subTest(name=name):
                markdown = (SYNTHETIC / "queue-records" / name).read_text(encoding="utf-8")
                self.assertEqual(markdown, render_record(parse_record(markdown)))

    def test_claim_commit_summary_line_is_not_canonical_but_json_is(self):
        """A claim commit renders ``str(owner)`` in the caller's insertion order; a later
        re-render sorts it. Only the human summary line differs; the JSON block is identical."""
        markdown = (SYNTHETIC / "queue-records" / "fleet-v1-claimed.md").read_text(encoding="utf-8")
        again = render_record(parse_record(markdown))
        self.assertEqual(self.extract(markdown), self.extract(again))
        self.assertEqual(markdown.split("<!-- augment-atlas-record-v1 -->")[1],
                         again.split("<!-- augment-atlas-record-v1 -->")[1])
        differing = [(a, b) for a, b in zip(markdown.splitlines(), again.splitlines()) if a != b]
        self.assertEqual(1, len(differing))
        self.assertTrue(differing[0][0].startswith("Attempt: "))

    def test_fleet_v1_record_passes_unchanged_b7a47f5_validator(self):
        old = _load_b7a47f5_schema()
        markdown = (SYNTHETIC / "queue-records" / "fleet-v1-claimed.md").read_text(encoding="utf-8")
        record = self.extract(markdown)
        old.validate_record(copy.deepcopy(record))
        self.assertEqual(record["job_id"], old.prepare_job(record["spec"])["job_id"])
        fs.validate_requirements(record["spec"]["requirements"])
        fs.validate_scheduling(record["spec"]["scheduling"])
        claim = record["history"][0]
        fs.validate_claim_fleet(claim["fleet"])
        self.assertEqual(claim["fleet"]["requirements_sha256"], digest(record["spec"]["requirements"]))
        self.assertEqual(record["owner"]["server"], record["owner"]["node_id"])
        for name in ("legacy-b7a47f5-pending.md", "legacy-b7a47f5-running.md"):
            old.validate_record(self.extract((SYNTHETIC / "queue-records" / name).read_text(encoding="utf-8")))

    def test_requirements_do_not_change_job_identity(self):
        old = _load_b7a47f5_schema()
        record = self.extract((SYNTHETIC / "queue-records" / "fleet-v1-claimed.md").read_text(encoding="utf-8"))
        spec = copy.deepcopy(record["spec"])
        for key in ("requirements", "scheduling", "job_id"):
            spec.pop(key)
        self.assertEqual(record["job_id"], old.prepare_job(spec)["job_id"])


if __name__ == "__main__":
    unittest.main()
