"""F1: node profile probe/validate/hash/card over the REAL sanitized x299 capture."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from augment_atlas import fleet_schema as fs
from augment_atlas import node_profile as npf
from augment_atlas import telemetry_v1 as tv

FIXTURES = Path(__file__).parent / "fixtures" / "telemetry_v1"
X299 = FIXTURES / "x299"
EXAMPLE = Path(__file__).parent.parent / "configs" / "node_profile.example.json"
NOW = "2026-10-19T12:38:41Z"
#: Read-only commands the probe may run (argv[0] basename + first argument).
ALLOWED_COMMANDS = {("nvidia-smi", "--query-gpu"), ("nvidia-smi", "--help-query-gpu"), ("nvidia-smi", "-q"),
                    ("nvidia-smi", "--version"), ("lscpu", None), ("docker", "--version"),
                    ("nvidia-ctk", "--version"), ("timedatectl", "show"), ("sensors", "-j"), ("sensors", None)}


def x299_probe(**kwargs):
    readers = tv.ReplayReaders(X299)
    return npf.probe(readers, "x299", NOW, **kwargs), readers


class ProbeX299Tests(unittest.TestCase):
    def test_hardware_facts_match_the_capture(self):
        profile, _ = x299_probe()
        hardware = profile["hardware"]
        self.assertEqual({"model": "Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz", "sockets": 1, "cores": 10,
                          "threads": 20, "tdp_w": None}, hardware["cpu"])
        self.assertEqual(32552160 * 1024, hardware["ram_bytes"])
        self.assertEqual([0, 1], [gpu["index"] for gpu in hardware["gpus"]])
        for gpu in hardware["gpus"]:
            self.assertEqual("NVIDIA GeForce GTX TITAN X", gpu["name"])
            self.assertEqual(12288, gpu["vram_mib"])
            self.assertTrue(gpu["uuid"].startswith("GPU-00000000-0000-0000-0000-00000000000"))
        self.assertEqual("580.178.04", hardware["software"]["nvidia_driver"])
        self.assertEqual([{"package": 0, "high_c": 85.0, "crit_c": 95.0, "source": "hwmon:coretemp"}],
                         hardware["thermal_reference"]["cpu_pkg"])
        self.assertEqual({(92, 97, 83)}, {(g["slowdown_c"], g["shutdown_c"], g["target_c"])
                                          for g in hardware["thermal_reference"]["gpus"]})

    def test_sensor_availability_matches_x299_facts(self):
        sensors = x299_probe()[0]["sensors"]
        for key in ("gpu.temp_c", "gpu.power_w", "gpu.fan_pct", "gpu.clocks_mhz", "gpu.pstate", "gpu.power_limit_w",
                    "gpu.clock_event_reasons", "cpu_pkg.temp_c", "cpu_core.temp_c", "cpu.temp_limits",
                    "acpi_zone.temp_c", "clock.ntp_synchronized", "host.boot_id", "host.var_crash_listing"):
            with self.subTest(key=key):
                self.assertEqual("available", sensors[key])
        expected = {"rapl.package": "permission_denied", "rapl.dram": "permission_denied",
                    "hwmon.fan_rpm": "not_exposed", "hwmon.pwm": "not_exposed",
                    "gpu.ecc_errors": "not_supported_by_device", "gpu.energy_counter": "not_exposed"}
        for key, reason in expected.items():
            with self.subTest(key=key):
                self.assertEqual({"status": "unavailable", "reason": reason},
                                 {k: v for k, v in sensors[key].items() if k != "evidence"})
        # Not in the capture: reported as unverified, never guessed.
        self.assertEqual("unverified", sensors["cpu.freq_mhz"])
        self.assertEqual("unverified", sensors["kernel_log"])
        self.assertIn("clocks_event_reasons.active", sensors["nvidia_smi_fields"])
        self.assertNotIn("total_energy_consumption", sensors["nvidia_smi_fields"])

    def test_policy_is_never_auto_detected(self):
        profile, _ = x299_probe()
        self.assertEqual("draining", profile["status"])
        self.assertIsNone(profile["hardware_class"])
        self.assertIsNone(profile["guardian"])
        self.assertEqual({"g_co2_per_kwh": None, "source": None, "region": None, "reference_period": None,
                          "retrieved_on": None}, profile["grid"])
        reasons = profile["null_reasons"]
        for key in ("grid.g_co2_per_kwh", "hardware_class", "guardian", "limits.thermal", "limits.allowed_gpus",
                    "limits.max_concurrent_trainings", "storage[0].reserve_bytes", "hardware.cpu.tdp_w"):
            with self.subTest(key=key):
                self.assertEqual("requires_operator_decision", reasons[key])
        self.assertEqual("not_captured", reasons["hardware.gpus[0].compute_capability"])
        self.assertEqual("never", profile["limits"]["training_retry"])

    def test_probe_only_runs_read_only_commands(self):
        _, readers = x299_probe()
        for call in readers.calls:
            first = next((part.split("=")[0] for part in call[1:2]), None)
            with self.subTest(call=call):
                self.assertIn((call[0], first), ALLOWED_COMMANDS)

    def test_example_config_is_the_probe_output(self):
        profile, _ = x299_probe()
        self.assertEqual(EXAMPLE.read_text(encoding="utf-8"), npf.dumps(profile))
        self.assertEqual(npf.profile_sha256(profile),
                         npf.profile_sha256(json.loads(EXAMPLE.read_text(encoding="utf-8"))))

    def test_tool_absent_still_yields_a_valid_profile(self):
        def run_cmd(argv, timeout):
            raise FileNotFoundError(argv[0])
        readers = tv.ReplayReaders(X299, run_cmd=run_cmd)
        profile = npf.probe(readers, "x299", NOW)
        self.assertEqual([], profile["hardware"]["gpus"])
        self.assertEqual({"status": "unavailable", "reason": "tool_absent"}, profile["sensors"]["gpu.temp_c"])
        self.assertEqual("available", profile["sensors"]["cpu_pkg.temp_c"])
        self.assertEqual("tool_absent", profile["null_reasons"]["hardware.cpu.model"])

    def test_node_id_must_be_an_alias(self):
        for bad in ("x299.lab.example", "X299", "", "10.0.0.5"):
            with self.subTest(bad=bad), self.assertRaises(fs.SchemaError):
                npf.probe(tv.ReplayReaders(X299), bad, NOW)


class HashAndCardTests(unittest.TestCase):
    def test_hash_is_canonical_and_content_sensitive(self):
        profile, _ = x299_probe()
        shuffled = json.loads(json.dumps(profile, sort_keys=False))
        reordered = dict(reversed(list(shuffled.items())))
        self.assertEqual(npf.profile_sha256(profile), npf.profile_sha256(reordered))
        changed = copy.deepcopy(profile)
        changed["liveness"]["heartbeat_seconds"] = 600
        self.assertNotEqual(npf.profile_sha256(profile), npf.profile_sha256(changed))

    def test_public_card(self):
        profile, _ = x299_probe()
        card = npf.public_card(profile, "2026-10-19T13:00:00Z")
        self.assertEqual(npf.profile_sha256(profile), card["profile_sha256"])
        self.assertTrue(npf.card_matches_profile(card, profile))
        stripped = npf.public_card(profile, "2026-10-19T13:00:00Z", strip_gpu_uuids=True)
        self.assertTrue(all("uuid" not in gpu for gpu in stripped["hardware"]["gpus"]))
        self.assertTrue(npf.card_matches_profile(stripped, profile))
        other = copy.deepcopy(profile)
        other["status"] = "active"
        self.assertFalse(npf.card_matches_profile(card, other))
        self.assertEqual([], fs.forbidden_strings(card))

    def test_validate_rejects_paths_ips_urls_and_incoherent_limits(self):
        profile, _ = x299_probe()
        cases = [lambda p: p["hardware"]["software"].update(docker="/usr/bin/docker"),
                 lambda p: p["hardware"]["cpu"].update(model="see https://example.org"),
                 lambda p: p["hardware"]["cpu"].update(model="10.20.30.40"),
                 lambda p: (p["limits"].update(thermal=[{"sensor": "gpu.temp_c", "ceiling_c": 85, "preventive_c": 89,
                                                         "claim_max_c": 75, "source": "nvidia-smi"}]),
                            p["null_reasons"].pop("limits.thermal")),
                 lambda p: (p["limits"].update(allowed_gpus=[3]), p["null_reasons"].pop("limits.allowed_gpus"))]
        for index, mutate in enumerate(cases):
            value = copy.deepcopy(profile)
            mutate(value)
            with self.subTest(index=index), self.assertRaises(fs.SchemaError):
                npf.validate(value)


class CliTests(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = npf.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_probe_validate_sha_card(self):
        with tempfile.TemporaryDirectory(prefix="atlas-profile-") as tmp:
            target = Path(tmp) / "profile.json"
            code, _, _ = self.run_main("probe", "--node-id", "x299", "--replay", str(X299), "--now", NOW,
                                       "--out", str(target))
            self.assertEqual(0, code)
            self.assertEqual(EXAMPLE.read_text(encoding="utf-8"), target.read_text(encoding="utf-8"))
            code, out, _ = self.run_main("validate", str(target))
            self.assertEqual((0, True), (code, json.loads(out)["valid"]))
            code, _, err = self.run_main("validate", str(target), "--complete")
            self.assertEqual(2, code)
            self.assertIn("pending operator decisions", err)
            code, out, _ = self.run_main("sha256", str(target))
            self.assertEqual(npf.profile_sha256(json.loads(target.read_text(encoding="utf-8"))), out.strip())
            code, out, _ = self.run_main("card", str(target), "--registered-at", "2026-10-19T13:00:00Z")
            self.assertEqual(0, code)
            fs.validate_public_card(json.loads(out))


if __name__ == "__main__":
    unittest.main()
