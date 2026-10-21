"""F2: telemetry v1 readers and parsers (docs/fleet-protocol.md section 8.2, cases 1-15 and 18).

x299 inputs are the REAL sanitized capture; everything under ``synthetic/`` is
SYNTHETIC and labelled as such. Permission errors are replayed by the injected
reader, never produced with chmod, so the suite runs on Windows and Linux.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from augment_atlas import fleet_schema as fs
from augment_atlas import telemetry_v1 as tv

FIXTURES = Path(__file__).parent / "fixtures" / "telemetry_v1"
X299 = FIXTURES / "x299"
SYNTHETIC = FIXTURES / "synthetic"
FAKE_NVIDIA_SMI = SYNTHETIC / "fake-nvidia-smi.py"
BOOT_X299 = "8c40d0a6-6a4f-46b4-adaf-bae6c4b8a2a6"


def read(path):
    return Path(path).read_text(encoding="utf-8")


def slow_record(sections, reasons, src, seq=1, mono=10.0, boot_id=BOOT_X299):
    record = tv.envelope("slow", "x299", boot_id, seq, mono, "2026-10-19T12:38:41.000Z")
    record.update(sections)
    record.update(null_reasons=reasons, src=src)
    return record


class SensorsTests(unittest.TestCase):
    """Cases 1 and 2."""

    def test_sensors_json_x299_package_cores_limits_and_acpitz_zone(self):
        cpu = tv.parse_sensors_json(read(X299 / "sensors-j.json"))
        self.assertEqual([{"package": 0, "temp_c": 32.0, "high_c": 85.0, "crit_c": 95.0}], cpu["packages"])
        self.assertEqual([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], [core["core"] for core in cpu["cores"]])
        self.assertEqual({"high_c": 85.0, "crit_c": 95.0}, cpu["limits"])
        self.assertEqual([{"chip": "acpitz", "label": "temp1", "temp_c": 16.8, "kind": "acpi_zone"}], cpu["zones"])
        self.assertNotIn(16.8, [item["temp_c"] for item in cpu["packages"] + cpu["cores"]])

    def test_text_fallback_is_equivalent(self):
        self.assertEqual(tv.parse_sensors_json(read(X299 / "sensors-j.json")),
                         tv.parse_sensors_text(read(X299 / "sensors.txt")))
        # Same bytes decoded as latin-1 (a broken locale) still parse.
        latin = (X299 / "sensors.txt").read_bytes().decode("latin-1")
        self.assertEqual(tv.parse_sensors_text(read(X299 / "sensors.txt")), tv.parse_sensors_text(latin))

    def test_cli_falls_back_from_json_to_text(self):
        replay = tv.ReplayReaders(X299)

        def run_cmd(argv, timeout):
            if "-j" in argv:
                raise tv.CommandFailed(argv, 1, "Unknown option -j")
            return replay.run_cmd(argv, timeout)
        cpu, reason, src = tv.read_cpu_sensors_cli(tv.Readers(run_cmd=run_cmd))
        self.assertEqual((None, "sensors-text"), (reason, src))
        self.assertEqual(tv.parse_sensors_json(read(X299 / "sensors-j.json")), cpu)

        def absent(argv, timeout):
            raise FileNotFoundError(argv[0])
        self.assertEqual((None, "tool_absent", None), tv.read_cpu_sensors_cli(tv.Readers(run_cmd=absent)))

    def test_hwmon_x299_matches_sensors_structure(self):
        cpu = tv.cpu_from_hwmon(tv.read_hwmon(tv.ReplayReaders(X299)))
        self.assertEqual([{"package": 0, "temp_c": 30.0, "high_c": 85.0, "crit_c": 95.0}], cpu["packages"])
        self.assertEqual(10, len(cpu["cores"]))
        self.assertEqual("acpi_zone", cpu["zones"][0]["kind"])
        zones = tv.read_thermal_zones(tv.ReplayReaders(X299))
        self.assertEqual([{"zone": 0, "type": "acpitz", "temp_c": 16.8}, {"zone": 1, "type": "x86_pkg_temp", "temp_c": 30.0}],
                         zones)


class FanTests(unittest.TestCase):
    """Case 3."""

    def test_x299_has_no_fans(self):
        self.assertEqual((None, "not_exposed"), tv.fans_from_hwmon(tv.read_hwmon(tv.ReplayReaders(X299))))
        sections, reasons, _, _ = tv.SlowCollector(tv.ReplayReaders(X299)).collect(1.0)
        self.assertIsNone(sections["fans"])
        self.assertEqual("not_exposed", reasons["fans"])

    def test_synthetic_fans_and_pwm(self):
        readers = tv.ReplayReaders(SYNTHETIC / "hwmon-with-fans")
        fans, reason = tv.fans_from_hwmon(tv.read_hwmon(readers))
        self.assertIsNone(reason)
        self.assertEqual([{"chip": "nct6798", "fan": 1, "rpm": 1180, "pwm": 128, "pwm_pct": 50.2, "pwm_enable": 5},
                          {"chip": "nct6798", "fan": 2, "rpm": 845, "pwm": 255, "pwm_pct": 100.0, "pwm_enable": 1}],
                         fans)
        cpu = tv.cpu_from_hwmon(tv.read_hwmon(readers))
        self.assertEqual(41.0, cpu["packages"][0]["temp_c"])
        self.assertEqual("other", cpu["zones"][0]["kind"])


class NvidiaSmiTests(unittest.TestCase):
    """Case 4 (+ -q parsing used by the slow tier)."""

    def test_header_and_noheader_parse_by_name_to_the_same_values(self):
        noheader, bad1 = tv.parse_nvidia_csv(read(X299 / "nvidia-smi-query.csv"), tv.FAST_QUERY_FIELDS)
        header, bad2 = tv.parse_nvidia_csv(read(X299 / "nvidia-smi-query-header.csv"))
        self.assertEqual((0, 0), (bad1, bad2))
        self.assertEqual(noheader, header)
        gpu0 = noheader[0]["values"]
        self.assertEqual({"index": 0, "temp_c": 34, "power_w": 14.35, "power_limit_w": 250.0, "fan_pct": 22,
                          "util_pct": 0, "mem_used_mib": 8, "mem_total_mib": 12288, "sm_mhz": 135, "mem_mhz": 405,
                          "pstate": "P8"}, {k: v for k, v in gpu0.items() if k not in {"uuid", "name"}})

    def test_permuted_header(self):
        rows, bad = tv.parse_nvidia_csv(read(SYNTHETIC / "nvidia-smi-permuted-header.csv"))
        self.assertEqual(0, bad)
        by_index = {row["index"]: row["values"] for row in rows}
        self.assertEqual((101.5, 51, 97), (by_index[1]["power_w"], by_index[1]["temp_c"], by_index[1]["util_pct"]))
        self.assertEqual((99.25, 49, 98), (by_index[0]["power_w"], by_index[0]["temp_c"], by_index[0]["util_pct"]))

    def test_na_and_not_supported_become_null_with_reason(self):
        rows, bad = tv.parse_nvidia_csv(read(SYNTHETIC / "nvidia-smi-na.csv"), tv.FAST_QUERY_FIELDS)
        self.assertEqual(0, bad)
        self.assertEqual((None, None), (rows[0]["values"]["power_w"], rows[0]["values"]["fan_pct"]))
        self.assertEqual({"power_w": "not_supported_by_device", "fan_pct": "not_supported_by_device"},
                         rows[0]["reasons"])
        self.assertEqual({"temp_c": "parse_error", "power_limit_w": "not_supported_by_device"}, rows[1]["reasons"])
        self.assertNotIn(0, [rows[0]["values"]["power_w"], rows[0]["values"]["fan_pct"]])

    def test_garbage_lines_are_dropped_and_counted(self):
        rows, bad = tv.parse_nvidia_csv(read(SYNTHETIC / "nvidia-smi-garbage.csv"), tv.FAST_QUERY_FIELDS)
        self.assertEqual([0], [row["index"] for row in rows])
        self.assertEqual(3, bad)

    def test_stream_capture_parses_every_line(self):
        rows, bad = tv.parse_nvidia_csv(read(X299 / "nvidia-smi-stream-10s.csv"), tv.FAST_QUERY_FIELDS)
        self.assertEqual((40, 0), (len(rows), bad))
        self.assertEqual({0, 1}, {row["index"] for row in rows})

    def test_q_output_x299(self):
        gpus = tv.parse_nvidia_q(read(X299 / "nvidia-smi-q-perf-power-temp-ecc.txt"))
        self.assertEqual(2, len(gpus))
        for gpu in gpus:
            self.assertEqual(["gpu_idle"], gpu["clock_event_reasons"])
            self.assertEqual(("P8", 250.0, 150.0, 275.0), (gpu["pstate"], gpu["power_limit_w"],
                                                         gpu["power_min_limit_w"], gpu["power_max_limit_w"]))
            self.assertEqual((92, 97, 83), (gpu["temp_slowdown_c"], gpu["temp_shutdown_c"], gpu["temp_target_c"]))
            self.assertIsNone(gpu["ecc_errors"])
            self.assertEqual({"ecc_errors": "not_supported_by_device"}, gpu["null_reasons"])
        self.assertEqual((14.26, 13.15), (gpus[0]["power_draw_w"], gpus[1]["power_draw_w"]))

    def test_help_query_lists_event_reasons_but_no_energy_counter(self):
        fields = tv.parse_help_query_gpu(read(X299 / "nvidia-smi-help-query-gpu.txt"))
        self.assertIn("clocks_event_reasons.active", fields)
        self.assertIn("clocks_throttle_reasons.active", fields)
        self.assertIn("compute_cap", fields)
        self.assertNotIn("total_energy_consumption", fields)
        self.assertEqual(["gpu_idle", "sw_power_cap"], tv.clock_event_names(0x5))
        self.assertEqual("580.178.04", tv.parse_nvidia_version(read(X299 / "nvidia-smi-version.txt"))["driver"])


class ToolFailureTests(unittest.TestCase):
    """Cases 5, 6 and 14 at reader level (sentinel-level in test_sentinel)."""

    def test_nvidia_smi_absent_gpu_null_tool_absent_others_filled(self):
        def absent(argv, timeout):
            raise FileNotFoundError(argv[0])
        collector = tv.SlowCollector(tv.ReplayReaders(X299, run_cmd=absent))
        sections, reasons, _, src = collector.collect(1.0)
        self.assertIsNone(sections["gpus"])
        self.assertEqual("tool_absent", reasons["gpus"])
        self.assertIsNotNone(sections["memory"])
        self.assertIsNotNone(sections["cpu"]["pkg_temps_c"])
        fs.validate_telemetry_record(slow_record(sections, reasons, src))
        gpus, gpu_reasons = tv.fast_gpu_entries({"status": "tool_absent", "reason": "tool_absent", "gpus": {}}, 5.0)
        self.assertEqual((None, {"gpus": "tool_absent"}), (gpus, gpu_reasons))

    def test_timeout_maps_to_tool_timeout_and_child_is_reaped(self):
        self.assertEqual("tool_timeout", tv.command_reason(subprocess.TimeoutExpired(["x"], 1)))
        started = time.monotonic()
        with mock.patch.dict(os.environ, {"FAKE_NVIDIA_SMI_MODE": "timeout"}),                 self.assertRaises(subprocess.TimeoutExpired):
            tv.run_command([sys.executable, str(FAKE_NVIDIA_SMI), "--version"], timeout=1.0)
        self.assertLess(time.monotonic() - started, 10.0)

    def test_reason_mapping(self):
        self.assertEqual("permission_denied", tv.file_reason(PermissionError()))
        self.assertEqual("not_exposed", tv.file_reason(FileNotFoundError()))
        self.assertEqual("parse_error", tv.file_reason(ValueError()))
        self.assertEqual("collector_error", tv.file_reason(RuntimeError()))
        self.assertEqual("tool_absent", tv.command_reason(FileNotFoundError()))
        self.assertEqual("not_exposed", tv.command_reason(tv.NotCaptured()))
        self.assertEqual("collector_error", tv.command_reason(tv.CommandFailed(["nvidia-smi"], 9)))

    def test_arbitrary_reader_exception_is_isolated(self):
        replay = tv.ReplayReaders(X299)

        def open_file(path):
            if Path(path).name == "meminfo":
                raise RuntimeError("synthetic reader crash")
            return replay.open_file(path)
        readers = tv.ReplayReaders(X299)
        readers.open_file = open_file
        collector = tv.SlowCollector(readers)
        sections, reasons, _, src = collector.collect(1.0)
        self.assertIsNone(sections["memory"])
        self.assertEqual("collector_error", reasons["memory"])
        self.assertIsNotNone(sections["cpu"]["pkg_temps_c"])
        self.assertIsNotNone(sections["gpus"])
        collector._zones = lambda reasons: (_ for _ in ()).throw(RuntimeError("boom"))
        sections, reasons, _, src = collector.collect(11.0)
        self.assertEqual("collector_error", reasons["thermal_zones"])
        self.assertIn(("thermal_zones", "RuntimeError"), collector.errors)
        self.assertIsNotNone(sections["rapl"])
        fs.validate_telemetry_record(slow_record(sections, reasons, src, seq=2, mono=11.0))


class RaplTests(unittest.TestCase):
    """Cases 8 and 9."""

    def test_x299_permission_denied_keeps_max_range(self):
        zones, reasons = tv.read_rapl(tv.ReplayReaders(X299))
        self.assertEqual([("package-0", "intel-rapl:0", 262143328850), ("dram", "intel-rapl:0:0", 65712999613)],
                         [(zone["domain"], zone["zone"], zone["max_range_uj"]) for zone in zones])
        self.assertEqual([None, None], [zone["energy_uj"] for zone in zones])
        self.assertEqual({"rapl[0].energy_uj": "permission_denied", "rapl[1].energy_uj": "permission_denied"}, reasons)

    def test_delta_and_wrap_arithmetic(self):
        self.assertEqual((5, None), tv.rapl_delta(10, 15, 100, 1.0, max_plausible_w=1e-6))
        self.assertEqual((15, "rapl_wrap"), tv.rapl_delta(90, 5, 100, 1.0, max_plausible_w=1e-6))
        self.assertEqual((None, "rapl_suspect_multi_wrap"), tv.rapl_delta(10, 15, 100, 1.0, max_plausible_w=1.0))
        self.assertEqual((None, None), tv.rapl_delta(None, 15, 100, 1.0))

    def test_readable_counter_with_wraparound(self):
        collector = tv.SlowCollector(tv.ReplayReaders(SYNTHETIC / "powercap-readable"))
        first, reasons, events, _ = collector.collect(0.0)
        self.assertEqual([None, None], [zone["delta_uj"] for zone in first["rapl"]])
        self.assertEqual("first_sample", reasons["rapl[0].delta_uj"])
        second, _, events, _ = collector.collect(10.0)
        package, dram = second["rapl"]
        self.assertEqual((262143328850 - 262143000000) + 1500000, package["delta_uj"])
        self.assertEqual(2000000, dram["delta_uj"])
        self.assertEqual([("rapl_wrap", "package-0")], [(event["event"], event["domain"]) for event in events])
        third, _, events, _ = collector.collect(20.0)
        self.assertEqual([20000000, 2000000], [zone["delta_uj"] for zone in third["rapl"]])
        self.assertEqual([], events)

    def test_multi_wrap_suspect_discards_delta(self):
        collector = tv.SlowCollector(tv.ReplayReaders(SYNTHETIC / "powercap-multiwrap"))
        collector.collect(0.0)
        sections, reasons, events, src = collector.collect(600.0)
        self.assertIsNone(sections["rapl"][0]["delta_uj"])
        self.assertEqual(["rapl_suspect_multi_wrap"], [event["event"] for event in events])
        self.assertEqual("stale", reasons["rapl[0].delta_uj"])


class EnergyTests(unittest.TestCase):
    """Cases 10 and 18 + classification (ADR D10)."""

    def test_trapezoid_exact_and_long_gap_excluded(self):
        series = [(0.0, 100.0), (0.5, 110.0), (1.0, 120.0), (4.0, 120.0), (4.5, 100.0)]
        result = tv.integrate_power(series)
        self.assertAlmostEqual(52.5 + 57.5 + 55.0, result["energy_j"], places=9)
        self.assertAlmostEqual(165.0 / 3.6e6, result["kwh"], places=15)
        self.assertAlmostEqual(1.5, result["covered_s"], places=9)
        self.assertAlmostEqual(1.5 / 4.5, result["coverage_fraction"], places=9)
        self.assertEqual(1, result["intervals_excluded"])

    def test_missing_power_is_excluded_not_zero(self):
        result = tv.integrate_power([(0.0, 100.0), (0.5, None), (1.0, 100.0), (1.5, 100.0)])
        self.assertAlmostEqual(50.0, result["energy_j"], places=9)
        self.assertAlmostEqual(0.5 / 1.5, result["coverage_fraction"], places=9)

    def test_constant_power_one_hour(self):
        series = [(i * 0.5, 250.0) for i in range(7201)]
        self.assertAlmostEqual(0.25, tv.integrate_power(series)["kwh"], places=12)

    def test_utc_jump_does_not_change_integrals(self):
        records = []
        for i in range(9):
            utc = 1792400000.0 + i * 0.5 - (3600.0 if i >= 4 else 0.0)  # NTP steps the wall clock back 1 h
            record = tv.envelope("fast", "x299", BOOT_X299, i + 1, 1000.0 + i * 0.5, tv.utc_label(utc))
            record.update(dt_s=None if i == 0 else 0.5, cpu_pkg=None,
                          gpus=[{"index": 0, "temp_c": 50, "power_w": 200.0, "util_pct": 99, "mem_used_mib": 1,
                                 "sm_mhz": 1, "mem_mhz": 1, "fan_pct": 1, "pstate": "P2", "age_s": 0.1}],
                          null_reasons={"cpu_pkg": "not_exposed", **({"dt_s": "first_sample"} if i == 0 else {})},
                          src={"gpu": "test"})
            records.append(record)
        fs.validate_telemetry_stream(records)
        result = tv.integrate_power(tv.gpu_power_series(records, 0))
        self.assertAlmostEqual(200.0 * 4.0, result["energy_j"], places=9)
        self.assertEqual(1.0, result["coverage_fraction"])

    def test_energy_classification(self):
        integrated = tv.integrate_power([(0.0, 100.0), (1.0, 100.0)])
        self.assertEqual("measured_counter", tv.energy_entry(counter_delta_j=3.6e6)["status"])
        self.assertEqual(1.0, tv.energy_entry(counter_delta_j=3.6e6)["value"])
        entry = tv.energy_entry(integrated=integrated, source="nvidia-smi power.draw 0.5s")
        self.assertEqual(("measured_integrated", 1.0), (entry["status"], entry["coverage_fraction"]))
        self.assertEqual("estimated_model", tv.energy_entry(estimate_j=10.0, method="cpu_tdp_util_linear_v1")["status"])
        with self.assertRaises(ValueError):
            tv.energy_entry(estimate_j=10.0)
        empty = tv.integrate_power([(0.0, None), (0.5, None)])
        self.assertEqual({"value": None, "status": "unavailable", "reason": "permission_denied"},
                         tv.energy_entry(integrated=empty, reason="permission_denied"))
        estimate = tv.estimate_cpu_energy_tdp_util_linear_v1(140.0, [(0.0, 50.0), (10.0, 50.0)])
        self.assertAlmostEqual(700.0, estimate, places=9)
        self.assertIsNone(tv.estimate_cpu_energy_tdp_util_linear_v1(None, [(0.0, 50.0)]))


class ProcfsTests(unittest.TestCase):
    """Case 11 + memory/boot facts."""

    def test_cpu_util_from_real_stat_pair(self):
        first = tv.parse_proc_stat(read(X299 / "proc" / "stat.0"))
        second = tv.parse_proc_stat(read(X299 / "proc" / "stat.1"))
        self.assertEqual((1792362334, 20), (first["btime"], first["cpus"]))
        self.assertEqual(round(100.0 * (1 - 2010 / 2011), 4), tv.cpu_util_pct(first, second))
        self.assertIsNone(tv.cpu_util_pct(None, second))
        self.assertIsNone(tv.cpu_util_pct(second, second))

    def test_first_sample_null_with_reason_then_value(self):
        collector = tv.SlowCollector(tv.ReplayReaders(X299))
        first, reasons, _, src = collector.collect(0.0)
        self.assertIsNone(first["cpu"]["util_pct"])
        self.assertEqual("first_sample", reasons["cpu.util_pct"])
        self.assertEqual("first_sample", reasons["dt_s"])
        fs.validate_telemetry_record(slow_record(first, reasons, src, seq=1, mono=0.0))
        second, reasons, _, src = collector.collect(1.0)
        self.assertAlmostEqual(0.0497, second["cpu"]["util_pct"], places=4)
        self.assertEqual(1.0, second["dt_s"])
        self.assertEqual({"high": 85.0, "crit": 95.0}, second["cpu"]["pkg_temp_limits_c"])
        fs.validate_telemetry_record(slow_record(second, reasons, src, seq=2, mono=1.0))

    def test_meminfo_uptime_loadavg_boot_id(self):
        memory = tv.parse_meminfo(read(X299 / "proc" / "meminfo"))
        self.assertEqual(32552160 * 1024, memory["ram_total_bytes"])
        self.assertEqual(3938300 * 1024, memory["swap_free_bytes"])
        self.assertEqual(51198.05, tv.parse_uptime(read(X299 / "proc" / "uptime")))
        self.assertEqual([0.05, 0.03, 0.0], tv.parse_loadavg(read(X299 / "proc" / "loadavg")))
        self.assertEqual((BOOT_X299, None), tv.read_boot_id(tv.ReplayReaders(X299)))
        self.assertTrue(tv.parse_timedatectl(read(X299 / "timedatectl.txt")))

    def test_lscpu_portuguese_and_english(self):
        parsed = tv.parse_lscpu(read(X299 / "lscpu.txt"))
        self.assertEqual({"model": "Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz", "sockets": 1, "cores_per_socket": 10,
                          "threads_per_core": 2, "cpus": 20}, parsed)
        english = "CPU(s): 8\nModel name: Synthetic CPU\nSocket(s): 1\nCore(s) per socket: 4\nThread(s) per core: 2\n"
        self.assertEqual(4, tv.parse_lscpu(english)["cores_per_socket"])


class RebootAndCrashTests(unittest.TestCase):
    """Cases 12 and 13."""

    def test_reboot_detected_with_prev_sample_and_gap(self):
        previous = json.loads(read(SYNTHETIC / "sentinel-state-prev.json"))
        boot_b = read(SYNTHETIC / "boot_id.b").strip()
        now = tv.parse_utc("2026-10-18T02:20:30.512Z")
        event = tv.detect_reboot(previous, boot_b, now)
        self.assertEqual({"prev_boot_id": read(SYNTHETIC / "boot_id.a").strip(),
                          "prev_last_sample_utc": "2026-10-18T02:15:30.512Z", "gap_s": 300.0}, event)
        self.assertIsNone(tv.detect_reboot(previous, read(SYNTHETIC / "boot_id.a").strip(), now))
        self.assertIsNone(tv.detect_reboot(None, boot_b, now))
        self.assertIsNone(tv.detect_reboot(previous, "unavailable", now))

    def test_new_crash_file_counted_names_only_in_events(self):
        replay = tv.ReplayReaders(X299)
        baseline = replay.listdir(replay.crash_dir)
        self.assertEqual(61, len(baseline))
        listings = iter([baseline, baseline + ["_usr_bin_synthetic.1003.crash"]])
        readers = tv.ReplayReaders(X299)
        original = readers.listdir
        readers.listdir = lambda path: next(listings) if Path(path) == readers.crash_dir else original(path)
        collector = tv.SlowCollector(readers)
        first, _, events, _ = collector.collect(0.0)
        self.assertEqual(({"var_crash_count": 61, "var_crash_new": 0}, []), (first["crash"], events))
        second, reasons, events, src = collector.collect(10.0)
        self.assertEqual({"var_crash_count": 62, "var_crash_new": 1}, second["crash"])
        self.assertEqual([{"event": "var_crash_new", "names": ["_usr_bin_synthetic.1003.crash"], "count": 1}], events)
        self.assertNotIn("synthetic.1003", json.dumps(slow_record(second, reasons, src)))

    def test_ls_parser_drops_owners(self):
        names = tv.parse_ls_names(read(X299 / "var-crash-ls.txt"))
        self.assertIn("_usr_bin_nvidia-smi.1003.crash", names)
        self.assertNotIn(".", names)
        self.assertFalse(any("whoopsie" in name or "atlasuser" in name for name in names))


class ReplayAndSchemaTests(unittest.TestCase):
    """Case 15 (records produced from the real capture validate) + replay mechanics."""

    def test_slow_record_from_x299_validates_and_has_expected_nulls(self):
        collector = tv.SlowCollector(tv.ReplayReaders(X299), disks={"runs": str(X299)}, reserves={"runs": 0})
        collector.collect(0.0, {0: "GPU-00000000-0000-0000-0000-000000000000"})
        sections, reasons, _, src = collector.collect(10.0, {0: "GPU-00000000-0000-0000-0000-000000000000"})
        record = slow_record(sections, reasons, src, seq=2)
        fs.validate_telemetry_record(record)
        self.assertEqual("permission_denied", reasons["rapl[0].energy_uj"])
        self.assertEqual("not_exposed", reasons["fans"])
        self.assertEqual("not_supported_by_device", reasons["gpus[0].ecc_errors"])
        self.assertEqual("disabled_by_config", reasons["containers"])
        self.assertEqual("disabled_by_config", reasons["kernel_xid"])
        self.assertEqual(["acpitz", "x86_pkg_temp"], [zone["type"] for zone in sections["thermal_zones"]])
        self.assertEqual("runs", sections["disks"][0]["key"])
        self.assertNotIn(str(X299), json.dumps(record))

    def test_fast_entries_mark_stale_gpu_and_field_reasons(self):
        snapshot = {"status": "ok", "reason": None, "gpus": {
            0: {"mono": 9.9, "values": {"temp_c": 40, "power_w": None, "util_pct": 1, "mem_used_mib": 1, "sm_mhz": 1,
                                        "mem_mhz": 1, "fan_pct": 1, "pstate": "P8"},
                "reasons": {"power_w": "not_supported_by_device"}},
            1: {"mono": 1.0, "values": {"temp_c": 40}, "reasons": {}}}}
        gpus, reasons = tv.fast_gpu_entries(snapshot, 10.0)
        self.assertEqual({"gpus[0].power_w": "not_supported_by_device", "gpus[1]": "stale"}, reasons)
        self.assertEqual(40, gpus[0]["temp_c"])
        self.assertIsNone(gpus[1]["temp_c"])

    def test_replay_accepts_colon_and_pua_spellings(self):
        with tempfile.TemporaryDirectory(prefix="atlas-replay-") as tmp:
            for spelling in ("intel-rapl" + tv.COLON_PUA + "0",):
                directory = Path(tmp) / "sys" / "class" / "powercap" / spelling
                directory.mkdir(parents=True)
                (directory / "name").write_text("package-0\n", encoding="utf-8")
                (directory / "max_energy_range_uj").write_text("100\n", encoding="utf-8")
                (directory / "energy_uj").write_text("7\n", encoding="utf-8")
            zones, reasons = tv.read_rapl(tv.ReplayReaders(tmp))
            self.assertEqual(([{"domain": "package-0", "zone": "intel-rapl:0", "energy_uj": 7, "max_range_uj": 100}], {}),
                             (zones, reasons))

    def test_replay_distinguishes_absent_from_not_captured(self):
        readers = tv.ReplayReaders(X299)
        with self.assertRaises(FileNotFoundError) as absent:
            readers.read(readers.sys("class", "hwmon", "hwmon0", "fan1_input"))
        self.assertNotIsInstance(absent.exception, tv.NotCaptured)
        with self.assertRaises(tv.NotCaptured):
            readers.read(readers.proc("cpuinfo"))
        with self.assertRaises(tv.NotCaptured):
            readers.run_cmd(["nvidia-smi", "-pl", "200"], 1)

    def test_utc_label_format(self):
        label = tv.utc_label(1792400000.5126)
        self.assertRegex(label, fs.UTC_Z)
        self.assertTrue(label.endswith(".512Z"))
        self.assertAlmostEqual(1792400000.512, tv.parse_utc(label), places=3)


if __name__ == "__main__":
    unittest.main()
