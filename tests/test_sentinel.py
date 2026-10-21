"""F3: sentinel process (docs/fleet-protocol.md section 8.2, cases 6, 7, 12, 14, 16, 17).

GPU data comes from ``synthetic/fake-nvidia-smi.py`` (SYNTHETIC); procfs/sysfs
come from the REAL sanitized x299 capture through ``ReplayReaders``. These are
host-side tests: they prove cadence/robustness of the collector on this
machine, not the behaviour of a real driver.

The cadence test runs ATLAS_SENTINEL_TEST_SECONDS (default 20) seconds. The
120 s check of slice F3 is manual::

    ATLAS_SENTINEL_TEST_SECONDS=120 PYTHONPATH=src python -m unittest tests.test_sentinel.CadenceTests -v
"""
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from augment_atlas import fleet_schema as fs
from augment_atlas import sentinel as sn
from augment_atlas import telemetry_v1 as tv

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "telemetry_v1"
X299 = FIXTURES / "x299"
SYNTHETIC = FIXTURES / "synthetic"
FAKE = [sys.executable, str(SYNTHETIC / "fake-nvidia-smi.py")]
SCRIPT = ROOT / "scripts" / "sentinel-observe-only.sh"
FORBIDDEN_NVIDIA_ARGS = {"-pl", "--power-limit", "-pm", "--persistence-mode", "-r", "--gpu-reset", "-ac", "-rac",
                         "-lgc", "-rgc", "-lmc", "-c", "--compute-mode", "-e", "--ecc-config", "--auto-boost-default"}


def load(out_dir):
    records = {"fast": [], "slow": [], "events": []}
    for boot in sorted(path for path in Path(out_dir).iterdir() if path.is_dir()):
        for prefix in records:
            for segment in sn.segment_files(boot, prefix):
                records[prefix].extend(sn.iter_segment_lines(segment))
    return records


def gaps(fast):
    return [b["mono_s"] - a["mono_s"] for a, b in zip(fast, fast[1:]) if a["boot_id"] == b["boot_id"]]


class RecordingReaders(tv.ReplayReaders):
    """x299 replay for files; real subprocesses (the fake tool) for commands, recorded."""

    def __init__(self, root=X299, run_cmd=None):
        self.commands_seen = []
        self._delegate = run_cmd or tv.run_command
        super().__init__(root, run_cmd=self._record)

    def _record(self, argv, timeout):
        self.commands_seen.append([str(part) for part in argv])
        return self._delegate(argv, timeout)


class SentinelHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="atlas-sentinel-")
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "sentinel"

    def run_sentinel(self, duration, env=None, readers=None, popen=None, **config):
        config.setdefault("nvidia_smi", FAKE)
        config.setdefault("nice", 0)
        config.setdefault("slow_interval_s", 2.0)
        cfg = sn.SentinelConfig(self.out, "sim-a", duration_s=duration, **config)
        sentinel = sn.Sentinel(cfg, readers or RecordingReaders(), popen=popen or subprocess.Popen)
        with mock.patch.dict(os.environ, env or {}):
            summary = sentinel.run()
        return sentinel, summary

    def assert_all_valid(self, records):
        everything = records["fast"] + records["slow"] + records["events"]
        for record in everything:
            fs.validate_telemetry_record(record)
        fs.validate_telemetry_stream(everything)


class CadenceTests(SentinelHarness):
    def test_short_run_cadence_schema_latest_and_observe_only(self):
        duration = float(os.environ.get("ATLAS_SENTINEL_TEST_SECONDS", "20"))
        launched = []

        def popen(argv, **kwargs):
            launched.append(list(argv))
            return subprocess.Popen(argv, **kwargs)
        readers = RecordingReaders()
        sentinel, summary = self.run_sentinel(duration, readers=readers, popen=popen,
                                              disks={"scratch": self.tmp.name}, reserves={"scratch": 0})
        records = load(self.out)
        fast = records["fast"]
        measured = gaps(fast)
        max_gap = max(measured)
        sys.stderr.write(f"\n[sentinel cadence] duration={duration}s fast={len(fast)} slow={len(records['slow'])} "
                         f"max_gap={max_gap:.3f}s p50={sorted(measured)[len(measured) // 2]:.3f}s "
                         f"cpu_pct={summary['cpu_pct']} fsyncs={summary['fsyncs_fast']}\n")
        self.assertEqual("duration", summary["reason"])
        self.assertLessEqual(max_gap, 1.0, "fast-tier gap criterion (F3)")
        self.assertGreaterEqual(len(fast), int(duration / 0.5) - 3)
        self.assertAlmostEqual(max_gap, summary["max_fast_gap_s"], delta=0.01)
        self.assert_all_valid(records)
        with_gpu = [record for record in fast if record["gpus"] and all(g["temp_c"] is not None for g in record["gpus"])]
        self.assertGreaterEqual(len(with_gpu), 0.9 * len(fast))
        self.assertLessEqual(summary["fsyncs_fast"], duration / 5.0 + 2)
        names = [event["event"] for event in records["events"]]
        self.assertEqual(("collector_start", "collector_stop"), (names[0], names[-1]))
        self.assertNotIn("sample_gap", names)
        self.assertNotIn("gpu_stream_gap", names)
        self.assertNotIn("reboot_detected", names)
        latest = json.loads((self.out / "latest.json").read_text(encoding="utf-8"))
        fs.validate_sentinel_latest(latest)
        self.assertEqual("GPU-00000000-0000-0000-0000-000000000000", latest["gpus"][0]["uuid"])
        self.assertEqual(0, latest["gpus"][0]["compute_apps"])
        self.assertEqual("scratch", latest["disks"][0]["key"])
        self.assertEqual("ok", sn.read_latest(self.out / "latest.json", now_mono=latest["mono_s"] + 1)["status"])
        state = json.loads((self.out / "state.json").read_text(encoding="utf-8"))
        self.assertEqual((fast[-1]["seq"], True), (state["last_fast_seq"], state["final"]))
        self.assertNotIn(self.tmp.name, (self.out / "latest.json").read_text(encoding="utf-8"))
        # Observe-only: one stream child with a pure query; every other command is a read-only query.
        self.assertEqual(1, len(launched))
        self.assertTrue(any(arg.startswith("--query-gpu=") for arg in launched[0]))
        self.assertIn("-lms", launched[0])
        for argv in readers.commands_seen + launched:
            with self.subTest(argv=argv):
                self.assertEqual(FAKE, argv[:2])
                self.assertFalse(FORBIDDEN_NVIDIA_ARGS & set(argv[2:]))
                self.assertTrue(argv[2] in {"-q"} or argv[2].startswith("--query-"))
        self.assertEqual([], [record for record in records["slow"] if record["gpus"] is None])
        self.assertTrue(all(code is not None for code in sentinel.stream.reaped))


class WatchdogTests(SentinelHarness):
    def test_stalled_stream_is_killed_reaped_and_restarted(self):
        state = Path(self.tmp.name) / "fake-state"
        sentinel, summary = self.run_sentinel(9.0, env={"FAKE_NVIDIA_SMI_MODE": "stall",
                                                         "FAKE_NVIDIA_SMI_STALL_AFTER": "3",
                                                         "FAKE_NVIDIA_SMI_STATE": str(state)}, slow_interval_s=5.0)
        records = load(self.out)
        stream_gaps = [event for event in records["events"] if event["event"] == "gpu_stream_gap"]
        self.assertEqual(["stall"], [event["reason"] for event in stream_gaps])
        self.assertGreater(stream_gaps[0]["gap_s"], 3.0)
        self.assertLessEqual(stream_gaps[0]["gap_s"], 3.6)
        self.assertEqual(1.0, stream_gaps[0]["restart_in_s"])
        self.assertEqual(2, int(state.read_text()))
        self.assertEqual(1, summary["gpu_stream_restarts"])
        self.assertTrue(sentinel.stream.reaped and all(code is not None for code in sentinel.stream.reaped))
        fast = records["fast"]
        stale = [record for record in fast if record["null_reasons"].get("gpus[0]") == "stale"]
        self.assertTrue(stale, "GPU values must be marked stale during the stall")
        self.assertIsNotNone(fast[-1]["gpus"][0]["temp_c"], "data resumes after the restart")
        self.assertLessEqual(max(gaps(fast)), 1.0, "the fast tier keeps its cadence while the stream stalls")
        changes = [(event["field"], event["from"], event["to"]) for event in records["events"]
                   if event["event"] == "sensor_state_change" and event["tier"] == "fast"]
        self.assertIn(("gpus[0]", "ok", "stale"), changes)
        self.assertIn(("gpus[0]", "stale", "ok"), changes)
        self.assert_all_valid(records)

    def test_exiting_stream_restarts_with_exponential_backoff(self):
        sentinel, summary = self.run_sentinel(8.0, env={"FAKE_NVIDIA_SMI_MODE": "exit",
                                                         "FAKE_NVIDIA_SMI_STALL_AFTER": "2"}, slow_interval_s=5.0)
        events = [event for event in load(self.out)["events"] if event["event"] == "gpu_stream_gap"]
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual({"exit"}, {event["reason"] for event in events})
        self.assertEqual({9}, {event["returncode"] for event in events})
        self.assertEqual([1.0, 2.0], [event["restart_in_s"] for event in events[:2]])
        self.assertTrue(all(code == 9 for code in sentinel.stream.reaped[:2]))

    def test_tool_absent_keeps_collecting(self):
        readers = RecordingReaders()
        sentinel, summary = self.run_sentinel(3.0, readers=readers, nvidia_smi=["atlas-no-such-nvidia-smi-xyz"],
                                              slow_interval_s=1.0)
        records = load(self.out)
        later = records["fast"][2:]
        self.assertTrue(later and all(record["gpus"] is None and record["null_reasons"]["gpus"] == "tool_absent"
                                      for record in later))
        self.assertTrue(all(record["cpu_pkg"][0]["temp_c"] == 30.0 for record in records["fast"]))
        events = [event for event in records["events"] if event["event"] == "gpu_stream_gap"]
        self.assertEqual(["tool_absent"], [event["reason"] for event in events])
        self.assertEqual({"tool_absent"}, {record["null_reasons"]["gpus"] for record in records["slow"]})
        self.assertGreaterEqual(summary["fast_samples"], 5)
        self.assert_all_valid(records)

    def test_slow_query_timeout_emits_gap_and_recovers(self):
        calls = {"q": 0}

        def flaky(argv, timeout):
            if "-q" in argv:
                calls["q"] += 1
                if calls["q"] == 1:
                    time.sleep(1.6)
                    raise subprocess.TimeoutExpired(argv, timeout)
            return tv.run_command(argv, timeout)
        readers = RecordingReaders(run_cmd=flaky)
        sentinel, summary = self.run_sentinel(4.5, readers=readers, slow_interval_s=0.5, cmd_timeout_s=1.0)
        records = load(self.out)
        slow = records["slow"]
        self.assertIsNone(slow[0]["gpus"])
        self.assertEqual("tool_timeout", slow[0]["null_reasons"]["gpus"])
        self.assertIsNotNone(slow[-1]["gpus"], "next cycle recovers")
        gap_events = [event for event in records["events"] if event["event"] == "sample_gap"]
        self.assertEqual(["slow"], [event["tier"] for event in gap_events])
        self.assertGreater(gap_events[0]["gap_s"], 1.0)
        self.assertLessEqual(max(gaps(records["fast"])), 1.0, "a hung slow command never delays the fast tier")
        self.assert_all_valid(records)

    def test_failing_reader_is_isolated_inside_a_tick(self):
        readers = RecordingReaders()
        original = readers.open_file

        def open_file(path):
            if Path(path).name == "temp1_input" and "hwmon1" in str(path):
                raise RuntimeError("synthetic sensor crash")
            return original(path)
        readers.open_file = open_file
        sentinel, summary = self.run_sentinel(2.0, readers=readers, gpu_mode="off", slow_interval_s=1.0)
        records = load(self.out)
        for record in records["fast"]:
            self.assertEqual([{"package": 0, "temp_c": None}], record["cpu_pkg"])
            self.assertEqual("collector_error", record["null_reasons"]["cpu_pkg[0].temp_c"])
            self.assertEqual("disabled_by_config", record["null_reasons"]["gpus"])
        self.assertEqual(0, summary["fast_errors"])
        self.assertGreaterEqual(summary["fast_samples"], 3)
        self.assert_all_valid(records)


class StreamTimeoutTests(SentinelHarness):
    """Cases 6 and 7 with the real fake tool sleeping (no injected exceptions)."""

    def test_stream_child_that_never_prints_is_killed_reaped_and_restarted(self):
        # FAKE_NVIDIA_SMI_MODE=timeout: every launch sleeps without output (a hung driver call).
        sentinel, summary = self.run_sentinel(5.5, env={"FAKE_NVIDIA_SMI_MODE": "timeout"}, slow_interval_s=5.0,
                                              cmd_timeout_s=1.0)
        records = load(self.out)
        stream_gaps = [event for event in records["events"] if event["event"] == "gpu_stream_gap"]
        self.assertGreaterEqual(len(stream_gaps), 1)
        self.assertEqual("stall", stream_gaps[0]["reason"])
        self.assertGreater(stream_gaps[0]["gap_s"], 3.0)
        self.assertLessEqual(stream_gaps[0]["gap_s"], 3.6, "watchdog fires within 3 s (+ poll granularity)")
        self.assertTrue(sentinel.stream.reaped, "the stalled child was waited for")
        self.assertTrue(all(code is not None for code in sentinel.stream.reaped), "no zombie child")
        fast = records["fast"]
        self.assertTrue(all(record["gpus"] is None for record in fast))
        self.assertEqual({"first_sample"}, {record["null_reasons"]["gpus"] for record in fast[:5]})
        self.assertEqual("tool_timeout", fast[-1]["null_reasons"]["gpus"])
        self.assertTrue(all(record["cpu_pkg"][0]["temp_c"] == 30.0 for record in fast), "other sections still filled")
        self.assertLessEqual(max(gaps(fast)), 1.0)
        slow = records["slow"]
        self.assertEqual("tool_timeout", slow[0]["null_reasons"]["gpus"], "slow queries hit the command timeout")
        self.assert_all_valid(records)

    def test_poll_mode_timeout_is_labelled_and_recovers(self):
        poll_argv = [str(part) for part in tv.nvidia_query_argv(FAKE, tv.FAST_QUERY_FIELDS)]
        calls = {"poll": 0}
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]

        def hanging_then_ok(argv, timeout):
            if [str(part) for part in argv] == poll_argv:
                calls["poll"] += 1
                if calls["poll"] <= 2:
                    # A real child that sleeps: subprocess.run must kill it at the timeout.
                    return tv.run_command(sleeper, timeout)
            return tv.run_command(argv, timeout)
        readers = RecordingReaders(run_cmd=hanging_then_ok)
        sentinel, summary = self.run_sentinel(7.0, readers=readers, gpu_mode="poll", cmd_timeout_s=0.8,
                                              slow_interval_s=5.0)
        records = load(self.out)
        fast = records["fast"]
        stream_gaps = [event for event in records["events"] if event["event"] == "gpu_stream_gap"]
        self.assertEqual(["timeout"], [event["reason"] for event in stream_gaps], "one event per status change")
        timed_out = [record for record in fast if record["gpus"] is None
                     and record["null_reasons"]["gpus"] == "tool_timeout"]
        self.assertTrue(timed_out, "fast records carry tool_timeout while the query hangs")
        self.assertIsNotNone(fast[-1]["gpus"], "the poll loop recovers after the backoff")
        self.assertIsNotNone(fast[-1]["gpus"][0]["temp_c"])
        self.assertGreaterEqual(calls["poll"], 3)
        self.assertLessEqual(max(gaps(fast)), 1.0, "a hanging GPU query never delays the fast tier")
        self.assert_all_valid(records)


class RebootAndRestartTests(SentinelHarness):
    def boot_root(self, boot_file):
        root = Path(self.tmp.name) / "host"
        target = root / "proc" / "sys" / "kernel" / "random"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SYNTHETIC / boot_file, target / "boot_id")
        return root

    def test_reboot_detected_with_last_sample_before_gap(self):
        previous = json.loads((SYNTHETIC / "sentinel-state-prev.json").read_text(encoding="utf-8"))
        prev_boot = previous["boot_id"]
        self.out.mkdir(parents=True)
        (self.out / "state.json").write_text(json.dumps(previous), encoding="utf-8")
        prev_dir = self.out / prev_boot
        prev_dir.mkdir()
        last = tv.envelope("fast", "sim-a", prev_boot, previous["last_fast_seq"], previous["last_sample_mono_s"],
                           previous["last_sample_utc"])
        last.update(dt_s=0.5, cpu_pkg=[{"package": 0, "temp_c": 66.0}], null_reasons={}, src={"gpu": "test"},
                    gpus=[{"index": 0, "temp_c": 83, "power_w": 181.0, "util_pct": 99, "mem_used_mib": 1, "sm_mhz": 1,
                           "mem_mhz": 1, "fan_pct": 61, "pstate": "P2", "age_s": 0.1}])
        segment = prev_dir / "fast-20261018T010000Z-0001.jsonl"
        segment.write_text(json.dumps(last) + "\n{\"truncated\": ", encoding="utf-8")
        readers = tv.ReplayReaders(self.boot_root("boot_id.b"))
        self.run_sentinel(1.2, readers=readers, gpu_mode="off", slow_interval_s=1.0)
        records = load(self.out)
        reboot = [event for event in records["events"] if event["event"] == "reboot_detected"]
        self.assertEqual(1, len(reboot))
        event = reboot[0]
        self.assertEqual(prev_boot, event["prev_boot_id"])
        self.assertEqual(previous["last_sample_utc"], event["prev_last_sample_utc"])
        self.assertGreater(event["gap_s"], 0)
        self.assertEqual({"index": 0, "temp_c": 83, "power_w": 181.0, "util_pct": 99},
                         event["prev_last_sample"]["gpus"][0])
        new_boot = (SYNTHETIC / "boot_id.b").read_text(encoding="utf-8").strip()
        self.assertEqual(new_boot, event["boot_id"])
        self.assertFalse(segment.exists(), "closed segments of the previous boot are compressed")
        self.assertTrue((prev_dir / (segment.name + ".gz")).exists())
        for item in records["events"] + [records["fast"][-1]]:
            if item["boot_id"] == new_boot:
                fs.validate_telemetry_record(item)
        self.assertEqual(new_boot, json.loads((self.out / "state.json").read_text(encoding="utf-8"))["boot_id"])

    def test_restart_in_same_boot_continues_seq_and_reports_gap(self):
        readers = tv.ReplayReaders(self.boot_root("boot_id.a"))
        self.run_sentinel(1.2, readers=readers, gpu_mode="off", slow_interval_s=1.0)
        time.sleep(1.6)
        self.run_sentinel(1.2, readers=tv.ReplayReaders(self.boot_root("boot_id.a")), gpu_mode="off",
                          slow_interval_s=1.0)
        records = load(self.out)
        starts = [event for event in records["events"] if event["event"] == "collector_start"]
        self.assertEqual([False, True], [event["prev_run_same_boot"] for event in starts])
        restart_gaps = [event for event in records["events"] if event["event"] == "sample_gap"]
        self.assertEqual(1, len(restart_gaps))
        self.assertTrue(restart_gaps[0]["across_restart"])
        self.assertGreater(restart_gaps[0]["gap_s"], 1.6)
        self.assertNotIn("reboot_detected", [event["event"] for event in records["events"]])
        self.assert_all_valid(records)

    def test_reboot_between_two_real_runs_uses_boot_id_and_last_synced_sample(self):
        self.run_sentinel(2.0, readers=tv.ReplayReaders(self.boot_root("boot_id.a")), slow_interval_s=1.0)
        boot_a = (SYNTHETIC / "boot_id.a").read_text(encoding="utf-8").strip()
        boot_b = (SYNTHETIC / "boot_id.b").read_text(encoding="utf-8").strip()
        first = load(self.out)
        last_a = first["fast"][-1]
        state_a = json.loads((self.out / "state.json").read_text(encoding="utf-8"))
        self.assertEqual((boot_a, last_a["seq"], last_a["utc"]),
                         (state_a["boot_id"], state_a["last_fast_seq"], state_a["last_sample_utc"]))
        time.sleep(1.0)
        # Same output directory, new boot_id (the host rebooted in between).
        self.run_sentinel(1.2, readers=tv.ReplayReaders(self.boot_root("boot_id.b")), slow_interval_s=1.0)
        records = load(self.out)
        reboot = [event for event in records["events"] if event["event"] == "reboot_detected"]
        self.assertEqual(1, len(reboot))
        event = reboot[0]
        self.assertEqual((boot_b, boot_a), (event["boot_id"], event["prev_boot_id"]))
        self.assertEqual(last_a["utc"], event["prev_last_sample_utc"])
        self.assertEqual((last_a["utc"], last_a["seq"]),
                         (event["prev_last_sample"]["utc"], event["prev_last_sample"]["seq"]))
        self.assertEqual([{key: gpu[key] for key in ("index", "temp_c", "power_w", "util_pct")}
                          for gpu in last_a["gpus"]], event["prev_last_sample"]["gpus"])
        self.assertGreaterEqual(event["gap_s"], 1.0, "UTC gap across the reboot (monotonic is not comparable)")
        self.assertNotIn("sample_gap", [item["event"] for item in records["events"] if item["boot_id"] == boot_b])
        new_fast = [record for record in records["fast"] if record["boot_id"] == boot_b]
        self.assertEqual(1, new_fast[0]["seq"], "seq restarts per boot")
        self.assertFalse(list((self.out / boot_a).glob("*.jsonl")), "segments of the old boot are compressed")
        self.assertEqual(boot_b, json.loads((self.out / "state.json").read_text(encoding="utf-8"))["boot_id"])
        self.assert_all_valid(records)

    def test_one_sentinel_per_directory(self):
        first = sn.Sentinel(sn.SentinelConfig(self.out, "sim-a", duration_s=2.5, gpu_mode="off", nice=0,
                                              slow_interval_s=1.0), RecordingReaders())
        thread = threading.Thread(target=first.run)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not (self.out / "sentinel.lock").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.3)
            second = sn.Sentinel(sn.SentinelConfig(self.out, "sim-a", duration_s=1.0, gpu_mode="off", nice=0),
                                 RecordingReaders())
            with self.assertRaisesRegex(RuntimeError, "another sentinel"):
                second.run()
        finally:
            thread.join(10)


class EndToEndWriterTests(SentinelHarness):
    """Cases 16 and 17 through the real sentinel loop (real clock, real files)."""

    def test_rotation_compresses_closed_segments_and_keeps_seq(self):
        sentinel, summary = self.run_sentinel(4.2, gpu_mode="off", slow_interval_s=1.0, rotate_seconds=1.5)
        boot = next(path for path in self.out.iterdir() if path.is_dir())
        fast_segments = sn.segment_files(boot, "fast")
        closed = [path for path in fast_segments if path.name.endswith(".jsonl.gz")]
        still_open = [path for path in fast_segments if path.name.endswith(".jsonl")]
        self.assertGreaterEqual(len(closed), 2, [path.name for path in fast_segments])
        self.assertEqual(1, len(still_open), "only the last segment stays uncompressed until the next start")
        fast = load(self.out)["fast"]
        self.assertEqual(list(range(1, len(fast) + 1)), [record["seq"] for record in fast])
        self.assertLessEqual(max(gaps(fast)), 1.0, "rotation + gzip never stall the fast tier")
        self.assertEqual([], list(boot.glob("*.gz.tmp")))

    def test_latest_json_is_never_partial_for_a_concurrent_reader(self):
        stop = threading.Event()
        seen = {"parsed": 0, "partial": 0, "seqs": []}

        def reader():
            target = self.out / "latest.json"
            while not stop.is_set():
                try:
                    raw = target.read_bytes()
                except OSError:  # not written yet, or the instant of the rename on Windows
                    time.sleep(0.002)
                    continue
                try:
                    document = json.loads(raw.decode("utf-8"))
                    fs.validate_sentinel_latest(document)
                except (ValueError, fs.SchemaError):
                    seen["partial"] += 1
                else:
                    seen["parsed"] += 1
                    seen["seqs"].append(document["seq"])
                time.sleep(0.001)
        thread = threading.Thread(target=reader)
        thread.start()
        try:
            sentinel, summary = self.run_sentinel(3.0, gpu_mode="off", slow_interval_s=1.0, fast_interval_s=0.1)
        finally:
            stop.set()
            thread.join(10)
        self.assertEqual(0, seen["partial"], "a reader never sees a truncated/invalid latest.json")
        self.assertGreater(seen["parsed"], 20)
        self.assertEqual(sorted(seen["seqs"]), seen["seqs"], "latest.json only moves forward")
        status = sn.read_latest(self.out / "latest.json", now_mono=time.monotonic())
        self.assertEqual("ok", status["status"], status)
        self.assertEqual("stale", sn.read_latest(self.out / "latest.json", now_mono=time.monotonic() + 6)["status"])
        self.assertEqual(0, summary["fast_errors"])


class WriterTests(unittest.TestCase):
    """Cases 16 and 17 with a simulated clock and a mocked fsync."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="atlas-writer-")
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_fsync_at_most_once_per_five_simulated_seconds(self):
        now = {"t": 0.0}
        fsyncs = []
        writer = sn.SegmentWriter(self.dir, "fast", wall=lambda: 1792400000.0, fsync_interval_s=5.0,
                                  rotate_seconds=3600.0, fsync=lambda fd: fsyncs.append(now["t"]))
        for index in range(121):
            now["t"] = index * 0.5
            writer.write({"seq": index}, now["t"])
        self.assertEqual([5.0 * k for k in range(1, 13)], fsyncs)
        self.assertEqual(60.0, writer.last_synced_record["seq"] * 0.5)
        writer.write({"seq": 121}, 60.2)
        writer.close()
        self.assertEqual(13, len(fsyncs), "close forces the pending fsync")

    def test_hourly_style_rotation_closes_and_compresses(self):
        # Injected wall clock (2026-10-20T04:20:00Z = 1792470000) advancing with the simulated
        # monotonic clock: segment names must come from it, never from the real clock.
        now = {"t": 0.0}
        writer = sn.SegmentWriter(self.dir, "fast", wall=lambda: 1792470000.0 + now["t"], fsync_interval_s=5.0,
                                  rotate_seconds=10.0, fsync=lambda fd: None, on_closed=sn.compress_segment)
        for index in range(51):
            now["t"] = index * 0.5
            writer.write({"seq": index}, now["t"])
        writer.close()
        names = sorted(path.name for path in self.dir.iterdir())
        self.assertEqual(["fast-20261020T042000Z-0001.jsonl.gz", "fast-20261020T042010Z-0002.jsonl.gz",
                          "fast-20261020T042020Z-0003.jsonl"], names)
        with gzip.open(self.dir / names[0], "rt", encoding="utf-8") as handle:
            self.assertEqual(list(range(20)), [json.loads(line)["seq"] for line in handle])
        seqs = [record["seq"] for path in sn.segment_files(self.dir, "fast") for record in sn.iter_segment_lines(path)]
        self.assertEqual(list(range(51)), seqs)
        self.assertEqual(50, sn.last_record(self.dir, "fast")["seq"])

    def latest(self, seq, mono, boot="8c40d0a6-6a4f-46b4-adaf-bae6c4b8a2a6"):
        return {"schema": "atlas.sentinel-latest.v1", "node_id": "sim-a", "boot_id": boot, "seq": seq,
                "utc": "2026-10-19T12:00:00.000Z", "mono_s": mono, "gpus": None, "cpu_pkg_max_temp_c": 30.0,
                "disks": None, "null_reasons": {"gpus": "disabled_by_config", "disks": "first_sample"}}

    def test_latest_atomic_replace_survives_crash_between_write_and_rename(self):
        target = self.dir / "latest.json"
        sn.atomic_write_json(target, self.latest(1, 100.0))

        def crash(source, destination):
            raise OSError("simulated power loss before rename")
        with self.assertRaises(OSError):
            sn.atomic_write_json(target, self.latest(2, 100.5), replace=crash)
        self.assertEqual(1, json.loads(target.read_text(encoding="utf-8"))["seq"])
        self.assertTrue(list(self.dir.glob(".latest.json.*.tmp")), "the partial temp file never replaced the target")
        status = sn.read_latest(target, now_mono=104.9)
        self.assertEqual(("ok", 1), (status["status"], status["document"]["seq"]))

    def test_consumer_sees_stale_other_boot_missing_invalid(self):
        target = self.dir / "latest.json"
        self.assertEqual("missing", sn.read_latest(target, now_mono=1.0)["status"])
        sn.atomic_write_json(target, self.latest(1, 100.0))
        self.assertEqual("ok", sn.read_latest(target, now_mono=105.0)["status"])
        self.assertEqual("stale", sn.read_latest(target, now_mono=105.1)["status"])
        self.assertEqual("other_boot", sn.read_latest(target, now_mono=100.0, boot_id="bbbbbbbb-0000-4000-8000-00000000000b")["status"])
        target.write_text("{\"schema\": \"atlas.sentinel-latest.v1\"", encoding="utf-8")
        self.assertEqual("missing", sn.read_latest(target, now_mono=100.0)["status"])
        target.write_text(json.dumps({"schema": "atlas.sentinel-latest.v1"}), encoding="utf-8")
        self.assertEqual("invalid", sn.read_latest(target, now_mono=100.0)["status"])


class CliAndScriptTests(unittest.TestCase):
    def test_cli_run_report_latest(self):
        with tempfile.TemporaryDirectory(prefix="atlas-sentinel-cli-") as tmp:
            out = Path(tmp) / "sentinel"
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            run = subprocess.run([sys.executable, "-m", "augment_atlas.sentinel", "run", "--dir", str(out),
                                  "--node-id", "sim-a", "--duration", "3", "--slow-interval", "1", "--nice", "0",
                                  "--replay", str(X299), "--disk", f"scratch={tmp}", "--nvidia-smi", *FAKE],
                                 capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual(0, run.returncode, run.stderr)
            summary = json.loads(run.stdout)
            self.assertEqual("duration", summary["reason"])
            report = subprocess.run([sys.executable, "-m", "augment_atlas.sentinel", "report", "--dir", str(out),
                                     "--validate"], capture_output=True, text=True, timeout=60, env=env)
            data = json.loads(report.stdout)
            self.assertEqual([], data["validation_errors"])
            self.assertLessEqual(data["boots"][0]["max_gap_s"], 1.0)
            latest = subprocess.run([sys.executable, "-m", "augment_atlas.sentinel", "latest", "--dir", str(out),
                                     "--max-age", "3600"], capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual(0, latest.returncode, latest.stdout)
            bad = subprocess.run([sys.executable, "-m", "augment_atlas.sentinel", "run", "--dir", str(out),
                                  "--node-id", "Host.Name", "--duration", "1"], capture_output=True, text=True,
                                 timeout=60, env=env)
            self.assertEqual(2, bad.returncode)

    def test_observe_only_script_is_bounded_unprivileged_and_parses(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("sudo", "ssh ", "docker run", "docker kill", "docker stop", "docker rm", "systemctl",
                          "nvidia-smi -pl", "--allow-root", "pip install", "apt"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)
        self.assertIn("augment-atlas-private/sentinel-", text)
        self.assertIn("umask 077", text)
        self.assertIn("--duration", text)
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash not available for the syntax check")
        result = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30)
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
