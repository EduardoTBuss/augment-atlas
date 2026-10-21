"""Training telemetry (atlas.train_telemetry.v1): writer, sensors, summary, trainer integration.

Pure tests use the stdlib only. The trainer integration runs the real toy torch
loop and is container-only, like the other CPU training tests:

    ATLAS_CONTAINER_CPU_TESTS=1 PYTHONPATH=src python -m unittest tests.test_train_telemetry -v

Integration contract: telemetry on vs. off (patched ``ENABLED=False``) yields
identical metrics (minus wall-clock ``utc``), model/optimizer/RNG state in the
checkpoints, best.pt and predictions; no evaluated-partition metric appears in
the telemetry; the validated artifact set is unchanged.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from augment_atlas import train_telemetry as tt

ROOT = Path(__file__).resolve().parents[1]
REQUEST_STACK = os.environ.get("ATLAS_CONTAINER_CPU_TESTS") == "1"
HAS_STACK = REQUEST_STACK and all(importlib.util.find_spec(n) is not None
                                  for n in ("torch", "torchvision", "numpy", "PIL"))
FORBIDDEN_METRIC_KEYS = {"test", "E_test", "E_f", "D_select", "evaluation", "clinical_evaluation", "results",
                         "predictions"}


class FakeClock:
    def __init__(self, start=100.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def lines(path):
    return [json.loads(raw) for raw in Path(path).read_text(encoding="utf-8").splitlines() if raw.strip()]


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


class WriterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-telemetry-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_schema_seq_and_common_fields(self):
        clock = FakeClock()
        writer = tt.JsonlWriter(self.root / "t" / "train.jsonl", session_id="s1", clock=clock)
        for index in range(5):
            clock.advance(0.5)
            self.assertTrue(writer.write("batch", {"step": index}))
        writer.close()
        rows = lines(self.root / "t" / "train.jsonl")
        self.assertEqual([row["seq"] for row in rows], list(range(5)))
        for row in rows:
            self.assertEqual(row["schema"], tt.SCHEMA)
            self.assertEqual((row["type"], row["session_id"]), ("batch", "s1"))
            self.assertIn("utc", row)
            self.assertIsInstance(row["mono_s"], float)
        self.assertEqual([row["mono_s"] for row in rows], sorted(row["mono_s"] for row in rows))

    def test_fsync_is_batched_by_count_time_and_explicit_sync(self):
        clock = FakeClock()
        with patch.object(tt.os, "fsync") as fsync:
            writer = tt.JsonlWriter(self.root / "train.jsonl", session_id="s", fsync_every=4, fsync_seconds=60,
                                    clock=clock)
            for _ in range(3):
                writer.write("batch", {})
            self.assertEqual(fsync.call_count, 0)
            writer.write("batch", {})
            self.assertEqual(fsync.call_count, 1)
            writer.write("batch", {})
            clock.advance(61)
            writer.write("batch", {})
            self.assertEqual(fsync.call_count, 2)
            writer.write("epoch", {}, sync=True)
            self.assertEqual(fsync.call_count, 3)
            writer.close()
            self.assertEqual(fsync.call_count, 4)
        self.assertEqual(len(lines(self.root / "train.jsonl")), 7)

    def test_append_only_across_sessions_and_torn_line_terminated(self):
        path = self.root / "train.jsonl"
        first = tt.JsonlWriter(path, session_id="a")
        first.write("session_start", {})
        first.close()
        with path.open("ab") as stream:
            stream.write(b'{"schema": "atlas.train_tele')  # crash mid-line
        second = tt.JsonlWriter(path, session_id="b")
        second.write("session_start", {})
        second.close()
        records, invalid = tt.read_records(path)
        self.assertEqual(invalid, 1)
        self.assertEqual([(row["session_id"], row["seq"]) for row in records], [("a", 0), ("b", 0)])

    def test_non_finite_value_disables_writer_after_one_error_line(self):
        path = self.root / "train.jsonl"
        writer = tt.JsonlWriter(path, session_id="s")
        self.assertTrue(writer.write("batch", {"loss": 1.0}))
        self.assertFalse(writer.write("batch", {"loss": float("nan")}))
        self.assertFalse(writer.enabled)
        self.assertFalse(writer.write("batch", {"loss": 2.0}))
        writer.close()
        records, invalid = tt.read_records(path)
        self.assertEqual(invalid, 0)
        self.assertEqual([row["type"] for row in records], ["batch", "error"])
        self.assertEqual(records[1]["telemetry"], "disabled")
        self.assertEqual(records[1]["error_type"], "ValueError")

    def test_io_error_disables_writer_without_raising(self):
        writer = tt.JsonlWriter(self.root / "train.jsonl", session_id="s")
        self.addCleanup(writer._stream.close)
        with patch.object(writer, "_stream") as stream:
            stream.write.side_effect = OSError("disk full")
            self.assertFalse(writer.write("batch", {}))
        self.assertFalse(writer.enabled)
        self.assertIn("disk full", writer.error)

    def test_concurrent_writers_keep_unique_monotonic_seq(self):
        path = self.root / "train.jsonl"
        writer = tt.JsonlWriter(path, session_id="s")
        barrier = threading.Barrier(8)

        def work():
            barrier.wait()
            for _ in range(200):
                writer.write("machine", {"x": 1})
        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        writer.close()
        records, invalid = tt.read_records(path)
        self.assertEqual(invalid, 0)
        self.assertEqual([row["seq"] for row in records], list(range(1600)))


class SensorTests(unittest.TestCase):
    def test_nvidia_smi_absent_is_null_with_reason(self):
        self.assertEqual(tt.query_gpus(which=lambda _: None), (None, {"gpus": "tool_absent"}))

    def test_nvidia_smi_timeout_and_error(self):
        def timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("nvidia-smi", 4)
        self.assertEqual(tt.query_gpus(runner=timeout, which=lambda _: "x"), (None, {"gpus": "tool_timeout"}))
        failed = lambda *_a, **_k: subprocess.CompletedProcess([], 9, "", "boom")
        self.assertEqual(tt.query_gpus(runner=failed, which=lambda _: "x"), (None, {"gpus": "collector_error"}))
        garbage = lambda *_a, **_k: subprocess.CompletedProcess([], 0, "1, 2\n", "")
        self.assertEqual(tt.query_gpus(runner=garbage, which=lambda _: "x"), (None, {"gpus": "parse_error"}))

    def test_nvidia_smi_parses_values_and_marks_unsupported_fields(self):
        line = "0, GPU-abc, NVIDIA GeForce GTX TITAN X, 83, 99, 10850, 12288, 181.20, 250.00, 1189, P2, [N/A]\n"
        ok = lambda *_a, **_k: subprocess.CompletedProcess([], 0, line, "")
        gpus, reasons = tt.query_gpus(runner=ok, which=lambda _: "x")
        self.assertEqual(gpus[0]["index"], 0)
        self.assertEqual((gpus[0]["uuid"], gpus[0]["temp_c"], gpus[0]["power_w"]), ("GPU-abc", 83.0, 181.2))
        self.assertEqual(gpus[0]["pstate"], "P2")
        self.assertIsNone(gpus[0]["fan_pct"])
        self.assertEqual(reasons, {"gpus[0].fan_pct": "not_supported_by_device"})

    def test_machine_sample_never_raises_and_every_null_has_a_reason(self):
        clock = FakeClock()

        def broken_gpu():
            raise RuntimeError("driver gone")
        sampler = tt.MachineSampler(gpu_query=broken_gpu, clock=clock, process_clock=lambda: 1.0)
        first = sampler.sample()
        clock.advance(30)
        second = sampler.sample()
        self.assertEqual(first["null_reasons"]["process_cpu_pct"], "first_sample")
        self.assertEqual(second["process_cpu_pct"], 0.0)
        for row in (first, second):
            self.assertEqual(row["null_reasons"]["gpus"], "collector_error")
            for key, value in row.items():
                if value is None:
                    self.assertIn(key, row["null_reasons"], key)
            for reason in row["null_reasons"].values():
                self.assertIn(reason, tt.REASONS)
            json.dumps(row, allow_nan=False)

    def test_linux_proc_and_cgroup_readings_are_parsed(self):
        clock = FakeClock()
        stats = iter(["cpu  100 0 100 700 100 0 0 0 0 0\n", "cpu  150 0 150 750 150 0 0 0 0 0\n"])
        files = {"/proc/self/status": "Name:\tpython\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\n",
                 "/sys/fs/cgroup/memory.current": "4096\n", "/sys/fs/cgroup/memory.max": "max\n"}

        def read(path):
            return next(stats) if path == "/proc/stat" else files.get(path)
        sampler = tt.MachineSampler(gpu_query=lambda: (None, {"gpus": "tool_absent"}), clock=clock,
                                    process_clock=lambda: 0.0)
        with patch.object(tt, "_read_text", side_effect=read), \
             patch.object(tt.os, "getloadavg", create=True, return_value=(1.0, 0.5, 0.25)):
            first = sampler.sample()
            clock.advance(30)
            second = sampler.sample()
        self.assertEqual((first["rss_bytes"], first["rss_peak_bytes"]), (1024 * 1024, 2048 * 1024))
        self.assertEqual(first["load_1_5_15"], [1.0, 0.5, 0.25])
        self.assertEqual(first["null_reasons"]["host_cpu_util_pct"], "first_sample")
        # delta total 200 ticks, idle+iowait delta 100 => 50 %.
        self.assertEqual(second["host_cpu_util_pct"], 50.0)
        self.assertEqual(second["cgroup_memory_bytes"], 4096)
        self.assertIsNone(second["cgroup_memory_limit_bytes"])  # cgroup v2 "max"
        self.assertEqual(second["null_reasons"]["cgroup_memory_limit_bytes"], "unlimited")


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)


class FacadeAndSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-telemetry-facade-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make(self, clock, sampler):
        return tt.TrainTelemetry(self.root, torch=None, device="cpu", context={"path": "unit"},
                                 batch_every=2, machine_interval_s=0, sampler=sampler, clock=clock,
                                 grad_clip_max_norm=1.0)

    def test_batch_epoch_records_and_summary_with_energy(self):
        clock = FakeClock(0.0)
        power = iter([100.0, 200.0, 200.0])
        temperature = iter([70.0, 72.0, 71.0])

        class Sampler:
            def sample(self):
                return {"gpus": [{"index": 0, "uuid": "GPU-x", "temp_c": next(temperature),
                                  "power_w": next(power)}], "rss_bytes": 10, "null_reasons": {}}
        telemetry = self.make(clock, Sampler())
        telemetry._machine_sample()
        telemetry.epoch_start(1, batches=3, images=10)
        for index, (images, loss, grad) in enumerate(((4, 1.0, None), (4, 0.5, FakeTensor(2.0)), (2, 0.25, None))):
            clock.advance(1.0)
            telemetry.batch(1, index, images, loss, 0.001, grad)
        telemetry.train_done(1)
        clock.advance(0.5)
        telemetry.epoch_end(1, {"epoch": 1, "train_loss": 0.65, "learning_rate": 0.0009, "samples_processed": 10,
                                "selection": {"loss": 0.7, "accuracy": 0.5, "samples": 4},
                                "test": {"loss": 9.9}}, selection_per_epoch=True)
        clock.advance(26.5)  # second machine sample at t=30
        telemetry._machine_sample()
        clock.advance(30.0)
        telemetry._machine_sample()
        telemetry.outcome("completed")
        telemetry.close()
        records, invalid = tt.read_records(self.root / tt.DIRECTORY / tt.JSONL_NAME)
        self.assertEqual(invalid, 0)
        kinds = [row["type"] for row in records]
        self.assertEqual(kinds, ["session_start", "machine", "batch", "batch", "epoch", "machine", "machine",
                                 "session_end"])
        batches = [row for row in records if row["type"] == "batch"]
        self.assertEqual([(row["step"], row["window_batches"]) for row in batches], [(2, 2), (3, 1)])
        self.assertEqual(batches[0]["loss"], 0.5)
        self.assertEqual(batches[0]["window_loss_mean"], 0.75)
        self.assertEqual(batches[0]["window_images_per_s"], 4.0)
        self.assertEqual((batches[0]["grad_norm_preclip"], batches[0]["grad_norm_step"], batches[0]["grad_clipped"]),
                         (2.0, 2, True))
        self.assertIsNone(batches[0]["vram_allocated_bytes"])
        self.assertEqual(batches[0]["null_reasons"]["vram_allocated_bytes"], "device_not_cuda")
        epoch = records[4]
        self.assertEqual((epoch["images"], epoch["train_s"], epoch["post_train_s"]), (10, 3.0, 0.5))
        self.assertEqual(epoch["selection"], {"loss": 0.7, "accuracy": 0.5, "samples": 4})
        self.assertNotIn("test", epoch)
        summary = json.loads((self.root / tt.DIRECTORY / tt.SUMMARY_NAME).read_text(encoding="utf-8"))
        self.assertEqual(summary["schema"], tt.SUMMARY_SCHEMA)
        self.assertEqual((summary["status"], summary["epochs_recorded"], summary["machine_samples"]),
                         ("completed", 1, 3))
        self.assertAlmostEqual(summary["throughput_img_s_mean"], 10 / 3)
        self.assertEqual(summary["gpu_temp_max_c"], 72.0)
        # Trapezoid: 30 s x 150 W + 30 s x 200 W = 10500 J.
        self.assertAlmostEqual(summary["energy_kwh"], 10500 / 3.6e6)
        self.assertEqual(summary["energy"]["status"], "measured_integrated")
        self.assertEqual(summary["energy"]["coverage_fraction"], 1.0)
        self.assertIsNone(summary["vram_max_allocated_bytes"])
        self.assertEqual(summary["null_reasons"]["vram_max_allocated_bytes"], "device_not_cuda")
        self.assertFalse(FORBIDDEN_METRIC_KEYS & set(walk_keys(records)))

    def test_energy_gap_is_not_integrated_and_absent_power_is_null(self):
        machine = lambda t, w: {"type": "machine", "mono_s": t, "session_id": "s",
                                "gpus": [{"uuid": "g", "power_w": w}], "null_reasons": {}}
        kwh, detail = tt.integrate_energy({"s": [machine(0, 100), machine(30, 100), machine(300, 100)]}, 30)
        self.assertAlmostEqual(kwh, 3000 / 3.6e6)
        self.assertAlmostEqual(detail["coverage_fraction"], 30 / 300)
        summary = tt.summarize([{"type": "machine", "mono_s": 1.0, "session_id": "s", "gpus": None,
                                 "null_reasons": {"gpus": "tool_absent"}}], status="completed")
        self.assertIsNone(summary["energy_kwh"])
        self.assertEqual(summary["null_reasons"]["energy_kwh"], "tool_absent")
        self.assertEqual(summary["null_reasons"]["gpu_temp_max_c"], "tool_absent")

    def test_failing_hooks_never_raise(self):
        clock = FakeClock()

        class Sampler:
            def sample(self):
                raise RuntimeError("sensor")
        telemetry = self.make(clock, Sampler())
        telemetry._machine_sample()
        telemetry.epoch_start(1, batches=1, images=1)
        telemetry.batch(1, 0, 1, float("nan"), float("inf"), FakeTensor(float("nan")))
        telemetry.epoch_end(1, {"train_loss": float("nan")}, selection_per_epoch=False)
        telemetry.close()
        records, _ = tt.read_records(self.root / tt.DIRECTORY / tt.JSONL_NAME)
        batch = next(row for row in records if row["type"] == "batch")
        self.assertEqual((batch["null_reasons"]["loss"], batch["null_reasons"]["grad_norm_preclip"]),
                         ("non_finite", "non_finite"))
        machine = next(row for row in records if row["type"] == "machine")
        self.assertEqual(machine["null_reasons"], {"machine": "collector_error"})
        epoch = next(row for row in records if row["type"] == "epoch")
        self.assertEqual(epoch["null_reasons"]["selection"], "not_evaluated_per_epoch")
        self.assertIsNone(epoch["train_loss"])
        self.assertEqual(records[-1]["status"], "failed")

    def test_disabled_and_broken_factory_return_null_telemetry(self):
        with patch.object(tt, "ENABLED", False):
            self.assertIsInstance(tt.open_for_attempt(self.root, torch=None, device="cpu", context={}),
                                  tt.NullTelemetry)
        self.assertFalse((self.root / tt.DIRECTORY).exists())
        with patch.object(tt, "JsonlWriter", side_effect=PermissionError("read-only")):
            self.assertIsInstance(tt.open_for_attempt(self.root, torch=None, device="cpu", context={}),
                                  tt.NullTelemetry)

    def test_daemon_thread_samples_and_stops_on_close(self):
        calls = []

        class Sampler:
            def sample(self):
                calls.append(1)
                return {"gpus": None, "null_reasons": {"gpus": "tool_absent"}}
        telemetry = tt.TrainTelemetry(self.root, torch=None, device="cpu", context={}, machine_interval_s=0.01,
                                      sampler=Sampler())
        self.assertTrue(telemetry._thread.daemon)
        deadline = threading.Event()
        for _ in range(200):
            if len(calls) >= 3:
                break
            deadline.wait(0.01)
        telemetry.outcome("interrupted")
        telemetry.close()
        self.assertFalse(telemetry._thread.is_alive())
        self.assertGreaterEqual(len(calls), 3)
        summary = json.loads((self.root / tt.DIRECTORY / tt.SUMMARY_NAME).read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["machine_samples"], len(calls))


if HAS_STACK:
    sys.path.insert(0, str(ROOT / "tests"))
    import test_global_trainer as _global
    import test_selection_only as _selection


def _strip_utc(history):
    return [{key: value for key, value in row.items() if key != "utc"} for row in history]


@unittest.skipUnless(HAS_STACK, "real toy torch loop is container-only (ATLAS_CONTAINER_CPU_TESTS=1)")
class TrainerIntegrationCPUTests(unittest.TestCase):
    """Telemetry on vs. off: identical training artifacts in both trainer paths."""

    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-telemetry-cpu-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def assert_same_state(self, left, right, path="state"):
        """Exact recursive equality of checkpoint payloads (tensors, numpy RNG, scalars)."""
        import numpy as np
        if isinstance(left, self.torch.Tensor):
            self.assertIsInstance(right, self.torch.Tensor, path)
            self.assertEqual((left.dtype, left.shape), (right.dtype, right.shape), path)
            self.assertTrue(self.torch.equal(left, right), path)
        elif isinstance(left, np.ndarray):
            self.assertTrue(np.array_equal(left, right) and left.dtype == right.dtype, path)
        elif isinstance(left, dict):
            self.assertEqual(sorted(left, key=repr), sorted(right, key=repr), path)
            for key in left:
                self.assert_same_state(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual((type(left), len(left)), (type(right), len(right)), path)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assert_same_state(a, b, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    def load(self, path):
        return self.torch.load(path, map_location="cpu", weights_only=False)

    def assert_identical_attempts(self, on, off, *, extra_bytes=()):
        self.assertEqual(_strip_utc(json.loads((on / "metrics.json").read_text())),
                         _strip_utc(json.loads((off / "metrics.json").read_text())))
        for name in ("config.json", "splits.json", "results.json", *extra_bytes):
            self.assertEqual((on / name).read_bytes(), (off / name).read_bytes(), name)
        for name in ("checkpoint.pt", "best.pt"):
            left, right = self.load(on / name), self.load(off / name)
            for state in (left, right):
                for row in state.get("metrics", []):
                    row.pop("utc", None)
            self.assert_same_state(left, right)
        on_completed = json.loads((on / "completed.json").read_text())
        off_completed = json.loads((off / "completed.json").read_text())
        self.assertEqual(sorted(on_completed["artifacts"]), sorted(off_completed["artifacts"]))
        self.assertFalse(any(name.startswith(tt.DIRECTORY) for name in on_completed["artifacts"]))
        self.assertTrue((on / tt.DIRECTORY / tt.JSONL_NAME).is_file())
        self.assertFalse((off / tt.DIRECTORY).exists())

    def telemetry(self, out, epochs):
        records, invalid = tt.read_records(out / tt.DIRECTORY / tt.JSONL_NAME)
        self.assertEqual(invalid, 0)
        kinds = [row["type"] for row in records]
        self.assertNotIn("error", kinds)
        self.assertEqual((kinds[0], kinds[-1]), ("session_start", "session_end"))
        self.assertEqual([row["epoch"] for row in records if row["type"] == "epoch"], list(range(1, epochs + 1)))
        self.assertTrue(any(kind == "batch" for kind in kinds))
        self.assertTrue(any(kind == "machine" for kind in kinds))
        for session in {row["session_id"] for row in records}:
            seqs = [row["seq"] for row in records if row["session_id"] == session]
            self.assertEqual(seqs, list(range(len(seqs))))
        # No evaluated-partition metric is ever written to the telemetry.
        self.assertFalse(FORBIDDEN_METRIC_KEYS & set(walk_keys(records)))
        for row in records:
            json.dumps(row, allow_nan=False)
            for key, value in row.items():
                if value is None and key not in ("selection", "epoch"):
                    self.assertIn(key, row.get("null_reasons", {}), (row["type"], key))
        summary = json.loads((out / tt.DIRECTORY / tt.SUMMARY_NAME).read_text(encoding="utf-8"))
        self.assertEqual((summary["status"], summary["epochs_recorded"]), ("completed", epochs))
        return records, summary

    def test_global_paths_identical_with_and_without_telemetry(self):
        for phase in ("hpo", "final"):
            with self.subTest(phase=phase):
                runner = _global.GlobalTrainerCPUTests(methodName="test_final_reads_E_f_exactly_once_after_last_epoch")
                runner.torch = self.torch
                runner.setUp()
                self.addCleanup(runner.doCleanups)
                cfg = runner.fx.hpo_cfg if phase == "hpo" else runner.fx.final("basic", fold=1)
                root = runner.fx.dview if phase == "hpo" else runner.fx.root
                on, off = runner.base / f"{phase}-on", runner.base / f"{phase}-off"
                code, counter_on = runner.guarded_run(cfg, on, root)
                self.assertEqual(code, 0)
                with patch.object(tt, "ENABLED", False):
                    code, counter_off = runner.guarded_run(cfg, off, root)
                self.assertEqual(code, 0)
                self.assertEqual(counter_on, counter_off)
                self.assertEqual(counter_on["early_eval_reads"], 0)
                self.assert_identical_attempts(on, off, extra_bytes=("predictions.json",))
                records, summary = self.telemetry(on, cfg["training"]["epochs"])
                for row in records:
                    if row["type"] == "epoch":
                        self.assertIsNone(row["selection"])
                self.assertIsNone(summary["vram_max_allocated_bytes"])
                start = records[0]["context"]
                self.assertEqual((start["path"], start["phase"], start["resume"]), ("global", phase, False))

    def test_legacy_path_identical_and_resume_appends_a_session(self):
        runner = _selection.SelectionOnlyCPUTrainingTests(
            methodName="test_selection_only_real_forward_backward_never_opens_external_test")
        runner.torch = self.torch
        cfg = runner.toy_config(None)
        on, off = self.base / "legacy-on", self.base / "legacy-off"
        code, counter_on, _ = runner.guarded_run(cfg, on, forbid_test=False)
        self.assertEqual(code, 0)
        with patch.object(tt, "ENABLED", False):
            code, counter_off, _ = runner.guarded_run(cfg, off, forbid_test=False)
        self.assertEqual(code, 0)
        self.assertEqual(counter_on, counter_off)
        self.assert_identical_attempts(on, off)
        records, _ = self.telemetry(on, cfg["training"]["epochs"])
        history = json.loads((on / "metrics.json").read_text())
        epochs = [row for row in records if row["type"] == "epoch"]
        for row, journal in zip(epochs, history):
            self.assertEqual(row["selection"], journal["selection"])
            self.assertEqual(row["train_loss"], journal["train_loss"])
            self.assertEqual(row["learning_rate"], journal["learning_rate"])
        # Interrupted + resumed attempt: two sessions in one file, same artifacts.
        resumed = self.base / "legacy-resumed"
        code, _, _ = runner.guarded_run(cfg, resumed, forbid_test=False, stop_after_epoch=1)
        self.assertEqual(code, 75)
        interrupted = json.loads((resumed / tt.DIRECTORY / tt.SUMMARY_NAME).read_text(encoding="utf-8"))
        self.assertEqual((interrupted["status"], interrupted["epochs_recorded"]), ("interrupted", 1))
        code, _, _ = runner.guarded_run(cfg, resumed, forbid_test=False, resume=str(resumed / "checkpoint.pt"))
        self.assertEqual(code, 0)
        records, summary = self.telemetry(resumed, cfg["training"]["epochs"])
        self.assertEqual(summary["sessions"], 2)
        self.assertEqual([row["context"]["resume"] for row in records if row["type"] == "session_start"],
                         [False, True])
        self.assertEqual(_strip_utc(json.loads((resumed / "metrics.json").read_text())),
                         _strip_utc(json.loads((off / "metrics.json").read_text())))

    def test_runtime_validation_ignores_the_telemetry_subdirectory(self):
        from augment_atlas import runtime
        runner = _selection.SelectionOnlyCPUTrainingTests(
            methodName="test_selection_only_real_forward_backward_never_opens_external_test")
        runner.torch = self.torch
        cfg = runner.toy_config()
        out = self.base / "validated"
        code, _, _ = runner.guarded_run(cfg, out)
        self.assertEqual(code, 0)
        self.assertTrue((out / tt.DIRECTORY / tt.SUMMARY_NAME).is_file())
        self.assertEqual(runtime.validate_artifacts(out)["artifact_count"], 8)


if __name__ == "__main__":
    unittest.main()
