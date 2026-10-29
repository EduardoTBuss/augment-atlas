"""Campaign subprocess contracts using only synthetic PATH stubs and stdlib.

Run only this file: python -m unittest discover -s tests -p test_campaign_watchdog.py -v
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "scripts" / "campaign"
SPEC = importlib.util.spec_from_file_location("campaign_watchdog", CAMPAIGN / "watchdog.py")
wd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wd)

# One executable fixture shared by all fake tools. Nothing calls real hardware,
# Docker, a queue or the real worker. Unexpected commands fail the test.
STUB = r'''
import json, os, sys
from pathlib import Path
base = Path(os.environ["CAMPAIGN_STUB"])
config = json.loads((base / "config.json").read_text())
tool, *args = sys.argv[1:]
with (base / "calls.jsonl").open("a") as f:
    f.write(json.dumps([tool, *args]) + "\n")
if tool == "nvidia-smi":
    assert int((base / "vigia.pid").read_text()) > 0
    print(config.get("gpu", "0, 40, 1, 10"))
    sys.exit(config.get("gpu_rc", 0))
if tool == "sensors":
    print(json.dumps({"coretemp-isa-0000": {"Package id 0": {"temp1_input": config.get("cpu", 40)}}}))
    sys.exit(config.get("cpu_rc", 0))
if tool == "df":
    print("Filesystem 1024-blocks Used Available Capacity Mounted on")
    print("fixture 999999999 0 " + str(config.get("free_kib", 999999999)) + " 0% /fixture")
    sys.exit(config.get("df_rc", 0))
if tool == "docker":
    containers = config["containers"]
    if args[0] == "ps":
        # Deliberately over-return: watchdog must independently inspect labels.
        print("\n".join(c["Id"] for c in containers))
    elif args[0] == "inspect":
        if args[1].startswith("atlas-attempt-"):
            if config.get("missing_timing"):
                sys.exit(1)
            print(json.dumps([{"State": {"Running": False, "StartedAt": "2026-01-01T00:00:00Z", "FinishedAt": "2026-01-01T00:02:00Z"}}]))
        else:
            selected = [c.copy() for c in containers if c["Id"] in args[1:]]
            stopped = json.loads((base / "stopped.json").read_text()) if (base / "stopped.json").exists() else []
            for c in selected:
                if c["Id"] in stopped:
                    c["State"] = {"Running": config.get("still_running", False), "ExitCode": config.get("exit_code", 143), "OOMKilled": config.get("oom", False)}
            print(json.dumps(selected))
    elif args[0] == "stop":
        assert (base / "vigia.pid").read_text().strip() == "0", "driver gate must close before grace"
        stopped = json.loads((base / "stopped.json").read_text()) if (base / "stopped.json").exists() else []
        stopped.append(args[-1])
        (base / "stopped.json").write_text(json.dumps(stopped))
        print(args[-1])
        sys.exit(config.get("stop_rc", 0))
    else:
        sys.exit(91)
    sys.exit(0)
if tool == "worker":
    assert args[-1] == "--once" and "--resume" not in args
    settings = json.loads(Path(args[args.index("--settings") + 1]).read_text())
    phase = args[args.index("--phase") + 1]
    assert phase == settings["phase"]
    previous = json.loads((base / "workers.json").read_text()) if (base / "workers.json").exists() else []
    previous.append(settings)
    (base / "workers.json").write_text(json.dumps(previous))
    if config.get("create_attempt", True):
        (Path(settings["runs"]) / "job" / ("attempt-" + str(len(previous)))).mkdir(parents=True)
    if config.get("drop_watchdog"):
        (base / "watchdog.pid").write_text("0")
    if len(previous) >= config.get("stop_after", 2):
        (base / "driver.stop").touch()
    sys.exit(config.get("worker_rc", 0))
sys.exit(92)
'''


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".campaign-test-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.runs = self.base / "runs"
        self.runs.mkdir()
        self.code = self.base / "code"
        (self.code / "scripts").mkdir(parents=True)
        self.script = self.base / "stub.py"
        self.script.write_text(STUB, encoding="utf-8")
        for tool in ("nvidia-smi", "sensors", "docker", "df"):
            if os.name == "nt":
                path = self.bin / (tool + ".cmd")
                path.write_text('@echo off\n"' + sys.executable + '" "' + str(self.script) + '" ' + tool + ' %*\n')
            else:
                path = self.bin / tool
                path.write_text('#!' + sys.executable + '\n' +
                                'import runpy, sys\nsys.argv = [' + repr(str(self.script)) + ', ' + repr(tool) + '] + sys.argv[1:]\nrunpy.run_path(' + repr(str(self.script)) + ', run_name="__main__")\n')
                path.chmod(0o755)
        # Worker is a bash fixture, invoked through the exact existing interface.
        (self.code / "scripts" / "worker.sh").write_text(
            '#!/usr/bin/env bash\nexec "$PYTHON" "$CAMPAIGN_STUB/stub.py" worker "$@"\n', encoding="utf-8")
        self.env = os.environ.copy()
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env["PATH"],
                        CAMPAIGN_STUB=str(self.base), PYTHON=sys.executable,
                        PYTHONDONTWRITEBYTECODE="1")
        for tool in ("nvidia-smi", "sensors", "docker", "df"):
            self.assertEqual(Path(shutil.which(tool, path=self.env["PATH"])).parent, self.bin)
        def container(cid, labels, source=None):
            return dict(Id=cid, Name="/" + cid, Config=dict(Labels=labels),
                        State=dict(Running=True), Mounts=[] if source is None else [
                            dict(Type="bind", Destination="/output", Source=str(source))])
        self.config = dict(containers=[
            container("campaign", {"campaign.test": "mine"}, self.runs / "j" / "a"),
            container("foreign", {"campaign.test": "other", wd.ATTEMPT_LABEL: "x"}, self.runs / "j" / "b"),
            container("unlabelled", {}, self.runs / "j" / "c"),
            container("outside", {"campaign.test": "mine"}, self.base / "runs-foreign" / "j" / "a"),
            container("native", {wd.ATTEMPT_LABEL: "atlas-x"}, self.runs / "j" / "d"),
        ])
        self.pid = self.base / "watchdog.pid"
        self.pid.write_text(str(os.getpid()))
        self.settings = []
        for i, phase in enumerate(("hpo", "final")):
            path = self.base / ("settings-" + str(i) + ".json")
            path.write_text(json.dumps(dict(code=str(self.code), runs=str(self.runs), phase=phase, gpu="9")))
            self.settings.append(path)

    def write_config(self, **changes):
        self.config.update(changes)
        (self.base / "config.json").write_text(json.dumps(self.config))

    def calls(self, tool=None):
        path = self.base / "calls.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        return [r for r in rows if tool is None or r[0] == tool]

    def watchdog(self, *, max_seconds=1, extra=(), **changes):
        self.write_config(**changes)
        # Dedicated watchdog pidfile, distinct from the driver's live test pid.
        cmd = [sys.executable, str(CAMPAIGN / "watchdog.py"), "--label", "campaign.test=mine",
               "--runs-root", str(self.runs), "--log", str(self.base / "samples.jsonl"),
               "--summary", str(self.base / "summary.json"), "--pidfile", str(self.base / "vigia.pid"),
               "--interval", "0.01", "--max-seconds", str(max_seconds), *extra]
        result = subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=20)
        self.assertFalse((self.base / "vigia.pid").exists(), result.stderr)
        summary = json.loads((self.base / "summary.json").read_text())
        rows = [json.loads(l) for l in (self.base / "samples.jsonl").read_text().splitlines()]
        return result, summary, rows

    def driver(self, *extra, **changes):
        self.write_config(**changes)
        cmd = [shutil.which("bash"), str(CAMPAIGN / "driver.sh"), "--label", "synthetic",
               "--gpu", "1", "--runs-root", str(self.runs), "--state-dir", str(self.base / "state"),
               "--watchdog-pidfile", str(self.pid), "--stop-file", str(self.base / "driver.stop")]
        for path in self.settings:
            cmd += ["--settings", str(path)]
        result = subprocess.run([*cmd, *extra], env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual("", result.stderr, result.stdout)
        return result, (self.base / "state" / "driver.log").read_text()

    def test_gpu_trip_filters_exact_label_and_runs_root_and_default_grace(self):
        result, summary, rows = self.watchdog(gpu="0, 89, 99, 200")
        self.assertEqual(3, result.returncode, result.stderr)
        self.assertEqual([['docker', 'stop', '-t', '120', 'campaign']], [c for c in self.calls('docker') if c[1] == 'stop'])
        self.assertEqual("trip", summary["exit_reason"])
        self.assertEqual(89, summary["thresholds"]["gpu_trip_c"])
        self.assertEqual(["campaign"], [c["name"] for c in summary["trip"]["stopped"]])
        self.assertTrue(summary["trip"]["stopped"][0]["confirmed_stopped"])
        self.assertFalse(any(r.get("event") == "killed_after_grace" for r in rows))

    def test_cpu_trip_inclusive_ceiling(self):
        result, summary, _ = self.watchdog(cpu=90)
        self.assertEqual(3, result.returncode)
        self.assertIn("trip:cpu_package>=90.0", summary["trip"]["reasons"])

    def test_alert_without_trip(self):
        result, summary, rows = self.watchdog(gpu="0, 85, 0, 10", cpu=89.9)
        self.assertEqual(0, result.returncode)
        self.assertGreater(summary["alerts"], 0)
        self.assertTrue(any(r["reasons"] for r in rows))
        self.assertFalse([c for c in self.calls('docker') if c[1] == 'stop'])

    def test_killed_after_grace_event_has_retry_instruction_and_state(self):
        result, summary, rows = self.watchdog(gpu="0, 89, 0, 10", exit_code=137)
        self.assertEqual(3, result.returncode)
        event, = [r for r in rows if r.get("event") == "killed_after_grace"]
        self.assertEqual("campaign", event["id"])
        self.assertEqual(120, event["grace_s"])
        self.assertIn("retry; do not use --resume", event["operator_action"])
        self.assertEqual(137, event["state_after"]["ExitCode"])
        self.assertTrue(summary["trip"]["stopped"][0]["killed_after_grace"])

    def test_oom_is_not_reported_as_grace_kill(self):
        _, _, rows = self.watchdog(gpu="0, 89, 0, 10", exit_code=137, oom=True)
        self.assertFalse(any(r.get("event") == "killed_after_grace" for r in rows))

    def test_missing_one_sensor_fails_safe(self):
        result, summary, _ = self.watchdog(cpu_rc=1, max_seconds=5)
        self.assertEqual(3, result.returncode)
        self.assertEqual(3, summary["sensor_failures_max_consecutive"])

    def test_stop_failure_is_visible_in_summary(self):
        result, summary, _ = self.watchdog(gpu="0, 89, 0, 10", still_running=True, stop_rc=1)
        self.assertEqual(3, result.returncode)
        entry = summary["trip"]["stopped"][0]
        self.assertFalse(entry["confirmed_stopped"])
        self.assertIn("still running", entry["error"])

    def test_native_label_requires_runs_root_and_never_selects_unlabelled(self):
        for item in self.config["containers"]:
            selected = wd.is_trial_container(item, (wd.ATTEMPT_LABEL, None), [self.runs])
            self.assertEqual(item["Id"] in ("native", "foreign"), selected)
        self.assertFalse(wd.is_trial_container({"Config": {"Labels": None}}, ("k", None), []))

    def test_parsers_and_boundaries(self):
        self.assertEqual([], wd.parse_nvidia("0, nan, 0, 0\n1, [N/A], 0, 0"))
        self.assertEqual(["1"], [r["index"] for r in wd.parse_nvidia("garbage\n1, 40, 0, [N/A]")])
        self.assertEqual(70, wd.parse_cpu_package({"coretemp-x": {"Package id 0": {"temp1_input": 60}, "Package id 1": {"temp2_input": 70}, "Core 0": {"temp3_input": 99}}}))
        self.assertEqual((False, ["alert:gpu>=85"]), wd.decide(88.9, 40, alert_c=85, gpu_trip_c=89, cpu_trip_c=90))

    def test_driver_stop_file_does_not_start_worker(self):
        (self.base / "driver.stop").touch()
        result, log = self.driver()
        self.assertEqual(0, result.returncode)
        self.assertIn("stop: stop file", log)
        self.assertEqual([], self.calls())

    def test_driver_disk_reserve_is_on_runs_filesystem(self):
        result, log = self.driver('--reserve-gib', '2', free_kib=1)
        self.assertEqual(0, result.returncode)
        self.assertIn("2 GiB reserve", log)
        self.assertEqual([['df', '-Pk', str(self.runs)]], self.calls('df'))
        self.assertEqual([], self.calls('worker'))

    def test_driver_dead_or_missing_watchdog(self):
        self.pid.unlink()
        result, log = self.driver()
        self.assertEqual(0, result.returncode)
        self.assertIn("watchdog not running", log)
        self.assertEqual([], self.calls())

    def test_driver_worker_failure_preserves_rc_and_stops_after_one(self):
        result, log = self.driver(worker_rc=7, stop_after=99)
        self.assertEqual(7, result.returncode)
        self.assertEqual(1, len(self.calls('worker')))
        self.assertRegex(log, r"turn 1 done rc=7 wall=\d+s gpu=120s")
        self.assertIn("stop: worker exit 7", log)

    def test_driver_round_robin_gpu_override_and_source_settings_untouched(self):
        before = [p.read_bytes() for p in self.settings]
        result, log = self.driver()
        self.assertEqual(0, result.returncode)
        workers = json.loads((self.base / "workers.json").read_text())
        self.assertEqual(["hpo", "final"], [w["phase"] for w in workers])
        self.assertEqual(["1", "1"], [w["gpu"] for w in workers])
        self.assertEqual([87, 87], [w["max_temperature_c"] for w in workers])
        self.assertEqual(before, [p.read_bytes() for p in self.settings])
        self.assertIn("turn 2 done rc=0", log)
        ledger = [json.loads(l) for l in (self.base / "state" / "gpu-ledger.jsonl").read_text().splitlines()]
        self.assertEqual([120, 120], [r["gpu_s"] for r in ledger])
        self.assertEqual([], list((self.base / "state").glob("worker-settings-*")))

    def test_driver_predictive_budget_and_restart_reuse_ledger(self):
        result, log = self.driver('--cap-gpu-s', '1000', stop_after=99)
        self.assertEqual(0, result.returncode)
        self.assertEqual(1, len(self.calls('worker')))
        self.assertIn("GPU cap", log)
        self.assertTrue((self.base / "state" / "CAP_REACHED").exists())
        result, _ = self.driver('--cap-gpu-s', '1000', stop_after=99)
        self.assertEqual(0, result.returncode)
        self.assertEqual(1, len(self.calls('worker')))

    def test_driver_missing_container_timing_accounts_wall(self):
        result, _ = self.driver(missing_timing=True, stop_after=1)
        self.assertEqual(0, result.returncode)
        row = json.loads((self.base / "state" / "gpu-ledger.jsonl").read_text())
        self.assertEqual("wall_fallback", row["timing"])
        self.assertEqual(row["wall_s"], row["gpu_s"])

    def test_driver_idle_full_cycle_exits_and_logs_done(self):
        result, log = self.driver(create_attempt=False, stop_after=99)
        self.assertEqual(0, result.returncode)
        self.assertEqual(2, len(self.calls('worker')))
        self.assertIn("no claimable job in a full cycle", log)

    def test_driver_corrupt_ledger_fails_closed(self):
        (self.base / "state").mkdir()
        (self.base / "state" / "gpu-ledger.jsonl").write_text('{invalid')
        result, log = self.driver()
        self.assertEqual(1, result.returncode)
        self.assertIn("stop: error", log)
        self.assertEqual([], self.calls('worker'))

    def test_custom_grace_and_trip_threshold(self):
        result, summary, _ = self.watchdog(gpu="0, 88, 0, 10", extra=("--stop-timeout", "7", "--gpu-trip-c", "88"))
        self.assertEqual(3, result.returncode)
        self.assertEqual(7, summary["stop_timeout_s"])
        self.assertIn(['docker', 'stop', '-t', '7', 'campaign'], self.calls('docker'))

    def test_driver_stale_pid_does_not_start_worker(self):
        self.pid.write_text("999999999")
        result, log = self.driver()
        self.assertEqual(0, result.returncode)
        self.assertIn("watchdog not running", log)
        self.assertEqual([], self.calls('worker'))

    def test_driver_df_failure_stops_before_worker(self):
        result, log = self.driver(df_rc=1)
        self.assertEqual(1, result.returncode)
        self.assertIn("stop: error CalledProcessError", log)
        self.assertEqual([], self.calls('worker'))

    def test_driver_refuses_overlapping_gpu_driver_lock(self):
        import hashlib
        marker = self.runs / (".campaign-driver-" + hashlib.sha256(b"1").hexdigest()[:16] + ".lock")
        marker.write_text("occupied")
        result, log = self.driver()
        self.assertEqual(2, result.returncode)
        self.assertIn("GPU driver lock already exists", log)
        self.assertEqual("occupied", marker.read_text())
        self.assertEqual([], self.calls('worker'))

    def test_driver_preserves_stricter_worker_threshold(self):
        settings = json.loads(self.settings[0].read_text())
        settings["max_temperature_c"] = 84
        self.settings[0].write_text(json.dumps(settings))
        result, _ = self.driver(stop_after=1)
        self.assertEqual(0, result.returncode)
        workers = json.loads((self.base / "workers.json").read_text())
        self.assertEqual(84, workers[0]["max_temperature_c"])

    def test_driver_observed_long_job_predicts_budget_stop(self):
        state = self.base / "state"
        state.mkdir()
        row = dict(label="synthetic", gpu="1", phase="hpo", wall_s=100, gpu_s=1200)
        (state / "gpu-ledger.jsonl").write_text(json.dumps(row) + "\n")
        result, log = self.driver('--cap-gpu-s', '2300')
        self.assertEqual(0, result.returncode)
        self.assertIn("next-job estimate 1200s", log)
        self.assertEqual([], self.calls('worker'))

    def test_configurable_alert_without_trip(self):
        result, summary, _ = self.watchdog(gpu="0, 80, 0, 10", extra=("--alert-c", "80"))
        self.assertEqual(0, result.returncode)
        self.assertGreater(summary["alerts"], 0)
        self.assertFalse([c for c in self.calls('docker') if c[1] == 'stop'])

    def test_watchdog_existing_stop_file_exits_without_sampling(self):
        stop = self.base / "watchdog.stop"
        stop.touch()
        result, summary, rows = self.watchdog(extra=("--stop-file", str(stop)))
        self.assertEqual(0, result.returncode)
        self.assertEqual("stop_file", summary["exit_reason"])
        self.assertEqual([], rows)
        self.assertEqual([], self.calls())

    def test_threshold_above_eduardo_ceiling_is_rejected(self):
        self.write_config()
        result = subprocess.run([sys.executable, str(CAMPAIGN / 'watchdog.py'),
                                 '--label', 'campaign.test=mine', '--log', str(self.base / 'l'),
                                 '--summary', str(self.base / 's'), '--gpu-trip-c', '91'],
                                env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(2, result.returncode)
        self.assertIn('90 C ceiling', result.stderr)
        self.assertEqual([], self.calls())

    def test_driver_runs_root_mismatch_fails_before_worker(self):
        settings = json.loads(self.settings[0].read_text())
        settings['runs'] = str(self.base / 'other-runs')
        self.settings[0].write_text(json.dumps(settings))
        result, log = self.driver()
        self.assertEqual(1, result.returncode)
        self.assertIn('settings.runs differs', log)
        self.assertEqual([], self.calls('worker'))

    def test_driver_watchdog_gate_closes_between_turns(self):
        result, log = self.driver(drop_watchdog=True, stop_after=99)
        self.assertEqual(0, result.returncode)
        self.assertIn('watchdog not running', log)
        self.assertEqual(1, len(self.calls('worker')))

    def test_nonfinite_cpu_limit_is_rejected(self):
        self.write_config()
        result = subprocess.run([sys.executable, str(CAMPAIGN / 'watchdog.py'),
                                 '--label', 'campaign.test=mine', '--log', str(self.base / 'l'),
                                 '--summary', str(self.base / 's'), '--cpu-trip-c', 'nan'],
                                env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(2, result.returncode)
        self.assertEqual([], self.calls())

    def test_shared_native_label_cannot_run_without_campaign_root(self):
        result = subprocess.run([sys.executable, str(CAMPAIGN / 'watchdog.py'),
                                 '--label', wd.ATTEMPT_LABEL, '--log', str(self.base / 'l'),
                                 '--summary', str(self.base / 's')],
                                env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(2, result.returncode)
        self.assertIn('--runs-root is required', result.stderr)
        self.assertEqual([], self.calls())


if __name__ == "__main__":
    unittest.main()
