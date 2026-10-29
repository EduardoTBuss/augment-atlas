#!/usr/bin/env python3
"""Campaign thermal watchdog; stdlib only. Samples all GPUs and CPU packages.

Only running containers with --label are eligible. When --runs-root is supplied,
their /output bind mount must ALSO lie below a supplied root. Labels accept KEY
(presence, for the worker's org.augment-atlas.attempt label) or KEY=VALUE.
Trips exit 3; sensor loss in either source fails safe after consecutive failures.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import signal
import subprocess
import shutil
import sys
import time
from pathlib import Path

ATTEMPT_LABEL = "org.augment-atlas.attempt"


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run(args: list[str], timeout: float = 20) -> subprocess.CompletedProcess:
    return subprocess.run([shutil.which(args[0]) or args[0], *args[1:]],
                          capture_output=True, text=True, timeout=timeout)


def parse_nvidia(text: str) -> list[dict]:
    """Rows 'index, temperature.gpu, utilization.gpu, power.draw' -> dicts.

    Unparseable fields become None; a row without a temperature is dropped.
    """
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        def num(value):
            try:
                return float(value) if math.isfinite(float(value)) else None
            except (TypeError, ValueError):
                return None
        temp = num(parts[1])
        if temp is None:
            continue
        rows.append({"index": parts[0], "temp_c": temp,
                     "util_pct": num(parts[2]) if len(parts) > 2 else None,
                     "power_w": num(parts[3]) if len(parts) > 3 else None})
    return rows


def parse_cpu_package(sensors_json: dict) -> float | None:
    """Highest 'Package id N' temp*_input over coretemp chips, or None."""
    if not isinstance(sensors_json, dict):
        return None
    best = None
    for chip, body in sensors_json.items():
        if not chip.startswith("coretemp") or not isinstance(body, dict):
            continue
        for name, values in body.items():
            if not name.startswith("Package id") or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if key.startswith("temp") and key.endswith("_input") and isinstance(value, (int, float)) and math.isfinite(value):
                    best = value if best is None else max(best, float(value))
    return None if best is None else float(best)


def decide(gpu_max: float | None, cpu_pkg: float | None, *, alert_c: float,
           gpu_trip_c: float, cpu_trip_c: float) -> tuple[bool, list[str]]:
    """Return (trip, reasons). Alerts are reasons prefixed 'alert:'."""
    reasons = []
    trip = False
    if gpu_max is not None and gpu_max >= gpu_trip_c:
        trip = True
        reasons.append(f"trip:gpu>={gpu_trip_c}")
    if cpu_pkg is not None and cpu_pkg >= cpu_trip_c:
        trip = True
        reasons.append(f"trip:cpu_package>={cpu_trip_c}")
    if not trip:
        if gpu_max is not None and gpu_max >= alert_c:
            reasons.append(f"alert:gpu>={alert_c}")
        if cpu_pkg is not None and cpu_pkg >= alert_c:
            reasons.append(f"alert:cpu_package>={alert_c}")
    return trip, reasons


def is_trial_container(inspect: dict, label: tuple[str, str | None], runs_roots: list[Path]) -> bool:
    """Label is mandatory; a runs-root is an additional restriction, never OR."""
    labels = (inspect.get("Config") or {}).get("Labels") or {}
    key, value = label
    if key not in labels or (value is not None and labels[key] != value):
        return False
    if not runs_roots:
        return True
    for mount in inspect.get("Mounts") or []:
        if mount.get("Destination") != "/output" or mount.get("Type") != "bind":
            continue
        source = Path(mount.get("Source", ""))
        if source.is_absolute() and any(source.resolve().is_relative_to(root) for root in runs_roots):
            return True
    return False


def trial_containers(label: tuple[str, str], runs_roots: list[Path]) -> list[dict]:
    query = run(["docker", "ps", "-q", "--no-trunc", "--filter",
                 "label=" + label[0] + ("=" + label[1] if label[1] is not None else "")])
    if query.returncode:
        raise RuntimeError("docker ps failed")
    ids = query.stdout.split()
    if not ids:
        return []
    result = run(["docker", "inspect", *ids])
    if result.returncode:
        # A container may vanish between ps and inspect; inspect individually.
        objects = []
        for cid in ids:
            r = run(["docker", "inspect", cid])
            if r.returncode == 0:
                objects.extend(json.loads(r.stdout))
    else:
        objects = json.loads(result.stdout)
    return [{"id": o["Id"], "name": o.get("Name", "").lstrip("/")}
            for o in objects if (o.get("State") or {}).get("Running") is True
            and is_trial_container(o, label, runs_roots)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="JSONL sample log")
    p.add_argument("--summary", required=True, help="summary JSON written on exit")
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--alert-c", type=float, default=85.0)
    p.add_argument("--gpu-trip-c", type=float, default=89.0)
    p.add_argument("--cpu-trip-c", type=float, default=90.0)
    p.add_argument("--label", required=True, help="KEY or KEY=VALUE campaign container selector")
    p.add_argument("--pidfile", help="exclusive pidfile; default: log path with .pid suffix")
    p.add_argument("--runs-root", action="append", default=[], help="repeatable; worker runs roots of the trial")
    p.add_argument("--stop-file", help="exit 0 when this file exists")
    p.add_argument("--max-sensor-failures", type=int, default=3)
    p.add_argument("--max-seconds", type=float, default=None, help="optional self-exit (exit 0)")
    p.add_argument("--stop-timeout", type=int, default=120,
                   help="docker stop -t seconds (grace for the end-of-epoch checkpoint before SIGKILL)")
    a = p.parse_args(argv)
    key, separator, value = a.label.partition("=")
    if not key or (separator and not value):
        p.error("--label must be KEY or KEY=VALUE")
    if (not math.isfinite(a.interval) or a.interval <= 0 or a.stop_timeout < 0
            or a.max_sensor_failures < 1):
        p.error("interval/failure count must be positive and grace nonnegative")
    if not (all(math.isfinite(v) for v in (a.alert_c, a.gpu_trip_c, a.cpu_trip_c))
            and 0 < a.alert_c <= min(a.gpu_trip_c, a.cpu_trip_c) <= 90
            and max(a.gpu_trip_c, a.cpu_trip_c) <= 90):
        p.error("require 0 < alert <= trip thresholds <= Eduardo's 90 C ceiling")
    if a.max_seconds is not None and (not math.isfinite(a.max_seconds) or a.max_seconds <= 0):
        p.error("--max-seconds must be finite and positive")
    label = (key, value if separator else None)
    roots = [Path(r).resolve() for r in a.runs_root]
    if key == ATTEMPT_LABEL and not separator and not roots:
        p.error("--runs-root is required with the shared native attempt label")
    if any(not root.is_dir() for root in roots):
        p.error("--runs-root directories must already exist")
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    summary = {"started_utc": utc(), "pid": os.getpid(), "thresholds": {"alert_c": a.alert_c,
               "gpu_trip_c": a.gpu_trip_c, "cpu_trip_c": a.cpu_trip_c}, "interval_s": a.interval,
               "label": a.label, "stop_timeout_s": a.stop_timeout, "runs_roots": [str(r) for r in roots], "samples": 0,
               "max_gpu_c": None, "max_cpu_package_c": None, "max_gap_s": 0.0, "alerts": 0,
               "sensor_failures_max_consecutive": 0, "trip": None, "exit_reason": None}
    failures = 0
    last = None
    started = time.monotonic()
    code = 0
    for target in (a.log, a.summary):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    pidfile = Path(a.pidfile) if a.pidfile else Path(a.log).with_suffix(".pid")
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite another vigia's ownership marker, including a stale one.
    with pidfile.open("x", encoding="ascii") as handle:
        handle.write(str(os.getpid()) + "\n")
    log = open(a.log, "a", encoding="utf-8", buffering=1)
    try:
        while not stop[0]:
            if a.stop_file and Path(a.stop_file).exists():
                summary["exit_reason"] = "stop_file"
                break
            if a.max_seconds is not None and time.monotonic() - started >= a.max_seconds:
                summary["exit_reason"] = "max_seconds"
                break
            now = time.monotonic()
            if last is not None:
                summary["max_gap_s"] = max(summary["max_gap_s"], round(now - last, 3))
            last = now
            gpus, cpu, errors = [], None, []
            try:
                r = run(["nvidia-smi", "--query-gpu=index,temperature.gpu,utilization.gpu,power.draw",
                         "--format=csv,noheader,nounits"])
                gpus = parse_nvidia(r.stdout) if r.returncode == 0 else []
                if r.returncode:
                    errors.append("nvidia-smi exit %d" % r.returncode)
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append("nvidia-smi " + type(exc).__name__)
            try:
                r = run(["sensors", "-j"])
                cpu = parse_cpu_package(json.loads(r.stdout)) if r.returncode == 0 else None
                if r.returncode:
                    errors.append("sensors exit %d" % r.returncode)
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                errors.append("sensors " + type(exc).__name__)
            gpu_max = max((g["temp_c"] for g in gpus), default=None)
            failures = failures + 1 if (gpu_max is None or cpu is None) else 0
            summary["sensor_failures_max_consecutive"] = max(summary["sensor_failures_max_consecutive"], failures)
            trip, reasons = decide(gpu_max, cpu, alert_c=a.alert_c, gpu_trip_c=a.gpu_trip_c, cpu_trip_c=a.cpu_trip_c)
            if failures >= a.max_sensor_failures:
                trip = True
                reasons.append(f"trip:sensor_unavailable x{failures}")
            try:
                trial = trial_containers(label, roots)
            except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
                trial = []
                errors.append("docker " + type(exc).__name__)
            summary["samples"] += 1
            if gpu_max is not None:
                summary["max_gpu_c"] = gpu_max if summary["max_gpu_c"] is None else max(summary["max_gpu_c"], gpu_max)
            if cpu is not None:
                summary["max_cpu_package_c"] = cpu if summary["max_cpu_package_c"] is None else max(summary["max_cpu_package_c"], cpu)
            if any(r.startswith("alert:") for r in reasons):
                summary["alerts"] += 1
            row = {"utc": utc(), "gpus": gpus, "cpu_package_c": cpu, "gpu_max_c": gpu_max,
                   "reasons": reasons, "trial_containers": [c["name"] for c in trial], "errors": errors}
            log.write(json.dumps(row, sort_keys=True) + "\n")
            if trip:
                # Close the driver gate BEFORE waiting for graceful container stops.
                pidfile.write_text("0\n", encoding="ascii")
                stopped = []
                for c in trial:
                    entry = {"name": c["name"], "id": c["id"]}
                    try:
                        r = run(["docker", "stop", "-t", str(a.stop_timeout), c["id"]],
                                timeout=a.stop_timeout + 60)
                        entry["stop_exit"] = r.returncode
                        after = run(["docker", "inspect", c["id"]])
                        if after.returncode:
                            raise RuntimeError("post-stop inspection failed (container must be retained)")
                        state = json.loads(after.stdout)[0]["State"]
                        entry["state_after"] = state
                        entry["confirmed_stopped"] = state.get("Running") is False
                        if not entry["confirmed_stopped"]:
                            entry["error"] = "container still running after stop"
                        if (r.returncode == 0 and entry["confirmed_stopped"] and state.get("ExitCode") == 137
                                and state.get("OOMKilled") is False):
                            event = {"utc": utc(), "event": "killed_after_grace", **entry,
                                     "grace_s": a.stop_timeout, "inferred_from_exit_137": True,
                                     "operator_action": "failed: retry; do not use --resume"}
                            log.write(json.dumps(event, sort_keys=True) + "\n")
                            entry["killed_after_grace"] = True
                    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError,
                            TypeError, IndexError, RuntimeError) as exc:
                        entry["error"] = type(exc).__name__ + ": " + str(exc)
                    stopped.append(entry)
                summary["trip"] = {"utc": utc(), "reasons": reasons, "gpu_max_c": gpu_max,
                                   "cpu_package_c": cpu, "stopped": stopped}
                log.write(json.dumps({"utc": utc(), "event": "TRIP", **summary["trip"]}, sort_keys=True) + "\n")
                summary["exit_reason"] = "trip"
                code = 3
                break
            # Sleep in small steps so SIGTERM and the stop file are honoured quickly.
            deadline = now + a.interval
            while not stop[0] and time.monotonic() < deadline:
                if a.stop_file and Path(a.stop_file).exists():
                    break
                if a.max_seconds is not None and time.monotonic() - started >= a.max_seconds:
                    break
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        if summary["exit_reason"] is None:
            summary["exit_reason"] = "signal"
    except Exception as exc:
        summary["exit_reason"] = "error"
        summary["error"] = type(exc).__name__ + ": " + str(exc)
        code = 1
    finally:
        pidfile.unlink(missing_ok=True)
        summary["ended_utc"] = utc()
        log.close()
        Path(a.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
