#!/usr/bin/env bash
# Serial campaign driver derived from the operational round-robin driver.
# Host Python stdlib runs the control loop; worker.sh remains the interface.
set -euo pipefail
umask 077
exec "${PYTHON:-python3}" - "$@" <<'PY'
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def alive(pidfile):
    try:
        pid = int(pidfile.read_text().strip())
        if pid <= 0:
            return False
        if os.name == "nt":
            # os.kill(pid, 0) terminates the target on Windows; query instead.
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x100000, False, pid)
            if not handle:
                return False
            try:
                return kernel.WaitForSingleObject(handle, 0) == 258
            finally:
                kernel.CloseHandle(handle)
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def attempts(runs):
    return {p for job in runs.iterdir() if job.is_dir() and not job.name.startswith("worker-")
            for p in job.iterdir() if p.is_dir()}


def gpu_seconds(created, fallback):
    total = 0.0
    for attempt in created:
        try:
            r = subprocess.run([shutil.which("docker") or "docker", "inspect", "atlas-" + attempt.name],
                               capture_output=True, text=True, timeout=20, check=True)
            state = json.loads(r.stdout)[0]["State"]
            def timestamp(value):
                if value.startswith("0001"):
                    raise ValueError("missing container timestamp")
                return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            start, end = timestamp(state["StartedAt"]), timestamp(state["FinishedAt"])
            duration = (end - start).total_seconds()
            if state.get("Running") or duration < 0:
                raise ValueError("nonterminal container timing")
            total += duration
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, IndexError):
            return fallback, "wall_fallback"
    return math.ceil(total), "container"


def main():
    p = argparse.ArgumentParser(description="Serial one-job-per-GPU campaign driver")
    p.add_argument("--label", required=True, help="operator campaign tag for logs/ledger")
    p.add_argument("--gpu", required=True, help="one GPU index or UUID")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--state-dir", required=True, help="private logs and persistent GPU ledger")
    p.add_argument("--watchdog-pidfile", required=True)
    p.add_argument("--stop-file", help="default: STATE_DIR/driver.stop")
    p.add_argument("--settings", action="append", required=True, help="repeat for round-robin")
    p.add_argument("--reserve-gib", type=float, default=20)
    p.add_argument("--cap-gpu-s", type=float, default=8 * 3600)
    p.add_argument("--default-estimate-s", type=float, default=900)
    p.add_argument("--worker-stop-c", type=float, default=87)
    a = p.parse_args()
    for name in ("reserve_gib", "cap_gpu_s", "default_estimate_s", "worker_stop_c"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            p.error(name + " must be finite and nonnegative")
    if not 0 < a.worker_stop_c <= 90 or a.default_estimate_s <= 0:
        p.error("worker threshold must be 0 < C <= 90; estimate must be positive")
    if not a.gpu or "," in a.gpu:
        p.error("--gpu must select exactly one GPU")
    runs = Path(a.runs_root).resolve()
    if not runs.is_dir():
        p.error("--runs-root must already exist on the runs filesystem")
    state = Path(a.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    stop = Path(a.stop_file) if a.stop_file else state / "driver.stop"
    pidfile = Path(a.watchdog_pidfile)
    ledger = state / "gpu-ledger.jsonl"
    logpath = state / "driver.log"
    def log(message):
        with logpath.open("a", encoding="utf-8") as handle:
            handle.write(utc() + " " + message + "\n")
    # SIGKILL leaves an ownership marker; verify no driver/worker remains before removal.
    lock = runs / (".campaign-driver-" + hashlib.sha256(a.gpu.encode()).hexdigest()[:16] + ".lock")
    try:
        with lock.open("x", encoding="ascii") as handle:
            handle.write(str(os.getpid()) + "\n")
    except FileExistsError:
        log("stop: GPU driver lock already exists")
        return 2
    try:
        configs = []
        for path in a.settings:
            source = Path(path).resolve()
            settings = json.loads(source.read_text(encoding="utf-8"))
            if settings.get("phase") not in ("smoke", "hpo", "final"):
                raise ValueError("invalid worker phase")
            if Path(settings["runs"]).resolve() != runs:
                raise ValueError("settings.runs differs from --runs-root")
            settings["gpu"] = a.gpu
            settings["max_temperature_c"] = min(float(settings.get("max_temperature_c", a.worker_stop_c)), a.worker_stop_c)
            if not math.isfinite(settings["max_temperature_c"]) or settings["max_temperature_c"] <= 0:
                raise ValueError("invalid worker temperature limit")
            configs.append((source.name, settings))
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
        for row in rows:
            duration = row.get("gpu_s", row["wall_s"])
            if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid ledger GPU time")
            if row.get("gpu", a.gpu) != a.gpu or row.get("label") != a.label:
                raise ValueError("state directory belongs to another GPU or campaign")
        idle = turn = 0
        log(f"start label={a.label} gpu={a.gpu} settings={len(configs)} pid={os.getpid()}")
        while True:
            name, settings = configs[turn % len(configs)]
            phase = settings["phase"]
            if stop.exists():
                log("stop: stop file")
                return 0
            if not alive(pidfile):
                log("stop: watchdog not running (trip or exit)")
                return 0
            disk = subprocess.run([shutil.which("df") or "df", "-Pk", str(runs)], capture_output=True, text=True, timeout=20, check=True)
            free = int(disk.stdout.splitlines()[-1].split()[3]) * 1024
            if free < a.reserve_gib * 1024 ** 3:
                log(f"stop: disk free {free} < {a.reserve_gib:g} GiB reserve")
                return 0
            used = sum(row.get("gpu_s", row["wall_s"]) for row in rows)
            observed = [row.get("gpu_s", row["wall_s"]) for row in rows if row["phase"] == phase]
            estimate = max([a.default_estimate_s, *observed])
            if used + estimate > a.cap_gpu_s:
                log(f"stop: GPU cap (used {used}s + next-job estimate {estimate}s > {a.cap_gpu_s}s)")
                (state / "CAP_REACHED").touch()
                return 0
            turn += 1
            before = attempts(runs)
            start = time.monotonic()
            log(f"turn {turn} phase={phase} settings={name} used={used}s")
            # Source settings stay untouched; never pass --resume automatically.
            with tempfile.TemporaryDirectory(prefix="worker-settings-", dir=state) as temporary:
                effective = Path(temporary) / "settings.json"
                effective.write_text(json.dumps(settings), encoding="utf-8")
                with (state / "worker.log").open("a", encoding="utf-8") as workerlog:
                    result = subprocess.run([shutil.which("bash") or "bash", "./scripts/worker.sh", "--settings", str(effective),
                                             "--phase", phase, "--once"], cwd=settings["code"],
                                            stdout=workerlog, stderr=subprocess.STDOUT)
            rc = result.returncode
            wall = math.ceil(time.monotonic() - start)
            created = attempts(runs) - before
            gpu, timing = gpu_seconds(created, wall) if created else (0, "no_attempt")
            if created:
                row = dict(utc=utc(), label=a.label, gpu=a.gpu, phase=phase, settings=name,
                           wall_s=wall, gpu_s=gpu, timing=timing, attempts=len(created), rc=rc)
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                rows.append(row)
                idle = 0
            else:
                idle += 1
                log(f"turn {turn}: no job claimed (rc={rc}, {wall}s)")
            log(f"turn {turn} done rc={rc} wall={wall}s gpu={gpu}s")
            if rc != 0:
                log(f"stop: worker exit {rc} (inspect)")
                return rc if 0 < rc < 126 else 1
            if idle >= len(configs):
                log("stop: no claimable job in a full cycle")
                return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as exc:
        log("stop: error " + type(exc).__name__ + ": " + str(exc))
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
PY
