"""atlas sentinel: observation-only node telemetry collector (fleet protocol section 5).

One process per node, independent of Git and of the training worker:

- fast tier (0.5 s): GPU stream values + CPU package temperature, ``latest.json``;
- slow tier (10 s, own thread): CPU/memory/disk/RAPL/fans/GPU details/crash list;
- events: collector_start/stop, reboot_detected, sample_gap, gpu_stream_gap,
  sensor_state_change, rapl_wrap, rapl_suspect_multi_wrap, var_crash_new,
  threshold_observed.

It never acts: no signal to any process except its own ``nvidia-smi`` child
(watchdog restart, section 5.1), no container/GPU/fan/power change, no
privilege (it refuses to run as root unless ``--allow-root``). A failing sensor
becomes ``null`` + reason; a failing tick is counted and the loop continues.

Shared state and locks (each lock is a leaf: no code path holds one lock while
acquiring another, so there is no lock ordering to violate):

- ``GpuStream._lock``   -> latest per-GPU values, status (stream reader vs. fast loop / slow thread);
- ``EventLog._lock``    -> event seq + events writer (fast loop, slow thread, stream supervisor);
- ``Sentinel._slow_lock`` -> latest slow snapshot used by ``latest.json`` (slow thread vs. fast loop);
- fast/slow writers and ``state.json`` are single-threaded (fast loop / slow thread only).

CLI (no change to ``cli.py``)::

    python -m augment_atlas.sentinel run --dir DIR --node-id ID [--duration S] [...]
    python -m augment_atlas.sentinel report --dir DIR [--validate]
    python -m augment_atlas.sentinel latest --dir DIR
    python -m augment_atlas.sentinel probe --node-id ID
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from . import fleet_schema
from . import telemetry_v1 as tv

STREAM_STATUS_REASON = {"starting": "first_sample", "tool_absent": "tool_absent",
                        "permission_denied": "permission_denied", "stalled": "tool_timeout",
                        "exited": "collector_error", "timeout": "tool_timeout", "disabled": "disabled_by_config",
                        "parse_error": "parse_error", "error": "collector_error"}


def _dumps(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


# ------------------------------------------------------------------ file helpers

def atomic_write_json(path, document, fsync=False, replace=os.replace, retries=50):
    """Write ``document`` to ``path`` via temp file + rename in the same directory.

    A reader sees either the previous or the new complete file, never a partial
    one. On Windows ``os.replace`` fails while a reader holds the target open, so
    the rename is retried briefly.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_dumps(document))
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    for _ in range(retries):
        try:
            replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.005)
    replace(temporary, path)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_latest(path, now_mono=None, boot_id=None, max_age_s=5.0):
    """Consumer side of ``latest.json`` (section 5.6). Age is monotonic, same boot only."""
    document = read_json(path)
    if document is None:
        return {"status": "missing", "age_s": None, "document": None}
    try:
        fleet_schema.validate_sentinel_latest(document)
    except fleet_schema.SchemaError as exc:
        return {"status": "invalid", "age_s": None, "document": None, "error": str(exc)}
    if boot_id is not None and document["boot_id"] != boot_id:
        return {"status": "other_boot", "age_s": None, "document": document}
    now_mono = time.monotonic() if now_mono is None else now_mono
    age = now_mono - document["mono_s"]
    return {"status": "stale" if age > max_age_s or age < -1.0 else "ok", "age_s": round(age, 3),
            "document": document}


def compress_segment(path):
    """gzip a closed segment atomically (``.gz.tmp`` -> fsync -> rename) and drop the original."""
    path = Path(path)
    target = path.with_name(path.name + ".gz")
    temporary = path.with_name(path.name + ".gz.tmp")
    with open(path, "rb") as source, open(temporary, "wb") as raw:
        with gzip.GzipFile(filename=path.name, mode="wb", fileobj=raw, mtime=0) as compressed:
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                compressed.write(chunk)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, target)
    path.unlink()
    return target


def iter_segment_lines(path):
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except (OSError, EOFError):
        return


def segment_files(directory, prefix):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(item for item in directory.iterdir()
                  if item.name.startswith(prefix + "-") and (item.name.endswith(".jsonl") or item.name.endswith(".jsonl.gz")))


def last_record(directory, prefix):
    """Last complete JSON line of the newest segment (``.jsonl`` tail or ``.gz``)."""
    for path in reversed(segment_files(directory, prefix)):
        if path.name.endswith(".jsonl"):
            try:
                with open(path, "rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 65536))
                    tail = handle.read().decode("utf-8", "replace").splitlines()
            except OSError:
                continue
            for line in reversed(tail):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
        else:
            found = None
            for found in iter_segment_lines(path):
                pass
            if found is not None:
                return found
    return None


class SegmentWriter:
    """JSON Lines writer with batched fsync and time-based rotation.

    - every line is written and flushed (visible to readers immediately);
    - ``fsync`` at most once per ``fsync_interval_s`` of monotonic time
      (``0`` = every write), so at most that much is lost before a host freeze;
    - after ``rotate_seconds`` the segment is closed (forced fsync) and handed
      to ``on_closed`` (gzip in the background).
    """

    def __init__(self, directory, prefix, wall=time.time, fsync_interval_s=5.0, rotate_seconds=3600.0,
                 fsync=os.fsync, on_closed=None):
        self.directory = Path(directory)
        self.prefix = prefix
        self.wall = wall
        self.fsync_interval_s = fsync_interval_s
        self.rotate_seconds = rotate_seconds
        self._fsync = fsync
        self.on_closed = on_closed
        self._file = None
        self._path = None
        self._opened_mono = None
        self._last_fsync = None
        self._pending = False
        self._counter = 0
        self.last_record = None
        self.last_synced_record = None
        self.fsync_count = 0
        self.segments = []

    @property
    def path(self):
        return self._path

    def _open(self, now_mono):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.wall()))
        while True:
            self._counter += 1
            candidate = self.directory / f"{self.prefix}-{stamp}-{self._counter:04d}.jsonl"
            try:
                self._file = open(candidate, "x", encoding="utf-8", newline="\n")
                break
            except FileExistsError:
                continue
        self._path = candidate
        self._opened_mono = now_mono
        self._last_fsync = now_mono
        self.segments.append(candidate)

    def write(self, record, now_mono):
        """Returns True if this write triggered an fsync."""
        if self._file is None:
            self._open(now_mono)
        elif now_mono - self._opened_mono >= self.rotate_seconds:
            self.rotate(now_mono)
        self._file.write(_dumps(record) + "\n")
        self._file.flush()
        self.last_record = record
        self._pending = True
        return self.maybe_fsync(now_mono)

    def maybe_fsync(self, now_mono, force=False):
        if self._file is None or not self._pending:
            return False
        if force or now_mono - self._last_fsync >= self.fsync_interval_s:
            self._fsync(self._file.fileno())
            self._last_fsync = now_mono
            self._pending = False
            self.fsync_count += 1
            self.last_synced_record = self.last_record
            return True
        return False

    def rotate(self, now_mono):
        closed = self._close_current()
        self._open(now_mono)
        if closed is not None and self.on_closed is not None:
            self.on_closed(closed)

    def _close_current(self):
        if self._file is None:
            return None
        self.maybe_fsync(0.0, force=True)
        self._file.close()
        closed, self._file = self._path, None
        return closed

    def close(self):
        self._close_current()


class EventLog:
    """Thread-safe event writer (fsync on every event: events are rare and evidential)."""

    def __init__(self, writer, node_id, boot_id, clock, wall, start_seq=0):
        self.writer = writer
        self.node_id = node_id
        self.boot_id = boot_id
        self.clock = clock
        self.wall = wall
        self.seq = start_seq
        self.counts = {}
        self._lock = threading.Lock()

    def emit(self, name, **payload):
        with self._lock:
            now = self.clock()
            self.seq += 1
            record = tv.envelope("event", self.node_id, self.boot_id, self.seq, now, tv.utc_label(self.wall()))
            record["event"] = name
            record.update(payload)
            # Nested nulls (e.g. a previous sample digest) are labelled, never left bare.
            record["null_reasons"] = {path: "not_exposed" for path in fleet_schema.uncovered_nulls(payload, {})}
            self.counts[name] = self.counts.get(name, 0) + 1
            try:
                self.writer.write(record, now)
            except (OSError, ValueError):
                pass
            return record


# --------------------------------------------------------------------- GPU feed

class GpuStream:
    """Long-lived ``nvidia-smi ... -lms 500`` child with a watchdog (or a poll loop).

    The watchdog kills and reaps the child if no line arrives for
    ``stall_timeout_s`` (or if it exits), emits ``gpu_stream_gap`` and restarts
    after a backoff of 1, 2, 4... up to ``backoff_max_s`` seconds; the backoff
    resets after ``healthy_reset_s`` of continuous output.
    """

    def __init__(self, nvidia_smi, emit, clock=time.monotonic, mode="stream", interval_s=0.5,
                 stall_timeout_s=3.0, backoff_initial_s=1.0, backoff_max_s=30.0, healthy_reset_s=30.0,
                 popen=subprocess.Popen, run_cmd=tv.run_command, cmd_timeout_s=4.0, fields=tv.FAST_QUERY_FIELDS):
        self.nvidia_smi = list(nvidia_smi)
        self.emit = emit
        self.clock = clock
        self.mode = mode
        self.interval_s = interval_s
        self.stall_timeout_s = stall_timeout_s
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self.healthy_reset_s = healthy_reset_s
        self.popen = popen
        self.run_cmd = run_cmd
        self.cmd_timeout_s = cmd_timeout_s
        self.fields = tuple(fields)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._latest = {}
        self._status = "disabled" if mode == "off" else "starting"
        self._last_line = None
        self.lines = 0
        self.bad_lines = 0
        self.restarts = 0
        self.reaped = []

    # -- public
    def start(self):
        if self.mode == "off":
            return
        target = self._supervise if self.mode == "stream" else self._poll
        self._thread = threading.Thread(target=target, name="sentinel-gpu", daemon=True)
        self._thread.start()

    def stop(self, timeout=10.0):
        self._stop.set()
        proc = self._proc
        if proc is not None:
            self._terminate(proc)
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self):
        with self._lock:
            return {"status": self._status, "reason": STREAM_STATUS_REASON.get(self._status),
                    "gpus": {index: {"mono": item["mono"], "values": dict(item["values"]),
                                     "reasons": dict(item["reasons"])} for index, item in self._latest.items()}}

    def uuids(self):
        with self._lock:
            return {index: item["values"].get("uuid") for index, item in self._latest.items()}

    # -- internals
    def _set_status(self, status):
        with self._lock:
            changed = self._status != status
            self._status = status
        return changed

    def _ingest(self, text, now):
        rows, bad = tv.parse_nvidia_csv(text, self.fields)
        with self._lock:
            for row in rows:
                self._latest[row["index"]] = {"mono": now, "values": row["values"], "reasons": row["reasons"]}
            self._last_line = now
            self.lines += text.count("\n") or 1
            self.bad_lines += bad
            if rows:
                self._status = "ok"

    def _terminate(self, proc):
        """Stop our own nvidia-smi child and reap it (no zombie)."""
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        except OSError:
            pass
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except OSError:
                pass

    def _reader(self, proc):
        try:
            for line in proc.stdout:
                self._ingest(line, self.clock())
        except (OSError, ValueError):
            return

    def _supervise(self):
        backoff = self.backoff_initial_s
        argv = tv.nvidia_query_argv(self.nvidia_smi, self.fields, loop_ms=int(round(self.interval_s * 1000)))
        env = dict(os.environ, LC_ALL="C", LANG="C")
        while not self._stop.is_set():
            try:
                proc = self.popen([str(part) for part in argv], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                  bufsize=1, env=env)
            except (FileNotFoundError, PermissionError) as exc:
                if isinstance(exc, PermissionError):
                    exc = tv.absent_or_denied(argv, exc)
                status = "tool_absent" if isinstance(exc, FileNotFoundError) else "permission_denied"
                if self._set_status(status):
                    self.emit("gpu_stream_gap", reason=status, gap_s=0.0, returncode=None, restart_in_s=backoff,
                              restarts=self.restarts)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.backoff_max_s)
                continue
            self._proc = proc
            started = self.clock()
            with self._lock:
                self._last_line = started
                lines_before = self.lines
            reader = threading.Thread(target=self._reader, args=(proc,), name="sentinel-gpu-reader", daemon=True)
            reader.start()
            cause = None
            while not self._stop.wait(0.1):
                if proc.poll() is not None:
                    cause = "exit"
                    break
                with self._lock:
                    silence = self.clock() - self._last_line
                if silence > self.stall_timeout_s:
                    cause = "stall"
                    break
            with self._lock:
                silence = self.clock() - self._last_line
                produced = self.lines - lines_before
            self._terminate(proc)
            reader.join(2.0)
            self._proc = None
            self.reaped.append(proc.returncode)
            if self._stop.is_set():
                break
            if produced and self.clock() - started - silence >= self.healthy_reset_s:
                backoff = self.backoff_initial_s
            self._set_status("stalled" if cause == "stall" else "exited")
            self.restarts += 1
            self.emit("gpu_stream_gap", reason=cause, gap_s=round(silence, 3), returncode=proc.returncode,
                      restart_in_s=backoff, restarts=self.restarts)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self.backoff_max_s)

    def _poll(self):
        argv = tv.nvidia_query_argv(self.nvidia_smi, self.fields)
        backoff = self.backoff_initial_s
        next_t = self.clock()
        while not self._stop.is_set():
            try:
                text = self.run_cmd(argv, self.cmd_timeout_s)
                self._ingest(text, self.clock())
                backoff = self.backoff_initial_s
                wait = self.interval_s
            except Exception as exc:  # noqa: BLE001
                reason = tv.command_reason(exc)
                status = {"tool_absent": "tool_absent", "tool_timeout": "timeout",
                          "permission_denied": "permission_denied"}.get(reason, "error")
                if self._set_status(status):
                    self.emit("gpu_stream_gap", reason=status, gap_s=0.0, returncode=None, restart_in_s=backoff,
                              restarts=self.restarts)
                wait = backoff
                backoff = min(backoff * 2, self.backoff_max_s)
            next_t = max(next_t + wait, self.clock())
            self._stop.wait(max(0.0, next_t - self.clock()))


# --------------------------------------------------------------------- sentinel

class SentinelConfig:
    def __init__(self, out_dir, node_id, fast_interval_s=0.5, slow_interval_s=10.0, duration_s=None,
                 gpu_mode="stream", nvidia_smi=("nvidia-smi",), stall_timeout_s=3.0, backoff_initial_s=1.0,
                 backoff_max_s=30.0, gpu_stale_after_s=1.5, fsync_interval_s=5.0, rotate_seconds=3600.0,
                 cmd_timeout_s=4.0, disks=None, reserves=None, containers=False, compress_closed=True,
                 thresholds=None, nice=10, max_state_change_events=16):
        fleet_schema._str(node_id, "node_id", fleet_schema.SLUG)
        if gpu_mode not in {"stream", "poll", "off"}:
            raise ValueError("gpu_mode must be stream, poll or off")
        if fast_interval_s <= 0 or slow_interval_s <= 0:
            raise ValueError("intervals must be positive")
        for key in (disks or {}):
            fleet_schema._str(key, "disk key", fleet_schema.STORAGE_KEY)
        unknown = sorted(set(thresholds or {}) - {"gpu.temp_c", "cpu_pkg.temp_c"})
        if unknown:
            raise ValueError(f"unsupported observe thresholds {unknown}")
        self.out_dir = Path(out_dir)
        self.node_id = node_id
        self.fast_interval_s = fast_interval_s
        self.slow_interval_s = slow_interval_s
        self.duration_s = duration_s
        self.gpu_mode = gpu_mode
        self.nvidia_smi = list(nvidia_smi)
        self.stall_timeout_s = stall_timeout_s
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self.gpu_stale_after_s = gpu_stale_after_s
        self.fsync_interval_s = fsync_interval_s
        self.rotate_seconds = rotate_seconds
        self.cmd_timeout_s = cmd_timeout_s
        self.disks = dict(disks or {})
        self.reserves = dict(reserves or {})
        self.containers = containers
        self.compress_closed = compress_closed
        self.thresholds = dict(thresholds or {})
        self.nice = nice
        self.max_state_change_events = max_state_change_events


class _InstanceLock:
    """One sentinel per output directory (advisory; flock on POSIX, msvcrt on Windows)."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def acquire(self):
        self.handle = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"another sentinel holds {self.path.name}") from None

    def release(self):
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.handle.close()
        self.handle = None


class Sentinel:
    def __init__(self, config, readers=None, clock=time.monotonic, wall=time.time, popen=subprocess.Popen):
        self.config = config
        self.readers = readers or tv.Readers()
        self.clock = clock
        self.wall = wall
        self.popen = popen
        self._stop = threading.Event()
        self._slow_lock = threading.Lock()
        self._slow_latest = None
        self._compress_queue = queue.Queue()
        self.stop_reason = None
        self.fast_samples = 0
        self.slow_samples = 0
        self.fast_errors = 0
        self.slow_errors = 0
        self.max_fast_gap_s = 0.0
        self.fast_gaps_over = 0
        self.dt_values = []
        self.pkg_paths = []
        self._pkg_discovered = None
        self._threshold_state = {}

    def stop(self, reason="signal"):
        if self.stop_reason is None:
            self.stop_reason = reason
        self._stop.set()

    # -------------------------------------------------------------- lifecycle
    def run(self):
        cfg = self.config
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(cfg.out_dir, 0o700)
        except OSError:
            pass
        lock = _InstanceLock(cfg.out_dir / "sentinel.lock")
        lock.acquire()
        try:
            return self._run_locked()
        finally:
            lock.release()

    def _run_locked(self):
        cfg = self.config
        self.boot_id, boot_reason = tv.read_boot_id(self.readers)
        self.boot_id = self.boot_id or "unavailable"
        previous = read_json(cfg.out_dir / "state.json")
        boot_dir = cfg.out_dir / self.boot_id
        boot_dir.mkdir(exist_ok=True)
        on_closed = self._compress_queue.put if cfg.compress_closed else None
        self.fast_writer = SegmentWriter(boot_dir, "fast", self.wall, cfg.fsync_interval_s, cfg.rotate_seconds,
                                         on_closed=on_closed)
        self.slow_writer = SegmentWriter(boot_dir, "slow", self.wall, cfg.fsync_interval_s, cfg.rotate_seconds,
                                         on_closed=on_closed)
        events_writer = SegmentWriter(boot_dir, "events", self.wall, 0.0, cfg.rotate_seconds, on_closed=on_closed)
        if previous and previous.get("node_id") != cfg.node_id:
            previous = None
        same_boot = bool(previous) and previous.get("boot_id") == self.boot_id
        last_fast = last_record(boot_dir, "fast") if same_boot else None
        last_slow = last_record(boot_dir, "slow") if same_boot else None
        last_event = last_record(boot_dir, "events") if same_boot else None
        self.fast_seq = (last_fast or {}).get("seq", 0)
        self.slow_seq = (last_slow or {}).get("seq", 0)
        self.events = EventLog(events_writer, cfg.node_id, self.boot_id, self.clock, self.wall,
                               (last_event or {}).get("seq", 0))
        reboot = tv.detect_reboot(previous, self.boot_id, self.wall())
        if reboot:
            reboot["prev_last_sample"] = self._sample_digest(last_record(cfg.out_dir / reboot["prev_boot_id"], "fast"))
            if reboot["gap_s"] is None:
                reboot["gap_s"] = 0.0
        # Leftover segments of earlier runs are compressed only AFTER their last records were read
        # (seq continuation, reboot digest): compressing first would race with those reads.
        compressor = threading.Thread(target=self._compressor, name="sentinel-gzip", daemon=True)
        compressor.start()
        if cfg.compress_closed:
            for directory in cfg.out_dir.iterdir():
                if directory.is_dir():
                    for leftover in directory.glob("*.jsonl"):
                        self._compress_queue.put(leftover)
        started_mono, started_cpu, started_children = self.clock(), time.process_time(), self._children_cpu()
        self.events.emit("collector_start", pid=os.getpid(), version=1, fast_interval_s=cfg.fast_interval_s,
                         slow_interval_s=cfg.slow_interval_s, gpu_mode=cfg.gpu_mode,
                         fsync_interval_s=cfg.fsync_interval_s, boot_id_reason=boot_reason or "ok",
                         prev_run_same_boot=same_boot)
        if reboot:
            self.events.emit("reboot_detected", **reboot)
        elif same_boot and isinstance(previous.get("last_sample_mono_s"), (int, float)):
            gap = started_mono - previous["last_sample_mono_s"]
            if gap > 2 * cfg.fast_interval_s:
                self.events.emit("sample_gap", tier="fast", gap_s=round(gap, 3), across_restart=True,
                                 prev_last_sample_utc=previous.get("last_sample_utc"))
        if cfg.nice and hasattr(os, "nice"):
            try:
                os.nice(cfg.nice)
            except OSError:
                pass
        self.stream = GpuStream(cfg.nvidia_smi, self.events.emit, self.clock, cfg.gpu_mode, cfg.fast_interval_s,
                                cfg.stall_timeout_s, cfg.backoff_initial_s, cfg.backoff_max_s, popen=self.popen,
                                run_cmd=self.readers.run_cmd, cmd_timeout_s=cfg.cmd_timeout_s)
        self.collector = tv.SlowCollector(self.readers, cfg.nvidia_smi, cfg.disks, cfg.reserves, cfg.cmd_timeout_s,
                                          cfg.containers, gpu_enabled=cfg.gpu_mode != "off")
        self._discover_packages(started_mono)
        self.stream.start()
        slow = threading.Thread(target=self._slow_loop, name="sentinel-slow", daemon=True)
        slow.start()
        try:
            self._fast_loop(started_mono)
        except BaseException:
            self.stop("error")
            raise
        finally:
            self._stop.set()
            self.stream.stop()
            slow.join(cfg.cmd_timeout_s * 4 + cfg.slow_interval_s)
            wall_s = self.clock() - started_mono
            cpu_s = time.process_time() - started_cpu
            children_s = self._children_cpu() - started_children
            self.summary = {
                "reason": self.stop_reason or "duration", "fast_samples": self.fast_samples,
                "slow_samples": self.slow_samples, "fast_errors": self.fast_errors, "slow_errors": self.slow_errors,
                "max_fast_gap_s": round(self.max_fast_gap_s, 3), "fast_gaps_over_2x": self.fast_gaps_over,
                "gpu_stream_restarts": self.stream.restarts, "gpu_stream_lines": self.stream.lines,
                "gpu_stream_bad_lines": self.stream.bad_lines, "wall_s": round(wall_s, 3),
                "cpu_process_s": round(cpu_s, 3), "cpu_children_s": round(children_s, 3),
                "cpu_pct": round(100.0 * (cpu_s + children_s) / wall_s, 3) if wall_s > 0 else 0.0,
                "fsyncs_fast": self.fast_writer.fsync_count,
            }
            self.events.emit("collector_stop", **self.summary)
            self.fast_writer.close()
            self.slow_writer.close()
            events_writer.close()
            self._write_state(final=True)
            self._compress_queue.put(None)
            compressor.join(30.0)
        return self.summary

    @staticmethod
    def _children_cpu():
        times = os.times()
        return times.children_user + times.children_system

    def _compressor(self):
        while True:
            item = self._compress_queue.get()
            if item is None:
                return
            try:
                if Path(item).exists() and Path(item) not in self._open_segments():
                    compress_segment(item)
            except OSError:
                continue

    def _open_segments(self):
        paths = set()
        for name in ("fast_writer", "slow_writer"):
            writer = getattr(self, name, None)
            if writer is not None and writer.path is not None:
                paths.add(writer.path)
        events = getattr(self, "events", None)
        if events is not None and events.writer.path is not None:
            paths.add(events.writer.path)
        return paths

    @staticmethod
    def _sample_digest(record):
        if not record:
            return None
        return {"utc": record.get("utc"), "seq": record.get("seq"),
                "gpus": [{key: gpu.get(key) for key in ("index", "temp_c", "power_w", "util_pct")}
                         for gpu in record.get("gpus") or []],
                "cpu_pkg": record.get("cpu_pkg")}

    def _write_state(self, final=False):
        record = self.fast_writer.last_record if final else self.fast_writer.last_synced_record
        if record is None:
            return
        state = {"schema": fleet_schema.SCHEMA_SENTINEL_STATE, "node_id": self.config.node_id,
                 "boot_id": self.boot_id, "last_fast_seq": record["seq"], "last_sample_utc": record["utc"],
                 "last_sample_mono_s": record["mono_s"], "updated_utc": tv.utc_label(self.wall()), "final": final}
        try:
            atomic_write_json(self.config.out_dir / "state.json", state, fsync=True)
        except OSError:
            pass

    def _discover_packages(self, now):
        self._pkg_discovered = now
        try:
            self.pkg_paths = tv.package_temp_paths(self.readers)
        except Exception:  # noqa: BLE001
            self.pkg_paths = []

    # -------------------------------------------------------------- fast tier
    def _fast_loop(self, started):
        cfg = self.config
        interval = cfg.fast_interval_s
        deadline = None if cfg.duration_s is None else started + cfg.duration_s
        next_t = started
        self._last_fast = None
        while not self._stop.is_set():
            now = self.clock()
            if deadline is not None and now >= deadline:
                self.stop("duration")
                break
            if now < next_t:
                self._stop.wait(next_t - now)
                continue
            try:
                self._fast_tick(now)
            except Exception:  # noqa: BLE001 - a failing tick never stops the collector
                self.fast_errors += 1
            next_t += interval
            after = self.clock()
            if next_t <= after:
                next_t += (int((after - next_t) // interval) + 1) * interval

    def _fast_tick(self, now):
        cfg = self.config
        self.fast_seq += 1
        reasons = {}
        dt = None if self._last_fast is None else now - self._last_fast
        if dt is None:
            reasons["dt_s"] = "first_sample"
        else:
            self.dt_values.append(dt)
            self.max_fast_gap_s = max(self.max_fast_gap_s, dt)
            if dt > 2 * cfg.fast_interval_s:
                self.fast_gaps_over += 1
                self.events.emit("sample_gap", tier="fast", gap_s=round(dt, 3), across_restart=False,
                                 prev_last_sample_utc=None if self.fast_writer.last_record is None
                                 else self.fast_writer.last_record["utc"])
        self._last_fast = now
        if cfg.gpu_mode == "off":
            gpus, gpu_reasons = None, {"gpus": "disabled_by_config"}
        else:
            gpus, gpu_reasons = tv.fast_gpu_entries(self.stream.snapshot(), now, cfg.gpu_stale_after_s)
        reasons.update(gpu_reasons)
        if not self.pkg_paths and now - self._pkg_discovered > 60.0:
            self._discover_packages(now)
        packages, pkg_reasons = tv.read_package_temps(self.readers, self.pkg_paths)
        reasons.update(pkg_reasons)
        if pkg_reasons and packages is not None and now - self._pkg_discovered > 60.0:
            self._discover_packages(now)
        record = tv.envelope("fast", cfg.node_id, self.boot_id, self.fast_seq, now, tv.utc_label(self.wall()))
        record.update({"dt_s": None if dt is None else round(dt, 3), "gpus": gpus, "cpu_pkg": packages,
                       "null_reasons": reasons,
                       "src": {"gpu": "nvidia-smi-" + cfg.gpu_mode,
                               "cpu_pkg": "hwmon:" + (self.pkg_paths[0][2] if self.pkg_paths else "none")}})
        synced = self.fast_writer.write(record, now)
        self.fast_samples += 1
        if synced:
            self._write_state()
        try:
            atomic_write_json(cfg.out_dir / "latest.json", self._latest_document(record))
        except OSError:
            self.fast_errors += 1
        self._state_changes("fast", reasons)
        self._observe_thresholds(record)

    def _latest_document(self, fast):
        reasons = {}
        uuids = self.stream.uuids() if self.config.gpu_mode != "off" else {}
        with self._slow_lock:
            slow = self._slow_latest
        apps, apps_reason = {}, {}
        slow_reason = "first_sample" if slow is None else None
        if slow is not None:
            if slow.get("gpus") is None:
                slow_reason = slow["null_reasons"].get("gpus", "not_exposed")
            for position, gpu in enumerate(slow.get("gpus") or []):
                apps[gpu["index"]] = gpu.get("compute_apps")
                apps_reason[gpu["index"]] = slow["null_reasons"].get(f"gpus[{position}].compute_apps")
        gpus = None
        if fast["gpus"] is None:
            reasons["gpus"] = fast["null_reasons"].get("gpus", "not_exposed")
        else:
            gpus = []
            for position, gpu in enumerate(fast["gpus"]):
                entry = {"index": gpu["index"], "uuid": uuids.get(gpu["index"]), "temp_c": gpu["temp_c"],
                         "util_pct": gpu["util_pct"], "compute_apps": apps.get(gpu["index"])}
                for key in ("temp_c", "util_pct"):
                    if entry[key] is None:
                        reasons[f"gpus[{position}].{key}"] = (fast["null_reasons"].get(f"gpus[{position}].{key}")
                                                              or fast["null_reasons"].get(f"gpus[{position}]")
                                                              or "not_exposed")
                if entry["uuid"] is None:
                    reasons[f"gpus[{position}].uuid"] = "not_exposed"
                if entry["compute_apps"] is None:
                    reasons[f"gpus[{position}].compute_apps"] = (slow_reason or apps_reason.get(gpu["index"])
                                                                 or "not_exposed")
                gpus.append(entry)
        temps = [item["temp_c"] for item in fast["cpu_pkg"] or [] if item["temp_c"] is not None]
        cpu_max = max(temps) if temps else None
        if cpu_max is None:
            reasons["cpu_pkg_max_temp_c"] = fast["null_reasons"].get("cpu_pkg", "not_exposed")
        disks = None
        if slow is None:
            reasons["disks"] = "first_sample"
        elif slow.get("disks") is None:
            reasons["disks"] = slow["null_reasons"].get("disks", "not_exposed")
        else:
            disks = [{"key": disk["key"], "free_bytes": disk["free_bytes"]} for disk in slow["disks"]]
            for position, disk in enumerate(disks):
                if disk["free_bytes"] is None:
                    reasons[f"disks[{position}].free_bytes"] = "not_exposed"
        return {"schema": fleet_schema.SCHEMA_SENTINEL_LATEST, "node_id": fast["node_id"], "boot_id": fast["boot_id"],
                "seq": fast["seq"], "utc": fast["utc"], "mono_s": fast["mono_s"], "gpus": gpus,
                "cpu_pkg_max_temp_c": cpu_max, "disks": disks, "null_reasons": reasons}

    def _state_changes(self, tier, reasons):
        attribute = f"_{tier}_reasons_prev"
        previous = getattr(self, attribute, None)
        current = {key: value for key, value in reasons.items() if value != "first_sample"}
        if previous is None:
            # The first record of a tier is the baseline; it already lists every null reason.
            setattr(self, attribute, current)
            return
        emitted = 0
        for key in sorted(set(previous) | set(current)):
            before, after = previous.get(key, "ok"), current.get(key, "ok")
            if before != after and emitted < self.config.max_state_change_events:
                self.events.emit("sensor_state_change", tier=tier, field=key, **{"from": before, "to": after})
                emitted += 1
        setattr(self, attribute, current)

    def _observe_thresholds(self, record):
        values = {"gpu.temp_c": [(f"gpu{gpu['index']}", gpu["temp_c"]) for gpu in record["gpus"] or []],
                  "cpu_pkg.temp_c": [(f"pkg{item['package']}", item["temp_c"]) for item in record["cpu_pkg"] or []]}
        for sensor, limit in self.config.thresholds.items():
            for name, value in values.get(sensor, []):
                if value is None:
                    continue
                above = value >= limit
                key = (sensor, name)
                previous = self._threshold_state.get(key)
                self._threshold_state[key] = above
                if (above and previous is not True) or (not above and previous is True):
                    self.events.emit("threshold_observed", sensor=sensor, device=name, value=value,
                                     threshold=limit, direction="up" if above else "down", action="none")

    # -------------------------------------------------------------- slow tier
    def _slow_loop(self):
        cfg = self.config
        interval = cfg.slow_interval_s
        next_t = self.clock()
        last = None
        while not self._stop.is_set():
            now = self.clock()
            if now < next_t:
                self._stop.wait(next_t - now)
                continue
            try:
                self._slow_tick(now, last)
            except Exception:  # noqa: BLE001
                self.slow_errors += 1
            last = now
            next_t += interval
            after = self.clock()
            if next_t <= after:
                next_t = after

    def _slow_tick(self, now, last):
        cfg = self.config
        uuids = self.stream.uuids() if cfg.gpu_mode != "off" else {}
        sections, reasons, events, src = self.collector.collect(now, uuids)
        if last is not None and now - last > 2 * cfg.slow_interval_s:
            self.events.emit("sample_gap", tier="slow", gap_s=round(now - last, 3), across_restart=False,
                             prev_last_sample_utc=None if self.slow_writer.last_record is None
                             else self.slow_writer.last_record["utc"])
        self.slow_seq += 1
        record = tv.envelope("slow", cfg.node_id, self.boot_id, self.slow_seq, now, tv.utc_label(self.wall()))
        record.update(sections)
        record["null_reasons"] = reasons
        record["src"] = src
        self.slow_writer.write(record, now)
        self.slow_samples += 1
        with self._slow_lock:
            self._slow_latest = record
        for event in events:
            name = event.pop("event")
            self.events.emit(name, **event)
        self._state_changes("slow", reasons)


# ------------------------------------------------------------------------ report

def summarize_run(out_dir, validate=False):
    """Coverage report over every boot directory: fast-tier gaps, events, stop stats."""
    out_dir = Path(out_dir)
    report = {"boots": [], "validation_errors": [] if validate else None}
    for boot_dir in sorted(item for item in out_dir.iterdir() if item.is_dir()) if out_dir.is_dir() else []:
        fast = [record for path in segment_files(boot_dir, "fast") for record in iter_segment_lines(path)]
        slow_count = sum(1 for path in segment_files(boot_dir, "slow") for _ in iter_segment_lines(path))
        events = [record for path in segment_files(boot_dir, "events") for record in iter_segment_lines(path)]
        if validate:
            for record in fast[:]:
                try:
                    fleet_schema.validate_telemetry_record(record)
                except fleet_schema.SchemaError as exc:
                    report["validation_errors"].append(str(exc))
        gaps = [later["mono_s"] - earlier["mono_s"] for earlier, later in zip(fast, fast[1:])
                if later["mono_s"] >= earlier["mono_s"]]
        ordered = sorted(gaps)
        gpu_null = sum(1 for record in fast if record.get("gpus") is None
                       or any(gpu.get("temp_c") is None for gpu in record["gpus"]))
        counts = {}
        for event in events:
            counts[event.get("event")] = counts.get(event.get("event"), 0) + 1
        stops = [event for event in events if event.get("event") == "collector_stop"]
        report["boots"].append({
            "boot_id": boot_dir.name, "fast_samples": len(fast), "slow_samples": slow_count,
            "first_utc": fast[0]["utc"] if fast else None, "last_utc": fast[-1]["utc"] if fast else None,
            "max_gap_s": round(max(gaps), 3) if gaps else None,
            "p50_gap_s": round(ordered[len(ordered) // 2], 3) if ordered else None,
            "p99_gap_s": round(ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], 3) if ordered else None,
            "gaps_over_1s": sum(1 for gap in gaps if gap > 1.0),
            "fast_samples_with_gpu_null": gpu_null, "events": counts,
            "collector_stop": stops[-1] if stops else None,
        })
    return report


# ---------------------------------------------------------------------------- CLI

def _key_values(items, kind):
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{kind} must be KEY=VALUE")
        key, value = item.split("=", 1)
        result[key.strip()] = value
    return result


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m augment_atlas.sentinel",
                                     description="Observation-only node telemetry collector (fleet v1).")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="collect until --duration or SIGTERM/SIGINT")
    run.add_argument("--dir", required=True, help="private output directory (one sentinel per directory)")
    run.add_argument("--node-id", required=True, help="public node alias (never the hostname)")
    run.add_argument("--duration", type=float, help="seconds; default: until signal")
    run.add_argument("--fast-interval", type=float, default=0.5)
    run.add_argument("--slow-interval", type=float, default=10.0)
    run.add_argument("--gpu-mode", choices=("stream", "poll", "off"), default="stream")
    run.add_argument("--nvidia-smi", nargs="+", default=["nvidia-smi"], metavar="ARG",
                     help="nvidia-smi command (tests pass: python fake-nvidia-smi.py)")
    run.add_argument("--stall-timeout", type=float, default=3.0)
    run.add_argument("--fsync-interval", type=float, default=5.0)
    run.add_argument("--rotate-seconds", type=float, default=3600.0)
    run.add_argument("--cmd-timeout", type=float, default=4.0)
    run.add_argument("--disk", action="append", metavar="KEY=PATH", help="statvfs target (only KEY is recorded)")
    run.add_argument("--reserve", action="append", metavar="KEY=BYTES")
    run.add_argument("--observe-threshold", action="append", metavar="SENSOR=C",
                     help="emit threshold_observed events (gpu.temp_c, cpu_pkg.temp_c); never acts")
    run.add_argument("--containers", action="store_true", help="also list atlas-* containers (docker ps, read-only)")
    run.add_argument("--no-compress", action="store_true")
    run.add_argument("--nice", type=int, default=10)
    run.add_argument("--replay", help="read procfs/sysfs from a captured fixture tree (testing)")
    run.add_argument("--allow-root", action="store_true", help="the sentinel needs no privilege; refuse root by default")
    report = commands.add_parser("report", help="coverage/gap report of a sentinel directory")
    report.add_argument("--dir", required=True)
    report.add_argument("--validate", action="store_true", help="validate every fast record against schema v1")
    latest = commands.add_parser("latest", help="print latest.json status (age from monotonic clock)")
    latest.add_argument("--dir", required=True)
    latest.add_argument("--max-age", type=float, default=5.0)
    probe = commands.add_parser("probe", help="one slow sample printed to stdout (read-only)")
    probe.add_argument("--node-id", required=True)
    probe.add_argument("--replay")
    probe.add_argument("--nvidia-smi", nargs="+", default=["nvidia-smi"], metavar="ARG")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "report":
        print(json.dumps(summarize_run(args.dir, args.validate), indent=2, sort_keys=True))
        return 0
    if args.command == "latest":
        status = read_latest(Path(args.dir) / "latest.json", max_age_s=args.max_age)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if status["status"] == "ok" else 1
    readers = tv.ReplayReaders(args.replay, run_cmd=tv.run_command) if args.replay else tv.Readers()
    if args.command == "probe":
        collector = tv.SlowCollector(readers, args.nvidia_smi)
        sections, reasons, _events, src = collector.collect(time.monotonic())
        boot_id = tv.read_boot_id(readers)[0] or "unavailable"
        record = tv.envelope("slow", args.node_id, boot_id, 1, time.monotonic(), tv.utc_label())
        record.update(sections)
        record.update(null_reasons=reasons, src=src)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if hasattr(os, "geteuid") and os.geteuid() == 0 and not args.allow_root:
        print("refusing to run as root: the sentinel only needs unprivileged reads", file=sys.stderr)
        return 2
    try:
        thresholds = {key: float(value) for key, value in _key_values(args.observe_threshold, "threshold").items()}
        reserves = {key: int(value) for key, value in _key_values(args.reserve, "reserve").items()}
        config = SentinelConfig(args.dir, args.node_id, args.fast_interval, args.slow_interval, args.duration,
                                args.gpu_mode, args.nvidia_smi, args.stall_timeout, fsync_interval_s=args.fsync_interval,
                                rotate_seconds=args.rotate_seconds, cmd_timeout_s=args.cmd_timeout,
                                disks=_key_values(args.disk, "disk"), reserves=reserves, containers=args.containers,
                                compress_closed=not args.no_compress, thresholds=thresholds, nice=args.nice)
    except (ValueError, fleet_schema.SchemaError) as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2
    sentinel = Sentinel(config, readers)
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), lambda *_: sentinel.stop("signal"))
            except (ValueError, OSError):
                pass
    try:
        summary = sentinel.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
