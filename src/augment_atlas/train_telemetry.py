"""Per-job training telemetry inside the attempt output (``atlas.train_telemetry.v1``).

Additive and best effort. ``telemetry/train.jsonl`` and ``telemetry/summary.json``
live in a subdirectory that is NOT one of the validated artifacts: the runtime
validators read only the names listed in ``completed.json``. Nothing here reads
or writes the training RNG streams, the batch order, the loss, checkpoints or
``metrics.json``; a failing sensor yields ``null`` plus a reason, and any writer
exception disables telemetry after one ``error`` line. Training never fails
because of telemetry.

Records (one JSON object per line, ``schema``/``type``/``session_id``/``seq``/
``utc``/``mono_s`` on every line, ``seq`` strictly increasing per session):

* ``session_start`` / ``session_end`` - one attempt process (a resume is a new
  session appended to the same file; ``mono_s`` is comparable only within a
  session).
* ``batch`` - every ``BATCH_EVERY`` batches and on the last batch of an epoch.
* ``epoch`` - wall time, throughput, training loss, learning rate and only the
  per-epoch selection metrics the trainer already computed. Evaluated-partition
  metrics (test / E_f / final D_select) are never written here.
* ``machine`` - every ``MACHINE_INTERVAL_S`` from a daemon thread: GPU via
  ``nvidia-smi --query-gpu``, process CPU/RSS, host CPU and load, cgroup memory.
* ``error`` - the writer failed; telemetry is off for the rest of the session.

Internal constants (deliberately not part of the validated config schema).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

SCHEMA = "atlas.train_telemetry.v1"
SUMMARY_SCHEMA = "atlas.train_telemetry_summary.v1"
ENABLED = True
BATCH_EVERY = 10
MACHINE_INTERVAL_S = 30.0
FSYNC_EVERY_RECORDS = 32
FSYNC_EVERY_S = 5.0
NVIDIA_SMI_TIMEOUT_S = 4.0
CLOSE_JOIN_S = 5.0
# Power samples further apart than this are not integrated (coverage drops).
ENERGY_MAX_GAP_FACTOR = 2.5
DIRECTORY = "telemetry"
JSONL_NAME = "train.jsonl"
SUMMARY_NAME = "summary.json"

# Fleet protocol 5.2 enum plus trainer-specific codes (documented in docs/telemetry.md).
REASONS = frozenset({
    "not_exposed", "permission_denied", "tool_absent", "tool_timeout", "parse_error",
    "not_supported_by_device", "disabled_by_config", "stale", "collector_error", "first_sample",
    "device_not_cuda", "no_optimizer_step", "non_finite", "no_samples", "not_evaluated_per_epoch",
    "zero_duration", "unlimited",
})

GPU_FIELDS = ("index", "uuid", "name", "temperature.gpu", "utilization.gpu", "memory.used", "memory.total",
              "power.draw", "power.limit", "clocks.sm", "pstate", "fan.speed")
_GPU_KEYS = {"index": "index", "uuid": "uuid", "name": "name", "temperature.gpu": "temp_c",
             "utilization.gpu": "util_pct", "memory.used": "mem_used_mib", "memory.total": "mem_total_mib",
             "power.draw": "power_w", "power.limit": "power_limit_w", "clocks.sm": "sm_mhz",
             "pstate": "pstate", "fan.speed": "fan_pct"}
_GPU_TEXT = {"uuid", "name", "pstate"}
_UNSUPPORTED = {"[n/a]", "n/a", "[not supported]", "not supported", "[unknown error]", ""}
# Keys an epoch record may copy from the trainer journal row. Never test/E_f.
_EPOCH_KEYS = ("train_loss", "learning_rate", "samples_processed", "optimizer_updates")
_SELECTION_KEYS = ("loss", "accuracy", "samples")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    """Float/int if finite, else None. Bools are not numbers here."""
    if type(value) is bool or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


class JsonlWriter:
    """Thread-safe append-only JSONL writer with batched fsync.

    A line is flushed to the OS on every write; ``fsync`` happens after
    ``fsync_every`` records, ``fsync_seconds`` or on ``sync=True``. A torn last
    line from a crashed earlier session is terminated before appending, so the
    file stays line-parseable (the torn line itself is skipped by readers).
    """

    def __init__(self, path, *, session_id, fsync_every=FSYNC_EVERY_RECORDS, fsync_seconds=FSYNC_EVERY_S,
                 clock=time.monotonic):
        self.path = Path(path)
        self.session_id = session_id
        self.fsync_every, self.fsync_seconds, self.clock = fsync_every, fsync_seconds, clock
        self.seq = 0
        self.fsyncs = 0
        self.enabled = True
        self.error = None
        self._pending = 0
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("ab")
        try:
            size = self.path.stat().st_size
            if size:
                with self.path.open("rb") as existing:
                    existing.seek(size - 1)
                    if existing.read(1) != b"\n":
                        self._stream.write(b"\n")
        except Exception:
            self._stream.close()
            raise
        self._last_sync = clock()

    def write(self, kind, payload=None, *, sync=False) -> bool:
        with self._lock:
            if not self.enabled:
                return False
            try:
                record = {"schema": SCHEMA, "type": kind, "session_id": self.session_id, "seq": self.seq,
                          "utc": utc_now(), "mono_s": self.clock()}
                record.update(payload or {})
                line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
                self._stream.write(line.encode("utf-8"))
                self._stream.flush()
                self.seq += 1
                self._pending += 1
                if sync or self._pending >= self.fsync_every or self.clock() - self._last_sync >= self.fsync_seconds:
                    self._sync()
                return True
            except Exception as error:  # telemetry never propagates into training
                self._fail(error)
                return False

    def fail(self, error):
        """Disable the writer after one ``error`` line (thread-safe)."""
        with self._lock:
            if self.enabled:
                self._fail(error)

    def _sync(self):
        os.fsync(self._stream.fileno())
        self._pending = 0
        self._last_sync = self.clock()
        self.fsyncs += 1

    def _fail(self, error):
        self.enabled = False
        self.error = f"{type(error).__name__}: {error}"[:300]
        try:
            line = json.dumps({"schema": SCHEMA, "type": "error", "session_id": self.session_id, "seq": self.seq,
                               "utc": utc_now(), "mono_s": self.clock(), "error_type": type(error).__name__,
                               "message": str(error)[:300], "telemetry": "disabled"}, allow_nan=False,
                              separators=(",", ":"))
            self._stream.write(b"\n" + line.encode("utf-8") + b"\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass

    def close(self):
        with self._lock:
            if not self.enabled:
                return
            self.enabled = False
            try:
                self._stream.flush()
                self._sync()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sensors (each isolated: missing => None + reason, never an exception)
# ---------------------------------------------------------------------------

def _read_text(path):
    try:
        return Path(path).read_text(encoding="ascii", errors="replace")
    except FileNotFoundError:
        return None
    except PermissionError:
        return PermissionError
    except OSError:
        return None


def _gpu_value(field, raw):
    text = raw.strip()
    if text.lower() in _UNSUPPORTED:
        return None, "not_supported_by_device"
    if field in _GPU_TEXT:
        return text, None
    try:
        value = float(text)
    except ValueError:
        return None, "parse_error"
    if not math.isfinite(value):
        return None, "parse_error"
    return (int(value) if field == "index" else value), None


def query_gpus(runner=subprocess.run, which=shutil.which):
    """Return (gpus|None, null_reasons). ``gpus`` lists every visible device."""
    if which("nvidia-smi") is None:
        return None, {"gpus": "tool_absent"}
    try:
        result = runner(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, {"gpus": "tool_timeout"}
    except PermissionError:
        return None, {"gpus": "permission_denied"}
    except OSError:
        return None, {"gpus": "tool_absent"}
    if result.returncode != 0:
        return None, {"gpus": "collector_error"}
    gpus, reasons = [], {}
    try:
        for position, values in enumerate(csv.reader(line for line in result.stdout.splitlines() if line.strip())):
            if len(values) != len(GPU_FIELDS):
                return None, {"gpus": "parse_error"}
            gpu = {}
            for field, raw in zip(GPU_FIELDS, values):
                value, reason = _gpu_value(field, raw)
                gpu[_GPU_KEYS[field]] = value
                if reason:
                    reasons[f"gpus[{position}].{_GPU_KEYS[field]}"] = reason
            gpus.append(gpu)
    except csv.Error:
        return None, {"gpus": "parse_error"}
    if not gpus:
        return None, {"gpus": "not_exposed"}
    return gpus, reasons


def _status_bytes(text, key):
    for line in text.splitlines():
        if line.startswith(key + ":"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


class MachineSampler:
    """Process, host and GPU readings. Holds only its own previous counters."""

    def __init__(self, gpu_query=query_gpus, clock=time.monotonic, process_clock=time.process_time):
        self.gpu_query, self.clock, self.process_clock = gpu_query, clock, process_clock
        self._previous_process = None
        self._previous_host = None

    def sample(self):
        start = self.clock()
        reasons = {}
        row = {}
        cpu_now = self.process_clock()
        if self._previous_process is None:
            row["process_cpu_pct"] = None
            reasons["process_cpu_pct"] = "first_sample"
        else:
            wall = start - self._previous_process[0]
            row["process_cpu_pct"] = finite(100.0 * (cpu_now - self._previous_process[1]) / wall) if wall > 0 else None
            if row["process_cpu_pct"] is None:
                reasons["process_cpu_pct"] = "stale"
        self._previous_process = (start, cpu_now)
        row["process_cpu_scope"] = "trainer_process_all_threads_excluding_dataloader_workers"
        try:
            times = os.times()
            row["children_cpu_s"] = finite(times.children_user + times.children_system)
        except (OSError, AttributeError):
            row["children_cpu_s"] = None
            reasons["children_cpu_s"] = "not_exposed"
        status = _read_text("/proc/self/status")
        if isinstance(status, str):
            row["rss_bytes"] = _status_bytes(status, "VmRSS")
            row["rss_peak_bytes"] = _status_bytes(status, "VmHWM")
            for key in ("rss_bytes", "rss_peak_bytes"):
                if row[key] is None:
                    reasons[key] = "parse_error"
        else:
            row["rss_bytes"] = row["rss_peak_bytes"] = None
            reasons["rss_bytes"] = reasons["rss_peak_bytes"] = (
                "permission_denied" if status is PermissionError else "not_exposed")
        try:
            row["load_1_5_15"] = [round(value, 3) for value in os.getloadavg()]
        except (OSError, AttributeError):
            row["load_1_5_15"] = None
            reasons["load_1_5_15"] = "not_exposed"
        row["host_cpu_util_pct"] = None
        stat = _read_text("/proc/stat")
        if isinstance(stat, str) and stat.startswith("cpu "):
            try:
                counts = [int(value) for value in stat.splitlines()[0].split()[1:9]]
                current = (sum(counts), counts[3] + counts[4])
                if self._previous_host is None:
                    reasons["host_cpu_util_pct"] = "first_sample"
                elif current[0] > self._previous_host[0]:
                    total = current[0] - self._previous_host[0]
                    row["host_cpu_util_pct"] = finite(100.0 * (1 - (current[1] - self._previous_host[1]) / total))
                else:
                    reasons["host_cpu_util_pct"] = "stale"
                self._previous_host = current
            except (ValueError, IndexError):
                reasons["host_cpu_util_pct"] = "parse_error"
        else:
            reasons["host_cpu_util_pct"] = "permission_denied" if stat is PermissionError else "not_exposed"
        row["cgroup_memory_bytes"], row["cgroup_memory_limit_bytes"] = None, None
        current = _read_text("/sys/fs/cgroup/memory.current")
        limit = _read_text("/sys/fs/cgroup/memory.max")
        if not isinstance(current, str):
            current = _read_text("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            limit = _read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if isinstance(current, str) and current.strip().isdigit():
            row["cgroup_memory_bytes"] = int(current.strip())
        else:
            reasons["cgroup_memory_bytes"] = "not_exposed"
        if isinstance(limit, str) and limit.strip().isdigit():
            row["cgroup_memory_limit_bytes"] = int(limit.strip())
        elif isinstance(limit, str) and limit.strip() == "max":
            reasons["cgroup_memory_limit_bytes"] = "unlimited"
        else:
            reasons["cgroup_memory_limit_bytes"] = "not_exposed"
        try:
            gpus, gpu_reasons = self.gpu_query()
        except Exception:
            gpus, gpu_reasons = None, {"gpus": "collector_error"}
        row["gpus"] = gpus
        reasons.update(gpu_reasons)
        row["collection_s"] = self.clock() - start
        row["null_reasons"] = reasons
        return row


# ---------------------------------------------------------------------------
# Trainer facade
# ---------------------------------------------------------------------------

class NullTelemetry:
    """Disabled telemetry: every hook is a no-op."""
    enabled = False

    def epoch_start(self, *args, **kwargs):
        pass

    def batch(self, *args, **kwargs):
        pass

    def train_done(self, *args, **kwargs):
        pass

    def epoch_end(self, *args, **kwargs):
        pass

    def outcome(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class TrainTelemetry:
    enabled = True

    def __init__(self, output, *, torch, device, context, batch_every=BATCH_EVERY,
                 machine_interval_s=MACHINE_INTERVAL_S, sampler=None, clock=None,
                 grad_clip_max_norm=None):
        # Durations use perf_counter (Windows monotonic ticks are ~16 ms);
        # record timestamps (mono_s) and energy integration use monotonic.
        self.directory = Path(output) / DIRECTORY
        self.torch, self.device, self.clock = torch, device, clock or time.perf_counter
        self.batch_every, self.machine_interval_s = batch_every, machine_interval_s
        self.grad_clip_max_norm = grad_clip_max_norm
        self.cuda = getattr(device, "type", str(device)) == "cuda"
        self.writer = JsonlWriter(self.directory / JSONL_NAME, session_id=os.urandom(8).hex(),
                                  clock=clock or time.monotonic)
        self.sampler = sampler or MachineSampler(clock=clock or time.monotonic)
        self._outcome = None
        self._closed = False
        self._epoch = None
        self._stop = threading.Event()
        self.writer.write("session_start", {"context": context, "batch_every": batch_every,
                                            "machine_interval_s": machine_interval_s,
                                            "cuda_memory": self.cuda}, sync=True)
        self._thread = None
        if machine_interval_s and machine_interval_s > 0:
            self._thread = threading.Thread(target=self._machine_loop, name="atlas-train-telemetry", daemon=True)
            self._thread.start()

    # -- machine thread -----------------------------------------------------
    def _machine_loop(self):
        while self.writer.enabled:
            self._machine_sample()
            if self._stop.wait(self.machine_interval_s):
                return

    def _machine_sample(self):
        try:
            try:
                row = self.sampler.sample()
            except Exception:
                row = {"gpus": None, "null_reasons": {"machine": "collector_error"}}
            state = self._epoch  # read once: the trainer thread may reset it
            row["epoch"] = state["epoch"] if state is not None else None
            self.writer.write("machine", row)
        except Exception as error:
            self.writer.fail(error)

    # -- trainer hooks ------------------------------------------------------
    def _memory(self):
        if not self.cuda:
            return {"vram_allocated_bytes": None, "vram_reserved_bytes": None, "vram_max_allocated_bytes": None,
                    "vram_max_reserved_bytes": None}, {key: "device_not_cuda" for key in (
                        "vram_allocated_bytes", "vram_reserved_bytes", "vram_max_allocated_bytes",
                        "vram_max_reserved_bytes")}
        cuda, values, reasons = self.torch.cuda, {}, {}
        for key, name in (("vram_allocated_bytes", "memory_allocated"), ("vram_reserved_bytes", "memory_reserved"),
                          ("vram_max_allocated_bytes", "max_memory_allocated"),
                          ("vram_max_reserved_bytes", "max_memory_reserved")):
            try:
                values[key] = int(getattr(cuda, name)(self.device))
            except Exception:
                values[key] = None
                reasons[key] = "collector_error"
        return values, reasons

    def epoch_start(self, epoch, *, batches, images):
        try:
            now = self.clock()
            self._epoch = {"epoch": epoch, "start": now, "mark": now, "batches": batches, "images_planned": images,
                           "images": 0, "train_done": None, "window_images": 0, "window_s": 0.0,
                           "window_loss": 0.0, "grad": None}
        except Exception:
            self._epoch = None

    def batch(self, epoch, index, images, loss_value, lr, grad_norm=None):
        """Called for EVERY batch (cheap); emits a record every ``batch_every``."""
        state = self._epoch
        if state is None or not self.writer.enabled:
            return
        try:
            now = self.clock()
            seconds = now - state["mark"]
            state["mark"] = now
            state["images"] += images
            state["window_images"] += images
            state["window_s"] += seconds
            state["window_loss"] += loss_value * images
            if grad_norm is not None:
                state["grad"] = (grad_norm, index)
            last = index + 1 == state["batches"]
            if (index + 1) % self.batch_every and not last:
                return
            reasons = {}
            record = {"epoch": epoch, "step": index + 1, "batches": state["batches"], "images": images,
                      "loss": finite(loss_value),
                      "window_loss_mean": finite(state["window_loss"] / state["window_images"]),
                      "lr": finite(lr), "batch_s": seconds,
                      "images_per_s": finite(images / seconds) if seconds > 0 else None,
                      "window_batches": None, "window_s": state["window_s"],
                      "window_images_per_s": (finite(state["window_images"] / state["window_s"])
                                              if state["window_s"] > 0 else None),
                      "grad_norm_preclip": None, "grad_norm_step": None, "grad_clipped": None,
                      "grad_clip_max_norm": self.grad_clip_max_norm}
            record["window_batches"] = (index % self.batch_every) + 1 if last else self.batch_every
            for key in ("loss", "window_loss_mean", "lr", "images_per_s", "window_images_per_s"):
                if record[key] is None:
                    reasons[key] = "non_finite"
            if seconds <= 0:
                reasons["images_per_s"] = "zero_duration"
            if state["window_s"] <= 0:
                reasons["window_images_per_s"] = "zero_duration"
            if state["grad"] is None:
                reasons["grad_norm_preclip"] = "no_optimizer_step"
            else:
                tensor, step = state["grad"]
                value = finite(float(tensor))
                record["grad_norm_step"] = step + 1
                record["grad_norm_preclip"] = value
                if value is None:
                    reasons["grad_norm_preclip"] = "non_finite"
                elif self.grad_clip_max_norm is not None:
                    record["grad_clipped"] = value > self.grad_clip_max_norm
            memory, memory_reasons = self._memory()
            record.update(memory)
            reasons.update(memory_reasons)
            record["null_reasons"] = reasons
            state["window_images"], state["window_s"], state["window_loss"] = 0, 0.0, 0.0
            self.writer.write("batch", record)
        except Exception as error:
            self.writer.fail(error)

    def train_done(self, epoch):
        if self._epoch is not None:
            self._epoch["train_done"] = self.clock()

    def epoch_end(self, epoch, row, *, selection_per_epoch):
        """``row`` is the trainer's metrics.json row; only whitelisted keys are copied."""
        state = self._epoch
        if state is None or not self.writer.enabled:
            return
        try:
            now = self.clock()
            done = state["train_done"] if state["train_done"] is not None else now
            train_s = done - state["start"]
            reasons = {}
            record = {"epoch": epoch, "batches": state["batches"], "images": state["images"],
                      "epoch_s": now - state["start"], "train_s": train_s, "post_train_s": now - done,
                      "images_per_s": finite(state["images"] / train_s) if train_s > 0 and state["images"] else None}
            if record["images_per_s"] is None:
                reasons["images_per_s"] = "no_samples"
            for key in _EPOCH_KEYS:
                if key in row:
                    record[key] = finite(row[key])
                    if record[key] is None:
                        reasons[key] = "non_finite"
            selection = row.get("selection") if selection_per_epoch else None
            if isinstance(selection, dict):
                record["selection"] = {key: finite(selection.get(key)) for key in _SELECTION_KEYS}
            else:
                record["selection"] = None
                reasons["selection"] = "not_evaluated_per_epoch"
            memory, memory_reasons = self._memory()
            record.update(memory)
            reasons.update(memory_reasons)
            record["null_reasons"] = reasons
            self.writer.write("epoch", record, sync=True)
        except Exception as error:
            self.writer.fail(error)
        finally:
            self._epoch = None

    def outcome(self, status):
        self._outcome = status

    def close(self, error=None):
        """Stop the sampler, write session_end and summary.json. Never raises."""
        if self._closed:
            return
        self._closed = True
        status = self._outcome or "failed"
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(CLOSE_JOIN_S)
            payload = {"status": status, "thread_alive": bool(self._thread and self._thread.is_alive())}
            if error is not None:
                payload["error_type"] = type(error).__name__
            self.writer.write("session_end", payload, sync=True)
        except Exception:
            pass
        finally:
            self.writer.close()
        try:
            write_summary(self.directory, status=status,
                          machine_interval_s=self.machine_interval_s or MACHINE_INTERVAL_S)
        except Exception:
            pass


def open_for_attempt(output, *, torch, device, context, grad_clip_max_norm=None):
    """Factory used by train.py. Disabled or failing setup => NullTelemetry."""
    if not ENABLED:
        return NullTelemetry()
    try:
        # ``context`` may be a callable so that building it cannot raise in the trainer.
        context = context() if callable(context) else context
        return TrainTelemetry(output, torch=torch, device=device, context=context,
                              grad_clip_max_norm=grad_clip_max_norm)
    except Exception:
        return NullTelemetry()


# ---------------------------------------------------------------------------
# Summary (re-derived from the JSONL, so it covers every resumed session)
# ---------------------------------------------------------------------------

def read_records(path):
    records, invalid = [], 0
    try:
        with Path(path).open("rb") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw)
                except ValueError:
                    invalid += 1
                    continue
                if isinstance(value, dict) and value.get("schema") == SCHEMA:
                    records.append(value)
                else:
                    invalid += 1
    except FileNotFoundError:
        pass
    return records, invalid


def _reason(records, key, default="not_exposed"):
    """Most recent null reason recorded for ``key`` (or a ``gpus`` umbrella)."""
    for record in reversed(records):
        reasons = record.get("null_reasons") or {}
        if key in reasons:
            return reasons[key]
    return default


def _max(values):
    values = [value for value in values if finite(value) is not None]
    return max(values) if values else None


def integrate_energy(sessions, machine_interval_s):
    """Trapezoid over power_w per GPU with monotonic dt, inside each session.

    Gaps larger than ENERGY_MAX_GAP_FACTOR x interval are not integrated.
    Returns (kwh|None, detail).
    """
    max_gap = ENERGY_MAX_GAP_FACTOR * machine_interval_s
    joules, integrated_s, span_s, devices = 0.0, 0.0, 0.0, set()
    powered = False
    for machine in sessions.values():
        previous = {}
        for record in machine:
            for gpu in record.get("gpus") or []:
                key = gpu.get("uuid") or gpu.get("index")
                power = finite(gpu.get("power_w"))
                if power is None:
                    previous.pop(key, None)
                    continue
                powered = True
                devices.add(str(key))
                if key in previous:
                    t0, p0 = previous[key]
                    dt = record["mono_s"] - t0
                    if dt > 0:
                        span_s += dt
                        if dt <= max_gap:
                            joules += 0.5 * (p0 + power) * dt
                            integrated_s += dt
                previous[key] = (record["mono_s"], power)
    if not powered:
        return None, {"status": "unavailable", "reason": "not_exposed", "source": None}
    detail = {"status": "measured_integrated", "source": f"nvidia-smi power.draw @{machine_interval_s:g}s",
              "boundary": "gpu_board_only", "gpus": sorted(devices), "integrated_s": integrated_s,
              "coverage_fraction": integrated_s / span_s if span_s > 0 else None,
              "note": "coarse sampling; not wall-plug energy"}
    if span_s <= 0:
        return None, dict(detail, status="unavailable", reason="first_sample")
    return joules / 3.6e6, detail


def summarize(records, invalid=0, *, status, machine_interval_s=MACHINE_INTERVAL_S):
    sessions, machines = {}, {}
    epochs, batches = {}, []
    errors = 0
    for record in records:
        session = record.get("session_id")
        sessions.setdefault(session, []).append(record)
        kind = record.get("type")
        if kind == "epoch":
            epochs[record.get("epoch")] = record
        elif kind == "batch":
            batches.append(record)
        elif kind == "machine":
            machines.setdefault(session, []).append(record)
        elif kind == "error":
            errors += 1
    duration = 0.0
    for rows in sessions.values():
        times = [row["mono_s"] for row in rows if finite(row.get("mono_s")) is not None]
        if times:
            duration += max(times) - min(times)
    train_s = sum(finite(row.get("train_s")) or 0.0 for row in epochs.values())
    images = sum(row.get("images") or 0 for row in epochs.values())
    gpu_rows = [gpu for rows in machines.values() for row in rows for gpu in (row.get("gpus") or [])]
    reasons = {}
    summary = {
        "schema": SUMMARY_SCHEMA, "status": status, "utc": utc_now(), "sessions": len(sessions),
        "records": len(records), "invalid_lines": invalid, "telemetry_errors": errors,
        "duration_s": duration, "epochs_recorded": len(epochs),
        "epochs": sorted(key for key in epochs if type(key) is int),
        "train_images_processed": images, "train_s": train_s,
        "throughput_img_s_mean": images / train_s if train_s > 0 and images else None,
        "vram_max_allocated_bytes": _max(row.get("vram_max_allocated_bytes") for row in
                                         list(batches) + list(epochs.values())),
        "vram_max_reserved_bytes": _max(row.get("vram_max_reserved_bytes") for row in
                                        list(batches) + list(epochs.values())),
        "gpu_temp_max_c": _max(gpu.get("temp_c") for gpu in gpu_rows),
        "gpu_power_max_w": _max(gpu.get("power_w") for gpu in gpu_rows),
        "gpu_mem_used_max_mib": _max(gpu.get("mem_used_mib") for gpu in gpu_rows),
        "process_rss_max_bytes": _max(row.get("rss_bytes") for rows in machines.values() for row in rows),
        "machine_samples": sum(len(rows) for rows in machines.values()),
        "batch_records": len(batches),
    }
    summary["energy_kwh"], summary["energy"] = integrate_energy(machines, machine_interval_s)
    machine_rows = [row for rows in machines.values() for row in rows]
    gpu_reason = _reason(machine_rows, "gpus")
    sources = {"throughput_img_s_mean": "no_samples",
               "vram_max_allocated_bytes": _reason(batches, "vram_max_allocated_bytes"),
               "vram_max_reserved_bytes": _reason(batches, "vram_max_reserved_bytes"),
               "gpu_temp_max_c": gpu_reason, "gpu_power_max_w": gpu_reason, "gpu_mem_used_max_mib": gpu_reason,
               "process_rss_max_bytes": _reason(machine_rows, "rss_bytes"), "energy_kwh": gpu_reason}
    if summary["energy_kwh"] is None:
        if summary["energy"].get("reason") == "not_exposed":
            summary["energy"]["reason"] = gpu_reason
        sources["energy_kwh"] = summary["energy"]["reason"]
    for key, reason in sources.items():
        if summary[key] is None:
            reasons[key] = reason
    summary["null_reasons"] = reasons
    return summary


def write_summary(directory, *, status, machine_interval_s=MACHINE_INTERVAL_S):
    from .contracts import atomic_json
    records, invalid = read_records(Path(directory) / JSONL_NAME)
    summary = summarize(records, invalid, status=status, machine_interval_s=machine_interval_s)
    atomic_json(Path(directory) / SUMMARY_NAME, summary)
    return summary
