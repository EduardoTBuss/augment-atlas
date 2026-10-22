"""Telemetry v1 readers and pure parsers (docs/fleet-protocol.md section 5).

Observation only: nothing here writes to sysfs/procfs, changes fans, power
limits, clocks, processes or containers, or asks for privileges. Every reader
receives a ``Readers`` object so tests can replay captured fixtures on Windows
and Linux (``procfs_root``, ``sysfs_root``, ``run_cmd``, ``open_file``,
``listdir``, ``disk_usage``). A missing or unreadable sensor becomes ``None``
plus a reason code from ``fleet_schema.REASONS``; it never raises to callers of
the ``collect_*`` functions.

Intervals always come from ``time.monotonic()``; UTC is only a label.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from .fleet_schema import SCHEMA_TELEMETRY

# ------------------------------------------------------------------- readers


class NotCaptured(FileNotFoundError):
    """A replayed capture does not contain this file/command (not a host fact)."""


class CommandFailed(OSError):
    def __init__(self, argv, returncode, stderr=""):
        super().__init__(f"{Path(str(argv[0])).name} exited {returncode}")
        self.returncode = returncode
        self.stderr = stderr


def run_command(argv, timeout):
    """Run a read-only command with a deadline. Returns stdout text.

    Raises FileNotFoundError (tool absent), subprocess.TimeoutExpired,
    PermissionError or CommandFailed (nonzero exit). ``subprocess.run`` kills and
    reaps the child on timeout, so no zombie is left behind.
    """
    env = dict(os.environ, LC_ALL="C", LANG="C")
    try:
        result = subprocess.run([str(part) for part in argv], stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    except PermissionError as exc:
        raise absent_or_denied(argv, exc) from exc
    if result.returncode != 0:
        raise CommandFailed(argv, result.returncode, result.stderr[-400:])
    return result.stdout


def absent_or_denied(argv, exc):
    """execvp reports EACCES, not ENOENT, for a bare name missing from PATH when
    some PATH entry is unreadable; that is an absent tool, not a denied one."""
    name = str(argv[0])
    if os.sep not in name and shutil.which(name) is None:
        return FileNotFoundError(2, "No such file or directory", name)
    return exc


def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


class Readers:
    """Injectable access to the host. Defaults read the live Linux host."""

    def __init__(self, procfs_root="/proc", sysfs_root="/sys", crash_dir="/var/crash", etc_root="/etc",
                 run_cmd=run_command, open_file=read_text_file, listdir=os.listdir,
                 disk_usage=shutil.disk_usage):
        self.procfs_root = Path(procfs_root)
        self.sysfs_root = Path(sysfs_root)
        self.crash_dir = Path(crash_dir)
        self.etc_root = Path(etc_root)
        self.run_cmd = run_cmd
        self.open_file = open_file
        self.listdir = listdir
        self.disk_usage = disk_usage

    def proc(self, *parts):
        return self.procfs_root.joinpath(*parts)

    def sys(self, *parts):
        return self.sysfs_root.joinpath(*parts)

    def read(self, path):
        return self.open_file(path)

    def read_strip(self, path):
        return self.open_file(path).strip()


#: Default capture layout of tests/fixtures/telemetry_v1/<node>/ (section 8.1).
FIXTURE_COMMANDS = {
    ("sensors", "-j"): "sensors-j.json",
    ("sensors",): "sensors.txt",
    ("nvidia-smi", "--help-query-gpu"): "nvidia-smi-help-query-gpu.txt",
    ("nvidia-smi", "-q", "-d", "PERFORMANCE,POWER,TEMPERATURE,ECC"): "nvidia-smi-q-perf-power-temp-ecc.txt",
    ("nvidia-smi", "--version"): "nvidia-smi-version.txt",
    ("timedatectl", "show", "-p", "NTPSynchronized"): "timedatectl.txt",
    ("lscpu",): "lscpu.txt",
}
#: Directories whose capture was a full listing: a missing file there is a host fact.
FIXTURE_AUTHORITATIVE = ("sys/class/hwmon", "sys/class/thermal", "sys/class/powercap")
#: MSYS/Cygwin store ``:`` of file names as U+F03A on NTFS; replay accepts both forms.
COLON_PUA = ""


class ReplayReaders(Readers):
    """Replays a captured fixture tree (``<root>/proc``, ``<root>/sys``...).

    - ``<file>.0``, ``<file>.1``... are served in order (sticks on the last);
    - ``<file>.attempt.txt`` mentioning "denied"/"negada" replays PermissionError;
    - a missing file inside ``FIXTURE_AUTHORITATIVE`` is FileNotFoundError, any
      other missing file or command is ``NotCaptured``;
    - ``var/crash`` is listed from ``var-crash-ls.txt`` (names only).
    """

    def __init__(self, root, commands=None, authoritative=FIXTURE_AUTHORITATIVE, run_cmd=None,
                 disk_usage=shutil.disk_usage, crash_listing="var-crash-ls.txt"):
        self.root = Path(root)
        if commands is None:
            commands = dict(FIXTURE_COMMANDS)
            commands[self._key(nvidia_query_argv(["nvidia-smi"], FAST_QUERY_FIELDS))] = "nvidia-smi-query.csv"
            commands[self._key(nvidia_query_argv(["nvidia-smi"], FAST_QUERY_FIELDS, header=True, units=True))] =                 "nvidia-smi-query-header.csv"
        self.commands = dict(commands)
        self.authoritative = tuple(authoritative)
        self.crash_listing = crash_listing
        self._sequence = {}
        self.calls = []
        super().__init__(procfs_root=self.root / "proc", sysfs_root=self.root / "sys",
                         crash_dir=self.root / "var" / "crash", etc_root=self.root / "etc",
                         run_cmd=run_cmd or self._replay_cmd, open_file=self._open, listdir=self._listdir,
                         disk_usage=disk_usage)

    def add_command(self, argv, filename):
        self.commands[self._key(argv)] = filename

    @staticmethod
    def _key(argv):
        return (Path(str(argv[0])).name,) + tuple(str(part) for part in argv[1:])

    def _relative(self, path):
        try:
            relative = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(self.root))).as_posix()
        except ValueError:
            raise NotCaptured(2, "outside the capture", str(path)) from None
        return relative.replace(COLON_PUA, ":")

    def _existing(self, relative, test):
        """Filesystem path for ``relative``. ``:`` may be stored as U+F03A (MSYS/Cygwin on NTFS)."""
        for candidate in dict.fromkeys((relative, relative.replace(":", COLON_PUA))):
            path = self.root.joinpath(*candidate.split("/"))
            try:
                if test(path):
                    return path
            except OSError:
                continue
        return None

    def _open(self, path):
        relative = self._relative(path)
        found = self._existing(relative, Path.is_file)
        if found is not None:
            return read_text_file(found)
        parent_rel, _, name = relative.rpartition("/")
        parent = self._existing(parent_rel, Path.is_dir)
        if parent is not None:
            indexed = sorted((item for item in parent.iterdir() if re.fullmatch(re.escape(name) + r"\.\d+", item.name)),
                             key=lambda item: int(item.suffix[1:]))
            if indexed:
                position = self._sequence.get(relative, 0)
                self._sequence[relative] = position + 1
                return read_text_file(indexed[min(position, len(indexed) - 1)])
            attempt = parent / (name + ".attempt.txt")
            if attempt.is_file():
                text = read_text_file(attempt).lower()
                if "denied" in text or "negada" in text:
                    raise PermissionError(13, "Permission denied (replayed capture)", relative)
        if any(relative == item or relative.startswith(item + "/") for item in self.authoritative):
            raise FileNotFoundError(2, "No such file (authoritative capture)", relative)
        raise NotCaptured(2, "not in capture", relative)

    def _listdir(self, path):
        relative = self._relative(path)
        if relative == "var/crash":
            listing = self.root / self.crash_listing
            if not listing.is_file():
                raise NotCaptured(2, "not in capture", relative)
            return parse_ls_names(read_text_file(listing))
        found = self._existing(relative, Path.is_dir)
        if found is not None:
            return sorted(item.name.replace(COLON_PUA, ":") for item in found.iterdir())
        if any(relative == item or relative.startswith(item + "/") for item in self.authoritative):
            raise FileNotFoundError(2, "No such directory (authoritative capture)", relative)
        raise NotCaptured(2, "not in capture", relative)

    def _replay_cmd(self, argv, timeout):
        key = self._key(argv)
        self.calls.append(key)
        name = self.commands.get(key)
        if name is None:
            raise NotCaptured(2, "command not in capture", " ".join(key))
        return read_text_file(self.root / name)


def file_reason(exc):
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, (FileNotFoundError, NotADirectoryError, IsADirectoryError)):
        return "not_exposed"
    if isinstance(exc, (ValueError, IndexError, KeyError)):
        return "parse_error"
    return "collector_error"


def command_reason(exc):
    if isinstance(exc, NotCaptured):
        return "not_exposed"
    if isinstance(exc, FileNotFoundError):
        return "tool_absent"
    if isinstance(exc, subprocess.TimeoutExpired):
        return "tool_timeout"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, (ValueError, IndexError, KeyError)):
        return "parse_error"
    return "collector_error"


# ---------------------------------------------------------------- time labels

def utc_label(epoch=None):
    """``2026-10-19T12:00:00.512Z`` (milliseconds, UTC)."""
    epoch = time.time() if epoch is None else epoch
    moment = dt.datetime.fromtimestamp(epoch, dt.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def parse_utc(text):
    value = dt.datetime.strptime(text.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z" if "." in text
                                 else "%Y-%m-%dT%H:%M:%S%z")
    return value.timestamp()


def _num(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return int(value) if value.is_integer() and "." not in text and "e" not in text.lower() else value


# ------------------------------------------------------------ lm-sensors / hwmon

_CORE = re.compile(r"^Core\s+(\d+)$")
_PACKAGE = re.compile(r"^Package id\s+(\d+)$")
_AMD_PACKAGE = {"Tctl", "Tdie"}


def _classify_chip(chip):
    name = chip.split("-", 1)[0]
    if name in {"coretemp", "k10temp", "zenpower"}:
        return "cpu"
    if name == "acpitz":
        return "acpi_zone"
    return "other"


def _sensor_entry(cpu, chip, label, temp, high, crit):
    kind = _classify_chip(chip)
    package = _PACKAGE.match(label)
    core = _CORE.match(label)
    if kind == "cpu" and package:
        cpu["packages"].append({"package": int(package.group(1)), "temp_c": temp, "high_c": high, "crit_c": crit})
    elif kind == "cpu" and label in _AMD_PACKAGE:
        cpu["packages"].append({"package": 0, "temp_c": temp, "high_c": high, "crit_c": crit})
    elif kind == "cpu" and core:
        cpu["cores"].append({"core": int(core.group(1)), "temp_c": temp})
    else:
        cpu["zones"].append({"chip": chip.split("-", 1)[0], "label": label, "temp_c": temp, "kind": kind})


def _finish_cpu(cpu):
    cpu["packages"].sort(key=lambda item: item["package"])
    cpu["cores"].sort(key=lambda item: item["core"])
    highs = {item["high_c"] for item in cpu["packages"] if item["high_c"] is not None}
    crits = {item["crit_c"] for item in cpu["packages"] if item["crit_c"] is not None}
    cpu["limits"] = {"high_c": max(highs) if highs else None, "crit_c": max(crits) if crits else None}
    return cpu


def parse_sensors_json(text):
    """``sensors -j`` -> {packages, cores, zones, limits}. ``acpitz`` is a zone, not CPU."""
    import json
    data = json.loads(text)
    cpu = {"packages": [], "cores": [], "zones": []}
    for chip, features in data.items():
        if not isinstance(features, dict):
            continue
        for label, values in features.items():
            if label == "Adapter" or not isinstance(values, dict):
                continue
            temp = high = crit = None
            for key, value in values.items():
                if re.fullmatch(r"temp\d+_input", key):
                    temp = float(value)
                elif re.fullmatch(r"temp\d+_max", key):
                    high = float(value)
                elif re.fullmatch(r"temp\d+_crit", key):
                    crit = float(value)
            if temp is not None:
                _sensor_entry(cpu, chip, label, temp, high, crit)
    return _finish_cpu(cpu)


_TEXT_LINE = re.compile(r"^(?P<label>[^:]+):\s+(?P<temp>[+-]?\d+(?:\.\d+)?)\s*°?C(?P<rest>.*)$")
_TEXT_LIMIT = re.compile(r"(high|crit|max)\s*=\s*([+-]?\d+(?:\.\d+)?)\s*°?C")


def parse_sensors_text(text):
    """Plain ``sensors`` output; same normalized structure as ``parse_sensors_json``."""
    cpu = {"packages": [], "cores": [], "zones": []}
    chip = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            chip = None
            continue
        if chip is None and ":" not in line:
            chip = line.strip()
            continue
        if chip is None or line.startswith("Adapter:"):
            continue
        match = _TEXT_LINE.match(line.replace("Â", ""))
        if not match:
            continue
        limits = {key: float(value) for key, value in _TEXT_LIMIT.findall(match.group("rest"))}
        _sensor_entry(cpu, chip, match.group("label").strip(), float(match.group("temp")),
                      limits.get("high", limits.get("max")), limits.get("crit"))
    return _finish_cpu(cpu)


def read_cpu_sensors_cli(readers, timeout=4.0):
    """``sensors -j`` with fallback to text ``sensors``. Returns (cpu, reason, src)."""
    try:
        return parse_sensors_json(readers.run_cmd(["sensors", "-j"], timeout)), None, "sensors-j"
    except Exception as first:  # noqa: BLE001 - every failure falls back
        try:
            return parse_sensors_text(readers.run_cmd(["sensors"], timeout)), None, "sensors-text"
        except Exception as second:  # noqa: BLE001
            reason = command_reason(second)
            if reason == "tool_absent" and not isinstance(first, FileNotFoundError):
                reason = command_reason(first)
            return None, reason, None


def _milli(text):
    return int(text.strip()) / 1000.0


def read_hwmon(readers):
    """All ``/sys/class/hwmon/hwmon*`` chips: temps (with label/max/crit), fans, pwm."""
    base = readers.sys("class", "hwmon")
    chips = []
    for entry in sorted(readers.listdir(base), key=lambda item: (len(item), item)):
        if not entry.startswith("hwmon"):
            continue
        directory = base / entry
        try:
            name = readers.read_strip(directory / "name")
        except OSError:
            name = "unknown"
        files = set(readers.listdir(directory))
        chip = {"hwmon": entry, "name": name, "temps": [], "fans": [], "pwms": []}
        for filename in sorted(files):
            temp = re.fullmatch(r"temp(\d+)_input", filename)
            fan = re.fullmatch(r"fan(\d+)_input", filename)
            pwm = re.fullmatch(r"pwm(\d+)", filename)
            try:
                if temp:
                    number = temp.group(1)
                    item = {"sensor": int(number), "label": None, "temp_c": None, "temp_reason": None,
                            "high_c": None, "crit_c": None}
                    for suffix, key in (("label", "label"), ("max", "high_c"), ("crit", "crit_c")):
                        other = f"temp{number}_{suffix}"
                        if other in files:
                            value = readers.read_strip(directory / other)
                            item[key] = value if suffix == "label" else int(value) / 1000.0
                    # A failing value read keeps the sensor (label, limits) with temp_c=None + reason,
                    # so discovery of the fast-tier package path never depends on one good reading.
                    try:
                        item["temp_c"] = _milli(readers.read(directory / filename))
                    except Exception as exc:  # noqa: BLE001 - fault isolation is the contract (5.1)
                        item["temp_reason"] = file_reason(exc)
                    chip["temps"].append(item)
                elif fan:
                    chip["fans"].append({"fan": int(fan.group(1)),
                                         "rpm": int(readers.read_strip(directory / filename))})
                elif pwm:
                    number = pwm.group(1)
                    enable = None
                    if f"pwm{number}_enable" in files:
                        enable = int(readers.read_strip(directory / f"pwm{number}_enable"))
                    chip["pwms"].append({"pwm": int(number), "value": int(readers.read_strip(directory / filename)),
                                         "enable": enable})
            except Exception:  # noqa: BLE001 - one bad file never hides the rest of the chip
                continue
        chip["temps"].sort(key=lambda item: item["sensor"])
        chip["fans"].sort(key=lambda item: item["fan"])
        chip["pwms"].sort(key=lambda item: item["pwm"])
        chips.append(chip)
    return chips


def cpu_from_hwmon(chips):
    """Normalize hwmon chips into the ``parse_sensors_*`` structure."""
    cpu = {"packages": [], "cores": [], "zones": []}
    for chip in chips:
        for item in chip["temps"]:
            label = item["label"] or f"temp{item['sensor']}"
            _sensor_entry(cpu, chip["name"], label, item["temp_c"], item["high_c"], item["crit_c"])
    return _finish_cpu(cpu)


def fans_from_hwmon(chips):
    """Returns (fans, reason). No fan/pwm file at all -> (None, ``not_exposed``)."""
    fans = []
    for chip in chips:
        pwms = {item["pwm"]: item for item in chip["pwms"]}
        numbers = sorted({item["fan"] for item in chip["fans"]} | set(pwms))
        rpm = {item["fan"]: item["rpm"] for item in chip["fans"]}
        for number in numbers:
            pwm = pwms.get(number)
            fans.append({"chip": chip["name"], "fan": number, "rpm": rpm.get(number),
                         "pwm": None if pwm is None else pwm["value"],
                         "pwm_pct": None if pwm is None else round(pwm["value"] * 100.0 / 255.0, 1),
                         "pwm_enable": None if pwm is None else pwm["enable"]})
    return (fans, None) if fans else (None, "not_exposed")


def hwmon_temp_reasons(chips):
    """{("package"|"core", number): reason} for CPU sensors whose value read failed."""
    failed = {}
    for chip in chips:
        if _classify_chip(chip["name"]) != "cpu":
            continue
        for item in chip["temps"]:
            if item.get("temp_reason") is None:
                continue
            label = item["label"] or ""
            package, core = _PACKAGE.match(label), _CORE.match(label)
            if package or label in _AMD_PACKAGE:
                failed[("package", int(package.group(1)) if package else 0)] = item["temp_reason"]
            elif core:
                failed[("core", int(core.group(1)))] = item["temp_reason"]
    return failed


def package_temp_paths(readers):
    """Discover hwmon ``temp*_input`` files of CPU packages, for the 0.5 s tier."""
    found = []
    for chip in read_hwmon(readers):
        if _classify_chip(chip["name"]) != "cpu":
            continue
        for item in chip["temps"]:
            label = item["label"] or ""
            package = _PACKAGE.match(label)
            if package or label in _AMD_PACKAGE:
                found.append((int(package.group(1)) if package else 0,
                              readers.sys("class", "hwmon", chip["hwmon"], f"temp{item['sensor']}_input"),
                              chip["name"]))
    return sorted(found)


def read_thermal_zones(readers):
    base = readers.sys("class", "thermal")
    zones = []
    for entry in sorted(readers.listdir(base), key=lambda item: (len(item), item)):
        if not entry.startswith("thermal_zone"):
            continue
        try:
            zones.append({"zone": int(entry[len("thermal_zone"):]),
                          "type": readers.read_strip(base / entry / "type"),
                          "temp_c": _milli(readers.read(base / entry / "temp"))})
        except (OSError, ValueError):
            continue
    return zones


# ------------------------------------------------------------------ nvidia-smi

#: Query used by the fast stream; identical to the x299 capture (section 8.1).
FAST_QUERY_FIELDS = ("index", "uuid", "name", "temperature.gpu", "power.draw", "power.limit", "fan.speed",
                     "utilization.gpu", "memory.used", "memory.total", "clocks.sm", "clocks.mem", "pstate")
PCIE_QUERY_FIELDS = ("index", "pcie.link.gen.current", "pcie.link.width.current")
CAPABILITY_QUERY_FIELDS = ("index", "compute_cap")
GPU_FIELD_MAP = {
    "index": "index", "uuid": "uuid", "name": "name", "temperature.gpu": "temp_c", "power.draw": "power_w",
    "power.limit": "power_limit_w", "fan.speed": "fan_pct", "utilization.gpu": "util_pct",
    "memory.used": "mem_used_mib", "memory.total": "mem_total_mib", "clocks.sm": "sm_mhz", "clocks.mem": "mem_mhz",
    "pstate": "pstate", "compute_cap": "compute_capability", "pcie.link.gen.current": "pcie_gen",
    "pcie.link.width.current": "pcie_width", "clocks_event_reasons.active": "clock_event_reasons",
    "total_energy_consumption": "energy_mj",
}
GPU_FIELD_ALIASES = {
    "clocks.current.sm": "clocks.sm", "clocks.current.memory": "clocks.mem", "gpu_name": "name",
    "gpu_uuid": "uuid", "clocks_throttle_reasons.active": "clocks_event_reasons.active",
    "gpu.power.requested_limit": "power.limit", "clocks.current.graphics": "clocks.gr",
}
_STRING_FIELDS = {"uuid", "name", "pstate"}
_NOT_SUPPORTED = {"[n/a]", "n/a", "[not supported]", "not supported"}
_VALUE_WITH_UNIT = re.compile(r"^([-+]?\d+(?:\.\d+)?)\s*(?:W|%|MiB|MHz|C|mJ)?$")
#: NVML clock event reason bits (nvml.h ``nvmlClocksEventReason*``).
CLOCK_EVENT_BITS = ((0x1, "gpu_idle"), (0x2, "applications_clocks_setting"), (0x4, "sw_power_cap"),
                    (0x8, "hw_slowdown"), (0x10, "sync_boost"), (0x20, "sw_thermal_slowdown"),
                    (0x40, "hw_thermal_slowdown"), (0x80, "hw_power_brake_slowdown"),
                    (0x100, "display_clock_setting"))


def nvidia_query_argv(prefix, fields, header=False, units=False, loop_ms=None):
    fmt = "csv" + ("" if header else ",noheader") + ("" if units else ",nounits")
    argv = list(prefix) + ["--query-gpu=" + ",".join(fields), "--format=" + fmt]
    if loop_ms is not None:
        argv += ["-lms", str(int(loop_ms))]
    return argv


def canonical_field(name):
    name = re.sub(r"\s*\[[^\]]*\]\s*$", "", name.strip())
    return GPU_FIELD_ALIASES.get(name, name)


def clock_event_names(mask):
    return [name for bit, name in CLOCK_EVENT_BITS if mask & bit]


def parse_gpu_value(field, raw):
    """Returns (value, reason). Never ``0`` for an absent value."""
    text = raw.strip()
    lowered = text.lower()
    if lowered in _NOT_SUPPORTED:
        return None, "not_supported_by_device"
    if lowered in {"[insufficient permissions]", "insufficient permissions"}:
        return None, "permission_denied"
    if text.startswith("[") or not text:
        return None, "parse_error" if not text else "collector_error"
    if field in _STRING_FIELDS:
        return text, None
    if field == "compute_cap":
        major, minor = text.split(".", 1)
        return [int(major), int(minor)], None
    if field == "clocks_event_reasons.active":
        return clock_event_names(int(text, 16)), None
    match = _VALUE_WITH_UNIT.match(text)
    if not match:
        return None, "parse_error"
    value = _num(match.group(1))
    if field in {"index", "pcie.link.gen.current", "pcie.link.width.current"}:
        value = int(value)
    return value, None


def parse_nvidia_csv(text, fields=None):
    """Parse ``--query-gpu`` CSV by field NAME.

    ``fields`` is the queried order for ``noheader`` output; ``None`` means the
    first line is the header (names may carry units like ``power.draw [W]``).
    Returns (rows, bad_lines); each row is {"index", "values", "reasons"} where
    values use the telemetry names (``temp_c``, ``power_w``...). Rows with a wrong
    column count or an unparseable index are dropped and counted.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if fields is None:
        if not lines:
            return [], 0
        header = next(csv.reader([lines[0]], skipinitialspace=True))
        fields = [canonical_field(name) for name in header]
        lines = lines[1:]
    else:
        fields = [canonical_field(name) for name in fields]
    rows, bad = [], 0
    for line in lines:
        try:
            cells = next(csv.reader([line], skipinitialspace=True))
        except (csv.Error, StopIteration):
            bad += 1
            continue
        if len(cells) != len(fields) or "index" not in fields:
            bad += 1
            continue
        values, reasons = {}, {}
        for field, cell in zip(fields, cells):
            key = GPU_FIELD_MAP.get(field, field)
            try:
                value, reason = parse_gpu_value(field, cell)
            except (ValueError, IndexError):
                value, reason = None, "parse_error"
            values[key] = value
            if reason:
                reasons[key] = reason
        if type(values.get("index")) is not int:
            bad += 1
            continue
        rows.append({"index": values["index"], "values": values, "reasons": reasons})
    return rows, bad


def parse_help_query_gpu(text):
    """Field names (and aliases) listed by ``nvidia-smi --help-query-gpu``."""
    fields = set()
    for line in text.splitlines():
        if line.startswith('"'):
            fields.update(re.findall(r'"([A-Za-z0-9_.]+)"', line))
    return fields


_Q_KEY = re.compile(r"^(?P<indent>\s+)(?P<key>[^:]+?)\s+:\s?(?P<value>.*)$")
_Q_SECTION = re.compile(r"^(?P<indent>\s+)(?P<key>\S.*?)\s*$")
_REASON_NAMES = {"Idle": "gpu_idle", "Applications Clocks Setting": "applications_clocks_setting",
                 "SW Power Cap": "sw_power_cap", "HW Slowdown": "hw_slowdown",
                 "HW Thermal Slowdown": "hw_thermal_slowdown", "HW Power Brake Slowdown": "hw_power_brake_slowdown",
                 "Sync Boost": "sync_boost", "SW Thermal Slowdown": "sw_thermal_slowdown",
                 "Display Clock Setting": "display_clock_setting"}


def _q_blocks(text):
    blocks, current = [], None
    for line in text.splitlines():
        if re.match(r"^GPU [0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$", line.strip()) \
                and not line.startswith(" "):
            current = {}
            blocks.append(current)
            stack = []
            continue
        if current is None:
            continue
        match = _Q_KEY.match(line)
        section = None if match else _Q_SECTION.match(line)
        source = match or section
        if not source:
            continue
        depth = len(source.group("indent"))
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if match:
            path = "/".join([item[1] for item in stack] + [match.group("key").strip()])
            current[path] = match.group("value").strip()
        else:
            stack.append((depth, section.group("key").strip()))
    return blocks


def _q_number(value):
    if value is None or value.upper().startswith("N/A"):
        return None
    match = re.match(r"^([-+]?\d+(?:\.\d+)?)", value)
    return _num(match.group(1)) if match else None


def parse_nvidia_q(text):
    """``nvidia-smi -q -d PERFORMANCE,POWER,TEMPERATURE,ECC`` -> one dict per GPU (index = order)."""
    gpus = []
    for index, block in enumerate(_q_blocks(text)):
        reasons, null_reasons = None, {}
        for prefix in ("Clocks Event Reasons", "Clocks Throttle Reasons"):
            items = {key[len(prefix) + 1:].rsplit("/", 1)[-1]: value for key, value in block.items()
                     if key.startswith(prefix + "/")}
            if items:
                reasons = sorted(_REASON_NAMES.get(name, re.sub(r"\W+", "_", name.lower()))
                                 for name, value in items.items() if value == "Active")
                break
        if reasons is None:
            null_reasons["clock_event_reasons"] = "not_exposed"
        ecc_mode = block.get("ECC Mode/Current")
        ecc_errors = None
        if ecc_mode is None:
            null_reasons["ecc_errors"] = "not_exposed"
        elif ecc_mode.upper().startswith("N/A"):
            null_reasons["ecc_errors"] = "not_supported_by_device"
        elif ecc_mode.lower() == "disabled":
            null_reasons["ecc_errors"] = "disabled_by_config"
        else:
            total = next((block[key] for key in ("ECC Errors/Volatile/Double Bit/Total",
                                                 "ECC Errors/Volatile/DRAM Uncorrectable",
                                                 "ECC Errors/Volatile/SRAM Uncorrectable") if key in block), None)
            ecc_errors = _q_number(total)
            if ecc_errors is None:
                null_reasons["ecc_errors"] = "parse_error"

        def power(name):
            for section in ("GPU Power Readings", "Power Readings"):
                if f"{section}/{name}" in block:
                    return _q_number(block[f"{section}/{name}"])
            return None
        entry = {
            "index": index,
            "pstate": block.get("Performance State"),
            "clock_event_reasons": reasons,
            "ecc_mode": ecc_mode,
            "ecc_errors": ecc_errors,
            "power_draw_w": power("Instantaneous Power Draw") if power("Instantaneous Power Draw") is not None
            else power("Power Draw"),
            "power_limit_w": power("Current Power Limit") if power("Current Power Limit") is not None
            else power("Power Limit"),
            "power_default_limit_w": power("Default Power Limit"),
            "power_min_limit_w": power("Min Power Limit"),
            "power_max_limit_w": power("Max Power Limit"),
            "temp_c": _q_number(block.get("Temperature/GPU Current Temp")),
            "temp_shutdown_c": _q_number(block.get("Temperature/GPU Shutdown Temp")),
            "temp_slowdown_c": _q_number(block.get("Temperature/GPU Slowdown Temp")),
            "temp_target_c": _q_number(block.get("Temperature/GPU Target Temperature")),
            "null_reasons": null_reasons,
        }
        gpus.append(entry)
    return gpus


def parse_nvidia_version(text):
    driver = re.search(r"DRIVER version\s*:\s*(\S+)", text, re.IGNORECASE)
    cuda = re.search(r"CUDA Version\s*:\s*(\S+)", text, re.IGNORECASE)
    return {"driver": driver.group(1) if driver else None, "cuda": cuda.group(1) if cuda else None}


def count_compute_apps(text, uuids):
    """``--query-compute-apps=gpu_uuid --format=csv,noheader`` -> {index: count}."""
    seen = [line.strip() for line in text.splitlines() if line.strip()]
    return {index: sum(1 for item in seen if item == uuid) for index, uuid in uuids.items() if uuid}


# --------------------------------------------------------------------- procfs

def parse_proc_stat(text):
    """Aggregate ``cpu`` ticks (user..steal), ``btime`` and CPU count."""
    ticks, btime, cpus = None, None, 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "cpu":
            ticks = [int(value) for value in parts[1:9]]
        elif re.fullmatch(r"cpu\d+", parts[0]):
            cpus += 1
        elif parts[0] == "btime":
            btime = int(parts[1])
    if ticks is None or len(ticks) < 4:
        raise ValueError("no aggregate cpu line")
    ticks += [0] * (8 - len(ticks))
    return {"ticks": ticks, "btime": btime, "cpus": cpus}


def cpu_util_pct(previous, current):
    """100 x busy share of the tick delta (idle = idle + iowait). None if no progress."""
    if previous is None:
        return None
    total = sum(current["ticks"]) - sum(previous["ticks"])
    idle = (current["ticks"][3] + current["ticks"][4]) - (previous["ticks"][3] + previous["ticks"][4])
    if total <= 0 or idle < 0:
        return None
    return round(100.0 * (1.0 - idle / total), 4)


def parse_meminfo(text):
    values = {}
    for line in text.splitlines():
        if ":" in line:
            key, rest = line.split(":", 1)
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return {"ram_total_bytes": values.get("MemTotal"), "ram_available_bytes": values.get("MemAvailable"),
            "swap_total_bytes": values.get("SwapTotal"), "swap_free_bytes": values.get("SwapFree")}


def parse_loadavg(text):
    return [float(value) for value in text.split()[:3]]


def parse_uptime(text):
    return float(text.split()[0])


def read_boot_id(readers):
    """Returns (boot_id, reason)."""
    try:
        value = readers.read_strip(readers.proc("sys", "kernel", "random", "boot_id")).lower()
    except Exception as exc:  # noqa: BLE001
        return None, file_reason(exc)
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value):
        return None, "parse_error"
    return value, None


def read_cpu_freq_mhz(readers):
    """``scaling_cur_freq`` (kHz) of every CPU -> {min, mean, max} MHz."""
    base = readers.sys("devices", "system", "cpu")
    values = []
    for entry in readers.listdir(base):
        if re.fullmatch(r"cpu\d+", entry):
            try:
                values.append(int(readers.read_strip(base / entry / "cpufreq" / "scaling_cur_freq")) / 1000.0)
            except FileNotFoundError:
                continue
    if not values:
        raise FileNotFoundError("no scaling_cur_freq")
    return {"min": min(values), "mean": round(sum(values) / len(values), 1), "max": max(values)}


def parse_ls_names(text):
    """Names from ``ls -la`` text (owners/sizes are dropped)."""
    names = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 9 and line[:1] in "-dlcbps":
            name = " ".join(parts[8:]).split(" -> ", 1)[0]
            if name not in {".", ".."}:
                names.append(name)
    return names


def parse_lscpu(text):
    """English (``LC_ALL=C``) and Portuguese ``lscpu`` labels."""
    labels = {
        "model": ("Model name", "Nome do modelo"),
        "sockets": ("Socket(s)", "Soquete(s)"),
        "cores_per_socket": ("Core(s) per socket", "Núcleo(s) por soquete"),
        "threads_per_core": ("Thread(s) per core", "Thread(s) per núcleo"),
        "cpus": ("CPU(s)",),
    }
    found = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        for name, options in labels.items():
            if key in options and name not in found:
                found[name] = value
    result = {"model": found.get("model")}
    for name in ("sockets", "cores_per_socket", "threads_per_core", "cpus"):
        result[name] = int(found[name]) if found.get(name, "").isdigit() else None
    return result


def parse_timedatectl(text):
    match = re.search(r"NTPSynchronized=(yes|no)", text)
    if not match:
        raise ValueError("NTPSynchronized not found")
    return match.group(1) == "yes"


# ------------------------------------------------------------------------ RAPL

_RAPL_ZONE = re.compile(r"^intel-rapl:\d+(?::\d+)*$")
#: Upper bound used to decide if more than one wrap may have happened (section 5.2).
DEFAULT_RAPL_MAX_PLAUSIBLE_W = 1000.0


def read_rapl(readers):
    """Returns (zones, reasons). ``energy_uj`` is root-only on many hosts (x299)."""
    base = readers.sys("class", "powercap")
    zones, reasons = [], {}
    for entry in sorted(readers.listdir(base)):
        if not _RAPL_ZONE.match(entry):
            continue
        index = len(zones)
        zone = {"domain": None, "zone": entry, "energy_uj": None, "max_range_uj": None}
        for key, filename in (("domain", "name"), ("max_range_uj", "max_energy_range_uj"), ("energy_uj", "energy_uj")):
            try:
                text = readers.read_strip(base / entry / filename)
                zone[key] = text if key == "domain" else int(text)
            except Exception as exc:  # noqa: BLE001
                reasons[f"rapl[{index}].{key}"] = file_reason(exc)
        zones.append(zone)
    return zones, reasons


def rapl_delta(old_uj, new_uj, max_range_uj, dt_s, max_plausible_w=DEFAULT_RAPL_MAX_PLAUSIBLE_W):
    """Energy delta with one wraparound. Returns (delta_uj | None, event | None).

    ``new >= old`` -> ``new - old``; otherwise ``(max_range - old) + new`` and the
    ``rapl_wrap`` event. If ``dt_s x max_plausible_w`` exceeds the counter range,
    several wraps are possible: the delta is discarded and
    ``rapl_suspect_multi_wrap`` is reported.
    """
    if old_uj is None or new_uj is None or max_range_uj is None or dt_s is None:
        return None, None
    if dt_s * max_plausible_w * 1e6 > max_range_uj:
        return None, "rapl_suspect_multi_wrap"
    if new_uj >= old_uj:
        return new_uj - old_uj, None
    return (max_range_uj - old_uj) + new_uj, "rapl_wrap"


# ---------------------------------------------------------------------- energy

def integrate_power(samples, max_dt_s=2.0):
    """Trapezoidal energy over ``(mono_s, power_w | None)`` samples.

    An interval is integrated only if both ends have power and ``0 < dt <= max_dt_s``.
    Returns joules, kWh, covered/window seconds and ``coverage_fraction``.
    """
    points = sorted(samples, key=lambda item: item[0])
    energy_j = covered = 0.0
    excluded = 0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        span = t1 - t0
        if span <= 0:
            continue
        if p0 is None or p1 is None or span > max_dt_s:
            excluded += 1
            continue
        energy_j += 0.5 * (p0 + p1) * span
        covered += span
    window = points[-1][0] - points[0][0] if len(points) > 1 else 0.0
    return {"energy_j": energy_j, "kwh": energy_j / 3.6e6, "covered_s": covered, "window_s": window,
            "coverage_fraction": (covered / window) if window > 0 else None, "intervals_excluded": excluded}


def gpu_power_series(fast_records, gpu_index):
    """``(mono_s, power_w)`` for one GPU from fast records of a single boot."""
    series = []
    for record in fast_records:
        power = None
        for gpu in record.get("gpus") or []:
            if gpu.get("index") == gpu_index:
                power = gpu.get("power_w")
        series.append((record["mono_s"], power))
    return series


def energy_entry(counter_delta_j=None, integrated=None, estimate_j=None, method=None, reason=None, source=None):
    """Classify one energy figure (section 5.4, ADR D10).

    Priority: hardware counter (``measured_counter``) > integrated sampled power
    (``measured_integrated``) > named model (``estimated_model``) > ``unavailable``.
    """
    if counter_delta_j is not None:
        return {"value": counter_delta_j / 3.6e6, "status": "measured_counter", "source": source}
    if integrated is not None and integrated.get("covered_s", 0) > 0:
        return {"value": integrated["kwh"], "status": "measured_integrated", "source": source,
                "coverage_fraction": integrated["coverage_fraction"]}
    if estimate_j is not None:
        if not method:
            raise ValueError("an estimate requires a named, versioned method")
        return {"value": estimate_j / 3.6e6, "status": "estimated_model", "method": method}
    return {"value": None, "status": "unavailable", "reason": reason or "not_exposed"}


def estimate_cpu_energy_tdp_util_linear_v1(tdp_w, util_samples, max_dt_s=20.0):
    """``cpu_tdp_util_linear_v1``: E = TDP x util% x dt over ``(mono_s, util_pct)`` (always an estimate)."""
    if tdp_w is None:
        return None
    points = sorted(item for item in util_samples if item[1] is not None)
    energy = 0.0
    for (t0, u0), (t1, u1) in zip(points, points[1:]):
        span = t1 - t0
        if 0 < span <= max_dt_s:
            energy += tdp_w * 0.5 * (u0 + u1) / 100.0 * span
    return energy


# ---------------------------------------------------------------- reboot / crash

def detect_reboot(previous_state, boot_id, now_epoch):
    """Compare the persisted sentinel state with the current boot.

    Returns the ``reboot_detected`` payload or ``None``. ``gap_s`` uses UTC
    because monotonic clocks are not comparable across boots.
    """
    if not previous_state or not boot_id or boot_id == "unavailable":
        return None
    prev_boot = previous_state.get("boot_id")
    if not prev_boot or prev_boot == boot_id:
        return None
    last_utc = previous_state.get("last_sample_utc")
    gap = None
    if isinstance(last_utc, str):
        try:
            gap = round(max(0.0, now_epoch - parse_utc(last_utc)), 3)
        except ValueError:
            gap = None
    return {"prev_boot_id": prev_boot, "prev_last_sample_utc": last_utc, "gap_s": gap}


def crash_delta(previous_names, names):
    """Returns (count, new_names). The first observation defines the baseline."""
    names = sorted(set(names))
    if previous_names is None:
        return len(names), []
    return len(names), sorted(set(names) - set(previous_names))


# ---------------------------------------------------------------- record helpers

def envelope(record_type, node_id, boot_id, seq, mono_s, utc):
    return {"schema": SCHEMA_TELEMETRY, "type": record_type, "node_id": node_id, "boot_id": boot_id,
            "seq": seq, "utc": utc, "mono_s": round(mono_s, 3)}


FAST_GPU_FIELDS = ("temp_c", "power_w", "util_pct", "mem_used_mib", "sm_mhz", "mem_mhz", "fan_pct", "pstate")


def fast_gpu_entries(snapshot, now_mono, stale_after_s=1.5):
    """GPU block of a fast record from a stream snapshot. Returns (gpus, reasons).

    ``snapshot`` = {"status", "reason", "gpus": {index: {"mono", "values", "reasons"}}}.
    """
    reasons = {}
    known = snapshot.get("gpus") or {}
    if not known:
        reasons["gpus"] = snapshot.get("reason") or "stale"
        return None, reasons
    gpus = []
    for position, index in enumerate(sorted(known)):
        item = known[index]
        age = max(0.0, now_mono - item["mono"])
        entry = {"index": index, "age_s": round(age, 3)}
        if age > stale_after_s:
            entry.update({field: None for field in FAST_GPU_FIELDS})
            reasons[f"gpus[{position}]"] = "stale"
        else:
            for field in FAST_GPU_FIELDS:
                entry[field] = item["values"].get(field)
                if entry[field] is None:
                    reasons[f"gpus[{position}].{field}"] = item["reasons"].get(field, "not_exposed")
        gpus.append(entry)
    return gpus, reasons


def read_package_temps(readers, paths):
    """Fast tier CPU package temperatures from cached hwmon paths. Returns (list, reasons)."""
    if not paths:
        return None, {"cpu_pkg": "not_exposed"}
    packages, reasons = [], {}
    for position, (package, path, _chip) in enumerate(paths):
        try:
            packages.append({"package": package, "temp_c": _milli(readers.read(path))})
        except Exception as exc:  # noqa: BLE001
            packages.append({"package": package, "temp_c": None})
            reasons[f"cpu_pkg[{position}].temp_c"] = file_reason(exc)
    return packages, reasons


class SlowCollector:
    """Builds the 10 s record sections. Every section is isolated: an arbitrary
    exception becomes ``None`` + ``collector_error`` and the others still fill."""

    def __init__(self, readers, nvidia_smi=("nvidia-smi",), disks=None, reserves=None, cmd_timeout_s=4.0,
                 containers=False, rapl_max_plausible_w=DEFAULT_RAPL_MAX_PLAUSIBLE_W, gpu_enabled=True):
        self.readers = readers
        self.nvidia_smi = list(nvidia_smi)
        self.disks = dict(disks or {})
        self.reserves = dict(reserves or {})
        self.cmd_timeout_s = cmd_timeout_s
        self.containers = containers
        self.gpu_enabled = gpu_enabled
        self.rapl_max_plausible_w = rapl_max_plausible_w
        self.prev_stat = None
        self.prev_rapl = {}
        self.prev_mono = None
        self.crash_names = None
        self.errors = []

    def _section(self, name, function, reasons, default=None):
        try:
            return function(reasons)
        except Exception as exc:  # noqa: BLE001 - fault isolation is the contract
            self.errors.append((name, type(exc).__name__))
            reasons[name] = "collector_error"
            return default

    def collect(self, mono_s, uuids=None):
        """Returns (sections, null_reasons, events, src). ``uuids`` maps GPU index -> UUID."""
        reasons, events = {}, []
        dt_s = None if self.prev_mono is None else round(mono_s - self.prev_mono, 3)
        if dt_s is None:
            reasons["dt_s"] = "first_sample"
        sections = {"dt_s": dt_s}
        stat = self._section("btime", lambda r: self._stat(r), reasons)
        sections["btime"] = stat["btime"] if stat else None
        if stat is None and "btime" not in reasons:
            reasons["btime"] = "not_exposed"
        sections["uptime_s"] = self._section("uptime_s", self._uptime, reasons)
        sections["cpu"] = self._section("cpu", lambda r: self._cpu(r, stat), reasons)
        sections["memory"] = self._section("memory", self._memory, reasons)
        sections["disks"] = self._section("disks", self._disks, reasons)
        sections["rapl"] = self._section("rapl", lambda r: self._rapl(r, dt_s, events), reasons)
        sections["fans"] = self._section("fans", self._fans, reasons)
        sections["gpus"] = self._section("gpus", lambda r: self._gpus(r, uuids or {}), reasons)
        sections["thermal_zones"] = self._section("thermal_zones", self._zones, reasons)
        sections["crash"] = self._section("crash", lambda r: self._crash(r, events), reasons)
        sections["containers"] = self._section("containers", self._containers, reasons)
        sections["kernel_xid"] = None
        reasons["kernel_xid"] = "disabled_by_config"
        self.prev_mono = mono_s
        for key in ("uptime_s", "cpu", "memory", "disks", "rapl", "fans", "gpus", "thermal_zones", "crash",
                    "containers"):
            if sections[key] is None:
                reasons.setdefault(key, "not_exposed")
        src = {"cpu_temp": "hwmon", "rapl": "sysfs:powercap", "gpu": "nvidia-smi-query", "proc": "procfs"}
        return sections, reasons, events, src

    def _stat(self, reasons):
        try:
            return parse_proc_stat(self.readers.read(self.readers.proc("stat")))
        except Exception as exc:  # noqa: BLE001
            reasons["btime"] = file_reason(exc)
            return None

    def _uptime(self, reasons):
        try:
            return parse_uptime(self.readers.read(self.readers.proc("uptime")))
        except Exception as exc:  # noqa: BLE001
            reasons["uptime_s"] = file_reason(exc)
            return None

    def _cpu(self, reasons, stat):
        cpu = {"util_pct": None, "load_1_5_15": None, "freq_mhz": None, "pkg_temps_c": None,
               "core_temps_c": None, "pkg_temp_limits_c": None}
        if stat is None:
            reasons["cpu.util_pct"] = reasons.get("btime", "not_exposed")
        else:
            cpu["util_pct"] = cpu_util_pct(self.prev_stat, stat)
            if cpu["util_pct"] is None:
                reasons["cpu.util_pct"] = "first_sample" if self.prev_stat is None else "stale"
            self.prev_stat = stat
        try:
            cpu["load_1_5_15"] = parse_loadavg(self.readers.read(self.readers.proc("loadavg")))
        except Exception as exc:  # noqa: BLE001
            reasons["cpu.load_1_5_15"] = file_reason(exc)
        try:
            cpu["freq_mhz"] = read_cpu_freq_mhz(self.readers)
        except Exception as exc:  # noqa: BLE001
            reasons["cpu.freq_mhz"] = file_reason(exc)
        try:
            chips = read_hwmon(self.readers)
            normalized = cpu_from_hwmon(chips)
            failed = hwmon_temp_reasons(chips)
            if normalized["packages"]:
                cpu["pkg_temps_c"] = [{"package": item["package"], "temp_c": item["temp_c"]}
                                      for item in normalized["packages"]]
                for position, item in enumerate(cpu["pkg_temps_c"]):
                    if item["temp_c"] is None:
                        reasons[f"cpu.pkg_temps_c[{position}].temp_c"] = failed.get(("package", item["package"]),
                                                                                    "not_exposed")
                cpu["pkg_temp_limits_c"] = {"high": normalized["limits"]["high_c"],
                                            "crit": normalized["limits"]["crit_c"]}
                for key in ("high", "crit"):
                    if cpu["pkg_temp_limits_c"][key] is None:
                        reasons[f"cpu.pkg_temp_limits_c.{key}"] = "not_exposed"
            if normalized["cores"]:
                cpu["core_temps_c"] = normalized["cores"]
                for position, item in enumerate(cpu["core_temps_c"]):
                    if item["temp_c"] is None:
                        reasons[f"cpu.core_temps_c[{position}].temp_c"] = failed.get(("core", item["core"]),
                                                                                     "not_exposed")
        except Exception as exc:  # noqa: BLE001
            reasons["cpu.pkg_temps_c"] = reasons["cpu.core_temps_c"] = file_reason(exc)
        for key in ("pkg_temps_c", "core_temps_c", "pkg_temp_limits_c"):
            if cpu[key] is None:
                reasons.setdefault(f"cpu.{key}", "not_exposed")
        return cpu

    def _memory(self, reasons):
        try:
            memory = parse_meminfo(self.readers.read(self.readers.proc("meminfo")))
        except Exception as exc:  # noqa: BLE001
            reasons["memory"] = file_reason(exc)
            return None
        for key, value in memory.items():
            if value is None:
                reasons[f"memory.{key}"] = "not_exposed"
        return memory

    def _disks(self, reasons):
        disks = []
        for position, key in enumerate(sorted(self.disks)):
            entry = {"key": key, "total_bytes": None, "free_bytes": None,
                     "reserve_bytes": self.reserves.get(key)}
            try:
                usage = self.readers.disk_usage(self.disks[key])
                entry["total_bytes"], entry["free_bytes"] = int(usage.total), int(usage.free)
            except Exception as exc:  # noqa: BLE001
                reasons[f"disks[{position}].total_bytes"] = reasons[f"disks[{position}].free_bytes"] = file_reason(exc)
            if entry["reserve_bytes"] is None:
                reasons[f"disks[{position}].reserve_bytes"] = "disabled_by_config"
            disks.append(entry)
        if not disks:
            reasons["disks"] = "disabled_by_config"
            return None
        return disks

    def _rapl(self, reasons, dt_s, events):
        try:
            zones, zone_reasons = read_rapl(self.readers)
        except Exception as exc:  # noqa: BLE001
            reasons["rapl"] = file_reason(exc)
            return None
        if not zones:
            reasons["rapl"] = "not_exposed"
            return None
        reasons.update(zone_reasons)
        for position, zone in enumerate(zones):
            previous = self.prev_rapl.get(zone["zone"])
            delta, event = rapl_delta(previous, zone["energy_uj"], zone["max_range_uj"], dt_s,
                                      self.rapl_max_plausible_w)
            zone["delta_uj"] = delta
            if event:
                events.append({"event": event, "domain": zone["domain"], "zone": zone["zone"], "dt_s": dt_s})
            if delta is None:
                if zone["energy_uj"] is None:
                    reasons[f"rapl[{position}].delta_uj"] = zone_reasons.get(f"rapl[{position}].energy_uj",
                                                                              "not_exposed")
                elif event == "rapl_suspect_multi_wrap":
                    reasons[f"rapl[{position}].delta_uj"] = "stale"
                else:
                    reasons[f"rapl[{position}].delta_uj"] = "first_sample"
            if zone["energy_uj"] is not None:
                self.prev_rapl[zone["zone"]] = zone["energy_uj"]
        return zones

    def _fans(self, reasons):
        fans, reason = fans_from_hwmon(read_hwmon(self.readers))
        if fans is None:
            reasons["fans"] = reason
            return None
        for position, fan in enumerate(fans):
            for key in ("rpm", "pwm", "pwm_pct", "pwm_enable"):
                if fan[key] is None:
                    reasons[f"fans[{position}].{key}"] = "not_exposed"
        return fans

    def _gpus(self, reasons, uuids):
        if not self.gpu_enabled:
            reasons["gpus"] = "disabled_by_config"
            return None
        timeout = self.cmd_timeout_s
        try:
            q = parse_nvidia_q(self.readers.run_cmd(self.nvidia_smi + ["-q", "-d", "PERFORMANCE,POWER,TEMPERATURE,ECC"],
                                                    timeout))
        except Exception as exc:  # noqa: BLE001
            reasons["gpus"] = command_reason(exc)
            return None
        pcie, pcie_reason = {}, None
        try:
            rows, _ = parse_nvidia_csv(self.readers.run_cmd(nvidia_query_argv(self.nvidia_smi, PCIE_QUERY_FIELDS),
                                                            timeout), PCIE_QUERY_FIELDS)
            pcie = {row["index"]: row for row in rows}
        except Exception as exc:  # noqa: BLE001
            pcie_reason = command_reason(exc)
        apps, apps_reason = {}, None
        if uuids:
            try:
                apps = count_compute_apps(self.readers.run_cmd(
                    self.nvidia_smi + ["--query-compute-apps=gpu_uuid", "--format=csv,noheader"], timeout), uuids)
            except Exception as exc:  # noqa: BLE001
                apps_reason = command_reason(exc)
        else:
            apps_reason = "stale"
        gpus = []
        for position, item in enumerate(q):
            index = item["index"]
            entry = {"index": index, "pstate": item["pstate"], "power_limit_w": item["power_limit_w"],
                     "clock_event_reasons": item["clock_event_reasons"], "ecc_errors": item["ecc_errors"],
                     "temp_slowdown_c": item["temp_slowdown_c"], "temp_shutdown_c": item["temp_shutdown_c"],
                     "temp_target_c": item["temp_target_c"], "pcie_gen": None, "pcie_width": None,
                     "compute_apps": apps.get(index)}
            for key, reason in item["null_reasons"].items():
                reasons[f"gpus[{position}].{key}"] = reason
            row = pcie.get(index)
            for key in ("pcie_gen", "pcie_width"):
                if row is not None:
                    entry[key] = row["values"].get(key)
                if entry[key] is None:
                    reasons[f"gpus[{position}].{key}"] = (row or {}).get("reasons", {}).get(key) or pcie_reason or "not_exposed"
            if entry["compute_apps"] is None:
                reasons[f"gpus[{position}].compute_apps"] = apps_reason or "not_exposed"
            for key in ("pstate", "power_limit_w", "temp_slowdown_c", "temp_shutdown_c", "temp_target_c"):
                if entry[key] is None:
                    reasons.setdefault(f"gpus[{position}].{key}", "not_supported_by_device")
            gpus.append(entry)
        if not gpus:
            reasons["gpus"] = "parse_error"
            return None
        return gpus

    def _zones(self, reasons):
        zones = read_thermal_zones(self.readers)
        if not zones:
            reasons["thermal_zones"] = "not_exposed"
            return None
        return [{"type": zone["type"], "temp_c": zone["temp_c"]} for zone in zones]

    def _crash(self, reasons, events):
        try:
            names = [name for name in self.readers.listdir(self.readers.crash_dir) if not name.startswith(".")]
        except Exception as exc:  # noqa: BLE001
            reasons["crash"] = file_reason(exc)
            return None
        count, new = crash_delta(self.crash_names, names)
        self.crash_names = sorted(set(names))
        if new:
            events.append({"event": "var_crash_new", "names": new, "count": len(new)})
        return {"var_crash_count": count, "var_crash_new": len(new)}

    def _containers(self, reasons):
        if not self.containers:
            reasons["containers"] = "disabled_by_config"
            return None
        try:
            text = self.readers.run_cmd(["docker", "ps", "--all", "--filter", "name=atlas-",
                                         "--format", "{{.Names}}\t{{.State}}"], self.cmd_timeout_s)
        except Exception as exc:  # noqa: BLE001
            reasons["containers"] = command_reason(exc)
            return None
        result = []
        for line in text.splitlines():
            if "\t" in line:
                name, state = line.split("\t", 1)
                result.append({"name": name.strip(), "running": state.strip() == "running"})
        return result
