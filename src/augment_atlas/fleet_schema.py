"""Pure validators for the fleet protocol v1 contracts (docs/fleet-protocol.md).

Every function here is side-effect free: no I/O, no clock, no subprocess. Each
validator either returns the (unmodified) document or raises ``SchemaError``
with a dotted path. ``UnsupportedSchema`` is raised for a well-formed document
whose ``schema`` marker is unknown, so callers can skip instead of failing
(forward compatibility, section 3.2a).

Contracts covered:
- ``atlas.requirements.v1``      (spec.requirements, section 3.2a)
- claim ``fleet`` block           (section 3.2d)
- ``atlas.heartbeat.v1``          (section 4.2)
- ``atlas.node-profile.v1``       (section 6.2) and the public card (section 6.3)
- ``atlas.telemetry.v1`` records  (fast / slow / event, section 5.2)
- ``atlas.sentinel-latest.v1``    (section 5.6)
- ``atlas.job-telemetry-summary.v1`` and its public subset (sections 5.4 and 5.5)
"""
from __future__ import annotations

import math
import re

from .schema import canonical_json

# --------------------------------------------------------------------------- enums

#: Reason codes for ``null`` telemetry values (section 5.2). Closed set.
REASONS = frozenset({
    "not_exposed", "permission_denied", "tool_absent", "tool_timeout", "parse_error",
    "not_supported_by_device", "disabled_by_config", "stale", "collector_error", "first_sample",
})
#: Node profiles additionally mark values that a human must decide or that the
#: probe could not observe in a replayed capture.
PROFILE_REASONS = REASONS | {"requires_operator_decision", "not_captured"}
ENERGY_STATUSES = frozenset({"measured_counter", "measured_integrated", "estimated_model", "unavailable"})
MEASURED_ENERGY_STATUSES = frozenset({"measured_counter", "measured_integrated"})
HEARTBEAT_STATUSES = frozenset({"idle", "running", "draining", "blocked_host_loss", "blocked_manual"})
GAP_CLASSIFICATIONS = frozenset({"git_outage", "coordinator_down", "host_reboot", "host_freeze", "node_lost",
                                 "unknown_observer", "unclassified"})
NODE_STATUSES = frozenset({"active", "draining", "retired"})
SENSOR_STATES = frozenset({"available", "unverified"})
EVENT_NAMES = frozenset({
    "collector_start", "collector_stop", "reboot_detected", "sample_gap", "gpu_stream_gap",
    "sensor_state_change", "rapl_wrap", "rapl_suspect_multi_wrap", "var_crash_new", "threshold_observed",
})
TELEMETRY_TYPES = frozenset({"fast", "slow", "event"})
INTERRUPTED_CAUSES = frozenset({"operator_stop", "thermal_preventive", "external_stop", "deadline",
                                "disk_reserve", "host_reboot"})
FAILED_CAUSES = frozenset({"nonzero_exit", "signal", "oom_killed", "training_error", "artifact_invalid",
                           "checkpoint_invalid", "launch_error", "host_reboot_checkpoint_invalid", "node_lost"})
#: Keys that the queue reserves inside history events (queue.GitQueue.update).
RESERVED_EVENT_KEYS = frozenset({"seq", "at", "attempt_id", "token", "payload_hash", "kind"})

SCHEMA_REQUIREMENTS = "atlas.requirements.v1"
SCHEMA_HEARTBEAT = "atlas.heartbeat.v1"
SCHEMA_NODE_PROFILE = "atlas.node-profile.v1"
SCHEMA_TELEMETRY = "atlas.telemetry.v1"
SCHEMA_SENTINEL_LATEST = "atlas.sentinel-latest.v1"
SCHEMA_SENTINEL_STATE = "atlas.sentinel-state.v1"
SCHEMA_JOB_SUMMARY = "atlas.job-telemetry-summary.v1"
FLEET_PROTOCOL = "atlas.fleet.v1"

#: Fixed public whitelist of the terminal ``telemetry_v1`` event block (section 5.5).
PUBLIC_TELEMETRY_KEYS = frozenset({
    "schema", "node_id", "hardware_class", "duration_s", "coverage_fraction", "max_gap_s", "gpu_temp_max_c",
    "gpu_temp_p95_c", "cpu_pkg_temp_max_c", "gpu_power_mean_w", "energy", "co2", "throughput_img_s_mean",
    "epochs_completed", "reboot_detected",
})
PUBLIC_ENERGY_KEYS = ("gpu_kwh", "cpu_pkg_kwh", "dram_kwh")
PUBLIC_CO2_KEYS = frozenset({"g", "boundary", "grid_factor_g_per_kwh", "factor_source_id"})
#: Keys never accepted in a public document (model metrics are private, section 5.5).
PRIVATE_METRIC_KEYS = frozenset({"train_loss", "loss", "accuracy", "selection", "learning_rate", "auc",
                                 "sensitivity", "specificity", "var_crash_names", "names"})

# ------------------------------------------------------------------------ patterns

HEX64 = re.compile(r"^[a-f0-9]{64}$")
HEX40 = re.compile(r"^[a-f0-9]{40}$")
HEX32 = re.compile(r"^[a-f0-9]{32}$")
IMAGE_ID = re.compile(r"^sha256:[a-f0-9]{64}$")
IMAGE_REF = re.compile(r"^[a-z0-9][a-z0-9.-]*(?::[0-9]+)?/[A-Za-z0-9._/-]+@sha256:[a-f0-9]{64}$")
SLUG = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")
STORAGE_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
BOOT_ID = re.compile(r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|unavailable)$")
GPU_UUID = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
UTC_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
HARDWARE_CLASS = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")
HEARTBEAT_REF = re.compile(r"^refs/heads/fleet-hb/[a-z0-9-]+/[a-z0-9-]+$")
FILE_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_ABS_PATH = re.compile(r"(?:^|[\s\"'=(,])(?:/|~/)[A-Za-z0-9._~-]|^[A-Za-z]:[\\/]|\\\\|[A-Za-z]:\\")
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_AT = re.compile(r"[^\s@]+@[^\s@]+")
_IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
_IPV6 = re.compile(r"(?<![0-9A-Za-z:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Za-z:])")
_SECRET = re.compile(r"-----BEGIN|\bghp_[A-Za-z0-9]{16,}|\bgithub_pat_|\bgho_[A-Za-z0-9]{16,}|\bglpat-|"
                     r"\bAKIA[0-9A-Z]{16}\b|\bssh-(?:rsa|ed25519|dss)\b|\bxox[abprs]-|\bsk-[A-Za-z0-9]{20,}")
_FORBIDDEN_KEY = re.compile(r"(?:^|_)(?:token|password|passwd|secret|api_?key|private_?key|hostname|"
                            r"ip_?address|remote|credential)s?(?:$|_)", re.IGNORECASE)
#: Keys whose dotted numeric values are software versions, not IPv4 addresses.
VERSION_KEYS = frozenset({"torch", "torchvision", "cuda", "cuda_runtime", "driver", "nvidia_driver", "kernel",
                          "docker", "nvidia_container_toolkit", "opencv", "os", "version", "python"})


class SchemaError(ValueError):
    """A document does not satisfy its v1 contract."""


class UnsupportedSchema(SchemaError):
    """The ``schema`` marker is well formed but unknown to this code (skip, do not fail)."""


# ------------------------------------------------------------------------- helpers

def _fail(path, message):
    raise SchemaError(f"{path or '<root>'}: {message}")


def _obj(value, path, required=(), optional=(), allow_extra=False):
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    missing = [key for key in required if key not in value]
    if missing:
        _fail(path, f"missing keys {sorted(missing)}")
    if not allow_extra:
        extra = sorted(set(value) - set(required) - set(optional))
        if extra:
            _fail(path, f"unknown keys {extra}")
    return value


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _num(value, path, minimum=None, maximum=None, exclusive_min=False, nullable=False):
    if value is None and nullable:
        return None
    if not is_number(value):
        _fail(path, "must be a finite number")
    if minimum is not None and (value <= minimum if exclusive_min else value < minimum):
        _fail(path, f"must be {'>' if exclusive_min else '>='} {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be <= {maximum}")
    return value


def _int(value, path, minimum=None, nullable=False):
    if value is None and nullable:
        return None
    if type(value) is not int:
        _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _bool(value, path, nullable=False):
    if value is None and nullable:
        return None
    if type(value) is not bool:
        _fail(path, "must be a boolean")
    return value


def _str(value, path, pattern=None, nullable=False, choices=None):
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value:
        _fail(path, "must be a nonempty string")
    if pattern is not None and not pattern.fullmatch(value):
        _fail(path, f"invalid format {value!r}")
    if choices is not None and value not in choices:
        _fail(path, f"must be one of {sorted(choices)}")
    return value


def _list(value, path, nullable=False, nonempty=False):
    if value is None and nullable:
        return None
    if not isinstance(value, list):
        _fail(path, "must be a list")
    if nonempty and not value:
        _fail(path, "must not be empty")
    return value


def _schema_marker(doc, expected, path=""):
    if not isinstance(doc, dict):
        _fail(path, "must be an object")
    marker = doc.get("schema")
    if marker == expected:
        return
    family = expected.rsplit(".v", 1)[0] + ".v"
    if isinstance(marker, str) and marker.startswith(family):
        raise UnsupportedSchema(f"{path or '<root>'}: unsupported schema {marker!r}; this node understands {expected!r}")
    _fail(path + ".schema" if path else "schema", f"must be {expected!r}")


def _canonical(doc, path=""):
    try:
        canonical_json(doc)
    except (TypeError, ValueError) as exc:
        _fail(path, f"not canonical JSON (NaN/Inf or non-JSON type): {exc}")


def join_path(prefix, key):
    if isinstance(key, int):
        return f"{prefix}[{key}]"
    return f"{prefix}.{key}" if prefix else str(key)


def null_paths(value, prefix="", skip=frozenset()):
    """Dotted paths of every ``None`` leaf (``a.b[0].c``) below ``value``."""
    if value is None:
        return [prefix]
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if not prefix and key in skip:
                continue
            found.extend(null_paths(item, join_path(prefix, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(null_paths(item, join_path(prefix, index)))
    return found


def _path_ancestors(path):
    """``gpus[0].temp_c`` -> [``gpus[0].temp_c``, ``gpus[0]``, ``gpus``]."""
    result = [path]
    while True:
        cut = max(path.rfind("."), path.rfind("["))
        if cut <= 0:
            break
        path = path[:cut]
        result.append(path)
    return result


def uncovered_nulls(doc, reasons, skip=frozenset()):
    """Null leaves with no reason on the leaf itself or on any ancestor."""
    return [path for path in null_paths(doc, skip=skip)
            if not any(candidate in reasons for candidate in _path_ancestors(path))]


def _null_reasons(doc, path, allowed, skip, exempt_prefixes=()):
    reasons = doc.get("null_reasons")
    if not isinstance(reasons, dict):
        _fail(join_path(path, "null_reasons"), "must be an object")
    for key, reason in reasons.items():
        if not isinstance(key, str) or not key:
            _fail(join_path(path, "null_reasons"), "keys must be nonempty field paths")
        if reason not in allowed:
            _fail(join_path(path, f"null_reasons[{key!r}]"), f"unknown reason {reason!r}")
    missing = [item for item in uncovered_nulls(doc, reasons, skip=skip)
               if not any(item == prefix or item.startswith(prefix + ".") or item.startswith(prefix + "[")
                          for prefix in exempt_prefixes)]
    if missing:
        _fail(path, f"null values without null_reasons: {missing[:8]}")


# --------------------------------------------------------------- public-data scan

def forbidden_strings(value, path="", version_keys=VERSION_KEYS, allow_at_keys=frozenset()):
    """List ``(path, kind)`` findings of data that must never be public.

    Detects absolute/home/Windows paths, URLs, ``user@host`` forms, IPv4/IPv6
    literals, credential-looking material and credential-like key names.
    Version-like keys (``torch``, ``driver``...) skip only the IPv4 check.
    """
    findings = []
    if isinstance(value, dict):
        for key, item in value.items():
            here = join_path(path, key)
            if isinstance(key, str) and _FORBIDDEN_KEY.search(key):
                findings.append((here, "forbidden_key"))
            findings.extend(forbidden_strings(item, here, version_keys, allow_at_keys)
                            if not (isinstance(key, str) and key in version_keys and isinstance(item, str))
                            else _scan_string(item, here, check_ipv4=False, allow_at=key in allow_at_keys))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(forbidden_strings(item, join_path(path, index), version_keys, allow_at_keys))
    elif isinstance(value, str):
        last = path.rsplit(".", 1)[-1]
        findings.extend(_scan_string(value, path, check_ipv4=True, allow_at=last in allow_at_keys))
    return findings


def _scan_string(text, path, check_ipv4=True, allow_at=False):
    findings = []
    if _ABS_PATH.search(text):
        findings.append((path, "path"))
    if _URL.search(text):
        findings.append((path, "url"))
    if not allow_at and _AT.search(text):
        findings.append((path, "at_sign"))
    if check_ipv4 and _IPV4.search(text):
        findings.append((path, "ipv4"))
    for match in _IPV6.finditer(text):
        token = match.group(0)
        if "::" in token or re.search(r"[A-Fa-f]", token):
            findings.append((path, "ipv6"))
            break
    if _SECRET.search(text):
        findings.append((path, "secret"))
    return findings


def assert_public(value, path="", allow_at_keys=frozenset()):
    findings = forbidden_strings(value, path, allow_at_keys=allow_at_keys)
    if findings:
        _fail(path, f"forbidden public content: {findings[:8]}")
    return value


# ------------------------------------------------------------------- requirements

REQUIREMENT_KEYS = ("schema", "gpu_required", "min_vram_mib", "min_compute_capability", "hardware_class",
                    "allowed_nodes", "image_id", "image_ref", "dataset", "weights_sha256",
                    "min_free_disk_bytes", "container", "expected_epoch_seconds_max")
REQUIRED_REQUIREMENT_KEYS = ("schema", "gpu_required", "min_vram_mib", "image_id", "dataset")


def _capability(value, path, nullable=True):
    if value is None and nullable:
        return None
    if (not isinstance(value, list) or len(value) != 2 or any(type(part) is not int or part < 0 for part in value)):
        _fail(path, "must be [major, minor] nonnegative integers")
    return value


def validate_requirements(req):
    """``spec.requirements`` v1. Unknown keys are rejected: a v1 node must not
    silently ignore a constraint it does not understand (it skips the job)."""
    _schema_marker(req, SCHEMA_REQUIREMENTS, "requirements")
    _obj(req, "requirements", REQUIRED_REQUIREMENT_KEYS, REQUIREMENT_KEYS)
    _bool(req["gpu_required"], "requirements.gpu_required")
    _int(req["min_vram_mib"], "requirements.min_vram_mib", 0)
    _capability(req.get("min_compute_capability"), "requirements.min_compute_capability")
    _str(req.get("hardware_class"), "requirements.hardware_class", HARDWARE_CLASS, nullable=True)
    nodes = _list(req.get("allowed_nodes"), "requirements.allowed_nodes", nullable=True, nonempty=True)
    if nodes is not None:
        for index, node in enumerate(nodes):
            _str(node, f"requirements.allowed_nodes[{index}]", SLUG)
        if len(set(nodes)) != len(nodes):
            _fail("requirements.allowed_nodes", "duplicate node IDs")
    _str(req["image_id"], "requirements.image_id", IMAGE_ID)
    _str(req.get("image_ref"), "requirements.image_ref", IMAGE_REF, nullable=True)
    dataset = _obj(req["dataset"], "requirements.dataset", ("id", "version", "manifest_sha256"))
    _str(dataset["id"], "requirements.dataset.id")
    _str(dataset["version"], "requirements.dataset.version")
    _str(dataset["manifest_sha256"], "requirements.dataset.manifest_sha256", HEX64, nullable=True)
    _str(req.get("weights_sha256"), "requirements.weights_sha256", HEX64, nullable=True)
    _int(req.get("min_free_disk_bytes", 0), "requirements.min_free_disk_bytes", 0)
    container = req.get("container")
    if container is not None:
        _obj(container, "requirements.container", ("cpus", "memory_bytes"))
        _num(container["cpus"], "requirements.container.cpus", 0, exclusive_min=True)
        _int(container["memory_bytes"], "requirements.container.memory_bytes", 1)
    _int(req.get("expected_epoch_seconds_max"), "requirements.expected_epoch_seconds_max", 1, nullable=True)
    assert_public({key: value for key, value in req.items() if key != "image_ref"}, "requirements")
    _canonical(req, "requirements")
    return req


def validate_scheduling(value):
    _obj(value, "scheduling", (), ("priority", "order"))
    _int(value.get("priority", 0), "scheduling.priority")
    _int(value.get("order"), "scheduling.order", nullable=True)
    return value


# ------------------------------------------------------------------ claim block

CLAIM_FLEET_KEYS = ("protocol", "node_profile_sha256", "coordinator_commit", "boot_id", "image_id",
                    "launch_plan_sha256", "requirements_sha256", "lease_seconds", "heartbeat_ref", "gate")


def validate_claim_fleet(block):
    """``history[].fleet`` of a claim event (section 3.2d)."""
    if not isinstance(block, dict):
        _fail("fleet", "must be an object")
    reserved = sorted(RESERVED_EVENT_KEYS & set(block))
    if reserved:
        _fail("fleet", f"uses reserved queue keys {reserved}")
    _obj(block, "fleet", CLAIM_FLEET_KEYS)
    _str(block["protocol"], "fleet.protocol", choices={FLEET_PROTOCOL})
    _str(block["node_profile_sha256"], "fleet.node_profile_sha256", HEX64)
    _str(block["coordinator_commit"], "fleet.coordinator_commit", HEX40)
    _str(block["boot_id"], "fleet.boot_id", BOOT_ID)
    _str(block["image_id"], "fleet.image_id", IMAGE_ID)
    _str(block["launch_plan_sha256"], "fleet.launch_plan_sha256", HEX64)
    _str(block["requirements_sha256"], "fleet.requirements_sha256", HEX64)
    _int(block["lease_seconds"], "fleet.lease_seconds", 1)
    _str(block["heartbeat_ref"], "fleet.heartbeat_ref", HEARTBEAT_REF)
    gate = _obj(block["gate"], "fleet.gate", ("observed_at_utc",), allow_extra=True)
    _str(gate["observed_at_utc"], "fleet.gate.observed_at_utc", UTC_Z)
    for key, value in gate.items():
        if key != "observed_at_utc" and not (value is None or is_number(value) or type(value) is bool):
            _fail(f"fleet.gate.{key}", "gate values must be numbers, booleans or null")
    assert_public(block, "fleet")
    _canonical(block, "fleet")
    return block


# --------------------------------------------------------------------- heartbeat

HEARTBEAT_KEYS = ("schema", "node_id", "worker", "seq", "observed_at_utc", "boot_id", "btime", "uptime_s",
                  "ntp_synchronized", "coordinator_commit", "node_profile_sha256", "status", "active",
                  "sentinel_last_sample_age_s", "spool_pending_events", "host_loss_detected", "gap_report")


def validate_heartbeat(doc):
    _schema_marker(doc, SCHEMA_HEARTBEAT, "heartbeat")
    _obj(doc, "heartbeat", HEARTBEAT_KEYS)
    _str(doc["node_id"], "heartbeat.node_id", SLUG)
    _str(doc["worker"], "heartbeat.worker", SLUG)
    _int(doc["seq"], "heartbeat.seq", 0)
    _str(doc["observed_at_utc"], "heartbeat.observed_at_utc", UTC_Z)
    _str(doc["boot_id"], "heartbeat.boot_id", BOOT_ID)
    _int(doc["btime"], "heartbeat.btime", 0, nullable=True)
    _num(doc["uptime_s"], "heartbeat.uptime_s", 0, nullable=True)
    _bool(doc["ntp_synchronized"], "heartbeat.ntp_synchronized", nullable=True)
    _str(doc["coordinator_commit"], "heartbeat.coordinator_commit", HEX40)
    _str(doc["node_profile_sha256"], "heartbeat.node_profile_sha256", HEX64, nullable=True)
    _str(doc["status"], "heartbeat.status", choices=HEARTBEAT_STATUSES)
    for index, item in enumerate(_list(doc["active"], "heartbeat.active")):
        here = f"heartbeat.active[{index}]"
        _obj(item, here, ("job_id", "attempt_id", "container"), ("epoch", "progress_age_s"))
        _str(item["job_id"], here + ".job_id", HEX64)
        _str(item["attempt_id"], here + ".attempt_id", HEX32)
        _str(item["container"], here + ".container",
             choices={"created", "running", "exited", "missing", "paused", "restarting", "dead", "unknown"})
        _int(item.get("epoch"), here + ".epoch", 0, nullable=True)
        _num(item.get("progress_age_s"), here + ".progress_age_s", 0, nullable=True)
    _num(doc["sentinel_last_sample_age_s"], "heartbeat.sentinel_last_sample_age_s", 0, nullable=True)
    _int(doc["spool_pending_events"], "heartbeat.spool_pending_events", 0)
    for index, item in enumerate(_list(doc["host_loss_detected"], "heartbeat.host_loss_detected")):
        here = f"heartbeat.host_loss_detected[{index}]"
        _obj(item, here, ("job_id", "attempt_id"), allow_extra=True)
        _str(item["job_id"], here + ".job_id", HEX64)
        _str(item["attempt_id"], here + ".attempt_id", HEX32)
    if doc["status"] == "blocked_host_loss" and not doc["host_loss_detected"]:
        _fail("heartbeat.host_loss_detected", "blocked_host_loss requires at least one attempt")
    gap = doc["gap_report"]
    if gap is not None:
        _obj(gap, "heartbeat.gap_report", ("gap_start_utc", "gap_end_utc", "classification"), allow_extra=True)
        _str(gap["gap_start_utc"], "heartbeat.gap_report.gap_start_utc", UTC_Z)
        _str(gap["gap_end_utc"], "heartbeat.gap_report.gap_end_utc", UTC_Z)
        _str(gap["classification"], "heartbeat.gap_report.classification", choices=GAP_CLASSIFICATIONS)
    assert_public(doc, "heartbeat")
    _canonical(doc, "heartbeat")
    return doc


# ------------------------------------------------------------------ node profile

PROFILE_KEYS = ("schema", "node_id", "status", "hardware_class", "hardware", "storage", "limits", "guardian",
                "liveness", "datasets", "images", "weights", "sensors", "grid", "updated_at_utc")
PROFILE_OPTIONAL_KEYS = ("null_reasons",)
LIMIT_KEYS = ("max_concurrent_trainings", "allowed_gpus", "container", "thermal",
              "min_idle_seconds_between_jobs", "training_retry", "forbidden_host_changes")
GRID_KEYS = ("g_co2_per_kwh", "source", "region", "reference_period", "retrieved_on")
#: Decisions that may stay ``null`` even in a claim-ready profile: the grid
#: factor only gates CO2 reporting and the CPU TDP only feeds an optional estimate.
OPTIONAL_OPERATOR_DECISIONS = ("grid", "hardware.cpu.tdp_w")
PROFILE_NULL_EXEMPT = ("datasets", "images", "weights", "hardware.thermal_reference")


def _validate_thermal_limit(item, path):
    _obj(item, path, ("sensor", "ceiling_c", "preventive_c", "claim_max_c", "source"))
    _str(item["sensor"], path + ".sensor")
    _str(item["source"], path + ".source")
    ceiling = _num(item["ceiling_c"], path + ".ceiling_c")
    preventive = _num(item["preventive_c"], path + ".preventive_c")
    claim_max = _num(item["claim_max_c"], path + ".claim_max_c")
    if not 0 < preventive < ceiling:
        _fail(path, "requires 0 < preventive_c < ceiling_c")
    if not claim_max < preventive:
        _fail(path, "requires claim_max_c < preventive_c")


def _validate_hardware(hw):
    _obj(hw, "hardware", ("cpu", "ram_bytes", "gpus", "software"), ("thermal_reference",))
    cpu = _obj(hw["cpu"], "hardware.cpu", ("model", "sockets", "cores", "threads", "tdp_w"))
    _str(cpu["model"], "hardware.cpu.model", nullable=True)
    for key in ("sockets", "cores", "threads"):
        _int(cpu[key], f"hardware.cpu.{key}", 1, nullable=True)
    _num(cpu["tdp_w"], "hardware.cpu.tdp_w", 0, exclusive_min=True, nullable=True)
    _int(hw["ram_bytes"], "hardware.ram_bytes", 1, nullable=True)
    indices = []
    for index, gpu in enumerate(_list(hw["gpus"], "hardware.gpus")):
        here = f"hardware.gpus[{index}]"
        _obj(gpu, here, ("index", "name", "vram_mib", "compute_capability"), ("uuid",))
        indices.append(_int(gpu["index"], here + ".index", 0))
        _str(gpu["name"], here + ".name", nullable=True)
        _int(gpu["vram_mib"], here + ".vram_mib", 1, nullable=True)
        _capability(gpu["compute_capability"], here + ".compute_capability")
        _str(gpu.get("uuid"), here + ".uuid", GPU_UUID, nullable=True)
    if len(set(indices)) != len(indices):
        _fail("hardware.gpus", "duplicate GPU index")
    software = _obj(hw["software"], "hardware.software",
                    ("os", "kernel", "nvidia_driver", "docker", "nvidia_container_toolkit"))
    for key, value in software.items():
        _str(value, f"hardware.software.{key}", nullable=True)
    reference = hw.get("thermal_reference")
    if reference is not None:
        _obj(reference, "hardware.thermal_reference", (), ("cpu_pkg", "gpus"))
        for key in ("cpu_pkg", "gpus"):
            for index, item in enumerate(_list(reference.get(key, []), f"hardware.thermal_reference.{key}")):
                here = f"hardware.thermal_reference.{key}[{index}]"
                _obj(item, here, ("source",), allow_extra=True)
                for field, value in item.items():
                    if field.endswith("_c"):
                        _num(value, f"{here}.{field}", nullable=True)
    return set(indices)


def validate_node_profile(profile, complete=False):
    """Structural and coherence rules of section 6.2.

    ``complete=True`` additionally refuses pending operator decisions
    (``requires_operator_decision``), except the grid factor and CPU TDP, and
    refuses ``null`` policy blocks. Probe output is valid but not complete.
    """
    _schema_marker(profile, SCHEMA_NODE_PROFILE, "profile")
    _obj(profile, "profile", PROFILE_KEYS, PROFILE_OPTIONAL_KEYS)
    _str(profile["node_id"], "profile.node_id", SLUG)
    _str(profile["status"], "profile.status", choices=NODE_STATUSES)
    _str(profile["hardware_class"], "profile.hardware_class", HARDWARE_CLASS, nullable=True)
    gpu_indices = _validate_hardware(profile["hardware"])
    keys = set()
    for index, item in enumerate(_list(profile["storage"], "profile.storage")):
        here = f"profile.storage[{index}]"
        _obj(item, here, ("key", "reserve_bytes"))
        keys.add(_str(item["key"], here + ".key", STORAGE_KEY))
        _int(item["reserve_bytes"], here + ".reserve_bytes", 0, nullable=True)
    if len(keys) != len(profile["storage"]):
        _fail("profile.storage", "duplicate storage key")
    limits = _obj(profile["limits"], "profile.limits", LIMIT_KEYS)
    _int(limits["max_concurrent_trainings"], "profile.limits.max_concurrent_trainings", 1, nullable=True)
    allowed = _list(limits["allowed_gpus"], "profile.limits.allowed_gpus", nullable=True)
    if allowed is not None:
        for index, gpu in enumerate(allowed):
            _int(gpu, f"profile.limits.allowed_gpus[{index}]", 0)
        if not set(allowed) <= gpu_indices:
            _fail("profile.limits.allowed_gpus", "must be a subset of hardware.gpus indices")
        if len(set(allowed)) != len(allowed):
            _fail("profile.limits.allowed_gpus", "duplicate GPU index")
    container = limits["container"]
    if container is not None:
        _obj(container, "profile.limits.container", ("cpus", "memory_bytes", "shm_bytes"))
        _num(container["cpus"], "profile.limits.container.cpus", 0, exclusive_min=True)
        _int(container["memory_bytes"], "profile.limits.container.memory_bytes", 1)
        _int(container["shm_bytes"], "profile.limits.container.shm_bytes", 1)
    thermal = _list(limits["thermal"], "profile.limits.thermal", nullable=True)
    for index, item in enumerate(thermal or []):
        _validate_thermal_limit(item, f"profile.limits.thermal[{index}]")
    _int(limits["min_idle_seconds_between_jobs"], "profile.limits.min_idle_seconds_between_jobs", 0, nullable=True)
    if limits["training_retry"] != "never":
        _fail("profile.limits.training_retry", "only 'never' is accepted in v1")
    for index, item in enumerate(_list(limits["forbidden_host_changes"], "profile.limits.forbidden_host_changes")):
        _str(item, f"profile.limits.forbidden_host_changes[{index}]")
    guardian = profile["guardian"]
    if guardian is not None:
        _obj(guardian, "profile.guardian", ("kind", "required_for_claim"), ("review_ref", "declared_thresholds"))
        _str(guardian["kind"], "profile.guardian.kind")
        _bool(guardian["required_for_claim"], "profile.guardian.required_for_claim")
    liveness = _obj(profile["liveness"], "profile.liveness", ("heartbeat_seconds", "lease_seconds", "ntp_required"))
    heartbeat = _int(liveness["heartbeat_seconds"], "profile.liveness.heartbeat_seconds", 1)
    lease = _int(liveness["lease_seconds"], "profile.liveness.lease_seconds", 1)
    _bool(liveness["ntp_required"], "profile.liveness.ntp_required")
    if lease < 2 * heartbeat:
        _fail("profile.liveness", "requires lease_seconds >= 2 * heartbeat_seconds")
    for index, item in enumerate(_list(profile["datasets"], "profile.datasets")):
        here = f"profile.datasets[{index}]"
        _obj(item, here, ("id", "version", "manifest_sha256"), allow_extra=True)
        _str(item["id"], here + ".id")
        _str(item["version"], here + ".version")
        _str(item["manifest_sha256"], here + ".manifest_sha256", HEX64, nullable=True)
        if item.get("storage_key") is not None and item["storage_key"] not in keys:
            _fail(here + ".storage_key", "must name a storage key")
    for index, item in enumerate(_list(profile["images"], "profile.images")):
        here = f"profile.images[{index}]"
        _obj(item, here, ("image_id",), allow_extra=True)
        _str(item["image_id"], here + ".image_id", IMAGE_ID)
    for index, item in enumerate(_list(profile["weights"], "profile.weights")):
        here = f"profile.weights[{index}]"
        _obj(item, here, ("file", "sha256"))
        _str(item["file"], here + ".file", FILE_BASENAME)
        _str(item["sha256"], here + ".sha256", HEX64)
    sensors = _obj(profile["sensors"], "profile.sensors", (), allow_extra=True)
    for key, value in sensors.items():
        here = f"profile.sensors[{key!r}]"
        if key == "nvidia_smi_fields":
            for index, item in enumerate(_list(value, here)):
                _str(item, f"{here}[{index}]")
        elif isinstance(value, str):
            _str(value, here, choices=SENSOR_STATES)
        else:
            _obj(value, here, ("status", "reason"), ("evidence",))
            _str(value["status"], here + ".status", choices={"unavailable"})
            _str(value["reason"], here + ".reason", choices=PROFILE_REASONS)
    grid = _obj(profile["grid"], "profile.grid", GRID_KEYS)
    factor = _num(grid["g_co2_per_kwh"], "profile.grid.g_co2_per_kwh", 0, exclusive_min=True, nullable=True)
    for key in ("source", "region", "reference_period", "retrieved_on"):
        _str(grid[key], f"profile.grid.{key}", nullable=True)
    if factor is not None and (grid["source"] is None or grid["reference_period"] is None):
        _fail("profile.grid", "a grid factor requires source and reference_period")
    _str(profile["updated_at_utc"], "profile.updated_at_utc", UTC_Z)
    reasons = profile.get("null_reasons", {})
    _null_reasons({**profile, "null_reasons": reasons}, "profile", PROFILE_REASONS, skip={"null_reasons"},
                  exempt_prefixes=PROFILE_NULL_EXEMPT)
    if complete:
        pending = sorted(key for key, reason in reasons.items() if reason == "requires_operator_decision"
                         and not any(key == item or key.startswith(item + ".") for item in OPTIONAL_OPERATOR_DECISIONS))
        if pending:
            _fail("profile", f"pending operator decisions: {pending}")
        for key in ("max_concurrent_trainings", "allowed_gpus", "container", "thermal",
                    "min_idle_seconds_between_jobs"):
            if limits[key] is None:
                _fail(f"profile.limits.{key}", "required in a complete profile")
        if guardian is None or profile["hardware_class"] is None:
            _fail("profile", "guardian and hardware_class are required in a complete profile")
        if any(item["reserve_bytes"] is None for item in profile["storage"]):
            _fail("profile.storage", "reserve_bytes are required in a complete profile")
    assert_public(profile, "profile")
    _canonical(profile, "profile")
    return profile


def validate_public_card(card):
    """``fleet/nodes/<node_id>.json``: the profile + ``profile_sha256`` + ``registered_at_utc``."""
    if not isinstance(card, dict):
        _fail("card", "must be an object")
    _str(card.get("profile_sha256"), "card.profile_sha256", HEX64)
    _str(card.get("registered_at_utc"), "card.registered_at_utc", UTC_Z)
    profile = {key: value for key, value in card.items() if key not in {"profile_sha256", "registered_at_utc"}}
    validate_node_profile(profile)
    return card


# --------------------------------------------------------------- telemetry v1

ENVELOPE_KEYS = ("schema", "type", "node_id", "boot_id", "seq", "utc", "mono_s")
FAST_KEYS = ENVELOPE_KEYS + ("dt_s", "gpus", "cpu_pkg", "null_reasons", "src")
SLOW_KEYS = ENVELOPE_KEYS + ("dt_s", "uptime_s", "btime", "cpu", "memory", "disks", "rapl", "fans", "gpus",
                             "thermal_zones", "crash", "containers", "kernel_xid", "null_reasons", "src")
EVENT_KEYS = ENVELOPE_KEYS + ("event", "null_reasons")
FAST_GPU_KEYS = ("index", "temp_c", "power_w", "util_pct", "mem_used_mib", "sm_mhz", "mem_mhz", "fan_pct",
                 "pstate", "age_s")


def _validate_envelope(rec, path):
    _schema_marker(rec, SCHEMA_TELEMETRY, path)
    _str(rec.get("type"), path + ".type", choices=TELEMETRY_TYPES)
    _str(rec.get("node_id"), path + ".node_id", SLUG)
    _str(rec.get("boot_id"), path + ".boot_id", BOOT_ID)
    _int(rec.get("seq"), path + ".seq", 0)
    _str(rec.get("utc"), path + ".utc", UTC_Z)
    _num(rec.get("mono_s"), path + ".mono_s", 0)


def _validate_fast(rec, path):
    _obj(rec, path, FAST_KEYS)
    _num(rec["dt_s"], path + ".dt_s", 0, nullable=True)
    gpus = _list(rec["gpus"], path + ".gpus", nullable=True)
    for index, gpu in enumerate(gpus or []):
        here = f"{path}.gpus[{index}]"
        _obj(gpu, here, FAST_GPU_KEYS)
        _int(gpu["index"], here + ".index", 0)
        for key in FAST_GPU_KEYS[1:-2]:
            _num(gpu[key], f"{here}.{key}", nullable=True)
        _str(gpu["pstate"], here + ".pstate", nullable=True)
        _num(gpu["age_s"], here + ".age_s", 0, nullable=True)
    packages = _list(rec["cpu_pkg"], path + ".cpu_pkg", nullable=True)
    for index, item in enumerate(packages or []):
        here = f"{path}.cpu_pkg[{index}]"
        _obj(item, here, ("package", "temp_c"))
        _int(item["package"], here + ".package", 0)
        _num(item["temp_c"], here + ".temp_c", nullable=True)
    _obj(rec["src"], path + ".src", (), allow_extra=True)


def _validate_slow(rec, path):
    _obj(rec, path, SLOW_KEYS)
    _num(rec["dt_s"], path + ".dt_s", 0, nullable=True)
    _num(rec["uptime_s"], path + ".uptime_s", 0, nullable=True)
    _int(rec["btime"], path + ".btime", 0, nullable=True)
    for key in ("cpu", "memory", "crash"):
        if rec[key] is not None:
            _obj(rec[key], f"{path}.{key}", (), allow_extra=True)
    for key in ("disks", "rapl", "fans", "gpus", "thermal_zones", "containers"):
        _list(rec[key], f"{path}.{key}", nullable=True)
    for index, zone in enumerate(rec["rapl"] or []):
        here = f"{path}.rapl[{index}]"
        _obj(zone, here, ("domain", "zone", "energy_uj", "max_range_uj", "delta_uj"))
        _int(zone["energy_uj"], here + ".energy_uj", 0, nullable=True)
        _int(zone["max_range_uj"], here + ".max_range_uj", 1, nullable=True)
        _int(zone["delta_uj"], here + ".delta_uj", 0, nullable=True)
    for index, disk in enumerate(rec["disks"] or []):
        here = f"{path}.disks[{index}]"
        _obj(disk, here, ("key", "total_bytes", "free_bytes"), ("reserve_bytes",))
        _str(disk["key"], here + ".key", STORAGE_KEY)
    _obj(rec["src"], path + ".src", (), allow_extra=True)


def _validate_event(rec, path):
    _obj(rec, path, EVENT_KEYS, allow_extra=True)
    _str(rec["event"], path + ".event", choices=EVENT_NAMES)
    if rec["event"] == "reboot_detected":
        _obj(rec, path, ("prev_boot_id", "prev_last_sample_utc", "gap_s"), allow_extra=True)
    if rec["event"] in {"sample_gap", "gpu_stream_gap"}:
        _num(rec.get("gap_s"), path + ".gap_s", 0)


def validate_telemetry_record(rec, path="record"):
    """One JSONL line of the sentinel (fast, slow or event)."""
    if not isinstance(rec, dict):
        _fail(path, "must be an object")
    _validate_envelope(rec, path)
    {"fast": _validate_fast, "slow": _validate_slow, "event": _validate_event}[rec["type"]](rec, path)
    _null_reasons(rec, path, REASONS, skip={"null_reasons", "src"})
    _canonical(rec, path)
    return rec


def validate_telemetry_stream(records):
    """Per (boot_id, type) the ``seq`` must strictly increase and ``mono_s`` must not go back."""
    last = {}
    for index, rec in enumerate(records):
        validate_telemetry_record(rec, f"records[{index}]")
        key = (rec["boot_id"], rec["type"])
        if key in last:
            seq, mono = last[key]
            if rec["seq"] <= seq:
                _fail(f"records[{index}].seq", f"not strictly increasing ({rec['seq']} after {seq})")
            if rec["mono_s"] < mono:
                _fail(f"records[{index}].mono_s", "monotonic clock went backwards")
        last[key] = (rec["seq"], rec["mono_s"])
    return records


def validate_sentinel_latest(doc):
    _schema_marker(doc, SCHEMA_SENTINEL_LATEST, "latest")
    _obj(doc, "latest", ("schema", "node_id", "boot_id", "seq", "utc", "mono_s", "gpus", "cpu_pkg_max_temp_c",
                         "disks", "null_reasons"))
    _str(doc["node_id"], "latest.node_id", SLUG)
    _str(doc["boot_id"], "latest.boot_id", BOOT_ID)
    _int(doc["seq"], "latest.seq", 0)
    _str(doc["utc"], "latest.utc", UTC_Z)
    _num(doc["mono_s"], "latest.mono_s", 0)
    for index, gpu in enumerate(_list(doc["gpus"], "latest.gpus", nullable=True) or []):
        here = f"latest.gpus[{index}]"
        _obj(gpu, here, ("index", "uuid", "temp_c", "util_pct", "compute_apps"))
        _int(gpu["index"], here + ".index", 0)
        _str(gpu["uuid"], here + ".uuid", GPU_UUID, nullable=True)
        _num(gpu["temp_c"], here + ".temp_c", nullable=True)
        _num(gpu["util_pct"], here + ".util_pct", nullable=True)
        _int(gpu["compute_apps"], here + ".compute_apps", 0, nullable=True)
    _num(doc["cpu_pkg_max_temp_c"], "latest.cpu_pkg_max_temp_c", nullable=True)
    for index, disk in enumerate(_list(doc["disks"], "latest.disks", nullable=True) or []):
        _obj(disk, f"latest.disks[{index}]", ("key", "free_bytes"))
        _str(disk["key"], f"latest.disks[{index}].key", STORAGE_KEY)
        _int(disk["free_bytes"], f"latest.disks[{index}].free_bytes", 0, nullable=True)
    _null_reasons(doc, "latest", REASONS, skip={"null_reasons"})
    _canonical(doc, "latest")
    return doc


# --------------------------------------------------------------- job summary v1

def _validate_energy_entry(entry, path):
    _obj(entry, path, ("value", "status"), ("source", "reason", "method", "gpu_index", "coverage_fraction"))
    status = _str(entry["status"], path + ".status", choices=ENERGY_STATUSES)
    value = _num(entry["value"], path + ".value", 0, nullable=True)
    if status == "unavailable" and value is not None:
        _fail(path, "unavailable energy must have value null")
    if value is None and entry.get("reason") not in REASONS:
        _fail(path, "null energy requires a reason code")
    if status == "estimated_model" and value is not None and not entry.get("method"):
        _fail(path, "an estimated value requires a named, versioned method")
    return status, value


def measured_co2_g(energy, factor):
    """Section 5.4: CO2 counts only measured kWh; estimates are never added."""
    if factor is None:
        return None
    total = 0.0
    for entry in energy.values():
        if entry.get("status") in MEASURED_ENERGY_STATUSES and entry.get("value") is not None:
            total += entry["value"]
    return total * factor


def validate_job_summary(summary):
    _schema_marker(summary, SCHEMA_JOB_SUMMARY, "summary")
    _obj(summary, "summary", ("schema", "node_id", "hardware_class", "job_id", "attempt_id", "window", "coverage",
                              "thermal", "power", "energy", "boundary", "co2", "attribution"), allow_extra=True)
    _str(summary["node_id"], "summary.node_id", SLUG)
    _str(summary["hardware_class"], "summary.hardware_class", HARDWARE_CLASS, nullable=True)
    window = _obj(summary["window"], "summary.window", ("start_utc", "end_utc", "duration_s"))
    _num(window["duration_s"], "summary.window.duration_s", 0)
    coverage = _obj(summary["coverage"], "summary.coverage", ("fast_fraction", "max_gap_s", "reboot_detected"))
    _num(coverage["fast_fraction"], "summary.coverage.fast_fraction", 0, 1, nullable=True)
    _num(coverage["max_gap_s"], "summary.coverage.max_gap_s", 0, nullable=True)
    _bool(coverage["reboot_detected"], "summary.coverage.reboot_detected")
    energy = _obj(summary["energy"], "summary.energy", PUBLIC_ENERGY_KEYS, ("cpu_kwh_estimated",))
    for key, entry in energy.items():
        _validate_energy_entry(entry, f"summary.energy.{key}")
    if energy.get("cpu_kwh_estimated", {}).get("status", "estimated_model") != "estimated_model":
        _fail("summary.energy.cpu_kwh_estimated.status", "must be estimated_model")
    for key in PUBLIC_ENERGY_KEYS:
        if energy[key]["status"] == "estimated_model":
            _fail(f"summary.energy.{key}.status", "estimates live only in *_estimated fields")
    boundary = _obj(summary["boundary"], "summary.boundary", ("measured", "estimated", "unavailable",
                                                               "excluded_always", "note"))
    if "wall" not in boundary["note"].lower():
        _fail("summary.boundary.note", "must state that this is not wall-plug energy")
    co2 = _obj(summary["co2"], "summary.co2", ("g", "boundary", "basis", "grid_factor_g_per_kwh",
                                                "factor_source_id"), ("reason", "co2_estimated_g"))
    factor = _num(co2["grid_factor_g_per_kwh"], "summary.co2.grid_factor_g_per_kwh", 0, exclusive_min=True,
                  nullable=True)
    if co2["basis"] != "measured_only":
        _fail("summary.co2.basis", "must be measured_only")
    if factor is None:
        if co2["g"] is not None or co2.get("reason") != "grid_factor_missing":
            _fail("summary.co2", "without a grid factor g must be null with reason grid_factor_missing")
    else:
        if co2["factor_source_id"] is None:
            _fail("summary.co2.factor_source_id", "a grid factor needs a source id")
        expected = measured_co2_g(energy, factor)
        if not is_number(co2["g"]) or abs(co2["g"] - expected) > 1e-9 * max(1.0, abs(expected)):
            _fail("summary.co2.g", f"must equal measured kWh x factor ({expected!r}); estimates are never added")
    _canonical(summary, "summary")
    return summary


def validate_public_telemetry(block):
    """The ``telemetry_v1`` block of a public terminal event: fixed whitelist only."""
    if not isinstance(block, dict):
        _fail("telemetry_v1", "must be an object")
    extra = sorted(set(block) - PUBLIC_TELEMETRY_KEYS)
    if extra:
        _fail("telemetry_v1", f"keys outside the public whitelist: {extra}")
    _schema_marker(block, SCHEMA_JOB_SUMMARY, "telemetry_v1")
    energy = block.get("energy")
    if energy is not None:
        _obj(energy, "telemetry_v1.energy", (), PUBLIC_ENERGY_KEYS)
        for key, entry in energy.items():
            _obj(entry, f"telemetry_v1.energy.{key}", ("value", "status"))
            _str(entry["status"], f"telemetry_v1.energy.{key}.status", choices=ENERGY_STATUSES)
            _num(entry["value"], f"telemetry_v1.energy.{key}.value", 0, nullable=True)
    co2 = block.get("co2")
    if co2 is not None:
        _obj(co2, "telemetry_v1.co2", (), PUBLIC_CO2_KEYS)

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in PRIVATE_METRIC_KEYS:
                    _fail(join_path(path, key), "model metrics and crash names are private")
                walk(item, join_path(path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, join_path(path, index))
    walk(block, "telemetry_v1")
    assert_public(block, "telemetry_v1")
    _canonical(block, "telemetry_v1")
    return block
