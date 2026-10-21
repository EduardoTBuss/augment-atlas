"""Node profile v1 (docs/fleet-protocol.md section 6): probe, validate, hash, public card.

The probe only reads (``Readers`` from ``telemetry_v1``), needs no privilege and
never derives the node ID from a hostname. Policy is never auto-detected:
limits, reserves, guardian, hardware class and the grid factor come out as
``null`` with ``null_reasons[path] = "requires_operator_decision"`` and the
profile starts as ``status: "draining"`` (onboarding step 9.1-12).

CLI (no change to ``cli.py``)::

    python -m augment_atlas.node_profile probe --node-id x299 [--replay DIR] [--now UTC] [--out FILE]
    python -m augment_atlas.node_profile validate FILE [--complete]
    python -m augment_atlas.node_profile sha256 FILE
    python -m augment_atlas.node_profile card FILE --registered-at UTC [--strip-gpu-uuids]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

from . import fleet_schema
from . import telemetry_v1 as tv
from .schema import digest

DEFAULT_STORAGE_KEYS = ("runs", "data")
#: Protocol defaults (sections 4.2/4.3), not host policy.
DEFAULT_LIVENESS = {"heartbeat_seconds": 900, "lease_seconds": 3600, "ntp_required": True}
FORBIDDEN_HOST_CHANGES = ["fan_control", "power_limit", "driver", "bios", "os_config"]
#: nvidia-smi query fields the fleet uses; the probe lists which ones the tool offers.
FIELDS_OF_INTEREST = (
    "index", "uuid", "name", "temperature.gpu", "power.draw", "power.limit", "fan.speed", "utilization.gpu",
    "memory.used", "memory.total", "clocks.sm", "clocks.mem", "pstate", "compute_cap",
    "clocks_event_reasons.active", "pcie.link.gen.current", "pcie.link.width.current",
    "ecc.errors.uncorrected.volatile.total", "total_energy_consumption", "power.draw.instant",
)
DECISION = "requires_operator_decision"


def _unavailable(reason, evidence=None):
    entry = {"status": "unavailable", "reason": reason}
    if evidence:
        entry["evidence"] = evidence
    return entry


def _cmd(readers, argv, timeout=8.0):
    """Returns (text, reason). ``NotCaptured`` maps to ``not_captured`` here (profile only)."""
    try:
        return readers.run_cmd(argv, timeout), None
    except tv.NotCaptured:
        return None, "not_captured"
    except Exception as exc:  # noqa: BLE001
        return None, tv.command_reason(exc)


def _file(readers, path):
    try:
        return readers.read(path), None
    except tv.NotCaptured:
        return None, "not_captured"
    except Exception as exc:  # noqa: BLE001
        return None, tv.file_reason(exc)


def _sensor_state(ok, reason):
    if ok:
        return "available"
    if reason == "not_captured":
        return "unverified"
    return _unavailable(reason)


def probe(readers, node_id, now_utc, storage_keys=DEFAULT_STORAGE_KEYS, nvidia_smi=("nvidia-smi",),
          include_gpu_uuids=True):
    """Read-only node probe. ``now_utc`` is the ``updated_at_utc`` label (injected for determinism)."""
    fleet_schema._str(node_id, "node_id", fleet_schema.SLUG)
    nvidia_smi = list(nvidia_smi)
    reasons, sensors = {}, {}

    # GPUs: the same query and order the sentinel streams.
    gpus = []
    text, reason = _cmd(readers, tv.nvidia_query_argv(nvidia_smi, tv.FAST_QUERY_FIELDS))
    rows = tv.parse_nvidia_csv(text, tv.FAST_QUERY_FIELDS)[0] if text is not None else []
    gpu_reason = reason or (None if rows else "parse_error")
    capabilities, cap_reason = {}, None
    cap_text, cap_reason = _cmd(readers, tv.nvidia_query_argv(nvidia_smi, tv.CAPABILITY_QUERY_FIELDS))
    if cap_text is not None:
        capabilities = {row["index"]: row["values"].get("compute_capability")
                        for row in tv.parse_nvidia_csv(cap_text, tv.CAPABILITY_QUERY_FIELDS)[0]}
    for position, row in enumerate(sorted(rows, key=lambda item: item["index"])):
        values = row["values"]
        entry = {"index": row["index"], "name": values.get("name"), "vram_mib": values.get("mem_total_mib"),
                 "compute_capability": capabilities.get(row["index"])}
        if include_gpu_uuids:
            entry["uuid"] = values.get("uuid")
        for key, source in (("name", "name"), ("vram_mib", "mem_total_mib"), ("uuid", "uuid")):
            if key in entry and entry[key] is None:
                reasons[f"hardware.gpus[{position}].{key}"] = row["reasons"].get(source, "not_exposed")
        if entry["compute_capability"] is None:
            reasons[f"hardware.gpus[{position}].compute_capability"] = cap_reason or "not_exposed"
        gpus.append(entry)
    for field, key in (("temp_c", "gpu.temp_c"), ("power_w", "gpu.power_w"), ("fan_pct", "gpu.fan_pct"),
                       ("util_pct", "gpu.util_pct"), ("mem_used_mib", "gpu.mem_used_mib"),
                       ("sm_mhz", "gpu.clocks_mhz"), ("pstate", "gpu.pstate"), ("power_limit_w", "gpu.power_limit_w")):
        if not rows:
            sensors[key] = _sensor_state(False, gpu_reason)
            continue
        missing = [row["reasons"].get(field) for row in rows if row["values"].get(field) is None]
        sensors[key] = "available" if not missing else _unavailable(missing[0] or "not_exposed")

    help_text, help_reason = _cmd(readers, nvidia_smi + ["--help-query-gpu"])
    supported = tv.parse_help_query_gpu(help_text) if help_text is not None else set()
    if help_text is not None:
        sensors["nvidia_smi_fields"] = sorted(field for field in FIELDS_OF_INTEREST if field in supported)
    q_text, q_reason = _cmd(readers, nvidia_smi + ["-q", "-d", "PERFORMANCE,POWER,TEMPERATURE,ECC"])
    q = tv.parse_nvidia_q(q_text) if q_text is not None else []
    if q:
        has_reasons = all(item["clock_event_reasons"] is not None for item in q)
        sensors["gpu.clock_event_reasons"] = "available" if has_reasons else _unavailable("not_exposed")
        ecc = {item["null_reasons"].get("ecc_errors") for item in q}
        sensors["gpu.ecc_errors"] = "available" if ecc == {None} else _unavailable(
            sorted(reason for reason in ecc if reason)[0], "nvidia-smi -q ECC Mode Current")
    else:
        sensors["gpu.clock_event_reasons"] = _sensor_state(False, q_reason or "parse_error")
        sensors["gpu.ecc_errors"] = _sensor_state(False, q_reason or "parse_error")
    if help_text is None:
        sensors["gpu.energy_counter"] = _sensor_state(False, help_reason)
    elif "total_energy_consumption" in supported:
        sensors["gpu.energy_counter"] = "unverified"
    else:
        sensors["gpu.energy_counter"] = _unavailable(
            "not_exposed", "total_energy_consumption absent from nvidia-smi --help-query-gpu")

    version_text, version_reason = _cmd(readers, nvidia_smi + ["--version"])
    driver = tv.parse_nvidia_version(version_text)["driver"] if version_text is not None else None

    # CPU, memory, software.
    lscpu_text, lscpu_reason = _cmd(readers, ["lscpu"])
    lscpu = tv.parse_lscpu(lscpu_text) if lscpu_text is not None else {}
    cores = (lscpu.get("cores_per_socket") or 0) * (lscpu.get("sockets") or 0) or None
    cpu = {"model": lscpu.get("model"), "sockets": lscpu.get("sockets"), "cores": cores,
           "threads": lscpu.get("cpus"), "tdp_w": None}
    for key in ("model", "sockets", "cores", "threads"):
        if cpu[key] is None:
            reasons[f"hardware.cpu.{key}"] = lscpu_reason or "parse_error"
    reasons["hardware.cpu.tdp_w"] = DECISION
    meminfo, mem_reason = _file(readers, readers.proc("meminfo"))
    ram = tv.parse_meminfo(meminfo)["ram_total_bytes"] if meminfo is not None else None
    if ram is None:
        reasons["hardware.ram_bytes"] = mem_reason or "parse_error"
    software = {"os": None, "kernel": None, "nvidia_driver": driver, "docker": None, "nvidia_container_toolkit": None}
    os_release, os_reason = _file(readers, readers.etc_root / "os-release")
    if os_release is not None:
        for line in os_release.splitlines():
            if line.startswith("PRETTY_NAME="):
                software["os"] = line.split("=", 1)[1].strip().strip('"') or None
    kernel, kernel_reason = _file(readers, readers.proc("sys", "kernel", "osrelease"))
    software["kernel"] = kernel.strip() if kernel else None
    docker, docker_reason = _cmd(readers, ["docker", "--version"])
    if docker:
        software["docker"] = docker.strip().replace("Docker version ", "").split(",", 1)[0] or None
    toolkit, toolkit_reason = _cmd(readers, ["nvidia-ctk", "--version"])
    if toolkit:
        software["nvidia_container_toolkit"] = toolkit.strip().splitlines()[0].rsplit(" ", 1)[-1] or None
    for key, why in (("os", os_reason), ("kernel", kernel_reason), ("nvidia_driver", version_reason),
                     ("docker", docker_reason), ("nvidia_container_toolkit", toolkit_reason)):
        if software[key] is None:
            reasons[f"hardware.software.{key}"] = why or "parse_error"

    # CPU temperatures: hwmon first (what the sentinel reads), sensors CLI as cross-check.
    thermal_reference = {"cpu_pkg": [], "gpus": []}
    try:
        chips = tv.read_hwmon(readers)
        normalized = tv.cpu_from_hwmon(chips)
        hwmon_reason = None
    except tv.NotCaptured:
        chips, normalized, hwmon_reason = [], None, "not_captured"
    except Exception as exc:  # noqa: BLE001
        chips, normalized, hwmon_reason = [], None, tv.file_reason(exc)
    if normalized is None or not normalized["packages"]:
        cli, cli_reason, _src = tv.read_cpu_sensors_cli(readers)
        normalized = cli if cli and cli["packages"] else normalized
        hwmon_reason = hwmon_reason or cli_reason
    cpu_chip = next((chip["name"] for chip in chips if tv._classify_chip(chip["name"]) == "cpu"), None)
    if normalized and normalized["packages"]:
        sensors["cpu_pkg.temp_c"] = "available"
        for item in normalized["packages"]:
            thermal_reference["cpu_pkg"].append({"package": item["package"], "high_c": item["high_c"],
                                                 "crit_c": item["crit_c"],
                                                 "source": f"hwmon:{cpu_chip}" if cpu_chip else "sensors"})
        limits_known = any(item["high_c"] is not None or item["crit_c"] is not None for item in normalized["packages"])
        sensors["cpu.temp_limits"] = "available" if limits_known else _unavailable("not_exposed")
    else:
        sensors["cpu_pkg.temp_c"] = _sensor_state(False, hwmon_reason or "not_exposed")
        sensors["cpu.temp_limits"] = _sensor_state(False, hwmon_reason or "not_exposed")
    sensors["cpu_core.temp_c"] = ("available" if normalized and normalized["cores"]
                                  else _sensor_state(False, hwmon_reason or "not_exposed"))
    acpi = normalized and any(zone["kind"] == "acpi_zone" for zone in normalized["zones"])
    sensors["acpi_zone.temp_c"] = "available" if acpi else _sensor_state(False, hwmon_reason or "not_exposed")
    if hwmon_reason == "not_captured":
        sensors["hwmon.fan_rpm"] = sensors["hwmon.pwm"] = "unverified"
    else:
        fans, _fan_reason = tv.fans_from_hwmon(chips)
        has_rpm = bool(fans) and any(fan["rpm"] is not None for fan in fans)
        has_pwm = bool(fans) and any(fan["pwm"] is not None for fan in fans)
        sensors["hwmon.fan_rpm"] = "available" if has_rpm else _unavailable(hwmon_reason or "not_exposed")
        sensors["hwmon.pwm"] = "available" if has_pwm else _unavailable(hwmon_reason or "not_exposed")
    for item in q:
        thermal_reference["gpus"].append({"index": item["index"], "slowdown_c": item["temp_slowdown_c"],
                                          "shutdown_c": item["temp_shutdown_c"], "target_c": item["temp_target_c"],
                                          "source": "nvidia-smi -q"})
    try:
        frequency = tv.read_cpu_freq_mhz(readers)
        sensors["cpu.freq_mhz"] = "available" if frequency else _unavailable("not_exposed")
    except tv.NotCaptured:
        sensors["cpu.freq_mhz"] = "unverified"
    except Exception as exc:  # noqa: BLE001
        sensors["cpu.freq_mhz"] = _unavailable(tv.file_reason(exc))

    # RAPL: aggregate per domain family (package, dram, core, uncore, psys).
    try:
        zones, zone_reasons = tv.read_rapl(readers)
        families = {}
        for position, zone in enumerate(zones):
            family = (zone["domain"] or "unknown").split("-", 1)[0]
            families.setdefault(family, []).append(zone_reasons.get(f"rapl[{position}].energy_uj"))
        for family, outcomes in sorted(families.items()):
            failures = [item for item in outcomes if item]
            sensors[f"rapl.{family}"] = "available" if not failures else _unavailable(failures[0])
        if not zones:
            sensors["rapl.package"] = _unavailable("not_exposed")
    except tv.NotCaptured:
        sensors["rapl.package"] = "unverified"
    except Exception as exc:  # noqa: BLE001
        sensors["rapl.package"] = _unavailable(tv.file_reason(exc))

    ntp_text, ntp_reason = _cmd(readers, ["timedatectl", "show", "-p", "NTPSynchronized"])
    if ntp_text is not None:
        try:
            tv.parse_timedatectl(ntp_text)
            sensors["clock.ntp_synchronized"] = "available"
        except ValueError:
            sensors["clock.ntp_synchronized"] = _unavailable("parse_error")
    else:
        sensors["clock.ntp_synchronized"] = _sensor_state(False, ntp_reason)
    boot_id, boot_reason = tv.read_boot_id(readers)
    sensors["host.boot_id"] = "available" if boot_id else _sensor_state(False, boot_reason)
    stat_text, stat_reason = _file(readers, readers.proc("stat"))
    sensors["host.proc_stat"] = "available" if stat_text else _sensor_state(False, stat_reason)
    sensors["host.meminfo"] = "available" if meminfo else _sensor_state(False, mem_reason)
    try:
        readers.listdir(readers.crash_dir)
        sensors["host.var_crash_listing"] = "available"
    except tv.NotCaptured:
        sensors["host.var_crash_listing"] = "unverified"
    except Exception as exc:  # noqa: BLE001
        sensors["host.var_crash_listing"] = _unavailable(tv.file_reason(exc))
    # The probe does not test kernel log access (P3 is an operator decision).
    sensors["kernel_log"] = "unverified"

    storage = [{"key": key, "reserve_bytes": None} for key in storage_keys]
    for position in range(len(storage)):
        reasons[f"storage[{position}].reserve_bytes"] = DECISION
    limits = {"max_concurrent_trainings": None, "allowed_gpus": None, "container": None, "thermal": None,
              "min_idle_seconds_between_jobs": None, "training_retry": "never",
              "forbidden_host_changes": list(FORBIDDEN_HOST_CHANGES)}
    for key in ("max_concurrent_trainings", "allowed_gpus", "container", "thermal", "min_idle_seconds_between_jobs"):
        reasons[f"limits.{key}"] = DECISION
    grid = {key: None for key in fleet_schema.GRID_KEYS}
    for key in fleet_schema.GRID_KEYS:
        reasons[f"grid.{key}"] = DECISION
    reasons["hardware_class"] = DECISION
    reasons["guardian"] = DECISION
    hardware = {"cpu": cpu, "ram_bytes": ram, "gpus": gpus, "software": software}
    if thermal_reference["cpu_pkg"] or thermal_reference["gpus"]:
        hardware["thermal_reference"] = thermal_reference
    profile = {
        "schema": fleet_schema.SCHEMA_NODE_PROFILE, "node_id": node_id, "status": "draining",
        "hardware_class": None, "hardware": hardware, "storage": storage, "limits": limits, "guardian": None,
        "liveness": dict(DEFAULT_LIVENESS), "datasets": [], "images": [], "weights": [], "sensors": sensors,
        "grid": grid, "updated_at_utc": now_utc, "null_reasons": dict(sorted(reasons.items())),
    }
    return validate(profile)


def validate(profile, complete=False):
    return fleet_schema.validate_node_profile(profile, complete=complete)


def profile_sha256(profile):
    """``digest(canonical_json(profile))`` of the profile exactly as stored."""
    return digest(profile)


def public_card(profile, registered_at_utc, strip_gpu_uuids=False):
    """Section 6.3: the whole profile + ``profile_sha256`` + ``registered_at_utc``.

    The hash is always of the full private profile, so a card with stripped
    UUIDs still proves which profile it came from.
    """
    validate(profile)
    card = copy.deepcopy(profile)
    if strip_gpu_uuids:
        for gpu in card["hardware"]["gpus"]:
            gpu.pop("uuid", None)
        card["null_reasons"] = {key: value for key, value in card.get("null_reasons", {}).items()
                                if not (key.startswith("hardware.gpus[") and key.endswith("].uuid"))}
    card["profile_sha256"] = profile_sha256(profile)
    card["registered_at_utc"] = registered_at_utc
    return fleet_schema.validate_public_card(card)


def card_matches_profile(card, profile):
    """The worker refuses to claim unless the published card hashes to the local profile."""
    return isinstance(card, dict) and card.get("profile_sha256") == profile_sha256(profile)


def dumps(document):
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m augment_atlas.node_profile",
                                     description="Read-only node profile tools (fleet protocol v1).")
    commands = parser.add_subparsers(dest="command", required=True)
    probe_cmd = commands.add_parser("probe", help="read-only probe of this host (or of a replayed capture)")
    probe_cmd.add_argument("--node-id", required=True, help="public node alias (never the hostname)")
    probe_cmd.add_argument("--replay", help="captured fixture directory (tests/fixtures/telemetry_v1/<node>)")
    probe_cmd.add_argument("--now", help="updated_at_utc label (default: current UTC)")
    probe_cmd.add_argument("--storage-key", action="append", help="logical storage keys (default: runs, data)")
    probe_cmd.add_argument("--nvidia-smi", default="nvidia-smi")
    probe_cmd.add_argument("--no-gpu-uuids", action="store_true")
    probe_cmd.add_argument("--out")
    validate_cmd = commands.add_parser("validate")
    validate_cmd.add_argument("file")
    validate_cmd.add_argument("--complete", action="store_true", help="also refuse pending operator decisions")
    sha_cmd = commands.add_parser("sha256")
    sha_cmd.add_argument("file")
    card_cmd = commands.add_parser("card")
    card_cmd.add_argument("file")
    card_cmd.add_argument("--registered-at", required=True)
    card_cmd.add_argument("--strip-gpu-uuids", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            readers = tv.ReplayReaders(args.replay) if args.replay else tv.Readers()
            now = args.now or tv.utc_label().split(".")[0] + "Z"
            document = probe(readers, args.node_id, now, tuple(args.storage_key or DEFAULT_STORAGE_KEYS),
                             nvidia_smi=[args.nvidia_smi], include_gpu_uuids=not args.no_gpu_uuids)
            text = dumps(document)
            if args.out:
                Path(args.out).write_text(text, encoding="utf-8", newline="\n")
            else:
                sys.stdout.write(text)
            return 0
        document = json.loads(Path(args.file).read_text(encoding="utf-8"))
        if args.command == "validate":
            validate(document, complete=args.complete)
            print(json.dumps({"valid": True, "complete": bool(args.complete),
                              "profile_sha256": profile_sha256(document)}))
        elif args.command == "sha256":
            print(profile_sha256(validate(document)))
        else:
            sys.stdout.write(dumps(public_card(document, args.registered_at, args.strip_gpu_uuids)))
        return 0
    except (fleet_schema.SchemaError, ValueError, OSError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
