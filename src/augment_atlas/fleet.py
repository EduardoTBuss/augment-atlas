"""F4: capacity-aware job selection (docs/fleet-protocol.md section 7.1).

Pure functions only: no I/O, no clock, no subprocess, no Git. Callers pass every
fact explicitly (records, node profile, public card, ``sentinel/latest.json``
document, the consumer's monotonic ``now`` and a ``worker`` facts dict), so the
same inputs always give the same decision (section 7.1, determinism).

Steps covered here (section 7.1): 0 (only the parts that are pure: card,
profile, sentinel freshness and live node gates), 1 (pending + dependencies),
2 (``requirements`` or ``derive_legacy``), 3 (static filter), 4 (per-job disk
gate) and 5 (deterministic order). Locks, ``active.json``, guardian arming,
``config_for`` and the CAS claim stay in the worker (F6).

``worker`` facts dict (all public, no paths)::

    {"node_id": "x299",              # == settings.server == profile.node_id
     "worker": "gpu0",               # optional label
     "phase": "smoke",               # settings.phase
     "code_commit": "<40hex>",       # HEAD of the training checkout
     "gpu_index": 0,                 # settings.gpu
     "gpu_uuid": "GPU-...",          # nvidia-smi uuid of that GPU (or None)
     "gpu_memory_mib": 12288,        # nvidia-smi memory.total (legacy compatible())
     "compute_capability": [5, 2],   # image probe, or None (falls back to the profile)
     "image_id": "sha256:<64hex>",   # settings.image resolved by docker image inspect
     "datasets": ["synthetic-tiny"], # settings.datasets (legacy filter only)
     "boot_id": "<uuid>"}            # /proc/sys/kernel/random/boot_id now

Every refusal carries a stable reason code from ``REASONS`` (catalogue below and
in docs/fleet-protocol.md section 7.1). ``null`` sensor values never raise: a
null in a field that a gate needs refuses with ``*_unavailable``; a null in a
field no gate reads (RAPL, fans, other disks...) is ignored.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import math

from . import fleet_schema as fs
from .schema import digest, job_start_refusal

#: In-memory marker of requirements derived from a pre-fleet spec. Never persisted:
#: a stored ``spec.requirements`` carrying it is refused as invalid.
LEGACY_SCHEMA = "atlas.requirements.legacy"
LATEST_MAX_AGE_S = 5.0          # section 5.6: older -> stale -> no claim
LATEST_MAX_FUTURE_S = 1.0       # same tolerance as sentinel.read_latest
GPU_IDLE_MAX_UTIL_PCT = 10      # section 7.1 step 0: util_pct < 10
RUNS_STORAGE_KEY = "runs"
OK = "ok"

#: Stable reason catalogue. Keys never change meaning; new ones may be added.
REASONS = {
    OK: "all checks passed",
    "expired": "seed request start deadline reached",
    "start_time_unknown": "seed request requires the current UTC start time",
    "invalid_seed_request": "seed request immutable binding is invalid",
    # -- step 1: queue state
    "not_pending": "record state is not pending",
    "dependency_not_succeeded": "a dependency is missing or not succeeded",
    # -- step 2/3: static filter (job x node x worker)
    "invalid_scheduling": "spec.scheduling is not {priority:int, order:int|null}",
    "unsupported_requirements_schema": "requirements.schema is a newer/unknown version (forward compatibility skip)",
    "invalid_requirements": "spec.requirements violates atlas.requirements.v1",
    "legacy_compatibility_invalid": "legacy compatibility.min_memory_mb is not a finite number",
    "node_profile_required": "requirements v1 need a node profile; legacy mode only runs legacy jobs",
    "phase_mismatch": "spec.phase differs from the worker phase",
    "code_commit_mismatch": "spec.code_commit differs from the training checkout HEAD",
    "node_not_allowed": "node_id not in requirements.allowed_nodes",
    "final_requires_hardware_class": "final jobs must pin hardware_class (decision P4)",
    "hardware_class_mismatch": "requirements.hardware_class differs from the profile",
    "gpu_not_allowed": "worker GPU not in profile.limits.allowed_gpus",
    "gpu_not_in_profile": "worker GPU index absent from profile.hardware.gpus",
    "gpu_identity_mismatch": "GPU UUID differs between worker, profile or sentinel",
    "gpu_vram_unknown": "no VRAM figure for the worker GPU",
    "insufficient_vram": "GPU VRAM below min_vram_mib",
    "compute_capability_unknown": "job needs a compute capability the node cannot prove",
    "insufficient_compute_capability": "GPU compute capability below the minimum",
    "image_mismatch": "job image differs from the image configured on this worker",
    "image_not_validated_on_node": "requirements.image_id not in profile.images",
    "gpu_uuid_mismatch": "legacy compatibility.gpu_uuid pins another GPU",
    "dataset_not_configured": "legacy spec.dataset not in settings.datasets",
    "dataset_not_validated_on_node": "(id, version, manifest_sha256) not in profile.datasets",
    "weights_not_validated_on_node": "requirements.weights_sha256 not in profile.weights",
    "container_limits_unset": "job sets container limits but the profile has none",
    "container_exceeds_node_limits": "job container cpus/memory above profile.limits.container",
    # -- step 4: per-job disk gate
    "disk_insufficient_for_job": "free(runs) - reserve < requirements.min_free_disk_bytes",
    # -- step 0: node gates
    "node_profile_invalid": "profile fails validate(complete=True)",
    "worker_node_mismatch": "worker node_id differs from profile.node_id",
    "node_card_missing": "no published card fleet/nodes/<node_id>.json",
    "node_card_invalid": "published card fails the card schema",
    "node_card_mismatch": "card.profile_sha256 differs from sha(profile)",
    "node_not_active": "card/profile status is not active (draining or retired)",
    "boot_id_unknown": "worker boot_id unknown; sentinel age cannot be trusted",
    "sentinel_latest_missing": "sentinel/latest.json absent",
    "sentinel_latest_invalid": "sentinel/latest.json unreadable or not atlas.sentinel-latest.v1",
    "sentinel_node_mismatch": "latest.json node_id differs from the profile",
    "sentinel_other_boot": "latest.json from another boot",
    "sentinel_stale": "latest.json older than 5 s (or from the future) on the monotonic clock",
    "thermal_sensor_unsupported": "profile thermal limit names a sensor this code cannot read",
    "gpu_temp_unavailable": "a GPU temperature needed by a thermal gate is null",
    "gpu_too_hot": "a GPU temperature is above claim_max_c",
    "cpu_pkg_temp_unavailable": "CPU package temperature needed by a thermal gate is null",
    "cpu_pkg_too_hot": "CPU package temperature is above claim_max_c",
    "disk_runs_unreported": "latest.json has no 'runs' disk entry",
    "disk_free_unavailable": "free_bytes of 'runs' is null",
    "disk_below_reserve": "free(runs) below the profile reserve",
    "gpu_not_reported": "latest.json has no entry for the worker GPU",
    "gpu_idle_unknown": "util_pct or compute_apps of the worker GPU is null",
    "gpu_busy": "worker GPU has compute apps or util_pct >= 10",
    "node_cooldown": "less than min_idle_seconds_between_jobs since the last job",
    # -- explain only
    "not_evaluated": "skipped because an earlier prerequisite failed",
}


def _check(name, reason, detail=None):
    if reason not in REASONS:
        raise AssertionError(f"reason {reason!r} missing from the catalogue")
    return {"check": name, "ok": reason == OK, "reason": reason, "detail": detail}


def _first_failure(checks):
    for item in checks:
        if not item["ok"]:
            return False, item["reason"]
    return True, OK


# ------------------------------------------------------------------ requirements

def derive_legacy(spec):
    """Implicit requirements of a pre-fleet job (no ``spec.requirements``, section 3.2a).

    Reproduces ``DockerRuntime.compatible()``: VRAM from ``compatibility.min_memory_mb``,
    image from ``compatibility.image_id`` or else the image configured on the worker,
    optional ``compatibility.gpu_uuid`` pin, dataset id in ``settings.datasets``,
    ``gpu_required = True``. The ``legacy`` block keeps "key absent" apart from
    "key present with null", exactly as ``dict.get(key, default)`` does there.
    """
    compat = spec.get("compatibility") if isinstance(spec, dict) else None
    compat = compat if isinstance(compat, dict) else {}
    raw = compat.get("min_memory_mb", 0)
    error = None
    try:
        vram = float(raw)
        if not math.isfinite(vram):
            raise ValueError("non-finite")
    except (TypeError, ValueError):
        vram, error = None, "legacy_compatibility_invalid"
    dataset = spec.get("dataset") if isinstance(spec, dict) else None
    return {"schema": LEGACY_SCHEMA, "gpu_required": True, "min_vram_mib": vram,
            "min_compute_capability": None, "hardware_class": None, "allowed_nodes": None,
            "image_id": compat.get("image_id"), "image_ref": None,
            "dataset": {"id": dataset, "version": None, "manifest_sha256": None},
            "weights_sha256": None, "min_free_disk_bytes": 0, "container": None,
            "expected_epoch_seconds_max": None,
            "legacy": {"image_id_declared": "image_id" in compat, "gpu_uuid_declared": "gpu_uuid" in compat,
                       "gpu_uuid": compat.get("gpu_uuid"), "error": error}}


def job_requirements(spec):
    """Step 2: ``spec.requirements`` if present (copied), else ``derive_legacy(spec)``."""
    if isinstance(spec, dict) and "requirements" in spec:
        return copy.deepcopy(spec["requirements"])
    return derive_legacy(spec)


def is_legacy(req):
    return isinstance(req, dict) and req.get("schema") == LEGACY_SCHEMA


def requirements_kind(req):
    """``("legacy"|"v1"|"unsupported"|"invalid", detail)`` without raising."""
    if is_legacy(req):
        return "legacy", None
    try:
        fs.validate_requirements(req)
        return "v1", None
    except fs.UnsupportedSchema as exc:
        return "unsupported", str(exc)
    except fs.SchemaError as exc:
        return "invalid", str(exc)


def _profile_gpu(profile, index):
    for gpu in (profile or {}).get("hardware", {}).get("gpus") or []:
        if gpu.get("index") == index:
            return gpu
    return None


def _dataset_validated(profile, dataset):
    for item in profile.get("datasets") or []:
        if (item.get("id") == dataset["id"] and item.get("version") in (dataset["version"], "*")
                and item.get("manifest_sha256") == dataset["manifest_sha256"]):
            return True
    return False


def static_checks(req, profile, worker, spec=None):
    """Step 3, every rule with its reason (for ``explain``). Stops at the first
    structural failure (schema/profile); evaluates all capability rules after that."""
    checks = []
    if spec is not None:
        scheduling = spec.get("scheduling")
        try:
            if scheduling is not None:
                fs.validate_scheduling(scheduling)
            checks.append(_check("scheduling", OK))
        except fs.SchemaError as exc:
            return checks + [_check("scheduling", "invalid_scheduling", str(exc))]
        if is_legacy(req) and "requirements" in spec:
            return checks + [_check("requirements.schema", "invalid_requirements", "legacy marker in a stored spec")]
    kind, detail = requirements_kind(req)
    if kind == "unsupported":
        return checks + [_check("requirements.schema", "unsupported_requirements_schema", detail)]
    if kind == "invalid":
        return checks + [_check("requirements.schema", "invalid_requirements", detail)]
    legacy = kind == "legacy"
    if legacy and req["legacy"]["error"]:
        return checks + [_check("requirements.legacy", req["legacy"]["error"])]
    checks.append(_check("requirements.schema", OK, kind))
    if not legacy and profile is None:
        return checks + [_check("node_profile", "node_profile_required")]
    w = worker
    if spec is not None:
        checks.append(_check("phase", OK if spec.get("phase") == w.get("phase") else "phase_mismatch",
                             {"job": spec.get("phase"), "worker": w.get("phase")}))
        checks.append(_check("code_commit", OK if spec.get("code_commit") == w.get("code_commit") else "code_commit_mismatch"))
    nodes = req.get("allowed_nodes")
    checks.append(_check("allowed_nodes", OK if nodes is None or w.get("node_id") in nodes else "node_not_allowed",
                         {"allowed": nodes, "node_id": w.get("node_id")}))
    hw_class = req.get("hardware_class")
    if spec is not None and spec.get("phase") == "final" and hw_class is None:
        checks.append(_check("hardware_class", "final_requires_hardware_class"))
    elif hw_class is not None and (profile is None or profile.get("hardware_class") != hw_class):
        checks.append(_check("hardware_class", "hardware_class_mismatch",
                             {"job": hw_class, "node": (profile or {}).get("hardware_class")}))
    else:
        checks.append(_check("hardware_class", OK))
    pgpu = _profile_gpu(profile, w.get("gpu_index")) if profile is not None else None
    if profile is not None:
        allowed = (profile.get("limits") or {}).get("allowed_gpus")
        if req.get("gpu_required"):
            checks.append(_check("allowed_gpus", OK if allowed is not None and w.get("gpu_index") in allowed
                                 else "gpu_not_allowed", {"allowed": allowed, "gpu": w.get("gpu_index")}))
        if pgpu is None:
            checks.append(_check("profile_gpu", "gpu_not_in_profile", {"gpu": w.get("gpu_index")}))
        elif pgpu.get("uuid") and w.get("gpu_uuid") and pgpu["uuid"] != w["gpu_uuid"]:
            checks.append(_check("profile_gpu", "gpu_identity_mismatch"))
        else:
            checks.append(_check("profile_gpu", OK))
    # VRAM: the measured worker figure wins (it is what compatible() used); profile as fallback.
    vram = w.get("gpu_memory_mib")
    if vram is None and pgpu is not None:
        vram = pgpu.get("vram_mib")
    need = req.get("min_vram_mib") or 0
    if need and vram is None:
        checks.append(_check("vram", "gpu_vram_unknown", {"min_vram_mib": need}))
    elif vram is not None and not need <= vram:
        checks.append(_check("vram", "insufficient_vram", {"min_vram_mib": need, "vram_mib": vram}))
    else:
        checks.append(_check("vram", OK, {"min_vram_mib": need, "vram_mib": vram}))
    min_cc = req.get("min_compute_capability")
    if min_cc is not None:
        cc = w.get("compute_capability") or (pgpu or {}).get("compute_capability")
        if cc is None:
            checks.append(_check("compute_capability", "compute_capability_unknown", {"min": min_cc}))
        else:
            checks.append(_check("compute_capability", OK if list(cc) >= list(min_cc) else "insufficient_compute_capability",
                                 {"min": min_cc, "gpu": list(cc)}))
    if legacy:
        expected = req["image_id"] if req["legacy"]["image_id_declared"] else w.get("image_id")
        checks.append(_check("image", OK if expected == w.get("image_id") else "image_mismatch"))
        if req["legacy"]["gpu_uuid_declared"]:
            checks.append(_check("gpu_uuid", OK if req["legacy"]["gpu_uuid"] == w.get("gpu_uuid") else "gpu_uuid_mismatch"))
        checks.append(_check("dataset", OK if req["dataset"]["id"] in (w.get("datasets") or []) else "dataset_not_configured",
                             {"dataset": req["dataset"]["id"]}))
        return checks
    checks.append(_check("image", OK if req["image_id"] == w.get("image_id") else "image_mismatch",
                         {"job": req["image_id"][:19], "worker": (w.get("image_id") or "")[:19]}))
    images = {item.get("image_id") for item in profile.get("images") or []}
    checks.append(_check("image_validated", OK if req["image_id"] in images else "image_not_validated_on_node"))
    checks.append(_check("dataset", OK if _dataset_validated(profile, req["dataset"]) else "dataset_not_validated_on_node",
                         {"id": req["dataset"]["id"], "version": req["dataset"]["version"]}))
    if req.get("weights_sha256") is not None:
        weights = {item.get("sha256") for item in profile.get("weights") or []}
        checks.append(_check("weights", OK if req["weights_sha256"] in weights else "weights_not_validated_on_node"))
    container = req.get("container")
    if container is not None:
        limit = (profile.get("limits") or {}).get("container")
        if limit is None:
            checks.append(_check("container", "container_limits_unset"))
        else:
            fits = container["cpus"] <= limit["cpus"] and container["memory_bytes"] <= limit["memory_bytes"]
            checks.append(_check("container", OK if fits else "container_exceeds_node_limits",
                                 {"job": container, "node": {"cpus": limit["cpus"], "memory_bytes": limit["memory_bytes"]}}))
    return checks


def static_match(req, profile, worker, spec=None):
    """Step 3 as ``(ok, reason)``. ``profile=None`` is the explicit legacy mode:
    only legacy requirements can match, with the exact ``compatible()`` decision."""
    return _first_failure(static_checks(req, profile, worker, spec))


# ------------------------------------------------------------------- node gates

def _runs_reserve(profile):
    for item in profile.get("storage") or []:
        if item.get("key") == RUNS_STORAGE_KEY:
            return item.get("reserve_bytes")
    return None


def _runs_free(latest):
    if not isinstance(latest, dict):
        return False, None
    for disk in latest.get("disks") or []:
        if disk.get("key") == RUNS_STORAGE_KEY:
            return True, disk.get("free_bytes")
    return False, None


def node_gate_checks(latest, profile, now, *, worker, card, idle_since_mono=None):
    """Step 0 (pure part), every gate with its reason. ``now`` is the consumer's
    ``time.monotonic()`` in the same boot; ``idle_since_mono`` is the monotonic
    end of the previous job in this boot (``None`` = no job since boot)."""
    checks = []
    try:
        fs.validate_node_profile(profile, complete=True)
    except fs.SchemaError as exc:
        return [_check("profile", "node_profile_invalid", str(exc))]
    checks.append(_check("profile", OK))
    checks.append(_check("worker_node", OK if worker.get("node_id") == profile["node_id"] else "worker_node_mismatch"))
    if card is None:
        checks.append(_check("card", "node_card_missing"))
    else:
        try:
            fs.validate_public_card(card)
            same = card["profile_sha256"] == digest(profile)
            checks.append(_check("card", OK if same else "node_card_mismatch"))
        except fs.SchemaError as exc:
            checks.append(_check("card", "node_card_invalid", str(exc)))
    status_ok = profile["status"] == "active" and (card is None or card.get("status") == "active")
    checks.append(_check("status", OK if status_ok else "node_not_active",
                         {"profile": profile["status"], "card": (card or {}).get("status")}))
    # -- sentinel freshness: everything below reads latest.json
    boot_id = worker.get("boot_id")
    if latest is None:
        fresh = _check("sentinel", "sentinel_latest_missing")
    else:
        try:
            fs.validate_sentinel_latest(latest)
            if latest["node_id"] != profile["node_id"]:
                fresh = _check("sentinel", "sentinel_node_mismatch")
            elif not boot_id:
                fresh = _check("sentinel", "boot_id_unknown")
            elif latest["boot_id"] != boot_id:
                fresh = _check("sentinel", "sentinel_other_boot")
            else:
                age = now - latest["mono_s"]
                fresh = _check("sentinel", OK if -LATEST_MAX_FUTURE_S <= age <= LATEST_MAX_AGE_S else "sentinel_stale",
                               {"age_s": round(age, 3), "max_age_s": LATEST_MAX_AGE_S})
        except fs.SchemaError as exc:
            fresh = _check("sentinel", "sentinel_latest_invalid", str(exc))
    checks.append(fresh)
    if not fresh["ok"]:
        return checks + [_check(name, "not_evaluated") for name in ("thermal", "disk", "gpu_idle", "cooldown")]
    limits = profile["limits"]
    for limit in limits["thermal"]:
        sensor, ceiling = limit["sensor"], limit["claim_max_c"]
        if sensor == "gpu.temp_c":
            gpus = latest["gpus"]
            temps = None if gpus is None or not gpus else [gpu["temp_c"] for gpu in gpus]
            if temps is None or any(value is None for value in temps):
                checks.append(_check("thermal:" + sensor, "gpu_temp_unavailable"))
            else:
                checks.append(_check("thermal:" + sensor, OK if max(temps) <= ceiling else "gpu_too_hot",
                                     {"max_c": max(temps), "claim_max_c": ceiling}))
        elif sensor == "cpu_pkg.temp_c":
            value = latest["cpu_pkg_max_temp_c"]
            if value is None:
                checks.append(_check("thermal:" + sensor, "cpu_pkg_temp_unavailable"))
            else:
                checks.append(_check("thermal:" + sensor, OK if value <= ceiling else "cpu_pkg_too_hot",
                                     {"max_c": value, "claim_max_c": ceiling}))
        else:
            checks.append(_check("thermal:" + sensor, "thermal_sensor_unsupported"))
    present, free = _runs_free(latest)
    reserve = _runs_reserve(profile)
    if not present:
        checks.append(_check("disk", "disk_runs_unreported"))
    elif free is None:
        checks.append(_check("disk", "disk_free_unavailable"))
    else:
        checks.append(_check("disk", OK if free - (reserve or 0) >= 0 else "disk_below_reserve",
                             {"free_bytes": free, "reserve_bytes": reserve}))
    entry = next((gpu for gpu in latest["gpus"] or [] if gpu["index"] == worker.get("gpu_index")), None)
    if entry is None:
        checks.append(_check("gpu_idle", "gpu_not_reported", {"gpu": worker.get("gpu_index")}))
    elif entry["uuid"] and worker.get("gpu_uuid") and entry["uuid"] != worker["gpu_uuid"]:
        checks.append(_check("gpu_idle", "gpu_identity_mismatch"))
    elif entry["util_pct"] is None or entry["compute_apps"] is None:
        checks.append(_check("gpu_idle", "gpu_idle_unknown",
                             {"null": sorted(key for key in ("util_pct", "compute_apps") if entry[key] is None)}))
    else:
        idle = entry["compute_apps"] == 0 and entry["util_pct"] < GPU_IDLE_MAX_UTIL_PCT
        checks.append(_check("gpu_idle", OK if idle else "gpu_busy",
                             {"util_pct": entry["util_pct"], "compute_apps": entry["compute_apps"]}))
    wait = limits["min_idle_seconds_between_jobs"]
    if idle_since_mono is None or now - idle_since_mono >= wait:
        checks.append(_check("cooldown", OK))
    else:
        checks.append(_check("cooldown", "node_cooldown", {"idle_s": round(now - idle_since_mono, 3), "required_s": wait}))
    return checks


def node_gates(latest, profile, now, *, worker, card, idle_since_mono=None):
    """Step 0 as ``(ok, reason)``: any failure means no claim and no push."""
    return _first_failure(node_gate_checks(latest, profile, now, worker=worker, card=card,
                                           idle_since_mono=idle_since_mono))


def job_disk_gate(req, latest, profile):
    """Step 4: ``free(runs) - reserve >= req.min_free_disk_bytes``."""
    present, free = _runs_free(latest)
    if not present:
        return False, "disk_runs_unreported"
    if free is None:
        return False, "disk_free_unavailable"
    need = req.get("min_free_disk_bytes") or 0
    return (True, OK) if free - (_runs_reserve(profile) or 0) >= need else (False, "disk_insufficient_for_job")


# ------------------------------------------------------------------------ order

def sort_key(candidate):
    """``(-priority, order nulls last, job_id)``; accepts a record or a spec.
    Malformed scheduling values fall back to the defaults (the static filter
    already refuses them), so ordering itself never raises."""
    spec = candidate.get("spec", candidate)
    scheduling = spec.get("scheduling") if isinstance(spec.get("scheduling"), dict) else {}
    priority = scheduling.get("priority", 0)
    priority = priority if type(priority) is int else 0
    position = scheduling.get("order")
    position = position if type(position) is int else None
    job_id = candidate.get("job_id", spec.get("job_id"))
    return (-priority, position is None, position if position is not None else 0, str(job_id))


def order(cands):
    """Deterministic order: any permutation of the input gives the same output."""
    return sorted(cands, key=sort_key)


# ---------------------------------------------------------------------- explain

def plan(records, profile, worker, *, latest=None, now=None, card=None, idle_since_mono=None, now_utc=None):
    """Steps 0-5 for every record, read-only. ``profile=None`` = legacy mode
    (no node gates; only legacy jobs, decided exactly as ``compatible()``)."""
    records = list(records)
    by_id = {record["job_id"]: record for record in records}
    if profile is None:
        node_checks = [_check("mode", OK, "legacy (no node profile): node gates not applicable")]
    else:
        node_checks = node_gate_checks(latest, profile, now, worker=worker, card=card, idle_since_mono=idle_since_mono)
    node_ok, node_reason = _first_failure(node_checks)
    # Capability rules read the profile; an invalid profile is reported once, at node level.
    profile_usable = profile is None or node_checks[0]["reason"] != "node_profile_invalid"
    sentinel_usable = any(item["check"] == "sentinel" and item["ok"] for item in node_checks)
    jobs = []
    for record in order(records):
        spec = record["spec"]
        req = job_requirements(spec)
        checks = []
        if record["state"] != "pending":
            checks.append(_check("state", "not_pending", record["state"]))
        else:
            refusal = job_start_refusal(spec, now_utc, worker.get("node_id"))
            checks.append(_check("seed_request_start", refusal or OK))
            deps = [dep for dep in spec.get("dependencies", []) if by_id.get(dep, {}).get("state") != "succeeded"]
            checks.append(_check("dependencies", OK if not deps else "dependency_not_succeeded", deps or None))
            if not profile_usable:
                checks.append(_check("static", "not_evaluated", "node profile invalid"))
            else:
                checks.extend(static_checks(req, profile, worker, spec))
            if profile is not None and profile_usable and _first_failure(checks)[0]:
                if sentinel_usable:
                    checks.append(_check("job_disk", *_disk_reason(req, latest, profile)))
                else:
                    checks.append(_check("job_disk", "not_evaluated", "sentinel latest.json unusable"))
        ok, reason = _first_failure(checks)
        jobs.append({"job_id": record["job_id"], "state": record["state"], "eligible": ok and node_ok,
                     "reason": reason if not ok else (OK if node_ok else node_reason),
                     "kind": requirements_kind(req)[0], "checks": checks})
    return {"mode": "legacy" if profile is None else "fleet", "node": {"ok": node_ok, "reason": node_reason,
            "checks": node_checks}, "jobs": jobs, "order": [job["job_id"] for job in jobs if job["eligible"]]}


def _disk_reason(req, latest, profile):
    _, reason = job_disk_gate(req, latest, profile)
    _, free = _runs_free(latest)
    return reason, {"free_bytes": free, "reserve_bytes": _runs_reserve(profile),
                    "min_free_disk_bytes": req.get("min_free_disk_bytes") or 0}


def explain(records, profile, worker, *, latest=None, now=None, card=None, idle_since_mono=None, job_id=None, now_utc=None):
    """``atlas fleet explain``: ``plan`` restricted to one job if ``job_id`` is given."""
    report = plan(records, profile, worker, latest=latest, now=now, card=card, idle_since_mono=idle_since_mono, now_utc=now_utc)
    if job_id is not None:
        report["jobs"] = [job for job in report["jobs"] if job["job_id"] == job_id]
        report["order"] = [item for item in report["order"] if item == job_id]
    return report


def format_explain(report):
    """Human-readable rendering of ``explain``; one line per check, stable codes."""
    def line(item):
        mark = "PASS" if item["ok"] else ("SKIP" if item["reason"] == "not_evaluated" else "FAIL")
        detail = "" if item["detail"] is None else f"  {item['detail']}"
        return f"    [{mark}] {item['check']}: {item['reason']}{detail}"
    out = [f"mode: {report['mode']}",
           f"node: {'PASS' if report['node']['ok'] else 'BLOCKED'} ({report['node']['reason']})"]
    out += [line(item) for item in report["node"]["checks"]]
    if not report["jobs"]:
        out.append("jobs: none")
    for job in report["jobs"]:
        out.append(f"job {job['job_id']} [{job['state']}, {job['kind']}]: "
                   f"{'ELIGIBLE' if job['eligible'] else 'SKIP'} ({job['reason']})")
        out += [line(item) for item in job["checks"]]
    out.append("claim order: " + (", ".join(item[:12] for item in report["order"]) or "none"))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------- seed

def seed_requirements(cfg, *, phase, image_id, weights_sha256=None, image_ref=None, min_vram_mib=1000,
                      min_compute_capability=None, hardware_class=None, allowed_nodes=None,
                      min_free_disk_bytes=0, container=None, expected_epoch_seconds_max=None):
    """``atlas.requirements.v1`` for a job seeded from ``cfg`` (validated).

    The dataset triple comes from the config (``manifest_sha256`` is the private
    manifest file hash the trainer already verifies; null for synthetic data).
    """
    if phase == "final" and hardware_class is None:
        raise ValueError("final jobs must pin requirements.hardware_class (decision P4)")
    req = {"schema": fs.SCHEMA_REQUIREMENTS, "gpu_required": True, "min_vram_mib": min_vram_mib,
           "min_compute_capability": min_compute_capability, "hardware_class": hardware_class,
           "allowed_nodes": allowed_nodes, "image_id": image_id, "image_ref": image_ref,
           "dataset": {"id": cfg["dataset"]["id"], "version": cfg["dataset"]["version"],
                       "manifest_sha256": cfg["dataset"].get("manifest_sha256")},
           "weights_sha256": weights_sha256, "min_free_disk_bytes": min_free_disk_bytes,
           "container": container, "expected_epoch_seconds_max": expected_epoch_seconds_max}
    try:
        return fs.validate_requirements(req)
    except fs.SchemaError as exc:
        raise ValueError(f"invalid seed requirements: {exc}") from exc


def with_requirements(spec, requirements, scheduling=None):
    """Additive fleet-v1 fields on a pre-fleet spec, without touching identity.

    ``compatibility.image_id``/``min_memory_mb`` mirror the requirements so an old
    coordinator (which only reads ``compatibility``) still enforces image and VRAM.
    """
    spec = copy.deepcopy(spec)
    spec["requirements"] = copy.deepcopy(requirements)
    compatibility = dict(spec.get("compatibility") or {})
    compatibility.update(image_id=requirements["image_id"], min_memory_mb=requirements["min_vram_mib"])
    spec["compatibility"] = compatibility
    if scheduling is not None:
        fs.validate_scheduling(scheduling)
        spec["scheduling"] = dict(scheduling)
    return spec


# ------------------------------------------------------------------ liveness (F5b)
# docs/fleet-protocol.md 4.3/4.4, ADR-3: derived by the observer, never persisted.
# Pure: no Git, no clock (``now_utc`` is injected), no write, never calls claim/retry/update.

HEARTBEAT_SECONDS = 900
DEFAULT_LEASE_SECONDS = 4 * HEARTBEAT_SECONDS
CLOCK_SKEW_SECONDS = 120
LIVENESS_STATES = ("n/a", "legacy_no_lease", "unknown_observer", "owner_rebooted", "alive", "late", "suspect_stale")
LIVE_RECORD_STATES = frozenset({"claimed", "running"})


def parse_utc(value):
    """Aware ``datetime`` or ISO-8601 UTC string (``Z`` or ``+00:00``) -> aware UTC datetime."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value:
        moment = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    else:
        raise ValueError("timestamp must be an ISO-8601 UTC string or an aware datetime")
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamp must carry a UTC offset (naive time refused)")
    return moment.astimezone(timezone.utc)


def _event_time(event):
    for key in ("observed_at_utc", "at"):
        if event.get(key):
            try:
                return parse_utc(event[key])
            except ValueError:
                continue
    return None


def _attempt_claim(record):
    attempt = record.get("attempt_id")
    for event in reversed(record.get("history") or []):
        if event.get("kind") == "claim" and event.get("attempt_id") == attempt:
            return event
    return None


def _result(state, age_s, **evidence):
    return {"state": state, "age_s": None if age_s is None else round(age_s, 3), "evidence": evidence}


def liveness(record, heartbeats_by_ref, all_worker_heartbeats=None, now_utc=None, observer_fetch_ok=True, *,
             heartbeat_seconds=HEARTBEAT_SECONDS, skew_seconds=CLOCK_SKEW_SECONDS):
    """Derived liveness of one job record (section 4.3) -> ``{state, age_s, evidence}``.

    ``heartbeats_by_ref``: ``{ref: heartbeat doc}`` as returned by
    ``heartbeat.fetch_heartbeats``; ``all_worker_heartbeats`` defaults to the same
    mapping (the other workers used by the 50 % observer rule); ``now_utc`` is the
    observer's UTC clock (required); ``observer_fetch_ok`` is ``False`` when the
    observer could not read the heartbeat refs.

    Boundaries (all inclusive on the healthy side, F5b clarification in 4.3):
    ``age <= 2*heartbeat`` alive, ``age <= lease`` late, ``age > lease`` suspect_stale
    (or unknown_observer). A heartbeat up to ``skew_seconds`` in the future counts as
    ``age 0``; further in the future, or ``ntp_synchronized`` not ``true``, gives
    ``late`` with ``evidence.clock_untrusted``.
    """
    if now_utc is None:
        raise ValueError("now_utc is required (the observer clock is injected)")
    now = parse_utc(now_utc)
    if heartbeat_seconds <= 0 or skew_seconds < 0:
        raise ValueError("heartbeat_seconds must be > 0 and skew_seconds >= 0")
    heartbeats_by_ref = heartbeats_by_ref or {}
    others_source = heartbeats_by_ref if all_worker_heartbeats is None else all_worker_heartbeats
    if record.get("state") not in LIVE_RECORD_STATES:
        return _result("n/a", None, record_state=record.get("state"))
    claim = _attempt_claim(record)
    block = (claim or {}).get("fleet")
    if not isinstance(block, dict):
        history = record.get("history") or []
        last = _event_time(history[-1]) if history else None
        return _result("legacy_no_lease", None if last is None else (now - last).total_seconds(),
                       last_event_utc=None if last is None else last.isoformat())
    ref, lease = block["heartbeat_ref"], block.get("lease_seconds", DEFAULT_LEASE_SECONDS)
    base = {"heartbeat_ref": ref, "lease_seconds": lease, "heartbeat_seconds": heartbeat_seconds}
    if observer_fetch_ok is not True:
        return _result("unknown_observer", None, observer_fetch_ok=False, **base)
    claimed_at = _event_time(claim)
    hb = heartbeats_by_ref.get(ref)
    if hb is not None and (hb.get("node_id"), hb.get("worker")) != tuple(ref.rsplit("/", 2)[-2:]):
        base["heartbeat_ignored"] = "document_of_another_worker"
        hb = None
    hb_at = None
    if hb is not None:
        hb_at = parse_utc(hb["observed_at_utc"])
        # A heartbeat older than the claim says nothing about this attempt's boot (4.3, F5b).
        predates_claim = claimed_at is not None and hb_at < claimed_at - timedelta(seconds=skew_seconds)
        base.update(seq=hb.get("seq"), heartbeat_utc=hb["observed_at_utc"], heartbeat_predates_claim=predates_claim)
        if not predates_claim and hb.get("boot_id") != block.get("boot_id"):
            return _result("owner_rebooted", (now - hb_at).total_seconds(), boot_id_at_claim=block.get("boot_id"),
                           boot_id_heartbeat=hb.get("boot_id"), **base)
    else:
        base["heartbeat_missing"] = True
    # Last sign of life of this attempt: the newer of its own heartbeat and its claim event.
    candidates = [moment for moment in (hb_at, claimed_at) if moment is not None]
    if not candidates:
        return _result("unknown_observer", None, reason="no_timestamp", **base)
    reference = max(candidates)
    age = (now - reference).total_seconds()
    if hb is not None and hb.get("ntp_synchronized") is not True:
        return _result("late", max(age, 0.0), clock_untrusted=True, reason="ntp_not_synchronized", **base)
    if age < -skew_seconds:
        return _result("late", 0.0, clock_untrusted=True, reason="future_timestamp", future_s=round(-age, 3), **base)
    if age < 0:
        base["future_s_tolerated"] = round(-age, 3)
        age = 0.0
    if age <= lease:
        return _result("late" if age > 2 * heartbeat_seconds else "alive", age, **base)
    node_id = ref.rsplit("/", 2)[-2]
    others = [doc for other_ref, doc in sorted((others_source or {}).items())
              if other_ref != ref and isinstance(doc, dict) and doc.get("node_id") != node_id]
    expired = sum(1 for doc in others if (now - parse_utc(doc["observed_at_utc"])).total_seconds() > lease)
    base.update(other_node_workers=len(others), other_node_workers_expired=expired)
    if others and 2 * expired >= len(others):
        return _result("unknown_observer", age, reason="most_other_nodes_expired", **base)
    return _result("suspect_stale", age, **base)
