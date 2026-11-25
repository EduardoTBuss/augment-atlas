"""Versioned, JSON-only public queue contracts; no training dependencies."""
from __future__ import annotations

import copy
import hashlib
import json
import re

IDENTITY_FIELDS = (
    "protocol_version", "dataset", "dataset_version", "split_hash", "recipe_hash", "phase",
    "fold", "train_seed", "arm", "code_commit",
)
STATES = frozenset({"pending", "claimed", "running", "succeeded", "failed", "cancelled", "interrupted"})
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
JOB_ID_RE = re.compile(r"^[a-f0-9]{64}$")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def prepare_job(spec):
    """Validate a spec and derive its stable ID without modifying the caller."""
    spec = copy.deepcopy(spec)
    if not isinstance(spec, dict):
        raise ValueError("job spec must be an object")
    for key in IDENTITY_FIELDS:
        value = spec.get(key)
        if key in {"fold", "train_seed"}:
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string")
    if spec["phase"] not in {"smoke", "hpo", "final"}:
        raise ValueError("phase must be explicit: smoke, hpo or final")
    expected = digest({key: spec[key] for key in IDENTITY_FIELDS})
    if spec.get("job_id", expected) != expected:
        raise ValueError("job_id does not match stable identity")
    spec["job_id"] = expected
    dependencies = spec.setdefault("dependencies", [])
    if not isinstance(dependencies, list) or any(not isinstance(item, str) or not JOB_ID_RE.fullmatch(item) for item in dependencies):
        raise ValueError("dependencies must be a list of job IDs")
    if len(set(dependencies)) != len(dependencies) or expected in dependencies:
        raise ValueError("duplicate or self dependency")
    if not isinstance(spec.setdefault("compatibility", {}), dict):
        raise ValueError("compatibility must be an object")
    canonical_json(spec)
    return spec


def new_record(spec):
    spec = prepare_job(spec)
    return {"schema_version": 1, "job_id": spec["job_id"], "spec": spec,
            "state": "pending", "version": 0, "seq": 0, "attempt_id": None,
            "token": None, "owner": None, "history": []}


def validate_record(record):
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("unsupported queue record schema")
    spec = prepare_job(record.get("spec"))
    if record.get("job_id") != spec["job_id"]:
        raise ValueError("record identity mismatch")
    if record.get("state") not in STATES:
        raise ValueError("invalid queue state")
    if type(record.get("version")) is not int or record["version"] < 0:
        raise ValueError("invalid version")
    if type(record.get("seq")) is not int or record["seq"] != record["version"] or not isinstance(record.get("history"), list):
        raise ValueError("invalid event sequence")
    if len(record["history"]) != record["version"]:
        raise ValueError("history and version differ")
    if record["state"] != "pending":
        if any(not isinstance(record.get(key), str) or not record[key] for key in ("attempt_id", "token")):
            raise ValueError("owned state lacks attempt identity")
        if not isinstance(record.get("owner"), dict) or not record["owner"]:
            raise ValueError("owned state lacks owner object")
    elif any(record.get(key) is not None for key in ("attempt_id", "token", "owner")):
        raise ValueError("pending state must not retain ownership")
    for seq, event in enumerate(record["history"], 1):
        if not isinstance(event, dict) or type(event.get("seq")) is not int or event["seq"] != seq:
            raise ValueError("invalid history sequence")
        if event.get("state") not in STATES or not isinstance(event.get("at"), str) or not event["at"]:
            raise ValueError("invalid history observation")
    if record["history"] and record["history"][-1]["state"] != record["state"]:
        raise ValueError("latest history state differs from snapshot")
    canonical_json(record)
    return record


def job_start_refusal(spec, now, node_id):
    """Pure start gate for file-request jobs; callers supply the actual UTC clock.

    A running container may finish after the deadline; every new start or resume
    must pass this gate. Missing/invalid bindings fail closed.
    """
    binding = spec.get("seed_request")
    if binding is None:
        return None
    from datetime import datetime, timezone
    try:
        required = {"request_id", "request_sha256", "node_id", "not_after_utc",
                    "stage_sha256", "anchor_commit", "environment_sha256", "node_profile_sha256"}
        if type(binding) is not dict or set(binding) != required:
            return "invalid_seed_request"
        for key in ("request_sha256", "stage_sha256", "environment_sha256", "node_profile_sha256"):
            if not isinstance(binding[key], str) or not JOB_ID_RE.fullmatch(binding[key]):
                return "invalid_seed_request"
        if (spec.get("requirements", {}).get("allowed_nodes") != [binding["node_id"]]
                or spec.get("environment_sha256") != binding["environment_sha256"]):
            return "invalid_seed_request"
        stamp = binding["not_after_utc"]
        if not isinstance(stamp, str) or not stamp.endswith("Z"):
            return "invalid_seed_request"
        deadline = datetime.fromisoformat(stamp[:-1] + "+00:00")
        if now is None or not isinstance(now, datetime) or now.utcoffset() is None:
            return "start_time_unknown"
        if now.astimezone(timezone.utc) >= deadline:
            return "expired"
        if node_id != binding["node_id"]:
            return "node_not_allowed"
    except (ValueError, TypeError, KeyError, AttributeError):
        return "invalid_seed_request"
    return None
