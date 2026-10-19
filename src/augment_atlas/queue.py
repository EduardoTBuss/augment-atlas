"""Conservative multi-writer queue using ordinary Git fast-forward pushes.

Training code lives in a different checkout. No merge, rebase, reset or forced
push is performed here. Every successful mutation is verified by fetching again.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import uuid

from .schema import JOB_ID_RE, TERMINAL_STATES, canonical_json, digest, new_record, prepare_job, validate_record
from . import fleet as fleet_rules
from . import fleet_schema


class QueueError(RuntimeError):
    pass


class QueueConflict(QueueError):
    pass


class StaleAttempt(QueueConflict):
    pass


class InvalidTransition(QueueConflict):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def render_record(record):
    spec = record["spec"]
    owner = record.get("owner") or {}
    latest = record["history"][-1] if record["history"] else {}
    progress = next((event for event in reversed(record["history"])
                     if event.get("attempt_id") == record.get("attempt_id") and "epoch" in event), {})
    # Keep human-facing values on one line; JSON is the authoritative payload.
    clean = lambda value: str(value).replace("\n", " ").replace("\r", " ")
    heading = f"# Job {record['job_id']}\n\n"
    summary = (f"State: **{record['state']}** | phase: {clean(spec['phase'])} | "
               f"fold: {spec['fold']} | train seed: {spec['train_seed']} | arm: {clean(spec['arm'])}\n\n"
               f"Attempt: {clean(record.get('attempt_id'))} | owner: {clean(owner)} | "
               f"last observation UTC: {clean(latest.get('observed_at_utc', latest.get('at', 'none')))} | sequence: {record['seq']}\n\n")
    if progress:
        summary += (f"Epoch: {clean(progress['epoch'])}/{clean(progress.get('total_epochs', progress.get('epochs', '?')))} | "
                    f"progress observed UTC: {clean(progress.get('observed_at_utc', progress['at']))}\n\n")
    return heading + summary + "<!-- augment-atlas-record-v1 -->\n```json\n" + json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n```\n"


def parse_record(markdown):
    marker = "<!-- augment-atlas-record-v1 -->\n```json\n"
    if markdown.count(marker) != 1:
        raise QueueError("missing or ambiguous structured job record")
    raw = markdown.split(marker, 1)[1].split("\n```", 1)[0]
    try:
        return validate_record(json.loads(raw))
    except (ValueError, TypeError, KeyError) as exc:
        raise QueueError(f"invalid structured job record: {exc}") from exc


class GitQueue:
    def __init__(self, checkout, remote, branch="queue", max_retries=16):
        self.checkout = Path(checkout).resolve()
        self.remote = str(remote)
        self.branch = branch
        self.max_retries = max_retries
        self._lock = threading.RLock()
        self.checkout.mkdir(parents=True, exist_ok=True)
        if not (self.checkout / ".git").exists():
            self._git("init")
        if self._git("check-ref-format", f"refs/heads/{branch}", check=False).returncode:
            raise ValueError("invalid queue branch")

    def _git(self, *args, input=None, check=True, env=None):
        settings = os.environ.copy()
        settings.update({"GIT_TERMINAL_PROMPT": "0", "GIT_AUTHOR_NAME": "AugmentAtlas queue",
                         "GIT_AUTHOR_EMAIL": "queue@example.invalid", "GIT_COMMITTER_NAME": "AugmentAtlas queue",
                         "GIT_COMMITTER_EMAIL": "queue@example.invalid"})
        if env:
            settings.update(env)
        try:
            result = subprocess.run(["git", "-C", str(self.checkout), *args], input=input,
                                    capture_output=True, text=True, encoding="utf-8", env=settings, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QueueError(f"Git operation unavailable: {type(exc).__name__}") from exc
        if check and result.returncode:
            # Avoid leaking authenticated remote URLs via Git's stderr.
            raise QueueError(f"Git {args[0]} failed (exit {result.returncode})")
        return result

    def _fetch_head(self):
        refs = self._git("ls-remote", "--heads", self.remote, f"refs/heads/{self.branch}").stdout.strip()
        if not refs:
            return None
        self._git("fetch", "--no-tags", self.remote, f"refs/heads/{self.branch}")
        return self._git("rev-parse", "FETCH_HEAD").stdout.strip()

    def _records(self, head):
        if head is None:
            return {}
        paths = self._git("ls-tree", "-r", "--name-only", head, "jobs/").stdout.splitlines()
        records = {}
        for path in paths:
            if not path.endswith(".md"):
                continue
            record = parse_record(self._git("show", f"{head}:{path}").stdout)
            if path != f"jobs/{record['job_id']}.md":
                raise QueueError("job path does not match structured identity")
            records[record["job_id"]] = record
        return records

    def list_jobs(self):
        with self._lock:
            return sorted(self._records(self._fetch_head()).values(), key=lambda record: record["job_id"])

    def _commit(self, head, records, message):
        # A private temporary index prevents unrelated checkout/index mutation.
        with tempfile.TemporaryDirectory(prefix="augment-atlas-index-") as directory:
            env = {"GIT_INDEX_FILE": str(Path(directory) / "index")}
            self._git("read-tree", head if head else "--empty", env=env)
            for record in records:
                data = render_record(record)
                blob = self._git("hash-object", "-w", "--stdin", input=data).stdout.strip()
                self._git("update-index", "--add", "--cacheinfo", f"100644,{blob},jobs/{record['job_id']}.md", env=env)
            tree = self._git("write-tree", env=env).stdout.strip()
            args = ["commit-tree", tree]
            if head:
                args.extend(["-p", head])
            return self._git(*args, input=message + "\n").stdout.strip()

    def _push(self, commit):
        # Normal push rejects a concurrent, divergent child of the observed HEAD.
        return self._git("push", self.remote, f"{commit}:refs/heads/{self.branch}", check=False).returncode == 0

    def _publish(self, head, changed, message):
        commit = self._commit(head, changed, message)
        try:
            self._push(commit)
        except QueueError:
            # An accepted push can lose its response. Only refetch can decide.
            pass
        return self._records(self._fetch_head())

    def seed(self, specs):
        prepared = [prepare_job(spec) for spec in specs]
        if len({spec["job_id"] for spec in prepared}) != len(prepared):
            raise ValueError("duplicate job IDs in seed")
        with self._lock:
            for _ in range(self.max_retries):
                head = self._fetch_head()
                records = self._records(head)
                for spec in prepared:
                    old = records.get(spec["job_id"])
                    if old and old["spec"] != spec:
                        raise QueueConflict("stable identity already has a different immutable spec")
                combined = {**records, **{spec["job_id"]: new_record(spec) for spec in prepared if spec["job_id"] not in records}}
                self._check_dependencies(combined)
                additions = [new_record(spec) for spec in prepared if spec["job_id"] not in records]
                if not additions:
                    return [records[spec["job_id"]] for spec in prepared]
                confirmed = self._publish(head, additions, "queue: add jobs")
                if all(spec["job_id"] in confirmed for spec in prepared):
                    # A concurrent seed with the same identity must match in full.
                    if any(confirmed[spec["job_id"]]["spec"] != spec for spec in prepared):
                        raise QueueConflict("concurrent seed changed immutable spec")
                    return [confirmed[spec["job_id"]] for spec in prepared]
            raise QueueConflict("seed exhausted concurrent retries")

    @staticmethod
    def _check_dependencies(records):
        visiting, visited = set(), set()
        def visit(job_id):
            if job_id in visiting:
                raise ValueError("dependency cycle")
            if job_id in visited:
                return
            if job_id not in records:
                raise ValueError("dependency does not exist")
            visiting.add(job_id)
            for dependency in records[job_id]["spec"]["dependencies"]:
                visit(dependency)
            visiting.remove(job_id)
            visited.add(job_id)
        for job_id in records:
            visit(job_id)

    @staticmethod
    def _get(records, job_id):
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise ValueError("invalid job ID")
        if job_id not in records:
            raise QueueError("job does not exist")
        return records[job_id]

    @staticmethod
    def _append(record, event):
        record["version"] += 1
        record["seq"] = record["version"]
        event = {**event, "seq": record["seq"], "at": utc_now()}
        record["history"].append(event)
        return validate_record(record)

    @staticmethod
    def _check_fleet_claim(owner, fleet):
        """Pure validation of a fleet claim (docs/fleet-protocol.md 3.2c/d); no Git I/O.

        Returns a private deep copy so later caller mutation cannot alter what is published.
        """
        try:
            block = fleet_schema.validate_claim_fleet(copy.deepcopy(fleet))
        except fleet_schema.SchemaError as exc:
            raise ValueError(f"invalid fleet claim block: {exc}") from exc
        node_id, worker = owner.get("node_id"), owner.get("worker")
        if any(not isinstance(value, str) or not value for value in (node_id, worker)):
            raise ValueError("fleet claim requires owner.node_id and owner.worker")
        # reconcile_stopped_attempt compares owner.server with settings.server (3.2c).
        if owner.get("server") != node_id:
            raise ValueError("fleet claim requires owner.server == owner.node_id")
        # Single writer per worker: the lease is read from this node's own ref (D3, 4.2).
        if block["heartbeat_ref"] != f"refs/heads/fleet-hb/{node_id}/{worker}":
            raise ValueError("fleet.heartbeat_ref does not belong to owner node/worker")
        return block

    def claim(self, job_id, owner, fleet=None):
        if not isinstance(owner, dict) or not owner:
            raise ValueError("owner must be a nonempty public alias object")
        canonical_json(owner)
        if fleet is not None:
            fleet = self._check_fleet_claim(owner, fleet)
        attempt_id, token = uuid.uuid4().hex, uuid.uuid4().hex
        with self._lock:
            for _ in range(self.max_retries):
                head = self._fetch_head()
                records = self._records(head)
                current = self._get(records, job_id)
                if current["attempt_id"] == attempt_id and current["token"] == token:
                    return current
                if current["state"] != "pending" or any(records[dependency]["state"] != "succeeded" for dependency in current["spec"]["dependencies"]):
                    return None
                event = {"kind": "claim", "attempt_id": attempt_id, "token": token, "owner": copy.deepcopy(owner), "state": "claimed"}
                if fleet is not None:
                    # The claim binds the node's plan to this immutable spec (F5, 3.3): refuse before writing.
                    if fleet["requirements_sha256"] != digest(fleet_rules.job_requirements(current["spec"])):
                        raise ValueError("fleet.requirements_sha256 does not match the job requirements")
                    event["fleet"] = copy.deepcopy(fleet)
                record = copy.deepcopy(current)
                record.update(state="claimed", attempt_id=attempt_id, token=token, owner=copy.deepcopy(owner))
                self._append(record, event)
                confirmed = self._publish(head, [record], f"queue: claim {job_id}")
                current = self._get(confirmed, job_id)
                if current["attempt_id"] == attempt_id and current["token"] == token:
                    return current
                if current["state"] != "pending":
                    return None
            raise QueueConflict("claim exhausted concurrent retries")

    def update(self, job_id, attempt_id, token, expected_version, event):
        event = copy.deepcopy(event)
        if not isinstance(event, dict) or not isinstance(event.get("event_id"), str) or not event["event_id"]:
            raise ValueError("event requires a durable unique event_id")
        if event.get("state") not in {"running", *TERMINAL_STATES}:
            raise ValueError("event requires a supported target state")
        if any(key in event for key in {"seq", "at", "attempt_id", "token", "payload_hash", "kind"}):
            raise ValueError("event uses reserved metadata")
        payload_hash = digest(event)
        with self._lock:
            for _ in range(self.max_retries):
                head = self._fetch_head()
                records = self._records(head)
                current = self._get(records, job_id)
                if current["attempt_id"] != attempt_id or current["token"] != token:
                    raise StaleAttempt("attempt is no longer the owner")
                for old in current["history"]:
                    if old.get("event_id") == event["event_id"]:
                        if old.get("payload_hash") != payload_hash or old.get("attempt_id") != attempt_id:
                            raise QueueConflict("event_id reused with different content")
                        return current
                if current["version"] != expected_version:
                    raise QueueConflict("state version changed; reconcile before retrying")
                manual_resume = (current["state"] == "interrupted" and event["state"] == "running"
                                 and event.get("manual_resume") is True)
                if (current["state"] in TERMINAL_STATES and not manual_resume) or current["state"] == "pending":
                    raise InvalidTransition("terminal/pending state cannot accept attempt updates")
                if event.get("manual_resume") and not manual_resume:
                    raise InvalidTransition("manual_resume is only valid from interrupted to running")
                if current["state"] == "claimed" and event["state"] == "succeeded":
                    raise InvalidTransition("success requires a running observation first")
                record = copy.deepcopy(current)
                record["state"] = event["state"]
                self._append(record, {**event, "kind": "update", "attempt_id": attempt_id, "payload_hash": payload_hash})
                confirmed = self._publish(head, [record], f"queue: {event['state']} {job_id}")
                latest = self._get(confirmed, job_id)
                if latest["attempt_id"] != attempt_id or latest["token"] != token:
                    raise StaleAttempt("attempt changed during publication")
                if any(item.get("event_id") == event["event_id"] and item.get("payload_hash") == payload_hash for item in latest["history"]):
                    return latest
            raise QueueConflict("update exhausted concurrent retries")

    def retry(self, job_id, expected_version, reason):
        """Manual recovery only after operators have confirmed the old child stopped."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("manual retry requires an audit reason")
        recovery_id = uuid.uuid4().hex
        with self._lock:
            for _ in range(self.max_retries):
                head = self._fetch_head()
                records = self._records(head)
                current = self._get(records, job_id)
                if any(item.get("recovery_id") == recovery_id for item in current["history"]):
                    return current
                if current["version"] != expected_version:
                    raise QueueConflict("state version changed")
                if current["state"] not in {"failed", "cancelled", "interrupted"}:
                    raise InvalidTransition("only failed/cancelled/interrupted jobs can be retried manually")
                record = copy.deepcopy(current)
                previous_attempt = record["attempt_id"]
                record.update(state="pending", attempt_id=None, token=None, owner=None)
                self._append(record, {"kind": "manual_retry", "state": "pending", "reason": reason,
                                      "previous_attempt_id": previous_attempt, "recovery_id": recovery_id})
                confirmed = self._publish(head, [record], f"queue: manual retry {job_id}")
                latest = self._get(confirmed, job_id)
                if any(item.get("recovery_id") == recovery_id for item in latest["history"]):
                    return latest
            raise QueueConflict("manual retry exhausted concurrent retries")
