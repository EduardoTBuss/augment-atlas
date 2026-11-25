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
import re
import subprocess
import tempfile
import threading
import uuid

from .schema import JOB_ID_RE, TERMINAL_STATES, canonical_json, digest, new_record, prepare_job, validate_record, job_start_refusal
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


#: Additive documents outside ``jobs/`` (3.2g): ``_records()`` never lists them.
DOCUMENT_PATH = re.compile(r"^fleet/(?:[a-z0-9][a-z0-9_-]{0,63}/){0,3}[a-z0-9][a-z0-9_-]{0,63}\.json$")
NODE_CARD_PATH = re.compile(r"^fleet/nodes/([a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?)\.json$")


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


# --------------------------------------------------------------------------
# Public hash anchors (global protocol, slice S3). Additive: jobs/ records,
# IDENTITY_FIELDS, the record envelope, states and retry are unchanged.
# --------------------------------------------------------------------------

ANCHOR_PROTOCOL = "global-anchor-v1"
ANCHOR_KINDS = ("pools", "recipe_freeze", "campaign", "stage_authorization")
# Recipe source of a freeze/campaign anchor (S3). One dataset/pools may hold one
# campaign per source (INbreast, 20/10/2026: external fixed recipe AND an
# exploratory HPO on D); the anchor slot is (kind, pools_sha256, recipe_source).
ANCHOR_RECIPE_SOURCES = ("hpo_d", "external_fixed")
ANCHOR_KEYS = frozenset({"schema_version", "protocol", "kind", "dataset", "pools_sha256", "document_sha256",
                         "freeze_sha256", "campaign_sha256", "recipe_source", "code_commit", "label",
                         "document_utc", "caveat", "sha256"})
# Verbatim public caveat of spec 6.9 (Bradshaw 2023, DOI 10.1148/ryai.220232).
ANCHOR_CAVEAT = ("Reembaralhar folds não remove viés se pacientes usados na seleção caírem no teste "
                 "(Bradshaw 2023, DOI 10.1148/ryai.220232); D fica fora de todos os testes.")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_ANCHOR_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _anchor_fail(code, message):
    raise ValueError(f"{code}: {message}")


def validate_anchor(anchor):
    """Closed public anchor (hashes only: no record IDs, groups or private paths)."""
    expected_keys = ANCHOR_KEYS | ({"stage"} if isinstance(anchor, dict) and anchor.get("kind") == "stage_authorization" else set())
    if type(anchor) is not dict or set(anchor) != expected_keys:
        _anchor_fail("anchor_invalid", "anchor must have exactly the global-anchor-v1 keys")
    if anchor["schema_version"] != 1 or type(anchor["schema_version"]) is not int or anchor["protocol"] != ANCHOR_PROTOCOL:
        _anchor_fail("anchor_invalid", "only global-anchor-v1 anchors are supported")
    kind = anchor["kind"]
    if kind not in ANCHOR_KINDS:
        _anchor_fail("anchor_invalid", "unknown anchor kind")
    dataset = anchor["dataset"]
    if (type(dataset) is not dict or set(dataset) != {"id", "version"}
            or any(type(dataset[k]) is not str or not _ANCHOR_NAME.fullmatch(dataset[k]) for k in dataset)):
        _anchor_fail("anchor_invalid", "dataset must be {id, version} safe names")
    for key in ("pools_sha256", "document_sha256", "sha256"):
        if type(anchor[key]) is not str or not _HEX64.fullmatch(anchor[key]):
            _anchor_fail("anchor_invalid", f"{key} must be a lowercase hex64")
    for key in ("freeze_sha256", "campaign_sha256"):
        if anchor[key] is not None and (type(anchor[key]) is not str or not _HEX64.fullmatch(anchor[key])):
            _anchor_fail("anchor_invalid", f"{key} must be null or a lowercase hex64")
    if anchor["code_commit"] is not None and (type(anchor["code_commit"]) is not str
                                              or not _HEX40.fullmatch(anchor["code_commit"])):
        _anchor_fail("anchor_invalid", "code_commit must be null or a full commit")
    if anchor["label"] not in (None, "exploratory", "confirmatory_candidate"):
        _anchor_fail("anchor_invalid", "unknown label")
    stamp = anchor["document_utc"]
    if type(stamp) is not str or not stamp.endswith("Z"):
        _anchor_fail("anchor_invalid", "document_utc must be an explicit UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        _anchor_fail("anchor_invalid", "document_utc must be ISO UTC")
    if anchor["caveat"] != ANCHOR_CAVEAT:
        _anchor_fail("anchor_invalid", "the Bradshaw 2023 selection-bias caveat is mandatory and verbatim")
    if kind == "pools":
        ok = (anchor["document_sha256"] == anchor["pools_sha256"] and anchor["freeze_sha256"] is None
              and anchor["campaign_sha256"] is None and anchor["code_commit"] is None and anchor["label"] is None
              and anchor["recipe_source"] is None)
    elif kind == "recipe_freeze":
        ok = (anchor["freeze_sha256"] == anchor["document_sha256"] and anchor["campaign_sha256"] is None
              and anchor["label"] is None and anchor["recipe_source"] in ANCHOR_RECIPE_SOURCES)
    elif kind == "stage_authorization":
        ok = (anchor["campaign_sha256"] is not None and anchor["freeze_sha256"] is not None
              and anchor["code_commit"] is not None and anchor["label"] is None
              and anchor["recipe_source"] in ANCHOR_RECIPE_SOURCES
              and type(anchor["stage"]) is int and anchor["stage"] >= 2)
    else:
        ok = (anchor["campaign_sha256"] == anchor["document_sha256"] and anchor["freeze_sha256"] is not None
              and anchor["code_commit"] is not None and anchor["label"] is not None
              and anchor["recipe_source"] in ANCHOR_RECIPE_SOURCES)
    if not ok:
        _anchor_fail("anchor_invalid", f"{kind} anchor fields are inconsistent with its kind")
    if digest({k: v for k, v in anchor.items() if k != "sha256"}) != anchor["sha256"]:
        _anchor_fail("anchor_invalid", "anchor checksum mismatch")
    return anchor


def anchor_slot(anchor):
    """At most one document per slot: (kind, pools_sha256, recipe_source)."""
    if anchor["kind"] == "stage_authorization":
        return anchor["kind"], anchor["campaign_sha256"], anchor["stage"]
    return anchor["kind"], anchor["pools_sha256"], anchor["recipe_source"]


def anchor_path(dataset_id, kind, document_sha256):
    if kind not in ANCHOR_KINDS or type(dataset_id) is not str or not _ANCHOR_NAME.fullmatch(dataset_id):
        raise ValueError("anchor_invalid: invalid anchor path parts")
    if type(document_sha256) is not str or not _HEX64.fullmatch(document_sha256):
        raise ValueError("anchor_invalid: document sha256 must be hex64")
    return f"anchors/{dataset_id}/{kind}-{document_sha256}.json"


def render_anchor(anchor):
    return json.dumps(anchor, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def _is_global_final(spec):
    return spec.get("phase") == "final" and isinstance(spec.get("global"), dict)


def _is_global_hpo(spec):
    return spec.get("phase") == "hpo" and isinstance(spec.get("global"), dict)


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

    def _git(self, *args, input=None, check=True, env=None, binary=False):
        settings = os.environ.copy()
        settings.update({"GIT_TERMINAL_PROMPT": "0", "GIT_AUTHOR_NAME": "AugmentAtlas queue",
                         "GIT_AUTHOR_EMAIL": "queue@example.invalid", "GIT_COMMITTER_NAME": "AugmentAtlas queue",
                         "GIT_COMMITTER_EMAIL": "queue@example.invalid"})
        if env:
            settings.update(env)
        try:
            # ``binary`` (documents only): exact bytes, no newline translation on Windows.
            text = {} if binary else {"text": True, "encoding": "utf-8"}
            result = subprocess.run(["git", "-C", str(self.checkout), *args], input=input,
                                    capture_output=True, env=settings, timeout=60, **text)
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
        if any(_is_global_final(spec) for spec in prepared):
            # Global final jobs need the campaign anchor in the parent commit.
            raise ValueError("campaign_anchor_missing: global final jobs are seeded only by seed_final")
        hpo_pools = {(spec["dataset"], spec["global"].get("pools_sha256")) for spec in prepared if _is_global_hpo(spec)}
        with self._lock:
            for _ in range(self.max_retries):
                head = self._fetch_head()
                if hpo_pools:
                    self._refuse_hpo_after_freeze(head, hpo_pools)
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

    def claim(self, job_id, owner, fleet=None, *, stage_context=None):
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
                refusal = job_start_refusal(current["spec"], datetime.now(timezone.utc), owner.get("server"))
                if refusal:
                    return None
                if _is_global_final(current["spec"]) and (stage_context is not None or "/stage-" in current["spec"].get("config", "")):
                    if stage_context is None:
                        raise QueueConflict("campaign_invalid: stage claim requires an audited stage context")
                    from . import global_campaign as gc
                    from . import stage_authorization as sa
                    campaign, authorization, code_repo = stage_context
                    if authorization is not None:
                        sa.audit(authorization, campaign, code_repo=code_repo)
                    # R1-01: seed eligibility comes from the audited campaign/authorization, not the path.
                    sa.job_binding(current["spec"], campaign, authorization)
                    gc.audit_chronology(self, campaign, head=head, stage_authorization=authorization, code_repo=code_repo,
                                        authorization_root=getattr(stage_context, "data_root", None))
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

    # ------------------------------------------------------------------
    # Global anchors and final seeding (slice S3, additive). Every mutation
    # uses the same fetch/commit/push/refetch loop as ``seed``.
    # ------------------------------------------------------------------

    def _git_bytes(self, *args, input=None):
        """Binary Git call: no newline translation (text-mode stdin turns LF into CRLF on Windows)."""
        settings = os.environ.copy()
        settings["GIT_TERMINAL_PROMPT"] = "0"
        try:
            result = subprocess.run(["git", "-C", str(self.checkout), *args], input=input, capture_output=True,
                                    env=settings, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QueueError(f"Git operation unavailable: {type(exc).__name__}") from exc
        if result.returncode:
            raise QueueError(f"Git {args[0]} failed (exit {result.returncode})")
        return result.stdout

    def read_path(self, head, path):
        """Exact UTF-8 text of ``path`` at commit ``head`` (no newline translation) or None."""
        if head is None:
            return None
        if self._git("cat-file", "-e", f"{head}:{path}", check=False).returncode:
            return None
        return self._git_bytes("cat-file", "blob", f"{head}:{path}").decode("utf-8")

    def list_anchors(self, head, dataset_id):
        """{path: validated anchor} under anchors/<dataset_id>/ at ``head``."""
        anchor_path(dataset_id, "pools", "0" * 64)  # validates the dataset id
        if head is None:
            return {}
        anchors = {}
        for path in self._git("ls-tree", "-r", "--name-only", head, f"anchors/{dataset_id}/").stdout.splitlines():
            try:
                anchor = validate_anchor(json.loads(self.read_path(head, path)))
            except (TypeError, ValueError) as error:
                raise QueueError(f"anchor_invalid: unreadable anchor {path}: {error}") from error
            if (path != anchor_path(dataset_id, anchor["kind"], anchor["document_sha256"])
                    or anchor["dataset"]["id"] != dataset_id):
                raise QueueError(f"anchor_invalid: anchor path does not match its content: {path}")
            anchors[path] = anchor
        return anchors

    def anchor_commit(self, head, path):
        """First first-parent commit reachable from ``head`` that introduced ``path``."""
        if head is None:
            return None
        commits = self._git("rev-list", "--first-parent", "--reverse", head, "--", path).stdout.split()
        return commits[0] if commits else None

    def _commit_files(self, head, files, message):
        # Same private-index technique as _commit; only the given paths change.
        with tempfile.TemporaryDirectory(prefix="augment-atlas-index-") as directory:
            env = {"GIT_INDEX_FILE": str(Path(directory) / "index")}
            self._git("read-tree", head if head else "--empty", env=env)
            for path, data in sorted(files.items()):
                blob = self._git_bytes("hash-object", "-w", "--stdin", input=data.encode("utf-8")).decode("ascii").strip()
                self._git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
            tree = self._git("write-tree", env=env).stdout.strip()
            args = ["commit-tree", tree]
            if head:
                args.extend(["-p", head])
            return self._git(*args, input=message + "\n").stdout.strip()

    @staticmethod
    def _final_jobs_of(records, pools_sha256):
        return [record for record in records.values()
                if _is_global_final(record["spec"]) and record["spec"]["global"].get("pools_sha256") == pools_sha256]

    @staticmethod
    def _hpo_jobs_of(records, pools_sha256):
        return [record for record in records.values()
                if _is_global_hpo(record["spec"]) and record["spec"]["global"].get("pools_sha256") == pools_sha256]

    def _refuse_hpo_after_freeze(self, head, hpo_pools):
        # Spec 4.2: no HPO job of these pools after its hpo_d C_freeze.
        for dataset_id, pools_sha256 in sorted(hpo_pools, key=repr):
            for path, anchor in self.list_anchors(head, dataset_id).items():
                if (anchor["kind"] == "recipe_freeze" and anchor["recipe_source"] == "hpo_d"
                        and anchor["pools_sha256"] == pools_sha256):
                    raise QueueConflict(f"hpo_job_after_freeze: an hpo_d freeze of these pools is anchored ({path})")

    def _check_anchor_preconditions(self, head, records, anchor):
        dataset_id = anchor["dataset"]["id"]
        for path, other in self.list_anchors(head, dataset_id).items():
            if anchor_slot(other) == anchor_slot(anchor) and other["document_sha256"] != anchor["document_sha256"]:
                raise QueueConflict(f"anchor_conflict: a different {anchor['kind']} anchor already holds this slot "
                                    f"(pools, recipe_source={anchor['recipe_source']}) ({path})")
        if anchor["kind"] == "pools":
            return
        if anchor["kind"] == "stage_authorization":
            raw = self.read_path(head, anchor_path(dataset_id, "campaign", anchor["campaign_sha256"]))
            if raw is None:
                raise QueueConflict("campaign_anchor_missing: stage authorization needs its original campaign anchor")
            campaign = validate_anchor(json.loads(raw))
            if any(campaign[key] != anchor[key] for key in ("dataset", "pools_sha256", "freeze_sha256", "recipe_source")):
                raise QueueConflict("anchor_conflict: stage authorization bound to another campaign")
            return
        # Both campaigns of one pools are sealed before ANY final job of the pools (spec 6.9).
        if self._final_jobs_of(records, anchor["pools_sha256"]):
            raise QueueConflict("final_job_before_seal: final jobs of these pools already exist in the queue")
        if anchor["kind"] == "recipe_freeze":
            hpo = self._hpo_jobs_of(records, anchor["pools_sha256"])
            if anchor["recipe_source"] == "external_fixed" and hpo:
                # The external recipe is anchored before any HPO score of these pools exists.
                raise QueueConflict("external_recipe_contaminated: HPO jobs of these pools already exist")
            if anchor["recipe_source"] == "hpo_d" and any(r["state"] not in TERMINAL_STATES for r in hpo):
                raise QueueConflict("hpo_job_after_freeze: HPO jobs of these pools are not terminal yet")
            return
        raw = self.read_path(head, anchor_path(dataset_id, "recipe_freeze", anchor["freeze_sha256"]))
        if raw is None:
            raise QueueConflict("freeze_anchor_missing: the campaign freeze is not anchored in the queue HEAD")
        freeze = validate_anchor(json.loads(raw))
        if (freeze["pools_sha256"] != anchor["pools_sha256"] or freeze["dataset"] != anchor["dataset"]
                or freeze["recipe_source"] != anchor["recipe_source"]):
            raise QueueConflict("anchor_conflict: freeze anchor is bound to other pools/dataset/recipe source")

    def publish_anchor(self, anchor):
        """Publish ``anchors/<ds>/<kind>-<document_sha256>.json`` (idempotent, no-clobber).

        Same bytes at the same path return the existing anchor; different
        bytes, or another document of the same kind for the same pools, raise
        ``QueueConflict("anchor_conflict: ...")``. Returns {path, commit, head}
        where ``commit`` is the first commit holding the anchor (C_freeze or
        C_campaign).
        """
        validate_anchor(anchor)
        path = anchor_path(anchor["dataset"]["id"], anchor["kind"], anchor["document_sha256"])
        data = render_anchor(anchor)
        with self._lock:
            for _ in range(self.max_retries):
                head = self._fetch_head()
                existing = self.read_path(head, path)
                if existing is not None:
                    if existing != data:
                        raise QueueConflict("anchor_conflict: anchor path exists with different bytes")
                    return {"path": path, "commit": self.anchor_commit(head, path), "head": head}
                self._check_anchor_preconditions(head, self._records(head), anchor)
                commit = self._commit_files(head, {path: data},
                                            f"queue: anchor {anchor['kind']} {anchor['dataset']['id']}")
                try:
                    self._push(commit)
                except QueueError:
                    # An accepted push can lose its response. Only refetch can decide.
                    pass
                latest = self._fetch_head()
                published = self.read_path(latest, path)
                if published is not None:
                    if published != data:
                        raise QueueConflict("anchor_conflict: concurrent anchor with different bytes")
                    return {"path": path, "commit": self.anchor_commit(latest, path), "head": latest}
            raise QueueConflict("publish_anchor exhausted concurrent retries")

    def seed_final(self, specs, campaign_sha256, *, stage_authorization=None, campaign=None, code_repo=None):
        """Seed global final jobs on top of a HEAD that already holds the campaign anchor.

        The anchor is checked inside the retry loop on the exact ``head`` the
        new commit is built on, so the jobs commit is always a child of a commit
        that contains the anchor (a descendant of ``C_campaign``, spec 6.9).
        """
        if type(campaign_sha256) is not str or not _HEX64.fullmatch(campaign_sha256):
            raise ValueError("campaign_anchor_missing: campaign_sha256 must be hex64")
        prepared = [prepare_job(spec) for spec in specs]
        if not prepared or len({spec["job_id"] for spec in prepared}) != len(prepared):
            raise ValueError("duplicate or empty job IDs in seed_final")
        for spec in prepared:
            if not _is_global_final(spec) or spec["global"].get("campaign_sha256") != campaign_sha256:
                raise ValueError("campaign_anchor_missing: every spec must be a global final job of this campaign")
            if spec["dependencies"]:
                raise ValueError("global_config_invalid: global final jobs have no dependencies")
        bindings = {(spec["dataset"], spec["dataset_version"], spec["global"].get("pools_sha256")) for spec in prepared}
        if len(bindings) != 1:
            raise ValueError("campaign_anchor_missing: seed_final takes one dataset/pools at a time")
        (dataset_id, dataset_version, pools_sha256), = bindings
        path = anchor_path(dataset_id, "campaign", campaign_sha256)
        with self._lock:
            for _ in range(self.max_retries):
                head = self._fetch_head()
                raw = self.read_path(head, path)
                if raw is None:
                    raise QueueConflict("campaign_anchor_missing: campaign anchor is not in the queue HEAD")
                try:
                    anchor = validate_anchor(json.loads(raw))
                except ValueError as error:
                    raise QueueConflict(f"campaign_anchor_missing: invalid campaign anchor ({error})") from error
                if (anchor["kind"] != "campaign" or anchor["pools_sha256"] != pools_sha256
                        or anchor["dataset"] != {"id": dataset_id, "version": dataset_version}):
                    raise QueueConflict("campaign_anchor_missing: campaign anchor bound to other dataset/pools")
                if stage_authorization is not None:
                    from . import stage_authorization as sa
                    from . import global_campaign as gc
                    sa.audit(stage_authorization, campaign, code_repo=code_repo)
                    if campaign["sha256"] != campaign_sha256:
                        raise QueueConflict("campaign_invalid: wrong campaign supplied for stage")
                    stage_chronology = gc.audit_chronology(self, campaign, head=head,
                                        stage_authorization=stage_authorization, code_repo=code_repo)
                for spec in prepared:
                    if (spec["global"].get("freeze_sha256") != anchor["freeze_sha256"]
                            or spec["global"].get("freeze_kind") != anchor["recipe_source"]
                            or spec["recipe_hash"] != anchor["freeze_sha256"]):
                        raise QueueConflict("campaign_anchor_missing: final job bound to another freeze than the "
                                            "anchored campaign")
                    if stage_authorization is not None:
                        sa.job_binding(spec, campaign, stage_authorization,
                                       anchor_commit=stage_chronology["C_stage_authorization"])
                        if spec["train_seed"] not in stage_authorization["seeds"]:
                            raise QueueConflict("campaign_invalid: stage seeding takes only that stage's jobs")
                    elif spec["code_commit"] != anchor["code_commit"]:
                        raise QueueConflict("code_identity_mismatch: final job code_commit differs from the "
                                            "anchored campaign code_commit")
                    elif "/stage-" in spec.get("config", ""):
                        raise QueueConflict("campaign_invalid: stage jobs need an audited authorization")
                    elif campaign is not None:
                        # R1-01: with the sealed campaign, admission follows its authorized seeds.
                        from .stage_gates import check_primary_job
                        check_primary_job(spec, campaign, campaign_sha256)
                records = self._records(head)
                for spec in prepared:
                    old = records.get(spec["job_id"])
                    if old and old["spec"] != spec:
                        raise QueueConflict("stable identity already has a different immutable spec")
                additions = [new_record(spec) for spec in prepared if spec["job_id"] not in records]
                if not additions:
                    return [records[spec["job_id"]] for spec in prepared]
                confirmed = self._publish(head, additions, "queue: add final jobs")
                if all(spec["job_id"] in confirmed for spec in prepared):
                    if any(confirmed[spec["job_id"]]["spec"] != spec for spec in prepared):
                        raise QueueConflict("concurrent seed changed immutable spec")
                    return [confirmed[spec["job_id"]] for spec in prepared]
            raise QueueConflict("seed_final exhausted concurrent retries")

    # ------------------------------------------------------------ documents (F5, 3.2g/6.3)

    @staticmethod
    def _document_bytes(path, content):
        if not isinstance(path, str) or not DOCUMENT_PATH.fullmatch(path):
            raise ValueError("document path must be fleet/<name>.json (never jobs/)")
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not isinstance(content, (bytes, bytearray)) or not content:
            raise ValueError("document content must be nonempty bytes or text")
        content = bytes(content)
        card = NODE_CARD_PATH.fullmatch(path)
        if card:
            # A public node card is validated before any Git I/O (6.3).
            try:
                document = json.loads(content.decode("utf-8"))
                fleet_schema.validate_public_card(document)
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValueError(f"invalid node card: {exc}") from exc
            if document.get("node_id") != card.group(1):
                raise ValueError("node card path does not match its node_id")
        return content

    def _document_blob(self, head, path):
        if head is None:
            return None
        listing = self._git("ls-tree", head, "--", path).stdout.strip()
        if not listing:
            return None
        meta, _, name = listing.partition("\t")
        _mode, kind, blob = meta.split()
        if name != path or kind != "blob":
            raise QueueError("document path is not a regular file")
        return blob

    def get_document(self, path):
        """``(content bytes, blob sha)`` of ``path`` on the queue branch, or ``(None, None)``."""
        if not isinstance(path, str) or not DOCUMENT_PATH.fullmatch(path):
            raise ValueError("document path must be fleet/<name>.json (never jobs/)")
        with self._lock:
            head = self._fetch_head()
            blob = self._document_blob(head, path)
            if blob is None:
                return None, None
            return self._git("cat-file", "blob", blob, binary=True).stdout, blob

    def put_document(self, path, content, expected_blob):
        """CAS write of one document outside ``jobs/``; returns the confirmed blob sha.

        ``expected_blob`` is the blob sha read with ``get_document`` (``None``: must not
        exist yet). Writing the bytes already present is idempotent. The commit keeps the
        rest of the tree (``jobs/`` included) exactly as observed.
        """
        data = self._document_bytes(path, content)
        if expected_blob is not None and (not isinstance(expected_blob, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_blob)):
            raise ValueError("expected_blob must be a 40-hex blob sha or None")
        with self._lock:
            target = self._git("hash-object", "-w", "--stdin", input=data, binary=True).stdout.decode("ascii").strip()
            for _ in range(self.max_retries):
                head = self._fetch_head()
                current = self._document_blob(head, path)
                if current == target:
                    return target
                if current != expected_blob:
                    raise QueueConflict("document changed; read it again before writing")
                with tempfile.TemporaryDirectory(prefix="augment-atlas-index-") as directory:
                    env = {"GIT_INDEX_FILE": str(Path(directory) / "index")}
                    self._git("read-tree", head if head else "--empty", env=env)
                    self._git("update-index", "--add", "--cacheinfo", f"100644,{target},{path}", env=env)
                    tree = self._git("write-tree", env=env).stdout.strip()
                args = ["commit-tree", tree] + (["-p", head] if head else [])
                commit = self._git(*args, input=f"queue: put {path}\n").stdout.strip()
                try:
                    self._push(commit)
                except QueueError:
                    pass  # Only the refetch decides.
                if self._document_blob(self._fetch_head(), path) == target:
                    return target
            raise QueueConflict("put_document exhausted concurrent retries")
