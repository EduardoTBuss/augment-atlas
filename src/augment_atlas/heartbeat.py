"""F5 part B: per-worker heartbeat in its own Git ref (docs/fleet-protocol.md 4.2, ADR-2).

* ``HeartbeatPublisher(checkout, remote, ref).publish(doc)`` writes one validated
  ``atlas.heartbeat.v1`` document as ``heartbeat.json`` in a commit that is a child
  of the current tip of ``refs/heads/fleet-hb/<node_id>/<worker>``, pushed with an
  ordinary fast-forward push (never ``--force``). It never touches the ``queue``
  branch nor any job ``.md``.
* Publishing is best effort: Git being unreachable, a rejected push or a timeout
  never raises into the training loop. ``publish`` returns a ``PublishResult`` and,
  when a ``journal`` path is given, appends one fsynced line to the local
  ``liveness.jsonl`` (4.2). The caller decides what to do.
* Programming errors are still errors: an invalid document, a document of another
  node/worker, or a ref outside ``refs/heads/fleet-hb/<node>/<worker>`` raise
  ``ValueError`` before any Git I/O.
* ``fetch_heartbeats(checkout, remote)`` is the observer side: it reads every
  ``fleet-hb`` ref and returns the snapshot consumed by ``fleet.liveness``.

Use a checkout owned by this publisher/observer only (``<queue_checkout>-hb``):
``FETCH_HEAD`` must not be shared between processes (4.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from . import fleet_schema
from .schema import canonical_json

HEARTBEAT_FILE = "heartbeat.json"
REF_PREFIX = "refs/heads/fleet-hb/"
GIT_TIMEOUT_S = 60


class HeartbeatGitError(RuntimeError):
    """Internal: Git unavailable/failed. Never escapes ``publish``."""


@dataclass(frozen=True)
class PublishResult:
    ok: bool
    reason: str            # published | already_published | git_unavailable | push_not_confirmed | seq_not_increasing | foreign_tip
    ref: str
    seq: int
    commit: str | None = None
    journal_ok: bool | None = None


@dataclass
class HeartbeatSnapshot:
    fetch_ok: bool
    heartbeats: dict = field(default_factory=dict)   # ref -> validated heartbeat document
    invalid: dict = field(default_factory=dict)      # ref -> short reason (never raised)


def parse_ref(ref):
    """``refs/heads/fleet-hb/<node>/<worker>`` -> ``(node, worker)``; anything else -> ``ValueError``."""
    if not isinstance(ref, str) or not ref.startswith(REF_PREFIX):
        raise ValueError("heartbeat ref must be refs/heads/fleet-hb/<node_id>/<worker>")
    parts = ref[len(REF_PREFIX):].split("/")
    if len(parts) != 2 or not all(fleet_schema.SLUG.fullmatch(part) for part in parts):
        raise ValueError("heartbeat ref must be refs/heads/fleet-hb/<node_id>/<worker>")
    return parts[0], parts[1]


def heartbeat_ref(node_id, worker):
    ref = f"{REF_PREFIX}{node_id}/{worker}"
    parse_ref(ref)
    return ref


def encode(doc):
    """Exact bytes stored in ``heartbeat.json`` (canonical JSON + LF, platform independent)."""
    return (canonical_json(doc) + "\n").encode("utf-8")


class _Git:
    def __init__(self, checkout, remote, timeout=GIT_TIMEOUT_S):
        self.checkout = Path(checkout).resolve()
        self.remote = str(remote)
        self.timeout = timeout

    def run(self, *args, input=None, check=True):
        env = os.environ.copy()
        env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_AUTHOR_NAME": "AugmentAtlas heartbeat",
                    "GIT_AUTHOR_EMAIL": "heartbeat@example.invalid", "GIT_COMMITTER_NAME": "AugmentAtlas heartbeat",
                    "GIT_COMMITTER_EMAIL": "heartbeat@example.invalid"})
        try:
            # Bytes mode: no newline translation of heartbeat.json on Windows.
            result = subprocess.run(["git", "-C", str(self.checkout), *args], input=input, capture_output=True,
                                    env=env, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HeartbeatGitError(f"Git operation unavailable: {type(exc).__name__}") from exc
        if check and result.returncode:
            # Never surface stderr: it can contain an authenticated remote URL.
            raise HeartbeatGitError(f"Git {args[0]} failed (exit {result.returncode})")
        return result

    def out(self, *args, **kwargs):
        return self.run(*args, **kwargs).stdout.decode("utf-8").strip()

    def ensure_repo(self):
        try:
            self.checkout.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HeartbeatGitError(f"checkout unavailable: {type(exc).__name__}") from exc
        if not (self.checkout / ".git").exists():
            self.run("init", "-q")

    def remote_tip(self, ref):
        lines = self.out("ls-remote", self.remote, ref).splitlines()
        tips = [line.split("\t", 1)[0] for line in lines if line.endswith("\t" + ref)]
        if len(tips) > 1:
            raise HeartbeatGitError("ambiguous heartbeat ref")
        return tips[0] if tips else None

    def fetch_blob(self, ref):
        """Fetch ``ref`` into FETCH_HEAD and return ``(commit, heartbeat.json bytes or None)``."""
        self.run("fetch", "--no-tags", "-q", self.remote, ref)
        commit = self.out("rev-parse", "FETCH_HEAD")
        shown = self.run("show", f"{commit}:{HEARTBEAT_FILE}", check=False)
        return commit, (shown.stdout if shown.returncode == 0 else None)


def _utc_z(now=None):
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class HeartbeatPublisher:
    """Single writer of one worker's heartbeat ref (4.2)."""

    def __init__(self, checkout, remote, ref, *, journal=None, timeout=GIT_TIMEOUT_S,
                 utc_clock=None, mono_clock=time.monotonic):
        self.node_id, self.worker = parse_ref(ref)
        self.ref = ref
        self.journal = Path(journal) if journal is not None else None
        self._git = _Git(checkout, remote, timeout)
        self._utc_clock = utc_clock or _utc_z
        self._mono_clock = mono_clock
        self._lock = threading.Lock()

    @property
    def checkout(self):
        return self._git.checkout

    def _check_doc(self, doc):
        try:
            fleet_schema.validate_heartbeat(doc)
        except fleet_schema.SchemaError as exc:
            raise ValueError(f"invalid heartbeat document: {exc}") from exc
        if doc["node_id"] != self.node_id or doc["worker"] != self.worker:
            raise ValueError("heartbeat document belongs to another node/worker than the ref")
        return encode(doc)

    def publish(self, doc):
        """Publish ``doc``; returns ``PublishResult``. Raises only ``ValueError`` (caller bug, no I/O done)."""
        data = self._check_doc(doc)
        with self._lock:
            try:
                result = self._publish(data, doc["seq"])
            except HeartbeatGitError:
                result = PublishResult(False, "git_unavailable", self.ref, doc["seq"])
        if not result.ok and self.journal is not None:
            result = PublishResult(result.ok, result.reason, result.ref, result.seq, result.commit,
                                   self._journal_failure(doc, result.reason))
        return result

    def _publish(self, data, seq):
        git = self._git
        git.ensure_repo()
        parent = git.remote_tip(self.ref)
        if parent is not None:
            fetched, previous = git.fetch_blob(self.ref)
            if fetched != parent:
                return PublishResult(False, "push_not_confirmed", self.ref, seq)
            if previous == data:
                return PublishResult(True, "already_published", self.ref, seq, parent)
            try:
                old = json.loads(previous) if previous is not None else None
            except ValueError:
                old = None
            if not isinstance(old, dict) or old.get("node_id") != self.node_id or old.get("worker") != self.worker:
                # Single writer: a tip we cannot attribute to ourselves is left untouched.
                return PublishResult(False, "foreign_tip", self.ref, seq, parent)
            if not isinstance(old.get("seq"), int) or seq <= old["seq"]:
                return PublishResult(False, "seq_not_increasing", self.ref, seq, parent)
        blob = git.out("hash-object", "-w", "--stdin", input=data)
        tree = git.out("mktree", input=f"100644 blob {blob}\t{HEARTBEAT_FILE}\n".encode("ascii"))
        args = ["commit-tree", tree] + (["-p", parent] if parent else [])
        commit = git.out(*args, input=f"heartbeat: {self.node_id}/{self.worker} seq {seq}\n".encode("utf-8"))
        try:
            # Ordinary push: the remote rejects anything that is not a fast-forward of ``parent``.
            git.run("push", "-q", git.remote, f"{commit}:{self.ref}", check=False)
        except HeartbeatGitError:
            pass  # An accepted push can lose its response; only the refetch decides.
        if git.remote_tip(self.ref) == commit:
            return PublishResult(True, "published", self.ref, seq, commit)
        return PublishResult(False, "push_not_confirmed", self.ref, seq)

    def _journal_failure(self, doc, reason):
        line = {"utc": self._utc_clock(), "mono_s": round(float(self._mono_clock()), 3), "boot_id": doc["boot_id"],
                "status": doc["status"], "git_ok_last": False, "event": "heartbeat_publish_failed",
                "reason": reason, "seq": doc["seq"]}
        try:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with open(self.journal, "ab") as handle:
                handle.write((canonical_json(line) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            return True
        except OSError:
            return False


def fetch_heartbeats(checkout, remote, *, timeout=GIT_TIMEOUT_S):
    """Observer side: every ``refs/heads/fleet-hb/*/*`` tip -> validated heartbeat document.

    Never raises for Git or data problems: ``fetch_ok=False`` when the remote cannot be
    read (``fleet.liveness`` then reports ``unknown_observer``); malformed refs or
    documents are listed in ``invalid`` and left out of ``heartbeats``.
    """
    git = _Git(checkout, remote, timeout)
    snapshot = HeartbeatSnapshot(fetch_ok=False)
    try:
        git.ensure_repo()
        listing = git.out("ls-remote", git.remote, REF_PREFIX + "*").splitlines()
        tips = {}
        for line in listing:
            sha, _, ref = line.partition("\t")
            try:
                parse_ref(ref)
            except ValueError:
                snapshot.invalid[ref] = "ref_pattern"
                continue
            tips[ref] = sha
        for ref, sha in sorted(tips.items()):
            commit, data = git.fetch_blob(ref)
            if commit != sha:
                snapshot.invalid[ref] = "ref_moved_during_fetch"
                continue
            try:
                doc = fleet_schema.validate_heartbeat(json.loads(data.decode("utf-8")))
            except (AttributeError, ValueError, UnicodeDecodeError):
                snapshot.invalid[ref] = "invalid_document"
                continue
            if heartbeat_ref(doc["node_id"], doc["worker"]) != ref:
                snapshot.invalid[ref] = "document_of_another_worker"
                continue
            snapshot.heartbeats[ref] = doc
    except HeartbeatGitError:
        return HeartbeatSnapshot(fetch_ok=False, invalid=snapshot.invalid)
    snapshot.fetch_ok = True
    return snapshot


def read_heartbeat(checkout, remote, ref, *, timeout=GIT_TIMEOUT_S):
    """One ref: ``(fetch_ok, doc_or_None)``. ``ValueError`` only for a malformed ``ref`` argument."""
    parse_ref(ref)
    git = _Git(checkout, remote, timeout)
    try:
        git.ensure_repo()
        tip = git.remote_tip(ref)
        if tip is None:
            return True, None
        commit, data = git.fetch_blob(ref)
    except HeartbeatGitError:
        return False, None
    if commit != tip or data is None:
        return True, None
    try:
        doc = fleet_schema.validate_heartbeat(json.loads(data.decode("utf-8")))
    except (ValueError, UnicodeDecodeError):
        return True, None
    return True, (doc if heartbeat_ref(doc["node_id"], doc["worker"]) == ref else None)
