"""Orchestration of ``probe``, ``canary``/``full`` and ``verify`` (plan 3.4).

Shared state and concurrency (single machine, one process tree):
* One acquisition per root: ``acquisition-v1/lock`` (flock / msvcrt), non-blocking.
* The main process is the only writer of the ledger, the derived directory renames
  and the spool deletions. Derive workers (spawned ProcessPoolExecutor, <= 2, at most 2
  jobs in flight per worker; the rest wait as FETCHED in a backlog) only read one
  spool file and write one ``<image>.jpg.tmp`` that no one else touches; results come
  back to the main process, which renames and commits. Hence no lock ordering exists
  beyond "acquisition lock, then everything else"; no deadlock is possible.
* Durability order per member: spool file fsync+rename -> ledger FETCHED -> derived
  ``.tmp`` fsync -> rename -> ledger DERIVED -> spool unlink. A kill between any two
  steps is repaired on the next session (``_recover``) without re-reading members that
  are already DERIVED/SKIPPED.
"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ..contracts import atomic_json, canonical_hash, file_hash
from . import csvsource, derive, groups, isic2020, splits_global, summary
from .config import CSV_ROLES, MIB
from .diskguard import DiskGuard, ProcessLock
from .errors import DivergenceError, Incomplete, IntegrityError, SpoolPaused
from .http_range import ObjectSpec, RangeClient, RetryPolicy, TokenBucket
from .ledger import Ledger, utc_now
from .stream import directory_usage, spool_path, stream_with_resume
from .zipcd import DEFAULT_SUFFIX_BYTES, EndRecord, load_central_directory, parse_central_directory, validate_entries

_SPOOL_FILE = re.compile(r"^member-(\d{6,})\.bin$")
_SAFE_IMAGE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PROBE_HEADERS = ("content-length", "etag", "last-modified", "accept-ranges", "content-type", "content-range")


@dataclass
class Deps:
    """Injectable collaborators (tests only; the CLI uses the defaults)."""

    free_fn: object = None
    deriver: object = derive.derive_member
    sleep: object = time.sleep
    clock: object = time.monotonic
    ssl_context: object = None
    log: object = None
    tool_commit: str | None = None
    docker_image_digest: str | None = None

    def emit(self, message):
        stream = self.log or sys.stderr
        print(f"[acquisition {utc_now()}] {message}", file=stream, flush=True)


class Layout:
    def __init__(self, root, cfg):
        self.root = Path(root).absolute()
        self.state = self.root / "acquisition-v1"
        self.lock = self.state / "lock"
        self.ledger = self.state / "ledger.sqlite"
        self.cd = self.state / "central-directory.bin"
        self.probe = self.state / "probe.json"
        self.canary = self.state / "canary.json"
        self.config_copy = self.state / "config.json"
        self.spool = self.state / "spool"
        self.release = self.root / "release"
        pid = cfg.preprocessing["id"]
        prefix = f"{cfg.dataset['name']}-"
        self.derived_name = pid[len(prefix):] if pid.startswith(prefix) else pid
        self.derived = self.root / self.derived_name / "images"
        self.manifest = self.root / "manifest"
        self.divergences = self.manifest / "divergences.json"
        self.ready = self.manifest / "READY.json"
        self.splits = self.root / "splits" / splits_global.ALGORITHM
        self.public = self.root / "public"

    def derived_rel(self, image):
        return f"{self.derived_name}/images/{image}.jpg"


def _mkdirs(layout):
    for path in (layout.root, layout.state, layout.spool, layout.release, layout.derived, layout.manifest):
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            os.chmod(path, 0o700)


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as sink:
        sink.write(data)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temp, path)


def _fsync_dir(path):
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _retry(cfg, deps):
    transfer = cfg.transfer
    return RetryPolicy(max_attempts=transfer["max_attempts"], base_delay_s=transfer["backoff_base_s"],
                       max_delay_s=transfer["backoff_max_s"], sleep=deps.sleep)


def _bucket(cfg, deps):
    return TokenBucket(cfg.transfer["max_rate_mib_s"] * MIB, clock=deps.clock, sleep=deps.sleep)


def _tool_commit(deps):
    if deps.tool_commit:
        return deps.tool_commit
    if os.environ.get("AUGMENT_ATLAS_TOOL_COMMIT"):
        return os.environ["AUGMENT_ATLAS_TOOL_COMMIT"]
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True,
                                text=True, timeout=10, check=False)
        if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", result.stdout.strip()):
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# ---------------------------------------------------------------------------
# probe (network) + analysis (offline, from retained files)
# ---------------------------------------------------------------------------

def _zip_client(cfg, deps, bucket, retry):
    source = cfg.source("images_zip")
    spec = ObjectSpec(url=source["url"], expected_size=source["bytes"], expected_etag=source["etag"])
    return RangeClient(spec, timeout_s=cfg.transfer["timeout_s"], bucket=bucket, retry=retry, ssl_context=deps.ssl_context)


def probe_network(cfg, layout, deps, guard):
    """Download/verify the 3 CSVs into ``release/``, HEAD the ZIP and retain its central directory."""
    retry, bucket = _retry(cfg, deps), _bucket(cfg, deps)
    report = {"probed_utc": utc_now(), "sources": {}}
    for role in CSV_ROLES:
        source = cfg.source(role)
        body, headers = csvsource.fetch_object(source["url"], expected_bytes=source["bytes"], expected_etag=source["etag"],
                                               timeout_s=cfg.transfer["timeout_s"], retry=retry, bucket=bucket,
                                               ssl_context=deps.ssl_context)
        digest = csvsource.verify_bytes(body, expected_bytes=source["bytes"], expected_etag=source["etag"])
        csvsource.parse_csv(body, source["header"], name=role)  # exact header (ADR-A8) before retaining anything
        guard.before_write(len(body), f"release {role}")
        csvsource.store_release_file(body, layout.release / cfg.filename(role))
        report["sources"][role] = {"url": source["url"], **digest, "utf8_bom": csvsource.has_utf8_bom(body),
                                   "headers": {k: v for k, v in headers.items() if k in PROBE_HEADERS}}
        deps.emit(f"{role}: {len(body)} B, ETag/MD5 and header verified")
    client = _zip_client(cfg, deps, bucket, retry)
    head = client.probe()
    suffix = client.fetch_suffix(DEFAULT_SUFFIX_BYTES)
    end, entries, cd_bytes = load_central_directory(client.fetch, head.size, suffix=suffix)
    if layout.cd.exists():
        if layout.cd.read_bytes() != cd_bytes:
            raise DivergenceError("central directory differs from the one retained by an earlier probe")
    else:
        guard.before_write(len(cd_bytes), "central directory")
        _atomic_bytes(layout.cd, cd_bytes)
    report["sources"]["images_zip"] = {
        "url": cfg.source("images_zip")["url"], "content_length": head.size, "etag": head.etag,
        "last_modified": head.last_modified, "zip64": end.zip64, "cd_offset": end.cd_offset, "cd_size": end.cd_size,
        "cd_entries": end.entry_count, "central_directory_sha256": hashlib.sha256(cd_bytes).hexdigest(),
        "headers": {k: v for k, v in head.headers.items() if k in PROBE_HEADERS}}
    deps.emit(f"images_zip: HEAD ok ({head.size} B), central directory with {end.entry_count} entries retained")
    return report, client


@dataclass
class Analysis:
    records: dict
    pairs: list
    observed: dict
    divergences: list
    end: object
    entries: list
    members: list
    cd_sha256: str
    release: dict = field(default_factory=dict)


def analyze(cfg, layout):
    """Everything derivable from the retained CSVs and central directory (no network)."""
    bodies, release = {}, {}
    for role in CSV_ROLES:
        path = layout.release / cfg.filename(role)
        if not path.exists():
            raise Incomplete(f"release file for {role} missing: run probe first")
        body = path.read_bytes()
        source = cfg.source(role)
        release[role] = csvsource.verify_bytes(body, expected_bytes=source["bytes"], expected_etag=source["etag"])
        bodies[role] = csvsource.parse_csv(body, source["header"], name=role)
    records, pairs, observed, divergences = isic2020.analyze_labels(bodies["ground_truth_v2"], bodies["ground_truth_v1"],
                                                                     bodies["duplicates"], cfg)
    if not layout.cd.exists():
        raise Incomplete("central directory not retained: run probe first")
    cd_bytes = layout.cd.read_bytes()
    size = cfg.source("images_zip")["bytes"]
    # The retained CD is self-describing only with its end record; re-fetching is avoided by
    # parsing the CD against an end record rebuilt from the probe report.
    probe = json.loads(layout.probe.read_text(encoding="utf-8"))["sources"]["images_zip"]
    if probe["central_directory_sha256"] != hashlib.sha256(cd_bytes).hexdigest():
        raise IntegrityError("retained central directory does not match the probe report")
    end = EndRecord(cd_offset=probe["cd_offset"], cd_size=probe["cd_size"], entry_count=probe["cd_entries"],
                    zip64=probe["zip64"], end_offset=probe["cd_offset"] + probe["cd_size"])
    if end.cd_offset + end.cd_size > size:
        raise IntegrityError("central directory range exceeds the pinned object size")
    entries = validate_entries(parse_central_directory(cd_bytes, end), end)
    members = isic2020.classify_members(entries, cfg)
    zip_observed, zip_divergences = isic2020.cross_check(members, records, cfg)
    observed.update(zip_observed)
    divergences += zip_divergences
    isic2020.apply_acceptance(divergences, cfg.accepted_divergences)
    return Analysis(records=records, pairs=pairs, observed=observed, divergences=divergences, end=end, entries=entries,
                    members=members, cd_sha256=hashlib.sha256(cd_bytes).hexdigest(), release=release)


def _write_divergences(layout, divergences, stage):
    atomic_json(layout.divergences, {"schema": "augment-atlas/acquisition-divergences/v1", "stage": stage,
                                     "written_utc": utc_now(), "divergences": divergences})


def _divergence_brief(items):
    return ", ".join(f"{d['field']} (documented {d['documented']}, observed {d['observed']})" for d in items)


def _public_counts(analysis):
    return {"observed": analysis.observed,
            "divergences": summary.public_divergences(analysis.divergences)}


def cmd_probe(cfg, root, deps):
    layout = Layout(root, cfg)
    guard = DiskGuard(layout.root, reserve_bytes=cfg.disk_bytes("reserve_gib"),
                      write_margin_bytes=cfg.disk_bytes("write_margin_gib"), free_fn=deps.free_fn)
    guard.before_write(cfg.source("ground_truth_v1")["bytes"] + cfg.source("ground_truth_v2")["bytes"], "probe")
    _mkdirs(layout)
    with ProcessLock(layout.lock):
        report, _client = probe_network(cfg, layout, deps, guard)
        atomic_json(layout.probe, report)
        analysis = analyze(cfg, layout)
        _write_divergences(layout, analysis.divergences, "probe")
        result = _public_counts(analysis)
        open_items = isic2020.pending(analysis.divergences)
        if open_items:
            deps.emit(f"probe: {len(open_items)} pending divergence(s): {_divergence_brief(open_items)}")
            raise DivergenceError(f"probe found {len(open_items)} pending divergence(s); see manifest/divergences.json")
        deps.emit("probe: no pending divergence")
        return result


# ---------------------------------------------------------------------------
# run (canary / full)
# ---------------------------------------------------------------------------

class _CanaryReached(Exception):
    pass


@dataclass
class _Job:
    idx: int
    image: str
    spool: Path
    tmp: Path
    estimate: int
    future: object = None
    result: object = None


class _Deriver:
    """Inline (workers=0) or process-pool derivation; results are committed by the caller.

    At most ``capacity`` jobs are in flight (2 per worker): FETCHED members wait in the
    runner's backlog, so the disk guard is checked right before each job really starts
    and an abort only has to wait for a handful of jobs.
    """

    def __init__(self, fn, workers):
        self.fn = fn
        # spawn (not fork): workers inherit no sqlite connection, signal handler or thread state,
        # and behave the same on every platform and Python version.
        self.pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
                     if workers else None)
        self.capacity = 2 * workers if workers else 1
        self.jobs = deque()

    def has_capacity(self):
        return len(self.jobs) < self.capacity

    def submit(self, job, params):
        if self.pool is None:
            try:
                job.result = self.fn(str(job.spool), str(job.tmp), params)
            except Exception as exc:
                job.result = {"status": "error", "exception": exc}
        else:
            job.future = self.pool.submit(self.fn, str(job.spool), str(job.tmp), params)
        self.jobs.append(job)

    def ready(self, block=False):
        """Pop finished jobs (block=True waits for the oldest one if none is finished)."""
        done = [job for job in self.jobs if job.future is None or job.future.done()]
        if not done and block and self.jobs:
            wait([self.jobs[0].future])
            done = [self.jobs[0]]
        for job in done:
            self.jobs.remove(job)
            if job.future is not None:
                exc = job.future.exception()
                job.result = {"status": "error", "exception": exc} if exc is not None else job.future.result()
        return done

    def shutdown(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=False)


class Runner:
    def __init__(self, cfg, layout, deps, mode, limit, strict):
        self.cfg, self.layout, self.deps, self.mode, self.limit, self.strict = cfg, layout, deps, mode, limit, strict
        self.guard = DiskGuard(layout.root, reserve_bytes=cfg.disk_bytes("reserve_gib"),
                               write_margin_bytes=cfg.disk_bytes("write_margin_gib"), free_fn=deps.free_fn)
        self.derived_cap = cfg.disk_bytes("derived_cap_gib")
        self.spool_cap = cfg.disk_bytes("spool_cap_gib")
        self.params = dict(cfg.preprocessing)
        self.projections = []
        self.backlog = deque()   # FETCHED members waiting for a derive slot (stream order)
        self.spool_bytes = 0     # bytes held in the spool, tracked in memory (no per-member scandir)
        self.started_clock = deps.clock()

    # -- helpers ---------------------------------------------------------------
    def needed_bytes(self, derived_written, spool_held):
        """Plan 3.4 step 1, resume-aware: what this acquisition may still write before it ends."""
        return (max(0, self.derived_cap - derived_written) + max(0, self.spool_cap - spool_held)
                + self.cfg.disk_bytes("precheck_margin_gib"))

    def _load_counters(self):
        """One-time SQL at session start; afterwards the counters are kept in memory (O(1) per member)."""
        self.n_derived, self.bytes_derived = self.ledger.derived_stats()
        conn = self.ledger.conn
        labelled = "kind='image' AND labelled=1"
        self.n_labelled_total = conn.execute(f"SELECT COUNT(*) FROM member WHERE {labelled}").fetchone()[0]
        self.n_labelled_done = conn.execute(f"SELECT COUNT(*) FROM member WHERE {labelled} AND state IN ('DERIVED', 'SKIPPED')"
                                            ).fetchone()[0]
        self.n_started = conn.execute(f"SELECT COUNT(*) FROM member WHERE {labelled} AND state != 'PENDING'").fetchone()[0]
        self.next_projection = max(self.cfg.canary["limit"], self.n_derived)
        self.session_derived = 0

    def _recover(self):
        """Repair what a kill can leave behind; never touches files the tool did not create."""
        fetched = {row["idx"] for row in self.ledger.members(state="FETCHED")}
        removed = 0
        for item in os.scandir(self.layout.spool):
            match = _SPOOL_FILE.match(item.name)
            if item.name.endswith(".part") or (match and int(match.group(1)) not in fetched):
                os.unlink(item.path)  # partial member, or member already committed past FETCHED / never committed
                removed += 1
        for item in os.scandir(self.layout.derived):
            if item.name.endswith(".jpg.tmp"):
                os.unlink(item.path)
                removed += 1
        for idx in fetched:
            if not spool_path(self.layout.spool, self.entries_by_idx[idx]).exists():
                raise IntegrityError(f"ledger says member {idx} is FETCHED but its spool file is gone")
        if removed:
            self.deps.emit(f"recovery: removed {removed} leftover temporary/stale file(s) from an interrupted session")

    def _submit(self, idx):
        row = self.ledger.member(idx)
        image = row["image"]
        if not _SAFE_IMAGE.match(image):
            raise IntegrityError("reconstructed image name is not a safe file name")
        estimate = row["usize"]  # conservative: q95 at 512 is smaller than the original
        self.guard.before_write(estimate, "derived image")
        if self.bytes_derived + estimate > self.derived_cap:
            raise DivergenceError(f"derived bytes would exceed derived_cap ({self.derived_cap} B)")
        job = _Job(idx=idx, image=image, spool=spool_path(self.layout.spool, self.entries_by_idx[idx]),
                   tmp=self.layout.derived / f"{image}.jpg.tmp", estimate=row["usize"])
        self.deriver.submit(job, self.params)

    def _release_spool(self, path, size):
        path.unlink(missing_ok=True)
        self.spool_bytes = max(0, self.spool_bytes - size)

    def _complete(self, jobs):
        """Commit every finished job first, then run the checks that may raise (no .tmp is left behind)."""
        worker_error = None
        for job in jobs:
            result = job.result
            if result["status"] == "error":
                job.tmp.unlink(missing_ok=True)  # member stays FETCHED; the next session re-derives it
                worker_error = worker_error or result["exception"]
                continue
            if result["status"] == "ok":
                final = self.layout.derived / f"{job.image}.jpg"
                os.replace(job.tmp, final)
                _fsync_dir(self.layout.derived)
                info = {"original": result["original"], "short_side_lt_target": result["short_side_lt_target"],
                        "derived": {**result["derived"], "path": self.layout.derived_rel(job.image)}}
                self.ledger.mark_derived(job.idx, info)
                self.n_derived += 1
                self.session_derived += 1
                self.bytes_derived += result["derived"]["bytes"]
            else:
                job.tmp.unlink(missing_ok=True)
                self.ledger.mark_skipped(job.idx, result["reason"], from_state="FETCHED")
            self.n_labelled_done += 1
            self._release_spool(job.spool, job.estimate)
        if worker_error is not None:
            raise worker_error
        if self.bytes_derived > self.derived_cap:
            raise DivergenceError(f"derived bytes {self.bytes_derived} exceed derived_cap ({self.derived_cap} B)")
        self._maybe_project()

    def _advance(self, block=False):
        """Commit finished jobs (waiting for one if ``block``), then fill free derive slots from the backlog."""
        self._complete(self.deriver.ready(block=block))
        while self.backlog and self.deriver.has_capacity():
            self._submit(self.backlog.popleft())
            if self.deriver.pool is None:  # inline: the job is already finished
                self._complete(self.deriver.ready())

    def _drain(self):
        while self.backlog or self.deriver.jobs:
            self._advance(block=True)

    def _abort_inflight(self):
        """After an error: leave the backlog FETCHED for the next session, settle in-flight jobs only."""
        self.backlog.clear()
        first_error = None
        while self.deriver.jobs:
            try:
                self._complete(self.deriver.ready(block=True))
            except BaseException as exc:  # keep settling so no .tmp is left; report the first error
                if first_error is None:
                    first_error = exc
        return first_error

    def projection(self):
        count, written = self.n_derived, self.bytes_derived
        if not count:
            return None
        mean = written / count
        remaining = self.n_labelled_total - self.n_labelled_done
        projected = mean * self.cfg.canary["projection_factor"] * remaining + written
        return {"derived": count, "written_bytes": written, "mean_bytes": mean, "remaining": remaining,
                "projected_bytes": projected, "derived_cap_bytes": self.derived_cap}

    def _maybe_project(self):
        count = self.n_derived
        if count < self.next_projection:
            return
        self.next_projection = count + self.cfg.canary["projection_every"]
        projection = self.projection()
        self.projections.append(projection)
        self.deps.emit(f"projection after {count} derived: {projection['projected_bytes'] / (1 << 30):.3f} GiB "
                       f"(cap {self.derived_cap / (1 << 30):.3f} GiB)")
        if projection["projected_bytes"] > self.derived_cap:
            raise DivergenceError(f"projected derived size {projection['projected_bytes']:.0f} B exceeds derived_cap")

    def _before_member(self, _position, entry):
        self.guard.before_write(entry.uncompressed_size, "spool member")

    def _on_member(self, result):
        idx = result.entry.index
        self.spool_bytes += result.size
        row = self.ledger.member(idx)
        if row["kind"] == "image" and row["labelled"]:
            self.ledger.mark_fetched(idx, result.sha256)
            self.n_started += 1
            self.backlog.append(idx)
        else:
            reason = "no-label" if row["kind"] == "image" else row["kind"]
            self.ledger.mark_skipped(idx, reason, sha256=result.sha256)
            self._release_spool(result.path, result.size)
        self._advance()
        if self.mode == "canary" and self.n_started >= self.limit:
            raise _CanaryReached()

    # -- main --------------------------------------------------------------------
    def run(self):
        cfg, layout, deps = self.cfg, self.layout, self.deps
        if deps.deriver is derive.derive_member and not derive.pillow_available():
            raise DivergenceError("Pillow is not importable here: canary/full must run inside the pinned container")
        fresh = not layout.ledger.exists()
        if fresh:
            self.guard.precheck(self.needed_bytes(0, 0))  # plan 3.4 step 1: before anything is written
        _mkdirs(layout)
        with ProcessLock(layout.lock), Ledger(layout.ledger) as ledger:
            self.ledger = ledger
            if not fresh:  # resume: only what is still to be written must fit above the reserve
                _count, written = ledger.derived_stats()
                self.guard.precheck(self.needed_bytes(written, directory_usage(layout.spool)))
            atomic_json(layout.config_copy, {"sha256": cfg.sha256, "identity_sha256": cfg.identity_sha256, "config": cfg.raw})
            report, client = probe_network(cfg, layout, deps, self.guard)
            atomic_json(layout.probe, report)
            analysis = analyze(cfg, layout)
            _write_divergences(layout, analysis.divergences, self.mode)
            open_items = isic2020.pending(analysis.divergences)
            if open_items:
                deps.emit(f"{len(open_items)} pending divergence(s): {_divergence_brief(open_items)}")
                if self.strict:
                    raise DivergenceError(f"strict mode: {len(open_items)} pending divergence(s) before streaming; "
                                          "see manifest/divergences.json (use --lenient to stream anyway; READY stays blocked)")
            # Identity is pinned only now, right before the ZIP body is streamed: until here a
            # divergence can still be fixed in the config without starting a new root.
            ledger.pin({"config_identity_sha256": cfg.identity_sha256,
                        "images_zip_etag": cfg.source("images_zip")["etag"],
                        "images_zip_bytes": cfg.source("images_zip")["bytes"],
                        "central_directory_sha256": analysis.cd_sha256})
            self.entries_by_idx = {entry.index: entry for entry in analysis.entries}
            ledger.init_members(analysis.entries, analysis.members, set(analysis.records))
            self._recover()
            self.spool_bytes = directory_usage(layout.spool)
            self._load_counters()
            start = ledger.first_pending_position(len(analysis.entries))
            session = ledger.start_session(self.mode, start)
            exit_code, note = 1, None
            self.deriver = _Deriver(deps.deriver, cfg.transfer["workers"])
            try:
                self.backlog.extend(row["idx"] for row in ledger.members(state="FETCHED"))
                if not (self.mode == "canary" and self.n_started >= self.limit):
                    self._stream(analysis, client, start)
                self._drain()
                exit_code = 0
            except BaseException as exc:
                exit_code = getattr(exc, "exit_code", 1)
                note = f"{type(exc).__name__}: {exc}"[:500]
                settle_error = self._abort_inflight()
                if settle_error is not None:  # the original error wins; record both
                    note += f"; settling in-flight jobs: {type(settle_error).__name__}"
                raise
            finally:
                self.deriver.shutdown()
                end_position = ledger.first_pending_position(len(analysis.entries))
                ledger.end_session(session, end_position, exit_code, note)
                deps.emit(f"session {self.mode} ended with exit {exit_code}: {ledger.counts()}; "
                          f"disk guard min headroom {self.guard.min_headroom} B")
            if self.mode == "canary":
                return self._canary_report(analysis)
        return cmd_verify(self.cfg, self.layout.root, self.deps)

    def _stream(self, analysis, client, start):
        position = start
        entries = analysis.entries
        while position < len(entries):
            try:
                position = stream_with_resume(
                    entries, client.open_range, retry=client.retry, start=position, cd_offset=analysis.end.cd_offset,
                    spool_dir=self.layout.spool, spool_cap_bytes=self.spool_cap, on_member=self._on_member,
                    before_member=self._before_member, spool_usage=lambda: self.spool_bytes)
            except _CanaryReached:
                return
            except SpoolPaused as exc:
                # Plan 3.4 step 8: the connection is already closed; drain below cap/2 (and until the
                # next member fits), then reopen at that member's local header.
                position = exc.next_index
                size = entries[position].uncompressed_size
                while (self.backlog or self.deriver.jobs) and (self.spool_bytes >= self.spool_cap / 2
                                                               or self.spool_bytes + size > self.spool_cap):
                    self._advance(block=True)
                if self.spool_bytes + size > self.spool_cap:
                    raise IntegrityError("spool cannot drain below its cap (foreign files in the spool?)")

    def _canary_report(self, analysis):
        count, written = self.n_derived, self.bytes_derived
        projection = self.projection()
        sessions = self.ledger.sessions()
        elapsed = max(0.0, self.deps.clock() - self.started_clock)
        timing = {"session_elapsed_s": round(elapsed, 3), "session_derived": self.session_derived,
                  "seconds_per_derived_image": round(elapsed / self.session_derived, 4) if self.session_derived else None,
                  "workers": self.cfg.transfer["workers"]}
        report = {"limit": self.limit, "derived": count, "derived_bytes": written, "projection": projection,
                  "projections": self.projections, "counts": self.ledger.counts(), "timing": timing,
                  "decoder": derive.decoder_versions() if self.deps.deriver is derive.derive_member else None,
                  "sessions": sessions, "disk_min_headroom_bytes": self.guard.min_headroom}
        atomic_json(self.layout.canary, report)
        if projection and projection["projected_bytes"] > self.derived_cap:
            raise DivergenceError("canary projection exceeds derived_cap")
        pending_items = isic2020.pending(analysis.divergences)
        if pending_items:
            raise DivergenceError(f"canary done, but {len(pending_items)} divergence(s) are pending (READY blocked)")
        public = {key: value for key, value in report.items() if key != "sessions"}
        self.deps.emit(f"canary: {count} derived, {written} B")
        return public


def cmd_run(cfg, root, deps, *, mode, limit=None, strict=True):
    if mode not in ("canary", "full"):
        raise ValueError("mode must be canary or full")
    limit = limit or cfg.canary["limit"]
    return Runner(cfg, Layout(root, cfg), deps, mode, limit, strict).run()


# ---------------------------------------------------------------------------
# verify -> manifest/* -> READY.json -> splits -> public summary
# ---------------------------------------------------------------------------

def _csv_bytes(rows, header):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def cmd_verify(cfg, root, deps):
    layout = Layout(root, cfg)
    if not layout.ledger.exists():
        raise Incomplete("no ledger: run canary/full first")
    with ProcessLock(layout.lock), Ledger(layout.ledger) as ledger:
        return _verify(cfg, layout, deps, ledger)


def _verify(cfg, layout, deps, ledger):
    analysis = analyze(cfg, layout)
    meta = ledger.meta()
    if meta.get("central_directory_sha256") != analysis.cd_sha256 or meta.get("config_identity_sha256") != cfg.identity_sha256:
        raise DivergenceError("ledger identity differs from the retained central directory or the config identity")
    counts = ledger.counts()
    if counts["PENDING"] or counts["FETCHED"]:
        raise Incomplete(f"acquisition incomplete: {counts['PENDING']} PENDING, {counts['FETCHED']} FETCHED; resume with full")
    leftovers = [item.name for item in os.scandir(layout.spool)]
    if leftovers:
        raise IntegrityError(f"spool is not empty ({len(leftovers)} files)")
    temps = [item.name for item in os.scandir(layout.derived) if not item.name.endswith(".jpg") or item.name.endswith(".tmp")]
    if temps:
        raise IntegrityError(f"derived directory holds {len(temps)} non-final files")
    rows = ledger.members()
    entries_by_idx = {entry.index: entry for entry in analysis.entries}
    derived_rows = {row["image"]: row for row in rows if row["state"] == "DERIVED"}
    on_disk = {name[:-4] for name in os.listdir(layout.derived)}
    if on_disk != set(derived_rows):
        raise IntegrityError(f"derived files ({len(on_disk)}) != DERIVED members ({len(derived_rows)})")
    derived_total = 0
    for image, row in derived_rows.items():
        info = json.loads(row["info"])["derived"]
        path = layout.derived / f"{image}.jpg"
        if path.stat().st_size != info["bytes"] or file_hash(path) != info["sha256"]:
            raise IntegrityError("a derived file does not match its ledger hash")
        derived_total += info["bytes"]
    sha_by_image = {row["image"]: row["sha256_original"] for row in rows
                    if row["kind"] == "image" and row["sha256_original"]}
    exact_pairs, hash_divergences = isic2020.exact_hash_check(analysis.pairs, sha_by_image)
    skipped_decode = [row["image"] for row in rows if row["kind"] == "image" and row["state"] == "SKIPPED"
                      and (row["reason"] or "").startswith("decode-failed")]
    divergences = list(analysis.divergences) + hash_divergences
    if skipped_decode:
        divergences.append(isic2020.divergence("derive.skipped_images", 0, len(skipped_decode), "images_zip", skipped_decode))
    isic2020.apply_acceptance(divergences, cfg.accepted_divergences)
    _write_divergences(layout, divergences, "verify")
    open_items = isic2020.pending(divergences)
    if open_items:
        deps.emit(f"verify: {len(open_items)} pending divergence(s): {_divergence_brief(open_items)}")
        raise DivergenceError(f"verify: {len(open_items)} pending divergence(s); READY not written")

    records = analysis.records
    all_pairs = sorted(set(analysis.pairs) | set(exact_pairs))
    patient_to_group, image_to_group, group_stats = groups.build_groups(records, all_pairs, cfg.group_prefix)
    excluded = groups.duplicate_policy(records, all_pairs)
    official = set(analysis.pairs)
    pair_rows = []
    for a, b in all_pairs:
        if a in records and b in records:
            pair_rows.append({"a": a, "b": b, "source": "official+exact-sha256" if (a, b) in official and (a, b) in exact_pairs
                              else ("official" if (a, b) in official else "exact-sha256"),
                              "same_patient": records[a]["patient_id"] == records[b]["patient_id"],
                              "label_conflict": records[a]["label"] != records[b]["label"]})
    by_image = {row["image"]: row for row in rows if row["kind"] == "image"}
    manifest_rows, record_docs, private_records, included = [], [], [], []
    obs = {"exif_orientation_ne_1": 0, "icc_present": 0, "non_rgb_mode": 0, "short_side_lt_512": 0}
    for image in sorted(records):
        rec = records[image]
        row = by_image.get(image)
        status, reason, original, derived_info = "missing-image", "accepted-divergence:labels_without_image", None, None
        if row is not None:
            entry = entries_by_idx[row["idx"]]
            original = {"zip_member": entry.name, "crc32": f"{entry.crc32:08x}", "compressed_size": entry.compressed_size,
                        "size": entry.uncompressed_size, "sha256": row["sha256_original"]}
            if row["state"] == "DERIVED":
                info = json.loads(row["info"])
                original.update(info["original"])
                derived_info = info["derived"]
                obs["exif_orientation_ne_1"] += int(info["original"].get("exif_orientation", 1) != 1)
                obs["icc_present"] += int(bool(info["original"].get("icc")))
                obs["non_rgb_mode"] += int(info["original"].get("mode", "RGB") != "RGB")
                obs["short_side_lt_512"] += int(bool(info.get("short_side_lt_target")))
                status, reason = ("excluded", excluded[image]) if image in excluded else ("included", None)
            else:
                status, reason = "skipped", row["reason"]
        group = image_to_group[image]
        private_records.append({"image": image, "patient_id": rec["patient_id"], "lesion_id": rec["lesion_id"],
                                "group": group, "label": rec["label"], "diagnosis": rec["diagnosis"],
                                "benign_malignant": rec["benign_malignant"], "status": status, "reason": reason,
                                "original": original, "derived": derived_info})
        if status == "included":
            path = layout.derived_rel(image)
            manifest_rows.append((path, group, rec["label"], derived_info["sha256"]))
            record_docs.append({"id": path, "path": path, "group": group, "label": rec["label"]})
            included.append({"group": group, "label": rec["label"]})
    manifest_csv = _csv_bytes(manifest_rows, ("path", "group", "label", "sha256"))
    records_json = {"schema": "augment-atlas/records/v1", "records": record_docs}
    records_json["canonical_hash"] = canonical_hash(records_json)
    _atomic_bytes(layout.manifest / "manifest.csv", manifest_csv)
    atomic_json(layout.manifest / "records.json", records_json)

    sessions = ledger.sessions()
    zip_report = json.loads(layout.probe.read_text(encoding="utf-8"))["sources"]["images_zip"]
    decoder = None
    if deps.deriver is derive.derive_member and derive.pillow_available():
        decoder = derive.decoder_versions()
    tool_commit = _tool_commit(deps)
    image_digest = deps.docker_image_digest or os.environ.get("AUGMENT_ATLAS_IMAGE_DIGEST")
    observed = dict(analysis.observed)
    observed.update(obs, duplicate_pairs_exact_hash=len(exact_pairs), near_duplicate_pairs=None,
                    skipped=sum(1 for r in private_records if r["status"] == "skipped"),
                    derived_bytes_total=derived_total)
    sources = [{"role": "images_zip", "url": zip_report["url"], "content_length": zip_report["content_length"],
                "etag": zip_report["etag"], "last_modified": zip_report["last_modified"], "sha256": None,
                "sha256_null_reason": "not computed: streaming verifies per-member CRC32 (CD) + SHA256",
                "chunked_sha256": None, "central_directory_sha256": analysis.cd_sha256}]
    for role in CSV_ROLES:
        source = cfg.source(role)
        sources.append({"role": role, "url": source["url"], "bytes": source["bytes"], "etag": source["etag"],
                        "sha256": analysis.release[role]["sha256"], "md5": analysis.release[role]["md5"],
                        "header": source["header"]})
    private = {
        "schema": "augment-atlas/dataset-manifest-private/v1",
        "dataset": {key: cfg.dataset.get(key) for key in ("name", "release", "doi", "license", "task", "label_column",
                                                           "patient_column")},
        "acquisition": {"tool_commit": tool_commit, "config_sha256": cfg.sha256, "config_identity_sha256": cfg.identity_sha256,
                        "docker_image_digest": image_digest, "decoder": decoder, "sources": sources,
                        "sessions": sessions},
        "preprocessing": cfg.preprocessing,
        "documented_counts": cfg.documented,
        "informational_counts": cfg.raw.get("informational_counts", {}),
        "observed_counts": observed,
        "divergences": divergences,
        "duplicates": {"policy": cfg.raw["duplicates"]["policy"], "pairs": pair_rows,
                       "excluded": [{"image": image, "reason": reason} for image, reason in sorted(excluded.items())]},
        "groups": {"algorithm": "patient-union-duplicates-v1", "n_groups": group_stats["groups"],
                   "patient_to_group": patient_to_group},
        "records": private_records,
        "outputs": {"manifest_csv": {"sha256": hashlib.sha256(manifest_csv).hexdigest(), "rows": len(manifest_rows)},
                    "records_json": {"sha256": file_hash(layout.manifest / "records.json"), "rows": len(record_docs)}},
    }
    private["sha256"] = canonical_hash(private)
    atomic_json(layout.manifest / "manifest-private.json", private)

    files = {name: file_hash(layout.manifest / name) for name in ("manifest.csv", "records.json", "manifest-private.json",
                                                                  "divergences.json")}
    ready = {"schema": "augment-atlas/dataset-ready/v1", "dataset": cfg.dataset["name"],
             "preprocessing_id": cfg.preprocessing["id"], "config_sha256": cfg.sha256,
             "config_identity_sha256": cfg.identity_sha256, "files": files, "written_utc": utc_now()}
    atomic_json(layout.ready, ready)  # written LAST among manifest/*

    pools_doc, folds_doc, audit_doc = splits_global.build(included, cfg.splits)
    layout.splits.mkdir(parents=True, exist_ok=True)
    if audit_doc["passed"]:
        atomic_json(layout.splits / "pools.json", pools_doc)
        atomic_json(layout.splits / "folds.json", folds_doc)
    atomic_json(layout.splits / "audit.json", audit_doc)

    identifiers = set(records) | {r["patient_id"] for r in records.values()} | {r["lesion_id"] for r in records.values()}
    identifiers |= set(patient_to_group.values()) | {entry.name for entry in analysis.entries if not entry.is_dir}
    identifiers |= {name for pair in analysis.pairs for name in pair}
    hashes = {"manifest_private_sha256": files["manifest-private.json"], "manifest_csv_sha256": files["manifest.csv"],
              "records_json_sha256": files["records.json"], "ready_sha256": file_hash(layout.ready)}
    cross = sum(1 for p in pair_rows if not p["same_patient"])
    dup_public = {"official_pairs": len(analysis.pairs), "exact_hash_pairs": len(exact_pairs), "near_pairs_reported": None,
                  "cross_patient_pairs": cross, "label_conflict_pairs": sum(1 for p in pair_rows if p["label_conflict"]),
                  "images_excluded_by_policy": len(excluded)}
    splits_public = {"algorithm": splits_global.ALGORITHM, "split_seed": cfg.splits["split_seed"],
                     "passed": audit_doc["passed"], "counts": audit_doc["counts"],
                     "failures_count": len(audit_doc["failures"]), "pools_sha256": pools_doc["canonical_hash"],
                     "folds_sha256": folds_doc["canonical_hash"]}
    document = summary.build_summary(
        cfg=cfg, tool_commit=tool_commit, docker_image_digest=image_digest,
        zip_source={"etag": zip_report["etag"], "content_length": zip_report["content_length"],
                    "central_directory_sha256": analysis.cd_sha256, "chunked_sha256_root": None},
        hashes=hashes, included=included,
        excluded_count=sum(1 for r in private_records if r["status"] != "included"), groups_stats=group_stats,
        duplicates=dup_public, divergences=divergences, derived_bytes_total=derived_total,
        splits_counts=splits_public, identifiers=identifiers)
    atomic_json(layout.public / f"{cfg.dataset['name']}-summary-public.json", document)
    deps.emit(f"verify: READY written; {len(manifest_rows)} images included, {group_stats['groups']} groups; "
              f"splits audit {'passed' if audit_doc['passed'] else 'FAILED'}")
    if not audit_doc["passed"]:
        raise DivergenceError(f"split audit failed ({len(audit_doc['failures'])} failure(s)); no seed retry, escalate")
    return document


__all__ = ["Analysis", "Deps", "Layout", "Runner", "analyze", "cmd_probe", "cmd_run", "cmd_verify", "probe_network"]
