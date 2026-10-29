"""Read-only Markdown campaign board from one committed queue snapshot (stdlib)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
from statistics import median
import subprocess

from .queue import QueueError, parse_record, validate_anchor


STATES = ("pending", "claimed", "running", "succeeded", "failed", "interrupted", "cancelled")
STALE_SECONDS = 30 * 60
RECENT_LIMIT = 20
_JOB_PATH = re.compile(r"jobs/[0-9a-f]{64}\.md\Z")
_ANCHOR_PATH = re.compile(r"anchors/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\.json\Z")


def _git(checkout, *args, check=True):
    return subprocess.run(["git", "-C", str(checkout), *args], check=check,
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


def read_snapshot(checkout):
    """Prefer fetched origin/queue, else local queue; never read working files/HEAD.

    Pin a full commit before listing blobs, so a concurrent fetch cannot mix two
    versions. Only regular job/anchor blobs from that tree are ever opened.
    """
    for ref in ("refs/remotes/origin/queue", "refs/heads/queue"):
        result = _git(checkout, "rev-parse", "--verify", ref + "^{commit}", check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
            break
    else:
        raise ValueError("no fetched origin/queue or local queue branch")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("invalid queue commit")
    tree = _git(checkout, "ls-tree", "-rz", commit, "--", "jobs/", "anchors/").stdout
    records, anchors = [], []
    for entry in sorted(filter(None, tree.split("\0"))):
        metadata, path = entry.split("\t", 1)
        mode, kind, blob = metadata.split()
        if not (_JOB_PATH.fullmatch(path) or _ANCHOR_PATH.fullmatch(path)):
            continue
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError(f"non-regular queue document: {path}")
        content = _git(checkout, "cat-file", "blob", blob).stdout
        try:
            if _JOB_PATH.fullmatch(path):
                record = parse_record(content)
                if path != f"jobs/{record['job_id']}.md":
                    raise ValueError("job filename differs from identity")
                records.append(record)
            else:
                anchor = validate_anchor(json.loads(content))
                if path.split("/")[1] != anchor["dataset"]["id"]:
                    raise ValueError("anchor directory differs from dataset")
                anchors.append(anchor)
        except (ValueError, TypeError, KeyError, QueueError) as error:
            raise ValueError(f"invalid queue document: {path}") from error
    return commit, records, anchors


def _time(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() is not None:
            return result.astimezone(timezone.utc)
    except ValueError:
        pass
    return None


def _stamp(value):
    return value.isoformat().replace("+00:00", "Z") if value else "unknown"


def _observed(event):
    # Invalid supplied observation must not fall back to a newer publication.
    return _time(event.get("observed_at_utc", event.get("at")))


def _attempt_events(record):
    return [event for event in record["history"]
            if event.get("attempt_id") == record.get("attempt_id")]


def _started(events):
    return next((_observed(event) for event in events if event.get("state") == "running"), None)


def _duration(record):
    events = _attempt_events(record)
    start = _started(events)
    end = _observed(events[-1]) if events and events[-1].get("state") == "succeeded" else None
    return (end - start).total_seconds() if start and end and end >= start else None


def _campaign(spec, anchors):
    global_spec = spec.get("global") or {}
    if global_spec.get("campaign_sha256"):
        return "campaign:" + global_spec["campaign_sha256"]
    campaigns = [a for a in anchors if a["kind"] == "campaign"
                 and a["dataset"] == {"id": spec["dataset"], "version": spec["dataset_version"]}]
    pools = global_spec.get("pools_sha256")
    freeze = global_spec.get("freeze_sha256")
    if freeze is None and spec["phase"] != "hpo":
        freeze = spec.get("recipe_hash")
    source = global_spec.get("freeze_kind") or ("hpo_d" if spec["phase"] == "hpo" else None)
    matches = [a for a in campaigns
               if (not pools or a["pools_sha256"] == pools)
               and (not source or a["recipe_source"] == source)
               and ((freeze and a["freeze_sha256"] == freeze)
                    or (not freeze and pools and spec["phase"] == "hpo"))]
    hashes = {a["campaign_sha256"] for a in matches}
    if len(hashes) == 1:
        return "campaign:" + hashes.pop()
    # Never guess across ambiguous campaigns or combine unrelated legacy jobs.
    if global_spec.get("freeze_sha256"):
        return "freeze:" + global_spec["freeze_sha256"]
    if global_spec.get("plan_sha256"):
        return "plan:" + global_spec["plan_sha256"]
    return "recipe:" + spec["recipe_hash"]


def _cell(value):
    return html.escape(" ".join(str(value).split()), quote=False).replace("|", "&#124;")


def _table(headers, rows):
    return ["| " + " | ".join(map(_cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(_cell, row)) + " |" for row in rows), ""]


def _elapsed(seconds):
    seconds = math.ceil(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_board(commit, records, anchors, now, seed_requests=None):
    """Deterministic rendering; no filesystem, clock, fleet refs or private I/O."""
    if now.utcoffset() is None:
        raise ValueError("now must include a UTC offset")
    now = now.astimezone(timezone.utc)
    totals = Counter(record["state"] for record in records)
    groups, campaigns = defaultdict(list), defaultdict(list)
    for record in records:
        spec = record["spec"]
        key = (spec["dataset"], spec["dataset_version"], _campaign(spec, anchors))
        groups[(*key, spec["phase"], spec["train_seed"])].append(record)
        campaigns[key].append(record)
    lines = ["# Queue board", "", f"Generated UTC: {_stamp(now)}", "",
             f"Queue commit: <code>{commit}</code>", "", f"Total jobs: {len(records)}", ""]
    lines += _table(STATES, [[totals[state] for state in STATES]])
    lines += ["## Campaign progress", "",
              "Completed = succeeded / total. Claimed remains a separate queue state.", ""]
    rows = []
    phase_order = {"smoke": 0, "hpo": 1, "final": 2}
    for key, jobs in sorted(groups.items(), key=lambda item: (*item[0][:3], phase_order[item[0][3]], item[0][4])):
        dataset, version, campaign, phase, seed = key
        count = Counter(job["state"] for job in jobs)
        rows.append([dataset, version, campaign, phase, seed, len(jobs),
                     *[count[state] for state in ("succeeded", "running", "pending", "failed", "interrupted", "cancelled", "claimed")],
                     f"{100 * count['succeeded'] / len(jobs):.1f}%"])
    lines += _table(["Dataset", "Version", "Campaign", "Phase", "Seed", "Total", "Succeeded", "Running",
                     "Pending", "Failed", "Interrupted", "Cancelled", "Claimed", "Completed"], rows)
    if not rows:
        lines += ["No jobs.", ""]
    lines += ["## Running jobs", "",
              "Stale = observation older than 30 min; unknown/future observations are flagged. No automatic recovery.", ""]
    rows = []
    for record in sorted(records, key=lambda r: (r["spec"]["dataset"], r["job_id"])):
        if record["state"] != "running":
            continue
        spec, owner = record["spec"], record.get("owner") or {}
        events = _attempt_events(record)
        observed = _observed(events[-1]) if events else None
        progress = next((e for e in reversed(events) if "epoch" in e), {})
        total = next((e.get("total_epochs", e.get("epochs")) for e in reversed(events)
                      if "total_epochs" in e or "epochs" in e), "?")
        age = (now - observed).total_seconds() if observed else None
        age_text = ("unknown" if age is None else "future" if age < 0 else
                    _elapsed(age) + (" stale" if age > STALE_SECONDS else ""))
        rows.append([record["job_id"], spec["dataset"], _campaign(spec, anchors), spec["arm"], spec["fold"],
                     spec["train_seed"], owner.get("server", "unknown"), owner.get("gpu", owner.get("gpu_uuid", "unknown")),
                     f"{progress.get('epoch', '?')}/{total}",
                     _stamp(_started(events)), _stamp(observed), age_text])
    lines += _table(["Job", "Dataset", "Campaign", "Arm", "Fold", "Seed", "Server", "GPU", "Epoch",
                     "Started UTC", "Last observation UTC", "Observation age (hh:mm:ss)"], rows)
    if not rows:
        lines += ["No running jobs.", ""]
    lines += ["## Recent failures / interruptions", "",
              f"Latest {RECENT_LIMIT} failure/interruption events, including prior attempts; observation time is preferred.", ""]
    failures = []
    for record in records:
        for event in record["history"]:
            if event["state"] not in ("failed", "interrupted"):
                continue
            observed = _observed(event)
            reason = next((event[k] for k in ("cause", "reason", "error", "message") if event.get(k)), "unspecified")
            if isinstance(reason, dict):
                reason = reason.get("code") or reason.get("reason") or "unspecified"
            reason = " ".join(str(reason).split())
            failures.append((observed or datetime.min.replace(tzinfo=timezone.utc), record["job_id"],
                             event["seq"], [record["job_id"], record["spec"]["dataset"], _campaign(record["spec"], anchors),
                                            event["state"], _stamp(observed), reason[:157] + "..." if len(reason) > 160 else reason]))
    lines += _table(["Job", "Dataset", "Campaign", "State", "Observed UTC", "Reason"],
                    [item[3] for item in sorted(failures, key=lambda item: item[:3], reverse=True)[:RECENT_LIMIT]])
    if not failures:
        lines += ["No failures or interruptions.", ""]
    lines += ["## Campaign ETA (estimate)", "",
              "Estimate = median successful current-attempt duration × pending / active GPUs in this campaign.",
              "Active GPUs are unique server + GPU UUID (GPU index fallback) on running jobs observed within 30 min.",
              "Stale/unknown/future observations and claimed jobs add no capacity; running remainder and retries are excluded.", ""]
    rows = []
    for (dataset, version, campaign), jobs in sorted(campaigns.items()):
        durations, active = [], set()
        for job in jobs:
            if job["state"] == "succeeded":
                duration = _duration(job)
                if duration is not None:
                    durations.append(duration)
            elif job["state"] == "running":
                events = _attempt_events(job)
                observed = _observed(events[-1]) if events else None
                owner = job.get("owner") or {}
                gpu = owner.get("gpu_uuid") or owner.get("gpu")
                if observed and 0 <= (now - observed).total_seconds() <= STALE_SECONDS and owner.get("server") and gpu is not None:
                    active.add((owner["server"], str(gpu)))
        pending = sum(job["state"] == "pending" for job in jobs)
        typical = median(durations) if durations else None
        eta = "00:00:00 estimate" if not pending else (
            _elapsed(typical * pending / len(active)) + " estimate" if typical is not None and active else "unavailable estimate")
        rows.append([dataset, version, campaign, len(durations), _elapsed(typical) if typical is not None else "unknown",
                     pending, len(active), eta])
    lines += _table(["Dataset", "Version", "Campaign", "Succeeded timed", "Median (hh:mm:ss)",
                     "Pending", "Active GPUs", "ETA (hh:mm:ss)"], rows)
    lines += ["## Seed requests", ""]
    request_rows = []
    for request in seed_requests or []:
        progress = request["progress"]
        per_seed = "; ".join("{}: {}/50 ".format(seed, counts["succeeded"]) +
                             ", ".join(f"{state}={count}" for state, count in counts["states"].items())
                             for seed, counts in progress.get("by_seed", {}).items())
        request_rows.append([request["request_id"], request["dataset"],
                             ", ".join(map(str, request["seeds"])), request["node_id"], request["status"],
                             "{}/{}".format(progress["succeeded"], progress["expected"]), per_seed,
                             request.get("error") or request.get("detail") or ", ".join(request.get("actions", []))])
    lines += _table(["Request", "Dataset", "Seeds", "Node", "Status", "Succeeded / expected",
                     "Jobs per seed", "Detail"], request_rows)
    if not request_rows:
        lines += ["No seed requests supplied.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-checkout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--control-checkout", type=Path)
    parser.add_argument("--node-cards", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--stop-at", default="2026-11-12T00:00:00Z")
    parser.add_argument("--now", help="ISO 8601 UTC timestamp for deterministic output")
    args = parser.parse_args(argv)
    now = _time(args.now) if args.now else datetime.now(timezone.utc)
    if now is None or (args.now and datetime.fromisoformat(args.now.replace("Z", "+00:00")).utcoffset().total_seconds() != 0):
        parser.error("--now must be an ISO 8601 UTC timestamp with Z or +00:00")
    # The output must never alter the supplied queue clone, even by accident.
    checkout, output = args.queue_checkout.resolve(), args.out.resolve()
    if output.is_relative_to(checkout):
        parser.error("--out must be outside --queue-checkout")
    try:
        commit, records, anchors = read_snapshot(checkout)
        seed_requests = None
        if args.control_checkout:
            if not args.node_cards or not args.state_dir:
                parser.error("--control-checkout requires --node-cards and --state-dir")
            from .seed_requests import board_requests
            seed_requests = board_requests(args.control_checkout, records, anchors, args.node_cards,
                                           args.state_dir, now=now, stop_at=args.stop_at)
        content = render_board(commit, records, anchors, now, seed_requests)
        output.write_text(content, encoding="utf-8", newline="\n")
    except (OSError, ValueError, QueueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"queue board: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
