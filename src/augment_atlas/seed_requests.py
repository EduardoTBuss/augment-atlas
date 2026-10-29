"""Offline seed-request control plane. No Git mutation, network or training imports."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol

from .fleet import with_requirements
from .fleet_schema import HEX64, IMAGE_ID, SLUG, validate_public_card, validate_requirements
from .global_jobs import ARMS
from .queue import parse_record, render_record, validate_anchor
from .schema import digest, new_record, prepare_job
from .storage import atomic_json

DEFAULT_STOP_AT = "2026-11-12T00:00:00Z"
REQUEST_FILE = "control/seed-requests.md"
STATE_FILE = "seed-requests-state.json"
_REQUIRED = {"request_id", "dataset", "campaign_sha256", "seeds", "node_id",
             "requested_on", "requested_by", "decided_after_results"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_FENCE = chr(96) * 3


def utc(value):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be UTC ending in Z")
    result = datetime.fromisoformat(value[:-1] + "+00:00")
    if result.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must be UTC")
    return result


def _now(value):
    if value.utcoffset() is None:
        raise ValueError("now must include a UTC offset")
    return value.astimezone(timezone.utc)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text):
    return json.loads(text, object_pairs_hook=_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))


def validate_request(value):
    if type(value) is not dict:
        raise ValueError("request must be a JSON object")
    missing, extra = _REQUIRED - value.keys(), value.keys() - _REQUIRED - {"note"}
    if missing or extra:
        raise ValueError(f"request schema: missing {sorted(missing)}; unknown {sorted(extra)}")
    for key in ("request_id", "dataset"):
        if not isinstance(value[key], str) or not _ID.fullmatch(value[key]):
            raise ValueError(f"{key}: expected a safe identifier")
    if not isinstance(value["node_id"], str) or not SLUG.fullmatch(value["node_id"]):
        raise ValueError("node_id: expected a fleet node identifier")
    if not isinstance(value["campaign_sha256"], str) or not HEX64.fullmatch(value["campaign_sha256"]):
        raise ValueError("campaign_sha256: expected lowercase hex64")
    seeds = value["seeds"]
    if (type(seeds) is not list or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("seeds: expected a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds: duplicate seed")
    if 11 in seeds:
        raise ValueError("seeds: seed 11 is not new")
    stamp = value["requested_on"]
    if not isinstance(stamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", stamp):
        raise ValueError("requested_on: expected YYYY-MM-DD")
    date.fromisoformat(stamp)
    for key in ("requested_by", "note"):
        if key in value and (not isinstance(value[key], str) or not value[key].strip()):
            raise ValueError(f"{key}: expected nonempty text")
    if type(value["decided_after_results"]) is not bool:
        raise ValueError("decided_after_results: explicit boolean is required")
    if value["decided_after_results"]:
        raise ValueError("decided_after_results: true conflicts with SAP item 5 (time/budget only)")
    return value


def parse_requests(markdown):
    """Strict closed JSON fences; prose is allowed outside fences.

    Parsing/schema errors fail the whole file instead of partially applying an
    ambiguous document. Contextual errors are reported per request by the planner.
    """
    requests, seen, block, start = [], set(), None, None
    for number, line in enumerate(markdown.splitlines(), 1):
        if line.lstrip().startswith(("~~~", _FENCE)):
            if block is None:
                if line != _FENCE + "json":
                    raise ValueError(f"line {number}: expected an exact JSON opening fence")
                block, start = [], number
            elif line == _FENCE:
                try:
                    request = validate_request(_json("\n".join(block)))
                except (ValueError, TypeError) as error:
                    raise ValueError(f"JSON block at line {start}: {error}") from error
                if request["request_id"] in seen:
                    raise ValueError(f"line {start}: duplicate request_id {request['request_id']}")
                seen.add(request["request_id"])
                requests.append(request)
                block = None
            else:
                raise ValueError(f"line {number}: invalid JSON closing fence")
        elif block is not None:
            block.append(line)
        elif line.lstrip().startswith(("{", "}")):
            raise ValueError(f"line {number}: request JSON must be inside a closed JSON fence")
    if block is not None:
        raise ValueError(f"line {start}: unclosed JSON block")
    return requests


@dataclass(frozen=True)
class AuthorizedStage:
    """Output AFTER sealing and durably anchoring the stage.

    Specs reference audited, materialized configs on execution_code_commit.
    The adapter rechecks new seeds, campaign/image/freeze/pools/plan, and is
    idempotent by request_sha256, including retry after a lost response.
    """
    stage_sha256: str
    anchor_commit: str
    image_id: str
    job_specs: list[dict]


class StageAuthorizer(Protocol):
    def authorize(self, *, request: dict, request_sha256: str, node_card: dict,
                  environment: dict, stop_at: str, queue_checkout: Path) -> AuthorizedStage:
        """Audit/seal/anchor; return only after the stage anchor is durable.

        The real adapter owns the private scientific inputs and verifies the
        additive execution commit. It MUST reject an image different from the
        sealed campaign. It MUST NOT create jobs; this module writes them later.
        """
        ...


def load_state(state_dir):
    path = Path(state_dir) / STATE_FILE
    if not path.exists():
        return {"schema_version": 1, "requests": {}}
    state = _json(path.read_text(encoding="utf-8"))
    if (type(state) is not dict or set(state) != {"schema_version", "requests", "sha256"}
            or state["schema_version"] != 1 or type(state["requests"]) is not dict
            or state["sha256"] != digest({k: v for k, v in state.items() if k != "sha256"})):
        raise ValueError("seed request state: invalid schema or checksum; do not reset state-dir")
    for key, item in state["requests"].items():
        if (type(item) is not dict or item.get("request", {}).get("request_id") != key
                or digest(item["request"]) != item.get("request_sha256")
                or item.get("status") not in {"authorizing", "authorized", "applied"}
                or not isinstance(item.get("stop_at"), str)):
            raise ValueError(f"seed request state: invalid entry {key}")
        validate_request(item["request"])
        utc(item["stop_at"])
    return {k: v for k, v in state.items() if k != "sha256"}


def _save(state_dir, state):
    atomic_json(Path(state_dir) / STATE_FILE, {**state, "sha256": digest(state)})


@contextmanager
def _lock(state_dir):
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    path = Path(state_dir) / "seed-requests.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("seed requests: state-dir locked; inspect the prior process before removing the lock") from error
    try:
        os.close(fd)
        yield
    finally:
        path.unlink()


def read_queue(checkout):
    """Read explicit regular queue files from a local checkout; never fetch."""
    checkout = Path(checkout)
    records, anchors = [], []
    for path in sorted((checkout / "jobs").glob("*.md")):
        if not HEX64.fullmatch(path.stem):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"non-regular queue job: {path.name}")
        record = parse_record(path.read_text(encoding="utf-8"))
        if path.stem != record["job_id"]:
            raise ValueError("queue job filename differs from identity")
        records.append(record)
    for path in sorted((checkout / "anchors").glob("*/*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("non-regular queue anchor")
        anchor = validate_anchor(_json(path.read_text(encoding="utf-8")))
        if path.parent.name != anchor["dataset"]["id"]:
            raise ValueError("queue anchor directory differs from dataset")
        anchors.append(anchor)
    return records, anchors


def read_cards(directory):
    cards = {}
    for path in sorted(Path(directory).glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("non-regular node card")
        card = validate_public_card(_json(path.read_text(encoding="utf-8")))
        if path.stem != card["node_id"]:
            raise ValueError("node card filename differs from node_id")
        cards[card["node_id"]] = card
    return cards


def _context(request, records, anchors, cards):
    matching = [a for a in anchors if a["kind"] == "campaign"
                and a["campaign_sha256"] == request["campaign_sha256"]
                and a["dataset"]["id"] == request["dataset"]]
    if len(matching) != 1:
        raise ValueError("campaign_sha256: campaign not uniquely anchored for this dataset")
    anchor = matching[0]
    card = cards.get(request["node_id"])
    if card is None:
        raise ValueError("node_id: node does not exist")
    if card["status"] != "active" or not card.get("hardware_class"):
        raise ValueError("node_id: node must be active with a hardware_class")
    dataset = anchor["dataset"]
    datasets = [d for d in card["datasets"] if d["id"] == dataset["id"]
                and d["version"] in (dataset["version"], "*")]
    if len(datasets) != 1:
        raise ValueError("node dataset: exact dataset/version must be validated once")
    jobs = [r for r in records if (r["spec"].get("global") or {}).get("campaign_sha256") == request["campaign_sha256"]
            and r["spec"]["dataset"] == request["dataset"]]
    pinned = {r["spec"].get("requirements", {}).get("image_id")
              or r["spec"].get("compatibility", {}).get("image_id") for r in jobs}
    pinned.discard(None)
    if len(pinned) > 1:
        raise ValueError("campaign image: conflicting pinned images in queue")
    images = [i for i in card["images"] if not pinned or i["image_id"] in pinned]
    if len(images) != 1 or not isinstance(images[0].get("probe"), dict) or not images[0]["probe"]:
        raise ValueError("node image: one matching pinned image with a probe is required")
    image = images[0]
    environment = {"image_id": image["image_id"], "probe": deepcopy(image["probe"])}
    environment["environment_sha256"] = digest(environment)
    environment["node_profile_sha256"] = card["profile_sha256"]
    return {"anchor": anchor, "card": card, "dataset": datasets[0],
            "environment": environment, "jobs": jobs}


def _progress(request, jobs):
    selected = [job for job in jobs if job["spec"]["dataset"] == request["dataset"]
                and (job["spec"].get("global") or {}).get("campaign_sha256") == request["campaign_sha256"]
                and job["spec"]["train_seed"] in request["seeds"] and job["spec"]["phase"] == "final"]
    counts = Counter(job["state"] for job in selected)
    by_seed = {}
    for seed in request["seeds"]:
        seed_counts = Counter(j["state"] for j in selected if j["spec"]["train_seed"] == seed)
        by_seed[str(seed)] = {"succeeded": seed_counts["succeeded"], "states": dict(sorted(seed_counts.items()))}
    return selected, {"succeeded": counts["succeeded"], "total": len(selected), "by_seed": by_seed,
                      "expected": len(ARMS) * 5 * len(request["seeds"]),
                      "states": dict(sorted(counts.items()))}


def plan_requests(requests, records, anchors, cards, state, *, now, stop_at=DEFAULT_STOP_AT,
                  published_snapshot=False):
    now, stop = _now(now), utc(stop_at)
    saved = state["requests"]
    reservations = {}
    for request in [*requests, *(s["request"] for s in saved.values())]:
        for seed in request["seeds"]:
            reservations.setdefault((request["campaign_sha256"], seed), set()).add(request["request_id"])
    rows = []
    for request in requests:
        rid = request["request_id"]
        selected, progress = _progress(request, records)
        old = saved.get(rid)
        row = {**deepcopy(request), "request_sha256": digest(request), "status": "pending",
               "actions": [], "progress": progress, "error": None}
        try:
            if old and old["request_sha256"] != row["request_sha256"]:
                raise ValueError("request is immutable after authorization: content changed")
            if old and old["stop_at"] != stop_at:
                raise ValueError("stop_at is immutable after authorization")
            for seed in request["seeds"]:
                owners = reservations[(request["campaign_sha256"], seed)] - {rid}
                if owners and (not old or any(other in saved for other in owners)):
                    raise ValueError(f"seed {seed}: duplicate campaign seed request")
            context = _context(request, records, anchors, cards)
            if not old and selected:
                raise ValueError("seeds: seed already exists in this campaign")
            if old:
                if old["environment"] != context["environment"]:
                    raise ValueError("node environment/profile changed after authorization")
                for job in selected:
                    spec = job["spec"]
                    if (spec.get("requirements", {}).get("allowed_nodes") != [request["node_id"]]
                            or spec.get("seed_request", {}).get("request_sha256") != row["request_sha256"]):
                        raise ValueError("seed affinity or immutable request binding differs in queue")
                if old.get("specs"):
                    expected = {s["job_id"]: s for s in old["specs"]}
                    if any(j["job_id"] not in expected or j["spec"] != expected[j["job_id"]] for j in selected):
                        raise ValueError("immutable authorized spec changed in queue")
            if old and old["status"] == "applied" and progress["total"] != progress["expected"] and not published_snapshot:
                raise ValueError("applied jobs disappeared from queue; do not recreate execution history")
            if progress["succeeded"] == progress["expected"] and progress["total"] == progress["expected"]:
                row["status"] = "done"
            elif any(j["state"] in {"claimed", "running"} for j in selected):
                row["status"] = "running"
            elif now >= stop or utc(request["requested_on"] + "T00:00:00Z") >= stop:
                row["status"] = "expired"
                for counts in progress["by_seed"].values():
                    counts["states"]["expired"] = counts["states"].get("pending", 0)
            elif old:
                row["status"] = "authorized"
                if old["status"] != "applied":
                    row["actions"] = ["resume_authorization" if old["status"] == "authorizing" else "materialize_jobs"]
                elif progress["total"] != progress["expected"] and published_snapshot:
                    row["detail"] = "awaiting_queue_publication"
            elif now.date() < date.fromisoformat(request["requested_on"]):
                row["actions"] = ["wait_until_requested_on"]
            else:
                row["actions"] = ["authorize_stage", "materialize_jobs"]
            if old and old["status"] == "authorized" and row["status"] == "running" and now < stop:
                row["actions"] = ["materialize_jobs"]
            if now >= stop:
                for counts in progress["by_seed"].values():
                    counts["states"]["expired"] = counts["states"].get("pending", 0)
            row["environment"] = context["environment"]
        except (ValueError, KeyError, TypeError) as error:
            row.update(status="invalid", actions=[], error=str(error))
        rows.append(row)
    for rid, old in sorted(saved.items()):
        if rid not in {r["request_id"] for r in requests}:
            _, progress = _progress(old["request"], records)
            rows.append({**old["request"], "status": "invalid", "actions": [], "progress": progress,
                         "error": "authorized request was removed from the control file"})
    return rows


def _stage_specs(stage, request, context, stop_at):
    if (not isinstance(stage, AuthorizedStage) or not HEX64.fullmatch(stage.stage_sha256)
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", stage.anchor_commit)
            or not IMAGE_ID.fullmatch(stage.image_id)
            or stage.image_id != context["environment"]["image_id"]):
        raise ValueError("authorizer: invalid durable stage receipt or pinned image mismatch")
    specs, identities = [], set()
    expected = {(seed, arm, fold) for seed in request["seeds"] for arm in ARMS for fold in range(5)}
    for raw in stage.job_specs:
        spec = deepcopy(raw)
        spec.pop("job_id", None)
        bound = spec.get("global", {})
        anchor = context["anchor"]
        if (spec.get("phase") != "final" or spec.get("dataset") != request["dataset"]
                or spec.get("dataset_version") != anchor["dataset"]["version"]
                or bound.get("campaign_sha256") != request["campaign_sha256"]
                or bound.get("pools_sha256") != anchor["pools_sha256"]
                or bound.get("freeze_sha256") != anchor["freeze_sha256"]
                or bound.get("freeze_kind") != anchor["recipe_source"]
                or spec.get("recipe_hash") != anchor["freeze_sha256"]
                or not re.fullmatch(r"[0-9a-f]{40}", spec.get("code_commit", ""))
                or not isinstance(spec.get("config"), str)
                or not HEX64.fullmatch(spec.get("config_sha256", ""))):
            raise ValueError("authorizer: spec differs from campaign/freeze/pools/config identity")
        identity = (spec.get("train_seed"), spec.get("arm"), spec.get("fold"))
        if identity not in expected or identity in identities:
            raise ValueError("authorizer: duplicate or unrequested seed/arm/fold")
        identities.add(identity)
        req = deepcopy(validate_requirements(spec.get("requirements")))
        if req["image_id"] != stage.image_id:
            raise ValueError("authorizer: spec image differs from pinned image")
        dataset = req["dataset"]
        if (dataset["id"] != request["dataset"] or dataset["version"] != anchor["dataset"]["version"]
                or dataset["manifest_sha256"] != context["dataset"]["manifest_sha256"]):
            raise ValueError("authorizer: spec dataset differs from node validated dataset")
        req.update(allowed_nodes=[request["node_id"]], hardware_class=context["card"]["hardware_class"])
        spec = with_requirements(spec, validate_requirements(req))
        spec["seed_request"] = {"request_id": request["request_id"], "request_sha256": digest(request),
                                "node_id": request["node_id"], "not_after_utc": stop_at,
                                "stage_sha256": stage.stage_sha256, "anchor_commit": stage.anchor_commit,
                                "environment_sha256": context["environment"]["environment_sha256"],
                                "node_profile_sha256": context["card"]["profile_sha256"]}
        spec["environment_sha256"] = context["environment"]["environment_sha256"]
        specs.append(prepare_job(spec))
    if identities != expected:
        raise ValueError("authorizer: incomplete matrix; require all ten arms x five folds per seed")
    return specs


def _materialize(checkout, specs):
    directory = Path(checkout) / "jobs"
    directory.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        path = directory / (spec["job_id"] + ".md")
        if path.is_symlink():
            raise ValueError("refusing symlink job")
        if path.exists() and parse_record(path.read_text(encoding="utf-8"))["spec"] != spec:
            raise ValueError("immutable spec conflict while materializing jobs")
    for spec in specs:
        path = directory / (spec["job_id"] + ".md")
        if path.exists():
            continue
        # Publish a complete file with an exclusive hard link: a killed writer
        # cannot leave a truncated .md, and a race cannot overwrite job history.
        fd, temporary = tempfile.mkstemp(prefix=".seed-request-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(render_record(new_record(spec)))
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or parse_record(path.read_text(encoding="utf-8"))["spec"] != spec:
                    raise ValueError("concurrent immutable spec conflict")
        finally:
            Path(temporary).unlink()


def process_requests(control_checkout, queue_checkout, node_cards, state_dir, *, apply=False,
                     authorizer: StageAuthorizer | None = None, now=None, stop_at=DEFAULT_STOP_AT):
    """Plan is read-only; apply locks state and journals before each side effect."""
    clock = (lambda: _now(now)) if now is not None else (lambda: datetime.now(timezone.utc))

    def process():
        state = load_state(state_dir)
        try:
            requests = parse_requests((Path(control_checkout) / REQUEST_FILE).read_text(encoding="utf-8"))
        except ValueError as error:
            return [{"request_id": "file", "dataset": "-", "seeds": [], "node_id": "-",
                     "status": "invalid", "actions": [], "error": str(error),
                     "progress": {"succeeded": 0, "total": 0, "expected": 0}}]
        records, anchors = read_queue(queue_checkout)
        cards = read_cards(node_cards)
        rows = plan_requests(requests, records, anchors, cards, state, now=clock(), stop_at=stop_at)
        if not apply:
            return rows
        for request, row in zip(requests, rows):
            if not row["actions"] or row["actions"] == ["wait_until_requested_on"]:
                continue
            rid = request["request_id"]
            try:
                old = state["requests"].get(rid)
                if not old or old["status"] == "authorizing":
                    if authorizer is None:
                        raise ValueError("StageAuthorizer adapter unavailable; configure --authorizer module:factory")
                    context = _context(request, records, anchors, cards)
                    if old and old["environment"] != context["environment"]:
                        raise ValueError("node environment changed during authorization; inspect the durable stage")
                    if clock() >= utc(stop_at):
                        row.update(status="expired", actions=[])
                        continue
                    old = {"request": deepcopy(request), "request_sha256": digest(request),
                           "status": "authorizing", "stop_at": stop_at,
                           "environment": context["environment"]}
                    state["requests"][rid] = old
                    _save(state_dir, state)
                    stage = authorizer.authorize(
                        request=deepcopy(request), request_sha256=digest(request), node_card=deepcopy(context["card"]),
                        environment=deepcopy(context["environment"]), stop_at=stop_at,
                        queue_checkout=Path(queue_checkout))
                    specs = _stage_specs(stage, request, context, stop_at)
                    old.update(status="authorized", specs=specs, specs_sha256=digest(specs),
                               stage_sha256=stage.stage_sha256, anchor_commit=stage.anchor_commit)
                    _save(state_dir, state)
                if digest(old["specs"]) != old["specs_sha256"]:
                    raise ValueError("authorized specs checksum mismatch")
                if clock() >= utc(stop_at):
                    row.update(status="expired", actions=[])
                    continue
                current_records, _ = read_queue(queue_checkout)
                selected, _ = _progress(request, current_records)
                expected = {spec["job_id"]: spec for spec in old["specs"]}
                if any(job["job_id"] not in expected or job["spec"] != expected[job["job_id"]] for job in selected):
                    raise ValueError("campaign seed already bound to different immutable specs")
                _materialize(queue_checkout, old["specs"])
                old["status"] = "applied"
                _save(state_dir, state)
                current_records, _ = read_queue(queue_checkout)
                _, progress = _progress(request, current_records)
                active = any(r["state"] in {"claimed", "running"} for r in current_records
                             if r["job_id"] in expected)
                row.update(status="running" if active else "authorized", actions=[], progress=progress)
            except (ValueError, OSError, RuntimeError, TypeError, KeyError) as error:
                row.update(status="invalid", actions=[], error=str(error))
        return rows

    if apply:
        if Path(queue_checkout).resolve() == Path(state_dir).resolve():
            raise ValueError("state-dir must be separate from queue-checkout")
        with _lock(queue_checkout), _lock(state_dir):
            return process()
    return process()


def main(argv=None, *, authorizer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply"))
    for name in ("control-checkout", "queue-checkout", "node-cards", "state-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--stop-at", default=DEFAULT_STOP_AT)
    parser.add_argument("--now", help="UTC timestamp for an offline plan (not allowed with apply)")
    parser.add_argument("--authorizer", help="trusted installed Python module:factory; no adapter is bundled")
    args = parser.parse_args(argv)
    if args.now and args.command == "apply":
        parser.error("--now is plan-only; apply always checks the real clock")
    try:
        if args.authorizer and args.command == "apply":
            module, factory = args.authorizer.split(":", 1)
            authorizer = getattr(importlib.import_module(module), factory)()
        rows = process_requests(args.control_checkout, args.queue_checkout, args.node_cards, args.state_dir,
                                apply=args.command == "apply", authorizer=authorizer,
                                now=utc(args.now) if args.now else None, stop_at=args.stop_at)
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 1 if any(r["status"] == "invalid" for r in rows) else 0
    except (ValueError, OSError, RuntimeError, ImportError, AttributeError) as error:
        parser.exit(1, f"seed requests: {error}\n")


def board_requests(control_checkout, records, anchors, node_cards, state_dir, *, now, stop_at=DEFAULT_STOP_AT):
    """Combine control input/state with the board's pinned committed queue snapshot."""
    try:
        requests = parse_requests((Path(control_checkout) / REQUEST_FILE).read_text(encoding="utf-8"))
        return plan_requests(requests, records, anchors, read_cards(node_cards), load_state(state_dir),
                             now=now, stop_at=stop_at, published_snapshot=True)
    except (ValueError, OSError) as error:
        return [{"request_id": "file", "dataset": "-", "seeds": [], "node_id": "-",
                 "status": "invalid", "actions": [], "error": str(error),
                 "progress": {"succeeded": 0, "total": 0, "expected": 0}}]


if __name__ == "__main__":
    raise SystemExit(main())
