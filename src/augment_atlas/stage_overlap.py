"""SAP amendment A1: private timing declarations and primary-worker observation.

Observing an anchored supplement never grants a primary worker permission to
execute it. Execution still requires the worker's explicit stage pin.
"""
import json
from pathlib import Path

from . import global_campaign as gc
from . import global_development as gd
from . import global_jobs as gj
from . import stage_authorization as sa

RECORD_KEYS = {"schema_version", "kind", "campaign_sha256", "stage_authorization_sha256",
               "allow_incomplete_stage1", "amendment_and_reason", "recorded_at_utc", "sha256"}


def declaration(text):
    if text is not None and (type(text) is not str or not text.strip()):
        gc._fail("cli_input_invalid", "--allow-incomplete-stage1 needs a non-empty amendment id and reason")
    return text


def record_path(data_root, document):
    return Path(data_root).resolve() / gj._private_name(
        "stage1-overlap", document["dataset"]["id"], document["sha256"])


def validate_record(record, campaign_sha256, authorization_sha256):
    if (type(record) is not dict or set(record) != RECORD_KEYS or gj._seal(record) != record
            or record["schema_version"] != 1 or record["kind"] != "stage1-overlap"
            or record["allow_incomplete_stage1"] is not True
            or record["campaign_sha256"] != campaign_sha256
            or record["stage_authorization_sha256"] != authorization_sha256):
        gc._fail("campaign_invalid", "invalid sealed stage-1 overlap declaration")
    declaration(record["amendment_and_reason"])
    if record["amendment_and_reason"] is None:
        gc._fail("campaign_invalid", "overlap declaration requires amendment id and reason")
    gc._parse_utc(record["recorded_at_utc"], "recorded_at_utc")
    return record


def publish_record(data_root, document, text):
    declaration(text)
    record = gj._seal({"schema_version": 1, "kind": "stage1-overlap",
                       "campaign_sha256": document["campaign_sha256"],
                       "stage_authorization_sha256": document["sha256"],
                       "allow_incomplete_stage1": True, "amendment_and_reason": text,
                       "recorded_at_utc": document["sealed_at_utc"]})
    validate_record(record, document["campaign_sha256"], document["sha256"])
    path = record_path(data_root, document)
    gd._publish(path, record, lambda value: validate_record(
        value, document["campaign_sha256"], document["sha256"]))
    return {"record_path": str(path), "record_sha256": record["sha256"]}


def load_record(data_root, document):
    path = record_path(data_root, document)
    if not path.exists():
        return None
    return validate_record(gd._read(path), document["campaign_sha256"], document["sha256"])


def anchored_authorization(anchors, campaign, data_root, code_repo):
    """Load only this campaign's anchored private document, then fully re-audit it."""
    matches = [anchor for _, anchor in anchors.values()
               if anchor["kind"] == "stage_authorization"
               and anchor["campaign_sha256"] == campaign["sha256"]]
    if not matches:
        return None
    if len(matches) != 1:
        gc._fail("campaign_invalid", "primary observation supports one later-stage authorization")
    anchor = matches[0]
    path = Path(data_root).resolve() / gj._private_name(
        "stage-authorization", campaign["dataset"]["id"], anchor["document_sha256"])
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError):
        gc._fail("campaign_anchor_missing", "anchored stage authorization private document is unavailable")
    sa.audit(document, campaign, code_repo=code_repo)
    if gc.build_anchor("stage_authorization", document) != anchor:
        gc._fail("campaign_invalid", "private stage authorization differs from its public anchor")
    return document


class ClaimContext(tuple):
    """Preserve the three-item execution context; carry private audit inputs separately."""
    def __new__(cls, campaign, authorization, code_repo, data_root):
        value = super().__new__(cls, (campaign, authorization, code_repo))
        value.data_root = data_root
        return value


def require_final_primary(expected, snapshot):
    """A1 primary collection requires every cell succeeded or terminal with a recorded outcome."""
    states = {row["job_id"]: row["state"] for row in snapshot}
    incomplete = sorted(job for job in expected
                        if states.get(job) not in ("succeeded", "failed", "cancelled"))
    if not expected or incomplete:
        gc._fail("stage1_incomplete", "stage1_current requires every primary cell final: "
                 + ", ".join(f"{job}={states.get(job, 'absent')}" for job in incomplete))


def require_current_report(campaign, cells, chronology):
    if not isinstance(chronology, dict) or chronology.get("structural_valid") is not True:
        gc._fail("chronology_invalid", "stage1_current requires a full current chronology audit")
    expected = {(arm, fold, seed) for arm in campaign["arms"] for fold in campaign["folds"]
                for seed in campaign["matrix"]["authorized_seeds"]}
    if set(cells) != expected or any(cell["status"] not in ("succeeded", "failed", "timeout")
                                     for cell in cells.values()):
        gc._fail("stage1_incomplete", "stage1_current requires every primary cell final")
