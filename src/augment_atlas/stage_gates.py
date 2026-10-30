"""R1 gates for later-stage execution on a queue shared with older orchestrators.

* Admission/claim (R1-01): a final job is eligible only through the audited
  campaign's ``matrix.authorized_seeds`` or an audited stage authorization;
  the config path naming (``/stage-``) never decides eligibility.
* Publication (R1-02): workers of the 835c70e lineage validate anchors with the
  closed three-kind ``global-anchor-v1`` schema and stop at ``anchor_invalid``
  once a ``stage_authorization`` anchor exists. Such an anchor is therefore
  published only while no primary (stage-1) final job of ANY dataset or
  campaign is pending, claimed, running or interrupted. Jobs of a later stage
  that is already anchored (same campaign and execution commit) do not block.
* Completion (R1-03): ``authorize-stage`` requires every authorized stage-1
  cell of the campaign to exist and be ``succeeded`` (SAP item 5), so the
  stage-1 snapshot at the parent of the authorization anchor is final.
"""
import json

from . import global_campaign as gc
from . import global_jobs as gj
from .queue import GitQueue, QueueConflict, _is_global_final, validate_anchor

ACTIVE_STATES = ("pending", "claimed", "running", "interrupted")


def check_primary_job(spec, campaign, campaign_sha256):
    """Stage-1 admission: the seed must be authorized by the sealed campaign itself."""
    if (type(campaign) is not dict or set(campaign) != gc._CAMPAIGN_KEYS or gj._seal(campaign) != campaign
            or campaign["sha256"] != campaign_sha256):
        gc._fail("campaign_invalid", "admission needs the sealed campaign named by the anchor")
    from . import stage_authorization as sa
    sa.job_binding(spec, campaign, None)


def _all_anchors(queue, head):
    if head is None:
        return []
    paths = queue._git("ls-tree", "-r", "--name-only", head, "anchors/").stdout.splitlines()
    return [validate_anchor(json.loads(queue.read_path(head, path))) for path in paths if path.endswith(".json")]


def active_primary_jobs(queue, head, records=None):
    """Sorted IDs of non-terminal final jobs not bound to an anchored later stage."""
    records = queue._records(head) if records is None else records
    later = {(anchor["campaign_sha256"], anchor["code_commit"]) for anchor in _all_anchors(queue, head)
             if anchor["kind"] == "stage_authorization"}
    return sorted(job_id for job_id, record in records.items()
                  if record["state"] in ACTIVE_STATES and _is_global_final(record["spec"])
                  and (record["spec"]["global"].get("campaign_sha256"), record["spec"].get("code_commit")) not in later)


def refuse_active_primary(queue, head, records=None, *, released_campaign=None):
    """Refuse while primary jobs are active. With ``released_campaign`` (declared release, D9), only primary
    jobs of that campaign block; the released jobs of other campaigns are returned."""
    active = active_primary_jobs(queue, head, records)
    if released_campaign is None:
        blocking, released = active, []
    else:
        records = queue._records(head) if records is None else records
        blocking = [job_id for job_id in active
                    if records[job_id]["spec"]["global"].get("campaign_sha256") == released_campaign]
        released = [job_id for job_id in active if job_id not in blocking]
    if blocking:
        raise QueueConflict(
            f"primary_workers_active: {len(blocking)} primary final job(s) are pending/claimed/running/interrupted; "
            "publish stage authorizations only after every primary worker has finished (older workers reject "
            "the stage_authorization anchor kind): " + ", ".join(blocking))
    return released


class StageGatedQueue(GitQueue):
    """GitQueue whose stage-authorization publication is gated on the exact HEAD it is built on.

    ``all_workers_upgraded`` (D9) is the operator's non-empty declaration that every worker reading the queue
    runs an orchestrator that understands stage_authorization anchors. It releases active primary jobs of OTHER
    campaigns only; primary jobs of the campaign being authorized always block. ``released_primary_jobs`` is
    None until the gate runs (idempotent republication of an existing anchor never runs it)."""

    def __init__(self, checkout, remote, branch="queue", max_retries=16, *, all_workers_upgraded=None):
        if all_workers_upgraded is not None and (type(all_workers_upgraded) is not str
                                                 or not all_workers_upgraded.strip()):
            raise ValueError("cli_input_invalid: --declare-all-workers-upgraded needs a non-empty declaration")
        super().__init__(checkout, remote, branch, max_retries)
        self.all_workers_upgraded = all_workers_upgraded
        self.released_primary_jobs = None
        self.release_head = None

    def _check_anchor_preconditions(self, head, records, anchor):
        super()._check_anchor_preconditions(head, records, anchor)
        if anchor["kind"] == "stage_authorization":
            released_campaign = anchor["campaign_sha256"] if self.all_workers_upgraded is not None else None
            self.released_primary_jobs = refuse_active_primary(self, head, records,
                                                               released_campaign=released_campaign)
            self.release_head = head


def require_complete_stage1(queue, expected):
    """``expected``: {job_id: (arm, fold, seed, ...)} of the campaign's stage-1 cells."""
    states = {record["job_id"]: record["state"] for record in queue.list_jobs()}
    missing = sorted(job_id for job_id in expected if states.get(job_id) != "succeeded")
    if not expected or missing:
        gc._fail("stage1_incomplete",
                 f"{len(missing)} of {len(expected)} authorized stage-1 cells are absent or not succeeded "
                 "(SAP item 5: later seeds only after all arms x folds of stage 1 complete): "
                 + ", ".join(f"{job_id}={states.get(job_id, 'absent')}" for job_id in missing))
