"""Local scientific adapter for the seed-request control plane.

The operator prepares and commits the additive stage configs beforehand. This
adapter never creates code commits: C9b audits the existing execution commit.
Queue publication uses GitQueue's anchor and seed_final transactions.
"""
from argparse import Namespace
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from . import global_campaign as gc, global_cli as cli, global_collect as col
from . import global_development as gd, global_jobs as gj, stage_authorization as sa
from .fleet_schema import validate_requirements
from .queue import GitQueue, anchor_path, validate_anchor
from .schema import digest
from .seed_requests import AuthorizedStage, _json, validate_request
from .stage_gates import StageGatedQueue, refuse_active_primary


class RealStageAuthorizer:
    def __init__(self, *, code_repo, data_root, source, remote, min_vram_mib, branch="queue"):
        self.code_repo = Path(code_repo).resolve()
        self.data_root = Path(data_root).resolve()
        self.source = Path(source).resolve()
        self.remote, self.branch = str(remote), branch
        if type(min_vram_mib) is not int or min_vram_mib < 0:
            raise ValueError("min_vram_mib must be a nonnegative integer")
        self.min_vram_mib = min_vram_mib

    @classmethod
    def from_file(cls, path):
        path = Path(path).resolve()
        config = _json(path.read_text(encoding="utf-8"))
        required = {"code_repo", "data_root", "source", "remote", "min_vram_mib"}
        if type(config) is not dict or not required <= config.keys() or config.keys() - required - {"branch"}:
            raise ValueError("invalid RealStageAuthorizer configuration")
        for key in ("code_repo", "data_root", "source"):
            config[key] = path.parent / config[key]
        return cls(**config)

    def _campaign(self, request):
        validate_request(request)
        args = Namespace(data_root=self.data_root, dataset=request["dataset"], source=self.source,
                         code=self.code_repo, campaign_sha256=request["campaign_sha256"])
        campaign = cli._document(args, "campaign", request["campaign_sha256"])
        args.pools_sha256 = campaign["pools_sha256"]
        inputs = cli._campaign(args)
        stages = [s for s in campaign["execution_stages"] if s["stage"] >= 2 and s["seeds"] == request["seeds"]]
        if len(stages) != 1:
            raise ValueError("seeds must exactly match one declared later campaign stage")
        return args, inputs, stages[0]["stage"]

    def validate_context(self, *, request, environment):
        _, inputs, _ = self._campaign(request)
        campaign = inputs[3]
        if environment["image_id"] != campaign["runtime"]["image_id"]:
            raise ValueError("node image must equal the sealed campaign image")
        if (environment["environment_sha256"] != campaign["runtime"]["environment_sha256"]
                and not request.get("cross_environment", False)):
            raise ValueError("node probe/environment differs from campaign; declare cross_environment: true")
        return inputs

    def read_queue(self, queue_checkout):
        """Use one committed HEAD; GitQueue does not populate its working tree."""
        queue = GitQueue(queue_checkout, self.remote, self.branch)
        head = queue._fetch_head()
        if head is None:
            return [], []
        anchors = []
        for path in queue._git("ls-tree", "-r", "--name-only", head, "anchors/").stdout.splitlines():
            if path.endswith(".json"):
                anchor = validate_anchor(_json(queue.read_path(head, path)))
                if path != anchor_path(anchor["dataset"]["id"], anchor["kind"], anchor["document_sha256"]):
                    raise ValueError("queue anchor path differs from identity")
                anchors.append(anchor)
        return list(queue._records(head).values()), anchors

    @staticmethod
    def public_error(error):
        return cli._redact(str(error))

    def _request_path(self, request):
        return self.data_root / gj._private_name("seed-request-authorization", request["dataset"], digest(request))

    def _document(self, request):
        document = _json(self._request_path(request).read_text(encoding="utf-8"))
        if document.get("seed_request", {}).get("request") != request:
            raise ValueError("durable authorization request binding differs")
        return document

    def authorize(self, *, request, request_sha256, node_card, environment, stop_at, queue_checkout):
        if digest(request) != request_sha256:
            raise ValueError("request_sha256 differs from request")
        self.validate_context(request=request, environment=environment)
        args, inputs, stage = self._campaign(request)
        source, records, pools, campaign, freeze, evidence, presets = inputs
        commit = cli._clean_head(self.code_repo)
        dataset = [d for d in node_card["datasets"] if d["id"] == request["dataset"]
                   and d["version"] in (campaign["dataset"]["version"], "*")]
        if (len(dataset) != 1 or node_card["node_id"] != request["node_id"]
                or dataset[0]["manifest_sha256"] != source["manifest_sha256"]):
            raise ValueError("node dataset or identity differs from request")
        requirements = validate_requirements({"schema": "atlas.requirements.v1", "gpu_required": True,
            "min_vram_mib": self.min_vram_mib, "image_id": campaign["runtime"]["image_id"],
            "dataset": {**dataset[0], "version": campaign["dataset"]["version"]},
            "hardware_class": node_card["hardware_class"], "allowed_nodes": [request["node_id"]]})
        binding = {"request": deepcopy(request), "environment": deepcopy(environment),
                   "requirements": requirements, "stop_at": stop_at}
        queue = StageGatedQueue(queue_checkout, self.remote, self.branch)
        saved = self._request_path(request)
        if saved.exists():
            document = self._document(request)
            sa.audit(document, campaign, code_repo=self.code_repo)
            if document["seed_request"] != binding or document["execution_code_commit"] != commit:
                raise ValueError("durable stage environment/request/commit changed; do not reauthorize")
        else:
            # R1-02: do not even seal while primary workers (older orchestrators) are active.
            refuse_active_primary(queue, queue._fetch_head())
            # Provenance comes from the request itself (no default); X5 refuses true upstream.
            document = sa.authorize_stage(campaign, stage=stage, execution_code_commit=commit,
                code_repo=self.code_repo, image_id=environment["image_id"],
                environment_sha256=environment["environment_sha256"],
                cross_environment=request.get("cross_environment", False), seed_request=binding,
                decided_on=request["requested_on"], decided_after_primary_results=request["decided_after_results"],
                rationale=request.get("note", "Additional sensitivity seeds within the declared time/budget."),
                sealed_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            # Persist before queue mutation; a lost anchor response reuses this exact seal.
            gd._publish(saved, document, lambda value: sa.audit(value, campaign, code_repo=self.code_repo))
        private = self.data_root / gj._private_name("stage-authorization", request["dataset"], document["sha256"])
        gd._publish(private, document, lambda value: sa.audit(value, campaign, code_repo=self.code_repo))
        sa.materialize(campaign, stage, self.code_repo, authorization=document)
        head = queue._fetch_head()
        anchored = queue.read_path(head, anchor_path(request["dataset"], "stage_authorization", document["sha256"]))
        gc.audit_chronology(queue, campaign, freeze=freeze, head=head,
                            stage_authorization=document if anchored is not None else None, code_repo=self.code_repo)
        receipt = queue.publish_anchor(gc.build_anchor("stage_authorization", document))
        expected = col.expected_final_jobs(campaign, records, pools, freeze=freeze, dataset_source=source,
            augmentation_presets=presets, evidence=evidence, stage_authorization=document,
            code_repo=self.code_repo, scope="stage")
        from .fleet import with_requirements
        specs = [with_requirements(item[5], requirements) for item in expected.values()]
        return AuthorizedStage(document["sha256"], receipt["commit"], document["runtime"]["image_id"], specs, "git")

    def seed_final(self, *, request, specs, queue_checkout):
        _, inputs, _ = self._campaign(request)
        pools, campaign = inputs[2], inputs[3]
        document = self._document(request)
        sa.audit(document, campaign, code_repo=self.code_repo)
        queue = GitQueue(queue_checkout, self.remote, self.branch)
        if pools["grouping"]["prior_exposure"] != "none_documented":
            sources = {a["recipe_source"] for a in cli._anchors(queue, request["dataset"])
                       if a["kind"] == "campaign" and a["pools_sha256"] == pools["sha256"]}
            if not {"external_fixed", "hpo_d"} <= sources:
                gc._fail("final_job_before_seal", "exposed dataset: anchor both campaigns first")
        return queue.seed_final(
            specs, campaign["sha256"], stage_authorization=document, campaign=campaign, code_repo=self.code_repo)
