"""R1 review fixes (real disposable Git repositories, no training).

R1-01 seed eligibility from the audited campaign/authorization (never the path),
R1-02 stage-authorization publication gated on finished primary workers,
provenance of decided_after_primary_results, re-signed changed pools, and the
closed anchor schema with the sealed optional X6 keys.
"""
import contextlib
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from augment_atlas import global_campaign as gc, global_collect as col, runtime, seed_requests as sr
from augment_atlas import stage_authorization as sa, stage_gates as gates
from augment_atlas.contracts import canonical_hash
from augment_atlas.queue import QueueConflict, anchor_path, render_anchor
from augment_atlas.schema import new_record, prepare_job
import test_global_cli as fixtures
import test_seed_stage_authorizer as seedauth
import test_stage_authorization as stagefixture
from test_global_campaign import resign

def ST():
    """Shared C9b fixture class (not bound at module level, so its tests are not collected here)."""
    return stagefixture.StageAuthorizationTests


class R1Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ST().setUpClass()
        cls.addClassCleanup(ST().tearDownClass)
        for key in ("env", "campaign", "freeze", "pools", "auth", "execution", "original", "primary_jobs",
                    "stage_jobs", "common"):
            setattr(cls, key, getattr(ST(), key))

    def setUp(self):
        self.fixture = ST()("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.queue = self.fixture.queue
        self.tmp = Path(self.fixture.tmp.name)
        self.sha = self.campaign["sha256"]

    def primary(self, arm="none", fold=0):
        return next(v[5] for v in self.primary_jobs.values() if v[:3] == (arm, fold, 11))

    def legacy(self, seed):
        """A later-stage seed with the campaign commit and a legacy-shaped (stage-1) config path."""
        spec = copy.deepcopy(self.primary())
        spec.pop("job_id")
        spec["train_seed"] = seed
        spec["config"] = self.campaign["configs"]["root"] + f"/none-f0-s{seed}.json"
        return prepare_job(spec)

    def seed_primary(self):
        return self.queue.seed_final([self.primary()], self.sha, campaign=self.campaign, code_repo=self.env.code)[0]

    def push(self, records, message):
        head = self.queue._fetch_head()
        self.assertTrue(self.queue._push(self.queue._commit(head, records, message)))


class SeedEligibilityTests(R1Case):
    """R1-01: seeds 13/19 with a legacy-shaped path and no authorization."""

    def test_legacy_path_stage_seeds_refused_at_seed_final(self):
        for seed in (13, 19):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "^campaign_invalid: seed is outside"):
                self.queue.seed_final([self.legacy(seed)], self.sha, campaign=self.campaign, code_repo=self.env.code)
        self.assertEqual(self.queue.list_jobs(), [])
        # A re-signed campaign that widens its own seeds is not the campaign named by the anchor.
        widened = copy.deepcopy(self.campaign)
        widened["matrix"]["authorized_seeds"] = [11, 13, 19]
        with self.assertRaisesRegex(ValueError, "^campaign_invalid:"):
            self.queue.seed_final([self.legacy(13)], self.sha, campaign=resign(widened), code_repo=self.env.code)
        self.assertEqual(self.queue.list_jobs(), [])
        # The legitimate stage-1 admission with the campaign is unchanged (same immutable spec and ID).
        job = self.seed_primary()
        self.assertEqual(job["spec"], self.primary())

    def test_legacy_path_stage_seeds_refused_at_claim(self):
        # Anchor-only queue admission (no campaign) stays as before; the audited claim decides by seeds.
        jobs = self.queue.seed_final([self.legacy(13), self.legacy(19)], self.sha)
        for job in jobs:
            with self.subTest(seed=job["spec"]["train_seed"]), \
                    self.assertRaisesRegex(ValueError, "^campaign_invalid: seed is outside"):
                self.queue.claim(job["job_id"], {"server": "synthetic"},
                                 stage_context=(self.campaign, None, self.env.code))
        self.assertEqual({job["state"] for job in self.queue.list_jobs()}, {"pending"})

    def test_primary_claim_with_campaign_context_still_succeeds(self):
        job = self.seed_primary()
        claimed = self.queue.claim(job["job_id"], {"server": "synthetic"},
                                   stage_context=(self.campaign, None, self.env.code))
        self.assertEqual(claimed["state"], "claimed")

    def host(self, **extra):
        settings = dict(code=str(self.env.code), data=str(self.env.data), phase="final", gpu="0",
                        campaign_sha256=self.sha, campaign_commit=self.fixture.c_campaign,
                        remote=str(self.queue.remote), queue_checkout=str(self.tmp / "worker"), **extra)
        host = runtime.DockerRuntime(settings)
        host.commit, host.image, host.environment = self.original, fixtures.IMAGE, json.loads(self.env.probe.read_bytes())
        return host

    def test_runtime_claim_context_carries_campaign_without_stage_pin(self):
        job = self.seed_primary()
        host = self.host()
        with patch.object(host, "weights_cache_for", return_value=None):
            host.global_config_for(self.primary())
        campaign, authorization, code = host._stage_claim_context
        self.assertEqual((campaign["sha256"], authorization, code), (self.sha, None, host.code))
        claimed = self.queue.claim(job["job_id"], {"server": "synthetic"}, stage_context=host._stage_claim_context)
        self.assertEqual(claimed["state"], "claimed")
        with patch.object(host, "weights_cache_for", return_value=None), self.assertRaises(ValueError):
            host.global_config_for(self.legacy(13))

    def test_worker_resume_revalidates_legacy_shaped_seed_without_pin(self):
        runs = self.tmp / "runs"
        slot = runs / "worker-synthetic-gpu"
        slot.mkdir(parents=True)
        spec = self.legacy(13)
        self.assertNotIn("/stage-", spec["config"])
        (slot / "active.json").write_text(json.dumps({
            "record": {"spec": spec}, "output": str(runs / "attempt"), "container": "synthetic-container",
            "image": fixtures.IMAGE, "stage": "resuming"}), encoding="utf-8")
        host = Mock()
        host.commit, host.image, host.gpu_uuid = self.original, fixtures.IMAGE, "synthetic-gpu"
        host.config_for.side_effect = ValueError("campaign_invalid: seed is outside the authorized stage")
        settings = dict(runs=str(runs), queue_checkout=str(self.tmp / "worker"),
                        remote=str(self.queue.remote), server="synthetic", gpu="0")
        with patch("augment_atlas.runtime.DockerRuntime", return_value=host), \
             patch("augment_atlas.runtime.gpu_lock", return_value=contextlib.nullcontext()), \
             patch("augment_atlas.runtime.signal.signal"):
            with self.assertRaisesRegex(ValueError, "^campaign_invalid:"):
                runtime.worker(settings, once=True)
        host.config_for.assert_called_once_with(spec)
        host.inspect.assert_not_called()


class PublicationGateTests(R1Case):
    """R1-02: older workers reject the stage_authorization anchor kind; publish only after primaries finish."""

    def gated(self):
        return gates.StageGatedQueue(self.queue.checkout, self.queue.remote)

    def stage_path(self, document=None):
        document = document or self.auth
        return anchor_path(document["dataset"]["id"], "stage_authorization", document["sha256"])

    def assertBlocked(self, job_ids, document=None):
        with self.assertRaisesRegex(QueueConflict, "^primary_workers_active:") as caught:
            self.gated().publish_anchor(gc.build_anchor("stage_authorization", document or self.auth))
        for job_id in job_ids:
            self.assertIn(job_id, str(caught.exception))
        self.assertIsNone(self.queue.read_path(self.queue._fetch_head(), self.stage_path(document)))

    def fake_authorization(self, dataset_id, label):
        """Another dataset's sealed campaign anchor plus a re-signed authorization bound to it."""
        campaign = {"sha256": canonical_hash(label), "pools_sha256": canonical_hash([label, "pools"]),
                    "freeze": {"sha256": canonical_hash([label, "freeze"]), "kind": "external_fixed"},
                    "runtime": {"code_commit": "a" * 40}, "label": "exploratory",
                    "sealed_at_utc": "2026-10-23T15:00:00Z", "dataset": {"id": dataset_id, "version": "v1"}}
        anchor = gc.build_anchor("campaign", campaign)
        head = self.queue._fetch_head()
        commit = self.queue._commit_files(head, {anchor_path(dataset_id, "campaign", campaign["sha256"]):
                                                 render_anchor(anchor)}, "synthetic other campaign anchor")
        self.assertTrue(self.queue._push(commit))
        document = copy.deepcopy(self.auth)
        document.update(campaign_sha256=campaign["sha256"], pools_sha256=campaign["pools_sha256"],
                        dataset=copy.deepcopy(campaign["dataset"]))
        document["freeze"]["sha256"] = campaign["freeze"]["sha256"]
        return resign(document)

    def test_every_non_terminal_primary_state_blocks_until_terminal(self):
        job = self.seed_primary()
        job_id = job["job_id"]
        self.assertBlocked([job_id])  # pending
        claimed = self.queue.claim(job_id, {"server": "synthetic"})
        self.assertBlocked([job_id])  # claimed
        running = self.queue.update(job_id, claimed["attempt_id"], claimed["token"], claimed["version"],
                                    {"event_id": "run", "state": "running"})
        self.assertBlocked([job_id])  # running
        stopped = self.queue.update(job_id, running["attempt_id"], running["token"], running["version"],
                                    {"event_id": "stop", "state": "interrupted"})
        self.assertBlocked([job_id])  # interrupted
        resumed = self.queue.update(job_id, stopped["attempt_id"], stopped["token"], stopped["version"],
                                    {"event_id": "resume", "state": "running", "manual_resume": True})
        self.queue.update(job_id, resumed["attempt_id"], resumed["token"], resumed["version"],
                          {"event_id": "fail", "state": "failed"})
        published = self.gated().publish_anchor(gc.build_anchor("stage_authorization", self.auth))
        self.assertEqual(published["path"], self.stage_path())
        again = self.gated().publish_anchor(gc.build_anchor("stage_authorization", self.auth))
        self.assertEqual(again["commit"], published["commit"])  # idempotent: no new publication

    def test_primary_job_of_another_dataset_and_campaign_blocks(self):
        other = copy.deepcopy(self.primary())
        other.pop("job_id")
        other["dataset"] = "synthetic_other"
        other["global"] = {**other["global"], "campaign_sha256": canonical_hash("another campaign")}
        other = prepare_job(other)
        self.push([new_record(other)], "synthetic primary job of another dataset")
        self.assertBlocked([other["job_id"]])

    def test_stage2_jobs_of_an_authorized_campaign_do_not_block_another_campaign(self):
        self.gated().publish_anchor(gc.build_anchor("stage_authorization", self.auth))
        self.queue.seed_final([v[5] for v in self.stage_jobs.values()], self.sha,
                              stage_authorization=self.auth, campaign=self.campaign, code_repo=self.env.code)
        records = self.queue.list_jobs()
        self.assertEqual((len(records), {r["state"] for r in records}), (100, {"pending"}))
        self.assertEqual(gates.active_primary_jobs(self.queue, self.queue._fetch_head()), [])
        second = self.fake_authorization("synthetic_other", "campaign B")
        published = self.gated().publish_anchor(gc.build_anchor("stage_authorization", second))
        self.assertEqual(published["path"], self.stage_path(second))
        # A still-pending primary job of the first campaign blocks any further publication.
        job = self.seed_primary()
        third = self.fake_authorization("synthetic_third", "campaign C")
        self.assertBlocked([job["job_id"]], third)

    def test_cli_anchor_refuses_and_lists_active_primary_jobs(self):
        job = self.seed_primary()
        argv = self.env.base("anchor") + ["--code", str(self.env.code), "--remote", str(self.queue.remote),
                                          "--checkout", str(self.queue.checkout), "--kind", "stage_authorization",
                                          "--document-sha256", self.auth["sha256"], "--image-id", fixtures.IMAGE,
                                          "--environment-probe", str(self.env.probe)]
        error = ST().tc.run_cli(argv, code=2)
        self.assertTrue(error.startswith("primary_workers_active:"), error)
        self.assertIn(job["job_id"], error)
        self.assertIsNone(self.queue.read_path(self.queue._fetch_head(), self.stage_path()))


class AdapterTests(unittest.TestCase):
    """RealStageAuthorizer: publication gate and decision provenance."""

    @classmethod
    def setUpClass(cls):
        ST().setUpClass()
        cls.addClassCleanup(ST().tearDownClass)
        for key in ("env", "campaign", "freeze", "pools"):
            setattr(seedauth.RealAuthorizerTests, key, getattr(ST(), key))

    def setUp(self):
        self.case = seedauth.RealAuthorizerTests("runTest")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.queue = self.case.fixture.queue

    def test_refuses_to_seal_or_anchor_while_a_primary_job_is_active(self):
        primary = next(v[5] for v in ST().primary_jobs.values() if v[:3] == ("none", 0, 11))
        job = self.queue.seed_final([primary], ST().campaign["sha256"], campaign=ST().campaign, code_repo=ST().env.code)[0]
        row = self.case.process()
        self.assertEqual(row["status"], "invalid", row)
        self.assertIn("primary_workers_active", row["error"])
        self.assertIn(job["job_id"], row["error"])
        self.assertFalse(self.case.adapter._request_path(self.case.request).exists())  # nothing sealed
        self.assertEqual([j["job_id"] for j in self.queue.list_jobs()], [job["job_id"]])
        claimed = self.queue.claim(job["job_id"], {"server": "synthetic"})
        running = self.queue.update(job["job_id"], claimed["attempt_id"], claimed["token"], claimed["version"],
                                    {"event_id": "run", "state": "running"})
        self.queue.update(job["job_id"], running["attempt_id"], running["token"], running["version"],
                          {"event_id": "end", "state": "succeeded"})
        row = self.case.process(adapter=self.case.make_adapter())
        self.assertEqual(row["status"], "authorized", row)
        self.assertEqual(len(self.queue.list_jobs()), 101)

    def test_decision_provenance_comes_from_the_request_without_default(self):
        missing = {k: v for k, v in self.case.request.items() if k != "decided_after_results"}
        with self.assertRaisesRegex(ValueError, "decided_after_results"):
            self.case.adapter._campaign(missing)
        with self.assertRaisesRegex(ValueError, "SAP item 5"):
            sr.validate_request({**self.case.request, "decided_after_results": True})
        # Bypass only X5's upstream refusal: the adapter must seal the request's own value, not a constant.
        permissive = lambda value: value
        self.case.request["decided_after_results"] = True
        with patch("augment_atlas.seed_requests.validate_request", side_effect=permissive), \
             patch("augment_atlas.seed_stage_authorizer.validate_request", side_effect=permissive):
            row = self.case.process()
        self.assertEqual(row["status"], "authorized", row)
        document = self.case.document()
        self.assertIs(document["decided_after_primary_results"], True)
        self.assertIs(document["seed_request"]["request"]["decided_after_results"], True)


class ResignedPoolsAndAnchorSchemaTests(R1Case):
    def test_resigned_changed_pools_are_refused_everywhere(self):
        changed = copy.deepcopy(self.pools)
        changed["source_sha256"] = canonical_hash("another synthetic manifest")
        changed = resign(changed)
        self.assertNotEqual(changed["sha256"], self.pools["sha256"])
        bound = resign(dict(copy.deepcopy(self.auth), pools_sha256=changed["sha256"]))
        with self.assertRaisesRegex(ValueError, "^campaign_invalid:"):
            sa.audit(bound, self.campaign, code_repo=self.env.code)
        with self.assertRaisesRegex(QueueConflict, "^anchor_conflict:"):
            gates.StageGatedQueue(self.queue.checkout, self.queue.remote).publish_anchor(
                gc.build_anchor("stage_authorization", bound))
        with self.assertRaises(ValueError):
            self.queue.seed_final([v[5] for v in self.stage_jobs.values()], self.sha, stage_authorization=bound,
                                  campaign=self.campaign, code_repo=self.env.code)
        with self.assertRaises(ValueError):
            col.expected_final_jobs(self.campaign, self.env.records, changed, **self.common)
        self.assertEqual(self.queue.list_jobs(), [])

    def test_anchor_accepts_only_sealed_optional_stage_keys(self):
        crossed = sa.authorize_stage(self.campaign, **dict(ST().auth_args, cross_environment=True,
                                                           environment_sha256=canonical_hash("other probe")))
        anchor = gc.build_anchor("stage_authorization", crossed)
        self.assertEqual(anchor["document_sha256"], crossed["sha256"])
        with self.assertRaisesRegex(ValueError, "^anchor_invalid:"):
            gc.build_anchor("stage_authorization", resign({**copy.deepcopy(self.auth), "unknown": True}))
        # A re-signed explicit false is not what authorize_stage seals: the auditor rejects it.
        with self.assertRaisesRegex(ValueError, "^campaign_invalid:"):
            sa.audit(resign({**copy.deepcopy(self.auth), "cross_environment": False}), self.campaign,
                     code_repo=self.env.code)


if __name__ == "__main__":
    unittest.main()
