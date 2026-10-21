"""Slice S3 anchors on the queue branch and chronology audit (real local Git, bare remote)."""
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest

from augment_atlas import global_campaign as gc
from augment_atlas import global_jobs as gj
from augment_atlas.contracts import canonical_hash
from augment_atlas.queue import GitQueue, QueueConflict, anchor_path, render_anchor, validate_anchor
from augment_atlas.schema import new_record
from test_global_campaign import CODE, final_spec, seal
from test_global_development import H
from test_global_jobs import DATASET_SOURCE, external_freeze, hpo_fixture


def hpo_spec(rows, pools, plan, *, candidate=None, seed=None):
    configs = gj.build_hpo_configs(rows, pools, expected_pools_sha256=pools["sha256"], dataset_source=DATASET_SOURCE,
                                   plan=plan, expected_plan_sha256=H(plan), search_space_sha256=H("search space"),
                                   hpo_view_sha256=H("hpo view"))
    names = sorted(configs)
    name = names[0] if candidate is None else f"{candidate}-s{seed}.json"
    cfg = configs[name]
    return gj.hpo_job_spec(cfg, config_path=f"{gj.config_dir(pools['dataset']['id'], 'hpo')}/{name}",
                           config_sha256=gj._bytes_sha256(gj.render_config_bytes(cfg)), code_commit=CODE)


class ChronologyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="augment-atlas-chronology-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.remote = root / "remote.git"
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True, capture_output=True)
        self.q = GitQueue(root / "host", self.remote)
        self.other = GitQueue(root / "auditor", self.remote)
        self.rows, self.pools, self.freeze = external_freeze()
        self.campaign = seal(self.rows, self.pools, self.freeze)

    def finals(self, campaign=None, rows=None, pools=None, freeze=None, kind="external_fixed", **extra):
        campaign = campaign or self.campaign
        rows, pools, freeze = rows or self.rows, pools or self.pools, freeze or self.freeze
        return [final_spec(rows, pools, freeze, campaign["sha256"], arm=arm, fold=fold, kind=kind, **extra)
                for arm, fold in (("none", 0), ("none", 1))]

    def anchor(self, kind, document):
        return gc.build_anchor(kind, document)

    def fabricate(self, files=None, records=None, message="fabricated"):
        """Commit arbitrary content on the remote HEAD (the fixture plays the repository owner)."""
        head = self.q._fetch_head()
        if records:
            commit = self.q._commit(head, records, message)
        else:
            commit = self.q._commit_files(head, files, message)
        self.assertTrue(self.q._push(commit))
        return commit

    def happy_path(self):
        freeze_anchor = self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        campaign_anchor = self.q.publish_anchor(self.anchor("campaign", self.campaign))
        jobs = self.q.seed_final(self.finals(), self.campaign["sha256"])
        return freeze_anchor, campaign_anchor, jobs


class AnchorAndSeedFinalTests(ChronologyTest):
    def test_order_freeze_campaign_final_is_audited(self):
        freeze_anchor, campaign_anchor, jobs = self.happy_path()
        self.assertEqual(len(jobs), 2)
        again = self.q.publish_anchor(self.anchor("campaign", self.campaign))
        self.assertEqual(again["commit"], campaign_anchor["commit"])
        claimed = self.q.claim(jobs[0]["job_id"], {"server": "fixture"})
        self.assertIsNotNone(claimed)
        report = gc.audit_chronology(self.other, self.campaign, freeze=self.freeze,
                                     recorded_campaign_commit=campaign_anchor["commit"])
        self.assertEqual((report["C_freeze"], report["C_campaign"]), (freeze_anchor["commit"], campaign_anchor["commit"]))
        self.assertEqual((report["final_jobs"], report["hpo_jobs"]), (2, 0))
        self.assertIsNotNone(report["first_final_claim_utc"])
        self.assertEqual((report["execution_verified"], report["scientific_eligible"]), (False, False))
        self.assertIn("Bradshaw 2023", report["caveat"])
        # Job commit is a child of a commit holding the campaign anchor.
        parent = self.q._git("rev-parse", f"{self.q.anchor_commit(report['head'], 'jobs/' + jobs[0]['job_id'] + '.md')}^").stdout.strip()
        self.assertIsNotNone(self.q.read_path(parent, anchor_path("synthetic_fixture", "campaign", self.campaign["sha256"])))
        with self.assertRaisesRegex(ValueError, "^chronology_invalid:"):
            gc.audit_chronology(self.other, self.campaign, recorded_campaign_commit=freeze_anchor["commit"])

    def test_anchor_files_have_hashes_only(self):
        self.happy_path()
        head = self.q._fetch_head()
        for path in self.q.list_anchors(head, "synthetic_fixture"):
            text = self.q.read_path(head, path)
            self.assertFalse([row["id"] for row in self.rows if row["id"] in text])
            self.assertNotIn("manifest", text)

    def test_seed_final_without_campaign_anchor_is_refused(self):
        with self.assertRaisesRegex(QueueConflict, "^campaign_anchor_missing:"):
            self.q.seed_final(self.finals(), self.campaign["sha256"])
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        with self.assertRaisesRegex(QueueConflict, "^campaign_anchor_missing:"):
            self.q.seed_final(self.finals(), self.campaign["sha256"])
        self.assertEqual(self.q.list_jobs(), [])
        with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
            self.q.seed(self.finals())  # legacy seed never takes global final jobs
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
            self.q.seed_final(self.finals(campaign={"sha256": H("other campaign")}), self.campaign["sha256"])
        self.assertEqual(len(self.q.seed_final(self.finals(), self.campaign["sha256"])), 2)
        self.assertEqual(len(self.q.seed_final(self.finals(), self.campaign["sha256"])), 2)  # idempotent

    def test_campaign_anchor_requires_freeze_anchor(self):
        with self.assertRaisesRegex(QueueConflict, "^freeze_anchor_missing:"):
            self.q.publish_anchor(self.anchor("campaign", self.campaign))

    def test_reseal_with_other_seeds_or_recipe_is_anchor_conflict(self):
        self.happy_path()
        reseeded = seal(self.rows, self.pools, self.freeze, execution_stages=[{"stage": 1, "seeds": [11, 13, 19]}])
        with self.assertRaisesRegex(QueueConflict, "^anchor_conflict:"):
            self.q.publish_anchor(self.anchor("campaign", reseeded))
        recipe = copy.deepcopy(self.freeze["recipe"])
        recipe["optimizer"]["parameters"]["lr"] = 0.001
        other = gc.seal_external_freeze(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"],
                                        recipe=recipe, source=gj.Q2_EXTERNAL_SOURCE, runtime=self.freeze["runtime"],
                                        final_policy=self.freeze["final_policy"],
                                        frozen_at_utc=self.freeze["frozen_at_utc"])
        with self.assertRaisesRegex(QueueConflict, "^anchor_conflict:"):
            self.q.publish_anchor(self.anchor("recipe_freeze", other))

    def test_same_path_with_other_bytes_is_anchor_conflict(self):
        anchor = self.anchor("recipe_freeze", self.freeze)
        path = anchor_path("synthetic_fixture", "recipe_freeze", self.freeze["sha256"])
        self.fabricate({path: render_anchor(anchor).replace("\n", "\n ", 1)})
        with self.assertRaisesRegex(QueueConflict, "^anchor_conflict:"):
            self.q.publish_anchor(anchor)

    def test_sealing_with_existing_final_job_is_refused(self):
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.fabricate(records=[new_record(spec) for spec in self.finals()])
        with self.assertRaisesRegex(ValueError, "^final_job_before_seal:"):
            seal(self.rows, self.pools, self.freeze, queue_records=self.q.list_jobs())
        with self.assertRaisesRegex(QueueConflict, "^final_job_before_seal:"):
            self.q.publish_anchor(self.anchor("campaign", self.campaign))


class FabricatedHistoryTests(ChronologyTest):
    def test_final_job_before_campaign_anchor_in_fabricated_history(self):
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.fabricate(records=[new_record(spec) for spec in self.finals()])
        campaign = self.anchor("campaign", self.campaign)
        self.fabricate({anchor_path("synthetic_fixture", "campaign", self.campaign["sha256"]): render_anchor(campaign)})
        with self.assertRaisesRegex(ValueError, "^final_job_before_seal:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_final_job_in_the_campaign_anchor_commit_is_before_seal(self):
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        head = self.q._fetch_head()
        with_jobs = self.q._commit(head, [new_record(spec) for spec in self.finals()], "jobs")
        path = anchor_path("synthetic_fixture", "campaign", self.campaign["sha256"])
        both = self.q._commit_files(with_jobs, {path: render_anchor(self.anchor("campaign", self.campaign))}, "x")
        squashed = self.q._git("commit-tree", f"{both}^{{tree}}", "-p", head, input="squashed\n").stdout.strip()
        self.assertTrue(self.q._push(squashed))
        with self.assertRaisesRegex(ValueError, "^final_job_before_seal:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_missing_anchors_and_inverted_anchor_order(self):
        with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
            gc.audit_chronology(self.other, self.campaign)
        path = anchor_path("synthetic_fixture", "campaign", self.campaign["sha256"])
        self.fabricate({path: render_anchor(self.anchor("campaign", self.campaign))})
        with self.assertRaisesRegex(ValueError, "^freeze_anchor_missing:"):
            gc.audit_chronology(self.other, self.campaign)
        self.fabricate({anchor_path("synthetic_fixture", "recipe_freeze", self.freeze["sha256"]):
                        render_anchor(self.anchor("recipe_freeze", self.freeze))})
        with self.assertRaisesRegex(ValueError, "^chronology_invalid:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_anchor_changed_or_removed_after_creation(self):
        self.happy_path()
        path = anchor_path("synthetic_fixture", "recipe_freeze", self.freeze["sha256"])
        anchor = render_anchor(self.anchor("recipe_freeze", self.freeze))
        self.fabricate({path: anchor.replace("\n", "\n ", 1)})
        with self.assertRaisesRegex(ValueError, "^anchor_conflict:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_merge_commit_is_not_a_linear_history(self):
        self.happy_path()
        head = self.q._fetch_head()
        side = self.q._commit_files(head, {"notes/side.txt": "side\n"}, "side")
        merge = self.q._git("commit-tree", f"{head}^{{tree}}", "-p", head, "-p", side, input="merge\n").stdout.strip()
        self.assertTrue(self.q._push(merge))
        with self.assertRaisesRegex(ValueError, "^chronology_invalid:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_timestamp_inversion_is_a_warning_not_an_error(self):
        late = seal(self.rows, self.pools, self.freeze, sealed_at_utc="2999-01-01T00:00:00Z")
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.q.publish_anchor(self.anchor("campaign", late))
        report = gc.audit_chronology(self.other, late, freeze=self.freeze)
        self.assertTrue(any("campaign.sealed_at_utc > C_campaign.commit_time" in w for w in report["warnings"]))
        self.assertFalse(report["execution_verified"])

    def test_external_freeze_after_hpo_jobs_is_contaminated(self):
        rows, pools, plan, _, _ = hpo_fixture()
        self.assertEqual(pools["sha256"], self.pools["sha256"])
        self.q.seed([hpo_spec(rows, pools, plan)])
        with self.assertRaisesRegex(QueueConflict, "^external_recipe_contaminated:"):
            self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        # The fixture plays the owner and anchors the external freeze anyway.
        self.fabricate({anchor_path("synthetic_fixture", "recipe_freeze", self.freeze["sha256"]):
                        render_anchor(self.anchor("recipe_freeze", self.freeze))})
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        self.q.seed_final(self.finals(), self.campaign["sha256"])
        with self.assertRaisesRegex(ValueError, "^external_recipe_contaminated:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_hpo_job_in_the_external_freeze_commit_is_contaminated(self):
        rows, pools, plan, _, _ = hpo_fixture()
        head = self.q._fetch_head()
        with_job = self.q._commit(head, [new_record(hpo_spec(rows, pools, plan))], "hpo")
        both = self.q._commit_files(with_job, {anchor_path("synthetic_fixture", "recipe_freeze", self.freeze["sha256"]):
                                               render_anchor(self.anchor("recipe_freeze", self.freeze))}, "x")
        self.assertTrue(self.q._push(both))
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        with self.assertRaisesRegex(ValueError, "^external_recipe_contaminated:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_rewritten_job_spec_is_not_auditable(self):
        self.happy_path()
        record = copy.deepcopy(self.q.list_jobs()[0])
        record["spec"]["global"]["campaign_sha256"] = H("rebound campaign")
        self.fabricate(records=[record])
        with self.assertRaisesRegex(ValueError, "^chronology_invalid:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_malformed_job_record_is_not_auditable(self):
        self.happy_path()
        self.fabricate({f"jobs/{'a' * 64}.md": "not a structured job record\n"})
        with self.assertRaisesRegex(ValueError, "^chronology_invalid:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_job_record_removed_is_not_auditable(self):
        self.happy_path()
        head = self.q._fetch_head()
        job = self.q.list_jobs()[0]["job_id"]
        with tempfile.TemporaryDirectory() as directory:
            env = {"GIT_INDEX_FILE": str(Path(directory) / "index")}
            self.q._git("read-tree", head, env=env)
            self.q._git("update-index", "--force-remove", f"jobs/{job}.md", env=env)
            tree = self.q._git("write-tree", env=env).stdout.strip()
        removed = self.q._git("commit-tree", tree, "-p", head, input="drop job\n").stdout.strip()
        self.assertTrue(self.q._push(removed))
        with self.assertRaisesRegex(ValueError, "^chronology_invalid:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_final_seed_outside_the_authorized_stage(self):
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        late = [final_spec(self.rows, self.pools, self.freeze, self.campaign["sha256"], seed=13)]
        self.q.seed_final(late, self.campaign["sha256"])  # the anchor carries no seeds
        with self.assertRaisesRegex(ValueError, "^campaign_invalid:"):
            gc.audit_chronology(self.other, self.campaign)


class HpoChronologyTests(ChronologyTest):
    def setUp(self):
        super().setUp()
        self.rows, self.pools, self.plan, self.evidence, self.freeze = hpo_fixture()
        self.campaign = seal(self.rows, self.pools, self.freeze, kind="hpo_d", evidence=self.evidence,
                             search_space_sha256=H("search space"))
        self.hpo = [hpo_spec(self.rows, self.pools, self.plan, candidate=c["id"], seed=s)
                    for c in self.plan["candidates"] for s in self.plan["seeds"]]

    def finals(self, campaign=None, **extra):
        return super().finals(campaign=campaign, kind="hpo_d", evidence=self.evidence,
                              expected_plan_sha256=self.freeze["selection_plan_sha256"])

    def finish(self, job_id):
        claimed = self.q.claim(job_id, {"server": "fixture"})
        running = self.q.update(job_id, claimed["attempt_id"], claimed["token"], claimed["version"],
                                {"event_id": "run", "state": "running", "epoch": 0})
        return self.q.update(job_id, running["attempt_id"], running["token"], running["version"],
                             {"event_id": "done", "state": "succeeded"})

    def test_hpo_jobs_terminal_before_freeze_pass(self):
        for record in self.q.seed(self.hpo[:-1]):
            self.finish(record["job_id"])
        self.happy_path()
        report = gc.audit_chronology(self.other, self.campaign, freeze=self.freeze)
        self.assertEqual(report["hpo_jobs"], len(self.hpo) - 1)

    def test_hpo_job_created_after_freeze(self):
        for record in self.q.seed(self.hpo[:-1]):
            self.finish(record["job_id"])
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        with self.assertRaisesRegex(QueueConflict, "^hpo_job_after_freeze:"):
            self.q.seed(self.hpo[-1:])
        self.fabricate(records=[new_record(self.hpo[-1])])  # owner bypasses the seed guard
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        self.q.seed_final(self.finals(), self.campaign["sha256"])
        with self.assertRaisesRegex(ValueError, "^hpo_job_after_freeze:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_hpo_job_not_terminal_before_freeze(self):
        self.q.seed(self.hpo[:1])
        with self.assertRaisesRegex(QueueConflict, "^hpo_job_after_freeze:"):
            self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.fabricate({anchor_path("synthetic_fixture", "recipe_freeze", self.freeze["sha256"]):
                        render_anchor(self.anchor("recipe_freeze", self.freeze))})
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        self.q.seed_final(self.finals(), self.campaign["sha256"])
        with self.assertRaisesRegex(ValueError, "^hpo_job_after_freeze:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_hpo_job_of_another_plan_is_refused(self):
        rows, pools, plan, _, _ = hpo_fixture(hpo_seeds=(103,))  # a second search on the same pools
        for record in self.q.seed(self.hpo + [hpo_spec(rows, pools, plan)]):
            self.finish(record["job_id"])
        self.happy_path()
        with self.assertRaisesRegex(ValueError, "^search_space_mismatch:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_hpo_job_retried_after_freeze(self):
        record = self.q.seed(self.hpo[:1])[0]
        claimed = self.q.claim(record["job_id"], {"server": "fixture"})
        failed = self.q.update(record["job_id"], claimed["attempt_id"], claimed["token"], claimed["version"],
                               {"event_id": "fail", "state": "failed"})
        self.happy_path()
        self.q.retry(record["job_id"], failed["version"], "operator retry after freeze")
        with self.assertRaisesRegex(ValueError, "^hpo_job_after_freeze:"):
            gc.audit_chronology(self.other, self.campaign)


class SeedFinalBindingTests(ChronologyTest):
    def test_seed_final_binds_freeze_and_code_commit_of_the_anchor(self):
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        with self.assertRaisesRegex(QueueConflict, "^code_identity_mismatch:"):
            self.q.seed_final(self.finals(code_commit="f" * 40), self.campaign["sha256"])
        rows, pools, _, evidence, freeze = hpo_fixture()
        other_freeze = self.finals(rows=rows, pools=pools, freeze=freeze, kind="hpo_d", evidence=evidence,
                                   expected_plan_sha256=freeze["selection_plan_sha256"])
        with self.assertRaisesRegex(QueueConflict, "^campaign_anchor_missing:"):
            self.q.seed_final(other_freeze, self.campaign["sha256"])
        self.assertEqual(self.q.list_jobs(), [])

    def test_anchor_recipe_source_is_derived_and_closed(self):
        self.assertEqual(self.anchor("recipe_freeze", self.freeze)["recipe_source"], "external_fixed")
        self.assertEqual(self.anchor("campaign", self.campaign)["recipe_source"], "external_fixed")
        pools_anchor = gc.build_anchor("pools", self.pools, document_utc="2026-10-19T11:00:00Z")
        self.assertIsNone(pools_anchor["recipe_source"])
        _, _, _, _, freeze = hpo_fixture()
        self.assertEqual(self.anchor("recipe_freeze", freeze)["recipe_source"], "hpo_d")
        for kind, document, source in (("pools", self.pools, "hpo_d"), ("recipe_freeze", self.freeze, None),
                                       ("campaign", self.campaign, "other")):
            anchor = gc.build_anchor(kind, document, document_utc="2026-10-19T11:00:00Z")
            anchor["recipe_source"] = source
            anchor["sha256"] = canonical_hash({k: v for k, v in anchor.items() if k != "sha256"})
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "^anchor_invalid:"):
                validate_anchor(anchor)


class TwoCampaignTests(ChronologyTest):
    """INbreast decision 20/10/2026: external fixed recipe (Q2) AND exploratory HPO on D, same pools."""

    def setUp(self):
        super().setUp()
        rows, pools, self.plan, self.evidence, self.hpo_freeze = hpo_fixture()
        self.assertEqual(pools["sha256"], self.pools["sha256"])
        self.hpo_campaign = seal(rows, pools, self.hpo_freeze, kind="hpo_d", evidence=self.evidence,
                                 search_space_sha256=H("search space"))
        self.hpo = [hpo_spec(rows, pools, self.plan, candidate=c["id"], seed=s)
                    for c in self.plan["candidates"] for s in self.plan["seeds"]]

    def hpo_finals(self, campaign=None, **extra):
        return self.finals(campaign=campaign or self.hpo_campaign, freeze=self.hpo_freeze, kind="hpo_d",
                           evidence=self.evidence, expected_plan_sha256=self.hpo_freeze["selection_plan_sha256"])

    def run_hpo(self):
        for record in self.q.seed(self.hpo):
            HpoChronologyTests.finish(self, record["job_id"])

    def test_both_campaigns_coexist_and_are_audited_separately(self):
        external = self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))  # before any HPO score
        self.run_hpo()
        tuned = self.q.publish_anchor(self.anchor("recipe_freeze", self.hpo_freeze))
        self.assertNotEqual(external["path"], tuned["path"])
        a = self.q.publish_anchor(self.anchor("campaign", self.campaign))
        b = self.q.publish_anchor(self.anchor("campaign", self.hpo_campaign))
        jobs_a = self.q.seed_final(self.finals(), self.campaign["sha256"])
        jobs_b = self.q.seed_final(self.hpo_finals(), self.hpo_campaign["sha256"])
        self.assertFalse({j["job_id"] for j in jobs_a} & {j["job_id"] for j in jobs_b})
        report_a = gc.audit_chronology(self.other, self.campaign, freeze=self.freeze,
                                       recorded_campaign_commit=a["commit"])
        report_b = gc.audit_chronology(self.other, self.hpo_campaign, freeze=self.hpo_freeze,
                                       recorded_campaign_commit=b["commit"])
        self.assertEqual((report_a["recipe_source"], report_b["recipe_source"]), ("external_fixed", "hpo_d"))
        self.assertEqual((report_a["final_jobs"], report_a["other_campaign_final_jobs"]), (2, 2))
        self.assertEqual((report_b["final_jobs"], report_b["other_campaign_final_jobs"]), (2, 2))
        self.assertEqual((report_a["hpo_jobs"], report_b["hpo_jobs"]), (len(self.hpo), len(self.hpo)))
        self.assertEqual(report_a["C_freeze"], external["commit"])
        self.assertEqual(report_b["C_freeze"], tuned["commit"])
        self.assertNotEqual(report_a["campaign_sha256"], report_b["campaign_sha256"])
        for report in (report_a, report_b):
            self.assertEqual((report["execution_verified"], report["scientific_eligible"]), (False, False))
        self.assertEqual((self.campaign["label"], self.hpo_campaign["label"]), ("exploratory", "exploratory"))
        self.assertEqual(len(self.q.list_anchors(self.q._fetch_head(), "synthetic_fixture")), 4)
        # One campaign's jobs never pass under the other campaign's anchor.
        with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
            self.q.seed_final(self.hpo_finals(), self.campaign["sha256"])
        with self.assertRaisesRegex(QueueConflict, "^campaign_anchor_missing:"):  # HPO-freeze jobs under campaign A
            self.q.seed_final(self.hpo_finals(campaign=self.campaign), self.campaign["sha256"])

    def test_second_campaign_after_the_first_campaign_finals_is_refused(self):
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.run_hpo()
        self.q.publish_anchor(self.anchor("recipe_freeze", self.hpo_freeze))
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        self.q.seed_final(self.finals(), self.campaign["sha256"])
        with self.assertRaisesRegex(QueueConflict, "^final_job_before_seal:"):
            self.q.publish_anchor(self.anchor("campaign", self.hpo_campaign))
        with self.assertRaisesRegex(ValueError, "^final_job_before_seal:"):
            seal(*hpo_fixture()[:2], self.hpo_freeze, kind="hpo_d", evidence=self.evidence,
                 search_space_sha256=H("search space"), queue_records=self.q.list_jobs())
        # Fabricated: the owner anchors the HPO campaign anyway, after campaign A's finals.
        self.fabricate({anchor_path("synthetic_fixture", "campaign", self.hpo_campaign["sha256"]):
                        render_anchor(self.anchor("campaign", self.hpo_campaign))})
        self.q.seed_final(self.hpo_finals(), self.hpo_campaign["sha256"])
        with self.assertRaisesRegex(ValueError, "^final_job_before_seal:"):
            gc.audit_chronology(self.other, self.hpo_campaign)
        gc.audit_chronology(self.other, self.campaign)  # campaign A itself stays auditable

    def test_one_document_per_recipe_source_slot(self):
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.run_hpo()
        self.q.publish_anchor(self.anchor("recipe_freeze", self.hpo_freeze))
        _, _, _, _, retuned = hpo_fixture(hpo_seeds=(103,))
        self.assertNotEqual(retuned["sha256"], self.hpo_freeze["sha256"])
        with self.assertRaisesRegex(QueueConflict, "^anchor_conflict:"):
            self.q.publish_anchor(self.anchor("recipe_freeze", retuned))
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        self.q.publish_anchor(self.anchor("campaign", self.hpo_campaign))
        # Fabricated second hpo_d freeze in the slot: every campaign audit refuses the history.
        self.fabricate({anchor_path("synthetic_fixture", "recipe_freeze", retuned["sha256"]):
                        render_anchor(self.anchor("recipe_freeze", retuned))})
        for campaign in (self.campaign, self.hpo_campaign):
            with self.subTest(source=campaign["freeze"]["kind"]), self.assertRaisesRegex(ValueError, "^anchor_conflict:"):
                gc.audit_chronology(self.other, campaign)

    def test_final_job_bound_to_an_unanchored_campaign(self):
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        self.q.seed_final(self.finals(), self.campaign["sha256"])
        # campaign_sha256 is an immutable extra, not an identity field: another fold gives a new job ID.
        stray = [final_spec(self.rows, self.pools, self.freeze, H("unanchored campaign"), fold=2)]
        self.fabricate(records=[new_record(spec) for spec in stray])
        with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
            gc.audit_chronology(self.other, self.campaign)

    def test_hpo_campaign_rejects_hpo_after_its_freeze_but_external_campaign_does_not(self):
        self.q.publish_anchor(self.anchor("recipe_freeze", self.freeze))
        self.run_hpo()
        self.q.publish_anchor(self.anchor("recipe_freeze", self.hpo_freeze))
        self.q.publish_anchor(self.anchor("campaign", self.campaign))
        self.q.publish_anchor(self.anchor("campaign", self.hpo_campaign))
        self.fabricate(records=[new_record(self.retuned_hpo())])  # a second search after C_freeze(hpo_d)
        with self.assertRaisesRegex(ValueError, "^hpo_job_after_freeze:"):
            gc.audit_chronology(self.other, self.hpo_campaign)
        report = gc.audit_chronology(self.other, self.campaign)
        self.assertEqual(report["hpo_jobs"], len(self.hpo) + 1)

    def retuned_hpo(self):
        rows, pools, plan, _, _ = hpo_fixture(hpo_seeds=(103,))
        return hpo_spec(rows, pools, plan)


if __name__ == "__main__":
    unittest.main()
