"""A1 adversarial authorization, publication and primary-worker checks (offline Git)."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from augment_atlas import global_campaign as gc, stage_gates as gates, stage_overlap as overlap
from augment_atlas.contracts import canonical_hash
from augment_atlas.queue import QueueConflict, anchor_path
from augment_atlas.schema import new_record, prepare_job
import test_global_cli as fixtures
import test_stage_gates_r1 as r1

AMENDMENT = "SAP v1 amendment A1: predeclared sensitivity runs in parallel on another node"
UPGRADED = "Every synthetic queue worker restarted on the A1 orchestrator between jobs"


class A1Case(r1.R1Case):
    def authorize_argv(self, *extra):
        return self.env.base("authorize-stage") + [
            "--code", str(self.env.code), "--remote", str(self.queue.remote),
            "--checkout", str(self.queue.checkout), "--campaign-sha256", self.sha, "--stage", "2",
            "--image-id", fixtures.IMAGE, "--environment-probe", str(self.env.probe),
            "--decided-on", "2026-10-29", "--decided-after-primary-results", "true",
            "--rationale", "Synthetic A1 timing test " + self._testMethodName,
            "--sealed-at", "2026-10-30T18:00:00Z", *extra]

    def run_cli(self, argv, code=0):
        return r1.ST().tc.run_cli(argv, code=code)

    def authorize(self):
        out = self.run_cli(self.authorize_argv("--allow-incomplete-stage1", AMENDMENT))
        return out, json.loads(Path(out["path"]).read_bytes())

    def anchor_argv(self, document, *extra):
        return self.env.base("anchor") + [
            "--code", str(self.env.code), "--remote", str(self.queue.remote),
            "--checkout", str(self.queue.checkout), "--kind", "stage_authorization",
            "--document-sha256", document["sha256"], "--image-id", fixtures.IMAGE,
            "--environment-probe", str(self.env.probe), *extra]

    def publish_and_seed(self):
        out, document = self.authorize()
        published = self.run_cli(self.anchor_argv(document, "--declare-all-workers-upgraded", UPGRADED))
        self.queue.seed_final([v[5] for v in self.stage_jobs.values()], self.sha,
                              campaign=self.campaign, stage_authorization=document, code_repo=self.env.code)
        return out, document, published

    def primary_host(self):
        return r1.SeedEligibilityTests.host(self)


class AuthorizationA1Tests(A1Case):
    def test_absent_flag_still_refuses_missing_and_every_incomplete_state(self):
        self.assertTrue(self.run_cli(self.authorize_argv(), code=2).startswith("stage1_incomplete:"))
        job = self.seed_primary()
        for state in ("pending", "claimed", "running", "interrupted", "failed"):
            if state == "claimed":
                job = self.queue.claim(job["job_id"], {"server": "synthetic"})
            elif state != "pending":
                if state == "failed":
                    job = self.queue.update(job["job_id"], job["attempt_id"], job["token"], job["version"],
                        {"event_id": "resume", "state": "running", "manual_resume": True})
                job = self.queue.update(job["job_id"], job["attempt_id"], job["token"], job["version"],
                    {"event_id": state, "state": state})
            with self.subTest(state=state):
                error = self.run_cli(self.authorize_argv(), code=2)
                self.assertIn(job["job_id"] + "=" + state, error)
                out, document = self.authorize()
                record = json.loads(Path(out["record_path"]).read_bytes())
                self.assertEqual(record["amendment_and_reason"], AMENDMENT)
                self.assertIs(record["allow_incomplete_stage1"], True)
                self.assertEqual(overlap.validate_record(record, self.sha, document["sha256"]), record)
                self.assertEqual(record["sha256"], out["record_sha256"])
                self.assertTrue(Path(out["record_path"]).is_relative_to(self.env.data))
                self.assertNotIn("allow_incomplete_stage1", document)
                self.assertNotIn("amendment_and_reason", gc.build_anchor("stage_authorization", document))

    def test_empty_amendment_is_refused_before_sealing(self):
        for text in ("", "   ", "\t\n"):
            with self.subTest(text=repr(text)):
                self.assertTrue(self.run_cli(self.authorize_argv("--allow-incomplete-stage1", text),
                                              code=2).startswith("cli_input_invalid:"))

    def test_private_record_cannot_be_rewritten_or_rebound_even_if_resigned(self):
        out, document = self.authorize()
        path = Path(out["record_path"])
        original = path.read_bytes()
        record = json.loads(original)
        for key, value in (("campaign_sha256", "0" * 64), ("stage_authorization_sha256", "0" * 64),
                           ("allow_incomplete_stage1", False), ("amendment_and_reason", " ")):
            bad = {**record, key: value}
            bad = r1.resign(bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                overlap.validate_record(bad, self.sha, document["sha256"])
        with self.assertRaisesRegex(ValueError, "no-clobber"):
            overlap.publish_record(self.env.data, document, "Another amendment and reason")
        self.assertEqual(path.read_bytes(), original)


class PublicationA1Tests(A1Case):
    def test_own_primary_release_requires_both_explicit_declarations(self):
        job = self.seed_primary()
        # Upgrade declaration alone still cannot release this campaign.
        error = self.run_cli(self.anchor_argv(self.auth, "--declare-all-workers-upgraded", UPGRADED), code=2)
        self.assertIn("primary_workers_active:", error)
        self.assertIn(job["job_id"], error)
        out, document = self.authorize()
        # A1 record alone still cannot publish while any primary worker is active.
        self.assertTrue(self.run_cli(self.anchor_argv(document), code=2).startswith("primary_workers_active:"))
        published = self.run_cli(self.anchor_argv(document, "--declare-all-workers-upgraded", UPGRADED))
        warning = published["warning"]
        self.assertEqual(warning["released_primary_jobs"], [job["job_id"]])
        record = json.loads(Path(warning["record_path"]).read_bytes())
        self.assertEqual(record["anchor_commit"], published["commit"])
        self.assertEqual(record["gate_head"], self.queue._git("rev-parse", published["commit"] + "^").stdout.strip())
        public = json.loads(self.queue.read_path(self.queue._fetch_head(), published["path"]))
        self.assertNotIn("allow_incomplete_stage1", public)
        self.assertNotIn(AMENDMENT, json.dumps(public))

    def test_forged_private_record_does_not_release_own_jobs(self):
        self.seed_primary()
        out, document = self.authorize()
        path = Path(out["record_path"])
        record = json.loads(path.read_bytes())
        record["stage_authorization_sha256"] = "0" * 64
        path.write_text(json.dumps(r1.resign(record)), encoding="utf-8")
        self.assertTrue(self.run_cli(self.anchor_argv(document, "--declare-all-workers-upgraded", UPGRADED),
                                      code=2).startswith("campaign_invalid:"))
        self.assertIsNone(self.queue.read_path(self.queue._fetch_head(),
            anchor_path(document["dataset"]["id"], "stage_authorization", document["sha256"])))

    def test_release_gate_rechecks_concurrent_own_primary_on_exact_parent(self):
        out, document = self.authorize()
        publisher = gates.StageGatedQueue(self.queue.checkout, self.queue.remote, all_workers_upgraded=UPGRADED,
                                          incomplete_stage1_record=overlap.load_record(self.env.data, document))
        job = self.seed_primary()
        published = publisher.publish_anchor(gc.build_anchor("stage_authorization", document))
        self.assertEqual(publisher.released_primary_jobs, [job["job_id"]])
        self.assertEqual(publisher.release_head, self.queue._git("rev-parse", published["commit"] + "^").stdout.strip())


class PrimaryObservationA1Tests(A1Case):
    def test_primary_runtime_and_claim_accept_anchored_stage_jobs_without_execution_pin(self):
        job = self.seed_primary()
        _, document, published = self.publish_and_seed()
        host = self.primary_host()
        with patch.object(host, "weights_cache_for", return_value=None):
            host.global_config_for(self.primary())
        self.assertIsNone(host._stage_claim_context[1])
        head = self.queue._fetch_head()
        audit = gc.audit_chronology(self.queue, self.campaign, head=head, code_repo=self.env.code,
                                    authorization_root=self.env.data)
        self.assertEqual((audit["head"], audit["final_jobs"]), (head, 101))
        self.assertEqual(audit["C_stage_authorization"], published["commit"])
        claimed = self.queue.claim(job["job_id"], {"server": "synthetic"}, stage_context=host._stage_claim_context)
        self.assertEqual(claimed["state"], "claimed")
        # Observing a supplement never gives the primary worker execution permission.
        stage = next(v[5] for v in self.stage_jobs.values())
        with self.assertRaisesRegex(ValueError, "campaign_invalid:"):
            self.queue.claim(stage["job_id"], {"server": "synthetic"}, stage_context=host._stage_claim_context)
        host.commit = self.execution
        with patch.object(host, "weights_cache_for", return_value=None), self.assertRaisesRegex(
                ValueError, "code_identity_mismatch:"):
            host.global_config_for(self.primary())

    def test_primary_audit_refuses_unanchored_later_job(self):
        self.seed_primary()
        stage = next(v[5] for v in self.stage_jobs.values())
        self.push([new_record(stage)], "synthetic unanchored stage job")
        with self.assertRaises(ValueError):
            gc.audit_chronology(self.queue, self.campaign, code_repo=self.env.code,
                                authorization_root=self.env.data)

    def test_primary_claim_refuses_forged_stage_config_even_with_anchor(self):
        job = self.seed_primary()
        self.publish_and_seed()
        host = self.primary_host()
        with patch.object(host, "weights_cache_for", return_value=None):
            host.global_config_for(self.primary())
        stage = copy.deepcopy(next(v[5] for v in self.stage_jobs.values()))
        stage.pop("job_id")
        stage["config_sha256"] = canonical_hash("forged stage config")
        stage = prepare_job(stage)
        self.push([new_record(stage)], "synthetic forged stage config")
        # Config digests do not change the stable job ID: rewriting one is
        # rejected by the immutable-spec replay before stage binding is reached.
        with self.assertRaisesRegex(ValueError, "chronology_invalid:.*spec was rewritten"):
            self.queue.claim(job["job_id"], {"server": "synthetic"}, stage_context=host._stage_claim_context)
        self.assertEqual(next(r["state"] for r in self.queue.list_jobs() if r["job_id"] == job["job_id"]), "pending")

    def test_primary_claim_refuses_uncovered_seed_despite_a_valid_stage_anchor(self):
        job = self.seed_primary()
        self.publish_and_seed()
        host = self.primary_host()
        with patch.object(host, "weights_cache_for", return_value=None):
            host.global_config_for(self.primary())
        stage = copy.deepcopy(next(v[5] for v in self.stage_jobs.values()))
        stage.pop("job_id")
        stage["train_seed"] = 17
        stage["config"] = stage["config"].replace("-s13.json", "-s17.json").replace("-s19.json", "-s17.json")
        self.push([new_record(prepare_job(stage))], "synthetic uncovered seed after valid anchor")
        with self.assertRaisesRegex(ValueError, "campaign_invalid: seed is outside"):
            self.queue.claim(job["job_id"], {"server": "synthetic"}, stage_context=host._stage_claim_context)

    def test_resigned_private_authorization_differing_from_anchor_is_refused(self):
        job = self.seed_primary()
        out, document, _ = self.publish_and_seed()
        path = Path(out["path"])
        document["rationale"] = "Resigned but not anchored"
        path.write_text(json.dumps(r1.resign(document)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "campaign_invalid:"):
            gc.audit_chronology(self.queue, self.campaign, code_repo=self.env.code,
                                authorization_root=self.env.data)


if __name__ == "__main__":
    unittest.main()
