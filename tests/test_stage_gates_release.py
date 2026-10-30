"""D9: declared release of the ``primary_workers_active`` gate (real disposable Git repositories, no training).

``anchor --kind stage_authorization --declare-all-workers-upgraded "<text>"`` lets the
operator publish one campaign's stage authorization while primary jobs of OTHER
campaigns are still active, after declaring that every worker reading the queue
runs the new orchestrator. The release never covers the campaign being
authorized, never reaches the seed-request route (RealStageAuthorizer), lists the
released jobs and records the declaration in a private record outside the repo.
"""
import copy
import inspect
import json
from pathlib import Path
import shutil
import unittest

# NOTE: CLI end-to-end release cases need a second, anchored campaign fixture (an unanchored foreign
# primary job makes the CLI chronology audit fail first with campaign_anchor_missing); pending post-ESANN.

from augment_atlas import global_campaign as gc, seed_stage_authorizer, stage_gates as gates
from augment_atlas.contracts import canonical_hash
from augment_atlas.queue import QueueConflict
from augment_atlas.schema import new_record, prepare_job
import test_global_cli as fixtures
import test_stage_gates_r1 as r1

DECLARATION = "synthetic queue: every worker restarted on the new orchestrator at 2026-10-30T12:00:00Z"


class ReleaseCase(r1.R1Case):
    def gated(self, declaration=None):
        if declaration is None:
            return gates.StageGatedQueue(self.queue.checkout, self.queue.remote)
        return gates.StageGatedQueue(self.queue.checkout, self.queue.remote, all_workers_upgraded=declaration)

    def stage_path(self, document=None):
        return r1.PublicationGateTests.stage_path(self, document)

    def fake_authorization(self, dataset_id, label):
        return r1.PublicationGateTests.fake_authorization(self, dataset_id, label)

    def foreign_primary(self, dataset_id="synthetic_other", campaign_sha256=None):
        """A pending primary final job of another dataset and campaign (an active stage-1 worker elsewhere)."""
        spec = copy.deepcopy(self.primary())
        spec.pop("job_id")
        spec["dataset"] = dataset_id
        spec["global"] = {**spec["global"], "campaign_sha256": campaign_sha256 or canonical_hash("another campaign")}
        spec = prepare_job(spec)
        self.push([new_record(spec)], "synthetic primary job of another campaign")
        return spec["job_id"]

    def anchor_argv(self, *extra):
        return self.env.base("anchor") + ["--code", str(self.env.code), "--remote", str(self.queue.remote),
                                          "--checkout", str(self.queue.checkout), "--kind", "stage_authorization",
                                          "--document-sha256", self.auth["sha256"], "--image-id", fixtures.IMAGE,
                                          "--environment-probe", str(self.env.probe), *extra]

    def run_cli(self, argv, code=0):
        return r1.ST().tc.run_cli(argv, code=code)

    def assertNotAnchored(self, document=None):
        self.assertIsNone(self.queue.read_path(self.queue._fetch_head(), self.stage_path(document)))


class CliReleaseTests(ReleaseCase):
    def test_c_empty_or_misplaced_declaration_is_invalid_input(self):
        self.foreign_primary()
        for text in ("", "   ", "\t\n"):
            with self.subTest(text=repr(text)):
                error = self.run_cli(self.anchor_argv("--declare-all-workers-upgraded", text), code=2)
                self.assertTrue(error.startswith("cli_input_invalid:"), error)
        campaign_anchor = self.env.base("anchor") + [
            "--code", str(self.env.code), "--remote", str(self.queue.remote), "--checkout", str(self.queue.checkout),
            "--kind", "campaign", "--document-sha256", self.sha, "--declare-all-workers-upgraded", DECLARATION]
        self.assertTrue(self.run_cli(campaign_anchor, code=2).startswith("cli_input_invalid:"))
        with self.assertRaisesRegex(ValueError, "^cli_input_invalid:"):
            self.gated(" ")
        self.assertNotAnchored()


class LibraryReleaseTests(ReleaseCase):
    def test_d_library_release_refuses_active_primary_of_the_authorized_campaign(self):
        other = self.fake_authorization("synthetic_other", "campaign B")
        foreign = self.foreign_primary("synthetic_third", canonical_hash("campaign C"))
        own = self.foreign_primary("synthetic_other", other["campaign_sha256"])
        with self.assertRaisesRegex(QueueConflict, "^primary_workers_active:") as caught:
            self.gated(DECLARATION).publish_anchor(gc.build_anchor("stage_authorization", other))
        self.assertIn(own, str(caught.exception))
        self.assertNotIn(foreign, str(caught.exception))
        self.assertNotAnchored(other)

    def test_e_library_release_is_idempotent_and_default_queue_is_unchanged(self):
        other = self.fake_authorization("synthetic_other", "campaign B")
        foreign = self.foreign_primary("synthetic_third", canonical_hash("campaign C"))
        with self.assertRaisesRegex(QueueConflict, "^primary_workers_active:"):
            self.gated().publish_anchor(gc.build_anchor("stage_authorization", other))
        released = self.gated(DECLARATION)
        published = released.publish_anchor(gc.build_anchor("stage_authorization", other))
        self.assertEqual(released.released_primary_jobs, [foreign])
        # The gate ran on the exact HEAD the anchor commit is built on.
        self.assertEqual(released.release_head,
                         self.queue._git("rev-parse", published["commit"] + "^").stdout.strip())
        again = self.gated(DECLARATION)
        self.assertEqual(again.publish_anchor(gc.build_anchor("stage_authorization", other))["commit"],
                         published["commit"])
        self.assertIsNone(again.released_primary_jobs)
        self.assertEqual(self.gated().publish_anchor(gc.build_anchor("stage_authorization", other))["commit"],
                         published["commit"])

    def test_seed_request_route_has_no_release(self):
        # X5 / RealStageAuthorizer stays forbidden for sensitivity stages; it never builds a released queue.
        self.assertNotIn("all_workers_upgraded", inspect.getsource(seed_stage_authorizer))
        self.assertIsNone(self.gated().all_workers_upgraded)
        job_id = self.foreign_primary()
        with self.assertRaisesRegex(QueueConflict, "^primary_workers_active:") as caught:
            gates.refuse_active_primary(self.queue, self.queue._fetch_head())
        self.assertIn(job_id, str(caught.exception))



class ReleaseRecordTests(unittest.TestCase):
    """CLI warning and private record of a declared release (unit level: no queue, no audit)."""

    def test_record_is_private_sealed_and_lists_released_jobs(self):
        import tempfile
        from types import SimpleNamespace
        from augment_atlas import global_cli
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(data_root=tmp, dataset="synthetic")
            document = {"campaign_sha256": "c" * 64, "sha256": "a" * 64}
            publisher = SimpleNamespace(released_primary_jobs=["j1", "j2"], release_head="h" * 40)
            warning = global_cli._release_record(args, DECLARATION, document, {"commit": "f" * 40}, publisher)
            self.assertEqual((warning["code"], warning["gate_evaluated"], warning["released_primary_jobs"]),
                             ("primary_workers_released", True, ["j1", "j2"]))
            path = Path(warning["record_path"])
            self.assertTrue(path.is_relative_to(Path(tmp).resolve()))
            record = json.loads(path.read_bytes())
            body = {key: value for key, value in record.items() if key != "sha256"}
            self.assertEqual(canonical_hash(body), record["sha256"])
            self.assertEqual(record["sha256"], warning["record_sha256"])
            self.assertEqual((record["declaration"], record["campaign_sha256"], record["stage_authorization_sha256"],
                              record["anchor_commit"], record["gate_head"]),
                             (DECLARATION, "c" * 64, "a" * 64, "f" * 40, "h" * 40))
            # Republication of an existing anchor: the gate did not run, nothing is released or recorded.
            again = global_cli._release_record(args, DECLARATION, document, {"commit": "f" * 40},
                                               SimpleNamespace(released_primary_jobs=None, release_head=None))
            self.assertEqual((again["gate_evaluated"], again["released_primary_jobs"]), (False, []))
            self.assertNotIn("record_path", again)


if __name__ == "__main__":
    unittest.main()
