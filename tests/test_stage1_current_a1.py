"""A1: current primary artifacts and chronology, with the historic snapshot intact."""
import json
import copy
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import test_stage_overlap_a1 as a1
from augment_atlas.queue import GitQueue


class BufferedFixtureQueue(GitQueue):
    """Use real claim/update transitions, then publish setup outcomes in one Git commit."""
    def __init__(self, records):
        self._lock = threading.RLock()
        self.max_retries = 1
        self.records = copy.deepcopy(records)

    def _fetch_head(self):
        return "buffered-fixture"

    def _records(self, head):
        return self.records

    def _publish(self, head, changed, message):
        self.records.update((r["job_id"], r) for r in changed)
        return self.records


class CurrentPrimaryA1Tests(a1.A1Case):
    def report_argv(self, scope, document, command="report"):
        return self.env.base(command) + [
            "--code", str(self.env.code), "--remote", str(self.queue.remote),
            "--checkout", str(self.queue.checkout), "--campaign-sha256", self.sha,
            "--stage-authorization-sha256", document["sha256"],
            "--recorded-commit", self.fixture.c_campaign, "--scope", scope,
            "--runs", str(self.env.runs), "--host-validation", "minimal"]

    def finish(self, job, state="succeeded"):
        return a1.r1.ST().tc.finish(self.queue, job, state,
            initial=a1.fixtures.H(["initial", job["spec"]["fold"], job["spec"]["train_seed"]]))

    def seed_matrix(self):
        return self.queue.seed_final([v[5] for v in self.primary_jobs.values()], self.sha,
                                     campaign=self.campaign, code_repo=self.env.code)

    def finish_many(self, jobs, state="succeeded"):
        # Only setup is batched. The cell whose post-authorization completion
        # is under test below uses the real Git queue claim/update path.
        buffered = BufferedFixtureQueue({r["job_id"]: r for r in self.queue.list_jobs()})
        for job in jobs:
            a1.r1.ST().tc.finish(buffered, job, state,
                initial=a1.fixtures.H(["initial", job["spec"]["fold"], job["spec"]["train_seed"]]))
        self.push([buffered.records[job["job_id"]] for job in jobs], "synthetic primary setup outcomes")

    def test_stage1_current_includes_cell_finished_after_authorization_and_preserves_snapshot_bytes(self):
        jobs = self.seed_matrix()
        self.finish_many(jobs[:-1])
        _, document, published = self.publish_and_seed()
        old = self.run_cli(self.report_argv("stage1", document))
        old_bytes = Path(old["path"]).read_bytes()
        self.assertEqual(old["status_counts"].get("missing"), 1)
        self.assertTrue(self.run_cli(self.report_argv("stage1_current", document), code=2).startswith("stage1_incomplete:"))
        last = jobs[-1]
        self.finish(last)
        head = self.queue._fetch_head()
        current = self.run_cli(self.report_argv("stage1_current", document))
        value = json.loads(Path(current["path"]).read_bytes())
        self.assertEqual(current["status_counts"], {"succeeded": 50})
        self.assertEqual(value["seeds"]["authorized"], [11])
        self.assertEqual(value["matrix"]["authorized_cells"], 50)
        self.assertEqual(value["chronology"]["head"], head)
        self.assertEqual(value["chronology"]["final_jobs"], 150)
        cell = next(c for c in value["matrix"]["cells"] if
                    (c["arm"], c["fold"], c["seed"]) ==
                    (last["spec"]["arm"], last["spec"]["fold"], 11))
        self.assertEqual(cell["status"], "succeeded")
        collected = self.run_cli(self.report_argv("stage1_current", document, "collect"))
        self.assertEqual((collected["authorized_cells"], collected["collected"]), (50, 50))
        historic = self.run_cli(self.report_argv("stage1", document))
        self.assertEqual((historic["report_sha256"], historic["collection_sha256"]),
                         (old["report_sha256"], old["collection_sha256"]))
        self.assertEqual(Path(historic["path"]).read_bytes(), old_bytes)
        sensitivity = self.run_cli(self.report_argv("all_authorized", document))
        sensitivity_value = json.loads(Path(sensitivity["path"]).read_bytes())
        self.assertEqual(sensitivity_value["seeds"]["authorized"], [11, 13, 19])

    def test_current_collection_refuses_absent_pending_claimed_running_and_interrupted(self):
        _, document, _ = self.publish_and_seed()
        command = self.report_argv("stage1_current", document, "collect")
        self.assertTrue(self.run_cli(command, code=2).startswith("stage1_incomplete:"))
        job = self.seed_primary()
        for state in ("pending", "claimed", "running", "interrupted"):
            if state == "claimed":
                job = self.queue.claim(job["job_id"], {"server": "synthetic"})
            elif state != "pending":
                job = self.queue.update(job["job_id"], job["attempt_id"], job["token"], job["version"],
                                         {"event_id": state, "state": state})
            with self.subTest(state=state):
                self.assertIn(job["job_id"] + "=" + state, self.run_cli(command, code=2))

    def test_current_report_keeps_final_failures_as_outcomes(self):
        jobs = self.seed_matrix()
        _, document, _ = self.publish_and_seed()
        self.finish_many(jobs, "failed")
        out = self.run_cli(self.report_argv("stage1_current", document))
        self.assertEqual(out["status_counts"], {"failed": 50})
        value = json.loads(Path(out["path"]).read_bytes())
        self.assertEqual(value["chronology"]["head"], self.queue._fetch_head())
        self.assertTrue(all(c["status"] == "failed" for c in value["matrix"]["cells"] if c["seed"] == 11))

    def test_current_scope_refuses_forged_stage_job_before_collecting(self):
        self.seed_matrix()
        _, document, _ = self.publish_and_seed()
        stage = dict(next(v[5] for v in self.stage_jobs.values()))
        stage.pop("job_id")
        stage["code_commit"] = self.original
        from augment_atlas.schema import new_record, prepare_job
        self.push([new_record(prepare_job(stage))], "synthetic forged stage identity")
        with patch("augment_atlas.global_collect.collect_final_entries") as collector:
            self.assertIn("code_identity_mismatch:", self.run_cli(self.report_argv("stage1_current", document), code=2))
        collector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
