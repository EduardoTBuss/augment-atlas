import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

from augment_atlas.queue import GitQueue, InvalidTransition, QueueConflict, QueueError, StaleAttempt, parse_record, render_record
from augment_atlas.schema import new_record, prepare_job


def spec(**changes):
    result = {"protocol_version": "smoke-v1", "dataset": "synthetic", "dataset_version": "synthetic-v1",
              "split_hash": "fixture-split-sha256", "recipe_hash": "fixture-recipe-sha256",
              "phase": "smoke", "fold": 0, "train_seed": 7, "arm": "none",
              "code_commit": "a" * 40, "compatibility": {"dataset": "synthetic", "min_gpu_memory_mb": 64}}
    result.update(changes)
    return result


class RenderTests(unittest.TestCase):
    def test_runtime_epochs_and_legacy_total_epochs_render_supplied_total(self):
        for metadata, expected in (({"epochs": 5}, "3/5"), ({"total_epochs": 7}, "3/7"), ({}, "3/?")):
            with self.subTest(metadata=metadata):
                record = new_record(spec())
                record.update(state="running", version=1, seq=1, attempt_id="attempt-fixture",
                              token="token-fixture", owner={"server": "synthetic-host"})
                record["history"] = [{"seq": 1, "state": "running", "attempt_id": "attempt-fixture",
                                      "epoch": 3, "at": "2026-10-15T20:00:00+00:00", **metadata}]
                before = copy.deepcopy(record)
                markdown = render_record(record)
                self.assertIn(f"Epoch: {expected} |", markdown)
                self.assertEqual(before, record)
                self.assertEqual(record, parse_record(markdown))


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="augment-atlas-test-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.remote = root / "remote.git"
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True, capture_output=True)
        self.q1 = GitQueue(root / "worker-a", self.remote)
        self.q2 = GitQueue(root / "worker-b", self.remote)
        self.record = self.q1.seed([spec()])[0]
        self.job_id = self.record["job_id"]

    def start(self, queue=None):
        queue = queue or self.q1
        claimed = queue.claim(self.job_id, {"server": "synthetic-host", "gpu": "GPU-fixture"})
        running = queue.update(self.job_id, claimed["attempt_id"], claimed["token"], claimed["version"],
                               {"event_id": "start", "state": "running", "epoch": 0})
        return running

    def test_identity_is_stable_and_changes_scientific_inputs(self):
        original = spec()
        before = copy.deepcopy(original)
        identity = prepare_job(original)["job_id"]
        self.assertEqual(before, original)
        self.assertEqual(identity, prepare_job(dict(reversed(list(original.items()))))["job_id"])
        for key, value in {"protocol_version": "v2", "dataset": "other-dataset", "dataset_version": "v2", "split_hash": "other",
                           "recipe_hash": "other", "phase": "hpo", "fold": 1, "train_seed": 8,
                           "arm": "gamma", "code_commit": "b" * 40}.items():
            self.assertNotEqual(identity, prepare_job(spec(**{key: value}))["job_id"], key)
        with self.assertRaises(ValueError):
            prepare_job(spec(phase=""))
        for invalid in (None, "", "   "):
            with self.subTest(dataset=invalid), self.assertRaises(ValueError):
                prepare_job(spec(dataset=invalid))
        missing_dataset = spec()
        del missing_dataset["dataset"]
        with self.assertRaises(ValueError):
            prepare_job(missing_dataset)

    def test_markdown_roundtrip_and_validation(self):
        self.assertEqual(self.record, parse_record(render_record(self.record)))
        with self.assertRaises(QueueError):
            parse_record("# missing record")
        broken = copy.deepcopy(self.record)
        broken["job_id"] = "b" * 64
        with self.assertRaises(QueueError):
            parse_record(render_record(broken))

    def test_two_clones_race_exactly_one_confirmed_claim(self):
        barrier = threading.Barrier(2)
        results = []
        for queue in (self.q1, self.q2):
            original = queue._push
            def racing_push(commit, original=original):
                barrier.wait(timeout=15)
                return original(commit)
            queue._push = racing_push
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(queue.claim, self.job_id, {"server": name})
                       for queue, name in ((self.q1, "worker-a"), (self.q2, "worker-b"))]
            results = [future.result(timeout=30) for future in futures]
        self.assertEqual(1, sum(record is not None for record in results))
        winner = next(record for record in results if record)
        remote = self.q1.list_jobs()[0]
        self.assertEqual(winner["token"], remote["token"])
        self.assertEqual(1, remote["version"])

    def test_push_accepted_response_lost_is_reconciled(self):
        original = self.q1._push
        def lost_response(commit):
            self.assertTrue(original(commit))
            raise QueueError("simulated accepted push lost response")
        self.q1._push = lost_response
        claimed = self.q1.claim(self.job_id, {"server": "worker-a"})
        self.assertEqual(claimed["token"], self.q2.list_jobs()[0]["token"])
        event = {"event_id": "durable-spool-1", "state": "running", "epoch": 1}
        updated = self.q1.update(self.job_id, claimed["attempt_id"], claimed["token"], claimed["version"], event)
        replayed = self.q1.update(self.job_id, claimed["attempt_id"], claimed["token"], claimed["version"], event)
        self.assertEqual(updated, replayed)
        self.assertEqual(2, updated["version"])

    def test_conflict_on_other_job_revalidates_then_retries(self):
        second = self.q1.seed([spec(arm="gamma")])[0]
        original = self.q1._push
        first = True
        def competing_push(commit):
            nonlocal first
            if first:
                first = False
                self.assertIsNotNone(self.q2.claim(second["job_id"], {"server": "worker-b"}))
            return original(commit)
        self.q1._push = competing_push
        self.assertIsNotNone(self.q1.claim(self.job_id, {"server": "worker-a"}))
        self.assertTrue(all(record["state"] == "claimed" for record in self.q1.list_jobs()))

    def test_stale_attempt_cannot_overwrite_new_attempt(self):
        running = self.start()
        stopped = self.q1.update(self.job_id, running["attempt_id"], running["token"], running["version"],
                                 {"event_id": "stopped", "state": "interrupted"})
        reset = self.q1.retry(self.job_id, stopped["version"], "operator verified child exited")
        new = self.q2.claim(self.job_id, {"server": "worker-b"})
        self.assertNotEqual(running["attempt_id"], new["attempt_id"])
        self.assertEqual("pending", reset["state"])
        with self.assertRaises(StaleAttempt):
            self.q1.update(self.job_id, running["attempt_id"], running["token"], running["version"],
                           {"event_id": "late-progress", "state": "running"})
        self.assertEqual(new["token"], self.q1.list_jobs()[0]["token"])

    def test_explicit_manual_resume_preserves_attempt_and_rejects_old_progress(self):
        running = self.start()
        stopped = self.q1.update(self.job_id, running["attempt_id"], running["token"], running["version"],
                                 {"event_id": "stop-for-checkpoint", "state": "interrupted", "epoch": 2})
        with self.assertRaises(InvalidTransition):
            self.q1.update(self.job_id, stopped["attempt_id"], stopped["token"], stopped["version"],
                           {"event_id": "implicit-resume", "state": "running"})
        resumed = self.q1.update(self.job_id, stopped["attempt_id"], stopped["token"], stopped["version"],
                                 {"event_id": "manual-resume", "state": "running", "manual_resume": True,
                                  "epoch": 2, "reason": "operator confirmed old child stopped"})
        self.assertEqual(stopped["attempt_id"], resumed["attempt_id"])
        self.assertEqual(stopped["token"], resumed["token"])
        self.assertEqual(stopped["version"] + 1, resumed["version"])
        with self.assertRaises(QueueConflict):
            self.q1.update(self.job_id, running["attempt_id"], running["token"], running["version"],
                           {"event_id": "old-progress", "state": "running", "epoch": 1})
        self.assertEqual(resumed, self.q2.list_jobs()[0])

    def test_terminal_state_never_regresses_and_conflicting_event_id_rejected(self):
        running = self.start()
        event = {"event_id": "complete", "state": "succeeded", "artifacts": {"checksum": "fixture"}}
        completed = self.q1.update(self.job_id, running["attempt_id"], running["token"], running["version"], event)
        # An old acknowledged event may be replayed without changing the terminal state.
        replay = self.q1.update(self.job_id, running["attempt_id"], running["token"], 1,
                                {"event_id": "start", "state": "running", "epoch": 0})
        self.assertEqual("succeeded", replay["state"])
        with self.assertRaises(InvalidTransition):
            self.q1.update(self.job_id, running["attempt_id"], running["token"], completed["version"],
                           {"event_id": "late", "state": "running"})
        with self.assertRaises(QueueConflict):
            self.q1.update(self.job_id, running["attempt_id"], running["token"], completed["version"],
                           {**event, "artifacts": {"checksum": "changed"}})
        with self.assertRaises(InvalidTransition):
            self.q1.retry(self.job_id, completed["version"], "cannot retry success")

    def test_version_mismatch_and_unconfirmed_claim_do_not_launch(self):
        running = self.start()
        with self.assertRaises(QueueConflict):
            self.q1.update(self.job_id, running["attempt_id"], running["token"], 1,
                           {"event_id": "new", "state": "running"})
        self.assertIsNone(self.q2.claim(self.job_id, {"server": "worker-b"}))

    def test_remote_unavailable_never_returns_claim(self):
        original = self.q1._push
        self.q1._push = lambda commit: False
        self.q1.max_retries = 2
        with self.assertRaises(QueueConflict):
            self.q1.claim(self.job_id, {"server": "worker-a"})
        self.assertEqual("pending", self.q2.list_jobs()[0]["state"])
        self.q1._push = original
        original_fetch = self.q1._fetch_head
        calls = 0
        def lost_verification():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise QueueError("verification network unavailable")
            return original_fetch()
        self.q1._fetch_head = lost_verification
        with self.assertRaises(QueueError):
            self.q1.claim(self.job_id, {"server": "worker-a"})
        self.assertEqual("claimed", self.q2.list_jobs()[0]["state"])
        self.assertIsNone(self.q2.claim(self.job_id, {"server": "worker-b"}))

    def test_dependencies_wait_for_success_and_immutable_seed(self):
        dependent = self.q1.seed([spec(arm="gamma", dependencies=[self.job_id])])[0]
        self.assertIsNone(self.q2.claim(dependent["job_id"], {"server": "worker-b"}))
        running = self.start()
        self.q1.update(self.job_id, running["attempt_id"], running["token"], running["version"],
                       {"event_id": "done", "state": "succeeded"})
        self.assertIsNotNone(self.q2.claim(dependent["job_id"], {"server": "worker-b"}))
        with self.assertRaises(QueueConflict):
            self.q1.seed([spec(compatibility={"different": True})])
        with self.assertRaises(ValueError):
            self.q1.seed([spec(arm="other", dependencies=["f" * 64])])

    def test_queue_sync_preserves_local_untracked_and_index(self):
        sentinel = self.q1.checkout / "local-notes.txt"
        sentinel.write_text("keep local edits", encoding="utf-8")
        self.q1._git("add", "local-notes.txt")
        before = self.q1._git("ls-files", "--stage").stdout
        self.q1.claim(self.job_id, {"server": "worker-a"})
        self.assertEqual("keep local edits", sentinel.read_text(encoding="utf-8"))
        self.assertEqual(before, self.q1._git("ls-files", "--stage").stdout)


if __name__ == "__main__":
    unittest.main()
