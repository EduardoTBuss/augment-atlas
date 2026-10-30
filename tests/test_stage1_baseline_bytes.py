"""R1-03/R1-04: authorize-stage needs the complete succeeded stage-1 matrix; stage-1 report bytes vs 835c70e.

The baseline module is read from Git (``git archive 835c70e src``) into a
temporary directory and runs in a separate Python process against the same
synthetic data root and queue. Without that commit in the clone, only the
baseline sub-tests are skipped (a skip is not a pass).
"""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from augment_atlas.queue import GitQueue
import test_global_cli as fixtures

BASELINE = "835c70e"


def baseline_src(target):
    """src/ of the baseline commit (read-only object access), or None if the clone lacks it."""
    probe = subprocess.run(["git", "-C", str(fixtures.ROOT), "cat-file", "-e", BASELINE + "^{commit}"],
                           capture_output=True)
    if probe.returncode:
        return None
    archive = subprocess.run(["git", "-C", str(fixtures.ROOT), "archive", "--format=zip", BASELINE, "src"],
                             capture_output=True, check=True).stdout
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        bundle.extractall(target)
    return Path(target) / "src"


class SecondAttempt:
    """Queue view for the fixture worker: same claim, event IDs suffixed for a retried job."""

    def __init__(self, queue):
        self.queue = queue

    def claim(self, job_id, owner):
        return self.queue.claim(job_id, owner)

    def update(self, job_id, attempt_id, token, version, event):
        return self.queue.update(job_id, attempt_id, token, version, {**event, "event_id": event["event_id"] + "-2"})


class Stage1CompletionTests(fixtures.CliCase):
    def setUp(self):
        super().setUp()
        work = tempfile.TemporaryDirectory(prefix="atlas-baseline-")
        self.addCleanup(work.cleanup)
        self.work = Path(work.name)
        self.baseline = baseline_src(self.work / "tree")

    def baseline_cli(self, argv, code=0):
        environment = {**os.environ, "PYTHONPATH": str(self.baseline)}
        result = subprocess.run([sys.executable, "-m", "augment_atlas.global_cli", *argv], cwd=self.work,
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, code, result.stderr)
        return json.loads(result.stdout) if code == 0 else json.loads(result.stderr)["error"]

    def take(self, output):
        """Move a published private report away so the next writer publishes its own bytes."""
        path = Path(output["path"])
        data = path.read_bytes()
        shutil.move(str(path), str(self.work / (path.name + f".{len(list(self.work.iterdir()))}")))
        return data

    def test_authorize_requires_complete_stage1_and_stage1_bytes_match_baseline(self):
        env = self.env
        freeze, campaign = self.to_campaign()
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "campaign",
                                           "--document-sha256", campaign["campaign_sha256"]])
        self.run_cli(env.base("seed-final") + self.campaign_args(campaign))
        materialize = env.base("materialize-final") + ["--code", str(env.code), *self.freeze_args(freeze),
                                                       "--campaign-sha256", campaign["campaign_sha256"], "--stage", "2"]
        self.run_cli(materialize)
        fixtures.git(env.code, "add", "-A")
        fixtures.git(env.code, "commit", "-qm", "synthetic stage additions")
        unqueued = env.base("authorize-stage") + [
            "--code", str(env.code), "--campaign-sha256", campaign["campaign_sha256"], "--stage", "2",
            "--image-id", fixtures.IMAGE, "--environment-probe", str(env.probe), "--decided-on", "2026-10-29",
            "--decided-after-primary-results", "true",
            "--rationale", "Manual route: decided after viewing synthetic primary results; sensitivity only.",
            "--sealed-at", "2026-10-29T18:00:00Z"]
        authorize = unqueued + env.queue_args()
        self.assertTrue(self.run_cli(unqueued, code=2).startswith("cli_input_invalid:"))
        self.assertTrue(self.run_cli(authorize, code=2).startswith("stage1_incomplete: 50 of 50"))

        queue = GitQueue(env.checkout, env.remote)
        jobs = sorted(queue.list_jobs(), key=lambda job: (job["spec"]["arm"], job["spec"]["fold"]))
        initial = lambda job: fixtures.H(["initial", job["spec"]["fold"], job["spec"]["train_seed"]])
        for job in jobs[:-1]:
            self.finish(queue, job, "succeeded", initial=initial(job))
        last = jobs[-1]
        self.finish(queue, last, "failed", initial=initial(last))
        error = self.run_cli(authorize, code=2)
        self.assertTrue(error.startswith("stage1_incomplete: 1 of 50"), error)
        self.assertIn(last["job_id"] + "=failed", error)
        failed = next(job for job in queue.list_jobs() if job["job_id"] == last["job_id"])
        queue.retry(last["job_id"], failed["version"], "synthetic infrastructure retry before authorization")
        self.assertIn(last["job_id"] + "=pending", self.run_cli(authorize, code=2))
        # finish() uses fixed event IDs; a second attempt needs fresh ones (the queue refuses reuse).
        self.finish(SecondAttempt(queue), last, "succeeded", initial=initial(last))

        report = env.base("report") + self.campaign_args(campaign) + ["--runs", str(env.runs),
                                                                      "--host-validation", "minimal"]
        baseline_bytes = None
        with self.subTest("835c70e baseline bytes before authorization"):
            if self.baseline is None:
                self.skipTest(f"{BASELINE} is not in this clone; baseline byte comparison not run")
            where = self.baseline_cli(["identity", "--data-root", str(env.data), "--source", str(env.source),
                                       "--dataset", fixtures.DATASET])
            self.assertEqual(where["records"], 16)
            module = subprocess.run([sys.executable, "-c", "import augment_atlas; print(augment_atlas.__file__)"],
                                    cwd=self.work, env={**os.environ, "PYTHONPATH": str(self.baseline)},
                                    capture_output=True, text=True, check=True).stdout.strip()
            self.assertTrue(Path(module).resolve().is_relative_to(self.baseline.resolve()), module)
            base = self.baseline_cli(report)
            baseline_bytes = self.take(base)
            current = self.run_cli(report)
            self.assertEqual((current["report_sha256"], current["collection_sha256"]),
                             (base["report_sha256"], base["collection_sha256"]))
            self.assertEqual(self.take(current), baseline_bytes)
        before = self.run_cli(report)
        before_bytes = self.take(before)

        authorized = self.run_cli(authorize)
        document = json.loads(Path(authorized["path"]).read_bytes())
        self.assertIs(document["decided_after_primary_results"], True)
        pin = ["--stage-authorization-sha256", authorized["stage_authorization_sha256"]]
        self.run_cli(materialize + pin)
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind",
                                           "stage_authorization", "--document-sha256",
                                           authorized["stage_authorization_sha256"], "--image-id", fixtures.IMAGE,
                                           "--environment-probe", str(env.probe)])
        with self.subTest("835c70e reader rejects the queue after the stage anchor (R1-02 rationale)"):
            if self.baseline is None:
                self.skipTest(f"{BASELINE} is not in this clone; old-reader coexistence not run")
            error = self.baseline_cli(env.base("audit-chronology") + self.campaign_args(campaign), code=2)
            self.assertTrue(error.startswith("anchor_invalid"), error)
        self.assertEqual(self.run_cli(env.base("seed-final") + self.campaign_args(campaign) + pin)["jobs"], 100)
        after = self.run_cli(report + pin)
        self.assertEqual(after["report_sha256"], before["report_sha256"])
        after_bytes = self.take(after)
        self.assertEqual(after_bytes, before_bytes)
        if baseline_bytes is not None:
            self.assertEqual(after_bytes, baseline_bytes)


if __name__ == "__main__":
    unittest.main()
