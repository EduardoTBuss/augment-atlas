"""Decision B1 (22/10/2026): HPO jobs run at a configs commit after the plan, under Q6.

The HPO configs embed ``plan_sha256`` (which covers ``plan.runtime.commit``), so
they can only be committed after the plan. The collector accepts jobs bound to
that later commit only with Q6 evidence (``same_code_identity_v1``). Synthetic
fixture of ``test_global_hpo``; no images, training or Docker.
"""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from augment_atlas import global_campaign as gc
from augment_atlas import global_jobs as gj
from augment_atlas.contracts import canonical_hash as H
from test_global_hpo import Fixture

CONFIGS_COMMIT = "e" * 40
IMAGE = "sha256:" + "d" * 64


def identity(plan_commit, **changes):
    value = {"policy": gc.Q6_POLICY, "policy_status": gc.Q6_POLICY_STATUS, "freeze_commit": plan_commit,
             "code_commit": CONFIGS_COMMIT, "image_id": IMAGE,
             "changed_paths": ["configs/global/synthetic_fixture/hpo/MANIFEST.json"],
             "freeze_code_tree_sha256": H("code tree"), "final_code_tree_sha256": H("code tree")}
    value.update(changes)
    return value


class B1Case(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fx = Fixture(tmp.name)
        for job in self.fx.jobs.values():  # the operator seeds at the configs commit, not at the plan commit
            job["spec"] = gj.hpo_job_spec(job["cfg"], config_path=job["spec"]["config"],
                                          config_sha256=job["config_sha"], code_commit=CONFIGS_COMMIT)

    def snapshot(self, image=IMAGE):
        records = self.fx.snapshot()
        for key in self.fx.jobs:
            path = self.fx.attempt_dir(key) / "runtime.json"
            provenance = json.loads(path.read_text())
            provenance["image_id"] = image
            path.write_text(json.dumps(provenance))
        return records

    def collect(self, snapshot, **kwargs):
        return self.fx.collect(snapshot, **kwargs)


class B1Tests(B1Case):
    def test_jobs_at_the_configs_commit_are_collected_with_q6_evidence(self):
        evidence, manifest = self.collect(self.snapshot(), code_identity=identity(self.fx.runtime["commit"]))
        self.assertEqual({row["status"] for row in evidence["rows"]}, {"completed"})
        self.assertEqual(manifest["code_commit"], CONFIGS_COMMIT)
        self.assertEqual(manifest["code_identity_sha256"], H(identity(self.fx.runtime["commit"])))

    def test_without_q6_evidence_jobs_at_another_commit_fail_closed(self):
        # Not silently "missing": the queue holds jobs of these pools that match no plan job.
        with self.assertRaisesRegex(ValueError, "^search_space_mismatch:"):
            self.collect(self.snapshot())

    def test_legacy_jobs_at_the_plan_commit_still_collect_without_evidence(self):
        for job in self.fx.jobs.values():
            job["spec"] = gj.hpo_job_spec(job["cfg"], config_path=job["spec"]["config"],
                                          config_sha256=job["config_sha"], code_commit=self.fx.runtime["commit"])
        evidence, manifest = self.collect(self.snapshot())
        self.assertEqual({row["status"] for row in evidence["rows"]}, {"completed"})
        self.assertEqual(manifest["code_commit"], self.fx.runtime["commit"])
        self.assertIsNone(manifest["code_identity_sha256"])

    def test_code_change_between_plan_and_configs_is_refused(self):
        plan_commit = self.fx.runtime["commit"]
        for bad in (identity(plan_commit, changed_paths=["src/augment_atlas/train.py"]),
                    identity(plan_commit, final_code_tree_sha256=H("other tree")),
                    identity("f" * 40)):  # evidence not anchored at the plan commit
            with self.subTest(bad=bad["changed_paths"]):
                with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
                    self.collect(self.snapshot(), code_identity=bad)

    def test_attempt_on_another_image_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^code_identity_mismatch: an HPO attempt ran on another image"):
            self.collect(self.snapshot(image="sha256:" + "a" * 64), code_identity=identity(self.fx.runtime["commit"]))

    def test_code_repo_without_identity_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
            self.collect(self.snapshot(), code_repo=Path(self.fx.root))


class B1GitTests(unittest.TestCase):
    """derive_code_identity on a real repository: configs-only change passes, a code change fails."""

    def git(self, *argv):
        return subprocess.run(["git", "-C", str(self.repo), *argv], check=True, capture_output=True,
                              text=True).stdout.strip()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        self.git("init", "-q"); self.git("config", "user.email", "t@example.invalid"); self.git("config", "user.name", "t")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "code.py").write_text("x = 1\n")
        self.git("add", "-A"); self.git("commit", "-qm", "plan commit")
        self.plan_commit = self.git("rev-parse", "HEAD")

    def commit(self, path, text):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        self.git("add", "-A"); self.git("commit", "-qm", f"change {path}")
        return self.git("rev-parse", "HEAD")

    def test_configs_only_commit_passes_and_code_change_fails(self):
        runtime = {"commit": self.plan_commit, "environment_sha256": H("env"), "imagenet_weights_sha256": H("w")}
        configs = self.commit("configs/global/ds/hpo/c00-s101.json", "{}\n")
        evidence = gc.derive_code_identity(self.repo, freeze_commit=self.plan_commit, code_commit=configs,
                                           image_id=IMAGE)
        self.assertEqual(gc.audit_code_identity(evidence, freeze_runtime=runtime, code_commit=configs, image_id=IMAGE),
                         evidence)
        changed = self.commit("src/code.py", "x = 2\n")
        evidence = gc.derive_code_identity(self.repo, freeze_commit=self.plan_commit, code_commit=changed,
                                           image_id=IMAGE)
        with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
            gc.audit_code_identity(evidence, freeze_runtime=runtime, code_commit=changed, image_id=IMAGE)


if __name__ == "__main__":
    unittest.main()
