"""Review finding F2: the HPO worker pins its image and probe to ``plan.runtime.environment_sha256``.

``environment_sha256(image, probe)`` (Q6, same immutable image) was checked by the operator CLI
(seed, collect, freeze, campaign seal) but not by the worker, which only compared the image with the
campaign for final jobs. ``runtime.worker`` now calls ``DockerRuntime.check_environment`` before
``queue.claim`` and again before the container is created. Docker, Git, the queue and the GPU are
mocks, as in ``test_global_runtime``; nothing runs for real.

The legacy runtime tests build ``DockerRuntime`` objects with a placeholder image and probe that do
not reproduce the plan hash, so the check cannot live in ``config_for``/``create`` without editing
them; it is a separate method that only the worker calls.
"""
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from augment_atlas import global_hpo as gh
from augment_atlas import runtime
from augment_atlas.contracts import canonical_hash as H
from augment_atlas.schema import new_record
from augment_atlas.storage import atomic_json
import test_global_runtime as legacy
import test_global_trainer as trainer

IMAGE = "sha256:" + "d" * 64
OTHER_IMAGE = "sha256:" + "e" * 64
PROBE = {"torch": "2.8.0", "torchvision": "0.23.0", "cuda": "12.8", "capability": [8, 6], "forward_backward": True,
         "opencv": None, "opencv_distribution": None}
OTHER_PROBE = {**PROBE, "torch": "2.9.0"}


class PinnedFixture(trainer.GlobalFixture):
    """The fixture of the legacy runtime tests, with a plan runtime that really is environment_sha256(IMAGE, PROBE)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime["environment_sha256"] = gh.environment_sha256(IMAGE, PROBE)


class F2Case(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="atlas-f2-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        with patch.object(legacy, "GlobalFixture", PinnedFixture):
            self.w = legacy.World(self.root)
        self.assertEqual(self.w.plan["runtime"]["environment_sha256"], gh.environment_sha256(IMAGE, PROBE))

    def host(self, image=IMAGE, probe=PROBE):
        host = self.w.docker_runtime()
        host.image, host.environment = image, probe
        host.gpu_uuid, host.memory_mb = "GPU-fixture", 24000.0
        return host

    def worker_settings(self):
        return {**self.w.settings, "runs": str(self.root / "runs"), "queue_checkout": str(self.root / "queue"),
                "remote": "mock-remote", "server": "host-fixture"}

    def run_worker(self, host, queue):
        with patch("augment_atlas.runtime.DockerRuntime", return_value=host), \
             patch.object(host, "preflight"), patch.object(host, "probe"), \
             patch.object(host, "compatible", return_value=True), \
             patch.object(host, "create") as create, \
             patch("augment_atlas.runtime.GitQueue", return_value=queue), \
             patch("augment_atlas.runtime.gpu_lock", return_value=contextlib.nullcontext()), \
             patch("augment_atlas.runtime.command") as docker:
            try:
                runtime.worker(self.worker_settings(), once=True)
            finally:
                self.docker, self.create = docker, create

    def pending_queue(self):
        queue = Mock()
        queue.list_jobs.return_value = [{"job_id": self.w.spec()["job_id"], "state": "pending", "spec": self.w.spec()}]
        queue.claim.return_value = None  # a correct worker reaches the claim; nobody else owns the job
        return queue


class WorkerChecksTheEnvironmentBeforeClaimTests(F2Case):
    def test_the_pinned_image_and_probe_still_reach_the_claim(self):
        queue = self.pending_queue()
        self.run_worker(self.host(), queue)
        queue.claim.assert_called_once()
        self.docker.assert_not_called()

    def test_another_image_is_refused_before_claim_and_before_docker(self):
        queue = self.pending_queue()
        with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
            self.run_worker(self.host(image=OTHER_IMAGE), queue)
        queue.claim.assert_not_called()
        self.docker.assert_not_called()
        self.create.assert_not_called()

    def test_another_probe_on_the_same_image_is_refused_before_claim(self):
        queue = self.pending_queue()
        with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
            self.run_worker(self.host(probe=OTHER_PROBE), queue)
        queue.claim.assert_not_called()
        self.docker.assert_not_called()

    def test_a_worker_without_a_recorded_probe_is_refused_before_claim(self):
        queue = self.pending_queue()
        host = self.host()
        del host.environment
        with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
            self.run_worker(host, queue)
        queue.claim.assert_not_called()

    def test_the_container_is_not_created_for_another_image_when_recovering_an_unclaimed_attempt(self):
        # active.json from an earlier run of this worker, container never created: the image is re-checked
        # before docker create, without going through the claim.
        host = self.host(image=OTHER_IMAGE)
        spec = self.w.spec()
        record = new_record(spec)
        record.update(state="claimed", attempt_id="a" * 32, token="token", owner={"gpu": "0"}, version=1, seq=1)
        out = self.root / "runs" / spec["job_id"] / record["attempt_id"]
        out.mkdir(parents=True)
        slot = self.root / "runs" / "worker-GPU-fixture"
        slot.mkdir(parents=True)
        atomic_json(slot / "active.json", dict(record=record, output=str(out), container="atlas-" + "a" * 32,
                                               image=OTHER_IMAGE, stage="prepared"))
        queue = Mock()
        host.inspect = Mock(return_value=None)
        with patch.object(runtime, "request_training_stop"):
            with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
                self.run_worker(host, queue)
        self.create.assert_not_called()
        self.docker.assert_not_called()
        queue.claim.assert_not_called()

        self.docker.reset_mock()  # the same journal with the pinned image proceeds to create
        host = self.host()
        host.inspect = Mock(side_effect=[None, {"State": {"Status": "exited", "Running": False, "ExitCode": 0}}] * 4)
        atomic_json(slot / "active.json", dict(record=record, output=str(out), container="atlas-" + "a" * 32,
                                               image=IMAGE, stage="prepared"))
        try:
            self.run_worker(host, queue)
        except Exception:  # whatever the mocked tail of the loop does, the container create was reached
            pass
        self.create.assert_called_once()


class AuditUnitTests(F2Case):
    def test_matching_environment_passes_and_returns_nothing(self):
        self.assertIsNone(self.host().check_environment(self.w.spec()))
        self.assertIsNone(gh.audit_worker_environment(self.host(), self.w.spec()))

    def test_final_and_smoke_specs_are_not_this_checks_business(self):
        host = self.host(image=OTHER_IMAGE)
        self.assertIsNone(host.check_environment(self.w.spec("final")))  # pinned to the campaign image by its own gate
        self.assertIsNone(host.check_environment({"phase": "smoke", "config": "configs/smoke/x.json"}))

    def test_refusal_does_not_depend_on_hashing_a_different_formula(self):
        # The formula is the one the operator CLI uses (seed, collect, freeze, campaign seal): unchanged.
        self.assertEqual(gh.environment_sha256(IMAGE, PROBE), H({"image_id": IMAGE, "probe": PROBE}))
        self.assertEqual(self.w.plan["runtime"]["environment_sha256"], H({"image_id": IMAGE, "probe": PROBE}))

    def test_the_plan_must_be_the_anchored_one(self):
        spec = self.w.spec()
        plan_file = self.w.data / "global" / legacy.DATASET / f"plan-{self.w.plan_sha}.json"
        plan = json.loads(plan_file.read_text())
        plan["runtime"]["environment_sha256"] = gh.environment_sha256(OTHER_IMAGE, PROBE)  # matches the OTHER image
        plan_file.write_text(json.dumps(plan, indent=2, sort_keys=True))
        with self.assertRaisesRegex(ValueError, "^plan_invalid:"):
            self.host(image=OTHER_IMAGE).check_environment(spec)  # an edited plan cannot legitimize another image
        plan_file.unlink()
        with self.assertRaisesRegex(ValueError, "^plan_invalid:"):
            self.host().check_environment(spec)

    def test_malformed_job_binding_is_refused(self):
        for bad in ({"global": None}, {"global": {"plan_sha256": "xyz"}}, {"dataset": None}):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "^plan_invalid:"):
                    self.host().check_environment({**self.w.spec(), **bad})


if __name__ == "__main__":
    unittest.main()
