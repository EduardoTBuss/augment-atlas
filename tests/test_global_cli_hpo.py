"""Global CLI, HPO path (campaign (b), decision B1) end to end on synthetic data.

Real files, a real code repository and a bare Git queue; the HPO and final
workers are played by writing the attempt artifacts a worker would write. No
training, images decoded or Docker. Plan at commit P, HPO configs at a later
commit X (only configs/global/** changed, Q6), final configs at a third commit.
"""
import hashlib
import json
from pathlib import Path
import unittest

from augment_atlas import global_development as gd
from augment_atlas import global_evaluation as ge
from augment_atlas import global_jobs as gj
from augment_atlas.contracts import canonical_hash as H
from augment_atlas.queue import GitQueue
from test_global_cli import DATASET, IMAGE, WEIGHTS, CliCase, git

GRID = [{"path": "optimizer.parameters.lr", "values": [0.00003, 0.0001, 0.0003, 0.001]}]
SPEC = {"grid": GRID, "default": {"optimizer.parameters.lr": 0.0001}, "seeds": [101],
        "budget": {"max_candidates": 6, "max_seeds": 2, "max_epochs": 20, "job_timeout_s": 3600,
                   "gpu_minutes_cap": 120},
        "failure_policy": "require_all_defined"}
# D_select scores per candidate -> AUROC 1.0 for c01, 0.5 for the others (c01 must win by the ranking).
GOOD, TIE = [[0.9, 0.1], [0.2, 0.8]], [[0.5, 0.5], [0.5, 0.5]]


class HpoCase(CliCase):
    def runtime_args(self):
        return ["--code", str(self.env.code), "--image-id", IMAGE, "--environment-probe", str(self.env.probe),
                "--weights-sha256", WEIGHTS]

    def to_seeded_hpo(self):
        env = self.env
        self.import_pools()
        self.view = self.run_cli(env.base("hpo-view"))
        self.run_cli(env.base("d-view") + ["--view-sha256", self.view["hpo_view_sha256"],
                                           "--views-root", str(env.views)])
        spec = Path(env.data).parent / "search-space-spec.json"
        spec.write_text(json.dumps(SPEC), encoding="utf-8")
        self.space = self.run_cli(env.base("search-space") + self.runtime_args() +
                                  ["--spec", str(spec), "--color", "grayscale_to_rgb"])
        self.assertEqual(self.space["candidates"], 4)
        self.plan = self.run_cli(env.base("build-plan") + self.runtime_args() + [
            "--search-space-sha256", self.space["search_space_sha256"],
            "--view-sha256", self.view["hpo_view_sha256"], "--final-seeds", "11,13,19"])
        made = self.run_cli(env.base("materialize-hpo") + ["--code", str(env.code), *self.pins()])
        self.assertEqual(made["configs"], 4)
        git(env.code, "add", "-A"); git(env.code, "commit", "-qm", "hpo configs")
        self.seeded = self.run_cli(env.base("seed-hpo") + ["--code", str(env.code), *self.pins(),
                                                           *env.queue_args(), "--image-id", IMAGE,
                                                           "--environment-probe", str(env.probe)])
        return self.seeded

    def pins(self):
        return ["--search-space-sha256", self.space["search_space_sha256"], "--plan-sha256", self.plan["plan_sha256"],
                "--view-sha256", self.view["hpo_view_sha256"]]

    def finish_hpo(self, queue, job, scores, image=IMAGE, end_state="succeeded"):
        """Play the HPO worker: claim, run, write the D_select artifacts, succeed."""
        env = self.env
        claimed = queue.claim(job["job_id"], {"server": "fixture"})
        running = queue.update(job["job_id"], claimed["attempt_id"], claimed["token"], claimed["version"],
                               {"event_id": "run-" + claimed["attempt_id"], "state": "running"})
        spec = job["spec"]
        cfg = json.loads((env.code / spec["config"]).read_text(encoding="ascii"))
        pools = gd._read(env.data / "global" / DATASET / f"pools-{env.pools_sha}.json")
        view = gd._read(env.data / "global" / DATASET / f"hpo-view-{self.view['hpo_view_sha256']}.json")
        artifact = ge.build_global_predictions(cfg, view, env.records, scores, pools=pools,
                                               epoch=cfg["training"]["epochs"])
        summary = ge.validate_global_predictions(cfg, view, env.records, artifact, pools=pools)
        out = env.runs / job["job_id"] / claimed["attempt_id"]
        out.mkdir(parents=True)
        (out / "predictions.json").write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
        (out / "results.json").write_text(json.dumps({"clinical_evaluation": {"artifact": ge.ARTIFACT_NAME, **{
            k: summary[k] for k in ("partition", "samples", "primary_metric", "primary_value", "primary_reason",
                                    "content_sha256")}}}), encoding="utf-8")
        mount = {"kind": "d_view", "source": str((env.views / env.pools_sha / "D").resolve()), "destination": "/data",
                 "readonly": True, "pools_sha256": env.pools_sha, "hpo_view_sha256": view["sha256"],
                 "view_files_sha256": view["view_files_sha256"]}
        (out / "runtime.json").write_text(json.dumps({"data_mount": mount, "image_id": image}), encoding="utf-8")
        (out / "completed.json").write_text(json.dumps({
            "status": "completed", "phase": "hpo", "config_sha256": spec["config_sha256"],
            "splits_sha256": view["sha256"], "trainer_override": None,
            "artifacts": {ge.ARTIFACT_NAME: hashlib.sha256((out / "predictions.json").read_bytes()).hexdigest()}}),
            encoding="utf-8")
        (out / "docker.log").write_text("fixture docker log\n", encoding="utf-8")
        queue.update(job["job_id"], running["attempt_id"], running["token"], running["version"],
                     {"event_id": "end-" + claimed["attempt_id"], "state": end_state})

    def fail_hpo(self, queue, job):
        """Play an HPO attempt that fails before writing any predictions (retry policy allows a retry)."""
        claimed = queue.claim(job["job_id"], {"server": "fixture"})
        running = queue.update(job["job_id"], claimed["attempt_id"], claimed["token"], claimed["version"],
                               {"event_id": "run", "state": "running"})
        return queue.update(job["job_id"], running["attempt_id"], running["token"], running["version"],
                            {"event_id": "end", "state": "failed"})

    def run_hpo(self, image=IMAGE):
        queue = GitQueue(self.env.checkout, self.env.remote)
        for job in queue.list_jobs():
            if job["spec"]["phase"] == "hpo":
                candidate = job["spec"]["global"]["candidate_id"]
                self.finish_hpo(queue, job, GOOD if candidate == "c01" else TIE, image=image)

    def collect_args(self):
        env = self.env
        return env.base("collect-hpo") + self.collection_tail()

    def collection_tail(self):
        env = self.env
        return ["--code", str(env.code), *self.pins(), *env.queue_args(), "--image-id", IMAGE,
                "--environment-probe", str(env.probe), "--runs", str(env.runs), "--views-root", str(env.views),
                "--configs-commit", self.seeded["code_commit"], "--host-validation", "minimal"]

    def seal_freeze(self, evidence_sha256, code=0):
        return self.run_cli(self.env.base("seal-freeze") + self.collection_tail() + [
            "--evidence-sha256", evidence_sha256, "--frozen-at", "2026-10-23T14:00:00Z"], code=code)


class HpoPathTests(HpoCase):
    def test_hpo_campaign_end_to_end(self):
        env = self.env
        seeded = self.to_seeded_hpo()
        self.assertEqual(seeded["jobs"], 4)
        self.assertNotEqual(seeded["code_commit"], self.plan["plan_commit"])  # B1: configs after the plan
        queue = GitQueue(env.checkout, env.remote)
        self.assertEqual({j["spec"]["code_commit"] for j in queue.list_jobs()}, {seeded["code_commit"]})
        self.run_hpo()
        collected = self.run_cli(self.collect_args())
        self.assertEqual(collected["status_counts"], {"completed": 4})
        freeze = self.seal_freeze(collected["evidence_sha256"])
        self.assertEqual(freeze["selected_candidate_id"], "c01")  # by the predeclared ranking
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "recipe_freeze",
                                           "--freeze-kind", "hpo_d", "--document-sha256", freeze["freeze_sha256"],
                                           *self.collection_tail()])
        freeze_args = ["--freeze-sha256", freeze["freeze_sha256"], "--freeze-kind", "hpo_d", "--stages", "11;13,19"]
        made = self.run_cli(env.base("materialize-final") + ["--code", str(env.code), *freeze_args])
        self.assertEqual(made["configs"], 50)
        git(env.code, "add", "-A"); git(env.code, "commit", "-qm", "final configs (hpo_d)")
        campaign = self.run_cli(env.base("seal-campaign") + [
            "--code", str(env.code), *freeze_args, *env.queue_args(), "--image-id", IMAGE,
            "--environment-probe", str(env.probe),
            "--sealed-at", "2026-10-23T15:00:00Z", "--search-space-sha256", self.space["search_space_sha256"]])
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "campaign",
                                           "--document-sha256", campaign["campaign_sha256"]])
        seeded_final = self.run_cli(env.base("seed-final") + self.campaign_args(campaign))
        self.assertEqual(seeded_final["jobs"], 50)
        chronology = self.run_cli(env.base("audit-chronology") + self.campaign_args(campaign))
        self.assertEqual((chronology["hpo_jobs"], chronology["final_jobs"], chronology["recipe_source"]),
                         (4, 50, "hpo_d"))
        finals = {(j["spec"]["arm"], j["spec"]["fold"]): j for j in GitQueue(env.checkout, env.remote).list_jobs()
                  if j["spec"]["phase"] == "final"}
        for fold in range(5):
            self.finish(queue, finals[("none", fold)], "succeeded", initial=H(["state", fold, 11]))
        report = self.run_cli(env.base("report") + self.campaign_args(campaign) +
                              ["--runs", str(env.runs), "--host-validation", "minimal"])
        self.assertEqual(report["status_counts"], {"succeeded": 5, "missing": 45})
        stored = gd._read(Path(report["path"]))
        self.assertEqual((stored["recipe_source"], stored["scientific_eligible"], stored["execution_verified"]),
                         ("hpo_d", False, False))

    def test_code_change_after_the_plan_blocks_seeding(self):
        env = self.env
        self.import_pools()
        self.view = self.run_cli(env.base("hpo-view"))
        spec = Path(env.data).parent / "search-space-spec.json"
        spec.write_text(json.dumps(SPEC), encoding="utf-8")
        self.space = self.run_cli(env.base("search-space") + self.runtime_args() +
                                  ["--spec", str(spec), "--color", "grayscale_to_rgb"])
        self.plan = self.run_cli(env.base("build-plan") + self.runtime_args() + [
            "--search-space-sha256", self.space["search_space_sha256"],
            "--view-sha256", self.view["hpo_view_sha256"], "--final-seeds", "11,13,19"])
        self.run_cli(env.base("materialize-hpo") + ["--code", str(env.code), *self.pins()])
        (env.code / "src" / "code.py").write_text("print('changed after the plan')\n")
        git(env.code, "add", "-A"); git(env.code, "commit", "-qm", "hpo configs + code change")
        error = self.run_cli(env.base("seed-hpo") + ["--code", str(env.code), *self.pins(), *env.queue_args(),
                                                     "--image-id", IMAGE, "--environment-probe", str(env.probe)], code=2)
        self.assertTrue(error.startswith("code_identity_mismatch:"), error)
        self.assertEqual(GitQueue(env.checkout, env.remote).list_jobs(), [])

    def test_edited_committed_config_blocks_seeding(self):
        env = self.env
        self.import_pools()
        self.view = self.run_cli(env.base("hpo-view"))
        spec = Path(env.data).parent / "search-space-spec.json"
        spec.write_text(json.dumps(SPEC), encoding="utf-8")
        self.space = self.run_cli(env.base("search-space") + self.runtime_args() +
                                  ["--spec", str(spec), "--color", "grayscale_to_rgb"])
        self.plan = self.run_cli(env.base("build-plan") + self.runtime_args() + [
            "--search-space-sha256", self.space["search_space_sha256"],
            "--view-sha256", self.view["hpo_view_sha256"], "--final-seeds", "11,13,19"])
        self.run_cli(env.base("materialize-hpo") + ["--code", str(env.code), *self.pins()])
        config = sorted((env.code / gj.config_dir(DATASET, "hpo")).glob("c0*.json"))[0]
        config.write_text(config.read_text().replace('"epochs": 20', '"epochs": 19'))
        git(env.code, "add", "-A"); git(env.code, "commit", "-qm", "edited hpo config")
        error = self.run_cli(env.base("seed-hpo") + ["--code", str(env.code), *self.pins(), *env.queue_args(),
                                                     "--image-id", IMAGE, "--environment-probe", str(env.probe)], code=2)
        self.assertTrue(error.startswith("config_identity_mismatch:"), error)

    def test_attempt_on_another_image_fails_collection(self):
        self.to_seeded_hpo()
        self.run_hpo(image="sha256:" + "a" * 64)
        error = self.run_cli(self.collect_args(), code=2)
        self.assertTrue(error.startswith("code_identity_mismatch:"), error)

    def test_exposed_dataset_needs_the_external_campaign_before_hpo(self):
        env = self.env
        candidate = json.loads(env.candidate.read_text())
        candidate["grouping"]["prior_exposure"] = "known"
        env.candidate.write_text(json.dumps(candidate))
        env.manifest_sha = gd.global_manifest_sha256(env.records, candidate)
        with self.assertRaises(AssertionError) as caught:  # run_cli asserts exit 0; capture the refusal instead
            self.to_seeded_hpo()
        self.assertIn("external_recipe_contaminated", str(caught.exception))
        self.assertEqual(GitQueue(env.checkout, env.remote).list_jobs(), [])

    def test_ranking_is_not_operator_choice_and_a_second_hpo_freeze_is_refused(self):
        env = self.env
        self.to_seeded_hpo()
        self.run_hpo()
        collected = self.run_cli(self.collect_args())
        freeze = self.seal_freeze(collected["evidence_sha256"])
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "recipe_freeze",
                                           "--freeze-kind", "hpo_d", "--document-sha256", freeze["freeze_sha256"],
                                           *self.collection_tail()])
        error = self.run_cli(env.base("seed-hpo") + ["--code", str(env.code), *self.pins(), *env.queue_args(),
                                                     "--image-id", IMAGE, "--environment-probe", str(env.probe)], code=2)
        self.assertTrue(error.startswith("hpo_job_after_freeze:"), error)
        from augment_atlas import global_cli as cli
        parser = cli._parser()
        seal_freeze = parser._subparsers._group_actions[0].choices["seal-freeze"]
        options = {o for action in seal_freeze._actions for o in action.option_strings}
        self.assertFalse({o for o in options if "candidate" in o or "winner" in o or "select" in o})


class FreezeSnapshotTests(HpoCase):
    """Audit blocker B-1: the freeze can only come from the evidence of the current, complete queue."""

    def hpo_jobs(self):
        queue = GitQueue(self.env.checkout, self.env.remote)
        return queue, {j["spec"]["global"]["candidate_id"]: j for j in queue.list_jobs() if j["spec"]["phase"] == "hpo"}

    def test_partial_collection_is_refused_and_publishes_nothing(self):
        self.to_seeded_hpo()
        queue, jobs = self.hpo_jobs()
        for candidate in ("c00", "c02", "c03"):
            self.finish_hpo(queue, jobs[candidate], TIE)
        error = self.run_cli(self.collect_args(), code=2)
        self.assertTrue(error.startswith("collection_incomplete:"), error)
        self.assertFalse(list((self.env.data / "global" / DATASET).glob("evidence-*.json")))

    def retry_and_finish_c01(self, queue, jobs, failed):
        queue.retry(jobs["c01"]["job_id"], failed["version"], "infrastructure failure before predictions")
        current = next(j for j in queue.list_jobs() if j["job_id"] == jobs["c01"]["job_id"])
        self.finish_hpo(queue, current, GOOD)

    def test_the_first_complete_collection_is_binding_a_later_retry_is_refused(self):
        self.to_seeded_hpo()
        queue, jobs = self.hpo_jobs()
        for candidate in ("c00", "c02", "c03"):
            self.finish_hpo(queue, jobs[candidate], TIE)
        failed = self.fail_hpo(queue, jobs["c01"])
        first = self.run_cli(self.collect_args())  # complete: c01 failed -> its row is explicit, never zero
        self.assertEqual(first["status_counts"], {"completed": 3, "failed": 1})
        self.retry_and_finish_c01(queue, jobs, failed)  # decided after the scores of the others were visible
        error = self.run_cli(self.collect_args(), code=2)
        self.assertTrue(error.startswith("collection_invalid:"), error)
        for evidence in (first["evidence_sha256"],):
            error = self.seal_freeze(evidence, code=2)
            self.assertTrue(error.startswith("collection_invalid:"), error)

    def test_a_freeze_sealed_before_a_retry_cannot_be_anchored(self):
        env = self.env
        self.to_seeded_hpo()
        queue, jobs = self.hpo_jobs()
        for candidate in ("c00", "c02", "c03"):
            self.finish_hpo(queue, jobs[candidate], TIE)
        failed = self.fail_hpo(queue, jobs["c01"])
        first = self.run_cli(self.collect_args())
        stale = self.seal_freeze(first["evidence_sha256"])
        self.retry_and_finish_c01(queue, jobs, failed)
        error = self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind",
                                                   "recipe_freeze", "--freeze-kind", "hpo_d", "--document-sha256",
                                                   stale["freeze_sha256"], *self.collection_tail()], code=2)
        self.assertTrue(error.startswith("collection_invalid:"), error)
        error = self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind",
                                                   "recipe_freeze", "--freeze-kind", "hpo_d", "--document-sha256",
                                                   stale["freeze_sha256"]], code=2)
        self.assertTrue(error.startswith("cli_input_invalid:"), error)  # hpo_d anchors need the collection

    def test_a_failed_attempt_that_already_wrote_scores_is_refused(self):
        # Audit B-1'': a job stopped after writing a (winning) score must not become a plain failed row.
        self.to_seeded_hpo()
        queue, jobs = self.hpo_jobs()
        for candidate in ("c00", "c02", "c03"):
            self.finish_hpo(queue, jobs[candidate], TIE)
        for end_state in ("failed",):
            self.finish_hpo(queue, jobs["c01"], GOOD, end_state=end_state)
        error = self.run_cli(self.collect_args(), code=2)
        self.assertTrue(error.startswith("collection_invalid:"), error)
        self.assertIn("already wrote scores", error)

    def test_an_edited_pin_is_refused(self):
        self.to_seeded_hpo()
        self.run_hpo()
        self.run_cli(self.collect_args())
        pin = next((self.env.data / "global" / DATASET).glob("hpo-evidence-pin-*.json"))
        document = json.loads(pin.read_text())
        document["evidence_sha256"] = "0" * 64  # checksum left stale
        pin.write_text(json.dumps(document))
        error = self.run_cli(self.collect_args(), code=2)
        self.assertIn("collection_invalid: the HPO evidence pin fails its checksum", error)

    def test_a_second_seeding_at_another_configs_commit_is_refused(self):
        env = self.env
        self.to_seeded_hpo()
        (env.code / "configs" / "global" / "README-note.txt").write_text("configs-only change\n")
        git(env.code, "add", "-A"); git(env.code, "commit", "-qm", "configs-only follow-up")
        error = self.run_cli(env.base("seed-hpo") + ["--code", str(env.code), *self.pins(), *env.queue_args(),
                                                     "--image-id", IMAGE, "--environment-probe", str(env.probe)],
                             code=2)
        self.assertTrue(error.startswith("search_space_mismatch:"), error)

    def test_another_image_or_probe_than_the_plan_is_refused_at_seed(self):
        env = self.env
        self.to_seeded_hpo()
        for image in ("sha256:" + "a" * 64,):
            error = self.run_cli(env.base("seed-hpo") + ["--code", str(env.code), *self.pins(), *env.queue_args(),
                                                         "--image-id", image, "--environment-probe", str(env.probe)],
                                 code=2)
            self.assertTrue(error.startswith("code_identity_mismatch:"), error)
        other_probe = Path(env.data).parent / "probe-other.json"
        other_probe.write_text(json.dumps({"torch": "other", "cuda": None}))
        error = self.run_cli(env.base("seed-hpo") + ["--code", str(env.code), *self.pins(), *env.queue_args(),
                                                     "--image-id", IMAGE, "--environment-probe", str(other_probe)],
                             code=2)
        self.assertTrue(error.startswith("code_identity_mismatch:"), error)


class WorkerPhaseHookTests(unittest.TestCase):
    """cli.py hook: --phase hpo|final reaches the runtime, whose guards still fail before any Docker/Git call."""

    def run_worker(self, cli_phase, **settings):
        import tempfile
        from augment_atlas import cli
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker.json"
            path.write_text(json.dumps({"code": tmp, "data": tmp, **settings}))
            cli.main(["worker", "--settings", str(path), "--phase", cli_phase])

    def test_phase_must_equal_settings_phase(self):
        with self.assertRaisesRegex(ValueError, "Settings phase must match"):
            self.run_worker("final", phase="hpo")

    def test_final_worker_without_campaign_pin_and_hpo_worker_without_views_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
            self.run_worker("final", phase="final")
        with self.assertRaisesRegex(ValueError, "^hpo_isolation_unproven:"):
            self.run_worker("hpo", phase="hpo")


class InbreastOrderTests(HpoCase):
    """Decision 7: external campaign (a) sealed and anchored before any HPO job; both campaigns before finals."""

    def test_two_exploratory_campaigns_in_the_documented_order(self):
        env = self.env
        candidate = json.loads(env.candidate.read_text())
        candidate["grouping"]["prior_exposure"] = "known"  # exposed dataset, like INbreast
        env.candidate.write_text(json.dumps(candidate))
        env.manifest_sha = gd.global_manifest_sha256(env.records, candidate)
        _, campaign_a = self.to_campaign()  # seal-external, anchor, final configs (a), seal campaign (a)
        self.assertEqual(campaign_a["label"], "exploratory")
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "campaign",
                                           "--document-sha256", campaign_a["campaign_sha256"]])
        commit_a = git(env.code, "rev-parse", "HEAD")
        error = self.run_cli(env.base("seed-final") + self.campaign_args(campaign_a), code=2)
        self.assertTrue(error.startswith("final_job_before_seal:"), error)  # campaign (b) not anchored yet
        self.view = self.run_cli(env.base("hpo-view"))
        self.run_cli(env.base("d-view") + ["--view-sha256", self.view["hpo_view_sha256"],
                                           "--views-root", str(env.views)])
        spec = Path(env.data).parent / "search-space-spec.json"
        spec.write_text(json.dumps(SPEC), encoding="utf-8")
        self.space = self.run_cli(env.base("search-space") + self.runtime_args() +
                                  ["--spec", str(spec), "--color", "grayscale_to_rgb"])
        self.plan = self.run_cli(env.base("build-plan") + self.runtime_args() + [
            "--search-space-sha256", self.space["search_space_sha256"],
            "--view-sha256", self.view["hpo_view_sha256"], "--final-seeds", "11,13,19"])
        self.run_cli(env.base("materialize-hpo") + ["--code", str(env.code), *self.pins()])
        git(env.code, "add", "-A"); git(env.code, "commit", "-qm", "hpo configs")
        self.seeded = self.run_cli(env.base("seed-hpo") + ["--code", str(env.code), *self.pins(),
                                                           *env.queue_args(), "--image-id", IMAGE,
                                                           "--environment-probe", str(env.probe)])
        self.run_hpo()
        collected = self.run_cli(self.collect_args())
        freeze = self.seal_freeze(collected["evidence_sha256"])
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "recipe_freeze",
                                           "--freeze-kind", "hpo_d", "--document-sha256", freeze["freeze_sha256"],
                                           *self.collection_tail()])
        # One final materialization per commit: replace campaign (a)'s final configs in a configs-only commit.
        git(env.code, "rm", "-rq", gj.config_dir(DATASET, "final"))
        freeze_args = ["--freeze-sha256", freeze["freeze_sha256"], "--freeze-kind", "hpo_d", "--stages", "11;13,19"]
        git(env.code, "commit", "-qm", "drop campaign (a) final configs")
        self.run_cli(env.base("materialize-final") + ["--code", str(env.code), *freeze_args])
        git(env.code, "add", "-A"); git(env.code, "commit", "-qm", "final configs (b)")
        campaign_b = self.run_cli(env.base("seal-campaign") + [
            "--code", str(env.code), *freeze_args, *env.queue_args(), "--image-id", IMAGE,
            "--environment-probe", str(env.probe),
            "--sealed-at", "2026-10-23T15:00:00Z", "--search-space-sha256", self.space["search_space_sha256"]])
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "campaign",
                                           "--document-sha256", campaign_b["campaign_sha256"]])
        commit_b = git(env.code, "rev-parse", "HEAD")
        # Each campaign's finals are seeded from its own sealed commit.
        git(env.code, "checkout", "-q", commit_a)
        self.assertEqual(self.run_cli(env.base("seed-final") + self.campaign_args(campaign_a))["jobs"], 50)
        error = self.run_cli(env.base("seed-final") + self.campaign_args(campaign_b), code=2)
        self.assertTrue(error.startswith(("code_identity_mismatch:", "config_identity_mismatch:")), error)
        git(env.code, "checkout", "-q", commit_b)
        self.assertEqual(self.run_cli(env.base("seed-final") + self.campaign_args(campaign_b))["jobs"], 50)
        for campaign, commit in ((campaign_a, commit_a), (campaign_b, commit_b)):
            git(env.code, "checkout", "-q", commit)
            report = self.run_cli(env.base("audit-chronology") + self.campaign_args(campaign))
            self.assertEqual((report["final_jobs"], report["other_campaign_final_jobs"]), (50, 50))
        queue = GitQueue(env.checkout, env.remote)
        reports = {}
        for campaign, commit in ((campaign_a, commit_a), (campaign_b, commit_b)):
            git(env.code, "checkout", "-q", commit)
            jobs = [j for j in queue.list_jobs() if j["spec"]["phase"] == "final"
                    and j["spec"]["global"]["campaign_sha256"] == campaign["campaign_sha256"]
                    and j["spec"]["arm"] == "none"]
            for job in jobs:
                self.finish(queue, job, "succeeded", initial=H(["state", job["spec"]["fold"], 11, commit]))
            reports[commit] = self.run_cli(env.base("report") + self.campaign_args(campaign) +
                                           ["--runs", str(env.runs), "--host-validation", "minimal"])
        comparison = self.run_cli(env.base("compare") + [
            "--report-a", reports[commit_a]["report_sha256"], "--report-b", reports[commit_b]["report_sha256"],
            "--out", str(Path(env.data).parent / "comparison")])
        written = json.loads((Path(comparison["out"]) / "campaign_comparison.json").read_text())
        self.assertIs(written["superiority_claim"], False)


if __name__ == "__main__":
    unittest.main()
