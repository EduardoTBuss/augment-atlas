"""Slice S1 (part C): global hpo/final branch of the host runtime.

Docker, GPU, Git and the queue are MOCKS (same pattern as test_runtime.py).
A private host data root, a pinned code checkout with materialized configs +
MANIFEST.json, an anchored plan, an hpo-view and a materialized D-view
(views/<pools_sha256>/D) are synthetic; images are placeholder bytes and the
checkpoints are opaque placeholders (no torch). Attempt outputs are written
the way ``train._run_global`` writes them; every mutation is RE-SIGNED
(completed.json checksums, content_sha256, config_sha256/MANIFEST) so that
nothing is rejected merely by a stale checksum.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import Mock, patch

from augment_atlas import global_evaluation as ge
from augment_atlas import global_jobs as gj
from augment_atlas import runtime
from augment_atlas.contracts import canonical_hash as H, required_artifacts
from augment_atlas.storage import atomic_json, digest
from test_global_trainer import DATASET, HPO_SEED, GlobalFixture, final_policy, sha256_bytes, signed

COMMIT = "b" * 40
CAMPAIGN = H("campaign fixture")
EPOCHS = 2


def checkpoint_bytes():
    """Opaque zip shaped like a torch archive (data.pkl + data/), never unpickled."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("archive/data.pkl", b"opaque placeholder, not a pickle")
        archive.writestr("archive/data/0", b"opaque tensor bytes")
    return stream.getvalue()


class World:
    """Host data root + code checkout + views root for one HPO and one final job."""

    def __init__(self, root):
        self.root = Path(root)
        self.fx = GlobalFixture(self.root, epochs=EPOCHS)
        self.data = self.fx.root
        self.code = self.root / "code"
        self.views = self.root / "views"
        pools = self.fx.pools
        self.plan = {
            "schema_version": 1, "dataset": copy.deepcopy(pools["dataset"]), "pools_sha256": pools["sha256"],
            "task": copy.deepcopy(pools["task"]), "endpoint": pools["task"]["endpoint"], "direction": "maximize",
            "candidates": [{"id": "c00", "parameters": {"lr": 0.0001}, "recipe": copy.deepcopy(self.fx.recipe),
                            "augmentation": {"arm": "none", "ops": []}}],
            "seeds": [HPO_SEED], "ranking": "mean_then_candidate_id", "failure_policy": "require_all_defined",
            "epoch_policy": "fixed_in_candidate_before_scores", "threshold_policy": "fixed_in_candidate_before_scores",
            "producer": {"id": "augment-atlas-global-hpo-v1", "code_sha256": H("code"),
                         "environment_sha256": self.fx.runtime["environment_sha256"],
                         "manifest_sha256": H("listing"), "logs_sha256": H("logs")},
            "runtime": copy.deepcopy(self.fx.runtime), "final_policy": final_policy()}
        self.plan_sha = H(self.plan)
        (self.data / "global" / DATASET / f"plan-{self.plan_sha}.json").write_text(
            json.dumps(self.plan, indent=2, sort_keys=True), encoding="utf-8")
        self.dview = self.views / pools["sha256"] / "D"
        d_ids = set(pools["lists"]["D"])
        for row in self.fx.rows:
            if row["id"] in d_ids:
                target = self.dview / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.data / row["path"], target)
        listing = sorted([row["path"], digest(self.dview / row["path"])] for row in self.fx.rows if row["id"] in d_ids)
        view = copy.deepcopy(self.fx.hpo_view)
        view["view_files_sha256"] = H(listing)
        self.view = signed(view)
        self.view_file = f"global/{DATASET}/hpo-view-{self.view['sha256']}.json"
        for base in (self.data, self.dview):
            self.write_json(base / self.view_file, self.view)
        manifest = (self.data / "manifest.csv").read_bytes()
        self.source = {"adapter": "manifest_csv", "manifest": "manifest.csv", "manifest_sha256": sha256_bytes(manifest)}
        hpo = gj.build_hpo_configs(self.fx.rows, pools, expected_pools_sha256=pools["sha256"],
                                   dataset_source=self.source, plan=self.plan, expected_plan_sha256=self.plan_sha,
                                   search_space_sha256=H("search space"), hpo_view_sha256=self.view["sha256"])
        gj.materialize_configs("hpo", self.code, hpo, pools_sha256=pools["sha256"], plan_sha256=self.plan_sha)
        gj.materialize_configs("final", self.code, self.fx.final_cfgs, pools_sha256=pools["sha256"],
                               freeze_sha256=self.fx.freeze["sha256"])
        self.hpo_name = f"configs/global/{DATASET}/hpo/c00-s{HPO_SEED}.json"
        self.final_name = f"configs/global/{DATASET}/final/none-f0-s11.json"
        cache = self.root / "torch-home" / "hub" / "checkpoints" / runtime.IMAGENET_V2_CHECKPOINT
        cache.parent.mkdir(parents=True)
        cache.write_bytes(self.fx.weights)
        self.settings = {"code": str(self.code), "data": str(self.data), "views": str(self.views),
                         "phase": "hpo", "gpu": "0", "torch_cache": str(self.root / "torch-home")}

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

    def cfg(self, phase):
        return json.loads((self.code / (self.hpo_name if phase == "hpo" else self.final_name)).read_text())

    def spec(self, phase="hpo"):
        name = self.hpo_name if phase == "hpo" else self.final_name
        cfg, sha = self.cfg(phase), digest(self.code / name)
        if phase == "hpo":
            return gj.hpo_job_spec(cfg, config_path=name, config_sha256=sha, code_commit=COMMIT)
        return gj.final_job_spec(cfg, self.fx.final_view(0), config_path=name, config_sha256=sha,
                                 code_commit=COMMIT, campaign_sha256=CAMPAIGN)

    def docker_runtime(self, phase="hpo"):
        host = runtime.DockerRuntime({**self.settings, "phase": phase})
        host.commit, host.image, host.environment = COMMIT, "sha256:image-fixture", {"opencv": None}
        return host

    # ------------------------------------------------------------------ attempt
    def write_attempt(self, out, phase="hpo", *, override=None):
        """Write the nine global artifacts exactly as train._run_global does."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg(phase)
        name = self.hpo_name if phase == "hpo" else self.final_name
        config_bytes = (self.code / name).read_bytes()
        config_sha = sha256_bytes(config_bytes)
        if phase == "hpo":
            view, partition = self.view, self.fx.pools["lists"]["D_select"]
        else:
            view = self.fx.final_view(0)
            partition = view["test"]
        labels = {row["id"]: row["label"] for row in self.fx.rows}
        scores = [[0.2, 0.8] if index % 3 else [0.6, 0.4] for index, _ in enumerate(partition)]
        if phase == "hpo":
            artifact = ge.build_hpo_predictions_from_view(cfg, view, scores, epoch=EPOCHS)
        else:
            artifact = ge.build_global_predictions(cfg, view, self.fx.rows, scores, pools=self.fx.pools, epoch=EPOCHS)
        correct = sum(max(range(2), key=lambda c: s[c]) == labels[i] for i, s in zip(partition, scores))
        evaluated = "test" if phase == "final" else "selection"
        split = view["sha256"]
        common = {"config_sha256": config_sha, "splits_sha256": split}
        initial = H("initial state")
        train_images = len(view["lists"]["train"]) if phase == "hpo" else len(view["train"])
        environment = {"schema_version": 1, "utc": "2026-10-19T12:00:00+00:00", "phase": phase,
                       "protocol_version": cfg["protocol_version"], "device": "cuda", "workers": 2,
                       "trainer_override": override, "trainer_constants": copy.deepcopy(gj.TRAINER_CONSTANTS_V1),
                       "train_images": train_images, "evaluated_partition": ge.PARTITIONS[phase],
                       "evaluated_images": len(partition), "weights": "IMAGENET1K_V2",
                       "weights_sha256": cfg["model"]["weights_sha256"], "initial_state_sha256": initial,
                       "code_commit": COMMIT, "image_id": "sha256:image-fixture",
                       "global": {k: cfg["global"][k] for k in ("pools_sha256", "hpo_view_sha256", "plan_sha256",
                                                                "candidate_id", "freeze_kind", "freeze_sha256", "fold")},
                       "dataset_integrity": ({"source": "hpo_view", "records": len(view["records"]),
                                              "view_sha256": split, "view_files_sha256": view["view_files_sha256"]}
                                             if phase == "hpo" else {"source": "manifest"}),
                       **common}
        primary = artifact["metrics"]["primary"]
        results = {"schema_version": 1, "phase": phase, "scientific_status": "provisional",
                   "protocol_version": cfg["protocol_version"], "checkpoint_selection": "last_epoch",
                   "best_epoch": EPOCHS, "epochs": EPOCHS, "initial_state_sha256": initial,
                   "trainer_override": override,
                   evaluated: {"loss": 0.5, "accuracy": correct / len(partition), "samples": len(partition)},
                   f"{evaluated}_evaluations": 1,
                   "clinical_evaluation": {"artifact": "predictions.json", "partition": artifact["partition"],
                                           "samples": len(artifact["rows"]), "primary_metric": primary["metric"],
                                           "primary_value": primary["value"], "primary_reason": primary["reason"],
                                           "content_sha256": artifact["content_sha256"]},
                   "execution_verified": False, "scientific_eligible": False, **common}
        metrics = [{"epoch": e, "utc": "2026-10-19T12:00:00+00:00", "train_loss": 0.4, "samples_processed": 4 * e,
                    "optimizer_updates": 1, "learning_rate": 0.0001} for e in range(1, EPOCHS + 1)]
        (out / "config.json").write_bytes(config_bytes)
        atomic_json(out / "splits.json", view)
        atomic_json(out / "environment.json", environment)
        atomic_json(out / "metrics.json", metrics)
        atomic_json(out / "results.json", results)
        atomic_json(out / "predictions.json", artifact)
        atomic_json(out / "progress.json", {"schema_version": 1, "status": "completed", "phase": phase,
                                            "epoch": EPOCHS, "epochs": EPOCHS, "samples_processed": 4 * EPOCHS,
                                            "checkpoint": "checkpoint.pt", **common})
        (out / "checkpoint.pt").write_bytes(checkpoint_bytes())
        (out / "best.pt").write_bytes(b"opaque best placeholder")
        mount = (dict(kind="d_view", source=str(self.dview.resolve()), destination="/data", readonly=True,
                      pools_sha256=self.fx.pools["sha256"], hpo_view_sha256=self.view["sha256"],
                      view_files_sha256=self.view["view_files_sha256"]) if phase == "hpo" else
                 dict(kind="data_root", source=str(self.data.resolve()), destination="/data", readonly=True))
        atomic_json(out / "runtime.json", {"job_id": "a" * 64, "attempt_id": "attempt-fixture",
                                           "code_commit": COMMIT, "data_mount": mount})
        self.completed = {"schema_version": 1, "status": "completed", "phase": phase, "epoch": EPOCHS,
                          "trainer_override": override, **common}
        self.resign(out)
        return out

    def resign(self, out):
        names = required_artifacts(self.cfg("hpo"))
        completed = dict(self.completed, artifacts={n: digest(Path(out) / n) for n in names})
        atomic_json(Path(out) / "completed.json", completed)

    def edit(self, out, name, change):
        path = Path(out) / name
        value = json.loads(path.read_text(encoding="utf-8"))
        change(value)
        atomic_json(path, value)
        self.resign(out)


def resign_predictions(artifact):
    artifact["content_sha256"] = H({k: v for k, v in artifact.items() if k != "content_sha256"})


class GlobalRuntimeTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="atlas-global-runtime-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.w = World(self.root)

    def create_args(self, host, spec, out):
        with patch("augment_atlas.runtime.command") as docker, \
             patch("augment_atlas.runtime.os.getuid", return_value=1000, create=True), \
             patch("augment_atlas.runtime.os.getgid", return_value=1000, create=True):
            host.create("container-fixture", spec, out)
        return docker.call_args.args[0]

    @staticmethod
    def mounts(args):
        return [args[i + 1] for i, arg in enumerate(args) if arg == "--mount"]


class HpoLaunchTests(GlobalRuntimeTest):
    def test_hpo_config_for_checks_everything_on_the_host_without_torch(self):
        import sys
        torch_before = {n for n in sys.modules if n == "torch" or n.startswith("torch.")}
        with patch("augment_atlas.global_evaluation.check_hpo_view", wraps=ge.check_hpo_view) as checked:
            name, cfg = self.w.docker_runtime().config_for(self.w.spec())
        self.assertEqual(name, self.w.hpo_name)
        self.assertEqual(cfg["phase"], "hpo")
        checked.assert_called_once()  # host-side check WITH pools (E identities) before launch
        self.assertEqual(torch_before, {n for n in sys.modules if n == "torch" or n.startswith("torch.")})

    def test_mock_create_mounts_only_the_d_view_as_data_and_records_it(self):
        out = self.root / "runs" / "job" / "attempt"
        out.mkdir(parents=True)
        atomic_json(out / "runtime.json", {"job_id": "a" * 64})
        args = self.create_args(self.w.docker_runtime(), self.w.spec(), out)
        mounts = self.mounts(args)
        self.assertIn(f"type=bind,src={self.w.dview.resolve()},dst=/data,readonly", mounts)
        self.assertEqual([m for m in mounts if "dst=/data" in m], [f"type=bind,src={self.w.dview.resolve()},dst=/data,readonly"])
        self.assertFalse(any(f"src={self.w.data.resolve()}," in m for m in mounts), "full data root must not be mounted")
        self.assertEqual(sorted(m.split("dst=")[1].split(",")[0] for m in mounts), ["/code", "/data", "/output", "/torch-cache"])
        self.assertEqual("none", args[args.index("--network") + 1])
        recorded = json.loads((out / "runtime.json").read_text())
        self.assertEqual(recorded["job_id"], "a" * 64)
        self.assertEqual(recorded["data_mount"]["kind"], "d_view")
        self.assertEqual(recorded["data_mount"]["view_files_sha256"], self.w.view["view_files_sha256"])

    def assert_isolation(self, settings_change=None, mutate=None, code="hpo_isolation_unproven"):
        if mutate:
            mutate()
        host = self.w.docker_runtime()
        if settings_change:
            settings_change(host.s)
        with patch("augment_atlas.runtime.command") as docker:
            with self.assertRaisesRegex(ValueError, f"^{code}:"):
                host.create("container-fixture", self.w.spec(), self.root / "out")
            docker.assert_not_called()

    def test_hpo_without_d_view_mount_is_isolation_unproven_and_never_falls_back(self):
        self.assert_isolation(settings_change=lambda s: s.pop("views"))
        self.assert_isolation(settings_change=lambda s: s.update(views=str(self.root / "missing")))
        with patch("augment_atlas.runtime.command") as docker:
            with self.assertRaisesRegex(ValueError, "^hpo_isolation_unproven:"):
                runtime.DockerRuntime({**self.w.settings, "views": ""}).preflight()
            docker.assert_not_called()
        shutil.rmtree(self.w.dview)
        self.assert_isolation()

    def test_d_view_listing_is_recomputed_before_mount(self):
        e_image = next(r for r in self.w.fx.rows if r["id"] in self.w.fx.pools["lists"]["E"])
        extra = self.w.dview / e_image["path"]
        self.assert_isolation(mutate=lambda: shutil.copyfile(self.w.data / e_image["path"], extra))
        extra.unlink()
        stray = self.w.dview / "notes.txt"
        self.assert_isolation(mutate=lambda: stray.write_text("E ids could hide here"))
        stray.unlink()
        d_image = self.w.dview / self.w.view["records"][0]["path"]
        original = d_image.read_bytes()
        self.assert_isolation(mutate=d_image.unlink)
        d_image.write_bytes(b"different bytes")
        self.assert_isolation(code="records_identity_mismatch")
        d_image.write_bytes(original)
        listing = self.w.dview / runtime.D_VIEW_LISTING
        self.assert_isolation(mutate=lambda: listing.write_text("[]"))
        listing.unlink()
        listing.write_text(json.dumps(runtime.d_view_listing(self.w.dview, self.w.view_file)))
        self.w.docker_runtime().config_for(self.w.spec())  # a correct LISTING.json is accepted
        inside = self.w.dview / self.w.view_file
        changed = copy.deepcopy(self.w.view)
        changed["records"] = changed["records"][1:]
        self.assert_isolation(mutate=lambda: inside.write_text(json.dumps(signed(changed))), code="view_hash_mismatch")

    def test_hpo_view_with_e_identity_resigned_is_rejected_on_host_before_launch(self):
        view = copy.deepcopy(self.w.view)
        e_row = next(r for r in self.w.fx.rows if r["id"] in self.w.fx.pools["lists"]["E"])
        view["records"] = sorted(view["records"] + [copy.deepcopy(e_row)], key=lambda r: r["id"])
        (self.w.data / self.w.view_file).write_text(json.dumps(signed(view)))
        self.assert_isolation(code="hpo_view_contains_E")

    def test_invalid_hpo_job_fails_real_worker_selection_before_mock_claim(self):
        shutil.rmtree(self.w.dview)
        host = self.w.docker_runtime()
        host.gpu_uuid = "GPU-fixture"
        queue = Mock()
        queue.list_jobs.return_value = [{"job_id": "a" * 64, "state": "pending", "spec": self.w.spec()}]
        settings = {**self.w.settings, "runs": str(self.root / "runs"), "queue_checkout": str(self.root / "queue"),
                    "remote": "mock-remote", "server": "host-fixture"}
        with patch("augment_atlas.runtime.DockerRuntime", return_value=host), \
             patch.object(host, "preflight"), patch.object(host, "probe"), \
             patch.object(host, "compatible", return_value=True), \
             patch("augment_atlas.runtime.GitQueue", return_value=queue), \
             patch("augment_atlas.runtime.gpu_lock", return_value=contextlib.nullcontext()), \
             patch("augment_atlas.runtime.command") as docker:
            with self.assertRaisesRegex(ValueError, "^hpo_isolation_unproven:"):
                runtime.worker(settings, once=True)
        queue.claim.assert_not_called()
        docker.assert_not_called()

    def test_spec_identity_must_match_materialized_config(self):
        host = self.w.docker_runtime()
        for key, value in (("split_hash", H("other view")), ("recipe_hash", H("other candidate")),
                           ("train_seed", HPO_SEED + 1), ("arm", "basic")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "^global_config_invalid:"):
                host.config_for({**self.w.spec(), key: value})
        for change in ({"config": f"configs/global/{DATASET}/hpo/../hpo/c00-s{HPO_SEED}.json"},
                       {"config": f"configs/smoke/c00-s{HPO_SEED}.json", "phase": "hpo"},
                       {"config": f"configs/global/{DATASET}/hpo/MANIFEST.json"},
                       {"phase": "final"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "^global_config_invalid:"):
                host.config_for({**self.w.spec(), **change})


class ConfigTamperTests(GlobalRuntimeTest):
    def rewrite_config(self, cfg=None, data=None, manifest=True):
        path = self.w.code / self.w.hpo_name
        data = gj.render_config_bytes(cfg) if data is None else data
        path.write_bytes(data)
        spec = self.w.spec() if cfg is not None else None
        if manifest:
            target = self.w.code / f"configs/global/{DATASET}/hpo/MANIFEST.json"
            document = json.loads(target.read_text())
            document["files"] = [{"path": f["path"], "sha256": sha256_bytes(data) if f["path"] == self.w.hpo_name else f["sha256"]}
                                 for f in document["files"]]
            target.write_bytes(gj.render_config_bytes(signed(document)))
        return spec

    def test_config_tampered_and_resigned_in_spec_only_fails_manifest(self):
        cfg = self.w.cfg("hpo")
        cfg["training"]["optimizer"]["lr"] = 0.001
        path = self.w.code / self.w.hpo_name
        path.write_bytes(gj.render_config_bytes(cfg))
        spec = {**self.w.spec(), "config_sha256": digest(path)}
        with self.assertRaisesRegex(ValueError, "^config_identity_mismatch:"):
            self.w.docker_runtime().config_for(spec)

    def test_config_tampered_with_manifest_and_spec_resigned_fails_anchored_recipe(self):
        cfg = self.w.cfg("hpo")
        cfg["training"]["optimizer"]["lr"] = 0.001
        self.rewrite_config(cfg)
        spec = {**self.w.spec(), "config_sha256": digest(self.w.code / self.w.hpo_name)}
        with self.assertRaisesRegex(ValueError, "^recipe_mismatch:"):
            self.w.docker_runtime().config_for(spec)

    def test_noncanonical_bytes_resigned_are_rejected(self):
        cfg = self.w.cfg("hpo")
        data = json.dumps(cfg).encode("ascii")
        spec = self.w.spec()
        self.rewrite_config(data=data)
        spec["config_sha256"] = sha256_bytes(data)
        with self.assertRaisesRegex(ValueError, "^config_identity_mismatch:"):
            self.w.docker_runtime().config_for(spec)

    def test_stale_checksum_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "^config_identity_mismatch:"):
            self.w.docker_runtime().config_for({**self.w.spec(), "config_sha256": "0" * 64})

    def test_attempt_config_json_tampered_and_resigned_is_rejected(self):
        out = self.w.write_attempt(self.root / "out")
        cfg = self.w.cfg("hpo")
        cfg["training"]["epochs"] = EPOCHS + 1
        (out / "config.json").write_bytes(gj.render_config_bytes(cfg))
        self.w.resign(out)
        with self.assertRaisesRegex(ValueError, "^config_identity_mismatch:"):
            runtime.validate_completion(self.w.settings, out, self.w.spec())


class CompletionTests(GlobalRuntimeTest):
    def test_valid_hpo_attempt_is_validated_with_host_trusted_inputs(self):
        out = self.w.write_attempt(self.root / "out")
        with patch("augment_atlas.global_evaluation.validate_global_predictions",
                   wraps=ge.validate_global_predictions) as recomputed:
            summary = runtime.validate_completion(self.w.settings, out, self.w.spec())
        self.assertEqual(summary["artifact_count"], 9)
        self.assertEqual((summary["epoch"], summary["epochs"]), (EPOCHS, EPOCHS))
        _, kwargs = recomputed.call_args
        self.assertEqual(kwargs["pools"]["sha256"], self.w.fx.pools["sha256"])  # host pools, not the attempt

    def test_hpo_completion_without_recorded_d_view_mount_is_isolation_unproven(self):
        out = self.w.write_attempt(self.root / "out")
        for change in (lambda r: r.pop("data_mount"),
                       lambda r: r["data_mount"].update(source=str(self.w.data.resolve())),
                       lambda r: r["data_mount"].update(view_files_sha256=H("other listing"))):
            with self.subTest(change=change):
                self.w.write_attempt(out)
                self.w.edit(out, "runtime.json", change)
                with self.assertRaisesRegex(ValueError, "^hpo_isolation_unproven:"):
                    runtime.validate_completion(self.w.settings, out, self.w.spec())
        self.w.write_attempt(out)
        with self.assertRaisesRegex(ValueError, "^hpo_isolation_unproven:"):
            runtime.validate_completion({k: v for k, v in self.w.settings.items() if k != "views"}, out, self.w.spec())

    def test_predictions_tampered_and_resigned_are_rejected(self):
        out = self.root / "out"

        def score(a):
            a["rows"][0]["scores"] = [0.9, 0.1]

        def metric(a):
            a["metrics"]["primary"]["value"] = 0.123

        def e_row(a):
            e_id = self.w.fx.pools["lists"]["E"][0]
            a["rows"].append({"id": e_id, "group": "g3", "target": 0, "scores": [0.5, 0.5], "prediction": 1})
            a["rows"].sort(key=lambda r: r["id"])

        for change, code in ((score, "(decision_recompute_mismatch|metric_recompute_mismatch)"),
                             (metric, "metric_recompute_mismatch"), (e_row, "hpo_view_contains_E")):
            with self.subTest(code=code):
                self.w.write_attempt(out)
                self.w.edit(out, "predictions.json", lambda a: (change(a), resign_predictions(a)))
                with self.assertRaisesRegex(ValueError, f"^{code}:"):
                    runtime.validate_completion(self.w.settings, out, self.w.spec())

    def test_results_summary_must_match_recomputed_artifact(self):
        out = self.root / "out"
        for change, code in ((lambda r: r["clinical_evaluation"].update(primary_value=0.99), "metric_recompute_mismatch"),
                             (lambda r: r["selection"].update(accuracy=0.0), "metric_recompute_mismatch"),
                             (lambda r: r.update(test={"loss": 0.1, "accuracy": 1.0, "samples": 2}), "global_artifacts_invalid"),
                             (lambda r: r.update(best_epoch=1), "epoch_selection_forbidden"),
                             (lambda r: r.update(scientific_eligible=True), "global_artifacts_invalid")):
            with self.subTest(code=code):
                self.w.write_attempt(out)
                self.w.edit(out, "results.json", change)
                with self.assertRaisesRegex(ValueError, f"^{code}:"):
                    runtime.validate_completion(self.w.settings, out, self.w.spec())

    def test_nonnull_trainer_override_is_rejected(self):
        out = self.root / "out"
        toy = {"device": "cpu", "workers": 0, "reason": "toy"}
        self.w.write_attempt(out, override=toy)
        with self.assertRaisesRegex(ValueError, "^trainer_override_forbidden:"):
            runtime.validate_completion(self.w.settings, out, self.w.spec())
        for name in ("completed.json", "environment.json", "results.json"):
            with self.subTest(name=name):
                self.w.write_attempt(out)
                if name == "completed.json":
                    self.w.completed["trainer_override"] = toy
                    self.w.resign(out)
                else:
                    self.w.edit(out, name, lambda d: d.update(trainer_override=toy))
                with self.assertRaisesRegex(ValueError, "^trainer_override_forbidden:"):
                    runtime.validate_completion(self.w.settings, out, self.w.spec())
        self.w.write_attempt(out)
        self.w.edit(out, "environment.json", lambda d: d.pop("trainer_override"))
        with self.assertRaisesRegex(ValueError, "^trainer_override_forbidden:"):
            runtime.validate_completion(self.w.settings, out, self.w.spec())

    def test_artifact_set_and_checksums_are_closed(self):
        out = self.w.write_attempt(self.root / "out")
        (out / "best.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "^global_artifacts_invalid:"):
            runtime.validate_completion(self.w.settings, out, self.w.spec())
        self.w.write_attempt(out)
        completed = json.loads((out / "completed.json").read_text())
        completed["artifacts"].pop("predictions.json")
        atomic_json(out / "completed.json", completed)
        with self.assertRaisesRegex(ValueError, "^global_artifacts_invalid:"):
            runtime.validate_completion(self.w.settings, out, self.w.spec())

    def test_legacy_validate_artifacts_refuses_global_attempt_with_a_clear_code(self):
        out = self.w.write_attempt(self.root / "out")
        with self.assertRaisesRegex(ValueError, "^global_config_invalid:"):
            runtime.validate_artifacts(out, self.w.spec())

    def test_stopped_global_checkpoint_uses_byte_identity_and_refuses_override(self):
        out = self.w.write_attempt(self.root / "out")
        numbers, checkpoint = runtime._stopped_checkpoint(out, self.w.spec(), self.w.code)
        self.assertEqual(numbers["epoch"], EPOCHS)
        self.assertEqual(checkpoint["config_sha256"], self.w.spec()["config_sha256"])
        self.w.edit(out, "environment.json", lambda d: d.update(trainer_override={"device": "cpu"}))
        with self.assertRaisesRegex(ValueError, "^checkpoint_trainer_override_forbidden$"):
            runtime._stopped_checkpoint(out, self.w.spec(), self.w.code)
        self.w.write_attempt(out)
        cfg = self.w.cfg("hpo")
        cfg["training"]["seed"] = 7
        (out / "config.json").write_bytes(gj.render_config_bytes(cfg))
        with self.assertRaisesRegex(ValueError, "^checkpoint_config_identity$"):
            runtime._stopped_checkpoint(out, self.w.spec(), self.w.code)


class FinalPhaseTests(GlobalRuntimeTest):
    def test_final_phase_is_refused_at_the_campaign_gate_before_claim(self):
        with patch("augment_atlas.runtime.command") as docker:
            with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
                runtime.DockerRuntime({**self.w.settings, "phase": "final"}).preflight()
            with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
                self.w.docker_runtime("final").config_for(self.w.spec("final"))
            docker.assert_not_called()

    def test_mock_create_of_gated_final_mounts_full_root_readonly_and_no_views(self):
        host = self.w.docker_runtime("final")
        out = self.root / "out"
        out.mkdir()
        with patch.object(host, "campaign_gate"):
            args = self.create_args(host, self.w.spec("final"), out)
        mounts = self.mounts(args)
        self.assertIn(f"type=bind,src={self.w.data.resolve()},dst=/data,readonly", mounts)
        self.assertFalse(any(str(self.w.views.resolve()) in m for m in mounts))
        self.assertEqual(sorted(m.split("dst=")[1].split(",")[0] for m in mounts), ["/code", "/data", "/output", "/torch-cache"])
        self.assertEqual(json.loads((out / "runtime.json").read_text())["data_mount"]["kind"], "data_root")

    def test_final_completion_recomputes_e_test_and_refuses_d(self):
        out = self.w.write_attempt(self.root / "out", "final")
        spec = self.w.spec("final")
        settings = {**self.w.settings, "phase": "final"}
        self.assertEqual(runtime.validate_completion(settings, out, spec)["artifact_count"], 9)

        def d_row(a):
            d_id = self.w.fx.pools["lists"]["D"][0]
            a["rows"].append({"id": d_id, "group": "g0", "target": 0, "scores": [0.5, 0.5], "prediction": 1})
            a["rows"].sort(key=lambda r: r["id"])
            resign_predictions(a)

        self.w.edit(out, "predictions.json", d_row)
        with self.assertRaisesRegex(ValueError, "^D_in_test:"):
            runtime.validate_completion(settings, out, spec)


class LegacyBoundaryTests(GlobalRuntimeTest):
    def test_global_config_disguised_under_smoke_prefix_fails_with_code_not_keyerror(self):
        smoke = self.w.code / "configs" / "smoke" / "global.json"
        smoke.parent.mkdir(parents=True)
        smoke.write_bytes((self.w.code / self.w.hpo_name).read_bytes())
        spec = {**self.w.spec(), "phase": "smoke", "config": "configs/smoke/global.json", "config_sha256": digest(smoke)}
        host = runtime.DockerRuntime({**self.w.settings, "phase": "smoke"})
        with self.assertRaisesRegex(ValueError, "^global_config_invalid:"):
            host.config_for(spec)
        with self.assertRaisesRegex(ValueError, "global_config_invalid"):
            runtime.validate_completion({**self.w.settings, "phase": "smoke"}, self.root / "none", spec)

    def test_legacy_and_clinical_specs_never_enter_the_global_branch(self):
        self.assertFalse(runtime.is_global_spec({"phase": "smoke", "config": "configs/smoke/synthetic.json"}))
        self.assertFalse(runtime.is_global_spec({"phase": "smoke", "config": "configs/smoke/clinical-fixture.json"}))
        self.assertFalse(runtime.is_global_spec(None))
        self.assertTrue(runtime.is_global_spec({"phase": "smoke", "config": "configs/global/x/hpo/c-s1.json"}))
        self.assertTrue(runtime.is_global_spec({"phase": "final"}))

    def test_runtime_error_codes_belong_to_the_spec_catalog(self):
        import re
        root = Path(__file__).resolve().parents[1]
        spec = (root / "docs/global-next-integration.md").read_text(encoding="utf-8")
        section = spec[spec.index("### 6.11"):spec.index("## 7.")]
        catalog = set(re.findall(r"`([A-Za-z_]+)(?::<reason>)?`", section))
        source = (root / "src/augment_atlas/runtime.py").read_text(encoding="utf-8")
        used = set(re.findall(r"_global_fail\('([A-Za-z_]+)'", source))
        self.assertTrue({"hpo_isolation_unproven", "trainer_override_forbidden", "config_identity_mismatch"} <= used)
        self.assertTrue(used <= catalog, used - catalog)


if __name__ == "__main__":
    unittest.main()
