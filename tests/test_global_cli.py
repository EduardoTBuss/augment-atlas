"""Global CLI, external fixed recipe path end to end (real files, real local Git, no training).

A synthetic private data root (manifest CSV + image files), a code repository
with the real candidate-v2 presets and a bare Git remote for the queue. Finished
final jobs are simulated by writing the attempt artifacts a worker would write.
"""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from augment_atlas import global_cli as cli
from augment_atlas import global_collect as gcol
from augment_atlas import global_development as gd
from augment_atlas import global_evaluation as ge
from augment_atlas import global_jobs as gj
from augment_atlas.contracts import ARMS, canonical_hash as H
from augment_atlas.queue import GitQueue
from augment_atlas.schema import new_record

ROOT = Path(__file__).resolve().parents[1]
DATASET = "synthetic_cli"
IMAGE = "sha256:" + "d" * 64
WEIGHTS = H("fixture weights")


def git(repo, *argv):
    return subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True, text=True).stdout.strip()


def rows():
    return [{"id": f"img/g{group}-{label}.png", "path": f"img/g{group}-{label}.png", "group": f"grp{group:02d}x",
             "label": label} for group in range(8) for label in range(2)]


def pools_candidate(records):
    ids = lambda groups: sorted(r["id"] for r in records if int(r["group"][3:5]) in groups)
    return {"schema_version": 1, "protocol": gd.PROTOCOL, "scientific_status": "provisional",
            "dataset": {"id": DATASET, "version": "v1"}, "records_sha256": H(records),
            "source_sha256": H("synthetic manifest bytes"),
            "provenance": {k: H(k) for k in ("preprocessing_sha256", "labels_sha256", "duplicates_sha256",
                                              "exclusions_sha256")},
            "grouping": {"namespace": "fixture-group", "source": "manual fixture", "identity_level": "group",
                         "linkage_status": "unverified", "evidence_sha256": None, "prior_exposure": "none_documented"},
            "task": {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
                     "decision_rule": "fixed_threshold", "missing_policy": "undefined_no_drop",
                     "requires_patient_identity": False},
            "partition_provenance": {"algorithm": "manual-fixture", "version": "1", "source": "fixed manual lists",
                                     "generator_version": "no generator",
                                     "seeds": {"development": 17, "selection": 29, "evaluation": 31},
                                     "intended_fractions": {"D": 0.2, "D_fit_given_D": 0.8, "E_test_given_E": 0.2}},
            "lists": {"D": ids(range(3)), "D_fit": ids(range(2)), "D_select": ids([2]), "E": ids(range(3, 8))},
            "evaluation_folds": [{"fold": i, "test_ids": ids([i + 3])} for i in range(5)]}


class Env:
    def __init__(self, root):
        root = Path(root)
        self.data, self.code, self.runs = root / "data", root / "code", root / "runs"
        self.remote, self.checkout, self.views = root / "queue.git", root / "queue", root / "views"
        self.records = rows()
        manifest = self.data / "manifest.csv"
        (self.data / "img").mkdir(parents=True)
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "group", "label", "sha256"])
            for row in self.records:
                data = f"pixels of {row['id']}".encode()
                (self.data / row["path"]).write_bytes(data)
                writer.writerow([row["path"], row["group"], row["label"], hashlib.sha256(data).hexdigest()])
        source = {"adapter": "manifest_csv", "manifest": "manifest.csv",
                  "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        self.source = root / "source.json"
        self.source.write_text(json.dumps(source), encoding="utf-8")
        self.source_doc = source
        self.candidate = root / "pools-candidate.json"
        self.candidate.write_text(json.dumps(pools_candidate(self.records)), encoding="utf-8")
        self.manifest_sha = gd.global_manifest_sha256(self.records, pools_candidate(self.records))
        self.probe = root / "probe.json"
        self.probe.write_text(json.dumps({"torch": "fixture", "cuda": None}), encoding="utf-8")
        (self.code / "src").mkdir(parents=True)
        (self.code / "src" / "code.py").write_text("print('pinned')\n", encoding="utf-8")
        shutil.copytree(ROOT / cli.PRESETS_DIR, self.code / cli.PRESETS_DIR)
        git(self.code, "init", "-q"); git(self.code, "config", "user.email", "t@example.invalid")
        git(self.code, "config", "user.name", "t"); git(self.code, "add", "-A"); git(self.code, "commit", "-qm", "code")
        subprocess.run(["git", "init", "--bare", "-q", str(self.remote)], check=True)
        self.pools_sha = None

    def base(self, command, pools=True):
        argv = [command, "--data-root", str(self.data), "--source", str(self.source), "--dataset", DATASET]
        if pools and self.pools_sha:
            argv += ["--pools-sha256", self.pools_sha]
        return argv

    def queue_args(self):
        return ["--remote", str(self.remote), "--checkout", str(self.checkout)]


class CliCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="atlas-global-cli-")
        self.addCleanup(tmp.cleanup)
        self.env = Env(tmp.name)

    def run_cli(self, argv, code=0):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = cli.main(argv)
        self.assertEqual(code, result, err.getvalue())
        text = out.getvalue()
        for row in self.env.records:
            self.assertNotIn(row["id"], text)
        return json.loads(text) if code == 0 else json.loads(err.getvalue())["error"]

    # -- the pipeline, step by step --------------------------------------------

    def import_pools(self):
        env = self.env
        out = self.run_cli(env.base("import-pools", pools=False) + ["--file", str(env.candidate),
                                                                    "--expected-manifest", env.manifest_sha])
        env.pools_sha = out["pools_sha256"]
        return out

    def seal_external(self):
        env = self.env
        return self.run_cli(env.base("seal-external") + [
            "--code", str(env.code), "--image-id", IMAGE, "--environment-probe", str(env.probe),
            "--weights-sha256", WEIGHTS, "--final-seeds", "11,13,19", "--color", "grayscale_to_rgb",
            "--frozen-at", "2026-10-23T12:00:00Z"])

    def freeze_args(self, freeze):
        return ["--freeze-sha256", freeze["freeze_sha256"], "--freeze-kind", "external_fixed", "--stages", "11;13,19"]

    def to_campaign(self):
        env = self.env
        self.import_pools()
        freeze = self.seal_external()
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "recipe_freeze",
                                           "--document-sha256", freeze["freeze_sha256"]])
        made = self.run_cli(env.base("materialize-final") + ["--code", str(env.code), *self.freeze_args(freeze)])
        self.assertEqual((made["configs"], made["authorized_seeds"]), (50, [11]))
        git(env.code, "add", "-A"); git(env.code, "commit", "-qm", "final configs")
        campaign = self.run_cli(env.base("seal-campaign") + [
            "--code", str(env.code), *self.freeze_args(freeze), *env.queue_args(), "--image-id", IMAGE,
            "--sealed-at", "2026-10-23T13:00:00Z"])
        return freeze, campaign

    def campaign_args(self, campaign):
        return ["--code", str(self.env.code), *self.env.queue_args(), "--campaign-sha256", campaign["campaign_sha256"]]

    def finish(self, queue, job, state, *, initial, scores=None):
        """Play the worker: claim, run, write the attempt artifacts, finish."""
        claimed = queue.claim(job["job_id"], {"server": "fixture"})
        running = queue.update(job["job_id"], claimed["attempt_id"], claimed["token"], claimed["version"],
                               {"event_id": "run", "state": "running"})
        if state == "succeeded":
            spec = job["spec"]
            cfg = json.loads((self.env.code / spec["config"]).read_text(encoding="ascii"))
            records = self.env.records
            pools = gd._read(self.env.data / "global" / DATASET / f"pools-{self.env.pools_sha}.json")
            view = gj.derive_final_view_doc(records, pools, cfg["global"]["fold"],
                                            expected_pools_sha256=pools["sha256"],
                                            freeze_sha256=cfg["global"]["freeze_sha256"])
            lookup = {r["id"]: r for r in records}
            probs = scores or [0.8 if lookup[i]["label"] else 0.3 for i in view["test"]]
            artifact = ge.build_global_predictions(cfg, view, records, [[1 - p, p] for p in probs], pools=pools,
                                                   epoch=cfg["training"]["epochs"])
            summary = ge.validate_global_predictions(cfg, view, records, artifact, pools=pools)
            out = self.env.runs / job["job_id"] / claimed["attempt_id"]
            out.mkdir(parents=True)
            (out / "predictions.json").write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
            results = {"initial_state_sha256": initial, "clinical_evaluation": {"artifact": ge.ARTIFACT_NAME, **{
                k: summary[k] for k in ("partition", "samples", "primary_metric", "primary_value", "primary_reason",
                                        "content_sha256")}}}
            (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
            (out / "runtime.json").write_text(json.dumps({"data_mount": {
                "kind": "data_root", "source": str(self.env.data), "destination": "/data", "readonly": True}}),
                encoding="utf-8")
            completed = {"status": "completed", "phase": "final", "config_sha256": spec["config_sha256"],
                         "splits_sha256": view["sha256"], "trainer_override": None,
                         "artifacts": {ge.ARTIFACT_NAME: hashlib.sha256((out / "predictions.json").read_bytes())
                                       .hexdigest()}}
            (out / "completed.json").write_text(json.dumps(completed), encoding="utf-8")
        return queue.update(job["job_id"], running["attempt_id"], running["token"], running["version"],
                            {"event_id": "end", "state": state})


class ExternalPathTests(CliCase):
    def test_identity_pins_records_and_global_manifest(self):
        env = self.env
        out = self.run_cli(env.base("identity", pools=False))
        self.assertEqual((out["records"], out["records_sha256"]), (16, H(env.records)))
        self.import_pools()
        out = self.run_cli(env.base("identity", pools=False) + ["--pools-sha256", env.pools_sha])
        self.assertEqual(out["global_manifest_sha256"], env.manifest_sha)
        self.assertTrue(out["pools_records_match"])
        audit = self.run_cli(env.base("audit-pools") + ["--expected-manifest", env.manifest_sha])
        self.assertIs(audit["structural_valid"], True)
        self.assertIs(audit["execution_verified"], False)

    def test_import_with_another_manifest_pin_is_refused(self):
        env = self.env
        error = self.run_cli(env.base("import-pools", pools=False) + [
            "--file", str(env.candidate), "--expected-manifest", H("other")], code=2)
        self.assertIn("manifest", error)
        self.assertFalse((env.data / "global").exists() and list((env.data / "global").rglob("pools-*.json")))

    def test_hpo_view_and_d_view_hold_only_d(self):
        env = self.env
        self.import_pools()
        view = self.run_cli(env.base("hpo-view"))
        document = gd._read(Path(view["path"]))
        pools = pools_candidate(env.records)
        self.assertEqual(sorted(r["id"] for r in document["records"]), pools["lists"]["D"])
        dview = self.run_cli(env.base("d-view") + ["--view-sha256", view["hpo_view_sha256"],
                                                   "--views-root", str(env.views)])
        self.assertEqual(dview["view_files_sha256"], view["view_files_sha256"])
        linked = {p.relative_to(dview["dest"]).as_posix() for p in Path(dview["dest"]).rglob("*.png")}
        self.assertEqual(linked, set(pools["lists"]["D"]))
        again = self.run_cli(env.base("hpo-view"))  # idempotent no-clobber
        self.assertEqual(again["hpo_view_sha256"], view["hpo_view_sha256"])

    def test_external_campaign_end_to_end(self):
        env = self.env
        freeze, campaign = self.to_campaign()
        self.assertEqual(campaign["matrix"]["authorized_cells"], 50)
        self.assertEqual(campaign["label"], "exploratory")
        # Final jobs before the campaign anchor are refused by the queue itself.
        error = self.run_cli(env.base("seed-final") + self.campaign_args(campaign), code=2)
        self.assertTrue(error.startswith("campaign_anchor_missing:"), error)
        anchored = self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(),
                                                      "--kind", "campaign",
                                                      "--document-sha256", campaign["campaign_sha256"]])
        seeded = self.run_cli(env.base("seed-final") + self.campaign_args(campaign))
        self.assertEqual(seeded["jobs"], 50)
        again = self.run_cli(env.base("seed-final") + self.campaign_args(campaign))  # idempotent
        self.assertEqual(again["jobs"], 50)
        chronology = self.run_cli(env.base("audit-chronology") + self.campaign_args(campaign))
        self.assertEqual(chronology["C_campaign"], anchored["commit"])
        self.assertEqual((chronology["final_jobs"], chronology["execution_verified"]), (50, False))

        queue = GitQueue(env.checkout, env.remote)
        jobs = {(j["spec"]["arm"], j["spec"]["fold"]): j for j in queue.list_jobs()}
        states = {("none", fold): "succeeded" for fold in range(5)}
        states.update({(ARMS[1], fold): "succeeded" for fold in range(4)})
        states[(ARMS[1], 4)] = "failed"
        for (arm, fold), state in states.items():
            self.finish(queue, jobs[(arm, fold)], state, initial=H(["state", fold, 11]))
        report = self.run_cli(env.base("report") + self.campaign_args(campaign) + ["--runs", str(env.runs)])
        self.assertEqual(report["authorized_cells"], 50)
        self.assertEqual(report["status_counts"], {"succeeded": 9, "failed": 1, "missing": 40})
        stored = gd._read(Path(report["path"]))
        self.assertIs(stored["scientific_eligible"], False)
        matrix = stored["matrix"]
        self.assertEqual((matrix["succeeded_cells"], matrix["failed_cells"], matrix["missing_cells"],
                          matrix["not_authorized_cells"]), (9, 1, 40, 100))
        cells = {(c["arm"], c["fold"], c["seed"]): c for c in matrix["cells"]}
        self.assertEqual((cells[(ARMS[1], 4, 11)]["status"], cells[(ARMS[1], 4, 11)]["reason"]), ("failed", "job_failed"))
        self.assertEqual(cells[("none", 0, 11)]["primary"]["value"], 1.0)  # positives 0.8 > negatives 0.3
        self.assertEqual(cells[(ARMS[2], 0, 11)]["reason"], "artifact_not_collected")
        summary = self.run_cli(env.base("summary") + ["--report-sha256", report["report_sha256"]])
        self.assertIs(summary["execution_verified"], False)
        article = Path(env.data).parent / "article"
        exported = self.run_cli(env.base("export-article") + ["--report-sha256", report["report_sha256"],
                                                              "--out", str(article)])
        self.assertTrue((article / "metrics_by_fold.csv").is_file(), exported)
        for path in article.iterdir():
            text = path.read_text(encoding="utf-8")
            for row in env.records:
                self.assertNotIn(row["id"], text)
        # Re-running the report on unchanged inputs is idempotent (no-clobber, same hash).
        self.assertEqual(report["report_sha256"], self.run_cli(
            env.base("report") + self.campaign_args(campaign) + ["--runs", str(env.runs)])["report_sha256"])

    def test_tampered_prediction_fails_the_whole_report(self):
        env = self.env
        _, campaign = self.to_campaign()
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "campaign",
                                           "--document-sha256", campaign["campaign_sha256"]])
        self.run_cli(env.base("seed-final") + self.campaign_args(campaign))
        queue = GitQueue(env.checkout, env.remote)
        job = next(j for j in queue.list_jobs() if j["spec"]["arm"] == "none" and j["spec"]["fold"] == 0)
        done = self.finish(queue, job, "succeeded", initial=H("s"))
        out = env.runs / job["job_id"] / done["attempt_id"]
        results = json.loads((out / "results.json").read_text())
        results["clinical_evaluation"]["primary_value"] = 0.99  # declared metric differs from the scores
        (out / "results.json").write_text(json.dumps(results))
        error = self.run_cli(env.base("report") + self.campaign_args(campaign) + ["--runs", str(env.runs)], code=2)
        self.assertTrue(error.startswith("metric_recompute_mismatch:"), error)
        self.assertFalse(list((env.data / "global" / DATASET).glob("report-*.json")))

    def test_dirty_checkout_and_edited_config_are_refused(self):
        env = self.env
        self.import_pools()
        freeze = self.seal_external()
        self.run_cli(env.base("materialize-final") + ["--code", str(env.code), *self.freeze_args(freeze)])
        error = self.run_cli(env.base("seal-campaign") + [
            "--code", str(env.code), *self.freeze_args(freeze), *env.queue_args(), "--image-id", IMAGE,
            "--sealed-at", "2026-10-23T13:00:00Z"], code=2)
        self.assertTrue(error.startswith("code_identity_mismatch:"), error)
        git(env.code, "add", "-A"); git(env.code, "commit", "-qm", "final configs")
        (env.code / "src" / "code.py").write_text("print('changed after the freeze')\n")
        git(env.code, "commit", "-qam", "code change")
        error = self.run_cli(env.base("seal-campaign") + [
            "--code", str(env.code), *self.freeze_args(freeze), *env.queue_args(), "--image-id", IMAGE,
            "--sealed-at", "2026-10-23T13:00:00Z"], code=2)
        self.assertTrue(error.startswith("code_identity_mismatch:"), error)

    def test_edited_freeze_file_and_second_freeze_anchor_are_refused(self):
        env = self.env
        self.import_pools()
        freeze = self.seal_external()
        path = env.data / "global" / DATASET / f"external-freeze-{freeze['freeze_sha256']}.json"
        original = path.read_text()
        document = json.loads(original)
        document["recipe"]["epochs"] = 5  # edited in place, not re-signed: the pin no longer matches
        path.write_text(json.dumps(document))
        error = self.run_cli(env.base("materialize-final") + ["--code", str(env.code), *self.freeze_args(freeze)],
                             code=2)
        self.assertNotEqual(error, "")
        self.assertFalse((env.code / gj.config_dir(DATASET, "final")).exists())
        path.write_text(original)
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "recipe_freeze",
                                           "--document-sha256", freeze["freeze_sha256"]])
        # A re-signed variant is a distinct, self-consistent freeze; the queue refuses to anchor a second one.
        document["sha256"] = H({k: v for k, v in document.items() if k != "sha256"})
        forged = env.data / "global" / DATASET / f"external-freeze-{document['sha256']}.json"
        forged.write_text(json.dumps(document))
        error = self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind",
                                                   "recipe_freeze", "--document-sha256", document["sha256"]], code=2)
        self.assertTrue(error.startswith("anchor_conflict:"), error)

    def test_module_entry_point(self):
        result = subprocess.run([sys.executable, "-m", "augment_atlas.global_cli", "--help"], capture_output=True,
                                text=True, timeout=60, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        self.assertEqual(0, result.returncode, result.stderr)
        for name in ("seal-external", "seal-campaign", "anchor", "seed-final", "report", "export-article"):
            self.assertIn(name, result.stdout)


class FinalCollectorTests(CliCase):
    def setUp(self):
        super().setUp()
        env = self.env
        _, self.campaign = self.to_campaign()
        self.run_cli(env.base("anchor") + ["--code", str(env.code), *env.queue_args(), "--kind", "campaign",
                                           "--document-sha256", self.campaign["campaign_sha256"]])
        self.run_cli(env.base("seed-final") + self.campaign_args(self.campaign))
        self.queue = GitQueue(env.checkout, env.remote)
        self.jobs = {(j["spec"]["arm"], j["spec"]["fold"]): j for j in self.queue.list_jobs()}
        args = cli._parser().parse_args(env.base("report") + self.campaign_args(self.campaign) + ["--runs", "x"])
        self.source, self.records, self.pools, self.doc, self.freeze, self.evidence, self.presets = cli._campaign(args)

    def collect(self, snapshot=None):
        return gcol.collect_final_entries(self.doc, self.queue.list_jobs() if snapshot is None else snapshot,
                                          self.env.runs, records=self.records, pools=self.pools, freeze=self.freeze,
                                          dataset_source=self.source, augmentation_presets=self.presets)

    def test_final_job_outside_the_authorized_stage_is_collection_invalid(self):
        configs = gj.build_final_configs(
            self.records, self.pools, expected_pools_sha256=self.pools["sha256"], dataset_source=self.source,
            freeze=self.freeze, freeze_kind="external_fixed", expected_freeze_sha256=self.freeze["sha256"],
            augmentation_presets=self.presets, seeds=[13])
        cfg = configs["none-f0-s13.json"]
        view = gj.derive_final_view_doc(self.records, self.pools, 0, expected_pools_sha256=self.pools["sha256"],
                                        freeze_sha256=self.freeze["sha256"])
        spec = gj.final_job_spec(cfg, view, config_path=f"{gj.config_dir(DATASET, 'final')}/none-f0-s13.json",
                                 config_sha256=ge.config_sha256(cfg),
                                 code_commit=self.doc["runtime"]["code_commit"], campaign_sha256=self.doc["sha256"])
        with self.assertRaisesRegex(ValueError, "^collection_invalid: the queue holds a final job of this campaign "
                                                "outside its authorized cells"):
            self.collect(self.queue.list_jobs() + [new_record(spec)])

    def test_retry_after_an_attempt_with_predictions_is_collection_invalid(self):
        job = self.jobs[("none", 0)]
        first = self.finish(self.queue, job, "succeeded", initial=H("s"))
        current = self.queue.list_jobs()
        record = next(r for r in current if r["job_id"] == job["job_id"])
        record = copy.deepcopy(record)
        record["history"].append({"seq": record["seq"] + 1, "kind": "manual_retry", "state": "pending",
                                  "at": "2026-10-23T15:00:00+00:00", "reason": "fixture"})
        record["history"].append({"seq": record["seq"] + 2, "kind": "claim", "state": "claimed",
                                  "attempt_id": "f" * 32, "at": "2026-10-23T15:01:00+00:00"})
        record.update(state="claimed", attempt_id="f" * 32, seq=record["seq"] + 2, version=record["version"] + 2)
        snapshot = [record if r["job_id"] == job["job_id"] else r for r in current]
        self.assertTrue((self.env.runs / job["job_id"] / first["attempt_id"] / "predictions.json").is_file())
        with self.assertRaisesRegex(ValueError, "^collection_invalid: a retried job already had a predictions.json"):
            self.collect(snapshot)

    def test_terminal_states_map_to_report_statuses_and_absent_cells_have_no_entry(self):
        self.finish(self.queue, self.jobs[("none", 0)], "interrupted", initial=None)
        self.finish(self.queue, self.jobs[("none", 1)], "cancelled", initial=None)
        self.finish(self.queue, self.jobs[("none", 2)], "failed", initial=None)
        entries, manifest = self.collect()
        by_cell = {(e["arm"], e["fold"]): e for e in entries}
        self.assertEqual(len(entries), 3)
        self.assertEqual((by_cell[("none", 0)]["status"], by_cell[("none", 0)]["reason"]),
                         ("interrupted", "job_interrupted"))
        self.assertEqual((by_cell[("none", 1)]["status"], by_cell[("none", 1)]["reason"]), ("failed", "job_cancelled"))
        self.assertEqual((by_cell[("none", 2)]["status"], by_cell[("none", 2)]["reason"]), ("failed", "job_failed"))
        self.assertTrue(all(e["predictions"] is None for e in entries))
        self.assertEqual(sum(1 for row in manifest["rows"] if row["status"] == "missing"), 47)
        self.assertEqual(manifest["sha256"], H({k: v for k, v in manifest.items() if k != "sha256"}))


if __name__ == "__main__":
    unittest.main()
