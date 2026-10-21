"""Host completion boundary for the clinical opt-in (ninth artifact).

Everything is synthetic: a tiny CSV manifest with placeholder image bytes in a
host "data" directory, a pinned "code" config and an attempt directory written
the way the trainer writes it. Checkpoints are opaque placeholders; no torch,
Docker, GPU or network is used. Queue/Docker interactions in the wiring tests
are the existing mocks from test_runtime.py.
"""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import runtime
from augment_atlas.contracts import canonical_hash, load_config, recipe_identity, required_artifacts
from augment_atlas.data import load_records, prepare_splits
from augment_atlas.evaluation import build_prediction_artifact, clinical_environment_block
from augment_atlas.storage import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("clinical_fixtures_for_runtime", ROOT / "tests/clinical_fixtures.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)
_rspec = importlib.util.spec_from_file_location("runtime_fixtures_for_predictions", ROOT / "tests/test_runtime.py")
_runtime_fixtures = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(_runtime_fixtures)

CONFIG = "configs/smoke/clinical-fixture.json"


def write_manifest(data, *, groups=20, moved=None):
    """Two placeholder images per group; labels are group-consistent."""
    data.mkdir(parents=True, exist_ok=True)
    with (data / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "group", "label"])
        for index in range(groups * 2):
            name = f"img-{index:03d}.png"
            (data / name).write_bytes(b"synthetic placeholder bytes, not an image\n")
            group = f"fixture-group-{index // 2:03d}"
            if moved == name:
                group = "fixture-group-moved"
            writer.writerow([name, group, (index // 2) % 2])


def csv_config(evaluation=fx.BINARY):
    cfg = load_config(ROOT / "configs/smoke/synthetic.json")
    cfg["dataset"] = {"id": "synthetic-csv-fixture", "version": "v1", "adapter": "manifest_csv",
                      "manifest": "manifest.csv"}
    cfg["model"]["num_classes"] = 2 if evaluation is None else len(evaluation["classes"])
    if evaluation is not None:
        cfg["evaluation"] = copy.deepcopy(evaluation)
    return cfg


def scientific_split_document(cfg, records):
    """Hand-built stratified-group-outer-v1 fold that passes the stdlib audit (no sklearn).

    Fixture only: groups 0-3 test, 4-7 selection, rest train; every partition
    holds both labels. It is not a StratifiedGroupKFold output.
    """
    from augment_atlas import scientific_splits as ss
    rows = sorted(records, key=lambda row: row["id"])
    order = sorted({row["group"] for row in rows})
    part = {group: "test" if i < 4 else "selection" if i < 8 else "train" for i, group in enumerate(order)}
    lists = {name: sorted(row["id"] for row in rows if part[row["group"]] == name) for name in ss.PARTITIONS}
    seed, fold = cfg["splits"]["split_seed"], cfg["splits"]["fold"]
    document = {"manifest_sha256": canonical_hash(rows), "dataset_version": cfg["dataset"]["version"],
                "split_seed": seed, "fold": fold, "algorithm": ss.ALGORITHM, "scientific_status": "provisional",
                "outer_folds": ss.FOLDS, "outer_seed": ss._seed(seed, "outer"),
                "selection_protocol": ss.SELECTION_PROTOCOL, "selection_folds": ss.FOLDS, "selection_fold": 0,
                "selection_seed": ss._seed(seed, "selection", fold), "sklearn_version": "fixture-not-generated",
                "lists": lists,
                "groups": {name: sorted({part_group for part_group in order if part[part_group] == name})
                           for name in ss.PARTITIONS},
                "diagnostics": {"manifest": ss.split_diagnostics(rows),
                                "partitions": ss._partition_diagnostics(rows, lists)}}
    document["sha256"] = canonical_hash(document)
    ss.audit_scientific_split(rows, document)
    return document


class ClinicalAttemptFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas-runtime-predictions-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.code, self.data, self.out = self.root / "code", self.root / "data", self.root / "attempt"
        write_manifest(self.data)
        self.write_attempt(csv_config())

    # -- fixture construction -------------------------------------------------
    def write_attempt(self, cfg, *, wrong=(0,)):
        for leftover in ("predictions.json", "completed.json", "splits.json"):
            (self.out / leftover).unlink(missing_ok=True)
        self.cfg = cfg
        atomic_json(self.code / CONFIG, cfg)
        self.settings = {"code": str(self.code), "data": str(self.data)}
        records = load_records(cfg, root=self.data)
        self.out.mkdir(parents=True, exist_ok=True)
        self.splits = prepare_splits(cfg, records, self.out, root=self.data)
        self.records = records
        config_hash = canonical_hash(cfg)
        atomic_json(self.out / "config.json", cfg)
        for name in ("checkpoint.pt", "best.pt"):
            (self.out / name).write_bytes(b"opaque checkpoint placeholder; never deserialized\n")
        scope = cfg["training"].get("evaluation_scope", "test")
        self.environment = {"schema_version": 1, "source": "synthetic-fixture", "utc": "2026-10-19T00:00:00+00:00",
                            "config_sha256": config_hash, "splits_sha256": self.splits["sha256"],
                            "evaluation_scope": scope, "code_commit": fx.CODE_COMMIT,
                            "image_id": "sha256:fixture-image"}
        if "evaluation" in cfg:
            self.environment["clinical_evaluation"] = clinical_environment_block(cfg)
        atomic_json(self.out / "environment.json", self.environment)
        history = [{"epoch": 1, "selection": {"loss": 0.7, "accuracy": 0.5, "samples": 2}},
                   {"epoch": 2, "selection": {"loss": 0.6, "accuracy": 0.5, "samples": 2}}]
        atomic_json(self.out / "metrics.json", history)
        atomic_json(self.out / "progress.json", {"schema_version": 1, "status": "completed", "epoch": 2, "epochs": 2,
                    "samples_processed": 64, "config_sha256": config_hash, "splits_sha256": self.splits["sha256"]})
        results = {"schema_version": 1, "phase": "smoke", "scientific_status": "provisional", "best_epoch": 2,
                   "selection_loss": 0.6, "config_sha256": config_hash, "splits_sha256": self.splits["sha256"]}
        partition = "selection" if scope == "selection-only" else "test"
        if scope == "selection-only":
            results["evaluation_scope"] = scope
        rows = [{r["id"]: r for r in records}[i] for i in self.splits["lists"][partition]]
        if "evaluation" in cfg:
            self.scores = fx.peaked_scores(rows, len(cfg["evaluation"]["classes"]), wrong=wrong)
            artifact, results["clinical_evaluation"] = build_prediction_artifact(
                cfg, self.splits, records, self.scores, best_epoch=2, environment=self.environment)
            atomic_json(self.out / "predictions.json", artifact)
        results[partition] = {"loss": 0.5, "accuracy": (len(rows) - len(wrong)) / len(rows), "samples": len(rows)}
        atomic_json(self.out / "results.json", results)
        names = required_artifacts(cfg)
        atomic_json(self.out / "completed.json", {"schema_version": 1, "status": "completed", "epoch": 2,
                    "config_sha256": config_hash, "splits_sha256": self.splits["sha256"],
                    "artifacts": {name: digest(self.out / name) for name in names},
                    "artifact_sizes": {name: (self.out / name).stat().st_size for name in names}})
        self.spec = {"config": CONFIG, "config_sha256": digest(self.code / CONFIG), "split_hash": self.splits["sha256"],
                     "recipe_hash": recipe_identity(cfg), "train_seed": cfg["training"]["seed"],
                     "arm": cfg["augmentation"]["arm"], "code_commit": fx.CODE_COMMIT}

    def read(self, name):
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def reseal(self, *names):
        """Recompute transport checksums: isolates semantic validation from file hashes."""
        completion = self.read("completed.json")
        for name in names:
            completion["artifacts"][name] = digest(self.out / name)
            completion["artifact_sizes"][name] = (self.out / name).stat().st_size
        atomic_json(self.out / "completed.json", completion)

    def rewrite(self, name, operation):
        document = self.read(name)
        result = operation(document)
        atomic_json(self.out / name, document if result is None else result)
        self.reseal(name)

    def rederive(self, scores, *, best_epoch=2, accuracy=None):
        """A forger rewrites probabilities and every derived value consistently."""
        artifact, clinical = build_prediction_artifact(self.cfg, self.splits, self.records, scores,
                                                       best_epoch=best_epoch, environment=self.environment)
        atomic_json(self.out / "predictions.json", artifact)

        def update(results):
            results["clinical_evaluation"] = clinical
            results["best_epoch"] = best_epoch
            if accuracy is not None:
                results["test"]["accuracy"] = accuracy
        self.rewrite("results.json", update)
        self.reseal("predictions.json")

    def validate(self, spec=None, settings=None):
        return runtime.validate_completion(self.settings if settings is None else settings, self.out,
                                           self.spec if spec is None else spec)

    def assert_rejected(self, pattern, **kwargs):
        with self.assertRaisesRegex(ValueError, pattern):
            self.validate(**kwargs)


class OptInCompletionTests(ClinicalAttemptFixture):
    def test_preconditions_both_classes_and_multi_image_groups_in_test(self):
        labels = {r["id"]: r for r in self.records}
        test = [labels[i] for i in self.splits["lists"]["test"]]
        self.assertEqual({row["label"] for row in test}, {0, 1})
        self.assertLess(len({row["group"] for row in test}), len(test))

    def test_valid_opt_in_requires_and_counts_nine_artifacts(self):
        summary = self.validate()
        self.assertEqual(summary["artifact_count"], 9)
        self.assertEqual(summary["epoch"], 2)
        # Public summary carries no clinical rows or metric values.
        self.assertFalse({"clinical_evaluation", "metrics", "rows"} & set(summary))
        trusted = runtime.load_trusted_evaluation_inputs(self.code / CONFIG, self.data,
                                                         expected_config_sha256=self.spec["config_sha256"])
        self.assertEqual(runtime.validate_artifacts(self.out, self.spec, trusted=trusted)["artifact_count"], 9)

    def test_legacy_call_without_trusted_inputs_fails_closed_for_opt_in(self):
        for spec in (None, self.spec):
            with self.subTest(spec=spec is not None), self.assertRaisesRegex(ValueError, "trusted host"):
                runtime.validate_artifacts(self.out, spec)
        for bogus in ({"cfg": self.cfg}, object()):
            with self.assertRaisesRegex(ValueError, "trusted host"):
                runtime.validate_artifacts(self.out, self.spec, trusted=bogus)

    def test_prediction_artifact_must_be_checksummed_and_present(self):
        completion = self.read("completed.json")
        del completion["artifacts"]["predictions.json"]
        atomic_json(self.out / "completed.json", completion)
        self.assert_rejected("cover the clinical prediction artifact")
        self.write_attempt(self.cfg)
        (self.out / "predictions.json").write_text("{}", encoding="utf-8")
        self.assert_rejected("Invalid completed artifact")
        self.write_attempt(self.cfg)
        (self.out / "predictions.json").unlink()
        self.assert_rejected("Invalid completed artifact")

    def test_resealed_row_tampering_rejected(self):
        def swap_ids(artifact):
            rows = artifact["rows"]
            rows[0]["id"], rows[-1]["id"] = rows[-1]["id"], rows[0]["id"]

        def swap_groups(artifact):
            rows = artifact["rows"]
            different = next(row for row in rows if row["group"] != rows[0]["group"])
            rows[0]["group"], different["group"] = different["group"], rows[0]["group"]

        def flip_truth(artifact):
            row = artifact["rows"][0]
            row["target_index"] = 1 - row["target_index"]
            row["truth"] = row["target_index"]

        def alter_score(artifact):
            artifact["rows"][1]["scores"] = [0.125, 0.875] if artifact["rows"][1]["scores"][1] > 0.5 else [0.875, 0.125]

        def alter_metric(results):
            results["clinical_evaluation"]["metrics"]["binary_auroc"]["value"] = 0.99

        for name, operation in (("predictions.json", swap_ids), ("predictions.json", swap_groups),
                                ("predictions.json", flip_truth), ("predictions.json", alter_score),
                                ("results.json", alter_metric)):
            with self.subTest(operation=operation.__name__):
                self.write_attempt(self.cfg)
                self.rewrite(name, operation)
                self.assert_rejected("mismatch")

    def test_resealed_self_declared_groups_and_splits_are_not_trusted(self):
        """The container rewrites rows AND splits.json consistently; host manifest wins."""
        moved = self.splits["lists"]["test"][0]

        def regroup_splits(splits):
            lookup = {r["id"]: dict(r) for r in self.records}
            lookup[moved]["group"] = "fixture-group-forged"
            splits["groups"] = {name: sorted({lookup[i]["group"] for i in ids}) for name, ids in splits["lists"].items()}
            splits["sha256"] = canonical_hash({k: v for k, v in splits.items() if k != "sha256"})

        def regroup_rows(artifact):
            for row in artifact["rows"]:
                if row["id"] == moved:
                    row["group"] = "fixture-group-forged"
        self.rewrite("splits.json", regroup_splits)
        self.rewrite("predictions.json", regroup_rows)
        forged = self.read("splits.json")["sha256"]
        for name in ("results.json", "environment.json", "progress.json"):
            self.rewrite(name, lambda document: document.update(splits_sha256=forged))
        completion = self.read("completed.json")
        completion["splits_sha256"] = forged
        atomic_json(self.out / "completed.json", completion)
        self.assert_rejected("another job|splits differ", spec={**self.spec, "split_hash": forged})
        self.assert_rejected("another job|splits differ")

    def test_host_manifest_is_authoritative_and_changes_are_detected(self):
        write_manifest(self.data, moved=self.splits["lists"]["test"][0])
        self.assert_rejected("splits differ")

    def test_data_root_comes_only_from_settings(self):
        other = self.root / "other-data"
        write_manifest(other, groups=12)
        with patch.dict(os.environ, {"ATLAS_DATA_ROOT": str(other)}), \
             patch("augment_atlas.data.data_root", side_effect=AssertionError("implicit data root consulted")):
            self.assertEqual(self.validate()["artifact_count"], 9)
        for settings, pattern in (({"code": str(self.code)}, "settings.data"),
                                  ({"code": str(self.code), "data": ""}, "settings.data"),
                                  ({"code": str(self.code), "data": None}, "settings.data"),
                                  ({"code": str(self.code), "data": str(self.root / "missing")}, "existing host directory")):
            with self.subTest(settings=settings), \
                 patch("augment_atlas.data.data_root", side_effect=AssertionError("no /data fallback")), \
                 patch.dict(os.environ, {"ATLAS_DATA_ROOT": str(self.data)}):
                self.assert_rejected(pattern, settings=settings)

    def test_split_file_is_located_only_under_settings_data(self):
        """Scientific path: dataset.split_file is resolved with the explicit host root (data.py)."""
        for algorithm in ("group-hash-v1", "stratified-group-outer-v1"):
            with self.subTest(algorithm=algorithm):
                cfg = csv_config()
                cfg["splits"]["version"] = algorithm
                cfg["dataset"]["split_file"] = "frozen-splits.json"
                records = load_records(cfg, root=self.data)
                if algorithm == "group-hash-v1":
                    plain = copy.deepcopy(cfg)
                    del plain["dataset"]["split_file"]
                    with tempfile.TemporaryDirectory() as directory:
                        frozen = prepare_splits(plain, records, directory, root=self.data)
                else:
                    frozen = scientific_split_document(cfg, records)
                atomic_json(self.data / "frozen-splits.json", frozen)
                self.write_attempt(cfg)
                self.assertEqual(self.splits["sha256"], frozen["sha256"])
                nowhere = self.root / "nowhere"
                nowhere.mkdir(exist_ok=True)
                with patch.dict(os.environ, {"ATLAS_DATA_ROOT": str(nowhere)}), \
                     patch("augment_atlas.data.data_root", side_effect=AssertionError("implicit data root consulted")):
                    self.assertEqual(self.validate()["artifact_count"], 9)
                    (self.data / "frozen-splits.json").unlink()
                    with self.assertRaises(OSError):
                        self.validate()

    def test_moved_checkout_cannot_decide_opt_in(self):
        """settings.code no longer at the job commit fails closed in both directions."""
        clinical_spec = dict(self.spec)
        # Clinical job; checkout now holds a legacy file; attempt resealed as legacy (8 artifacts).
        self.write_attempt(csv_config(None))
        forged = {**self.spec, "config_sha256": clinical_spec["config_sha256"],
                  "recipe_hash": clinical_spec["recipe_hash"]}
        self.assertNotEqual(digest(self.code / CONFIG), forged["config_sha256"])
        self.assert_rejected("Pinned job config checksum", spec=forged)
        # Legacy job; checkout now holds a clinical file.
        legacy_spec = dict(self.spec)
        atomic_json(self.code / CONFIG, csv_config())
        self.assert_rejected("Pinned job config checksum", spec=legacy_spec)
        with self.assertRaisesRegex(ValueError, "Pinned job config checksum"):
            runtime.trusted_evaluation_inputs(self.settings, {k: v for k, v in legacy_spec.items()
                                                              if k != "config_sha256"})

    def test_extra_top_level_result_claims_rejected(self):
        for change in ({"scientifically_eligible": True}, {"clinical_claim": "validated"},
                       {"scientific_status": "confirmatory"}, {"phase": "final"},
                       {"selection": {"loss": 0.5, "accuracy": 1.0, "samples": 2}},
                       {"inbreast_previously_examined_in_pilot": True},
                       {"inbreast_previously_examined_in_pilot": 0}):
            with self.subTest(change=change):
                self.write_attempt(self.cfg)
                self.rewrite("results.json", lambda results: results.update(change))
                self.assert_rejected("outside the trainer contract")
        self.write_attempt(self.cfg)
        self.rewrite("results.json", lambda results: results.update(inbreast_previously_examined_in_pilot=False))
        self.assertEqual(self.validate()["artifact_count"], 9)

    def test_parser_failures_are_attempt_validation_failures(self):
        """RecursionError/csv.Error become ValueError, the worker's failed-attempt path."""
        (self.out / "predictions.json").write_text("[" * 200000 + "]" * 200000, encoding="utf-8")
        self.reseal("predictions.json")
        self.assert_rejected("cannot be parsed")
        self.write_attempt(self.cfg)
        with (self.data / "manifest.csv").open("a", encoding="utf-8", newline="") as stream:
            stream.write("img-000.png,fixture-group-000," + "9" * 200000 + "\n")
        self.assert_rejected("cannot be parsed")

    def test_spec_identity_is_required_for_opt_in(self):
        for change, pattern in (({"recipe_hash": "f" * 64}, "recipe/split/code"),
                                ({"code_commit": "c" * 40}, "code commit"),
                                ({"code_commit": ""}, "recipe/split/code"),
                                ({"split_hash": "0" * 64}, "another job"),
                                ({"config_sha256": "0" * 64}, "config checksum")):
            with self.subTest(change=change):
                self.assert_rejected(pattern, spec={**self.spec, **change})
        without = {k: v for k, v in self.spec.items() if k != "code_commit"}
        self.assert_rejected("recipe/split/code", spec=without)
        trusted = runtime.trusted_evaluation_inputs(self.settings, self.spec)
        with self.assertRaisesRegex(ValueError, "recipe/split/code"):
            runtime.validate_artifacts(self.out, None, trusted=trusted)
        with self.assertRaisesRegex(ValueError, "recipe/split/code"):
            runtime.validate_artifacts(self.out, {**self.spec, "config_sha256": "0" * 64}, trusted=trusted)

    def test_environment_identity_is_checked(self):
        def block(environment):
            environment["clinical_evaluation"]["scientifically_eligible"] = True
        self.rewrite("environment.json", block)
        self.assert_rejected("environment does not declare")
        self.write_attempt(self.cfg)
        self.rewrite("environment.json", lambda environment: environment.update(code_commit="c" * 40))
        self.assert_rejected("code commit")
        self.write_attempt(self.cfg)
        self.rewrite("environment.json", lambda environment: environment.update(utc="2027-01-01T00:00:00+00:00"))
        self.assert_rejected("provenance")

    def test_selected_epoch_comes_from_selection_journal(self):
        self.rederive(self.scores, best_epoch=1)
        self.assert_rejected("Selected epoch")
        self.write_attempt(self.cfg)

        def better_first_epoch(history):
            history[0]["selection"]["loss"] = 0.1
        self.rewrite("metrics.json", better_first_epoch)
        self.assert_rejected("Selected epoch")

    def test_stripping_evaluation_from_attempt_config_cannot_downgrade_to_legacy(self):
        legacy = {k: v for k, v in self.cfg.items() if k != "evaluation"}
        legacy_hash = canonical_hash(legacy)
        atomic_json(self.out / "config.json", legacy)
        for name in ("results.json", "environment.json", "progress.json"):
            def update(document):
                document["config_sha256"] = legacy_hash
                document.pop("clinical_evaluation", None)
            self.rewrite(name, update)
        (self.out / "predictions.json").unlink()
        completion = self.read("completed.json")
        completion["config_sha256"] = legacy_hash
        completion["artifacts"].pop("predictions.json")
        completion["artifact_sizes"].pop("predictions.json")
        atomic_json(self.out / "completed.json", completion)
        self.reseal("config.json")
        # The bare legacy validator cannot know the job opted in ...
        self.assertEqual(runtime.validate_artifacts(self.out, self.spec)["artifact_count"], 8)
        # ... the worker/reconciliation entry derives it from the pinned checkout.
        self.assert_rejected("differs from the trusted clinical job config")

    def test_partition_accuracy_must_agree_with_probabilities(self):
        flipped = copy.deepcopy(self.scores)
        flipped[1] = list(reversed(flipped[1]))  # a correct row becomes wrong
        self.rederive(flipped)
        self.assert_rejected("accuracy is inconsistent")
        self.write_attempt(self.cfg)
        self.rewrite("results.json", lambda results: results["test"].update(samples=results["test"]["samples"] - 1,
                     accuracy=0.5))
        self.assert_rejected("accuracy is inconsistent")

    def test_limit_fully_rederived_forgery_is_not_execution_evidence(self):
        """Documented limit: consistent forged probabilities are undetectable here.

        This is why results keep execution_verified=false and scientifically_eligible=false.
        """
        forged = copy.deepcopy(self.scores)
        forged[1] = [0.125, 0.875] if forged[1][1] > 0.5 else [0.875, 0.125]  # same argmax outcome
        self.rederive(forged)
        self.assertEqual(self.validate()["artifact_count"], 9)
        clinical = self.read("results.json")["clinical_evaluation"]
        self.assertIs(clinical["execution_verified"], False)
        self.assertIs(clinical["scientifically_eligible"], False)

    def test_selection_only_opt_in_validates_selection_partition(self):
        cfg = csv_config()
        cfg["training"]["evaluation_scope"] = "selection-only"
        cfg["recipe"]["hpo"]["protocol"] = "within-outer-fold"
        self.write_attempt(cfg, wrong=())
        self.assertNotIn("test", self.read("results.json"))
        self.assertEqual(self.validate()["artifact_count"], 9)
        rows = self.read("predictions.json")["rows"]
        self.assertEqual([row["id"] for row in rows], self.splits["lists"]["selection"])

    def test_multiclass_and_ordinal_opt_in(self):
        for evaluation in (fx.MULTICLASS, fx.ORDINAL):
            with self.subTest(task=evaluation["task"]):
                write_manifest(self.data)
                self.write_attempt(csv_config(evaluation))
                self.assertEqual(self.validate()["artifact_count"], 9)


class LegacyCompatibilityTests(ClinicalAttemptFixture):
    def setUp(self):
        super().setUp()
        self.write_attempt(csv_config(None))

    def test_legacy_config_uses_exact_legacy_call(self):
        self.assertIsNone(runtime.trusted_evaluation_inputs(self.settings, self.spec))
        with patch("augment_atlas.runtime.validate_artifacts", return_value={"mock": True}) as validate:
            self.assertEqual(runtime.validate_completion(self.settings, self.out, self.spec), {"mock": True})
        validate.assert_called_once_with(self.out, self.spec)
        summary = runtime.validate_completion(self.settings, self.out, self.spec)
        self.assertEqual(summary, runtime.validate_artifacts(self.out, self.spec))
        self.assertEqual(summary["artifact_count"], 8)

    def test_legacy_ignores_data_settings_and_specs_without_config(self):
        # No manifest is read for legacy jobs, so absent/odd data settings do not matter.
        self.assertEqual(runtime.validate_completion({"code": str(self.code)}, self.out, self.spec)["artifact_count"], 8)
        spec = {k: v for k, v in self.spec.items() if k != "config"}
        self.assertIsNone(runtime.trusted_evaluation_inputs({}, spec))

    def test_trusted_inputs_cannot_be_attached_to_legacy_attempts(self):
        clinical = self.root / "clinical"
        atomic_json(clinical / CONFIG, csv_config())
        trusted = runtime.load_trusted_evaluation_inputs(clinical / CONFIG, self.data,
                                                         expected_config_sha256=digest(clinical / CONFIG))
        with self.assertRaisesRegex(ValueError, "differs from the trusted"):
            runtime.validate_artifacts(self.out, self.spec, trusted=trusted)
        with self.assertRaisesRegex(ValueError, "only defined for clinical"):
            runtime.load_trusted_evaluation_inputs(self.code / CONFIG, self.data,
                                                   expected_config_sha256=self.spec["config_sha256"])

    def test_legacy_job_with_moved_checkout_fails_closed(self):
        """Documented: a legacy job validated after its pinned file changed or vanished fails."""
        atomic_json(self.code / CONFIG, {**self.cfg, "notes": "checkout moved after claim"})
        with self.assertRaisesRegex(ValueError, "Pinned job config checksum"):
            runtime.validate_completion(self.settings, self.out, self.spec)
        (self.code / CONFIG).unlink()
        with self.assertRaisesRegex(ValueError, "Invalid training config path"):
            runtime.validate_completion(self.settings, self.out, self.spec)

    def test_legacy_deeply_nested_results_is_a_validation_failure(self):
        (self.out / "results.json").write_text('{"a":' * 200000 + "1" + "}" * 200000, encoding="utf-8")
        self.reseal("results.json")
        with self.assertRaisesRegex(ValueError, "cannot be parsed"):
            runtime.validate_completion(self.settings, self.out, self.spec)

    def test_invalid_config_paths_rejected_for_lookup(self):
        for name in ("configs/other/x.json", "configs/smoke/../../escape.json", "configs/smoke/missing.json", 7):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.validate_completion(self.settings, self.out, {**self.spec, "config": name})


class WiringTests(unittest.TestCase):
    def test_reconciliation_of_completed_attempt_uses_validate_completion(self):
        fixture = _runtime_fixtures.StoppedReconciliationTests(methodName="test_completed_success_is_preserved_after_worker_crash")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.container["State"]["ExitCode"] = 0
        atomic_json(fixture.out / "completed.json", {"mock": "validated separately"})
        with patch("augment_atlas.runtime.validate_completion",
                   return_value=dict(epoch=5, epochs=5, samples_processed=80)) as validate:
            result = fixture.reconcile()
        validate.assert_called_once_with(fixture.settings, fixture.out, fixture.record["spec"])
        self.assertEqual(result["state"], "succeeded")

    def test_reconciliation_marks_untrusted_clinical_completion_failed(self):
        fixture = _runtime_fixtures.StoppedReconciliationTests(methodName="test_completed_success_is_preserved_after_worker_crash")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.container["State"]["ExitCode"] = 0
        atomic_json(fixture.out / "completed.json", {"mock": "never reached"})
        with patch("augment_atlas.runtime.trusted_evaluation_inputs",
                   side_effect=ValueError("Trusted clinical config checksum mismatch")):
            result = fixture.reconcile()
        self.assertEqual(result["state"], "failed")
        self.assertEqual(fixture.queue.list_jobs()[0]["history"][-1]["validation_error"], "checkpoint_artifacts_invalid")

    def test_worker_success_path_uses_validate_completion(self):
        fixture = _runtime_fixtures.ResumeCrashTests(methodName="test_mock_resume_acknowledged_before_active_journal_saved")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with patch("augment_atlas.runtime.validate_completion", wraps=runtime.validate_completion) as validate:
            fixture.run_recovery()
        validate.assert_called_once()
        settings, out, spec = validate.call_args.args
        self.assertEqual(spec, _runtime_fixtures.owned_record(3, "interrupted")["spec"])
        self.assertEqual(Path(out).name, "attempt-fixture")
        self.assertEqual(settings["phase"], "smoke")


if __name__ == "__main__":
    unittest.main()
