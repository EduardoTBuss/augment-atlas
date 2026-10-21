"""Synthetic selection evidence; these tests never run HPO or open images."""
import copy
from itertools import permutations, product
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas.contracts import ARMS
from augment_atlas import global_development as gd
from test_global_development import H, signed, pool_fixture


def freeze_fixture(*, pools=None, rows=None):
    if pools is None:
        rows, pools, _ = pool_fixture()
    runtime = {"commit": "c" * 40, "environment_sha256": H("fixture environment"), "imagenet_weights_sha256": H("fixture weights")}
    producer = {"id": "SYNTHETIC-NOT-A-REAL-PRODUCER", "code_sha256": H("fixture code"),
                "environment_sha256": runtime["environment_sha256"], "manifest_sha256": H("fixture access declarations"),
                "logs_sha256": H("fixture producer logs")}
    recipe = {"model": {"name": "resnet50", "weights": "IMAGENET1K_V2", "weights_sha256": runtime["imagenet_weights_sha256"]},
              "input": {"size": 224, "color": "grayscale_to_rgb"}, "preprocessing_sha256": pools["provenance"]["preprocessing_sha256"],
              "optimizer": {"name": "sgd", "parameters": {"lr": 0.01, "momentum": 0.9}},
              "scheduler": {"name": "cosine", "parameters": {"minimum_lr": 0.0}},
              "batch_size": 8, "accumulation_steps": 2, "epochs": 3,
              "loss": {"name": "cross_entropy", "parameters": {"class_weights": [1.0, 1.0]}, "learned_on": "D_fit"},
              "prediction": {"rule": "scores_only", "threshold_policy": "not_applicable", "thresholds": []}}
    if pools["task"]["decision_rule"] == "fixed_threshold":
        recipe["prediction"] = {"rule": "fixed_threshold", "threshold_policy": "fixed_before_E", "thresholds": [0.5]}
    final_policy = {"initialization": "fresh_imagenet_each_arm_fold_seed", "warm_start": False,
                    "training_pool": "D_union_E_minus_test", "folds": list(range(5)), "seeds": [11, 13, 19],
                    "arms": list(ARMS), "augmentation_presets_sha256": H("fixture ten presets"),
                    "scheduler_duration": "fixed_epochs_recompute_steps_from_final_pool", "early_stopping": False,
                    "step_count": {"unit": "optimizer_update", "drop_last": False, "accumulation_tail": "step",
                                   "per_epoch": "ceil(ceil(train_images/batch_size)/accumulation_steps)"},
                    "checkpoint_selection": "last_epoch", "selection_on_E": False}
    candidates = []
    for name, lr in (("a", 0.01), ("b", 0.02)):
        other = copy.deepcopy(recipe)
        other["optimizer"]["parameters"]["lr"] = lr
        candidates.append({"id": name, "parameters": {"lr": lr}, "recipe": other,
                           "augmentation": {"arm": "none", "ops": []}})
    plan = {"schema_version": 1, "dataset": copy.deepcopy(pools["dataset"]), "pools_sha256": pools["sha256"],
            "task": copy.deepcopy(pools["task"]), "endpoint": pools["task"]["endpoint"], "direction": "maximize",
            "candidates": candidates, "seeds": [11, 13], "ranking": "mean_then_candidate_id", "failure_policy": "require_all_defined",
            "epoch_policy": "fixed_in_candidate_before_scores", "threshold_policy": "fixed_in_candidate_before_scores",
            "producer": producer, "runtime": runtime, "final_policy": final_policy}
    ledger = []
    for candidate in candidates:
        for seed in plan["seeds"]:
            ledger.append({"candidate_id": candidate["id"], "seed": seed, "config_sha256": H(candidate),
                           "train_ids": list(pools["lists"]["D_fit"]), "select_ids": list(pools["lists"]["D_select"]),
                           "epoch_ids": list(pools["lists"]["D_select"]), "threshold_ids": list(pools["lists"]["D_select"]),
                           "artifact_sha256": H(["fixture artifact", candidate["id"], seed]), "log_sha256": H(["fixture log", candidate["id"], seed]),
                           "status": "completed", "metric": {"endpoint": plan["endpoint"], "state": "defined",
                                                                  "value": 0.8 if candidate["id"] == "a" else 0.7, "reason": None}, "reason": None})
    evidence = signed({"schema_version": 1, "kind": "synthetic_fixture", "producer": copy.deepcopy(producer),
                       "dataset": copy.deepcopy(pools["dataset"]), "pools_sha256": pools["sha256"], "plan_sha256": H(plan),
                       "task": copy.deepcopy(pools["task"]), "rows": ledger})
    freeze = {"schema_version": 1, "protocol": gd.FREEZE_PROTOCOL, "scientific_status": "provisional",
              "dataset": copy.deepcopy(pools["dataset"]), "task": copy.deepcopy(pools["task"]), "pools_sha256": pools["sha256"],
              "selection_plan": plan, "selection_plan_sha256": H(plan), "selection_evidence_sha256": evidence["sha256"],
              "selected_candidate_id": "a", "selected_parameters": copy.deepcopy(candidates[0]["parameters"]),
              "frozen_recipe": copy.deepcopy(recipe), "final_policy": copy.deepcopy(final_policy), "runtime": copy.deepcopy(runtime),
              "selection_scope": {"train_ids": list(pools["lists"]["D_fit"]), "select_ids": list(pools["lists"]["D_select"]),
                                  "epoch_ids": list(pools["lists"]["D_select"]), "threshold_ids": list(pools["lists"]["D_select"]),
                                  "augmentation": {"arm": "none", "ops": []}, "producer": copy.deepcopy(producer)},
              "frozen_at_utc": "2026-10-16T22:00:00Z"}
    return rows, pools, freeze, evidence


class RecipeFreezeTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.pools, self.freeze, self.evidence = freeze_fixture()
        self.pool_anchor = self.pools["sha256"]
        self.plan_anchor = H(self.freeze["selection_plan"])

    def seal(self, candidate=None, evidence=None, plan_anchor=None, output=None):
        if output is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            output = Path(temporary.name) / "recipe-freeze.json"
        return gd.seal_recipe_freeze(self.rows, self.pools, self.freeze if candidate is None else candidate,
                                      self.evidence if evidence is None else evidence,
                                      expected_pools_sha256=self.pool_anchor,
                                      expected_plan_sha256=self.plan_anchor if plan_anchor is None else plan_anchor, output=output)

    def audit(self, freeze, evidence=None, plan_anchor=None):
        return gd.audit_recipe_freeze(self.rows, self.pools, freeze, self.evidence if evidence is None else evidence,
                                       expected_pools_sha256=self.pool_anchor,
                                       expected_plan_sha256=self.plan_anchor if plan_anchor is None else plan_anchor)

    def rebind_evidence(self, candidate, evidence):
        evidence = signed(evidence)
        candidate["selection_evidence_sha256"] = evidence["sha256"]
        return evidence

    def rebind_plan(self, candidate, evidence):
        plan = candidate["selection_plan"]
        anchor = H(plan)
        candidate["selection_plan_sha256"] = anchor
        evidence["plan_sha256"] = anchor
        return self.rebind_evidence(candidate, evidence), anchor

    def test_synthetic_complete_ledger_seals_without_execution_claim(self):
        before = copy.deepcopy((self.rows, self.pools, self.freeze, self.evidence))
        frozen = self.seal()
        result = self.audit(frozen)
        self.assertTrue(result["structural_valid"])
        self.assertTrue(result["eligibility"]["contract_eligible"])
        self.assertFalse(result["eligibility"]["scientific_eligible"])
        self.assertFalse(result["execution_verified"])
        self.assertIn("synthetic_fixture", result["eligibility"]["scientific_reasons"])
        self.assertEqual(result["selected_candidate_id"], "a")
        self.assertEqual(result["candidate_means"], {"a": 0.8, "b": 0.7})
        self.assertEqual(result["final_steps"], [{"fold": index, "train_images": 14,
                                                "optimizer_updates_per_epoch": 1,
                                                "total_optimizer_updates": 3} for index in range(5)])
        self.assertEqual((self.rows, self.pools, self.freeze, self.evidence), before)
        with patch.dict("sys.modules", {name: None for name in ("torch", "torchvision", "numpy", "sklearn")}), \
             patch.object(Path, "open", side_effect=AssertionError("images unavailable")), \
             patch("subprocess.Popen", side_effect=AssertionError("no execution")):
            self.audit(frozen)

    def test_rehashed_freeze_recipe_epoch_threshold_weights_and_final_policy_mutants(self):
        frozen = self.seal()
        mutations = [lambda f: f["frozen_recipe"].update(epochs=4),
                     lambda f: f["frozen_recipe"].update(epochs=True),
                     lambda f: f["frozen_recipe"]["prediction"].update(thresholds=[0.4]),
                     lambda f: f["frozen_recipe"]["model"].update(weights_sha256=H("candidate checkpoint")),
                     lambda f: f["final_policy"].update(initialization="warm_start_candidate"),
                     lambda f: f["final_policy"].update(early_stopping=True),
                     lambda f: f["final_policy"].update(selection_on_E=True),
                     lambda f: f["final_policy"].update(checkpoint_selection="best_E_score"),
                     lambda f: f["final_policy"].update(scheduler_duration="retune_on_E"),
                     lambda f: f["final_policy"]["step_count"].update(drop_last=True),
                     lambda f: f["final_policy"].update(seeds=[11]),
                     lambda f: f["final_policy"].update(arms=["none"]),
                     lambda f: f["final_policy"].update(folds=[False, True, 2, 3, 4]),
                     lambda f: f["runtime"].update(commit="d" * 40),
                     lambda f: f["selection_scope"]["train_ids"].append("g3-0"),
                     lambda f: f["selection_scope"]["augmentation"].update(arm="basic", ops=[{"op": "basic"}]),
                     lambda f: f["selected_parameters"].update(lr=0.02),
                     lambda f: f["eligibility"].update(scientific_eligible=True),
                     lambda f: f.update(frozen_at_utc="2026-10-16T22:00:00"),
                     lambda f: f.update(frozen_at_utc="2026-10-16T22:00:00-03:00")]
        for mutate in mutations:
            doc = copy.deepcopy(frozen)
            mutate(doc)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.audit(signed(doc))

    def test_internal_plan_reseal_does_not_replace_external_anchor(self):
        freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
        freeze["selection_plan"]["candidates"][0]["recipe"]["epochs"] = 9
        evidence, _ = self.rebind_plan(freeze, evidence)
        with self.assertRaisesRegex(ValueError, "anchored selection plan"):
            self.seal(freeze, evidence)
        pools = copy.deepcopy(self.pools)
        pools["dataset"]["version"] = "replacement"
        self.pools = signed(pools)
        with self.assertRaisesRegex(ValueError, "anchored pools"):
            self.seal()

    def test_rehashed_evidence_dataset_partition_candidate_and_scope_mutants(self):
        mutations = [lambda e: e["dataset"].update(id="another dataset"),
                     lambda e: e.update(pools_sha256=H("other partition")),
                     lambda e: e.update(plan_sha256=H("other plan")),
                     lambda e: e["producer"].update(code_sha256=H("different producer")),
                     lambda e: e["rows"][0].update(candidate_id="unknown"),
                     lambda e: e["rows"][0].update(config_sha256=H("other config")),
                     lambda e: e["rows"][0]["train_ids"].append("g3-0"),
                     lambda e: e["rows"][0]["select_ids"].append("g3-0"),
                     lambda e: e["rows"][0]["epoch_ids"].append("g3-0"),
                     lambda e: e["rows"][0]["threshold_ids"].append("g3-0"),
                     lambda e: e["rows"][0]["metric"].update(endpoint="accuracy"),
                     lambda e: e["rows"].pop(),
                     lambda e: e["rows"].append(copy.deepcopy(e["rows"][0])),
                     lambda e: e["rows"][0].update(seed=True),
                     lambda e: e.update(kind="verified_real_hpo")]
        for mutate in mutations:
            freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
            mutate(evidence)
            evidence = self.rebind_evidence(freeze, evidence)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.seal(freeze, evidence)

    def test_none_augmentation_and_fixed_final_policy_enforced_even_with_new_plan_anchor(self):
        mutations = [lambda p: p["candidates"][0]["augmentation"].update(arm="basic"),
                     lambda p: p["candidates"][0]["augmentation"].update(ops=[{"op": "gamma"}]),
                     lambda p: p["candidates"][0]["recipe"].update(epochs=True),
                     lambda p: p["candidates"][0]["recipe"]["loss"].update(learned_on="E"),
                     lambda p: p["final_policy"].update(warm_start=True),
                     lambda p: p["final_policy"].update(early_stopping=True),
                     lambda p: p["final_policy"].update(selection_on_E=True),
                     lambda p: p["final_policy"]["step_count"].update(per_epoch="tune_using_E"),
                     lambda p: p["final_policy"].update(folds=[False, True, 2, 3, 4]),
                     lambda p: p.update(epoch_policy="smoke_selection_loss"),
                     lambda p: p.update(endpoint="accuracy"),
                     lambda p: p.update(failure_policy="ignore_missing_as_zero")]
        for mutate in mutations:
            freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
            mutate(freeze["selection_plan"])
            evidence, anchor = self.rebind_plan(freeze, evidence)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.seal(freeze, evidence, anchor)

    def test_complete_ledger_nondefined_failures_and_predeclared_exclusion_policy(self):
        for status, state in (("completed", "undefined"), ("failed", "unavailable"),
                              ("timeout", "unavailable"), ("missing", "unavailable")):
            freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
            row = evidence["rows"][-1]  # b fails; a remains rankable
            row.update(status=status, reason="explicit fixture failure", artifact_sha256=None)
            row["metric"].update(state=state, value=None, reason="undefined fixture metric")
            evidence = self.rebind_evidence(freeze, evidence)
            result = self.audit(self.seal(freeze, evidence), evidence)
            self.assertFalse(result["eligibility"]["contract_eligible"])
            self.assertIn("ledger_contains_nondefined_trials", result["eligibility"]["reasons"])
            self.assertNotIn("b", result["candidate_means"])
            self.assertEqual(result["nondefined_trials"][0]["status"], status)
            # Exclusion is a NEW externally approved plan, not a post-hoc patch.
            freeze["selection_plan"]["failure_policy"] = "exclude_candidate_on_any_nondefined"
            evidence, anchor = self.rebind_plan(freeze, evidence)
            result = self.audit(self.seal(freeze, evidence, anchor), evidence, anchor)
            self.assertTrue(result["eligibility"]["contract_eligible"])
            self.assertFalse(result["eligibility"]["scientific_eligible"])

    def test_nondefined_is_never_silently_zero_and_all_candidates_undefined_cannot_select(self):
        for value, reason, state in ((0, "failed", "unavailable"), (None, None, "unavailable"),
                                      (0.8, "failed", "defined")):
            freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
            evidence["rows"][-1].update(status="failed", reason=reason)
            evidence["rows"][-1]["metric"].update(value=value, reason=reason, state=state)
            evidence = self.rebind_evidence(freeze, evidence)
            with self.subTest(value=value, state=state), self.assertRaises(ValueError):
                self.seal(freeze, evidence)
        freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
        for row in evidence["rows"]:
            row.update(status="completed", reason="metric undefined")
            row["metric"].update(state="undefined", value=None, reason="metric undefined")
        evidence = self.rebind_evidence(freeze, evidence)
        with self.assertRaisesRegex(ValueError, "no fully defined candidate"):
            self.seal(freeze, evidence)

    def test_nan_infinity_bool_and_out_of_range_scores_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, -0.1, 1.1):
            freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
            evidence["rows"][0]["metric"]["value"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                # canonical signing itself must reject non-finite JSON.
                evidence = self.rebind_evidence(freeze, evidence)
                self.seal(freeze, evidence)

    def test_missing_class_and_unproven_patient_cannot_be_eligible(self):
        for case in ("class", "patient", "exposure"):
            rows, pools, _ = pool_fixture()
            if case == "class":
                rows[5]["label"] = 0
                pools["records_sha256"] = H(rows)
            elif case == "patient":
                pools["task"]["requires_patient_identity"] = True
            else:
                pools["grouping"]["prior_exposure"] = "known"
            pools = signed(pools)
            self.rows, self.pools, freeze, evidence = freeze_fixture(rows=rows, pools=pools)
            self.pool_anchor = pools["sha256"]
            self.plan_anchor = H(freeze["selection_plan"])
            frozen = self.seal(freeze, evidence)
            result = self.audit(frozen, evidence)
            self.assertFalse(result["eligibility"]["contract_eligible"])
            self.assertTrue(result["eligibility"]["reasons"])
            self.assertFalse(result["eligibility"]["scientific_eligible"])

    def test_ranking_ties_and_explicit_preselected_threshold(self):
        freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
        for row in evidence["rows"]:
            row["metric"]["value"] = 0.8
        evidence = self.rebind_evidence(freeze, evidence)
        result = self.audit(self.seal(freeze, evidence), evidence)
        self.assertEqual(result["selected_candidate_id"], "a")
        freeze["selected_candidate_id"] = "b"
        with self.assertRaisesRegex(ValueError, "ranking"):
            self.seal(freeze, evidence)
        rows, pools, _ = pool_fixture()
        pools["task"]["decision_rule"] = "fixed_threshold"
        pools = signed(pools)
        self.rows, self.pools, freeze, evidence = freeze_fixture(rows=rows, pools=pools)
        self.pool_anchor, self.plan_anchor = pools["sha256"], H(freeze["selection_plan"])
        frozen = self.seal(freeze, evidence)
        frozen["frozen_recipe"]["prediction"]["thresholds"] = [0.4]
        with self.assertRaisesRegex(ValueError, "frozen recipe"):
            self.audit(signed(frozen), evidence)

    def test_ledger_permutations_preserve_means_ranking_and_frozen_recipe(self):
        freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
        freeze["selection_plan"]["seeds"] = [11, 13, 17]
        original_rows = evidence["rows"]
        rows = []
        for candidate_id in ("a", "b"):
            template = next(row for row in original_rows if row["candidate_id"] == candidate_id)
            for seed, score in zip((11, 13, 17), (0.1, 0.2, 0.3)):
                row = copy.deepcopy(template)
                row.update(seed=seed, artifact_sha256=H(["fixture artifact", candidate_id, seed]),
                           log_sha256=H(["fixture log", candidate_id, seed]))
                row["metric"]["value"] = score
                rows.append(row)
        evidence["rows"] = rows
        evidence, anchor = self.rebind_plan(freeze, evidence)
        expected_mean = math.fsum((0.1, 0.2, 0.3)) / 3
        expected_recipe = copy.deepcopy(freeze["frozen_recipe"])
        expected_final_policy = copy.deepcopy(freeze["final_policy"])
        # All within-candidate permutations, interleaved and reversed globally.
        # The old ordinary sum elected b for some of these identical mappings.
        for left, right in product(permutations(rows[:3]), permutations(rows[3:])):
            for reverse in (False, True):
                reordered = copy.deepcopy(evidence)
                reordered["rows"] = [row for pair in zip(left, right) for row in pair]
                if reverse:
                    reordered["rows"].reverse()
                candidate = copy.deepcopy(freeze)
                reordered = self.rebind_evidence(candidate, reordered)
                with self.subTest(order=[(row["candidate_id"], row["seed"]) for row in reordered["rows"]]):
                    frozen = self.seal(candidate, reordered, anchor)
                    result = self.audit(frozen, reordered, anchor)
                    self.assertEqual(result["candidate_means"], {"a": expected_mean, "b": expected_mean})
                    self.assertEqual(result["selected_candidate_id"], "a")
                    self.assertEqual(frozen["frozen_recipe"], expected_recipe)
                    self.assertEqual(frozen["final_policy"], expected_final_policy)
                    self.assertTrue(result["eligibility"]["contract_eligible"])

    def test_freeze_no_clobber_and_external_unverified_evidence_is_not_execution(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "freeze.json"
            self.seal(output=path)
            before = path.read_bytes()
            self.seal(output=path)
            self.assertEqual(before, path.read_bytes())
            changed = copy.deepcopy(self.freeze)
            changed["frozen_at_utc"] = "2026-10-17T00:00:00Z"
            with self.assertRaisesRegex(ValueError, "no-clobber"):
                self.seal(changed, output=path)
            self.assertEqual(before, path.read_bytes())
        freeze, evidence = copy.deepcopy(self.freeze), copy.deepcopy(self.evidence)
        evidence["kind"] = "external_unverified"
        evidence = self.rebind_evidence(freeze, evidence)
        result = self.audit(self.seal(freeze, evidence), evidence)
        self.assertFalse(result["execution_verified"])
        self.assertFalse(result["eligibility"]["scientific_eligible"])


if __name__ == "__main__":
    unittest.main()
