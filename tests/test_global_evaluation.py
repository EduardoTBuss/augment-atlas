"""Slice S1 (part A): global predictions.json build/validation by recomputation.

Synthetic records/pools only; no images, training or processes. Every
adversarial mutation is RE-SIGNED (content_sha256 / view sha256 recomputed)
so that it cannot be caught merely by a stale checksum.
"""
import copy
import json
import math
import unittest

from augment_atlas import clinical_metrics as cm
from augment_atlas import global_development as gd
from augment_atlas import global_evaluation as ge
from augment_atlas import global_jobs as gj
from augment_atlas.contracts import canonical_hash as H

TASKS = {
    "binary": {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
               "decision_rule": "fixed_threshold"},
    "ordinal": {"type": "ordinal", "classes": [0, 1, 2], "positive_class": None,
                "endpoint": "quadratic_weighted_kappa", "decision_rule": "argmax"},
    "multiclass": {"type": "multiclass", "classes": [0, 1, 2], "positive_class": None,
                   "endpoint": "macro_auroc_ovr", "decision_rule": "argmax"},
}
DECISIONS = {"binary": {"rule": "fixed_threshold", "threshold": 0.5, "comparison": "ge", "tie_break": None},
             "argmax": {"rule": "argmax", "threshold": None, "comparison": None, "tie_break": "first"}}
FREEZE = H("synthetic external freeze")
OTHER_FREEZE = H("another campaign freeze")


def signed(value, key="sha256"):
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = H(result)
    return result


def resign(artifact):
    return signed(artifact, "content_sha256")


def make_pools(kind, drop=()):
    """8 groups, one image per class per group; D = g0..g2, D_select = g2, E_f = g(f+3)."""
    task = dict(TASKS[kind], missing_policy="undefined_no_drop", requires_patient_identity=False)
    rows = [{"id": f"g{g}-{label}", "path": f"unavailable/g{g}-{label}.png", "group": f"g{g}", "label": label}
            for g in range(8) for label in task["classes"] if f"g{g}-{label}" not in drop]
    ids = lambda groups: sorted(r["id"] for r in rows if int(r["group"][1:]) in groups)
    document = {"schema_version": 1, "protocol": gd.PROTOCOL, "scientific_status": "provisional",
                "dataset": {"id": "synthetic_eval", "version": "v1"}, "records_sha256": H(sorted(rows, key=lambda r: r["id"])),
                "source_sha256": H("synthetic manifest bytes"),
                "provenance": {key: H(key) for key in ("preprocessing_sha256", "labels_sha256",
                                                       "duplicates_sha256", "exclusions_sha256")},
                "grouping": {"namespace": "fixture-group", "source": "manual fixture", "identity_level": "group",
                             "linkage_status": "unverified", "evidence_sha256": None,
                             "prior_exposure": "none_documented"},
                "task": task,
                "partition_provenance": {"algorithm": "manual-fixture", "version": "1", "source": "fixed lists",
                                         "generator_version": "no generator",
                                         "seeds": {"development": 17, "selection": 29, "evaluation": 31},
                                         "intended_fractions": {"D": 0.2, "D_fit_given_D": 0.8, "E_test_given_E": 0.2}},
                "lists": {"D": ids(range(3)), "D_fit": ids(range(2)), "D_select": ids([2]), "E": ids(range(3, 8))},
                "evaluation_folds": [{"fold": f, "test_ids": ids([f + 3])} for f in range(5)]}
    return rows, signed(document)


def recipe():
    return {"optimizer": {"name": "adamw", "parameters": {"lr": 0.0001, "weight_decay": 0.0001}},
            "scheduler": {"name": "cosine_per_epoch", "parameters": {"eta_min": 0.0}},
            "loss": {"name": "cross_entropy", "parameters": {}}, "batch_size": 16, "accumulation_steps": 1,
            "epochs": 20}


def hpo_view(rows, pools, **override):
    d = set(pools["lists"]["D"])
    view = {"schema_version": 1, "protocol": ge.HPO_VIEW_PROTOCOL, "scientific_status": "provisional",
            "dataset": copy.deepcopy(pools["dataset"]), "task": copy.deepcopy(pools["task"]),
            "pools_sha256": pools["sha256"],
            "records": sorted((copy.deepcopy(r) for r in rows if r["id"] in d), key=lambda r: r["id"]),
            "lists": {"train": list(pools["lists"]["D_fit"]), "selection": list(pools["lists"]["D_select"])},
            "view_files_sha256": H("D view listing")}
    view.update(override)
    return signed(view)


def make_cfg(pools, phase, *, fold=0, view_sha256=None, freeze=FREEZE, seed=11):
    task = pools["task"]
    ds = pools["dataset"]["id"]
    sha = pools["sha256"]
    hpo = phase == "hpo"
    glob = {"pools_file": f"global/{ds}/pools-{sha}.json", "pools_sha256": sha, "view": phase,
            "hpo_view_file": f"global/{ds}/hpo-view-{view_sha256}.json" if hpo else None,
            "hpo_view_sha256": view_sha256 if hpo else None,
            "plan_sha256": H("plan") if hpo else None, "search_space_sha256": H("search space") if hpo else None,
            "candidate_id": "c00" if hpo else None, "candidate_config_sha256": H("c00") if hpo else None,
            "freeze_kind": None if hpo else "external_fixed",
            "freeze_file": None if hpo else f"global/{ds}/external-freeze-{freeze}.json",
            "freeze_sha256": None if hpo else freeze, "fold": None if hpo else fold}
    cfg = {"schema_version": 1, "phase": phase, "scientific_status": "provisional",
           "protocol_version": gj.HPO_PROTOCOL if hpo else gj.FINAL_PROTOCOL,
           "dataset": {"id": ds, "version": pools["dataset"]["version"], "adapter": "manifest_csv",
                       "manifest": "synthetic/manifest.csv", "manifest_sha256": H("manifest"),
                       "require_image_hashes": True},
           "global": glob,
           "model": {"name": "resnet50", "weights": "IMAGENET1K_V2", "weights_sha256": H("weights"),
                     "num_outputs": len(task["classes"]), "head": "softmax", "init": "imagenet_fresh"},
           "input": {"size": 224, "color": "rgb"}, "augmentation": {"arm": "none", "ops": []},
           "training": gj.recipe_to_training(recipe(), seed),
           "evaluation": {"schema_version": 1, "task": task["type"], "classes": list(task["classes"]),
                          "positive_class": task["positive_class"], "endpoint": task["endpoint"],
                          "score_kind": "softmax_probability",
                          "decision": copy.deepcopy(DECISIONS["binary" if task["type"] == "binary" else "argmax"]),
                          "frozen": True}}
    return gj.validate_global_config(cfg)


def fake_scores(ids, lookup, width):
    """Deterministic, imperfect probabilities (exact binary fractions sum to 1)."""
    out = []
    for index, item in enumerate(ids):
        label = lookup[item]["label"]
        row = [0.125] * width
        hit = label if index % 3 else (label + 1) % width  # every third row is wrong
        row[hit] += 1.0 - 0.125 * width
        out.append(row)
    return out


class Case:
    def __init__(self, kind, phase, *, fold=0, drop=(), freeze=FREEZE):
        self.kind, self.phase = kind, phase
        self.rows, self.pools = make_pools(kind, drop)
        self.lookup = {r["id"]: r for r in self.rows}
        if phase == "hpo":
            self.view = hpo_view(self.rows, self.pools)
            self.cfg = make_cfg(self.pools, "hpo", view_sha256=self.view["sha256"])
            self.ids = list(self.pools["lists"]["D_select"])
        else:
            self.view = gj.derive_final_view_doc(self.rows, self.pools, fold, expected_pools_sha256=self.pools["sha256"],
                                                 freeze_sha256=freeze)
            self.cfg = make_cfg(self.pools, "final", fold=fold, freeze=freeze)
            self.ids = list(self.view["test"])
        self.scores = fake_scores(self.ids, self.lookup, len(self.pools["task"]["classes"]))

    def build(self, scores=None, **kw):
        return ge.build_global_predictions(self.cfg, self.view, self.rows, self.scores if scores is None else scores,
                                           pools=self.pools, epoch=kw.get("epoch", 20))

    def validate(self, artifact, view=None, cfg=None):
        return ge.validate_global_predictions(self.cfg if cfg is None else cfg, self.view if view is None else view,
                                              self.rows, artifact, pools=self.pools)


class GlobalEvaluationValidTests(unittest.TestCase):
    def test_every_scalar_task_and_phase_round_trips_through_json(self):
        for kind in TASKS:
            for phase in ("hpo", "final"):
                with self.subTest(kind=kind, phase=phase):
                    case = Case(kind, phase)
                    artifact = json.loads(json.dumps(case.build(), allow_nan=False))
                    summary = case.validate(artifact)
                    self.assertIs(summary["execution_verified"], False)
                    self.assertIs(summary["scientific_eligible"], False)
                    self.assertEqual(summary["partition"], "D_select" if phase == "hpo" else "E_test")
                    self.assertEqual([r["id"] for r in artifact["rows"]], case.ids)
                    self.assertEqual(artifact["epoch"], artifact["epochs"])
                    self.assertEqual(artifact["checkpoint_selection"], "last_epoch")
                    self.assertEqual(artifact["metrics"]["primary"]["metric"], TASKS[kind]["endpoint"])
                    self.assertEqual(set(artifact["metrics"]["secondary"]), set(ge.SECONDARY_METRICS))
                    self.assertEqual(artifact["content_sha256"],
                                     H({k: v for k, v in artifact.items() if k != "content_sha256"}))
                    self.assertEqual(artifact["config_sha256"],
                                     gj._bytes_sha256(gj.render_config_bytes(case.cfg)))
                    if phase == "hpo":
                        self.assertEqual((artifact["candidate_id"], artifact["fold"], artifact["freeze_sha256"]),
                                         ("c00", None, None))
                        self.assertFalse(set(case.ids) & set(case.pools["lists"]["E"]))
                    else:
                        self.assertEqual((artifact["fold"], artifact["freeze_sha256"], artifact["plan_sha256"]),
                                         (0, FREEZE, None))
                        self.assertFalse(set(case.ids) & set(case.pools["lists"]["D"]))

    def test_metrics_are_the_clinical_metrics_values(self):
        case = Case("binary", "final")
        artifact = case.build()
        truth = [case.lookup[i]["label"] for i in case.ids]
        positive = [row[1] for row in case.scores]
        auroc = cm.binary_auroc(truth, positive, classes=[0, 1], positive_label=1)
        ap = cm.binary_average_precision(truth, positive, classes=[0, 1], positive_label=1)
        self.assertEqual(artifact["metrics"]["primary"]["value"], auroc["value"])
        self.assertEqual(artifact["metrics"]["secondary"]["average_precision"]["value"], ap["value"])
        predictions = cm.binary_predictions(positive, classes=[0, 1], positive_label=1, threshold=0.5, comparison="ge")
        self.assertEqual([r["prediction"] for r in artifact["rows"]], predictions)
        self.assertEqual(artifact["metrics"]["secondary"]["macro_f1"]["value"],
                         cm.macro_f1(truth, predictions, classes=[0, 1])["value"])
        case = Case("ordinal", "final", fold=2)
        artifact = case.build()
        truth = [case.lookup[i]["label"] for i in case.ids]
        predictions = cm.multiclass_predictions(case.scores, classes=[0, 1, 2], tie_break="first")
        self.assertEqual(artifact["metrics"]["primary"]["value"],
                         cm.quadratic_weighted_kappa(truth, predictions, classes=[0, 1, 2])["value"])
        case = Case("multiclass", "hpo")
        artifact = case.build()
        truth = [case.lookup[i]["label"] for i in case.ids]
        self.assertEqual(artifact["metrics"]["primary"]["value"],
                         cm.macro_ovr_auroc(truth, case.scores, classes=[0, 1, 2])["value"])

    def test_q4_threshold_is_ge_and_argmax_ties_pick_first(self):
        case = Case("binary", "final")
        scores = [[0.5, 0.5]] + case.scores[1:]  # exactly at threshold => positive (ge)
        artifact = case.build(scores)
        self.assertEqual(artifact["rows"][0]["prediction"], 1)
        self.assertEqual(case.validate(artifact)["samples"], len(case.ids))
        case = Case("multiclass", "final")
        scores = [[0.25, 0.375, 0.375]] + case.scores[1:]
        artifact = case.build(scores)
        self.assertEqual(artifact["rows"][0]["prediction"], 1)
        case.validate(artifact)

    def test_absent_class_is_null_with_reason_and_no_row_is_dropped(self):
        # Fold 0 tests g3; drop its positive image: E_0 has no positive example.
        case = Case("binary", "final", fold=0, drop=("g3-1",))
        artifact = case.build()
        self.assertEqual(len(artifact["rows"]), len(case.ids))
        primary = artifact["metrics"]["primary"]
        self.assertIsNone(primary["value"])
        self.assertEqual(primary["reason"], "metric_undefined:no_positive_examples")
        self.assertEqual(primary["coverage"]["n_samples"], len(case.ids))
        balanced = artifact["metrics"]["secondary"]["balanced_accuracy"]
        self.assertIsNone(balanced["value"])
        self.assertEqual(balanced["reason"], "metric_undefined:undefined_class_components")
        self.assertEqual(len(balanced["per_class"]), 2)  # absent class stays in the denominator
        summary = case.validate(json.loads(json.dumps(artifact)))
        self.assertIsNone(summary["primary_value"])
        self.assertEqual(summary["primary_reason"], "metric_undefined:no_positive_examples")
        case = Case("multiclass", "final", fold=1, drop=("g4-2",))
        artifact = case.build()
        primary = artifact["metrics"]["primary"]
        self.assertIsNone(primary["value"])
        self.assertEqual(primary["coverage"]["n_classes_expected"], 3)
        self.assertEqual(primary["coverage"]["n_classes_defined"], 2)
        self.assertIsNone(artifact["metrics"]["secondary"]["average_precision"]["value"])
        case.validate(artifact)


class GlobalEvaluationAdversarialTests(unittest.TestCase):
    def assertCode(self, code, fn, *args, **kwargs):
        with self.assertRaisesRegex(ValueError, f"^{code}: "):
            fn(*args, **kwargs)

    def mutated(self, case, mutate):
        artifact = copy.deepcopy(case.build())
        mutate(artifact)
        return resign(artifact)

    def test_D_identity_in_final_test_is_rejected(self):
        case = Case("binary", "final")
        d_row = case.pools["lists"]["D_select"][0]

        def add_d(artifact):
            artifact["rows"].append({"id": d_row, "group": case.lookup[d_row]["group"],
                                     "target": case.lookup[d_row]["label"], "scores": [0.5, 0.5], "prediction": 1})
            artifact["rows"].sort(key=lambda r: r["id"])

        def replace_with_d(artifact):
            artifact["rows"][0].update(id=d_row, group=case.lookup[d_row]["group"])
        for mutate in (add_d, replace_with_d):
            with self.subTest(mutate=mutate.__name__):
                self.assertCode("D_in_test", case.validate, self.mutated(case, mutate))
        view = copy.deepcopy(case.view)
        view["test"] = sorted(view["test"] + [d_row])
        self.assertCode("D_in_test", case.validate, case.build(), view=signed(view))
        self.assertCode("D_in_test", ge.build_global_predictions, case.cfg, signed(view), case.rows,
                        case.scores, pools=case.pools, epoch=20)

    def test_E_identity_in_hpo_artifact_or_view_is_rejected(self):
        case = Case("binary", "hpo")
        e_row = case.pools["lists"]["E"][0]

        def add_e(artifact):
            artifact["rows"].append({"id": e_row, "group": case.lookup[e_row]["group"],
                                     "target": case.lookup[e_row]["label"], "scores": [0.5, 0.5], "prediction": 1})
        self.assertCode("hpo_view_contains_E", case.validate, self.mutated(case, add_e))
        record = copy.deepcopy(case.lookup[e_row])
        smuggled = [
            hpo_view(case.rows, case.pools, records=sorted(hpo_view(case.rows, case.pools)["records"] + [record],
                                                           key=lambda r: r["id"])),
            hpo_view(case.rows, case.pools, lists={"train": sorted(case.pools["lists"]["D_fit"] + [e_row]),
                                                  "selection": case.pools["lists"]["D_select"]}),
        ]
        for view in smuggled:
            # Even a config re-bound to the smuggled view's hash is refused.
            cfg = make_cfg(case.pools, "hpo", view_sha256=view["sha256"])
            self.assertCode("hpo_view_contains_E", case.validate, case.build(), view=view, cfg=cfg)
            self.assertCode("hpo_view_contains_E", ge.build_global_predictions, cfg, view, case.rows,
                            case.scores, pools=case.pools, epoch=20)

    def test_final_coverage_must_be_exactly_E_f_once(self):
        case = Case("ordinal", "final", fold=1)
        other_fold = case.pools["evaluation_folds"][2]["test_ids"][0]

        def drop(artifact):
            artifact["rows"].pop()

        def duplicate(artifact):
            artifact["rows"].insert(0, copy.deepcopy(artifact["rows"][0]))

        def other_fold_row(artifact):
            artifact["rows"][-1].update(id=other_fold, group=case.lookup[other_fold]["group"])

        def unknown(artifact):
            artifact["rows"][-1]["id"] = "zz-not-a-record"
        for mutate in (drop, duplicate, other_fold_row, unknown):
            with self.subTest(mutate=mutate.__name__):
                self.assertCode("E_coverage_violation", case.validate, self.mutated(case, mutate))
        self.assertCode("predictions_invalid", case.validate,
                        self.mutated(case, lambda a: a["rows"].reverse()))

    def test_hpo_coverage_must_be_exactly_D_select_once(self):
        case = Case("multiclass", "hpo")
        d_fit = case.pools["lists"]["D_fit"][0]

        def fit_row(artifact):
            artifact["rows"][0].update(id=d_fit, group=case.lookup[d_fit]["group"])
        for mutate in (fit_row, lambda a: a["rows"].pop(), lambda a: a["rows"].append(copy.deepcopy(a["rows"][0]))):
            self.assertCode("selection_coverage_violation", case.validate, self.mutated(case, mutate))

    def test_invalid_probabilities_are_rejected(self):
        case = Case("multiclass", "final")
        bad_rows = {"nan": [math.nan, 0.5, 0.5], "inf": [math.inf, 0.0, 0.0], "narrow": [0.5, 0.5],
                    "wide": [0.25, 0.25, 0.25, 0.25], "sum": [0.25, 0.25, 0.5 + 2e-6], "negative": [-0.25, 0.75, 0.5],
                    "bool": [True, False, False], "string": ["0.5", 0.25, 0.25], "int": [1, 0, 0]}
        for name, row in bad_rows.items():
            with self.subTest(name=name):
                artifact = copy.deepcopy(case.build())
                artifact["rows"][0]["scores"] = row
                if name not in ("nan", "inf"):  # non-finite JSON cannot be canonically re-signed
                    artifact = resign(artifact)
                self.assertCode("prediction_scores_invalid", case.validate, artifact)
                if name != "int":  # the builder accepts JSON integers and normalizes to float
                    scores = copy.deepcopy(case.scores)
                    scores[0] = row
                    self.assertCode("prediction_scores_invalid", case.build, scores)
        self.assertCode("prediction_scores_invalid", case.build, case.scores[:-1])
        within = copy.deepcopy(case.scores)
        within[0] = [0.25, 0.25, 0.5 + 5e-7]
        case.validate(case.build(within))

    def test_declared_metric_or_decision_differs_from_recomputation(self):
        case = Case("binary", "final", fold=3)

        def primary(artifact):
            artifact["metrics"]["primary"]["value"] = 1.0

        def secondary(artifact):
            artifact["metrics"]["secondary"]["macro_f1"]["value"] = 0.99

        def reason(artifact):
            artifact["metrics"]["primary"]["reason"] = "metric_undefined:no_positive_examples"

        def scores_changed(artifact):  # decision unchanged (still >= 0.5), AUROC 0.5 -> 0.0
            row = next(r for r in artifact["rows"] if r["target"] == 0)
            self.assertEqual((row["scores"], row["prediction"]), ([0.125, 0.875], 1))
            row["scores"] = [0.0625, 0.9375]
        for mutate in (primary, secondary, reason, scores_changed):
            with self.subTest(mutate=mutate.__name__):
                self.assertCode("metric_recompute_mismatch", case.validate, self.mutated(case, mutate))

        def flip(artifact):
            artifact["rows"][0]["prediction"] = 1 - artifact["rows"][0]["prediction"]
        self.assertCode("decision_recompute_mismatch", case.validate, self.mutated(case, flip))
        self.assertCode("decision_recompute_mismatch", case.validate,
                        self.mutated(case, lambda a: a["rows"][0].update(prediction=None)))

    def test_artifact_of_another_campaign_config_or_fold_is_rejected(self):
        case = Case("binary", "final", fold=0)
        foreign = Case("binary", "final", fold=0, freeze=OTHER_FREEZE).build()
        self.assertCode("prediction_identity_mismatch", case.validate, foreign)
        other_seed = make_cfg(case.pools, "final", fold=0, seed=13)
        self.assertCode("prediction_identity_mismatch", case.validate, case.build(), cfg=other_seed)
        for key, value in (("config_sha256", H("other config")), ("pools_sha256", H("other pools")),
                           ("view_sha256", H("other view")), ("arm", "gamma"), ("fold", 1),
                           ("decision", {"rule": "fixed_threshold", "threshold": 0.4, "comparison": "ge",
                                         "tie_break": None})):
            with self.subTest(key=key):
                self.assertCode("prediction_identity_mismatch", case.validate,
                                self.mutated(case, lambda a: a.__setitem__(key, value)))
        other_fold = Case("binary", "final", fold=4)
        self.assertCode("prediction_identity_mismatch", case.validate, other_fold.build())
        hpo = Case("binary", "hpo")
        self.assertCode("prediction_identity_mismatch", hpo.validate,
                        self.mutated(hpo, lambda a: a.__setitem__("candidate_id", "c01")))

    def test_wrong_view_or_pools_anchor_is_rejected(self):
        case = Case("binary", "final", fold=0)
        other_view = gj.derive_final_view_doc(case.rows, case.pools, 1, expected_pools_sha256=case.pools["sha256"],
                                              freeze_sha256=FREEZE)
        self.assertCode("view_hash_mismatch", case.validate, case.build(), view=other_view)
        stale = copy.deepcopy(case.view)
        stale["test"] = stale["test"][:-1]
        self.assertCode("view_hash_mismatch", case.validate, case.build(), view=stale)  # not re-signed
        self.assertCode("view_hash_mismatch", case.validate, case.build(), view=signed(stale))  # re-signed
        rows, other_pools = make_pools("binary", drop=("g7-0",))
        self.assertCode("pools_anchor_mismatch", ge.validate_global_predictions, case.cfg, case.view, case.rows,
                        case.build(), pools=other_pools)
        hpo = Case("binary", "hpo")
        bad_lists = hpo_view(hpo.rows, hpo.pools, lists={"train": hpo.pools["lists"]["D_select"],
                                                        "selection": hpo.pools["lists"]["D_fit"]})
        cfg = make_cfg(hpo.pools, "hpo", view_sha256=bad_lists["sha256"])
        self.assertCode("view_hash_mismatch", hpo.validate, hpo.build(), view=bad_lists, cfg=cfg)
        missing = hpo_view(hpo.rows, hpo.pools, records=hpo.view["records"][:-1])
        cfg = make_cfg(hpo.pools, "hpo", view_sha256=missing["sha256"])
        self.assertCode("records_identity_mismatch", hpo.validate, hpo.build(), view=missing, cfg=cfg)

    def test_truth_comes_from_trusted_records_not_the_artifact(self):
        case = Case("binary", "final")
        for key, value in (("target", None), ("group", "g0"), ("target", True)):
            with self.subTest(key=key, value=value):
                self.assertCode("records_identity_mismatch", case.validate,
                                self.mutated(case, lambda a: a["rows"][0].__setitem__(key, value)))
        self.assertCode("records_identity_mismatch", case.validate,
                        self.mutated(case, lambda a: a["rows"][0].update(target=1 - a["rows"][0]["target"])))
        tampered = copy.deepcopy(case.rows)  # "trusted" records not matching the pools anchor
        row = next(r for r in tampered if r["id"] == case.ids[0])
        row["label"] = 1 - row["label"]
        self.assertCode("records_identity_mismatch", ge.validate_global_predictions, case.cfg, case.view, tampered,
                        case.build(), pools=case.pools)

    def test_structure_checksum_and_epoch(self):
        case = Case("ordinal", "hpo")
        artifact = case.build()
        stale = copy.deepcopy(artifact)
        stale["metrics"]["primary"]["value"] = 1.0  # not re-signed
        self.assertCode("predictions_invalid", case.validate, stale)
        self.assertCode("predictions_invalid", case.validate,
                        self.mutated(case, lambda a: a.__setitem__("execution_verified", True)))
        self.assertCode("predictions_invalid", case.validate,
                        self.mutated(case, lambda a: a["rows"][0].__setitem__("extra", 1)))
        self.assertCode("predictions_invalid", case.validate, [])
        for key, value in (("epoch", 19), ("epochs", 21), ("checkpoint_selection", "best_selection_loss")):
            with self.subTest(key=key):
                self.assertCode("epoch_selection_forbidden", case.validate,
                                self.mutated(case, lambda a: a.__setitem__(key, value)))
        self.assertCode("epoch_selection_forbidden", case.build, epoch=19)

    def test_multilabel_is_slice_S6(self):
        case = Case("binary", "final")
        artifact = self.mutated(case, lambda a: a["task"].__setitem__("type", "multilabel"))
        with self.assertRaisesRegex(ValueError, "^recipe_adapter_unsupported: .*S6"):
            case.validate(artifact)
        cfg = copy.deepcopy(case.cfg)
        cfg["evaluation"]["task"] = "multilabel"
        with self.assertRaisesRegex(ValueError, "^recipe_adapter_unsupported: .*S6"):
            ge.build_global_predictions(cfg, case.view, case.rows, case.scores, pools=case.pools, epoch=20)
        with self.assertRaisesRegex(ValueError, "^recipe_adapter_unsupported: .*S6"):
            case.validate(case.build(), cfg=cfg)


if __name__ == "__main__":
    unittest.main()
