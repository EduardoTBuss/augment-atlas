"""Slice S4: private report, ID-free public summary and campaign comparison.

Synthetic records/pools/campaigns only (no images, training, Docker or Git).
Every adversarial artifact is RE-SIGNED (``content_sha256`` recomputed) so it
cannot be caught merely by a stale checksum.
"""
import copy
import json
import math
from pathlib import Path
import re
import tempfile
import unittest

from augment_atlas import clinical_metrics as cm
from augment_atlas import global_campaign as gc
from augment_atlas import global_evaluation as ge
from augment_atlas import global_jobs as gj
from augment_atlas import global_report as gr
from augment_atlas.contracts import ARMS, canonical_hash as H
from test_global_campaign import seal
from test_global_evaluation import make_pools
from test_global_jobs import DATASET_SOURCE, binary_pools, external_freeze, hpo_fixture, presets

SEED = 11
STATE = {seed: H(["initial state", seed]) for seed in (11, 13, 19)}
# Shape of ``telemetry/summary.json`` (atlas.train_telemetry_summary.v1, docs/telemetry.md).
TELEMETRY_SUMMARY = {
    "schema": "atlas.train_telemetry_summary.v1", "status": "completed", "utc": "2026-10-20T12:00:00Z",
    "sessions": 1, "records": 120, "invalid_lines": 0, "telemetry_errors": 0, "duration_s": 312.5,
    "epochs_recorded": 20, "epochs": list(range(1, 21)), "train_images_processed": 6500, "train_s": 290.0,
    "throughput_img_s_mean": 22.5, "vram_max_allocated_bytes": 2147483648, "vram_max_reserved_bytes": None,
    "gpu_temp_max_c": 81.0, "gpu_power_max_w": 180.5, "gpu_mem_used_max_mib": 5120.0, "process_rss_max_bytes": None,
    "machine_samples": 11, "batch_records": 40, "energy_kwh": 0.0125,
    "energy": {"status": "measured_integrated", "source": "nvidia-smi power.draw @30s", "coverage_fraction": 0.95},
    "null_reasons": {"vram_max_reserved_bytes": "not_exposed", "process_rss_max_bytes": "not_exposed"}}


def resign(artifact):
    value = copy.deepcopy(artifact)
    value.pop("content_sha256", None)
    value["content_sha256"] = H(value)
    return value


def probability(index, label, arm_index, fold, flavor):
    """Exact binary fraction in [2/16, 14/16]; imperfect but label-informative."""
    return ((index * 5 + arm_index * 3 + fold + flavor) % 9 + 4 * label + 2) / 16


class Fixture:
    """One sealed campaign (external_fixed or hpo_d) and builders for its final artifacts."""

    def __init__(self, kind="external_fixed", rows=None, pools=None):
        self.kind = kind
        if kind == "external_fixed":
            self.rows, self.pools, self.freeze = external_freeze(rows, pools)
            self.evidence, plan_sha256 = None, None
            self.campaign = seal(self.rows, self.pools, self.freeze)
        else:
            self.rows, self.pools, _, self.evidence, self.freeze = hpo_fixture()
            plan_sha256 = self.freeze["selection_plan_sha256"]
            self.campaign = seal(self.rows, self.pools, self.freeze, kind="hpo_d", evidence=self.evidence,
                                 search_space_sha256=H("search space"))
        self.configs = gj.build_final_configs(
            self.rows, self.pools, expected_pools_sha256=self.pools["sha256"], dataset_source=DATASET_SOURCE,
            freeze=self.freeze, freeze_kind=kind, expected_freeze_sha256=self.freeze["sha256"],
            augmentation_presets=presets(), seeds=[SEED], evidence=self.evidence, expected_plan_sha256=plan_sha256)
        self.views = {fold: gj.derive_final_view_doc(self.rows, self.pools, fold,
                                                     expected_pools_sha256=self.pools["sha256"],
                                                     freeze_sha256=self.freeze["sha256"]) for fold in range(5)}
        self.lookup = {row["id"]: row for row in self.rows}
        self._cache = {}

    def artifact(self, arm, fold, flavor=0):
        key = (arm, fold, flavor)
        if key not in self._cache:
            ids = self.views[fold]["test"]
            scores = []
            for index, item in enumerate(ids):
                p = probability(index, self.lookup[item]["label"], ARMS.index(arm), fold, flavor)
                scores.append([1 - p, p])
            cfg = self.configs[f"{arm}-f{fold}-s{SEED}.json"]
            self._cache[key] = ge.build_global_predictions(cfg, self.views[fold], self.rows, scores, pools=self.pools,
                                                           epoch=cfg["training"]["epochs"])
        return copy.deepcopy(self._cache[key])

    def entry(self, arm, fold, *, flavor=0, seed=SEED, status="succeeded", reason=None, predictions="build",
              initial_state="seed", telemetry=None, campaign_sha256=None):
        if predictions == "build":
            predictions = self.artifact(arm, fold, flavor) if status == "succeeded" else None
        return {"campaign_sha256": campaign_sha256 or self.campaign["sha256"], "arm": arm, "fold": fold,
                "seed": seed, "status": status, "reason": reason, "predictions": predictions,
                "initial_state_sha256": STATE[seed] if initial_state == "seed" else initial_state,
                "telemetry_summary": telemetry}

    def entries(self, skip=(), flavor=0):
        return [self.entry(arm, fold, flavor=flavor) for arm in ARMS for fold in range(5) if (arm, fold) not in skip]

    def report(self, entries=None, **changes):
        args = dict(expected_campaign_sha256=self.campaign["sha256"], freeze=self.freeze,
                    dataset_source=DATASET_SOURCE, augmentation_presets=presets(), evidence=self.evidence)
        args.update(changes)
        return gr.build_report(self.campaign, self.entries() if entries is None else entries, self.pools, self.rows,
                               **args)


_FIXTURES = {}


def fixture(kind="external_fixed"):
    if kind not in _FIXTURES:
        _FIXTURES[kind] = Fixture(kind)
    return _FIXTURES[kind]


def identifiers(rows):
    return ({row["id"] for row in rows}, {row["group"] for row in rows}, {row["path"] for row in rows})


def assert_no_identifiers(test, text, rows):
    """Raw-text scan: no record ID or path anywhere, no group as a JSON/CSV token."""
    ids, groups, paths = identifiers(rows)
    for value in sorted(ids | paths):
        test.assertNotIn(value, text)
    for value in sorted(groups):
        test.assertIsNone(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(value)}(?![A-Za-z0-9_-])", text), value)
    test.assertNotIn("unavailable/", text)


def keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys(item)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixture()

    def fold_auroc(self, artifact):
        positive = [row["scores"][1] for row in artifact["rows"]]
        return cm.binary_auroc([row["target"] for row in artifact["rows"]], positive, classes=[0, 1],
                               positive_label=1)["value"]

    def test_stage_one_report_header_matrix_and_q5_aggregation(self):
        report = self.fx.report()
        self.assertEqual((report["execution_verified"], report["scientific_eligible"]), (False, False))
        self.assertEqual(report["caveat"], gc.SELECTION_BIAS_CAVEAT)
        self.assertIn("Bradshaw 2023", report["caveat"])
        self.assertIn("10.1148/ryai.220232", report["caveat"])
        self.assertIn(gc.BRADSHAW_CAVEAT_ID, report["caveats"])
        self.assertEqual((report["label"], report["exploratory"], report["descriptive_only"]),
                         ("exploratory", True, True))
        self.assertEqual(report["recipe_source"], "external_fixed")
        self.assertEqual(report["blockers"][:3], list(gc.BASE_BLOCKERS))
        matrix = report["matrix"]
        self.assertEqual((matrix["declared_cells"], matrix["authorized_cells"], matrix["succeeded_cells"],
                          matrix["missing_cells"], matrix["failed_cells"], matrix["not_authorized_cells"]),
                         (150, 50, 50, 0, 0, 100))
        self.assertEqual((matrix["complete"], matrix["authorized_complete"]), (False, True))
        self.assertEqual(len(matrix["cells"]), 150)
        self.assertEqual({c["reason"] for c in matrix["cells"] if c["seed"] != SEED},
                         {"stage_not_authorized_by_this_campaign"})
        # Primary: mean over the 5 folds (recomputed here independently), then over the executed seed.
        for arm in ARMS:
            folds = [self.fold_auroc(self.fx.artifact(arm, fold)) for fold in range(5)]
            expected = math.fsum(folds) / 5
            per_seed = next(r for r in report["per_seed"] if r["arm"] == arm)
            self.assertEqual(per_seed["primary"]["value"], expected)
            mean = sum(folds) / 5
            self.assertAlmostEqual(per_seed["primary"]["sd"],
                                   math.sqrt(sum((v - mean) ** 2 for v in folds) / 4), places=12)
            per_arm = next(r for r in report["per_arm"] if r["arm"] == arm)
            self.assertEqual(per_arm["primary"]["value"], expected)
            self.assertEqual(per_arm["primary"]["aggregation"], "mean_over_folds_then_seeds")
            self.assertIsNone(per_arm["primary"]["interval"]["ci_low"])
            self.assertEqual(per_arm["primary"]["interval"]["reason"], gr.NO_INFERENCE_REASON)
            # Sensitivity (Q5): AUROC over the concatenated E predictions of the seed.
            rows = [row for fold in range(5) for row in self.fx.artifact(arm, fold)["rows"]]
            pooled = cm.binary_auroc([r["target"] for r in rows], [r["scores"][1] for r in rows], classes=[0, 1],
                                     positive_label=1)["value"]
            self.assertEqual(per_seed["pooled_E"]["primary"]["value"], pooled)
        self.assertEqual(report["pairing"]["status"], "consistent")
        self.assertEqual(report["pairing"]["groups_checked"], 5)
        self.assertEqual(report["chronology"], {"audited": False})
        self.assertEqual(report["sha256"], H({k: v for k, v in report.items() if k != "sha256"}))

    def test_paired_differences_are_descriptive_arm_minus_none(self):
        report = self.fx.report()
        paired = report["paired_differences"]
        self.assertEqual(paired["inference"], "descriptive_only")
        self.assertEqual(len(paired["cells"]), 9 * 5)
        cell = next(c for c in paired["cells"] if c["arm"] == "basic" and c["fold"] == 3)
        self.assertEqual(cell["value"], self.fold_auroc(self.fx.artifact("basic", 3))
                         - self.fold_auroc(self.fx.artifact("none", 3)))

    def test_report_is_deterministic_and_order_independent(self):
        entries = self.fx.entries()
        first = self.fx.report(entries)
        second = self.fx.report(list(reversed(entries)))
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            self.fx.report(entries, output=output)
            data = output.read_bytes()
            self.fx.report(entries, output=output)
            self.assertEqual(data, output.read_bytes())

    # --- adversarials of spec 9 S4 (all re-signed) --------------------------

    def run_with(self, arm, fold, artifact):
        entries = self.fx.entries(skip={(arm, fold)}) + [self.fx.entry(arm, fold, predictions=artifact)]
        return self.fx.report(entries)

    def test_prediction_with_a_D_identity_is_D_in_test(self):
        artifact = self.fx.artifact("basic", 1)
        d_id = self.fx.pools["lists"]["D"][0]
        artifact["rows"][0].update(id=d_id, group=self.fx.lookup[d_id]["group"], target=self.fx.lookup[d_id]["label"])
        with self.assertRaisesRegex(ValueError, "^D_in_test:"):
            self.run_with("basic", 1, resign(artifact))

    def test_duplicated_absent_or_foreign_E_identity_is_E_coverage_violation(self):
        base = self.fx.artifact("gamma_light", 2)
        duplicated = copy.deepcopy(base)
        duplicated["rows"][1] = copy.deepcopy(duplicated["rows"][0])
        absent = copy.deepcopy(base)
        absent["rows"].pop()
        foreign = copy.deepcopy(base)
        other = self.fx.views[4]["test"][0]
        foreign["rows"][0].update(id=other, group=self.fx.lookup[other]["group"],
                                  target=self.fx.lookup[other]["label"])
        for name, artifact in (("duplicated", duplicated), ("absent", absent), ("other fold", foreign)):
            with self.subTest(name), self.assertRaisesRegex(ValueError, "^E_coverage_violation:"):
                self.run_with("gamma_light", 2, resign(artifact))

    def test_two_artifacts_for_one_cell_cover_E_twice(self):
        entries = self.fx.entries() + [self.fx.entry("none", 0)]
        with self.assertRaisesRegex(ValueError, "^E_coverage_violation:"):
            self.fx.report(entries)
        entries = self.fx.entries() + [self.fx.entry("none", 0, status="failed", reason="docker_exit_1")]
        with self.assertRaisesRegex(ValueError, "^report_input_invalid:"):
            self.fx.report(entries)

    def test_artifact_of_another_campaign_is_rejected(self):
        other = fixture("hpo_d")
        self.assertEqual(other.pools["sha256"], self.fx.pools["sha256"])
        # (a) declared as another campaign's job
        entries = self.fx.entries(skip={("none", 0)}) + [
            self.fx.entry("none", 0, campaign_sha256=other.campaign["sha256"])]
        with self.assertRaisesRegex(ValueError, "^prediction_identity_mismatch:"):
            self.fx.report(entries)
        # (b) smuggled under this campaign's identity: bound to the other freeze/config
        with self.assertRaisesRegex(ValueError, "^prediction_identity_mismatch:"):
            self.run_with("none", 0, other.artifact("none", 0))
        # (c) a forged binding to this freeze, re-signed, still has the other config hash
        forged = other.artifact("none", 0)
        forged["freeze_sha256"] = self.fx.freeze["sha256"]
        with self.assertRaisesRegex(ValueError, "^prediction_identity_mismatch:"):
            self.run_with("none", 0, resign(forged))

    def test_tampered_metric_or_decision_is_rejected_by_recomputation(self):
        artifact = self.fx.artifact("clahe_heavy", 3)
        artifact["metrics"]["primary"]["value"] = 0.99
        with self.assertRaisesRegex(ValueError, "^metric_recompute_mismatch:"):
            self.run_with("clahe_heavy", 3, resign(artifact))
        artifact = self.fx.artifact("clahe_heavy", 3)
        artifact["rows"][0]["prediction"] = 1 - artifact["rows"][0]["prediction"]
        with self.assertRaisesRegex(ValueError, "^decision_recompute_mismatch:"):
            self.run_with("clahe_heavy", 3, resign(artifact))
        stale = self.fx.artifact("clahe_heavy", 3)
        stale["rows"][0]["scores"] = [0.5, 0.5]
        with self.assertRaisesRegex(ValueError, "^predictions_invalid:"):
            self.run_with("clahe_heavy", 3, stale)

    def test_missing_and_failed_cells_are_explicit_and_block_partial_means(self):
        entries = self.fx.entries(skip={("basic", 2), ("gamma_heavy", 4)})
        entries.append(self.fx.entry("gamma_heavy", 4, status="timeout", reason="job_timeout_3600s"))
        report = self.fx.report(entries)
        matrix = report["matrix"]
        self.assertEqual((matrix["succeeded_cells"], matrix["missing_cells"], matrix["failed_cells"],
                          matrix["authorized_complete"]), (48, 1, 1, False))
        cells = {(c["arm"], c["fold"], c["seed"]): c for c in matrix["cells"]}
        self.assertEqual((cells[("basic", 2, SEED)]["status"], cells[("basic", 2, SEED)]["reason"]),
                         ("missing", "artifact_not_collected"))
        self.assertIsNone(cells[("basic", 2, SEED)]["primary"])
        self.assertEqual((cells[("gamma_heavy", 4, SEED)]["status"], cells[("gamma_heavy", 4, SEED)]["reason"]),
                         ("timeout", "job_timeout_3600s"))
        basic = next(r for r in report["per_seed"] if r["arm"] == "basic")
        self.assertIsNone(basic["primary"]["value"])
        self.assertEqual(basic["primary"]["reason"], "metric_undefined:missing_folds")
        self.assertEqual((basic["primary"]["missing"], basic["primary"]["n_defined"]), ([2], 4))
        self.assertEqual(basic["pooled_E"]["primary"]["reason"], "metric_undefined:missing_folds")
        self.assertIsNone(next(r for r in report["per_arm"] if r["arm"] == "basic")["primary"]["value"])
        self.assertIsNotNone(next(r for r in report["per_arm"] if r["arm"] == "none")["primary"]["value"])
        diff = next(c for c in report["paired_differences"]["cells"] if c["arm"] == "basic" and c["fold"] == 2)
        self.assertEqual((diff["value"], diff["reason"]), (None, "metric_undefined:missing_cell"))
        self.assertEqual(report["pairing"]["status"], "incomplete")

    def test_undefined_fold_metric_makes_primary_null_without_partial_mean(self):
        rows, pools = make_pools("binary", drop=("g3-1",))  # fold 0 of E has no positive example
        fx = Fixture(rows=rows, pools=pools)
        report = fx.report()
        cell = next(c for c in report["matrix"]["cells"] if (c["arm"], c["fold"], c["seed"]) == ("none", 0, SEED))
        self.assertEqual(cell["status"], "succeeded")
        self.assertIsNone(cell["primary"]["value"])
        self.assertEqual(cell["primary"]["reason"], "metric_undefined:no_positive_examples")
        per_seed = next(r for r in report["per_seed"] if r["arm"] == "none")
        self.assertIsNone(per_seed["primary"]["value"])
        self.assertEqual(per_seed["primary"]["reason"], "metric_undefined:undefined_in_folds")
        self.assertEqual((per_seed["primary"]["undefined"], per_seed["primary"]["n_defined"]), ([0], 4))
        self.assertIsNone(per_seed["primary"]["sd"])
        per_arm = next(r for r in report["per_arm"] if r["arm"] == "none")
        self.assertEqual((per_arm["primary"]["value"], per_arm["primary"]["reason"]),
                         (None, "metric_undefined:undefined_in_seeds"))
        # Sensitivity on concatenated E stays defined: it is reported, never substituted.
        self.assertIsNotNone(per_seed["pooled_E"]["primary"]["value"])

    def test_entry_identity_and_stage_guards(self):
        cases = [
            ("campaign_invalid", self.fx.entry("none", 0, seed=13, status="failed", reason="x", initial_state=None)),
            ("prediction_identity_mismatch", dict(self.fx.entry("none", 0), seed=99)),
            ("prediction_identity_mismatch", dict(self.fx.entry("none", 0), arm="unknown_arm")),
            ("prediction_identity_mismatch", dict(self.fx.entry("none", 0), fold=True)),
            ("report_input_invalid", dict(self.fx.entry("none", 0), extra=1)),
            ("report_input_invalid", dict(self.fx.entry("none", 0), reason="should be null")),
            ("report_input_invalid", self.fx.entry("none", 0, status="failed", reason=None)),
            ("report_input_invalid", dict(self.fx.entry("none", 0), status="succeded")),
            ("report_input_invalid", dict(self.fx.entry("none", 0), initial_state_sha256="not-a-hash")),
            ("telemetry_summary_invalid", self.fx.entry("none", 0, telemetry={"duration_s": 70.0})),
            ("telemetry_summary_invalid", self.fx.entry("none", 0, telemetry=dict(TELEMETRY_SUMMARY, schema="v0"))),
            ("telemetry_summary_invalid",
             self.fx.entry("none", 0, telemetry=dict(TELEMETRY_SUMMARY, duration_s=float("nan")))),
            ("telemetry_summary_invalid",
             self.fx.entry("none", 0, telemetry=dict(TELEMETRY_SUMMARY, train_images_processed=1.5))),
            ("telemetry_summary_invalid", self.fx.entry("none", 0, telemetry=dict(TELEMETRY_SUMMARY, status="ok"))),
            ("telemetry_summary_invalid", self.fx.entry("none", 0, telemetry=dict(TELEMETRY_SUMMARY, energy=[]))),
        ]
        for code, entry in cases:
            with self.subTest(code=code, entry={k: v for k, v in entry.items() if k != "predictions"}):
                with self.assertRaisesRegex(ValueError, f"^{code}:"):
                    self.fx.report(self.fx.entries(skip={("none", 0)}) + [entry])
        with self.assertRaisesRegex(ValueError, "^report_input_invalid:"):
            self.fx.report({"not": "a list"})

    def test_pairing_requires_same_initial_state_across_arms_and_distinct_seeds(self):
        entries = self.fx.entries(skip={("basic", 1)}) + [self.fx.entry("basic", 1, initial_state=H("warm start"))]
        with self.assertRaisesRegex(ValueError, "^pairing_violation:"):
            self.fx.report(entries)
        entries = self.fx.entries(skip={("basic", 1)}) + [self.fx.entry("basic", 1, initial_state=None)]
        report = self.fx.report(entries)
        self.assertEqual((report["pairing"]["status"], report["pairing"]["groups_incomplete"]), ("incomplete", 1))

    def test_campaign_is_reaudited_and_must_match_its_pin(self):
        with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
            self.fx.report(expected_campaign_sha256=H("another pin"))
        tampered = copy.deepcopy(self.fx.campaign)
        tampered["eligibility"]["scientific_eligible"] = True
        tampered.pop("sha256")
        tampered["sha256"] = H(tampered)
        with self.assertRaisesRegex(ValueError, "^campaign_invalid:"):
            gr.build_report(tampered, self.fx.entries(), self.fx.pools, self.fx.rows,
                            expected_campaign_sha256=tampered["sha256"], freeze=self.fx.freeze,
                            dataset_source=DATASET_SOURCE, augmentation_presets=presets())

    def test_chronology_result_is_bound_to_the_campaign(self):
        chronology = {"protocol": gc.CHRONOLOGY_PROTOCOL, "campaign_sha256": self.fx.campaign["sha256"],
                      "structural_valid": True, "execution_verified": False, "scientific_eligible": False,
                      "head": "1" * 40, "C_freeze": "2" * 40, "C_campaign": "3" * 40, "final_jobs": 50,
                      "other_campaign_final_jobs": 0, "warnings": []}
        report = self.fx.report(chronology=chronology)
        self.assertEqual(report["chronology"]["audited"], True)
        self.assertEqual(report["chronology"]["C_campaign"], "3" * 40)
        for change in ({"campaign_sha256": H("other")}, {"scientific_eligible": True}, {"structural_valid": False}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "^chronology_invalid:"):
                self.fx.report(chronology=dict(chronology, **change))

    def test_telemetry_summary_is_carried_with_fixed_fields(self):
        entries = self.fx.entries(skip={("none", 0)}) + [
            self.fx.entry("none", 0, telemetry=TELEMETRY_SUMMARY)]
        report = self.fx.report(entries)
        self.assertEqual(len(report["telemetry"]), 1)
        row = report["telemetry"][0]
        self.assertEqual(set(row), {"arm", "fold", "seed", *gr.TELEMETRY_FIELDS})
        self.assertEqual((row["duration_s"], row["gpu_temp_max_c"], row["energy_coverage_fraction"],
                          row["vram_max_reserved_bytes"], row["status"]), (312.5, 81.0, 0.95, None, "completed"))
        self.assertNotIn("utc", row)

    def test_projection_accepts_the_real_trainer_summary(self):
        try:
            from augment_atlas import train_telemetry
        except ImportError:  # trainer telemetry lands in a parallel slice
            self.skipTest("augment_atlas.train_telemetry not available")
        summary = json.loads(json.dumps(train_telemetry.summarize([], status="completed")))
        row = gr._check_telemetry(summary)
        self.assertEqual(set(row), set(gr.TELEMETRY_FIELDS))
        self.assertEqual(row["status"], "completed")
        missing = [name for name in gr.TELEMETRY_FIELDS if name != "energy_coverage_fraction" and name not in summary]
        self.assertEqual(missing, [], "TELEMETRY_FIELDS drifted from the trainer summary")

    def test_hpo_d_report_carries_candidates_and_selection_metric(self):
        report = fixture("hpo_d").report()
        self.assertEqual(report["recipe_source"], "hpo_d")
        hpo = report["recipe"]["hpo"]
        self.assertEqual((hpo["partition"], hpo["selected_candidate_id"], hpo["label"]), ("D_select", "a", "exploratory"))
        self.assertEqual([(c["id"], c["mean"], c["selected"]) for c in hpo["candidates"]],
                         [("a", 0.8, True), ("b", 0.7, False)])
        self.assertEqual(hpo["selected_mean"], 0.8)
        self.assertEqual(report["seeds"]["hpo"], [101])
        self.assertNotIn("train_ids", json.dumps(report))

    # --- no identifiers -----------------------------------------------------

    def test_public_summary_and_report_carry_no_identifier(self):
        entries = self.fx.entries(skip={("basic", 2)})
        report = self.fx.report(entries)
        summary = gr.public_summary(report)
        self.assertEqual((summary["visibility"], summary["execution_verified"], summary["scientific_eligible"]),
                         ("public", False, False))
        self.assertIn("10.1148/ryai.220232", summary["caveat"])
        self.assertEqual(summary["report_sha256"], report["sha256"])
        self.assertEqual(summary["matrix"]["missing_cells"], 1)
        for document in (summary, report):
            text = json.dumps(document, sort_keys=True)
            assert_no_identifiers(self, text, self.fx.rows)
            self.assertFalse({"group", "path", "rows", "job_id", "attempt_id", "train_ids", "test_ids"}
                             & set(keys(document)))
        public_text = json.dumps(summary)
        for private in ("predictions_content_sha256", "config_sha256", "telemetry", "records_sha256"):
            self.assertNotIn(private, public_text)
        self.assertEqual(summary["sha256"], H({k: v for k, v in summary.items() if k != "sha256"}))

    def test_identifier_scan_refuses_leaks(self):
        rows = self.fx.rows
        for document in ({"note": rows[0]["id"]}, {rows[3]["group"]: 1}, [{"x": [rows[5]["path"]]}]):
            with self.subTest(document=document), self.assertRaisesRegex(ValueError, "^identifier_leak:"):
                gr._scan_identifiers(document, rows, "probe")
        gr._scan_identifiers({"note": "g3x", "hash": H("g3")}, rows, "probe")

    def test_tampered_report_is_refused(self):
        report = self.fx.report()
        flagged = copy.deepcopy(report)
        flagged["scientific_eligible"] = True
        flagged.pop("sha256")
        flagged["sha256"] = H(flagged)
        stale = copy.deepcopy(report)
        stale["per_arm"][0]["primary"]["value"] = 0.123
        uncaveated = copy.deepcopy(report)
        uncaveated["caveats"] = uncaveated["caveats"][1:]
        uncaveated.pop("sha256")
        uncaveated["sha256"] = H(uncaveated)
        for name, document in (("flag", flagged), ("stale", stale), ("caveat", uncaveated),
                               ("extra key", dict(report, rows=[]))):
            with self.subTest(name):
                for function in (gr.public_summary, lambda r: gr.compare_campaigns(r, fixture("hpo_d").report())):
                    with self.assertRaisesRegex(ValueError, "^report_invalid:"):
                        function(document)


class CampaignComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.external = fixture().report()
        hpo = fixture("hpo_d")
        cls.hpo = hpo.report(hpo.entries(flavor=3))

    def test_difference_per_arm_is_hpo_minus_external_and_descriptive(self):
        comparison = gr.compare_campaigns(self.external, self.hpo)
        self.assertEqual(comparison, gr.compare_campaigns(self.hpo, self.external))
        self.assertEqual((comparison["inference"], comparison["hypothesis_test"], comparison["superiority_claim"]),
                         ("descriptive_only", None, False))
        self.assertEqual((comparison["execution_verified"], comparison["scientific_eligible"]), (False, False))
        self.assertEqual(comparison["difference"], "hpo_d_minus_external_fixed")
        self.assertEqual(comparison["label"], "exploratory")
        self.assertIn(gr.INBREAST_D_SELECT_WARNING, comparison["warnings"])
        self.assertTrue(any(w.startswith("D_select used by the exploratory HPO has 2 images in 1 groups")
                            for w in comparison["warnings"]))
        self.assertTrue(any("incomplete" in w for w in comparison["warnings"]))
        self.assertEqual(comparison["common_seeds"], [SEED])
        for row in comparison["arms"]:
            a = next(r for r in self.external["per_arm"] if r["arm"] == row["arm"])["primary"]["value"]
            b = next(r for r in self.hpo["per_arm"] if r["arm"] == row["arm"])["primary"]["value"]
            self.assertEqual(row["difference"], b - a)
            self.assertEqual(len(row["per_cell"]), 5)
        text = json.dumps(comparison)
        assert_no_identifiers(self, text, fixture().rows)
        self.assertTrue(any("no superiority claim" in w for w in comparison["warnings"]))
        neutral = text.replace("superiority_claim", "").replace("no superiority claim", "")
        for word in ("p_value", "significan", "superior", "better", "outperform"):
            self.assertNotIn(word, neutral)

    def test_undefined_side_gives_null_difference(self):
        hpo = fixture("hpo_d")
        report = hpo.report(hpo.entries(skip={("basic", 0)}, flavor=3))
        comparison = gr.compare_campaigns(self.external, report)
        basic = next(r for r in comparison["arms"] if r["arm"] == "basic")
        self.assertEqual((basic["difference"], basic["reason"]),
                         (None, "metric_undefined:primary_undefined_in_a_campaign"))
        self.assertEqual(basic["per_cell"][0]["reason"], "metric_undefined:missing_cell")

    def test_comparison_requires_two_sources_on_the_same_pools(self):
        with self.assertRaisesRegex(ValueError, "^campaign_comparison_invalid:"):
            gr.compare_campaigns(self.external, self.external)
        rows, pools = binary_pools(evaluation=37)
        other = Fixture(rows=rows, pools=pools).report()
        with self.assertRaisesRegex(ValueError, "^campaign_comparison_invalid:"):
            gr.compare_campaigns(other, self.hpo)


if __name__ == "__main__":
    unittest.main()
