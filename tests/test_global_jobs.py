"""Slice S0 global job contracts; synthetic declarations, no images or training."""
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from augment_atlas import global_development as gd
from augment_atlas import global_jobs as gj
from augment_atlas.contracts import ARMS
from augment_atlas.schema import IDENTITY_FIELDS, prepare_job
from test_global_development import H, signed, pool_fixture
from test_recipe_freeze import freeze_fixture

ROOT = Path(__file__).resolve().parents[1]
DATASET_SOURCE = {"adapter": "manifest_csv", "manifest": "synthetic/manifest.csv", "manifest_sha256": H("manifest bytes")}
LEGACY_JOB_ID = "c5be13e49c860292cb59c8e8b1a93d7f653f0ee4ec8be47b41f2ab9b05f12e07"  # computed at HEAD 0dd0d8a


def presets():
    """The ten real candidate-v2 presets from the repository."""
    return {arm: json.loads((ROOT / "configs/smoke/candidate-v2" / f"{arm}.json").read_text(encoding="utf-8"))["augmentation"]
            for arm in ARMS}


def binary_pools(**seed_overrides):
    rows, pools, _ = pool_fixture()
    pools = copy.deepcopy(pools)
    pools["task"]["decision_rule"] = "fixed_threshold"
    pools["partition_provenance"]["seeds"].update(seed_overrides)
    return rows, signed(pools)


def runtime():
    return {"commit": "c" * 40, "environment_sha256": H("fixture environment"), "imagenet_weights_sha256": H("fixture weights")}


def final_policy():
    _, _, freeze, _ = freeze_fixture(pools=binary_pools()[1], rows=binary_pools()[0])
    policy = copy.deepcopy(freeze["final_policy"])
    policy["augmentation_presets_sha256"] = H(presets())
    return policy


def external_freeze(rows=None, pools=None, source=None):
    if pools is None:
        rows, pools = binary_pools()
    recipe = gj.external_recipe_q2(pools, runtime(), color="grayscale_to_rgb")
    return rows, pools, gj.build_external_freeze(
        rows, pools, expected_pools_sha256=pools["sha256"], recipe=recipe,
        source=gj.Q2_EXTERNAL_SOURCE if source is None else source, runtime=runtime(),
        final_policy=final_policy(), frozen_at_utc="2026-10-19T12:00:00Z")


def register(recipe, lr):
    recipe["optimizer"] = {"name": "adamw", "parameters": {"lr": lr, "weight_decay": 0.0001}}
    recipe["scheduler"] = {"name": "cosine_per_epoch", "parameters": {"eta_min": 0.0}}
    recipe["loss"] = {"name": "cross_entropy", "parameters": {}, "learned_on": "D_fit"}


def hpo_fixture(hpo_seeds=(101,), registered=True):
    """A v1 plan/evidence/freeze whose candidates use the closed registry."""
    rows, pools = binary_pools()
    _, _, freeze, evidence = freeze_fixture(pools=pools, rows=rows)
    plan = freeze["selection_plan"]
    plan["seeds"] = list(hpo_seeds)
    for policy in (plan["final_policy"], freeze["final_policy"]):
        policy["augmentation_presets_sha256"] = H(presets())
    if registered:
        for candidate in plan["candidates"]:
            register(candidate["recipe"], candidate["parameters"]["lr"])
        register(freeze["frozen_recipe"], plan["candidates"][0]["parameters"]["lr"])
    candidates = {c["id"]: c for c in plan["candidates"]}
    template = evidence["rows"][0]
    ledger = []
    for candidate_id in sorted(candidates):
        for seed in plan["seeds"]:
            row = copy.deepcopy(template)
            row.update(candidate_id=candidate_id, seed=seed, config_sha256=H(candidates[candidate_id]),
                       artifact_sha256=H(["artifact", candidate_id, seed]), log_sha256=H(["log", candidate_id, seed]))
            row["metric"]["value"] = 0.8 if candidate_id == "a" else 0.7
            ledger.append(row)
    evidence["rows"] = ledger
    evidence["plan_sha256"] = H(plan)
    evidence = signed(evidence)
    freeze["selection_plan_sha256"] = H(plan)
    freeze["selection_evidence_sha256"] = evidence["sha256"]
    with tempfile.TemporaryDirectory() as directory:
        sealed = gd.seal_recipe_freeze(rows, pools, freeze, evidence, expected_pools_sha256=pools["sha256"],
                                       expected_plan_sha256=H(plan), output=Path(directory) / "freeze.json")
    return rows, pools, plan, evidence, sealed


def final_configs(rows, pools, freeze, kind="external_fixed", **extra):
    return gj.build_final_configs(rows, pools, expected_pools_sha256=pools["sha256"], dataset_source=DATASET_SOURCE,
                                  freeze=freeze, freeze_kind=kind, expected_freeze_sha256=freeze["sha256"],
                                  augmentation_presets=presets(), **extra)


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


class GlobalJobsTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.pools, self.freeze = external_freeze()
        self.configs = final_configs(self.rows, self.pools, self.freeze, seeds=[11])

    def temp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def materialize(self, root, configs=None):
        return gj.materialize_configs("final", root, self.configs if configs is None else configs,
                                      pools_sha256=self.pools["sha256"], freeze_sha256=self.freeze["sha256"])

    # --- Q2 external recipe and deterministic materialization -------------

    def test_q2_external_recipe_is_representable_sealed_and_never_eligible(self):
        audit = gj.audit_external_freeze(self.rows, self.pools, self.freeze, expected_pools_sha256=self.pools["sha256"])
        self.assertFalse(audit["execution_verified"])
        self.assertFalse(audit["scientific_eligible"])
        eligibility = self.freeze["eligibility"]
        self.assertEqual((eligibility["contract_eligible"], eligibility["scientific_eligible"],
                          eligibility["execution_verified"]), (False, False, False))
        self.assertIn("external_recipe_not_tuned_on_D", eligibility["reasons"])
        self.assertEqual(self.freeze["source"]["reference"], "convenção de fine-tuning, sem scores do INbreast")
        self.assertIs(self.freeze["source"]["derived_from_this_dataset_scores"], False)
        self.assertEqual(self.freeze["label"], "exploratory")
        cfg = self.configs["none-f0-s11.json"]
        self.assertEqual(cfg["training"], {
            "seed": 11, "epochs": 20, "batch_size": 16, "accumulation_steps": 1,
            "optimizer": {"name": "adamw", "lr": 0.0001, "weight_decay": 0.0001},
            "scheduler": {"name": "cosine_per_epoch", "eta_min": 0.0}, "loss": {"name": "cross_entropy"},
            "checkpoint_selection": "last_epoch", "early_stopping": False, "device": "cuda", "workers": 2,
            "deterministic": True, "autocast": False, "tf32": False})
        self.assertEqual(cfg["input"], {"size": 224, "color": "grayscale_to_rgb"})
        self.assertEqual(cfg["evaluation"]["decision"],
                         {"rule": "fixed_threshold", "threshold": 0.5, "comparison": "ge", "tie_break": None})
        self.assertEqual(cfg["model"]["init"], "imagenet_fresh")
        self.assertEqual(len(self.configs), 50)  # Q13 pilot: 10 arms x 5 folds x seed[0]
        self.assertEqual(len(final_configs(self.rows, self.pools, self.freeze)), 150)

    def test_materialization_is_byte_identical_and_idempotent(self):
        first, second = self.temp(), self.temp()
        manifest = self.materialize(first)
        rebuilt = final_configs(*external_freeze(), seeds=[11])
        self.assertEqual(self.materialize(second, rebuilt), manifest)
        self.assertEqual(tree(first), tree(second))
        before = tree(first)
        self.assertEqual(self.materialize(first), manifest)  # idempotent regeneration
        self.assertEqual(tree(first), before)
        self.assertEqual(len(manifest["files"]), 50)
        self.assertEqual(manifest["sha256"], H({k: v for k, v in manifest.items() if k != "sha256"}))
        directory = first / "configs/global/synthetic_fixture/final"
        for entry in manifest["files"]:
            data = (first / entry["path"]).read_bytes()
            self.assertEqual(gj._bytes_sha256(data), entry["sha256"])
            self.assertNotIn(b"\r\n", data)
        text = "".join(path.read_text(encoding="ascii") for path in directory.glob("*.json"))
        for row in self.rows:  # private names and hashes only, never record IDs or groups
            self.assertNotIn(f'"{row["id"]}"', text)
            self.assertNotIn(f'"{row["group"]}"', text)

    def test_conflicting_or_stray_materialization_fails_without_changes(self):
        root = self.temp()
        self.materialize(root)
        directory = root / "configs/global/synthetic_fixture/final"
        target = directory / "none-f0-s11.json"
        target.write_bytes(target.read_bytes().replace(b'"lr": 0.0001', b'"lr": 0.001'))
        mutated = tree(root)
        with self.assertRaisesRegex(ValueError, "^materialization_conflict: "):
            self.materialize(root)
        self.assertEqual(tree(root), mutated)
        root = self.temp()
        self.materialize(root)
        (root / "configs/global/synthetic_fixture/final/extra.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "^materialization_conflict: "):
            self.materialize(root)
        other = copy.deepcopy(self.configs)
        other["none-f0-s11.json"]["global"]["fold"] = 1
        with self.assertRaisesRegex(ValueError, "^global_config_invalid: "):
            self.materialize(self.temp(), other)

    # --- forbidden warm start / evaluation scope ---------------------------

    def test_init_checkpoint_and_evaluation_scope_are_rejected_anywhere(self):
        cfg = self.configs["basic-f2-s11.json"]
        for key in sorted(gj.FORBIDDEN_KEYS):
            for section in (None, "training", "model", "global", "augmentation"):
                mutant = copy.deepcopy(cfg)
                (mutant if section is None else mutant[section])[key] = "runs/hpo/a/best.pt" if key != "warm_start" else True
                with self.subTest(key=key, section=section), self.assertRaisesRegex(ValueError, "^init_checkpoint_forbidden: "):
                    gj.validate_global_config(mutant)
        bad = presets()
        bad["basic"]["evaluation_scope"] = "selection-only"
        with self.assertRaisesRegex(ValueError, "^freeze_not_sealed: "):  # presets are bound by the sealed hash
            gj.build_final_configs(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"],
                                   dataset_source=DATASET_SOURCE, freeze=self.freeze, freeze_kind="external_fixed",
                                   expected_freeze_sha256=self.freeze["sha256"], augmentation_presets=bad)
        unknown = copy.deepcopy(cfg)
        unknown["training"]["resume"] = True
        with self.assertRaisesRegex(ValueError, "^global_config_invalid: "):
            gj.validate_global_config(unknown)

    # --- closed recipe -> trainer registry --------------------------------

    def test_optimizer_scheduler_loss_outside_registry_are_rejected(self):
        recipe = self.freeze["recipe"]
        mutants = [lambda r: r["optimizer"].update(name="sgd"),
                   lambda r: r["optimizer"]["parameters"].update(momentum=0.9),
                   lambda r: r["optimizer"]["parameters"].update(lr=0.0),
                   lambda r: r["optimizer"]["parameters"].update(lr=float("nan")),
                   lambda r: r["optimizer"]["parameters"].update(lr=True),
                   lambda r: r["scheduler"].update(name="cosine"),
                   lambda r: r["scheduler"]["parameters"].update(eta_min=1.0),
                   lambda r: r["loss"].update(name="focal"),
                   lambda r: r["loss"].update(parameters={"class_weights": [1.0, 2.0]}),
                   lambda r: r.update(epochs=0)]
        for mutate in mutants:
            mutant = copy.deepcopy(recipe)
            mutate(mutant)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "^recipe_adapter_unsupported: "):
                gj.recipe_to_training(mutant, 11)
        cfg = copy.deepcopy(self.configs["none-f0-s11.json"])
        cfg["training"]["optimizer"]["name"] = "sgd"
        with self.assertRaisesRegex(ValueError, "^recipe_adapter_unsupported: "):
            gj.validate_global_config(cfg)
        cfg = copy.deepcopy(self.configs["none-f0-s11.json"])
        cfg["training"]["optimizer"]["momentum"] = 0.9
        with self.assertRaisesRegex(ValueError, "^recipe_adapter_unsupported: "):
            gj.validate_global_config(cfg)
        # The accepted v1 fixture (sgd/cosine/class weights) is a valid v1
        # freeze but cannot be mapped onto the global trainer.
        rows, pools, plan, evidence, freeze = hpo_fixture(registered=False)
        with self.assertRaisesRegex(ValueError, "^recipe_adapter_unsupported: "):
            final_configs(rows, pools, freeze, "hpo_d", evidence=evidence, expected_plan_sha256=H(plan))

    def test_training_must_equal_recipe_mapping(self):
        cfg = self.configs["gamma_light-f3-s11.json"]
        gj.validate_global_config(cfg, recipe=self.freeze["recipe"])
        for mutate in (lambda r: r["optimizer"]["parameters"].update(lr=0.0003),
                       lambda r: r.update(epochs=19), lambda r: r.update(batch_size=8),
                       lambda r: r["input"].update(color="rgb"),
                       lambda r: r["model"].update(weights_sha256=H("other weights"))):
            recipe = copy.deepcopy(self.freeze["recipe"])
            mutate(recipe)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "^recipe_mismatch: "):
                gj.validate_global_config(cfg, recipe=recipe)
        for key, value in (("checkpoint_selection", "best_selection_loss"), ("early_stopping", True),
                           ("workers", 0), ("autocast", True), ("deterministic", 1)):
            mutant = copy.deepcopy(cfg)
            mutant["training"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "^global_config_invalid: "):
                gj.validate_global_config(mutant)

    def test_binary_decision_is_fixed_half_ge_and_scores_only_binary_is_unsupported(self):
        cfg = copy.deepcopy(self.configs["none-f0-s11.json"])
        for decision in ({"rule": "fixed_threshold", "threshold": 0.4, "comparison": "ge", "tie_break": None},
                         {"rule": "fixed_threshold", "threshold": 0.5, "comparison": "gt", "tie_break": None}):
            cfg["evaluation"]["decision"] = decision
            with self.subTest(decision=decision), self.assertRaisesRegex(ValueError, "^recipe_adapter_unsupported: "):
                gj.validate_global_config(cfg)
        rows, pools, _ = pool_fixture()  # binary scores_only in the v1 fixture
        with self.assertRaisesRegex(ValueError, "^recipe_adapter_unsupported: "):
            external_freeze(rows, pools)

    # --- queue identity ----------------------------------------------------

    def test_legacy_identity_fields_and_job_ids_are_unchanged(self):
        self.assertEqual(IDENTITY_FIELDS, ("protocol_version", "dataset", "dataset_version", "split_hash", "recipe_hash",
                                           "phase", "fold", "train_seed", "arm", "code_commit"))
        legacy = dict(protocol_version="candidate-v2", dataset="synthetic-tiny", dataset_version="v1", split_hash="a" * 64,
                      recipe_hash="b" * 64, phase="smoke", fold=0, train_seed=3, arm="basic", code_commit="c" * 40,
                      config="configs/smoke/candidate-v2/basic.json", config_sha256="d" * 64, dependencies=[],
                      compatibility={"min_memory_mb": 1000})
        self.assertEqual(prepare_job(legacy)["job_id"], LEGACY_JOB_ID)

    def test_final_view_and_job_specs_bind_fold_freeze_and_campaign(self):
        campaign = H("campaign fixture")
        specs, views = [], {}
        for name in sorted(self.configs):
            cfg = self.configs[name]
            fold = cfg["global"]["fold"]
            views.setdefault(fold, gj.derive_final_view_doc(self.rows, self.pools, fold, expected_pools_sha256=self.pools["sha256"],
                                                            freeze_sha256=self.freeze["sha256"]))
            path = f"configs/global/synthetic_fixture/final/{name}"
            spec = gj.final_job_spec(cfg, views[fold], config_path=path, config_sha256=gj._bytes_sha256(gj.render_config_bytes(cfg)),
                                     code_commit="e" * 40, campaign_sha256=campaign)
            self.assertEqual((spec["phase"], spec["protocol_version"], spec["fold"], spec["recipe_hash"], spec["split_hash"]),
                             ("final", "global-final-v1", fold, self.freeze["sha256"], views[fold]["sha256"]))
            self.assertEqual(spec["global"]["campaign_sha256"], campaign)
            self.assertEqual(spec["dependencies"], [])
            specs.append(spec)
        self.assertEqual(len({spec["job_id"] for spec in specs}), 50)
        tests = []
        for fold, view in views.items():
            self.assertNotIn("selection", view)
            self.assertFalse(set(view["test"]) & set(self.pools["lists"]["D"]))  # D is never tested
            self.assertTrue(set(self.pools["lists"]["D"]) <= set(view["train"]))
            self.assertEqual(set(view["train"]) | set(view["test"]), {row["id"] for row in self.rows})
            tests.extend(view["test"])
        self.assertEqual(sorted(tests), self.pools["lists"]["E"])  # E covered exactly once
        cfg = self.configs["none-f1-s11.json"]
        sha = gj._bytes_sha256(gj.render_config_bytes(cfg))
        path = "configs/global/synthetic_fixture/final/none-f1-s11.json"
        with self.assertRaisesRegex(ValueError, "^view_hash_mismatch: "):
            gj.final_job_spec(cfg, views[0], config_path=path, config_sha256=sha, code_commit="e" * 40, campaign_sha256=campaign)
        other_freeze = gj.derive_final_view_doc(self.rows, self.pools, 1, expected_pools_sha256=self.pools["sha256"],
                                                freeze_sha256=H("other freeze"))
        self.assertNotEqual(other_freeze["sha256"], views[1]["sha256"])
        with self.assertRaisesRegex(ValueError, "^view_hash_mismatch: "):
            gj.final_job_spec(cfg, other_freeze, config_path=path, config_sha256=sha, code_commit="e" * 40, campaign_sha256=campaign)
        tampered = copy.deepcopy(views[1])
        tampered["test"].append(self.pools["lists"]["D"][0])
        with self.assertRaisesRegex(ValueError, "^view_hash_mismatch: "):
            gj.final_job_spec(cfg, tampered, config_path=path, config_sha256=sha, code_commit="e" * 40, campaign_sha256=campaign)
        with self.assertRaisesRegex(ValueError, "^recipe_mismatch: "):
            gj.final_job_spec(cfg, views[1], config_path=path, config_sha256=H("x"), code_commit="e" * 40, campaign_sha256=campaign)
        with self.assertRaisesRegex(ValueError, "^global_config_invalid: "):
            gj.final_job_spec(cfg, views[1], config_path="configs/smoke/none.json", config_sha256=sha,
                              code_commit="e" * 40, campaign_sha256=campaign)

    # --- HPO producer side ---------------------------------------------------

    def test_hpo_configs_specs_and_arm_none_only(self):
        rows, pools, plan, evidence, freeze = hpo_fixture()
        view_sha, space_sha = H("hpo view"), H("search space")
        build = lambda: gj.build_hpo_configs(rows, pools, expected_pools_sha256=pools["sha256"], dataset_source=DATASET_SOURCE,
                                             plan=plan, expected_plan_sha256=H(plan), search_space_sha256=space_sha,
                                             hpo_view_sha256=view_sha)
        configs = build()
        self.assertEqual(sorted(configs), ["a-s101.json", "b-s101.json"])
        candidates = {c["id"]: c for c in plan["candidates"]}
        roots = [self.temp(), self.temp()]
        manifests = [gj.materialize_configs("hpo", root, cfgs, pools_sha256=pools["sha256"], plan_sha256=H(plan))
                     for root, cfgs in zip(roots, (configs, build()))]
        self.assertEqual(manifests[0], manifests[1])
        self.assertEqual(tree(roots[0]), tree(roots[1]))
        for name, cfg in configs.items():
            candidate = candidates[cfg["global"]["candidate_id"]]
            gj.validate_global_config(cfg, recipe=candidate["recipe"])
            self.assertEqual(cfg["augmentation"], {"arm": "none", "ops": []})
            self.assertIsNone(cfg["global"]["fold"])
            self.assertIsNone(cfg["global"]["freeze_sha256"])
            spec = gj.hpo_job_spec(cfg, config_path=f"configs/global/synthetic_fixture/hpo/{name}",
                                   config_sha256=gj._bytes_sha256(gj.render_config_bytes(cfg)), code_commit="e" * 40)
            self.assertEqual((spec["phase"], spec["arm"], spec["fold"], spec["split_hash"], spec["recipe_hash"]),
                             ("hpo", "none", 0, view_sha, H(candidate)))
            self.assertEqual(spec["recipe_hash"], next(r["config_sha256"] for r in evidence["rows"]
                                                        if r["candidate_id"] == candidate["id"]))
        cfg = configs["a-s101.json"]
        for arm in ("basic", "gamma_light"):
            mutant = copy.deepcopy(cfg)
            mutant["augmentation"] = presets()[arm]
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "^hpo_augmentation_forbidden: "):
                gj.validate_global_config(mutant)
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "^hpo_augmentation_forbidden: "):
                gj.hpo_job_spec(mutant, config_path="configs/global/synthetic_fixture/hpo/a-s101.json",
                                config_sha256=gj._bytes_sha256(gj.render_config_bytes(mutant)), code_commit="e" * 40)
        for key, value in (("fold", 0), ("freeze_sha256", H("f")), ("hpo_view_sha256", None)):
            mutant = copy.deepcopy(cfg)
            mutant["global"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "^global_config_invalid: "):
                gj.validate_global_config(mutant)
        with self.assertRaisesRegex(ValueError, "^plan_invalid: "):
            gj.build_hpo_configs(rows, pools, expected_pools_sha256=pools["sha256"], dataset_source=DATASET_SOURCE,
                                 plan=plan, expected_plan_sha256=H("other plan"), search_space_sha256=space_sha,
                                 hpo_view_sha256=view_sha)

    def test_hpo_freeze_materializes_final_and_requires_disjoint_seeds(self):
        rows, pools, plan, evidence, freeze = hpo_fixture()
        configs = final_configs(rows, pools, freeze, "hpo_d", evidence=evidence, expected_plan_sha256=H(plan), seeds=[11])
        cfg = configs["clahe_heavy-f4-s11.json"]
        self.assertEqual(cfg["global"]["freeze_file"], f"global/synthetic_fixture/freeze-{freeze['sha256']}.json")
        self.assertEqual(cfg["training"]["optimizer"]["lr"], 0.01)  # winner "a"
        with self.assertRaisesRegex(ValueError, "^freeze_not_sealed: "):
            final_configs(rows, pools, freeze, "hpo_d")  # evidence and plan anchor are mandatory
        with self.assertRaisesRegex(ValueError, "^freeze_not_sealed: "):
            final_configs(rows, pools, freeze, "hpo_d", evidence=evidence, expected_plan_sha256=H("forged plan"))
        rows, pools, plan, evidence, freeze = hpo_fixture(hpo_seeds=(11, 13))  # overlaps final seeds 11/13/19
        with self.assertRaisesRegex(ValueError, "^seed_reuse_forbidden: "):
            final_configs(rows, pools, freeze, "hpo_d", evidence=evidence, expected_plan_sha256=H(plan))
        rows, pools = binary_pools(evaluation=17)  # evaluation seed == development seed
        rows, pools, freeze = external_freeze(rows, pools)
        with self.assertRaisesRegex(ValueError, "^seed_reuse_forbidden: "):
            final_configs(rows, pools, freeze)
        for seeds in ([19, 11], [7], [], [11, 11]):
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "^global_config_invalid: "):
                final_configs(self.rows, self.pools, self.freeze, seeds=seeds)

    # --- external freeze and anchors -----------------------------------------

    def test_external_freeze_contamination_tamper_and_anchor_mismatch(self):
        source = dict(gj.Q2_EXTERNAL_SOURCE, derived_from_this_dataset_scores=True)
        with self.assertRaisesRegex(ValueError, "^external_recipe_contaminated: "):
            external_freeze(source=source)
        for mutate in (lambda d: d.update(label="confirmatory_candidate"),
                       lambda d: d["eligibility"].update(scientific_eligible=True),
                       lambda d: d["eligibility"].update(execution_verified=True),
                       lambda d: d["selection_scope"].update(data_used_for_selection="D_select"),
                       lambda d: d["final_policy"].update(warm_start=True),
                       lambda d: d["recipe"]["optimizer"].update(name="sgd")):
            mutant = copy.deepcopy(self.freeze)
            mutate(mutant)
            mutant = signed(mutant)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "^(freeze_not_sealed|recipe_adapter_unsupported): "):
                gj.audit_external_freeze(self.rows, self.pools, mutant, expected_pools_sha256=self.pools["sha256"])
        with self.assertRaisesRegex(ValueError, "^freeze_not_sealed: "):
            final_configs(self.rows, self.pools, signed(dict(self.freeze, frozen_at_utc="2026-10-19T12:00:00Z ")))
        with self.assertRaisesRegex(ValueError, "^pools_anchor_mismatch: "):
            gj.build_final_configs(self.rows, self.pools, expected_pools_sha256=H("other pools"), dataset_source=DATASET_SOURCE,
                                   freeze=self.freeze, freeze_kind="external_fixed",
                                   expected_freeze_sha256=self.freeze["sha256"], augmentation_presets=presets())
        rows = copy.deepcopy(self.rows)
        rows[0]["label"] = 1 - rows[0]["label"]
        with self.assertRaisesRegex(ValueError, "^records_identity_mismatch: "):
            gj.derive_final_view_doc(rows, self.pools, 0, expected_pools_sha256=self.pools["sha256"],
                                     freeze_sha256=self.freeze["sha256"])

    # --- documentation and import hygiene ------------------------------------

    def test_error_codes_belong_to_the_spec_catalog_and_caveat_is_documented(self):
        spec = (ROOT / "docs/global-next-integration.md").read_text(encoding="utf-8")
        section = spec[spec.index("### 6.11"):spec.index("## 7.")]
        catalog = set(re.findall(r"`([A-Za-z_]+)(?::<reason>)?`", section))
        source = (ROOT / "src/augment_atlas/global_jobs.py").read_text(encoding="utf-8")
        used = set(re.findall(r'_fail\("([A-Za-z_]+)"', source)) | {"global_config_invalid"}
        self.assertTrue(used <= catalog, used - catalog)
        self.assertIn("10.1148/ryai.220232", gj.__doc__)
        self.assertIn("never a test set", gj.__doc__)
        self.assertEqual(gj.TRAINER_CONSTANTS_V1["gradient_clip_max_norm"], 1.0)
        self.assertEqual(gj.TRAINER_CONSTANTS_V1["checkpoint_selection"], "last_epoch")

    def test_import_has_no_ml_dependencies(self):
        code = ("import sys; import augment_atlas.global_jobs; "
                "print(sorted(m for m in ('torch','torchvision','numpy','PIL','cv2') if m in sys.modules))")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                                env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
