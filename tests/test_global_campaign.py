"""Slice S3 campaign seal, external freeze seal and Q6 code identity (synthetic, offline)."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from augment_atlas import global_campaign as gc
from augment_atlas import global_jobs as gj
from augment_atlas.contracts import ARMS, canonical_hash
from augment_atlas.queue import ANCHOR_CAVEAT
from augment_atlas.schema import new_record
from test_global_development import H, signed
from test_global_jobs import DATASET_SOURCE, binary_pools, external_freeze, hpo_fixture, presets, runtime

IMAGE = "sha256:" + "d" * 64
CODE = "e" * 40
STAGES = [{"stage": 1, "seeds": [11]}, {"stage": 2, "seeds": [13, 19]}]
SEALED_AT = "2026-10-19T13:00:00Z"


def identity(freeze_commit="c" * 40, code_commit=CODE, changed=None, tree=None, **changes):
    changed = ["configs/global/synthetic_fixture/final/MANIFEST.json"] if changed is None else changed
    tree = tree or H("code tree")
    value = {"policy": gc.Q6_POLICY, "policy_status": gc.Q6_POLICY_STATUS, "freeze_commit": freeze_commit,
             "code_commit": code_commit, "image_id": IMAGE, "changed_paths": sorted(changed),
             "freeze_code_tree_sha256": tree, "final_code_tree_sha256": tree}
    value.update(changes)
    return value


def seal(rows, pools, freeze, kind="external_fixed", **changes):
    args = dict(expected_pools_sha256=pools["sha256"], freeze=freeze, freeze_kind=kind,
                expected_freeze_sha256=freeze["sha256"], dataset_source=DATASET_SOURCE,
                augmentation_presets=presets(), execution_stages=copy.deepcopy(STAGES), code_commit=CODE,
                image_id=IMAGE, code_identity=identity(freeze_commit=freeze["runtime"]["commit"]),
                label="exploratory", caveats=list(gc.CAMPAIGN_CAVEATS), sealed_at_utc=SEALED_AT, queue_records=[])
    args.update(changes)
    return gc.seal_campaign(rows, pools, **args)


def audit(rows, pools, freeze, campaign, **changes):
    args = dict(expected_campaign_sha256=campaign["sha256"], freeze=freeze, dataset_source=DATASET_SOURCE,
                augmentation_presets=presets())
    args.update(changes)
    return gc.audit_campaign(rows, pools, campaign, **args)


def final_spec(rows, pools, freeze, campaign_sha256, *, arm="none", fold=0, seed=11, kind="external_fixed",
               code_commit=CODE, **extra):
    cfg = gj.build_final_configs(rows, pools, expected_pools_sha256=pools["sha256"], dataset_source=DATASET_SOURCE,
                                 freeze=freeze, freeze_kind=kind, expected_freeze_sha256=freeze["sha256"],
                                 augmentation_presets=presets(), seeds=[seed], **extra)[f"{arm}-f{fold}-s{seed}.json"]
    view = gj.derive_final_view_doc(rows, pools, fold, expected_pools_sha256=pools["sha256"],
                                    freeze_sha256=freeze["sha256"])
    path = f"{gj.config_dir(pools['dataset']['id'], 'final')}/{arm}-f{fold}-s{seed}.json"
    return gj.final_job_spec(cfg, view, config_path=path, config_sha256=gj._bytes_sha256(gj.render_config_bytes(cfg)),
                             code_commit=code_commit, campaign_sha256=campaign_sha256)


def inbreast_pools():
    rows, pools = binary_pools()
    pools = copy.deepcopy(pools)
    pools["dataset"]["id"] = "inbreast"
    pools["grouping"]["prior_exposure"] = "known"
    return rows, signed(pools)


def resign(document):
    value = copy.deepcopy(document)
    value.pop("sha256", None)
    value["sha256"] = canonical_hash(value)
    return value


class CampaignSealTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.pools, self.freeze = external_freeze()

    def test_pilot_declares_full_matrix_and_authorizes_only_stage_one(self):
        before = copy.deepcopy((self.rows, self.pools, self.freeze))
        campaign = seal(self.rows, self.pools, self.freeze)
        self.assertEqual(before, (self.rows, self.pools, self.freeze))
        self.assertEqual(campaign["sha256"], seal(self.rows, self.pools, self.freeze)["sha256"])
        matrix = campaign["matrix"]
        self.assertEqual((matrix["declared_cells"], matrix["authorized_cells"], matrix["missing_cells"]), (150, 50, 100))
        self.assertEqual(matrix["missing"], [{"stage": 2, "seeds": [13, 19], "cells": 100,
                                              "reason": "stage_not_authorized_by_this_campaign"}])
        self.assertFalse(matrix["complete"])
        self.assertEqual([s["status"] for s in campaign["execution_stages"]], ["authorized", "declared_not_authorized"])
        self.assertEqual(campaign["seeds"], [11, 13, 19])
        self.assertEqual(campaign["configs"]["count"], 50)
        self.assertEqual(campaign["arms"], list(ARMS))
        with tempfile.TemporaryDirectory() as directory:
            configs = gj.build_final_configs(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"],
                                             dataset_source=DATASET_SOURCE, freeze=self.freeze,
                                             freeze_kind="external_fixed", expected_freeze_sha256=self.freeze["sha256"],
                                             augmentation_presets=presets(), seeds=[11])
            manifest = gj.materialize_configs("final", directory, configs, pools_sha256=self.pools["sha256"],
                                              freeze_sha256=self.freeze["sha256"])
        self.assertEqual(campaign["configs"]["manifest_sha256"], manifest["sha256"])
        self.assertEqual(campaign["evaluation"]["decision"],
                         {"rule": "fixed_threshold", "threshold": 0.5, "comparison": "ge", "tie_break": None})
        self.assertEqual(campaign["runtime"]["trainer_constants"], gj.TRAINER_CONSTANTS_V1)
        self.assertEqual(campaign["q6_policy"], "same_code_identity_v1")
        self.assertEqual(campaign["code_identity"]["policy_status"], "pending_eduardo_confirmation")
        self.assertEqual(campaign["label"], "exploratory")
        self.assertEqual(campaign["label_reasons"], ["patient_identity_unproven", "analysis_plan_missing"])
        self.assertEqual(campaign["eligibility"], {
            "execution_verified": False, "scientific_eligible": False,
            "blockers": ["execution_unverified", "runtime_isolation_unverified", "chronology_unverified",
                         "patient_identity_unproven", "analysis_plan_missing"]})
        result = audit(self.rows, self.pools, self.freeze, campaign, expected_manifest_sha256=manifest["sha256"])
        self.assertEqual((result["execution_verified"], result["scientific_eligible"]), (False, False))
        self.assertEqual(result["authorized_seeds"], [11])

    def test_single_stage_campaign_is_the_complete_matrix(self):
        campaign = seal(self.rows, self.pools, self.freeze, execution_stages=[{"stage": 1, "seeds": [11, 13, 19]}])
        self.assertEqual(campaign["configs"]["count"], 150)
        self.assertTrue(campaign["matrix"]["complete"])
        self.assertEqual(campaign["matrix"]["missing"], [])

    def test_stages_must_partition_policy_seeds_in_order(self):
        for stages in ([{"stage": 1, "seeds": [11]}], [{"stage": 1, "seeds": [13]}, {"stage": 2, "seeds": [11, 19]}],
                       [{"stage": 2, "seeds": [11, 13, 19]}], [{"stage": 1, "seeds": [11, 13, 19], "extra": 1}],
                       [{"stage": 1, "seeds": [11, 13, 19, 23]}], []):
            with self.subTest(stages=stages), self.assertRaisesRegex(ValueError, "^campaign_invalid:"):
                seal(self.rows, self.pools, self.freeze, execution_stages=stages)

    def test_existing_final_job_refuses_seal(self):
        record = new_record(final_spec(self.rows, self.pools, self.freeze, H("other campaign")))
        with self.assertRaisesRegex(ValueError, "^final_job_before_seal:"):
            seal(self.rows, self.pools, self.freeze, queue_records=[record])
        other_pools = copy.deepcopy(record)
        other_pools["spec"]["global"]["pools_sha256"] = H("other pools")
        seal(self.rows, self.pools, self.freeze, queue_records=[other_pools])

    def test_final_seed_equal_to_hpo_seed_is_forbidden(self):
        rows, pools, _, evidence, freeze = hpo_fixture(hpo_seeds=(11,))
        with self.assertRaisesRegex(ValueError, "^seed_reuse_forbidden:"):
            seal(rows, pools, freeze, kind="hpo_d", evidence=evidence, search_space_sha256=H("search space"))

    def test_evaluation_seed_equal_to_development_seed_is_forbidden(self):
        rows, pools = binary_pools(evaluation=17)
        rows, pools, freeze = external_freeze(rows, pools)
        with self.assertRaisesRegex(ValueError, "^seed_reuse_forbidden:"):
            seal(rows, pools, freeze)

    def test_hpo_d_campaign_binds_plan_evidence_and_search_space(self):
        rows, pools, plan, evidence, freeze = hpo_fixture()
        campaign = seal(rows, pools, freeze, kind="hpo_d", evidence=evidence, search_space_sha256=H("search space"))
        self.assertEqual(campaign["freeze"], {"kind": "hpo_d", "sha256": freeze["sha256"], "plan_sha256": H(plan),
                                              "evidence_sha256": evidence["sha256"],
                                              "search_space_sha256": H("search space")})
        audit(rows, pools, freeze, campaign, evidence=evidence)
        with self.assertRaisesRegex(ValueError, "^campaign_invalid:"):
            seal(rows, pools, freeze, kind="hpo_d", evidence=evidence)
        with self.assertRaisesRegex(ValueError, "^freeze_not_sealed:"):
            seal(rows, pools, freeze, kind="hpo_d", search_space_sha256=H("search space"))

    def test_external_recipe_derived_from_dataset_scores_is_refused(self):
        source = dict(gj.Q2_EXTERNAL_SOURCE, derived_from_this_dataset_scores=True)
        recipe = gj.external_recipe_q2(self.pools, runtime(), color="grayscale_to_rgb")
        with self.assertRaisesRegex(ValueError, "^external_recipe_contaminated:"):
            gc.seal_external_freeze(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"], recipe=recipe,
                                    source=source, runtime=runtime(), final_policy=self.freeze["final_policy"],
                                    frozen_at_utc="2026-10-19T12:00:00Z")
        tampered = copy.deepcopy(self.freeze)
        tampered["source"]["derived_from_this_dataset_scores"] = True
        tampered = resign(tampered)
        with self.assertRaisesRegex(ValueError, "^external_recipe_contaminated:"):
            gc.audit_external_freeze(self.rows, self.pools, tampered, expected_pools_sha256=self.pools["sha256"])
        with self.assertRaisesRegex(ValueError, "^external_recipe_contaminated:"):
            seal(self.rows, self.pools, tampered)

    def test_seal_external_freeze_is_no_clobber(self):
        recipe = gj.external_recipe_q2(self.pools, runtime(), color="grayscale_to_rgb")
        args = dict(expected_pools_sha256=self.pools["sha256"], recipe=recipe, source=gj.Q2_EXTERNAL_SOURCE,
                    runtime=runtime(), final_policy=self.freeze["final_policy"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "external-freeze.json"
            first = gc.seal_external_freeze(self.rows, self.pools, frozen_at_utc="2026-10-19T12:00:00Z",
                                            output=output, **args)
            self.assertEqual(first, self.freeze)
            data = output.read_bytes()
            gc.seal_external_freeze(self.rows, self.pools, frozen_at_utc="2026-10-19T12:00:00Z", output=output, **args)
            with self.assertRaisesRegex(ValueError, "no-clobber"):
                gc.seal_external_freeze(self.rows, self.pools, frozen_at_utc="2026-10-19T12:30:00Z",
                                        output=output, **args)
            self.assertEqual(data, output.read_bytes())

    def test_inbreast_confirmatory_label_is_refused(self):
        rows, pools = inbreast_pools()
        rows, pools, freeze = external_freeze(rows, pools)
        with self.assertRaisesRegex(ValueError, "^campaign_label_forbidden:"):
            seal(rows, pools, freeze, label="confirmatory_candidate", analysis_plan_sha256=H("analysis plan"))
        campaign = seal(rows, pools, freeze, analysis_plan_sha256=H("analysis plan"),
                        code_identity=identity(changed=["configs/global/inbreast/final/MANIFEST.json"]))
        self.assertEqual(campaign["label_reasons"], ["prior_exposure_known", "patient_identity_unproven",
                                                     "external_freeze_on_exposed_dataset", "inbreast_pilot_exploratory"])
        with self.assertRaisesRegex(ValueError, "^campaign_label_forbidden:"):
            seal(self.rows, self.pools, self.freeze, label="confirmatory_candidate")
        with self.assertRaisesRegex(ValueError, "^campaign_label_forbidden:"):
            seal(self.rows, self.pools, self.freeze, label="confirmatory")

    def test_missing_bradshaw_caveat_is_refused(self):
        without = [c for c in gc.CAMPAIGN_CAVEATS if c != gc.BRADSHAW_CAVEAT_ID]
        for caveats in (without, [], None, list(reversed(gc.CAMPAIGN_CAVEATS)), list(gc.CAMPAIGN_CAVEATS) + ["x"]):
            with self.subTest(caveats=caveats), self.assertRaisesRegex(ValueError, "^campaign_caveat_missing:"):
                seal(self.rows, self.pools, self.freeze, caveats=caveats)
        campaign = seal(self.rows, self.pools, self.freeze)
        stripped = resign(dict(campaign, caveats=without))
        with self.assertRaisesRegex(ValueError, "^campaign_caveat_missing:"):
            audit(self.rows, self.pools, self.freeze, stripped, expected_campaign_sha256=stripped["sha256"])
        # The three textual presences of the mandatory caveat.
        for text in (gc.SELECTION_BIAS_CAVEAT, ANCHOR_CAVEAT, gc.__doc__):
            self.assertIn("Bradshaw 2023", text)
            self.assertIn("10.1148/ryai.220232", text)
        self.assertIn("Bradshaw 2023", gc.audit_campaign(self.rows, self.pools, campaign,
                                                         expected_campaign_sha256=campaign["sha256"], freeze=self.freeze,
                                                         dataset_source=DATASET_SOURCE,
                                                         augmentation_presets=presets())["caveat"])

    def test_every_global_module_docstring_cites_the_caveat(self):
        import ast
        modules = sorted((Path(gc.__file__).parent).glob("global_*.py"))
        self.assertGreaterEqual(len(modules), 4)
        for module in modules:
            doc = ast.get_docstring(ast.parse(module.read_text(encoding="utf-8"))) or ""
            with self.subTest(module=module.name):
                self.assertIn("Bradshaw 2023", doc)
                self.assertIn("10.1148/ryai.220232", doc)

    def test_audit_refuses_reasigned_tampering(self):
        campaign = seal(self.rows, self.pools, self.freeze)
        mutations = {
            "matrix": lambda c: c["matrix"].update(authorized_cells=150, missing_cells=0, missing=[], complete=True),
            "flags": lambda c: c["eligibility"].update(scientific_eligible=True),
            "blockers": lambda c: c["eligibility"].update(blockers=[]),
            "label_reasons": lambda c: c.update(label_reasons=[]),
            "decision": lambda c: c["evaluation"]["decision"].update(comparison="gt"),
            "configs": lambda c: c["configs"].update(count=150),
            "seeds": lambda c: c.update(seeds=[11, 13, 23]),
            "aggregation": lambda c: c["evaluation"]["aggregation"].update(primary="pooled"),
            "trainer": lambda c: c["runtime"]["trainer_constants"].update(workers=0),
        }
        for name, mutate in mutations.items():
            tampered = copy.deepcopy(campaign)
            mutate(tampered)
            tampered = resign(tampered)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "^(campaign_invalid|seed_reuse_forbidden):"):
                audit(self.rows, self.pools, self.freeze, tampered, expected_campaign_sha256=tampered["sha256"])
        with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
            audit(self.rows, self.pools, self.freeze, campaign, expected_campaign_sha256=H("other"))
        unsigned = copy.deepcopy(campaign)
        unsigned["label"] = "confirmatory_candidate"
        with self.assertRaisesRegex(ValueError, "^campaign_invalid:"):
            audit(self.rows, self.pools, self.freeze, unsigned)
        with self.assertRaisesRegex(ValueError, "^config_identity_mismatch:"):
            audit(self.rows, self.pools, self.freeze, campaign, expected_manifest_sha256=H("other manifest"))
        with self.assertRaisesRegex(ValueError, "^config_identity_mismatch:"):
            seal(self.rows, self.pools, self.freeze, expected_manifest_sha256=H("other manifest"))
        other_presets = presets()
        other_presets["none"] = {"arm": "none", "ops": [], "backend": "x"}
        with self.assertRaises(ValueError):
            audit(self.rows, self.pools, self.freeze, campaign, augmentation_presets=other_presets)

    def test_reseal_with_other_seeds_or_recipe_changes_the_hash(self):
        campaign = seal(self.rows, self.pools, self.freeze)
        other_stage = seal(self.rows, self.pools, self.freeze, execution_stages=[{"stage": 1, "seeds": [11, 13, 19]}])
        self.assertNotEqual(campaign["sha256"], other_stage["sha256"])
        recipe = copy.deepcopy(self.freeze["recipe"])
        recipe["optimizer"]["parameters"]["lr"] = 0.001
        other_freeze = gc.seal_external_freeze(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"],
                                               recipe=recipe, source=gj.Q2_EXTERNAL_SOURCE, runtime=runtime(),
                                               final_policy=self.freeze["final_policy"],
                                               frozen_at_utc=self.freeze["frozen_at_utc"])
        self.assertNotEqual(other_freeze["sha256"], self.freeze["sha256"])
        self.assertNotEqual(seal(self.rows, self.pools, other_freeze)["sha256"], campaign["sha256"])

    def test_anchors_carry_hashes_only(self):
        campaign = seal(self.rows, self.pools, self.freeze)
        for kind, document in (("recipe_freeze", self.freeze), ("campaign", campaign),
                               ("pools", self.pools)):
            anchor = gc.build_anchor(kind, document, document_utc="2026-10-19T11:00:00Z" if kind == "pools" else None)
            text = json.dumps(anchor)
            self.assertFalse([row["id"] for row in self.rows if row["id"] in text])
            self.assertFalse([row["group"] for row in self.rows if f'"{row["group"]}"' in text])
            self.assertNotIn("global/", text)
            self.assertEqual(anchor["document_sha256"], document["sha256"])
        anchor = gc.build_anchor("campaign", campaign)
        self.assertEqual((anchor["freeze_sha256"], anchor["code_commit"], anchor["label"]),
                         (self.freeze["sha256"], CODE, "exploratory"))


class Q6CodeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.pools, self.freeze = external_freeze()

    def test_policy_rejects_code_changes_and_unbound_evidence(self):
        cases = {
            "src change": identity(changed=["configs/global/x/final/a.json", "src/augment_atlas/train.py"]),
            "docs change": identity(changed=["docs/protocol.md"]),
            "smoke configs": identity(changed=["configs/smoke/none.json"]),
            "traversal": identity(changed=["configs/global/../../src/x.py"]),
            "tree differs": identity(final_code_tree_sha256=H("other tree")),
            "freeze commit": identity(freeze_commit="f" * 40),
            "code commit": identity(code_commit="f" * 40),
            "image": identity(image_id="sha256:" + "0" * 64),
            "policy": identity(policy="same_commit_literal"),
            "status": identity(policy_status="confirmed"),
            "unsorted": dict(identity(), changed_paths=["configs/global/b", "configs/global/a"]),
        }
        for name, value in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
                seal(self.rows, self.pools, self.freeze, code_identity=value)
        literal = identity(freeze_commit=CODE, changed=[])
        gc.audit_code_identity(literal, freeze_runtime={"commit": CODE}, code_commit=CODE, image_id=IMAGE)
        with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
            gc.audit_code_identity(identity(freeze_commit=CODE), freeze_runtime={"commit": CODE},
                                   code_commit=CODE, image_id=IMAGE)

    def test_policy_function_is_isolated_and_swappable(self):
        self.assertIs(gc.Q6_POLICIES["same_code_identity_v1"], gc._q6_same_code_identity_v1)
        self.assertIn("pending", gc.Q6_POLICY_STATUS)

    def git(self, repo, *args):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                               *args], check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, repo, files, message):
        for path, text in files.items():
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode())
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-q", "-m", message)
        return self.git(repo, "rev-parse", "HEAD")

    def test_git_derived_identity_allows_only_global_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.git(repo, "init", "-q")
            freeze_commit = self.commit(repo, {"src/augment_atlas/a.py": "x = 1\n", "scripts/run.sh": "echo\n",
                                               "docs/a.md": "doc\n"}, "code")
            final_commit = self.commit(repo, {"configs/global/synthetic_fixture/final/MANIFEST.json": "{}\n"}, "configs")
            derived = gc.derive_code_identity(repo, freeze_commit=freeze_commit, code_commit=final_commit, image_id=IMAGE)
            self.assertEqual(derived["changed_paths"], ["configs/global/synthetic_fixture/final/MANIFEST.json"])
            self.assertEqual(derived["freeze_code_tree_sha256"], derived["final_code_tree_sha256"])
            gc.audit_code_identity(derived, freeze_runtime={"commit": freeze_commit}, code_commit=final_commit,
                                   image_id=IMAGE)
            for files in ({"src/augment_atlas/a.py": "x = 2\n"}, {"docs/a.md": "changed\n"},
                          {"scripts/new.sh": "echo\n"}):
                with self.subTest(files=files):
                    later = self.commit(repo, files, "drift")
                    drifted = gc.derive_code_identity(repo, freeze_commit=freeze_commit, code_commit=later,
                                                      image_id=IMAGE)
                    with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
                        gc.audit_code_identity(drifted, freeze_runtime={"commit": freeze_commit}, code_commit=later,
                                               image_id=IMAGE)
            with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
                gc.derive_code_identity(repo, freeze_commit=final_commit, code_commit=freeze_commit, image_id=IMAGE)


if __name__ == "__main__":
    unittest.main()
