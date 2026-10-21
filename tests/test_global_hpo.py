"""Slice S2 global HPO producer: search space, plan, hpo-view, D-view, collector.

Synthetic records and files only: no images are decoded, nothing trains, no
Docker or Git. Every adversarial mutation is RE-SIGNED (checksum recomputed) so
that it can only be caught by the semantic rule under test, never by a stale hash.
"""
import copy
import errno
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from augment_atlas import global_development as gd
from augment_atlas import global_evaluation as ge
from augment_atlas import global_hpo as gh
from augment_atlas import global_jobs as gj
from augment_atlas import runtime
from augment_atlas.contracts import canonical_hash as H
from augment_atlas.schema import new_record
from test_global_jobs import DATASET_SOURCE, binary_pools, final_policy
from test_global_jobs import runtime as runtime_decl
from test_recipe_freeze import freeze_fixture

PROFILES = {  # D_select scores -> AUROC: good = 1.0, tie = 0.5, bad = 0.0
    "good": [[0.9, 0.1], [0.2, 0.8]],
    "tie": [[0.5, 0.5], [0.5, 0.5]],
    "bad": [[0.2, 0.8], [0.9, 0.1]],
}
DEFAULT_PROFILE = {"c00": "tie", "c01": "good", "c02": "tie", "c03": "bad"}
T0 = "2026-10-20T10:00:00+00:00"
ATTEMPT = "a" * 32
RETRIED = "b" * 32


def signed(value, key="sha256"):
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = H(result)
    return result


def sha_of(data):
    return hashlib.sha256(data).hexdigest()


class Fixture:
    """Pools, data root, search space, plan, view and the expected HPO jobs."""

    def __init__(self, root, *, seeds=(101,), budget=None, grid=None, default=None):
        self.root = Path(root)
        self.rows, self.pools = binary_pools()
        self.data = self.root / "data"
        self.views = self.root / "views"
        self.runs = self.root / "runs"
        self.image_sha256 = {}
        for row in self.rows:
            data = f"image bytes of {row['id']}".encode()
            path = self.data.joinpath(*row["path"].split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.image_sha256[row["path"]] = sha_of(data)
        self.runtime = runtime_decl()
        base = gj.external_recipe_q2(self.pools, self.runtime, color="grayscale_to_rgb")
        self.grid = grid or [{"path": "optimizer.parameters.lr", "values": [0.00003, 0.0001, 0.0003, 0.001]}]
        self.default = default or {"optimizer.parameters.lr": 0.0001}
        self.budget = budget or {"max_candidates": 6, "max_seeds": 2, "max_epochs": 20, "job_timeout_s": 3600,
                                 "gpu_minutes_cap": 120}
        self.space = gh.build_search_space(self.pools, base_recipe=base, grid=self.grid, default=self.default,
                                           seeds=list(seeds), budget=self.budget)
        self.view = gh.derive_hpo_view(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"],
                                       image_sha256=self.image_sha256)
        self.producer = gh.make_producer(code_sha256=H("fixture code tree"),
                                         environment_sha256=self.runtime["environment_sha256"],
                                         view_files_sha256=self.view["view_files_sha256"])
        self.plan = gh.build_selection_plan(self.space, self.pools, producer=self.producer, runtime=self.runtime,
                                            final_policy=final_policy())
        self.plan_sha256 = H(self.plan)
        self.configs = gj.build_hpo_configs(
            self.rows, self.pools, expected_pools_sha256=self.pools["sha256"], dataset_source=DATASET_SOURCE,
            plan=self.plan, expected_plan_sha256=self.plan_sha256, search_space_sha256=self.space["sha256"],
            hpo_view_sha256=self.view["sha256"])
        self.jobs = {}  # (candidate_id, seed) -> dict(cfg, config_sha, spec)
        for name, cfg in self.configs.items():
            sha = ge.config_sha256(cfg)
            spec = gj.hpo_job_spec(cfg, config_path=f"{gj.config_dir(cfg['dataset']['id'], 'hpo')}/{name}",
                                   config_sha256=sha, code_commit=self.runtime["commit"])
            self.jobs[(cfg["global"]["candidate_id"], cfg["training"]["seed"])] = dict(cfg=cfg, config_sha=sha, spec=spec)
        self.mount = dict(kind="d_view", source=str((self.views / self.pools["sha256"] / "D").resolve()),
                          destination="/data", readonly=True, pools_sha256=self.pools["sha256"],
                          hpo_view_sha256=self.view["sha256"], view_files_sha256=self.view["view_files_sha256"])

    # --- attempts and queue records ----------------------------------------

    def attempt_dir(self, key, attempt=ATTEMPT):
        return self.runs / self.jobs[key]["spec"]["job_id"] / attempt

    def write_attempt(self, key, profile="tie", attempt=ATTEMPT, scores=None, mount=None, tweak=None):
        job = self.jobs[key]
        cfg, out = job["cfg"], self.attempt_dir(key, attempt)
        out.mkdir(parents=True, exist_ok=True)
        scores = copy.deepcopy(PROFILES[profile] if scores is None else scores)
        artifact = ge.build_global_predictions(cfg, self.view, self.rows, scores, pools=self.pools,
                                               epoch=cfg["training"]["epochs"])
        summary = ge.validate_global_predictions(cfg, self.view, self.rows, artifact, pools=self.pools)
        results = {"clinical_evaluation": {"artifact": ge.ARTIFACT_NAME, **{
            k: summary[k] for k in ("partition", "samples", "primary_metric", "primary_value", "primary_reason",
                                    "content_sha256")}}}
        if tweak:
            artifact, results = tweak(artifact, results)
        self.store(out, artifact, results, key, mount)
        (out / "docker.log").write_text(f"docker log of {key}\n", encoding="utf-8")
        return out

    def store(self, out, artifact, results, key, mount=None):
        job = self.jobs[key]
        (out / ge.ARTIFACT_NAME).write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
        (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
        completed = {"status": "completed", "phase": "hpo", "config_sha256": job["config_sha"],
                     "splits_sha256": self.view["sha256"], "trainer_override": None,
                     "artifacts": {ge.ARTIFACT_NAME: sha_of((out / ge.ARTIFACT_NAME).read_bytes())}}
        (out / "completed.json").write_text(json.dumps(completed), encoding="utf-8")
        (out / "runtime.json").write_text(json.dumps({"data_mount": self.mount if mount is None else mount}),
                                          encoding="utf-8")

    def record(self, key, state, attempt=ATTEMPT, minutes=1.0, last=None):
        record = new_record(self.jobs[key]["spec"])
        if state == "pending":
            return record
        events = [{"seq": 1, "kind": "claim", "state": "claimed", "attempt_id": attempt, "at": T0}]
        if state != "claimed":
            events.append({"seq": 2, "kind": "update", "state": "running", "attempt_id": attempt,
                           "at": "2026-10-20T10:01:00+00:00"})
        if state not in ("claimed", "running"):
            end = 60 + int(minutes * 60)
            events.append({"seq": 3, "kind": "update", "state": state, "attempt_id": attempt,
                           "at": f"2026-10-20T{10 + end // 3600:02d}:{end % 3600 // 60:02d}:{end % 60:02d}+00:00",
                           **(last or {})})
        record.update(state=state, attempt_id=attempt, token="token", owner={"gpu": "0"}, version=len(events),
                      seq=len(events), history=events)
        return record

    def retried_record(self, key, minutes=(3, 2)):
        """Attempt RETRIED failed, an operator retried it, attempt ATTEMPT succeeded."""
        record = new_record(self.jobs[key]["spec"])
        stamp = lambda minute: f"2026-10-20T10:{minute:02d}:00+00:00"
        first, second = minutes
        events = [("claim", "claimed", RETRIED, 0), ("update", "running", RETRIED, 1),
                  ("update", "failed", RETRIED, 1 + first), ("manual_retry", "pending", None, 2 + first),
                  ("claim", "claimed", ATTEMPT, 3 + first), ("update", "running", ATTEMPT, 4 + first),
                  ("update", "succeeded", ATTEMPT, 4 + first + second)]
        history = [{"seq": n, "kind": kind, "state": state, "at": stamp(minute),
                    **({"attempt_id": attempt} if attempt else {})}
                   for n, (kind, state, attempt, minute) in enumerate(events, 1)]
        record.update(state="succeeded", attempt_id=ATTEMPT, token="token", owner={"gpu": "0"}, version=len(history),
                      seq=len(history), history=history)
        return record

    def snapshot(self, states=None, profile=None, **kwargs):
        """All jobs succeeded (attempts written) unless ``states`` overrides a key."""
        profile = DEFAULT_PROFILE if profile is None else profile
        states = states or {}
        records = []
        for key in sorted(self.jobs):
            state = states.get(key, "succeeded")
            if state == "absent":
                continue
            if state == "succeeded":
                self.write_attempt(key, profile[key[0]])
            records.append(self.record(key, state, **kwargs))
        return records

    def collect(self, snapshot, **overrides):
        arguments = dict(search_space=self.space, plan=self.plan, plan_sha256=self.plan_sha256, view=self.view,
                         queue_snapshot=snapshot, runs_root=self.runs, records=self.rows, pools=self.pools,
                         dataset_source=DATASET_SOURCE, image_sha256=self.image_sha256)
        arguments.update(overrides)
        return gh.collect_hpo_evidence(arguments.pop("search_space"), arguments.pop("plan"),
                                       arguments.pop("plan_sha256"), arguments.pop("view"),
                                       arguments.pop("queue_snapshot"), arguments.pop("runs_root"), **arguments)

    def freeze_candidate(self, winner, evidence):
        candidate = next(c for c in self.plan["candidates"] if c["id"] == winner)
        return {"schema_version": 1, "protocol": gd.FREEZE_PROTOCOL, "scientific_status": "provisional",
                "dataset": copy.deepcopy(self.pools["dataset"]), "task": copy.deepcopy(self.pools["task"]),
                "pools_sha256": self.pools["sha256"], "selection_plan": copy.deepcopy(self.plan),
                "selection_plan_sha256": self.plan_sha256, "selection_evidence_sha256": evidence["sha256"],
                "selected_candidate_id": winner, "selected_parameters": copy.deepcopy(candidate["parameters"]),
                "frozen_recipe": copy.deepcopy(candidate["recipe"]),
                "final_policy": copy.deepcopy(self.plan["final_policy"]), "runtime": copy.deepcopy(self.plan["runtime"]),
                "selection_scope": {"train_ids": list(self.pools["lists"]["D_fit"]),
                                    "select_ids": list(self.pools["lists"]["D_select"]),
                                    "epoch_ids": list(self.pools["lists"]["D_select"]),
                                    "threshold_ids": list(self.pools["lists"]["D_select"]),
                                    "augmentation": {"arm": "none", "ops": []},
                                    "producer": copy.deepcopy(self.plan["producer"])},
                "frozen_at_utc": "2026-10-21T12:00:00Z"}

    def seal(self, evidence, winner, output):
        return gd.seal_recipe_freeze(self.rows, self.pools, self.freeze_candidate(winner, evidence), evidence,
                                     expected_pools_sha256=self.pools["sha256"], expected_plan_sha256=self.plan_sha256,
                                     output=output)

    def audit(self, evidence, freeze):
        return gd.audit_recipe_freeze(self.rows, self.pools, freeze, evidence,
                                      expected_pools_sha256=self.pools["sha256"], expected_plan_sha256=self.plan_sha256)


class Base(unittest.TestCase):
    def temp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def fixture(self, **kwargs):
        return Fixture(self.temp(), **kwargs)

    def assertCode(self, expected):
        return self.assertRaisesRegex(ValueError, f"^{expected}:")


# ==========================================================================
# Search space and candidate expansion (6.1)
# ==========================================================================

class SearchSpaceTests(Base):
    def setUp(self):
        self.fx = self.fixture()
        self.pools = self.fx.pools

    def build(self, **overrides):
        arguments = dict(base_recipe=copy.deepcopy(self.fx.space["base_recipe"]), grid=copy.deepcopy(self.fx.grid),
                         default=copy.deepcopy(self.fx.default), seeds=[101], budget=copy.deepcopy(self.fx.budget))
        arguments.update(overrides)
        return gh.build_search_space(self.pools, **arguments)

    def resigned(self, mutate):
        document = copy.deepcopy(self.fx.space)
        mutate(document)
        return signed(document)

    def test_expansion_puts_default_first_then_lexicographic_and_is_deterministic(self):
        candidates = gh.expand_candidates(self.fx.space)
        self.assertEqual([c["id"] for c in candidates], ["c00", "c01", "c02", "c03"])
        self.assertEqual([c["parameters"]["lr"] for c in candidates], [0.0001, 0.00003, 0.0003, 0.001])
        for candidate in candidates:
            self.assertEqual(candidate["augmentation"], {"arm": "none", "ops": []})
            self.assertEqual(candidate["recipe"]["optimizer"]["parameters"]["lr"], candidate["parameters"]["lr"])
            self.assertEqual(set(candidate["parameters"]), {"lr"})
        self.assertEqual(H(candidates), H(gh.expand_candidates(copy.deepcopy(self.fx.space))))
        base = self.fx.space["base_recipe"]
        self.assertEqual(candidates[0]["recipe"], base)
        self.assertEqual(base["optimizer"]["parameters"]["lr"], 0.0001)  # the search space is not mutated

    def test_two_axis_grid_is_lexicographic_in_grid_order_and_skips_the_default_once(self):
        grid = [{"path": "optimizer.parameters.lr", "values": [0.0001, 0.0003]},
                {"path": "optimizer.parameters.weight_decay", "values": [0, 0.0001, 0.001]}]
        default = {"optimizer.parameters.lr": 0.0003, "optimizer.parameters.weight_decay": 0.0001}
        base = copy.deepcopy(self.fx.space["base_recipe"])
        base["optimizer"]["parameters"] = {"lr": 0.0003, "weight_decay": 0.0001}
        space = self.build(grid=grid, default=default, base_recipe=base)
        candidates = gh.expand_candidates(space)
        got = [(c["parameters"]["lr"], c["parameters"]["weight_decay"]) for c in candidates]
        self.assertEqual(got, [(0.0003, 0.0001), (0.0001, 0), (0.0001, 0.0001), (0.0001, 0.001),
                               (0.0003, 0), (0.0003, 0.001)])
        self.assertEqual(len(set(got)), 6)

    def test_epochs_axis_and_budget_ceilings(self):
        base = copy.deepcopy(self.fx.space["base_recipe"])
        base["epochs"] = 10
        space = self.build(grid=[{"path": "epochs", "values": [5, 10, 20]}], default={"epochs": 10}, base_recipe=base)
        self.assertEqual([c["parameters"]["epochs"] for c in gh.expand_candidates(space)], [10, 5, 20])
        for budget in ({"max_candidates": 7}, {"max_seeds": 3}, {"max_epochs": 21}):
            with self.subTest(budget=budget), self.assertCode("search_space_invalid"):
                self.build(budget={**self.fx.budget, **budget})

    def test_closed_registry_ordering_budget_and_default_rules(self):
        lr = "optimizer.parameters.lr"
        cases = {
            "unknown path": dict(grid=[{"path": "batch_size", "values": [8, 16]}], default={"batch_size": 16}),
            "repeated path": dict(grid=[{"path": lr, "values": [0.0001]}, {"path": lr, "values": [0.0003]}]),
            "unsorted values": dict(grid=[{"path": lr, "values": [0.0003, 0.0001]}]),
            "duplicate values": dict(grid=[{"path": lr, "values": [0.0001, 0.0001]}]),
            "zero lr": dict(grid=[{"path": lr, "values": [0, 0.0001]}]),
            "bool value": dict(grid=[{"path": "epochs", "values": [True, 5]}], default={"epochs": 5}),
            "epochs over budget": dict(grid=[{"path": "epochs", "values": [5, 25]}], default={"epochs": 5}),
            "empty grid": dict(grid=[]),
            "default outside grid": dict(default={lr: 0.0002}),
            "default misses a path": dict(default={}),
            "product over max_candidates": dict(
                grid=[{"path": lr, "values": [0.0001, 0.0002, 0.0003]},
                      {"path": "optimizer.parameters.weight_decay", "values": [0, 0.0001, 0.001]}],
                default={lr: 0.0001, "optimizer.parameters.weight_decay": 0}),
            "too many seeds": dict(seeds=[1, 2, 3]),
            "duplicate seeds": dict(seeds=[1, 1]),
            "bool seed": dict(seeds=[True]),
            "no seeds": dict(seeds=[]),
        }
        for label, overrides in cases.items():
            with self.subTest(label), self.assertCode("search_space_invalid"):
                self.build(**overrides)

    def test_base_recipe_must_carry_the_default_and_be_in_the_trainer_registry(self):
        base = copy.deepcopy(self.fx.space["base_recipe"])
        base["optimizer"]["parameters"]["lr"] = 0.0003
        with self.assertCode("search_space_invalid"):
            self.build(base_recipe=base)
        for mutate in (lambda r: r["optimizer"].update(name="sgd"),
                       lambda r: r["scheduler"].update(name="cosine"),
                       lambda r: r["loss"].update(parameters={"class_weights": [1.0, 2.0]})):
            base = copy.deepcopy(self.fx.space["base_recipe"])
            mutate(base)
            with self.subTest(mutate=mutate), self.assertCode("recipe_adapter_unsupported"):
                self.build(base_recipe=base)

    def test_seal_binds_pools_policies_and_checksum_after_resign(self):
        validated = gh.validate_search_space(self.fx.space, pools=self.pools)
        self.assertEqual(validated["protocol"], gh.SEARCH_SPACE_PROTOCOL)
        tampered = copy.deepcopy(self.fx.space)
        tampered["seeds"] = [102]
        with self.assertCode("search_space_invalid"):  # stale checksum
            gh.validate_search_space(tampered)
        for label, mutate in {
            "retry policy": lambda d: d.update(retry_policy="retry_after_scores"),
            "failure policy": lambda d: d.update(failure_policy="drop_failed_silently"),
            "scientific status": lambda d: d.update(scientific_status="confirmatory"),
            "extra key": lambda d: d.update(note="x"),
            "protocol": lambda d: d.update(protocol="other"),
            "budget key": lambda d: d["budget"].pop("gpu_minutes_cap"),
        }.items():
            with self.subTest(label), self.assertCode("search_space_invalid"):
                gh.validate_search_space(self.resigned(mutate))
        other = self.resigned(lambda d: d.update(pools_sha256=H("other pools")))
        with self.assertCode("search_space_mismatch"):
            gh.validate_search_space(other, pools=self.pools)
        other = self.resigned(lambda d: d["task"].update(positive_class=0))
        with self.assertCode("search_space_mismatch"):
            gh.validate_search_space(other, pools=self.pools)

    def test_every_expanded_candidate_must_fit_the_trainer_registry_before_anchoring(self):
        base = copy.deepcopy(self.fx.space["base_recipe"])
        base["scheduler"]["parameters"]["eta_min"] = 0.0005  # a candidate with lr 3e-5 would sit below eta_min
        base["optimizer"]["parameters"]["lr"] = 0.001
        with self.assertCode("recipe_adapter_unsupported"):
            self.build(base_recipe=base, default={"optimizer.parameters.lr": 0.001})

    def test_numbers_beyond_float_range_are_a_coded_error(self):
        with self.assertCode("search_space_invalid"):
            self.build(grid=[{"path": "optimizer.parameters.lr", "values": [0.0001, 10 ** 400]}])

    def test_a_fixed_threshold_other_than_the_global_constant_is_refused(self):
        base = copy.deepcopy(self.fx.space["base_recipe"])
        base["prediction"]["thresholds"] = [0.3]
        with self.assertCode("recipe_adapter_unsupported"):
            self.build(base_recipe=base)


# ==========================================================================
# Plan v1 from the search space
# ==========================================================================

class PlanTests(Base):
    def setUp(self):
        self.fx = self.fixture()

    def test_plan_is_the_existing_v1_schema_and_audits_against_its_search_space(self):
        plan = self.fx.plan
        candidates = gd._plan(plan, self.fx.pools, self.fx.plan_sha256)
        self.assertEqual(sorted(candidates), ["c00", "c01", "c02", "c03"])
        self.assertEqual(plan["seeds"], [101])
        self.assertEqual(plan["failure_policy"], "require_all_defined")
        audit = gh.audit_plan_from_search_space(plan, self.fx.space)
        self.assertEqual(audit["candidate_ids"], ["c00", "c01", "c02", "c03"])
        self.assertEqual(audit["search_space_sha256"], self.fx.space["sha256"])
        self.assertEqual(plan["producer"]["id"], "augment-atlas-global-hpo-v1")
        self.assertEqual(plan["producer"]["logs_sha256"], H({"docker_log": "docker.log", "per_row": True}))
        self.assertEqual(plan["producer"]["manifest_sha256"], self.fx.view["view_files_sha256"])

    def test_plan_that_is_not_the_expansion_of_the_search_space_is_refused(self):
        def edit(mutate):
            plan = copy.deepcopy(self.fx.plan)
            mutate(plan)
            return plan
        cases = {
            "candidate lr edited": lambda p: p["candidates"][2]["recipe"]["optimizer"]["parameters"].update(lr=0.0002),
            "candidate dropped": lambda p: p["candidates"].pop(),
            "candidates reordered": lambda p: p["candidates"].reverse(),
            "candidate id renamed": lambda p: p["candidates"][1].update(id="best"),
            "augmentation added": lambda p: p["candidates"][1].update(augmentation={"arm": "basic", "ops": []}),
            "seed added": lambda p: p["seeds"].append(102),
            "failure policy": lambda p: p.update(failure_policy="exclude_candidate_on_any_nondefined"),
            "other pools": lambda p: p.update(pools_sha256=H("x")),
            "ranking": lambda p: p.update(ranking="best_single_seed"),
        }
        for label, mutate in cases.items():
            with self.subTest(label), self.assertCode("plan_not_from_search_space"):
                gh.audit_plan_from_search_space(edit(mutate), self.fx.space)

    def test_search_space_changed_after_the_plan_is_not_the_plan_origin(self):
        changed = copy.deepcopy(self.fx.space)
        changed["grid"][0]["values"] = [0.00003, 0.0001, 0.0003, 0.002]
        with self.assertCode("plan_not_from_search_space"):
            gh.audit_plan_from_search_space(self.fx.plan, signed(changed))

    def test_producer_runtime_and_seed_rules(self):
        build = lambda **kw: gh.build_selection_plan(
            self.fx.space, self.fx.pools, **{"producer": self.fx.producer, "runtime": self.fx.runtime,
                                              "final_policy": final_policy(), **kw})
        for label, producer in {
            "synthetic producer id": {**self.fx.producer, "id": "SYNTHETIC-NOT-A-REAL-PRODUCER"},
            "logs hash of real logs": {**self.fx.producer, "logs_sha256": H("real docker logs")},
            "other environment": {**self.fx.producer, "environment_sha256": H("another environment")},
        }.items():
            with self.subTest(label), self.assertCode("plan_invalid"):
                build(producer=producer)
        policy = final_policy()
        policy["seeds"] = [101, 13, 19]
        with self.assertCode("seed_reuse_forbidden"):
            build(final_policy=policy)
        with self.assertCode("plan_invalid"):
            build(runtime={**self.fx.runtime, "commit": "short"})

    def test_junk_runtime_is_a_coded_plan_error(self):
        with self.assertCode("plan_invalid"):
            gh.build_selection_plan(self.fx.space, self.fx.pools, producer=self.fx.producer, runtime={"commit": "x"},
                                    final_policy=final_policy())

    def test_evaluation_seed_equal_to_a_development_seed_is_refused(self):
        rows, pools = binary_pools(evaluation=29)  # selection seed is 29
        base = gj.external_recipe_q2(pools, self.fx.runtime, color="grayscale_to_rgb")
        space = gh.build_search_space(pools, base_recipe=base, grid=self.fx.grid, default=self.fx.default,
                                      seeds=[101], budget=self.fx.budget)
        with self.assertCode("seed_reuse_forbidden"):
            gh.build_selection_plan(space, pools, producer=self.fx.producer, runtime=self.fx.runtime,
                                    final_policy=final_policy())

    def test_helpers_hash_code_and_environment_deterministically(self):
        directory = self.temp()
        (directory / "a.py").write_text("x = 1\n")
        (directory / "data.json").write_text("{}\n")  # non-Python files are part of the code identity
        (directory / "pkg").mkdir()
        (directory / "pkg" / "b.py").write_text("y = 2\n")
        (directory / "__pycache__").mkdir()
        (directory / "__pycache__" / "c.py").write_text("ignored\n")
        first = gh.code_tree_sha256(directory)
        self.assertEqual(first, gh.code_tree_sha256(directory))
        (directory / "pkg" / "b.py").write_text("y = 3\n")
        self.assertNotEqual(first, gh.code_tree_sha256(directory))
        second = gh.code_tree_sha256(directory)
        (directory / "data.json").write_text('{"changed": true}\n')
        self.assertNotEqual(second, gh.code_tree_sha256(directory))  # every regular file counts, not only *.py
        other = self.temp()
        (directory / "dirlink").symlink_to(other, target_is_directory=True)
        with self.assertCode("plan_invalid"):
            gh.code_tree_sha256(directory)
        (directory / "dirlink").unlink()
        with self.assertCode("plan_invalid"):
            gh.code_tree_sha256(self.temp())
        (directory / "link.py").symlink_to(directory / "a.py")
        with self.assertCode("plan_invalid"):
            gh.code_tree_sha256(directory)
        self.assertEqual(gh.environment_sha256("sha256:abc", {"python": "3.11"}),
                         H({"image_id": "sha256:abc", "probe": {"python": "3.11"}}))


# ==========================================================================
# hpo-view and D-view (6.3)
# ==========================================================================

class ViewTests(Base):
    def setUp(self):
        self.fx = self.fixture()
        self.rows, self.pools, self.view = self.fx.rows, self.fx.pools, self.fx.view

    def audit(self, view, **kwargs):
        return gh.audit_hpo_view(view, self.rows, self.pools, expected_pools_sha256=self.pools["sha256"], **kwargs)

    def test_view_is_exactly_d_with_no_e_identity_and_is_deterministic(self):
        lists = self.pools["lists"]
        d_ids = set(lists["D"])
        self.assertEqual([r["id"] for r in self.view["records"]], sorted(d_ids))
        self.assertEqual(self.view["lists"], {"train": lists["D_fit"], "selection": lists["D_select"]})
        e_groups = {r["group"] for r in self.rows if r["id"] in set(lists["E"])}
        text = json.dumps(self.view)
        self.assertFalse(any(i in text for i in lists["E"]))
        self.assertFalse(any(json.dumps(g) in text for g in e_groups))
        self.assertEqual(self.view["view_files_sha256"], H(sorted([r["path"], self.fx.image_sha256[r["path"]]]
                                                                 for r in self.view["records"])))
        again = gh.derive_hpo_view(list(reversed(self.rows)), self.pools, expected_pools_sha256=self.pools["sha256"],
                                   image_sha256=self.fx.image_sha256)
        self.assertEqual(again, self.view)
        self.assertEqual(self.audit(self.view, image_sha256=self.fx.image_sha256), self.view)
        self.assertEqual(gh.hpo_view_file(self.view), f"global/synthetic_fixture/hpo-view-{self.view['sha256']}.json")

    def test_view_depends_on_the_d_image_bytes_but_not_on_e_image_bytes(self):
        other = dict(self.fx.image_sha256)
        e_path = next(r["path"] for r in self.rows if r["id"] in set(self.pools["lists"]["E"]))
        other[e_path] = H("different E bytes")
        self.assertEqual(gh.derive_hpo_view(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"],
                                            image_sha256=other), self.view)
        d_path = self.view["records"][0]["path"]
        other[d_path] = H("different D bytes")
        changed = gh.derive_hpo_view(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"],
                                     image_sha256=other)
        self.assertNotEqual(changed["view_files_sha256"], self.view["view_files_sha256"])
        with self.assertCode("view_hash_mismatch"):
            self.audit(changed, image_sha256=self.fx.image_sha256)

    def test_derivation_needs_anchored_pools_and_every_d_image_hash(self):
        with self.assertCode("pools_anchor_mismatch"):
            gh.derive_hpo_view(self.rows, self.pools, expected_pools_sha256=H("not the pools"),
                               image_sha256=self.fx.image_sha256)
        missing = dict(self.fx.image_sha256)
        del missing[self.view["records"][0]["path"]]
        with self.assertCode("records_identity_mismatch"):
            gh.derive_hpo_view(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"], image_sha256=missing)
        bad = dict(self.fx.image_sha256)
        bad[self.view["records"][0]["path"]] = "ABC"
        with self.assertCode("records_identity_mismatch"):
            gh.derive_hpo_view(self.rows, self.pools, expected_pools_sha256=self.pools["sha256"], image_sha256=bad)
        rows = copy.deepcopy(self.rows)
        rows[0]["label"] = 1 - rows[0]["label"]
        with self.assertCode("records_identity_mismatch"):
            gh.derive_hpo_view(rows, self.pools, expected_pools_sha256=self.pools["sha256"],
                               image_sha256=self.fx.image_sha256)

    def test_resigned_view_with_an_e_identity_is_hpo_view_contains_E(self):
        e_row = next(r for r in self.rows if r["id"] in set(self.pools["lists"]["E"]))
        e_other_group = next(r for r in self.rows if r["group"] == e_row["group"] and r["id"] != e_row["id"])

        def add_record(view):
            view["records"] = sorted(view["records"] + [copy.deepcopy(e_row)], key=lambda r: r["id"])

        def add_to_train(view):
            view["lists"]["train"] = sorted(view["lists"]["train"] + [e_row["id"]])

        def add_to_selection(view):
            view["lists"]["selection"] = sorted(view["lists"]["selection"] + [e_row["id"]])

        def relabel_group(view):  # a D record carrying an E group (no E ID at all)
            view["records"][0]["group"] = e_row["group"]

        def only_other_e_sibling(view):
            view["records"].append(copy.deepcopy(e_other_group))

        for mutate in (add_record, add_to_train, add_to_selection, relabel_group, only_other_e_sibling):
            view = copy.deepcopy(self.view)
            mutate(view)
            view = signed(view)  # re-signed: the checksum is valid
            with self.subTest(mutate=mutate.__name__), self.assertCode("hpo_view_contains_E"):
                self.audit(view)

    def test_other_tampered_views_are_refused_even_when_resigned(self):
        cases = {
            "stale checksum": (lambda v: v["records"].pop(), False, "view_hash_mismatch"),
            "extra key": (lambda v: v.update(note="x"), True, "view_hash_mismatch"),
            "other pools": (lambda v: v.update(pools_sha256=H("other")), True, "view_hash_mismatch"),
            "protocol": (lambda v: v.update(protocol="global-hpo-view-v2"), True, "view_hash_mismatch"),
            "D record dropped": (lambda v: v["records"].pop(), True, "records_identity_mismatch"),
            "D label flipped": (lambda v: v["records"][0].update(label=1 - v["records"][0]["label"]), True,
                                "records_identity_mismatch"),
            "swapped lists": (lambda v: v["lists"].update(train=v["lists"]["selection"],
                                                          selection=v["lists"]["train"]), True, "view_hash_mismatch"),
            "task edited": (lambda v: v["task"].update(positive_class=0), True, "view_hash_mismatch"),
            "bad listing hash": (lambda v: v.update(view_files_sha256="xyz"), True, "view_hash_mismatch"),
        }
        for label, (mutate, resign, expected) in cases.items():
            view = copy.deepcopy(self.view)
            mutate(view)
            view = signed(view) if resign else view
            with self.subTest(label), self.assertCode(expected):
                self.audit(view)
        with self.assertCode("view_hash_mismatch"):
            self.audit("not a view")

    def test_deep_symlinked_subdirectory_creates_nothing_outside_the_view(self):
        outside = self.temp()
        dest = self.fx.views / self.pools["sha256"] / "D"
        dest.mkdir(parents=True)
        first = Path(self.view["records"][0]["path"]).parts[0]
        (dest / first).symlink_to(outside, target_is_directory=True)
        deep = {**self.view["records"][0], "path": f"{first}/sub/deep/img.png"}
        rows = [deep if r["id"] == deep["id"] else r for r in self.rows]
        with self.assertRaises(ValueError):  # the path is not in the trusted records either way
            gh.materialize_d_view(self.fx.data, self.fx.views, self.view, records=rows, pools=self.pools,
                                  image_sha256=self.fx.image_sha256)
        self.assertEqual(list(outside.iterdir()), [])
        with self.assertCode("hpo_isolation_unproven"):
            gh._confined_parent(dest, dest / first / "sub" / "deep" / "img.png")
        self.assertEqual(list(outside.iterdir()), [])

    def test_view_records_with_unhashable_identities_are_a_coded_error(self):
        for field in ("id", "group"):
            view = copy.deepcopy(self.view)
            view["records"][0][field] = ["x"]
            with self.subTest(field), self.assertCode("view_hash_mismatch"):
                self.audit(signed(view))

    def test_ge_check_hpo_view_agrees_on_the_e_leak(self):
        e_row = next(r for r in self.rows if r["id"] in set(self.pools["lists"]["E"]))
        view = copy.deepcopy(self.view)
        view["records"].append(e_row)
        view = signed(view)
        cfg = self.fx.jobs[("c00", 101)]["cfg"]
        with self.assertCode("hpo_view_contains_E"):
            ge.check_hpo_view(view, cfg, self.rows, self.pools)

    # --- physical D-view ------------------------------------------------------

    def materialize(self, fx=None, **kwargs):
        fx = fx or self.fx
        return gh.materialize_d_view(fx.data, fx.views, fx.view, records=fx.rows, pools=fx.pools,
                                     image_sha256=fx.image_sha256, **kwargs)

    def test_d_view_holds_only_hardlinked_d_images_and_reproduces_the_listing(self):
        result = self.materialize()
        dest = Path(result["dest"])
        self.assertEqual(dest, (self.fx.views / self.pools["sha256"] / "D").resolve())
        self.assertEqual(result["methods"], {"hardlink": len(self.view["records"]), "copy": 0, "existing": 0})
        files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
        expected = sorted([r["path"] for r in self.view["records"]] + [result["view_file"], gh.LISTING_NAME])
        self.assertEqual(files, expected)
        for record in self.view["records"]:
            source, target = self.fx.data / record["path"], dest / record["path"]
            self.assertTrue(os.path.samefile(source, target))
            self.assertGreaterEqual(target.stat().st_nlink, 2)
        for record in self.rows:
            if record["id"] in set(self.pools["lists"]["E"]):
                self.assertFalse((dest / record["path"]).exists())
        listing = runtime.d_view_listing(dest, result["view_file"])
        self.assertEqual(H(listing), self.view["view_files_sha256"])
        self.assertEqual(json.loads((dest / result["view_file"]).read_text()), self.view)
        self.assertEqual(json.loads((dest / gh.LISTING_NAME).read_text()), listing)

    def test_worker_runtime_accepts_exactly_this_view_and_refuses_a_stray_file(self):
        result = self.materialize()
        cfg = self.fx.jobs[("c00", 101)]["cfg"]
        trusted = mock.Mock(cfg=cfg, view=self.view, image_sha256=self.fx.image_sha256)
        mount = runtime.verify_d_view({"views": str(self.fx.views)}, trusted)
        self.assertEqual((mount["kind"], mount["destination"], mount["readonly"]), ("d_view", "/data", True))
        self.assertEqual(mount["view_files_sha256"], self.view["view_files_sha256"])
        stray = Path(result["dest"]) / "leak.png"
        stray.write_bytes(b"E bytes")
        with self.assertCode("hpo_isolation_unproven"):
            runtime.verify_d_view({"views": str(self.fx.views)}, trusted)
        with self.assertCode("hpo_isolation_unproven"):
            self.materialize()  # the producer refuses to bless a dirty view too
        stray.unlink()
        self.assertEqual(self.materialize()["methods"]["existing"], len(self.view["records"]))

    def test_materialization_is_idempotent_and_never_overwrites(self):
        self.materialize()
        again = self.materialize()
        self.assertEqual(again["methods"], {"hardlink": 0, "copy": 0, "existing": len(self.view["records"])})
        target = Path(again["dest"]) / self.view["records"][0]["path"]
        target.unlink()
        target.write_bytes(b"different bytes")
        with self.assertCode("materialization_conflict"):
            self.materialize()
        self.assertEqual(target.read_bytes(), b"different bytes")

    def test_source_bytes_must_match_the_manifest_and_symlinks_are_refused(self):
        path = self.fx.data / self.view["records"][0]["path"]
        original = path.read_bytes()
        path.write_bytes(b"tampered after hashing")
        with self.assertCode("records_identity_mismatch"):
            self.materialize()
        self.assertFalse((self.fx.views / self.pools["sha256"] / "D" / self.view["records"][0]["path"]).exists())
        path.write_bytes(original)
        e_row = next(r for r in self.rows if r["id"] in set(self.pools["lists"]["E"]))
        e_path = self.fx.data / e_row["path"]
        d_path = self.fx.data / self.view["records"][1]["path"]
        d_path.unlink()
        d_path.symlink_to(e_path)  # a D name pointing at an E file
        with self.assertCode("hpo_isolation_unproven"):
            self.materialize()

    def test_a_symlinked_view_directory_never_receives_writes(self):
        outside = self.temp()
        base = self.fx.views / self.pools["sha256"]
        base.mkdir(parents=True)
        (base / "D").symlink_to(outside, target_is_directory=True)
        with self.assertCode("hpo_isolation_unproven"):
            self.materialize()
        self.assertEqual(list(outside.iterdir()), [])
        base.joinpath("D").unlink()
        base.rmdir()
        shutil_link = self.fx.views / self.pools["sha256"]
        shutil_link.symlink_to(outside, target_is_directory=True)
        with self.assertCode("hpo_isolation_unproven"):
            self.materialize()
        self.assertEqual(list(outside.iterdir()), [])

    def test_a_symlinked_subdirectory_inside_the_view_cannot_redirect_a_write(self):
        outside = self.temp()
        dest = self.fx.views / self.pools["sha256"] / "D"
        dest.mkdir(parents=True)
        first = Path(self.view["records"][0]["path"]).parts[0]
        (dest / first).symlink_to(outside, target_is_directory=True)
        with self.assertCode("hpo_isolation_unproven"):
            self.materialize()
        self.assertEqual(list(outside.iterdir()), [])

    def test_copy_fallback_keeps_the_source_permissions(self):
        for record in self.view["records"]:
            os.chmod(self.fx.data / record["path"], 0o644)
        real_link = os.link
        data_root = str(self.fx.data.resolve())

        def link(source, target, *args, **kwargs):
            if str(source).startswith(data_root):
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_link(source, target, *args, **kwargs)

        with mock.patch.object(gh.os, "link", side_effect=link):
            result = self.materialize()
        for record in self.view["records"]:
            self.assertEqual((Path(result["dest"]) / record["path"]).stat().st_mode & 0o777, 0o644)

    def test_a_copy_that_differs_from_its_source_is_refused(self):
        real_link = os.link
        data_root = str(self.fx.data.resolve())

        def link(source, target, *args, **kwargs):
            if str(source).startswith(data_root):
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_link(source, target, *args, **kwargs)

        def bad_copy(source, destination):
            Path(destination).write_bytes(b"corrupted in transit")

        with mock.patch.object(gh.os, "link", side_effect=link), mock.patch.object(gh.shutil, "copyfile", side_effect=bad_copy):
            with self.assertCode("d_view_materialization_failed"):
                self.materialize()

    def test_view_with_an_e_identity_is_never_materialized(self):
        e_row = next(r for r in self.rows if r["id"] in set(self.pools["lists"]["E"]))
        view = copy.deepcopy(self.view)
        view["records"].append(e_row)
        view = signed(view)
        with self.assertCode("hpo_view_contains_E"):
            gh.materialize_d_view(self.fx.data, self.fx.views, view, records=self.rows, pools=self.pools,
                                  image_sha256=self.fx.image_sha256)
        self.assertFalse((self.fx.views / self.pools["sha256"]).exists())

    def test_filesystem_without_hardlinks_gets_a_hash_verified_copy_reported_as_such(self):
        real_link = os.link
        data_root = str(self.fx.data.resolve())

        def link(source, target, *args, **kwargs):
            if str(source).startswith(data_root):
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_link(source, target, *args, **kwargs)

        with mock.patch.object(gh.os, "link", side_effect=link):
            result = self.materialize()
        self.assertEqual(result["methods"], {"hardlink": 0, "copy": len(self.view["records"]), "existing": 0})
        dest = Path(result["dest"])
        for record in self.view["records"]:
            self.assertFalse(os.path.samefile(self.fx.data / record["path"], dest / record["path"]))
            self.assertEqual((dest / record["path"]).read_bytes(), (self.fx.data / record["path"]).read_bytes())
        self.assertEqual(H(runtime.d_view_listing(dest, result["view_file"])), self.view["view_files_sha256"])
        self.assertEqual([p.name for p in dest.rglob(".*")], [])  # no temporary files left behind

    def test_no_space_or_other_link_failure_is_an_error_not_a_silent_fallback(self):
        real_link = os.link
        data_root = str(self.fx.data.resolve())

        def no_hardlink(source, target, *args, **kwargs):
            if str(source).startswith(data_root):
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_link(source, target, *args, **kwargs)

        with mock.patch.object(gh.os, "link", side_effect=no_hardlink), \
                mock.patch.object(gh.shutil, "copyfile", side_effect=OSError(errno.ENOSPC, "No space left")):
            with self.assertCode("d_view_materialization_failed"):
                self.materialize()
        self.assertEqual([p for p in (self.fx.views / self.pools["sha256"] / "D").rglob("*") if p.is_file()], [])

        def denied(source, target, *args, **kwargs):
            if str(source).startswith(data_root):
                raise OSError(errno.EACCES, "Permission denied")
            return real_link(source, target, *args, **kwargs)

        with mock.patch.object(gh.os, "link", side_effect=denied), \
                mock.patch.object(gh.shutil, "copyfile") as copy_file:
            with self.assertCode("d_view_materialization_failed"):
                self.materialize()
        copy_file.assert_not_called()


# ==========================================================================
# Evidence collector (6.7)
# ==========================================================================

class CollectorTests(Base):
    def setUp(self):
        self.fx = self.fixture()

    def test_happy_path_recomputes_every_metric_and_feeds_the_v1_freeze(self):
        fx = self.fx
        evidence, manifest = fx.collect(fx.snapshot())
        self.assertEqual(evidence["kind"], "queue_collected_v1")
        self.assertEqual(evidence["plan_sha256"], fx.plan_sha256)
        self.assertEqual(evidence["producer"], fx.plan["producer"])
        self.assertEqual([(r["candidate_id"], r["seed"]) for r in evidence["rows"]],
                         [("c00", 101), ("c01", 101), ("c02", 101), ("c03", 101)])
        expected = {"c00": 0.5, "c01": 1.0, "c02": 0.5, "c03": 0.0}
        for row in evidence["rows"]:
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["metric"], {"endpoint": "binary_auroc", "state": "defined",
                                             "value": expected[row["candidate_id"]], "reason": None})
            self.assertEqual(row["train_ids"], fx.pools["lists"]["D_fit"])
            self.assertEqual(row["select_ids"], fx.pools["lists"]["D_select"])
            self.assertEqual(row["epoch_ids"], row["threshold_ids"])
            key = (row["candidate_id"], row["seed"])
            out = fx.attempt_dir(key)
            artifact = json.loads((out / "predictions.json").read_text())
            self.assertEqual(row["artifact_sha256"], artifact["content_sha256"])
            self.assertEqual(row["log_sha256"], sha_of((out / "docker.log").read_bytes()))
            self.assertEqual(row["config_sha256"], H(next(c for c in fx.plan["candidates"] if c["id"] == row["candidate_id"])))
        for item in manifest["rows"]:
            self.assertEqual((item["status"], item["queue_state"], item["attempts"], item["data_mount_source_kind"]),
                             ("completed", "succeeded", 1, "d_view"))
            self.assertEqual(item["view_files_sha256"], fx.view["view_files_sha256"])
            self.assertEqual(item["recomputed_metric"], expected[item["candidate_id"]])
        self.assertEqual(manifest["evidence_sha256"], evidence["sha256"])
        directory = self.temp()
        freeze = fx.seal(evidence, "c01", directory / "freeze.json")
        self.assertEqual(freeze["selected_candidate_id"], "c01")
        audit = fx.audit(evidence, freeze)
        self.assertEqual(audit["candidate_means"], {"c00": 0.5, "c01": 1.0, "c02": 0.5, "c03": 0.0})
        self.assertEqual(freeze["eligibility"]["scientific_reasons"], [
            "execution_unverified", "runtime_isolation_unverified", "chronology_unverified", "queue_collected_v1"])
        self.assertIs(freeze["eligibility"]["scientific_eligible"], False)
        self.assertIs(audit["execution_verified"], False)
        recipe, policy, frozen_runtime = gj._freeze_parts(  # the S3/S1 consumers accept this freeze too
            fx.rows, fx.pools, freeze=freeze, freeze_kind="hpo_d", expected_pools_sha256=fx.pools["sha256"],
            expected_freeze_sha256=freeze["sha256"], evidence=evidence, expected_plan_sha256=fx.plan_sha256)
        self.assertEqual(recipe["optimizer"]["parameters"]["lr"], 0.00003)
        self.assertEqual(frozen_runtime, fx.runtime)

    def test_collection_is_deterministic_and_independent_of_snapshot_order(self):
        snapshot = self.fx.snapshot()
        first = self.fx.collect(snapshot)
        self.assertEqual(first, self.fx.collect(snapshot))
        self.assertEqual(first, self.fx.collect(list(reversed(snapshot))))

    def test_exact_tie_goes_to_c00_and_row_permutations_never_change_the_winner(self):
        fx = self.fx
        evidence, _ = fx.collect(fx.snapshot(profile={c: "tie" for c in ("c00", "c01", "c02", "c03")}))
        self.assertEqual({r["metric"]["value"] for r in evidence["rows"]}, {0.5})
        directory = self.temp()
        freeze = fx.seal(evidence, "c00", directory / "c00.json")
        self.assertEqual(fx.audit(evidence, freeze)["selected_candidate_id"], "c00")
        with self.assertRaisesRegex(ValueError, "violates predeclared deterministic ranking"):
            fx.seal(evidence, "c02", directory / "c02.json")
        for order in permutations(range(4)):
            shuffled = signed({**evidence, "rows": [copy.deepcopy(evidence["rows"][i]) for i in order]})
            shuffled_freeze = signed({**freeze, "selection_evidence_sha256": shuffled["sha256"]})
            self.assertEqual(fx.audit(shuffled, shuffled_freeze)["selected_candidate_id"], "c00", order)

    def test_two_seed_ledger_ranking_does_not_depend_on_row_order(self):
        fx = self.fixture(seeds=(101, 102))
        profile = {"c00": "tie", "c01": "tie", "c02": "good", "c03": "tie"}
        evidence, _ = fx.collect(fx.snapshot(profile=profile))
        self.assertEqual(len(evidence["rows"]), 8)
        winners = set()
        for order in ((0, 1, 2, 3, 4, 5, 6, 7), (7, 6, 5, 4, 3, 2, 1, 0), (3, 5, 1, 7, 0, 6, 2, 4)):
            shuffled = signed({**evidence, "rows": [copy.deepcopy(evidence["rows"][i]) for i in order]})
            freeze = fx.freeze_candidate("c02", shuffled)
            sealed = signed(freeze_with_eligibility(fx, freeze, shuffled))
            winners.add(fx.audit(shuffled, sealed)["selected_candidate_id"])
        self.assertEqual(winners, {"c02"})

    # --- missing / failed / non-terminal ----------------------------------

    def test_absent_job_is_an_explicit_missing_row_never_zero(self):
        fx = self.fx
        evidence, manifest = fx.collect(fx.snapshot({("c03", 101): "absent"}))
        row = next(r for r in evidence["rows"] if r["candidate_id"] == "c03")
        self.assertEqual((row["status"], row["artifact_sha256"]), ("missing", None))
        self.assertEqual(row["metric"], {"endpoint": "binary_auroc", "state": "unavailable", "value": None,
                                         "reason": "job_missing"})
        self.assertEqual(row["reason"], "job_missing")
        item = next(r for r in manifest["rows"] if r["candidate_id"] == "c03")
        self.assertEqual((item["status"], item["queue_state"], item["attempts"], item["predictions_sha256"]),
                         ("missing", None, 0, None))
        self.assertEqual(len(evidence["rows"]), 4)
        directory = self.temp()
        freeze = fx.seal(evidence, "c01", directory / "freeze.json")
        self.assertIn("ledger_contains_nondefined_trials", freeze["eligibility"]["reasons"])
        self.assertEqual(fx.audit(evidence, freeze)["nondefined_trials"],
                         [{"candidate_id": "c03", "seed": 101, "status": "missing", "reason": "job_missing"}])

    def test_candidate_with_a_missing_seed_is_excluded_not_averaged_over_what_remains(self):
        fx = self.fixture(seeds=(101, 102))
        profile = {"c00": "tie", "c01": "good", "c02": "tie", "c03": "bad"}
        evidence, _ = fx.collect(fx.snapshot({("c01", 102): "absent"}, profile=profile))
        directory = self.temp()
        freeze = fx.seal(evidence, "c00", directory / "freeze.json")
        audit = fx.audit(evidence, freeze)
        self.assertNotIn("c01", audit["candidate_means"])
        self.assertEqual(audit["selected_candidate_id"], "c00")
        with self.assertRaisesRegex(ValueError, "violates predeclared deterministic ranking"):
            fx.seal(evidence, "c01", directory / "partial.json")

    def test_nothing_defined_still_returns_the_ledger_but_cannot_be_sealed(self):
        fx = self.fx
        evidence, manifest = fx.collect([])
        self.assertEqual({r["status"] for r in evidence["rows"]}, {"missing"})
        self.assertEqual({r["metric"]["value"] for r in evidence["rows"]}, {None})
        self.assertEqual(len(manifest["rows"]), 4)
        with self.assertRaisesRegex(ValueError, "no fully defined candidate"):
            fx.seal(evidence, "c00", self.temp() / "freeze.json")

    def test_non_terminal_and_terminal_failure_states_are_explicit_rows(self):
        fx = self.fx
        states = {("c00", 101): "pending", ("c01", 101): "running", ("c02", 101): "failed", ("c03", 101): "interrupted"}
        fx.runs.mkdir(parents=True, exist_ok=True)
        (fx.attempt_dir(("c02", 101))).mkdir(parents=True)
        (fx.attempt_dir(("c02", 101)) / "docker.log").write_text("OOM killed\n")
        evidence, manifest = fx.collect(fx.snapshot(states))
        rows = {r["candidate_id"]: r for r in evidence["rows"]}
        self.assertEqual((rows["c00"]["status"], rows["c00"]["reason"]), ("missing", "job_not_terminal:pending"))
        self.assertEqual((rows["c01"]["status"], rows["c01"]["reason"]), ("missing", "job_not_terminal:running"))
        self.assertEqual((rows["c02"]["status"], rows["c02"]["reason"]), ("failed", "job_failed"))
        self.assertEqual(rows["c02"]["log_sha256"], sha_of(b"OOM killed\n"))
        self.assertEqual((rows["c03"]["status"], rows["c03"]["reason"]), ("failed", "job_interrupted"))
        self.assertRegex(rows["c03"]["log_sha256"], r"^[0-9a-f]{64}$")
        for row in rows.values():
            self.assertEqual((row["metric"]["state"], row["metric"]["value"], row["artifact_sha256"]),
                             ("unavailable", None, None))
        items = {r["candidate_id"]: r for r in manifest["rows"]}
        self.assertEqual(items["c02"]["docker_log_sha256"], sha_of(b"OOM killed\n"))
        self.assertIsNone(items["c03"]["docker_log_sha256"])

    def test_failed_job_whose_last_event_says_timeout_is_a_timeout_row(self):
        fx = self.fx
        records = [fx.record(key, "failed", last={"reason": "job_timeout_s exceeded"}) if key == ("c01", 101)
                   else fx.record(key, "pending") for key in sorted(fx.jobs)]
        evidence, _ = fx.collect(records)
        row = next(r for r in evidence["rows"] if r["candidate_id"] == "c01")
        self.assertEqual((row["status"], row["reason"]), ("timeout", "job_timeout"))

    def test_gpu_minutes_cap_turns_overrunning_rows_into_timeouts_in_canonical_order(self):
        fx = self.fixture(budget={"max_candidates": 6, "max_seeds": 2, "max_epochs": 20, "job_timeout_s": 3600,
                                  "gpu_minutes_cap": 5})
        evidence, manifest = fx.collect(fx.snapshot(minutes=2.0))
        statuses = [(r["candidate_id"], r["status"]) for r in evidence["rows"]]
        self.assertEqual(statuses, [("c00", "completed"), ("c01", "completed"), ("c02", "timeout"), ("c03", "timeout")])
        capped = next(r for r in evidence["rows"] if r["candidate_id"] == "c02")
        self.assertEqual((capped["metric"]["value"], capped["reason"], capped["artifact_sha256"]),
                         (None, "gpu_minutes_cap_exceeded", None))
        self.assertAlmostEqual(manifest["gpu_minutes_counted"], 4.0)
        self.assertEqual(manifest["gpu_minutes_cap"], 5)
        item = next(r for r in manifest["rows"] if r["candidate_id"] == "c02")
        self.assertEqual((item["status"], item["queue_state"], item["predictions_sha256"]), ("timeout", "succeeded", None))

    # --- isolation ---------------------------------------------------------

    def test_runtime_without_a_d_view_mount_is_hpo_isolation_unproven(self):
        fx = self.fx
        key = ("c01", 101)
        for label, mount in {
            "no mount recorded": {"missing": True},
            "full data root": dict(kind="data_root", source=str(fx.data), destination="/data", readonly=True),
            "writable": {**fx.mount, "readonly": False},
            "readonly as integer": {**fx.mount, "readonly": 1},
            "other view hash": {**fx.mount, "view_files_sha256": H("another listing")},
            "other hpo-view": {**fx.mount, "hpo_view_sha256": H("another view")},
            "other pools": {**fx.mount, "pools_sha256": H("another pools")},
            "extra key": {**fx.mount, "also": "/e"},
            "wrong source dir": {**fx.mount, "source": str(fx.data)},
        }.items():
            records = fx.snapshot()
            out = fx.attempt_dir(key)
            (out / "runtime.json").write_text(json.dumps({} if "missing" in mount else {"data_mount": mount}))
            with self.subTest(label), self.assertCode("hpo_isolation_unproven"):
                fx.collect(records)
        records = fx.snapshot()
        (fx.attempt_dir(key) / "runtime.json").write_text(json.dumps({"data_mount": fx.mount}))
        with self.assertCode("hpo_isolation_unproven"):  # operator pins the real views root
            fx.collect(records, views_root=self.temp())
        fx.collect(records, views_root=fx.views)

    def test_missing_runtime_json_or_docker_log_is_rejected(self):
        fx = self.fx
        records = fx.snapshot()
        (fx.attempt_dir(("c00", 101)) / "runtime.json").unlink()
        with self.assertCode("global_artifacts_invalid"):
            fx.collect(records)
        records = fx.snapshot()
        (fx.attempt_dir(("c00", 101)) / "docker.log").unlink()
        with self.assertCode("global_artifacts_invalid"):
            fx.collect(records)

    # --- declared metric != recomputed -----------------------------------

    def test_declared_metric_that_differs_from_the_recomputation_is_rejected_after_resign(self):
        fx = self.fx
        key = ("c00", 101)

        def inflate(artifact, results):
            artifact["metrics"]["primary"]["value"] = 0.99
            return signed(artifact, "content_sha256"), results

        fx.write_attempt(key, "tie", tweak=inflate)
        records = [fx.record(k, "succeeded") for k in sorted(fx.jobs)]
        for other in sorted(fx.jobs):
            if other != key:
                fx.write_attempt(other, DEFAULT_PROFILE[other[0]])
        with self.assertCode("metric_recompute_mismatch"):
            fx.collect(records)

    def test_scores_edited_with_stale_metrics_are_rejected_after_resign(self):
        fx = self.fx
        key = ("c00", 101)

        def edit(artifact, results):
            artifact["rows"][0]["scores"] = [0.1, 0.9]
            artifact["rows"][1]["scores"] = [0.9, 0.1]
            return signed(artifact, "content_sha256"), results

        for other in sorted(fx.jobs):
            fx.write_attempt(other, DEFAULT_PROFILE[other[0]], tweak=edit if other == key else None)
        records = [fx.record(k, "succeeded") for k in sorted(fx.jobs)]
        with self.assertRaisesRegex(ValueError, "^(decision_recompute_mismatch|metric_recompute_mismatch):"):
            fx.collect(records)

    def test_results_json_declaring_another_metric_is_rejected(self):
        fx = self.fx
        key = ("c00", 101)

        def lie(artifact, results):
            results["clinical_evaluation"]["primary_value"] = 0.75
            return artifact, results

        for other in sorted(fx.jobs):
            fx.write_attempt(other, DEFAULT_PROFILE[other[0]], tweak=lie if other == key else None)
        with self.assertCode("metric_recompute_mismatch"):
            fx.collect([fx.record(k, "succeeded") for k in sorted(fx.jobs)])

    def test_artifact_defects_are_rejected_not_demoted_to_rows(self):
        fx = self.fx
        key = ("c01", 101)
        e_row = next(r for r in fx.rows if r["id"] in set(fx.pools["lists"]["E"]))

        def with_e(artifact, results):
            artifact["rows"].append({"id": e_row["id"], "group": e_row["group"], "target": e_row["label"],
                                     "scores": [0.5, 0.5], "prediction": 1})
            artifact["rows"].sort(key=lambda r: r["id"])
            return signed(artifact, "content_sha256"), results

        def dropped(artifact, results):
            artifact["rows"].pop()
            return signed(artifact, "content_sha256"), results

        def other_config(artifact, results):
            artifact["config_sha256"] = fx.jobs[("c02", 101)]["config_sha"]
            return signed(artifact, "content_sha256"), results

        def wrong_epoch(artifact, results):
            artifact["epoch"] = artifact["epochs"] - 1
            return signed(artifact, "content_sha256"), results

        def not_last(artifact, results):
            artifact["checkpoint_selection"] = "best_selection_loss"
            return signed(artifact, "content_sha256"), results

        cases = {with_e: "hpo_view_contains_E", dropped: "selection_coverage_violation",
                 other_config: "prediction_identity_mismatch", wrong_epoch: "epoch_selection_forbidden",
                 not_last: "epoch_selection_forbidden"}
        for tweak, expected in cases.items():
            for other in sorted(fx.jobs):
                fx.write_attempt(other, DEFAULT_PROFILE[other[0]], tweak=tweak if other == key else None)
            records = [fx.record(k, "succeeded") for k in sorted(fx.jobs)]
            with self.subTest(tweak.__name__), self.assertCode(expected):
                fx.collect(records)

    def test_completion_that_does_not_cover_or_belong_to_the_job_is_rejected(self):
        fx = self.fx
        records = fx.snapshot()
        out = fx.attempt_dir(("c00", 101))
        completed = json.loads((out / "completed.json").read_text())
        for label, mutate, expected in (
                ("override", lambda c: c.update(trainer_override={"device": "cpu"}), "trainer_override_forbidden"),
                ("no override key", lambda c: c.pop("trainer_override"), "trainer_override_forbidden"),
                ("other config", lambda c: c.update(config_sha256=H("x")), "global_artifacts_invalid"),
                ("other split", lambda c: c.update(splits_sha256=H("x")), "global_artifacts_invalid"),
                ("not completed", lambda c: c.update(status="running"), "global_artifacts_invalid"),
                ("predictions unchecked", lambda c: c.update(artifacts={}), "global_artifacts_invalid"),
                ("predictions stale", lambda c: c.update(artifacts={"predictions.json": H("x")}), "global_artifacts_invalid")):
            changed = copy.deepcopy(completed)
            mutate(changed)
            (out / "completed.json").write_text(json.dumps(changed))
            with self.subTest(label), self.assertCode(expected):
                fx.collect(records)

    def test_full_host_validation_runs_for_succeeded_jobs_when_settings_are_given(self):
        fx = self.fx
        records = fx.snapshot()
        with mock.patch.object(runtime, "validate_global_completion") as validate:
            fx.collect(records, settings={"code": "c", "data": "d"})
        self.assertEqual(validate.call_count, 4)
        specs = {call.args[2]["job_id"] for call in validate.call_args_list}
        self.assertEqual(specs, {job["spec"]["job_id"] for job in fx.jobs.values()})
        with mock.patch.object(runtime, "validate_global_completion",
                               side_effect=ValueError("global_artifacts_invalid: nine artifacts expected")):
            with self.assertCode("global_artifacts_invalid"):
                fx.collect(records, settings={"code": "c", "data": "d"})

    # --- search-space / identity binding ---------------------------------------

    def test_search_space_changed_after_jobs_were_seeded_is_search_space_mismatch(self):
        fx = self.fx
        other = self.fixture(grid=[{"path": "optimizer.parameters.lr", "values": [0.00003, 0.0001, 0.0003, 0.002]}])
        self.assertNotEqual(other.space["sha256"], fx.space["sha256"])
        seeded = [other.record(key, "pending") for key in sorted(other.jobs)]
        with self.assertCode("search_space_mismatch"):
            fx.collect(seeded)
        with self.assertCode("search_space_mismatch"):  # c03 only exists in the altered space
            fx.collect(fx.snapshot() + seeded[3:])

    def test_job_of_the_same_pools_bound_to_another_plan_or_extra_seed_is_refused(self):
        fx = self.fx
        records = fx.snapshot()
        spec = copy.deepcopy(records[0]["spec"])
        spec["global"]["plan_sha256"] = H("another plan")  # same job_id, different sealed plan
        with self.assertCode("search_space_mismatch"):
            fx.collect([new_record(spec)] + records[1:])
        extra_seed = copy.deepcopy(records[0]["spec"])
        extra_seed["train_seed"] = 999
        extra_seed.pop("job_id")
        with self.assertCode("search_space_mismatch"):
            fx.collect(records + [new_record(extra_seed)])

    def test_queue_job_with_another_materialized_config_is_config_identity_mismatch(self):
        fx = self.fx
        records = fx.snapshot()
        records[0]["spec"]["config_sha256"] = H("a different config")
        with self.assertCode("config_identity_mismatch"):
            fx.collect(records)

    def test_repeated_job_in_the_snapshot_is_refused(self):
        fx = self.fx
        records = fx.snapshot()
        with self.assertCode("collection_invalid"):
            fx.collect(records + records[:1])

    def test_view_with_an_e_id_or_foreign_listing_never_reaches_the_jobs(self):
        fx = self.fx
        records = fx.snapshot()
        e_row = next(r for r in fx.rows if r["id"] in set(fx.pools["lists"]["E"]))
        leaky = copy.deepcopy(fx.view)
        leaky["lists"]["train"] = sorted(leaky["lists"]["train"] + [e_row["id"]])
        with self.assertCode("hpo_view_contains_E"):
            fx.collect(records, view=signed(leaky))
        other_listing = signed({**fx.view, "view_files_sha256": H("another listing")})
        with self.assertCode("view_hash_mismatch"):
            fx.collect(records, view=other_listing)
        producer = {**fx.producer, "manifest_sha256": H("another listing")}
        plan = copy.deepcopy(fx.plan)
        plan["producer"] = producer
        with self.assertCode("view_hash_mismatch"):
            fx.collect(records, plan=plan, plan_sha256=H(plan))

    def test_plan_not_matching_the_search_space_or_its_anchor_is_refused(self):
        fx = self.fx
        records = fx.snapshot()
        with self.assertCode("plan_invalid"):
            fx.collect(records, plan_sha256=H("not the plan"))
        plan = copy.deepcopy(fx.plan)
        plan["candidates"][1]["recipe"]["optimizer"]["parameters"]["lr"] = 0.0002
        with self.assertCode("plan_not_from_search_space"):
            fx.collect(records, plan=plan, plan_sha256=H(plan))

    # --- v1 ledger invariants on collected evidence ---------------------------

    def test_ledger_row_with_an_e_identity_is_rejected_by_v1_after_resign(self):
        fx = self.fx
        evidence, _ = fx.collect(fx.snapshot())
        e_id = fx.pools["lists"]["E"][0]
        for key in ("train_ids", "select_ids", "epoch_ids", "threshold_ids"):
            mutated = copy.deepcopy(evidence)
            mutated["rows"][1][key] = sorted(mutated["rows"][1][key] + [e_id])
            mutated = signed(mutated)
            freeze = signed(freeze_with_eligibility(fx, fx.freeze_candidate("c01", mutated), mutated))
            with self.subTest(key), self.assertRaisesRegex(ValueError, "never E"):
                fx.audit(mutated, freeze)
            with self.subTest(key, how="seal"), self.assertRaisesRegex(ValueError, "never E"):
                fx.seal(mutated, "c01", self.temp() / "freeze.json")

    def test_ledger_declaring_another_search_candidate_config_is_rejected(self):
        fx = self.fx
        evidence, _ = fx.collect(fx.snapshot())
        mutated = copy.deepcopy(evidence)
        mutated["rows"][1]["config_sha256"] = H("candidate with a different lr")
        mutated = signed(mutated)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            fx.seal(mutated, "c01", self.temp() / "freeze.json")

    def test_queue_collected_evidence_requires_the_real_producer_and_old_kinds_keep_their_reasons(self):
        rows, pools, freeze, evidence = freeze_fixture()
        collected = signed({**evidence, "kind": "queue_collected_v1"})
        freeze = {**freeze, "selection_evidence_sha256": collected["sha256"]}
        with self.assertRaisesRegex(ValueError, "real HPO producer"):
            gd.seal_recipe_freeze(rows, pools, freeze, collected, expected_pools_sha256=pools["sha256"],
                                  expected_plan_sha256=H(freeze["selection_plan"]), output=self.temp() / "f.json")
        sealed = gd.seal_recipe_freeze(rows, pools, {**freeze, "selection_evidence_sha256": evidence["sha256"]},
                                       evidence, expected_pools_sha256=pools["sha256"],
                                       expected_plan_sha256=H(freeze["selection_plan"]), output=self.temp() / "g.json")
        self.assertEqual(sealed["eligibility"]["scientific_reasons"],
                         ["no_integrated_evidence_producer", "synthetic_fixture"])

    # --- review iteration 1 ---------------------------------------------------

    def test_undefined_metric_is_a_completed_undefined_row_with_its_reason(self):
        fx = self.fx
        key = ("c01", 101)
        reason = "metric_undefined:fixture_missing_class"

        def undefine(artifact, results):
            artifact["metrics"]["primary"].update(value=None, reason=reason)
            artifact = signed(artifact, "content_sha256")
            results["clinical_evaluation"].update(primary_value=None, primary_reason=reason,
                                                  content_sha256=artifact["content_sha256"])
            return artifact, results

        for other in sorted(fx.jobs):
            fx.write_attempt(other, DEFAULT_PROFILE[other[0]], tweak=undefine if other == key else None)

        def trusted_summary(cfg, view, records, artifact, *, pools):  # the real recomputation would (rightly) refuse this edit
            primary = artifact["metrics"]["primary"]
            return {"partition": artifact["partition"], "samples": len(artifact["rows"]),
                    "primary_metric": primary["metric"], "primary_value": primary["value"],
                    "primary_reason": primary["reason"], "content_sha256": artifact["content_sha256"]}

        with mock.patch.object(gh.ge, "validate_global_predictions", side_effect=trusted_summary):
            evidence, manifest = fx.collect([fx.record(k, "succeeded") for k in sorted(fx.jobs)])
        row = next(r for r in evidence["rows"] if r["candidate_id"] == "c01")
        self.assertEqual((row["status"], row["metric"]["state"], row["metric"]["value"], row["reason"]),
                         ("completed", "undefined", None, reason))
        self.assertRegex(row["artifact_sha256"], r"^[0-9a-f]{64}$")
        directory = self.temp()
        freeze = fx.seal(evidence, "c00", directory / "freeze.json")
        self.assertIn("ledger_contains_nondefined_trials", freeze["eligibility"]["reasons"])
        self.assertNotIn("c01", fx.audit(evidence, freeze)["candidate_means"])

    def test_retried_job_may_not_hide_an_earlier_attempt_with_predictions(self):
        fx = self.fx
        key = ("c01", 101)
        records = fx.snapshot({key: "absent"})
        fx.write_attempt(key, "good")
        retried = fx.retried_record(key)
        evidence, manifest = fx.collect(records + [retried])
        item = next(r for r in manifest["rows"] if r["candidate_id"] == "c01")
        self.assertEqual((item["attempts"], item["attempt_id"], item["status"]), (2, ATTEMPT, "completed"))
        self.assertAlmostEqual(item["minutes"], 5.0)  # both attempts count against the GPU cap
        fx.write_attempt(key, "bad", attempt=RETRIED)  # the failed attempt had scores: a prediction is never redone
        with self.assertCode("collection_invalid"):
            fx.collect(records + [retried])

    def test_attempt_ids_must_be_the_ids_the_queue_assigns(self):
        fx = self.fx
        records = fx.snapshot()
        records[0]["attempt_id"] = "../" + records[1]["job_id"] + "/" + ATTEMPT
        with self.assertCode("collection_invalid"):
            fx.collect(records)

    def test_terminal_job_without_a_derivable_wall_time_is_rejected_not_exempted(self):
        fx = self.fx
        for mutate in (lambda e: e.update(at="garbage"), lambda e: e.update(at="2026-10-20T10:00:00")):
            records = fx.snapshot()
            mutate(records[0]["history"][-1])
            with self.subTest(), self.assertCode("collection_invalid"):
                fx.collect(records)
        records = fx.snapshot()  # a terminal record whose attempt has a single event has no duration
        only = records[0]["history"][-1]
        only["seq"] = 1
        records[0]["history"] = [only]
        records[0]["version"] = records[0]["seq"] = 1
        with self.assertCode("collection_invalid"):
            fx.collect(records)

    def test_gpu_cap_is_monotone_once_crossed_and_the_boundary_is_inclusive(self):
        fx = self.fixture(budget={"max_candidates": 6, "max_seeds": 2, "max_epochs": 20, "job_timeout_s": 3600,
                                  "gpu_minutes_cap": 5})
        records = fx.snapshot()
        durations = {("c00", 101): 2.0, ("c01", 101): 4.0, ("c02", 101): 1.0, ("c03", 101): 1.0}
        records = [fx.record(key, "succeeded", minutes=durations[key]) for key in sorted(fx.jobs)]
        evidence, manifest = fx.collect(records)
        self.assertEqual([r["status"] for r in evidence["rows"]], ["completed", "timeout", "timeout", "timeout"])
        self.assertAlmostEqual(manifest["gpu_minutes_counted"], 2.0)
        exact = [fx.record(key, "succeeded", minutes=m) for key, m in
                 zip(sorted(fx.jobs), (2.0, 1.5, 1.0, 0.5))]  # exactly 5.0 minutes: not an overrun
        evidence, manifest = fx.collect(exact)
        self.assertEqual({r["status"] for r in evidence["rows"]}, {"completed"})
        self.assertAlmostEqual(manifest["gpu_minutes_counted"], 5.0)

    def test_search_space_hash_alone_binds_the_queue_jobs(self):
        fx = self.fx
        records = fx.snapshot()
        spec = copy.deepcopy(records[0]["spec"])
        spec["global"]["search_space_sha256"] = H("same plan, altered budget or failure policy")
        with self.assertCode("search_space_mismatch"):
            fx.collect([new_record(spec)] + records[1:])

    def test_manifest_records_whether_the_full_host_validation_ran(self):
        fx = self.fx
        records = fx.snapshot()
        self.assertEqual(fx.collect(records)[1]["host_validation"], "minimal")
        with mock.patch.object(runtime, "validate_global_completion"):
            self.assertEqual(fx.collect(records, settings={"code": "c", "data": "d"})[1]["host_validation"], "full")

    # --- review iteration 2 ---------------------------------------------------

    def test_per_job_timeout_is_enforced_by_the_collector_independently_of_scores(self):
        fx = self.fixture(budget={"max_candidates": 6, "max_seeds": 2, "max_epochs": 20, "job_timeout_s": 90,
                                  "gpu_minutes_cap": 120})
        records = [fx.record(key, "succeeded", minutes=2.0 if key[0] == "c01" else 1.0) for key in sorted(fx.jobs)]
        for key in sorted(fx.jobs):
            fx.write_attempt(key, DEFAULT_PROFILE[key[0]])
        evidence, manifest = fx.collect(records)
        rows = {r["candidate_id"]: r for r in evidence["rows"]}
        self.assertEqual((rows["c01"]["status"], rows["c01"]["reason"], rows["c01"]["metric"]["value"]),
                         ("timeout", "job_timeout_s_exceeded", None))  # the best scorer is NOT kept
        self.assertEqual({rows[c]["status"] for c in ("c00", "c02", "c03")}, {"completed"})
        directory = self.temp()
        self.assertEqual(fx.audit(evidence, fx.seal(evidence, "c00", directory / "f.json"))["selected_candidate_id"], "c00")

    def test_attempt_id_traversal_is_refused_wherever_it_appears(self):
        fx = self.fx
        traversal = "../" + "c" * 32
        records = fx.snapshot()
        record = records[0]
        record["attempt_id"] = traversal
        for event in record["history"]:
            event["attempt_id"] = traversal  # consistent history, no earlier attempt to trip the retry rule
        with self.assertCode("collection_invalid"):
            fx.collect(records)
        records = fx.snapshot()
        records[0]["history"][0]["attempt_id"] = traversal  # only the history is malformed
        with self.assertCode("collection_invalid"):
            fx.collect(records)

    def test_history_not_containing_the_current_attempt_has_no_derivable_wall_time(self):
        fx = self.fx
        records = fx.snapshot()
        for event in records[0]["history"]:
            event.pop("attempt_id")
        with self.assertCode("collection_invalid"):
            fx.collect(records)
        records = fx.snapshot()
        for event in records[0]["history"]:
            event["attempt_id"] = RETRIED  # a different attempt than record.attempt_id
        with self.assertCode("collection_invalid"):
            fx.collect(records)

    def test_unfinished_retry_cannot_hide_an_earlier_attempt_with_predictions(self):
        fx = self.fx
        key = ("c01", 101)
        records = fx.snapshot({key: "absent"})
        retried = fx.retried_record(key)
        retried["state"] = "pending"
        retried["attempt_id"] = retried["token"] = retried["owner"] = None
        retried["history"] = retried["history"][:4]
        retried["version"] = retried["seq"] = 4
        fx.write_attempt(key, "good", attempt=RETRIED)
        with self.assertCode("collection_invalid"):
            fx.collect(records + [retried])
        earlier_scores = fx.attempt_dir(key, RETRIED) / "predictions.json"
        earlier_scores.unlink()
        evidence, _ = fx.collect(records + [retried])
        self.assertEqual(next(r for r in evidence["rows"] if r["candidate_id"] == "c01")["status"], "missing")

    def test_results_declaring_a_non_finite_metric_is_a_metric_mismatch(self):
        fx = self.fx
        key = ("c00", 101)
        for other in sorted(fx.jobs):
            fx.write_attempt(other, DEFAULT_PROFILE[other[0]])
        path = fx.attempt_dir(key) / "results.json"
        results = json.loads(path.read_text())
        results["clinical_evaluation"]["primary_value"] = float("nan")
        path.write_text(json.dumps(results))  # JSON NaN: still parseable by Python
        with self.assertCode("metric_recompute_mismatch"):
            fx.collect([fx.record(k, "succeeded") for k in sorted(fx.jobs)])

    def test_plan_without_runtime_is_a_coded_error_and_flags_describe_the_validation_level(self):
        fx = self.fx
        plan = copy.deepcopy(fx.plan)
        del plan["runtime"]
        with self.assertRaisesRegex(ValueError, "^[A-Za-z_]+: "):
            fx.collect(fx.snapshot(), plan=plan, plan_sha256=H(plan))
        records = fx.snapshot()
        weak = fx.collect(records)[1]
        self.assertEqual((weak["host_validation"], weak["mount_source_pinned"], weak["listing_verified"]),
                         ("minimal", False, True))
        with mock.patch.object(runtime, "validate_global_completion"):
            strong = fx.collect(records, settings={"code": "c", "data": "d"}, views_root=fx.views,
                                image_sha256=None)[1]
        self.assertEqual((strong["host_validation"], strong["mount_source_pinned"], strong["listing_verified"]),
                         ("full", True, False))

    def test_queue_collected_evidence_also_requires_the_log_policy_declaration(self):
        rows, pools, freeze, evidence = freeze_fixture()
        plan = copy.deepcopy(freeze["selection_plan"])
        plan["producer"]["id"] = gd.QUEUE_PRODUCER_ID
        collected = signed({**evidence, "kind": "queue_collected_v1", "producer": copy.deepcopy(plan["producer"]),
                            "plan_sha256": H(plan)})
        candidates = gd._plan(plan, pools, H(plan))
        with self.assertRaisesRegex(ValueError, "real HPO producer"):  # right id, logs hash of real logs
            gd._evidence(collected, plan, candidates, pools)
        plan["producer"]["logs_sha256"] = H(gd.QUEUE_LOGS_DECLARATION)
        collected = signed({**collected, "producer": copy.deepcopy(plan["producer"]), "plan_sha256": H(plan)})
        candidates = gd._plan(plan, pools, H(plan))
        self.assertEqual(gd._evidence(collected, plan, candidates, pools)[0], "a")

    # --- review iteration 3 ---------------------------------------------------

    def test_timeout_boundary_per_attempt_rounding_and_negative_durations(self):
        fx = self.fixture(budget={"max_candidates": 6, "max_seeds": 2, "max_epochs": 20, "job_timeout_s": 120,
                                  "gpu_minutes_cap": 120})
        for key in sorted(fx.jobs):
            fx.write_attempt(key, DEFAULT_PROFILE[key[0]])
        exact = [fx.record(key, "succeeded", minutes=2.0) for key in sorted(fx.jobs)]  # == job_timeout_s: allowed
        self.assertEqual({r["status"] for r in fx.collect(exact)[0]["rows"]}, {"completed"})
        over = [fx.record(key, "succeeded", minutes=2.0) for key in sorted(fx.jobs)]
        over[0]["history"][-1]["at"] = "2026-10-20T10:03:01.200000+00:00"  # 120.2 s rounds UP to 121 s
        rows = fx.collect(over)[0]["rows"]
        self.assertEqual((rows[0]["status"], rows[0]["reason"]), ("timeout", "job_timeout_s_exceeded"))
        skew = [fx.record(key, "succeeded", minutes=1.0) for key in sorted(fx.jobs)]
        skew[0]["history"][-1]["at"] = "2026-10-20T10:00:30+00:00"  # terminal event before `running`
        with self.assertCode("collection_invalid"):
            fx.collect(skew)

    def test_timeout_is_per_attempt_and_an_overlong_job_still_consumes_the_gpu_cap(self):
        fx = self.fixture(budget={"max_candidates": 6, "max_seeds": 2, "max_epochs": 20, "job_timeout_s": 240,
                                  "gpu_minutes_cap": 8})
        key = ("c01", 101)
        records = fx.snapshot({key: "absent"})
        fx.write_attempt(key, "good")
        evidence, manifest = fx.collect(records + [fx.retried_record(key, minutes=(3, 3))])
        row = next(r for r in evidence["rows"] if r["candidate_id"] == "c01")
        self.assertEqual(row["status"], "completed")  # two 3-minute attempts: each below the 4-minute timeout
        item = next(r for r in manifest["rows"] if r["candidate_id"] == "c01")
        self.assertAlmostEqual(item["minutes"], 6.0)
        fx2 = self.fixture(budget={"max_candidates": 6, "max_seeds": 2, "max_epochs": 20, "job_timeout_s": 90,
                                   "gpu_minutes_cap": 4})
        for k in sorted(fx2.jobs):
            fx2.write_attempt(k, DEFAULT_PROFILE[k[0]])
        durations = {("c00", 101): 1.0, ("c01", 101): 3.0, ("c02", 101): 1.0, ("c03", 101): 1.0}
        records = [fx2.record(k, "succeeded", minutes=durations[k]) for k in sorted(fx2.jobs)]
        evidence, manifest = fx2.collect(records)
        self.assertEqual([(r["status"], r["reason"]) for r in evidence["rows"]],
                         [("completed", None), ("timeout", "job_timeout_s_exceeded"),
                          ("timeout", "gpu_minutes_cap_exceeded"), ("timeout", "gpu_minutes_cap_exceeded")])
        self.assertAlmostEqual(manifest["gpu_minutes_counted"], 4.0)  # the overlong job's 3 min were spent

    def test_traversal_attempt_id_on_an_earlier_multi_event_attempt_is_refused(self):
        fx = self.fx
        key = ("c01", 101)
        records = fx.snapshot({key: "absent"})
        fx.write_attempt(key, "good")
        retried = fx.retried_record(key)
        for event in retried["history"][:3]:
            event["attempt_id"] = "../" + "d" * 32
        with self.assertCode("collection_invalid"):
            fx.collect(records + [retried])

    def test_a_failed_job_with_a_wrong_mount_is_still_isolation_unproven(self):
        fx = self.fx
        key = ("c02", 101)
        records = fx.snapshot({key: "failed"})
        out = fx.attempt_dir(key)
        out.mkdir(parents=True)
        (out / "runtime.json").write_text(json.dumps({"data_mount": dict(kind="data_root", source=str(fx.data),
                                                                       destination="/data", readonly=True)}))
        with self.assertCode("hpo_isolation_unproven"):
            fx.collect(records)
        (out / "runtime.json").write_text(json.dumps({"data_mount": fx.mount}))
        self.assertEqual(next(r for r in fx.collect(records)[0]["rows"] if r["candidate_id"] == "c02")["status"], "failed")

    def test_collector_output_never_claims_execution_or_scientific_eligibility(self):
        evidence, manifest = self.fx.collect(self.fx.snapshot())
        text = json.dumps([evidence, manifest])
        self.assertNotIn('"execution_verified": true', text)
        self.assertNotIn('"scientific_eligible": true', text)
        self.assertNotIn("eligibility", evidence)


def freeze_with_eligibility(fx, freeze, evidence):
    """A candidate freeze with derived eligibility, as ``seal_recipe_freeze`` would write it."""
    freeze = copy.deepcopy(freeze)
    pool_audit = gd._anchored_pools(fx.rows, fx.pools, fx.pools["sha256"])
    reasons = list(pool_audit["eligibility"]["reasons"])
    freeze["eligibility"] = {"contract_eligible": not reasons, "scientific_eligible": False, "reasons": reasons,
                             "scientific_reasons": gd._scientific_reasons(evidence)}
    freeze["selection_evidence_sha256"] = evidence["sha256"]
    return freeze


if __name__ == "__main__":
    unittest.main()
