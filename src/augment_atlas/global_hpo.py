"""Global HPO producer (slice S2): pre-registered search space, deterministic
candidate expansion, the v1 selection plan, the D-only HPO view and its
physical D-view, and the collector that turns finished HPO jobs into
``queue_collected_v1`` evidence. Offline: no images are decoded, nothing trains.

Fixed protocol rules (decided by Eduardo, not negotiable in this version):

* One HPO per dataset, without augmentation, only inside the development
  partition D (D_fit trains, D_select selects). No E identity is ever part of
  the view, the plan, a job or the ledger; the HPO container mounts a D-only
  directory of hardlinks instead of the data root (ADR-06).
* The search space is a small fixed grid (at most 6 candidates, 2 seeds,
  20 epochs) sealed BEFORE any HPO job exists. ``c00`` is the declared default
  and the exact-tie winner (the v1 ranking breaks ties by ``candidate_id``).
* Selection is the mean over seeds of the last-epoch D_select metric, recomputed
  by the host from the stored scores; a missing/failed job is an explicit ledger
  row (never a zero) and, under ``require_all_defined``, excludes the candidate.
* Caveat (Bradshaw 2023, DOI 10.1148/ryai.220232): reshuffling folds does not
  remove selection bias if patients used to choose hyperparameters fall into a
  test set; that is why D stays out of every test.
* ``execution_verified`` and ``scientific_eligible`` are always false: the
  collector audits files written by a container it cannot observe.

Structural errors are ``ValueError("<code>: <message>")`` with the stable codes
of section 6.11 of ``docs/global-next-integration.md``.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import errno
import itertools
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from . import global_development as gd
from . import global_evaluation as ge
from . import global_jobs as gj
from .contracts import canonical_hash, file_hash
from .data import confined_path
from .schema import validate_record

SEARCH_SPACE_PROTOCOL = "global-hpo-search-space-v1"
COLLECTION_PROTOCOL = "global-hpo-collection-v1"
PRODUCER_ID = gd.QUEUE_PRODUCER_ID
LOGS_DECLARATION = gd.QUEUE_LOGS_DECLARATION
RETRY_POLICY = "infrastructure_only_before_predictions"
FAILURE_POLICIES = ("require_all_defined", "exclude_candidate_on_any_nondefined")
# Closed registry of tunable recipe paths -> the short name stored in candidate.parameters.
GRID_PATHS = {"optimizer.parameters.lr": "lr", "optimizer.parameters.weight_decay": "weight_decay",
              "epochs": "epochs"}
# Hard ceilings of the v1 cheap-HPO budget (spec section 3 and 6.1, Q7).
MAX_CANDIDATES, MAX_SEEDS, MAX_EPOCHS = 6, 2, 20
LISTING_NAME = "LISTING.json"  # same name the host runtime recomputes (runtime.D_VIEW_LISTING)

_SS_KEYS = {"schema_version", "protocol", "scientific_status", "dataset", "pools_sha256", "task", "base_recipe",
            "grid", "default", "seeds", "failure_policy", "budget", "retry_policy", "sha256"}
_BUDGET_KEYS = {"max_candidates", "max_seeds", "max_epochs", "job_timeout_s", "gpu_minutes_cap"}
_LINK_UNSUPPORTED = {errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP, errno.EOPNOTSUPP}
_CODES = frozenset({
    "records_identity_mismatch", "pools_anchor_mismatch", "view_hash_mismatch", "hpo_view_contains_E",
    "hpo_isolation_unproven", "search_space_mismatch", "plan_not_from_search_space", "recipe_adapter_unsupported",
    "recipe_mismatch", "freeze_not_sealed", "freeze_anchor_missing", "campaign_anchor_missing", "anchor_conflict",
    "final_job_before_seal", "hpo_job_after_freeze", "seed_reuse_forbidden", "init_checkpoint_forbidden",
    "checkpoint_lineage_mismatch", "test_access_before_last_epoch", "epoch_selection_forbidden", "D_in_test",
    "E_coverage_violation", "external_recipe_contaminated", "multilabel_policy_missing", "global_config_invalid",
    "hpo_augmentation_forbidden", "materialization_conflict", "plan_invalid", "predictions_invalid",
    "prediction_scores_invalid", "prediction_identity_mismatch", "selection_coverage_violation",
    "decision_recompute_mismatch", "metric_recompute_mismatch", "config_identity_mismatch",
    "trainer_override_forbidden", "global_artifacts_invalid", "search_space_invalid",
    "d_view_materialization_failed", "collection_invalid"})


def _fail(code, message):
    raise ValueError(f"{code}: {message}")


def _guard(code, function, *args, **kwargs):
    """Run a v1 validator; its uncoded ValueErrors become ``code``."""
    try:
        return function(*args, **kwargs)
    except ValueError as error:
        text = str(error)
        if text.partition(":")[0] in _CODES:
            raise
        _fail(code, text)


def _number(value, name, minimum=None, strict=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:  # ints beyond float range
        finite = False
    if not finite:
        _fail("search_space_invalid", f"{name}: finite number required")
    if minimum is not None and (value < minimum or (strict and value == minimum)):
        _fail("search_space_invalid", f"{name}: must be {'>' if strict else '>='} {minimum}")


def _count(value, name, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        _fail("search_space_invalid", f"{name}: integer in [{minimum}, {maximum}] required")


def _closed(value, keys, name, code="search_space_invalid"):
    if type(value) is not dict or set(value) != set(keys):
        _fail(code, f"{name}: closed schema requires exactly {sorted(keys)}")


def _recipe_get(recipe, path):
    node = recipe
    for part in path.split("."):
        node = node[part]
    return node


def _recipe_set(recipe, path, value):
    *parents, leaf = path.split(".")
    node = recipe
    for part in parents:
        node = node[part]
    node[leaf] = value


# --------------------------------------------------------------------------
# Search space (spec 6.1)
# --------------------------------------------------------------------------

def _validate_grid(search_space):
    budget, grid = search_space["budget"], search_space["grid"]
    _closed(budget, _BUDGET_KEYS, "budget")
    _count(budget["max_candidates"], "budget.max_candidates", 1, MAX_CANDIDATES)
    _count(budget["max_seeds"], "budget.max_seeds", 1, MAX_SEEDS)
    _count(budget["max_epochs"], "budget.max_epochs", 1, MAX_EPOCHS)
    _count(budget["job_timeout_s"], "budget.job_timeout_s", 1)
    _count(budget["gpu_minutes_cap"], "budget.gpu_minutes_cap", 1)
    if type(grid) is not list or not grid:
        _fail("search_space_invalid", "grid: nonempty list of {path, values} required")
    seen, size = set(), 1
    for axis in grid:
        _closed(axis, {"path", "values"}, "grid axis")
        path, values = axis["path"], axis["values"]
        if path not in GRID_PATHS or path in seen:
            _fail("search_space_invalid", f"grid path {path!r} is unknown or repeated; allowed: {sorted(GRID_PATHS)}")
        seen.add(path)
        if type(values) is not list or not values:
            _fail("search_space_invalid", f"grid {path}: nonempty value list required")
        for value in values:
            if path == "epochs":
                _count(value, f"grid {path} value", 1, budget["max_epochs"])
            else:
                _number(value, f"grid {path} value", 0, strict=path.endswith("lr"))
        if any(not left < right for left, right in zip(values, values[1:])):
            _fail("search_space_invalid", f"grid {path}: values must be strictly ascending (ordered, unique)")
        size *= len(values)
    if size > budget["max_candidates"]:
        _fail("search_space_invalid", f"grid has {size} candidates; budget.max_candidates is {budget['max_candidates']}")
    default = search_space["default"]
    if type(default) is not dict or set(default) != seen:
        _fail("search_space_invalid", "default must assign exactly the grid paths")
    for axis in grid:
        if not any(canonical_hash(default[axis["path"]]) == canonical_hash(value) for value in axis["values"]):
            _fail("search_space_invalid", f"default for {axis['path']} does not belong to the grid")
    seeds = search_space["seeds"]
    if (type(seeds) is not list or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds) or len(seeds) > budget["max_seeds"]):
        _fail("search_space_invalid", "seeds: unique nonnegative integers, at most budget.max_seeds")


def validate_search_space(search_space, *, pools=None, runtime=None):
    """Validate a sealed search space; with ``pools`` also bind it to them.

    Without ``pools`` only the self-contained rules run (this is what the plan
    audit needs); with ``pools`` the base recipe goes through the v1 recipe
    contract and the closed trainer registry, and dataset/task/pools must match.
    """
    _closed(search_space, _SS_KEYS, "search space")
    try:
        body = canonical_hash({k: v for k, v in search_space.items() if k != "sha256"})
    except (TypeError, ValueError) as error:
        _fail("search_space_invalid", f"search space is not finite canonical JSON ({error})")
    if search_space["sha256"] != body:
        _fail("search_space_invalid", "search space checksum mismatch")
    if (type(search_space["schema_version"]) is not int or search_space["schema_version"] != 1
            or search_space["protocol"] != SEARCH_SPACE_PROTOCOL or search_space["scientific_status"] != "provisional"):
        _fail("search_space_invalid", "only provisional global-hpo-search-space-v1 supported")
    _guard("search_space_invalid", gd._object, search_space["dataset"], {"id", "version"}, "dataset")
    _guard("search_space_invalid", gd._hash, search_space["pools_sha256"], "pools_sha256")
    if search_space["failure_policy"] not in FAILURE_POLICIES:
        _fail("search_space_invalid", "failure_policy must be explicit and supported")
    if search_space["retry_policy"] != RETRY_POLICY:
        _fail("search_space_invalid", f"retry_policy must be {RETRY_POLICY}")
    _validate_grid(search_space)
    recipe = search_space["base_recipe"]
    _guard("recipe_adapter_unsupported", gj._registry, recipe)
    if recipe["epochs"] > search_space["budget"]["max_epochs"]:
        _fail("search_space_invalid", "base_recipe.epochs exceeds budget.max_epochs")
    for axis in search_space["grid"]:
        if canonical_hash(_recipe_get(recipe, axis["path"])) != canonical_hash(search_space["default"][axis["path"]]):
            _fail("search_space_invalid", f"base_recipe must carry the default value at {axis['path']}")
    for candidate in _expand(search_space):  # a candidate the trainer registry refuses must fail before anchoring
        _guard("recipe_adapter_unsupported", gj._registry, candidate["recipe"])
    if pools is not None:
        if type(pools) is not dict or search_space["pools_sha256"] != pools.get("sha256"):
            _fail("search_space_mismatch", "search space is bound to other pools")
        for key in ("dataset", "task"):
            if canonical_hash(search_space[key]) != canonical_hash(pools[key]):
                _fail("search_space_mismatch", f"search space {key} differs from the pools")
        _guard("recipe_adapter_unsupported", gj._global_task, pools["task"])
        weights = (runtime or {"imagenet_weights_sha256": recipe["model"].get("weights_sha256")
                               if type(recipe["model"]) is dict else None})
        _guard("search_space_invalid", gd._recipe, recipe, pools, weights)
        _guard("recipe_adapter_unsupported", gj._decision, pools["task"], recipe["prediction"])
    return search_space


def build_search_space(pools, *, base_recipe, grid, default, seeds, budget,
                       failure_policy="require_all_defined"):
    """Sealed search space bound to ``pools`` (the caller anchors its ``sha256``)."""
    document = gj._seal({
        "schema_version": 1, "protocol": SEARCH_SPACE_PROTOCOL, "scientific_status": "provisional",
        "dataset": deepcopy(pools["dataset"]), "pools_sha256": pools["sha256"], "task": deepcopy(pools["task"]),
        "base_recipe": deepcopy(base_recipe), "grid": deepcopy(grid), "default": deepcopy(default),
        "seeds": list(seeds), "failure_policy": failure_policy, "budget": deepcopy(budget),
        "retry_policy": RETRY_POLICY})
    return validate_search_space(document, pools=pools)


def expand_candidates(search_space):
    """Deterministic candidates: ``c00`` = default, then the product in grid order.

    The rest of the Cartesian product follows the grid order and each axis'
    value order (lexicographic). The v1 ranking breaks exact ties by ascending
    ``candidate_id``, so an exact tie favours the default (pre-declared and
    conservative). Every candidate augmentation is ``none`` with no operations.
    """
    validate_search_space(search_space)
    return _expand(search_space)


def _expand(search_space):
    axes = [(axis["path"], list(axis["values"])) for axis in search_space["grid"]]
    default = tuple(search_space["default"][path] for path, _ in axes)
    rest = [combo for combo in itertools.product(*(values for _, values in axes))
            if canonical_hash(list(combo)) != canonical_hash(list(default))]
    candidates = []
    for index, combo in enumerate([default, *rest]):
        recipe = deepcopy(search_space["base_recipe"])
        for (path, _), value in zip(axes, combo):
            _recipe_set(recipe, path, value)
        candidates.append({"id": f"c{index:02d}",
                           "parameters": {GRID_PATHS[path]: value for (path, _), value in zip(axes, combo)},
                           "recipe": recipe, "augmentation": {"arm": "none", "ops": []}})
    return candidates


# --------------------------------------------------------------------------
# Selection plan v1 (the existing schema, unchanged)
# --------------------------------------------------------------------------

def code_tree_sha256(src_root):
    """Canonical hash of ``[(relative path, sha256)]`` of every regular file in ``src_root``.

    Symlinks are refused; ``__pycache__`` and ``*.pyc`` are skipped. Sorted by POSIX path text
    (same on every OS). It hashes the working tree: the caller must run it on a clean checkout
    of ``runtime.commit``.
    """
    root = Path(src_root).resolve()
    files = []
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            _fail("plan_invalid", "the source tree must contain only regular files")
        if path.is_dir():
            continue
        if not path.is_file():
            _fail("plan_invalid", "the source tree must contain only regular files")
        files.append((path.relative_to(root).as_posix(), path))
    if not files:
        _fail("plan_invalid", "no source files found to hash")
    return canonical_hash([[name, file_hash(path)] for name, path in sorted(files)])


def environment_sha256(image_id, probe):
    return canonical_hash({"image_id": image_id, "probe": probe})


def make_producer(*, code_sha256, environment_sha256, view_files_sha256):
    """The real producer declaration (lacuna L8: ``logs_sha256`` hashes the POLICY, not logs)."""
    return {"id": PRODUCER_ID, "code_sha256": code_sha256, "environment_sha256": environment_sha256,
            "manifest_sha256": view_files_sha256, "logs_sha256": canonical_hash(LOGS_DECLARATION)}


def plan_hash(plan):
    return canonical_hash(plan)


def build_selection_plan(search_space, pools, *, producer, runtime, final_policy):
    """Plan v1 (existing schema) generated from a sealed search space."""
    _guard("pools_anchor_mismatch", gd._checksum, pools)
    _guard("plan_invalid", gd._runtime, runtime)
    validate_search_space(search_space, pools=pools, runtime=runtime)
    _guard("plan_invalid", gd._producer, producer)
    if (producer["id"] != PRODUCER_ID or producer["logs_sha256"] != canonical_hash(LOGS_DECLARATION)
            or producer["environment_sha256"] != runtime["environment_sha256"]):
        _fail("plan_invalid", "producer must be the real HPO producer, declare the log policy and share the runtime environment")
    _guard("plan_invalid", gd._final_policy, final_policy)
    if set(search_space["seeds"]) & set(final_policy["seeds"]):
        _fail("seed_reuse_forbidden", "HPO seeds must be disjoint from the final seeds")
    gj._check_evaluation_seed(pools)
    candidates = expand_candidates(search_space)
    for candidate in candidates:
        gj._registry(candidate["recipe"])
        gj._decision(pools["task"], candidate["recipe"]["prediction"])
    plan = {"schema_version": 1, "dataset": deepcopy(pools["dataset"]), "pools_sha256": pools["sha256"],
            "task": deepcopy(pools["task"]), "endpoint": pools["task"]["endpoint"], "direction": "maximize",
            "candidates": candidates, "seeds": list(search_space["seeds"]), "ranking": "mean_then_candidate_id",
            "failure_policy": search_space["failure_policy"],
            "epoch_policy": "fixed_in_candidate_before_scores", "threshold_policy": "fixed_in_candidate_before_scores",
            "producer": deepcopy(producer), "runtime": deepcopy(runtime), "final_policy": deepcopy(final_policy)}
    _guard("plan_invalid", gd._plan, plan, pools, plan_hash(plan))
    audit_plan_from_search_space(plan, search_space)
    return plan


def audit_plan_from_search_space(plan, search_space):
    """Recompute the expansion and require canonical equality with ``plan``."""
    validate_search_space(search_space)
    if type(plan) is not dict:
        _fail("plan_not_from_search_space", "plan must be an object")
    expected = {"dataset": search_space["dataset"], "pools_sha256": search_space["pools_sha256"],
                "task": search_space["task"], "endpoint": search_space["task"]["endpoint"],
                "candidates": expand_candidates(search_space), "seeds": search_space["seeds"],
                "failure_policy": search_space["failure_policy"], "direction": "maximize",
                "ranking": "mean_then_candidate_id"}
    for key, value in expected.items():
        if key not in plan or canonical_hash(plan[key]) != canonical_hash(value):
            _fail("plan_not_from_search_space", f"plan {key} is not what the sealed search space expands to")
    return {"search_space_sha256": search_space["sha256"], "plan_sha256": plan_hash(plan),
            "candidate_ids": [candidate["id"] for candidate in expected["candidates"]]}


# --------------------------------------------------------------------------
# hpo-view and the physical D-view (spec 6.3)
# --------------------------------------------------------------------------

def _listing(view_records, image_sha256):
    """``[[path, sha256], ...]`` sorted by path, exactly like ``runtime.d_view_listing``."""
    if type(image_sha256) is not dict:
        _fail("records_identity_mismatch", "image_sha256 must map every D image path to its sha256")
    listing = []
    for row in view_records:
        sha = image_sha256.get(row["path"])
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            _fail("records_identity_mismatch", f"no manifest sha256 for D image {row['id']!r}")
        listing.append([row["path"], sha])
    return sorted(listing)


def _d_records(records, pools):
    d_ids = set(pools["lists"]["D"])
    return sorted(({key: row[key] for key in ("id", "path", "group", "label")} for row in records
                   if row["id"] in d_ids), key=lambda row: row["id"])


def hpo_view_file(view):
    return gj._private_name("hpo-view", view["dataset"]["id"], view["sha256"])


def derive_hpo_view(records, pools, *, expected_pools_sha256, image_sha256):
    """The hpo-view: exactly the D records, D_fit/D_select lists, no E identity.

    ``image_sha256`` maps each D image path to the sha256 the host manifest
    declares; the D-view listing hash derived from it enters the view.
    """
    gj._anchored(records, pools, expected_pools_sha256, global_task=True)
    view_records = _d_records(records, pools)
    document = {"schema_version": 1, "protocol": ge.HPO_VIEW_PROTOCOL, "scientific_status": "provisional",
                "dataset": deepcopy(pools["dataset"]), "task": deepcopy(pools["task"]),
                "pools_sha256": expected_pools_sha256, "records": view_records,
                "lists": {"train": list(pools["lists"]["D_fit"]), "selection": list(pools["lists"]["D_select"])},
                "view_files_sha256": canonical_hash(_listing(view_records, image_sha256))}
    return gj._seal(document)


def audit_hpo_view(view, records, pools, *, expected_pools_sha256, image_sha256=None):
    """Check an hpo-view against trusted records/pools; E leakage is reported first.

    A re-signed view that smuggles an E ID or group is ``hpo_view_contains_E``
    (not merely a hash mismatch). With ``image_sha256`` the listing hash is
    recomputed too; without it only its shape is checked.
    """
    gj._anchored(records, pools, expected_pools_sha256, global_task=True)
    if type(view) is not dict or set(view) != ge._HPO_VIEW_KEYS:
        _fail("view_hash_mismatch", f"hpo view: closed schema requires exactly {sorted(ge._HPO_VIEW_KEYS)}")
    try:
        body = canonical_hash({k: v for k, v in view.items() if k != "sha256"})
    except (TypeError, ValueError) as error:
        _fail("view_hash_mismatch", f"hpo view is not canonical JSON ({error})")
    if view["sha256"] != body:
        _fail("view_hash_mismatch", "hpo view checksum mismatch")
    lookup = {row["id"]: row for row in records}
    e_ids = set(pools["lists"]["E"])
    e_groups = {lookup[item]["group"] for item in e_ids}
    shown = view["records"] if type(view["records"]) is list else []
    if any(type(row) is not dict or type(row.get("id")) is not str or type(row.get("group")) is not str
           for row in shown):
        _fail("view_hash_mismatch", "hpo view records must be {id, path, group, label} objects with string identities")
    lists = view["lists"] if type(view["lists"]) is dict else {}
    ids = {row.get("id") for row in shown if type(row) is dict}
    groups = {row.get("group") for row in shown if type(row) is dict}
    for name in ("train", "selection"):
        if type(lists.get(name)) is list:
            ids |= {item for item in lists[name] if type(item) is str}
    if ids & e_ids or groups & e_groups:
        _fail("hpo_view_contains_E", "hpo view declares an E identity or group")
    if (view["schema_version"] != 1 or type(view["schema_version"]) is not int or view["protocol"] != ge.HPO_VIEW_PROTOCOL
            or view["scientific_status"] != "provisional" or view["pools_sha256"] != expected_pools_sha256):
        _fail("view_hash_mismatch", "hpo view is not an hpo-view of these pools")
    for key in ("dataset", "task"):
        if canonical_hash(view[key]) != canonical_hash(pools[key]):
            _fail("view_hash_mismatch", f"hpo view {key} differs from the pools")
    if canonical_hash(view["records"]) != canonical_hash(_d_records(records, pools)):
        _fail("records_identity_mismatch", "hpo view records are not exactly the trusted D records")
    if canonical_hash(view["lists"]) != canonical_hash({"train": pools["lists"]["D_fit"],
                                                        "selection": pools["lists"]["D_select"]}):
        _fail("view_hash_mismatch", "hpo view lists must be exactly D_fit/D_select")
    _guard("view_hash_mismatch", gd._hash, view["view_files_sha256"], "view_files_sha256")
    if image_sha256 is not None and view["view_files_sha256"] != canonical_hash(_listing(view["records"], image_sha256)):
        _fail("view_hash_mismatch", "view_files_sha256 differs from the D image hashes of the host manifest")
    return deepcopy(view)


def _link_or_copy(source, target):
    """Hardlink; on a filesystem that cannot, a hash-verified copy. Returns the method.

    The method is returned (never silent). A copy that cannot complete
    (no space, I/O error) is an error: there is no degraded fallback.
    """
    try:
        os.link(source, target)
        return "hardlink"
    except FileExistsError:
        raise
    except OSError as error:
        if error.errno not in _LINK_UNSUPPORTED:
            _fail("d_view_materialization_failed", f"cannot hardlink a D image ({error.strerror or error.errno})")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    os.close(descriptor)
    try:
        try:
            shutil.copyfile(source, temporary)
        except OSError as error:
            _fail("d_view_materialization_failed",
                  f"hardlink unavailable and verified copy failed ({error.strerror or error.errno}); no silent fallback")
        if file_hash(temporary) != file_hash(source):
            _fail("d_view_materialization_failed", "copied D image differs from its source")
        shutil.copymode(source, temporary)  # mkstemp creates 0600; keep the source mode like a hardlink would
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return "copy"


def _confined_parent(dest, target):
    """Create ``target``'s directory one level at a time, refusing any symlinked component."""
    current = dest
    for part in target.parent.relative_to(dest).parts:
        current = current / part
        if current.is_symlink():
            _fail("hpo_isolation_unproven", "a D-view subdirectory is a symlink; it could redirect writes outside the view")
        current.mkdir(exist_ok=True)


def materialize_d_view(data_root, views_root, view, *, records, pools, image_sha256):
    """Build ``<views_root>/<pools_sha256>/D``: D images (hardlinks), the hpo-view, LISTING.json.

    Every source is checked against the host manifest hash before it enters the
    view, nothing outside D can be linked, and the finished directory must
    reproduce ``view.view_files_sha256`` through ``runtime.d_view_listing`` (the
    exact recomputation the worker performs before mounting). Re-running on a
    finished view is idempotent; any stray or different file is a conflict.
    """
    from . import runtime
    pools_sha = pools["sha256"]
    audit_hpo_view(view, records, pools, expected_pools_sha256=pools_sha, image_sha256=image_sha256)
    gj._safe_name(view["dataset"]["id"], "dataset.id")
    root = Path(data_root).resolve()
    base = Path(views_root).resolve()
    dest = base / pools_sha / "D"
    for part in (base / pools_sha, dest):
        if part.is_symlink():
            _fail("hpo_isolation_unproven", "the D-view directory is a symlink; it could reach outside the views root")
    dest.mkdir(parents=True, exist_ok=True)
    if not dest.resolve().is_relative_to(base):
        _fail("hpo_isolation_unproven", "the D-view directory escapes the views root")
    methods = {"hardlink": 0, "copy": 0, "existing": 0}
    for row in view["records"]:
        relative = row["path"]
        source = confined_path(root, relative)
        if source != root.joinpath(*relative.split("/")) or not source.is_file():
            _fail("hpo_isolation_unproven", f"D image {row['id']!r} is missing or reached through a symlink")
        if file_hash(source) != image_sha256[relative]:
            _fail("records_identity_mismatch", f"D image {row['id']!r} bytes differ from the manifest sha256")
        target = dest.joinpath(*relative.split("/"))
        _confined_parent(dest, target)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or file_hash(target) != image_sha256[relative]:
                _fail("materialization_conflict", f"{relative} exists in the D-view with different content")
            methods["existing"] += 1
        else:
            methods[_link_or_copy(source, target)] += 1
    view_file = hpo_view_file(view)
    listing = _listing(view["records"], image_sha256)
    for relative, data in ((view_file, gj.render_config_bytes(view)), (LISTING_NAME, gj.render_config_bytes(listing))):
        target = dest.joinpath(*relative.split("/"))
        _confined_parent(dest, target)
        gj._write_exclusive(target, data)
    recomputed = runtime.d_view_listing(dest, view_file)
    if (recomputed != listing or [path for path, _ in recomputed] != sorted(row["path"] for row in view["records"])
            or canonical_hash(recomputed) != view["view_files_sha256"]):
        _fail("hpo_isolation_unproven", "the materialized D-view differs from the hpo-view (stray or missing files)")
    return {"dest": str(dest), "view_file": view_file, "files": len(listing), "methods": methods,
            "view_files_sha256": view["view_files_sha256"]}


# --------------------------------------------------------------------------
# Evidence collector (spec 6.7)
# --------------------------------------------------------------------------

def _utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _attempt_ids(record):
    """Attempt IDs in claim order (the queue assigns uuid4 hex)."""
    seen = []
    for event in record["history"]:
        attempt = event.get("attempt_id")
        if attempt is not None and attempt not in seen:
            seen.append(attempt)
    return seen


def _attempt_seconds(record):
    """Wall seconds (integers) of EVERY attempt from queue history, or None when not derivable."""
    ids = _attempt_ids(record)
    if not ids or (record["attempt_id"] is not None and record["attempt_id"] not in ids):
        return None
    seconds = []
    for attempt in ids:
        history = [event for event in record["history"] if event.get("attempt_id") == attempt]
        if len(history) < 2:
            return None
        start = next((event for event in history if event.get("state") == "running"), history[0])
        try:
            elapsed = (_utc(history[-1]["at"]) - _utc(start["at"])).total_seconds()
        except (KeyError, TypeError, ValueError):
            return None
        if elapsed < 0:  # clock skew between queue writers: never clamp a duration to zero
            return None
        seconds.append(math.ceil(elapsed))  # round up: the cap and the timeout are never under-enforced
    return seconds


def _attempts(record):
    return sum(1 for event in record["history"] if event.get("kind") == "claim")


_ATTEMPT_ID = re.compile(r"[0-9a-f]{32}")


def _check_retries(record, runs_root):
    """retry_policy: only an attempt WITHOUT predictions may be repeated (spec 6.1)."""
    current = record["attempt_id"]
    if current is not None and not _ATTEMPT_ID.fullmatch(str(current)):
        _fail("collection_invalid", "queue attempt_id must be the 32-hex id the queue assigns")
    for attempt in _attempt_ids(record):
        if not _ATTEMPT_ID.fullmatch(attempt):
            _fail("collection_invalid", "queue history holds a malformed attempt id")
        if attempt != current and (Path(runs_root) / record["job_id"] / attempt / ge.ARTIFACT_NAME).exists():
            _fail("collection_invalid", "a retried job already had a predictions.json in an earlier attempt")


def _timed_out(record):
    last = record["history"][-1] if record["history"] else {}
    return any("timeout" in str(last.get(key, "")).lower() for key in ("reason", "error", "message", "kind")) or last.get("timeout") is True


def _unavailable_log(job_id, reason):
    """Stable placeholder for the mandatory ``log_sha256`` of a row that has no docker.log."""
    return canonical_hash({"no_docker_log": True, "job_id": job_id, "reason": reason})


def _read_json(path, what):
    try:
        return gd._read(path)
    except (OSError, ValueError, RecursionError) as error:
        _fail("global_artifacts_invalid", f"{what} is missing or unreadable ({type(error).__name__})")


def _check_mount(provenance, view, pools_sha, views_root):
    mount = provenance.get("data_mount") if type(provenance) is dict else None
    expected = {"kind": "d_view", "destination": "/data", "readonly": True, "pools_sha256": pools_sha,
                "hpo_view_sha256": view["sha256"], "view_files_sha256": view["view_files_sha256"]}
    source = mount.get("source") if type(mount) is dict else None
    ok = (type(mount) is dict and isinstance(source, str)
          and canonical_hash({k: v for k, v in mount.items() if k != "source"}) == canonical_hash(expected)
          and Path(source).parts[-2:] == (pools_sha, "D"))
    if ok and views_root is not None:
        ok = Path(source) == Path(views_root).resolve() / pools_sha / "D"
    if not ok:
        _fail("hpo_isolation_unproven", "runtime.json does not record the verified D-view as the only /data mount")


def _read_attempt(out, record, cfg, config_sha, view, records, pools, views_root, settings):
    """Validate one finished attempt; return (artifact, summary, shas)."""
    out = Path(out)
    if out.is_symlink() or not out.is_dir():
        _fail("global_artifacts_invalid", "attempt directory is missing")
    completed = _read_json(out / "completed.json", "completed.json")
    if type(completed) is not dict or "trainer_override" not in completed or completed["trainer_override"] is not None:
        _fail("trainer_override_forbidden", "completed.json records a trainer_override; toy/CPU attempts are never accepted")
    if (completed.get("status") != "completed" or completed.get("phase") != "hpo"
            or completed.get("config_sha256") != config_sha or completed.get("splits_sha256") != view["sha256"]):
        _fail("global_artifacts_invalid", "completion config/split/phase identity mismatch")
    _check_mount(_read_json(out / "runtime.json", "runtime.json"), view, pools["sha256"], views_root)
    if settings is not None:
        from . import runtime
        runtime.validate_global_completion(settings, out, record["spec"])
    prediction_path = out / ge.ARTIFACT_NAME
    files = completed.get("artifacts")
    if (type(files) is not dict or prediction_path.is_symlink() or not prediction_path.is_file()
            or files.get(ge.ARTIFACT_NAME) != file_hash(prediction_path)):
        _fail("global_artifacts_invalid", "predictions.json is not covered by the completion checksums")
    artifact = _read_json(prediction_path, "predictions.json")
    summary = ge.validate_global_predictions(cfg, view, records, artifact, pools=pools)
    results = _read_json(out / "results.json", "results.json")
    declared = results.get("clinical_evaluation") if type(results) is dict else None
    recomputed = {"artifact": ge.ARTIFACT_NAME,
                  **{key: summary[key] for key in ("partition", "samples", "primary_metric", "primary_value",
                                                   "primary_reason", "content_sha256")}}
    try:
        same = canonical_hash(declared) == canonical_hash(recomputed)
    except (TypeError, ValueError):  # NaN/inf or non-JSON values in a declared metric
        same = False
    if not same:
        _fail("metric_recompute_mismatch", "results.clinical_evaluation differs from the metric recomputed from the scores")
    log = out / "docker.log"
    if log.is_symlink() or not log.is_file():
        _fail("global_artifacts_invalid", "docker.log is missing from a succeeded attempt")
    return artifact, summary, {"completed_sha256": file_hash(out / "completed.json"),
                               "predictions_sha256": file_hash(prediction_path), "docker_log_sha256": file_hash(log)}


def _row(candidate, seed, pools, endpoint, status, *, metric_state, value=None, reason=None, artifact=None, log):
    return {"candidate_id": candidate["id"], "seed": seed, "config_sha256": canonical_hash(candidate),
            "train_ids": list(pools["lists"]["D_fit"]), "select_ids": list(pools["lists"]["D_select"]),
            "epoch_ids": list(pools["lists"]["D_select"]), "threshold_ids": list(pools["lists"]["D_select"]),
            "artifact_sha256": artifact, "log_sha256": log, "status": status,
            "metric": {"endpoint": endpoint, "state": metric_state, "value": value, "reason": reason},
            "reason": reason}


def _manifest_row(candidate_id, seed, job_id, status, **fields):
    row = {"candidate_id": candidate_id, "seed": seed, "job_id": job_id, "status": status, "attempt_id": None,
           "attempts": 0, "queue_state": None, "completed_sha256": None, "predictions_sha256": None,
           "recomputed_metric": None, "docker_log_sha256": None, "data_mount_source_kind": None,
           "view_files_sha256": None, "minutes": None}
    row.update(fields)
    return row


def _hpo_code_commit(plan, code_identity, code_repo):
    """(commit the HPO jobs must carry, pinned image or None) under decision B1 / Q6."""
    commit = plan.get("runtime", {}).get("commit") if type(plan) is dict and type(plan.get("runtime")) is dict else None
    if code_identity is None:
        if code_repo is not None:
            _fail("code_identity_mismatch", "code_repo needs the code_identity it re-derives")
        return commit, None
    from . import global_campaign as gc
    if type(code_identity) is not dict:
        _fail("code_identity_mismatch", "code_identity must be the Q6 evidence object")
    identity = gc.audit_code_identity(code_identity, freeze_runtime=plan["runtime"],
                                      code_commit=code_identity.get("code_commit"),
                                      image_id=code_identity.get("image_id"))
    if code_repo is not None:
        derived = gc.derive_code_identity(code_repo, freeze_commit=identity["freeze_commit"],
                                          code_commit=identity["code_commit"], image_id=identity["image_id"])
        if derived != identity:
            _fail("code_identity_mismatch", "Git-derived Q6 evidence differs from the HPO code_identity")
    return identity["code_commit"], identity["image_id"]


def collect_hpo_evidence(search_space, plan, plan_sha256, view, queue_snapshot, runs_root, *, records, pools,
                         dataset_source, image_sha256=None, views_root=None, settings=None, code_identity=None,
                         code_repo=None):
    """Turn finished HPO jobs into ``queue_collected_v1`` evidence + a private manifest.

    For every (candidate, seed) the job is located by its recomputed identity.
    ``succeeded`` jobs are validated and their metric RECOMPUTED from the stored
    scores (a declared value that differs is a rejection, not a row);
    ``failed``/``cancelled``/``interrupted`` become ``failed``/``timeout`` rows;
    absent or not yet terminal jobs become explicit ``missing`` rows. Rows that
    overrun ``budget.gpu_minutes_cap`` (canonical order) become ``timeout`` rows;
    no new round is ever chosen after seeing scores. ``settings`` additionally
    runs the full host validation (``runtime.validate_global_completion``).

    Decision B1 (Eduardo, 22/10/2026): the HPO configs embed ``plan_sha256``,
    which covers ``plan.runtime.commit``, so they are committed AFTER the plan.
    ``code_identity`` is the Q6 evidence (``global_campaign.derive_code_identity``)
    from ``plan.runtime.commit`` to that configs commit: it must pass
    ``same_code_identity_v1`` (ancestor, only ``configs/global/**`` changed,
    identical code tree, pinned image), job identities are then recomputed with
    the configs commit, and every succeeded attempt must record the pinned
    image. ``code_repo`` re-derives the identity from Git and requires equality.
    Without ``code_identity`` the jobs must carry ``plan.runtime.commit`` itself.
    """
    pools_sha = pools["sha256"]
    validate_search_space(search_space, pools=pools, runtime=plan.get("runtime") if type(plan) is dict else None)
    code_commit, image_id = _hpo_code_commit(plan, code_identity, code_repo)
    audit_plan_from_search_space(plan, search_space)
    _guard("plan_invalid", gd._plan, plan, pools, plan_sha256)
    candidates = {c["id"]: c for c in plan["candidates"]}
    view = audit_hpo_view(view, records, pools, expected_pools_sha256=pools_sha, image_sha256=image_sha256)
    if plan["producer"]["manifest_sha256"] != view["view_files_sha256"]:
        _fail("view_hash_mismatch", "plan.producer.manifest_sha256 is not the view_files_sha256 of this D-view")
    configs = gj.build_hpo_configs(records, pools, expected_pools_sha256=pools_sha, dataset_source=dataset_source,
                                   plan=plan, expected_plan_sha256=plan_sha256,
                                   search_space_sha256=search_space["sha256"], hpo_view_sha256=view["sha256"])
    directory = gj.config_dir(pools["dataset"]["id"], "hpo")
    expected = {}
    for name, cfg in configs.items():
        sha = ge.config_sha256(cfg)
        spec = gj.hpo_job_spec(cfg, config_path=f"{directory}/{name}", config_sha256=sha, code_commit=code_commit)
        expected[spec["job_id"]] = (cfg["global"]["candidate_id"], cfg["training"]["seed"], cfg, sha, spec)
    by_id = {}
    for record in queue_snapshot:
        _guard("collection_invalid", validate_record, record)
        if record["job_id"] in by_id:
            _fail("collection_invalid", "queue snapshot repeats a job")
        by_id[record["job_id"]] = record
    for job_id, record in by_id.items():
        spec = record["spec"]
        bound = spec.get("global") if type(spec.get("global")) is dict else {}
        if spec.get("phase") != "hpo" or bound.get("pools_sha256") != pools_sha:
            continue
        if (job_id not in expected or bound.get("search_space_sha256") != search_space["sha256"]
                or bound.get("plan_sha256") != plan_sha256):
            _fail("search_space_mismatch", "the queue holds an HPO job of these pools outside the sealed search space/plan")
    endpoint = plan["endpoint"]
    cap = search_space["budget"]["gpu_minutes_cap"]
    timeout_s = search_space["budget"]["job_timeout_s"]
    ledger, manifest_rows, spent, over_cap = [], [], 0, False  # spent: integer seconds
    order = sorted(expected.items(), key=lambda item: (item[1][0], item[1][1]))
    for job_id, (candidate_id, seed, cfg, config_sha, spec) in order:
        candidate = candidates[candidate_id]
        record = by_id.get(job_id)
        state = None if record is None else record["state"]
        if record is not None:
            for key, value in spec.items():
                if key not in record["spec"] or canonical_hash(record["spec"][key]) != canonical_hash(value):
                    _fail("config_identity_mismatch", f"queue job {key} differs from the candidate's materialized config")
        if record is not None:
            _check_retries(record, runs_root)  # also for non-terminal jobs: an earlier attempt with scores stays visible
        if record is None or state not in ("succeeded", "failed", "cancelled", "interrupted"):
            reason = "job_missing" if record is None else f"job_not_terminal:{state}"
            ledger.append(_row(candidate, seed, pools, endpoint, "missing", metric_state="unavailable", reason=reason,
                               log=_unavailable_log(job_id, reason)))
            manifest_rows.append(_manifest_row(candidate_id, seed, job_id, "missing", queue_state=state,
                                               attempts=0 if record is None else _attempts(record),
                                               attempt_id=None if record is None else record["attempt_id"]))
            continue
        attempt_seconds = _attempt_seconds(record)
        if attempt_seconds is None:
            _fail("collection_invalid", f"cannot derive the wall time of terminal job {job_id[:12]} from the queue history")
        seconds = sum(attempt_seconds)
        common = dict(attempt_id=record["attempt_id"], attempts=_attempts(record), queue_state=state,
                      minutes=seconds / 60)
        out = Path(runs_root) / job_id / record["attempt_id"]
        log_path = out / "docker.log"
        has_log = log_path.is_file() and not log_path.is_symlink()
        if state != "succeeded" and (out / "runtime.json").is_file():  # a mis-mounted job that failed is still unproven
            _check_mount(_read_json(out / "runtime.json", "runtime.json"), view, pools_sha, views_root)
        if state != "succeeded" and any(os.path.lexists(out / name) for name in (ge.ARTIFACT_NAME, "results.json")):
            # B-1'': scores were written and then the job ended without success. This runs BEFORE the cap and
            # timeout shortcuts below, which would otherwise turn such a row into a plain timeout (a candidate
            # dropped after its own score was visible). A succeeded attempt keeps its legitimate timeout row.
            _fail("collection_invalid", f"non-succeeded HPO job {job_id[:12]} already wrote scores")
        if over_cap or spent + seconds > cap * 60:
            over_cap = True  # monotone: once the cap is crossed every later row is a timeout
            reason = "gpu_minutes_cap_exceeded"
            log = file_hash(log_path) if has_log else _unavailable_log(job_id, reason)
            ledger.append(_row(candidate, seed, pools, endpoint, "timeout", metric_state="unavailable",
                               reason=reason, log=log))
            manifest_rows.append(_manifest_row(candidate_id, seed, job_id, "timeout",
                                               docker_log_sha256=file_hash(log_path) if has_log else None, **common))
            continue
        spent += seconds
        if max(attempt_seconds) > timeout_s:  # score-independent: the pre-registered per-job timeout
            reason = "job_timeout_s_exceeded"
            ledger.append(_row(candidate, seed, pools, endpoint, "timeout", metric_state="unavailable", reason=reason,
                               log=file_hash(log_path) if has_log else _unavailable_log(job_id, reason)))
            manifest_rows.append(_manifest_row(candidate_id, seed, job_id, "timeout",
                                               docker_log_sha256=file_hash(log_path) if has_log else None, **common))
            continue
        if state != "succeeded":
            status = "timeout" if _timed_out(record) else "failed"
            reason = f"job_{'timeout' if status == 'timeout' else state}"
            ledger.append(_row(candidate, seed, pools, endpoint, status, metric_state="unavailable", reason=reason,
                               log=file_hash(log_path) if has_log else _unavailable_log(job_id, reason)))
            manifest_rows.append(_manifest_row(candidate_id, seed, job_id, status,
                                               docker_log_sha256=file_hash(log_path) if has_log else None, **common))
            continue
        artifact, summary, shas = _read_attempt(out, record, cfg, config_sha, view, records, pools, views_root, settings)
        if image_id is not None:
            provenance = _read_json(out / "runtime.json", "runtime.json")
            if type(provenance) is not dict or provenance.get("image_id") != image_id:
                _fail("code_identity_mismatch", "an HPO attempt ran on another image than the pinned Q6 image")
        primary = artifact["metrics"]["primary"]
        if primary["value"] is None:
            ledger.append(_row(candidate, seed, pools, endpoint, "completed", metric_state="undefined",
                               reason=primary["reason"], artifact=artifact["content_sha256"], log=shas["docker_log_sha256"]))
        else:
            ledger.append(_row(candidate, seed, pools, endpoint, "completed", metric_state="defined",
                               value=primary["value"], artifact=artifact["content_sha256"], log=shas["docker_log_sha256"]))
        manifest_rows.append(_manifest_row(
            candidate_id, seed, job_id, "completed", completed_sha256=shas["completed_sha256"],
            predictions_sha256=shas["predictions_sha256"], recomputed_metric=primary["value"],
            docker_log_sha256=shas["docker_log_sha256"], data_mount_source_kind="d_view",
            view_files_sha256=view["view_files_sha256"], **common))
    evidence = gj._seal({"schema_version": 1, "kind": gd.QUEUE_COLLECTED_KIND, "producer": deepcopy(plan["producer"]),
                         "dataset": deepcopy(pools["dataset"]), "pools_sha256": pools_sha, "plan_sha256": plan_sha256,
                         "task": deepcopy(pools["task"]), "rows": ledger})
    try:
        gd._evidence(evidence, plan, candidates, pools)
    except ValueError as error:
        # A ledger with no fully defined candidate is a valid (useless) record of what ran.
        if not str(error).startswith("no fully defined candidate"):
            raise
    manifest = gj._seal({"schema_version": 1, "protocol": COLLECTION_PROTOCOL, "pools_sha256": pools_sha,
                         "plan_sha256": plan_sha256, "search_space_sha256": search_space["sha256"],
                         "evidence_sha256": evidence["sha256"], "hpo_view_sha256": view["sha256"],
                         "host_validation": "minimal" if settings is None else "full",
                         "mount_source_pinned": views_root is not None, "listing_verified": image_sha256 is not None,
                         "code_commit": code_commit,
                         "code_identity_sha256": None if code_identity is None else canonical_hash(code_identity),
                         "gpu_minutes_counted": spent / 60, "gpu_minutes_cap": cap, "rows": manifest_rows})
    return evidence, manifest
