"""Global D/E job contracts (slice S0): closed hpo/final configs, the closed
recipe->trainer registry, final fold views, queue specs and deterministic
config materialization. Offline only: no images, training, queue or Docker.

Fixed protocol rules (decided by Eduardo, not negotiable in this version):

* D (about 20 % of patients/groups) is never a test set. HPO only ever sees
  D_fit (training) and D_select (selection); no E identity reaches HPO.
* Recipe, preprocessing, epochs and thresholds are frozen and sealed before E
  is opened; the 5 E folds use a new evaluation seed distinct from the
  development/selection seeds.
* Final training of fold f uses D union (E minus E_f), a fresh ImageNet model,
  fixed epochs and the last epoch; no epoch/model is chosen on E_f and no HPO
  checkpoint is ever loaded (there is no config key that could request it).
* Caveat (Bradshaw 2023, DOI 10.1148/ryai.220232): reshuffling folds does not
  remove selection bias if patients used to choose hyperparameters fall into
  a test set; that is why D stays out of every test.
* ``execution_verified`` and ``scientific_eligible`` are always false here.

Structural errors are ``ValueError("<code>: <message>")`` with the stable codes
of section 6.11 of ``docs/global-next-integration.md``.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from . import global_development as gd
from .contracts import ARMS, canonical_hash
from .schema import prepare_job

HPO_PROTOCOL = "global-hpo-v1"
FINAL_PROTOCOL = "global-final-v1"
FINAL_VIEW_ALGORITHM = "global-final-view-v1"
EXTERNAL_FREEZE_PROTOCOL = "global-recipe-external-v1"
MANIFEST_PROTOCOL = "global-configs-manifest-v1"
FREEZE_KINDS = ("hpo_d", "external_fixed")
FORBIDDEN_KEYS = frozenset({"init_checkpoint", "pretrained_path", "warm_start", "evaluation_scope"})
BINARY_THRESHOLD_V1 = 0.5  # Q4: fixed a priori, compared with "ge"; never tuned on D_select.
DEFAULT_COMPATIBILITY = {"min_memory_mb": 1000}

# Closed recipe->trainer registry v1. Anything else is recipe_adapter_unsupported.
OPTIMIZERS_V1 = {"adamw": ("lr", "weight_decay")}
SCHEDULERS_V1 = {"cosine_per_epoch": ("eta_min",)}
LOSSES_V1 = {"cross_entropy": "softmax", "bce_with_logits_masked": "sigmoid"}
SCALAR_TASKS = {"binary": "binary_auroc", "ordinal": "quadratic_weighted_kappa",
                "multiclass": "macro_auroc_ovr"}

# Trainer behaviour that is NOT part of the recipe; copied into the campaign seal.
TRAINER_CONSTANTS_V1 = {
    "version": "global-trainer-v1",
    "architecture": "resnet50",
    "initialization": "imagenet_fresh_seeded_linear_head",
    "gradient_clip_max_norm": 1.0,
    "batchnorm": "frozen_eval_mode",
    "precision": "float32",
    "autocast": False,
    "tf32": False,
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "resize": "bilinear_square_input_size",
    "resize_order": "resize_before_augmentation",
    "pixel_scale": "divide_by_255",
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "scheduler_step": "per_epoch",
    "partial_batch": "sample_weighted_mean_within_accumulation_window",
    "drop_last": False,
    "accumulation_tail": "step",
    "checkpoint_selection": "last_epoch",
    "early_stopping": False,
    "device": "cuda",
    "workers": 2,
}

# Q2 (decided 19/10/2026): external fixed recipe for INbreast, no HPO.
Q2_EXTERNAL_SOURCE = {
    "kind": "convention",
    "reference": "convenção de fine-tuning, sem scores do INbreast",
    "derived_from_this_dataset_scores": False,
    "notes": ("AdamW lr 1e-4, wd 1e-4, cosine per epoch, 20 epochs, batch 16, 224 px, "
              "unweighted cross-entropy; nothing chosen from the INbreast pilot."),
}

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CONFIG_KEYS = {"schema_version", "phase", "scientific_status", "protocol_version", "dataset", "global",
                "model", "input", "augmentation", "training", "evaluation"}
_GLOBAL_KEYS = {"pools_file", "pools_sha256", "view", "hpo_view_file", "hpo_view_sha256", "plan_sha256",
                "search_space_sha256", "candidate_id", "candidate_config_sha256", "freeze_kind",
                "freeze_file", "freeze_sha256", "fold"}
_HPO_ONLY = ("hpo_view_file", "hpo_view_sha256", "plan_sha256", "search_space_sha256", "candidate_id",
             "candidate_config_sha256")
_FINAL_ONLY = ("freeze_kind", "freeze_file", "freeze_sha256", "fold")
_TRAINING_KEYS = {"seed", "epochs", "batch_size", "accumulation_steps", "optimizer", "scheduler", "loss",
                  "checkpoint_selection", "early_stopping", "device", "workers", "deterministic",
                  "autocast", "tf32"}


def _fail(code, message):
    raise ValueError(f"{code}: {message}")


def _invalid(message):
    _fail("global_config_invalid", message)


def _closed(value, keys, name, code="global_config_invalid"):
    if type(value) is not dict or set(value) != set(keys):
        _fail(code, f"{name}: closed schema requires exactly {sorted(keys)}")


def _text(value, name):
    if type(value) is not str or not value or value != value.strip():
        _invalid(f"{name}: nonempty trimmed string required")


def _int(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        _invalid(f"{name}: integer in [{minimum}, {maximum}] required")


def _hex64(value, name):
    if type(value) is not str or not _HEX64.fullmatch(value):
        _invalid(f"{name}: lowercase SHA256 required")


def _commit(value, name="code_commit"):
    if type(value) is not str or not _HEX40.fullmatch(value):
        _invalid(f"{name}: full lowercase 40-hex commit required")


def _finite(value, name, minimum, strict=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < minimum or (strict and value == minimum)):
        _fail("recipe_adapter_unsupported", f"{name}: finite number {'>' if strict else '>='} {minimum} required")


def _safe_name(value, name):
    if type(value) is not str or not _SAFE_NAME.fullmatch(value) or ".." in value:
        _invalid(f"{name}: path-safe name [A-Za-z0-9._-] required")


def _relative_path(value, name):
    _text(value, name)
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or ":" in value or "\\" in value
            or path.as_posix() != value or not path.parts):
        _invalid(f"{name}: normalized, relative and confined POSIX path required")


def _scan_forbidden(value, where="config"):
    """Warm-start/checkpoint keys are refused anywhere, before the closed schema."""
    if type(value) is dict:
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                _fail("init_checkpoint_forbidden", f"{where}.{key} is not representable in global jobs")
            _scan_forbidden(item, f"{where}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _scan_forbidden(item, f"{where}[{index}]")


def _expected_ops(arm):
    ops = ["basic"] if arm == "basic" or arm.startswith("basic_") else []
    if "gamma" in arm:
        ops.append("gamma")
    if "clahe" in arm:
        ops.append("clahe")
    return ops


def _augmentation(value, arm=None):
    if type(value) is not dict or "arm" not in value or "ops" not in value:
        _invalid("augmentation requires arm and ops")
    if value["arm"] not in ARMS or (arm is not None and value["arm"] != arm):
        _invalid("augmentation arm unknown or not the declared arm")
    ops = value["ops"]
    if type(ops) is not list or any(type(op) is not dict for op in ops):
        _invalid("augmentation.ops must be a list of objects")
    if [op.get("op") for op in ops] != _expected_ops(value["arm"]):
        _invalid("augmentation operations/order must match the declared arm")
    _json_check(value, "augmentation")


def _json_check(value, name):
    try:
        gd._json(value, name)
    except ValueError as error:
        _invalid(str(error))


# --------------------------------------------------------------------------
# Recipe -> trainer registry
# --------------------------------------------------------------------------

def _registry(recipe):
    """Check the training-relevant recipe fields against the closed registry."""
    if type(recipe) is not dict:
        _fail("recipe_adapter_unsupported", "recipe must be an object")
    for key in ("optimizer", "scheduler", "loss", "batch_size", "accumulation_steps", "epochs"):
        if key not in recipe:
            _fail("recipe_adapter_unsupported", f"recipe.{key} missing")
    for key in ("batch_size", "accumulation_steps", "epochs"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            _fail("recipe_adapter_unsupported", f"recipe.{key}: integer >= 1 required")
    for section, registry in (("optimizer", OPTIMIZERS_V1), ("scheduler", SCHEDULERS_V1)):
        block = recipe[section]
        if type(block) is not dict or set(block) != {"name", "parameters"}:
            _fail("recipe_adapter_unsupported", f"recipe.{section} requires name and parameters")
        if block["name"] not in registry:
            _fail("recipe_adapter_unsupported", f"{section} {block['name']!r} is not in the v1 trainer registry")
        if type(block["parameters"]) is not dict or set(block["parameters"]) != set(registry[block["name"]]):
            _fail("recipe_adapter_unsupported",
                  f"{section} {block['name']} requires parameters exactly {list(registry[block['name']])}")
    parameters = recipe["optimizer"]["parameters"]
    _finite(parameters["lr"], "optimizer.lr", 0, strict=True)
    _finite(parameters["weight_decay"], "optimizer.weight_decay", 0)
    _finite(recipe["scheduler"]["parameters"]["eta_min"], "scheduler.eta_min", 0)
    if recipe["scheduler"]["parameters"]["eta_min"] >= parameters["lr"]:
        _fail("recipe_adapter_unsupported", "scheduler.eta_min must be below optimizer.lr")
    loss = recipe["loss"]
    if type(loss) is not dict or "name" not in loss or loss["name"] not in LOSSES_V1:
        _fail("recipe_adapter_unsupported", "loss is not in the v1 trainer registry")
    if loss.get("parameters") != {}:
        _fail("recipe_adapter_unsupported", "v1 trainer losses take no parameters (no class weights)")


def _fixed_training():
    constants = TRAINER_CONSTANTS_V1
    return {"checkpoint_selection": constants["checkpoint_selection"],
            "early_stopping": constants["early_stopping"], "device": constants["device"],
            "workers": constants["workers"], "deterministic": constants["deterministic_algorithms"],
            "autocast": constants["autocast"], "tf32": constants["tf32"]}


def recipe_to_training(recipe, seed):
    """The only mapping from a frozen/candidate recipe to the trainer block."""
    _registry(recipe)
    if type(seed) is not int or seed < 0:
        _invalid("training seed must be a nonnegative integer")
    return {"seed": seed, "epochs": recipe["epochs"], "batch_size": recipe["batch_size"],
            "accumulation_steps": recipe["accumulation_steps"],
            "optimizer": {"name": recipe["optimizer"]["name"],
                          "lr": recipe["optimizer"]["parameters"]["lr"],
                          "weight_decay": recipe["optimizer"]["parameters"]["weight_decay"]},
            "scheduler": {"name": recipe["scheduler"]["name"],
                          "eta_min": recipe["scheduler"]["parameters"]["eta_min"]},
            "loss": {"name": recipe["loss"]["name"]}, **_fixed_training()}


def _decision(task, prediction):
    """Global v1 decision constants per task (Q4; spec 6.9)."""
    kind = task["type"]
    if kind not in SCALAR_TASKS:
        _fail("recipe_adapter_unsupported", "multilabel global jobs are slice S6 (protocol v2)")
    if type(prediction) is not dict:
        _fail("recipe_adapter_unsupported", "recipe.prediction must be an object")
    rule = prediction.get("rule")
    if kind == "binary":
        if (rule != "fixed_threshold" or prediction.get("threshold_policy") != "fixed_before_E"
                or prediction.get("thresholds") != [BINARY_THRESHOLD_V1]):
            _fail("recipe_adapter_unsupported", "global v1 binary decision is fixed_threshold 0.5 with ge (Q4)")
        return {"rule": "fixed_threshold", "threshold": BINARY_THRESHOLD_V1, "comparison": "ge", "tie_break": None}
    if rule != "argmax":
        _fail("recipe_adapter_unsupported", "global v1 ordinal/multiclass decision is argmax with tie_break first")
    return {"rule": "argmax", "threshold": None, "comparison": None, "tie_break": "first"}


def _evaluation_block(task, recipe):
    return {"schema_version": 1, "task": task["type"], "classes": list(task["classes"]),
            "positive_class": task["positive_class"], "endpoint": task["endpoint"],
            "score_kind": "softmax_probability", "decision": _decision(task, recipe["prediction"]),
            "frozen": True}


def _model_block(recipe, task):
    return {"name": "resnet50", "weights": "IMAGENET1K_V2", "weights_sha256": recipe["model"]["weights_sha256"],
            "num_outputs": len(task["classes"]), "head": LOSSES_V1[recipe["loss"]["name"]],
            "init": "imagenet_fresh"}


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

def _private_name(kind, dataset_id, sha):
    return f"global/{dataset_id}/{kind}-{sha}.json"


def validate_global_config(cfg, *, recipe=None):
    """Validate a closed phase=hpo|final config; with ``recipe``, bind it too.

    ``recipe`` is the frozen recipe (final) or the candidate recipe (hpo); the
    training/model/input/decision blocks must equal what it maps to.
    """
    _scan_forbidden(cfg)
    _closed(cfg, _CONFIG_KEYS, "global config")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        _invalid("schema_version must be integer 1")
    if cfg["phase"] not in ("hpo", "final") or cfg["scientific_status"] != "provisional":
        _invalid("phase must be hpo|final with scientific_status=provisional")
    hpo = cfg["phase"] == "hpo"
    if cfg["protocol_version"] != (HPO_PROTOCOL if hpo else FINAL_PROTOCOL):
        _invalid("protocol_version does not match phase")

    dataset = cfg["dataset"]
    _closed(dataset, {"id", "version", "adapter", "manifest", "manifest_sha256", "require_image_hashes"}, "dataset")
    _safe_name(dataset["id"], "dataset.id")
    _text(dataset["version"], "dataset.version")
    if dataset["adapter"] not in ("manifest_csv", "inbreast_csv"):
        _invalid("dataset.adapter must be manifest_csv or inbreast_csv")
    _relative_path(dataset["manifest"], "dataset.manifest")
    _hex64(dataset["manifest_sha256"], "dataset.manifest_sha256")
    if dataset["require_image_hashes"] is not True:
        _invalid("dataset.require_image_hashes must be true")

    view = cfg["global"]
    _closed(view, _GLOBAL_KEYS, "global")
    _hex64(view["pools_sha256"], "global.pools_sha256")
    if view["pools_file"] != _private_name("pools", dataset["id"], view["pools_sha256"]):
        _invalid("global.pools_file must be global/<dataset>/pools-<pools_sha256>.json")
    if view["view"] != cfg["phase"]:
        _invalid("global.view must equal phase")
    present, absent = (_HPO_ONLY, _FINAL_ONLY) if hpo else (_FINAL_ONLY, _HPO_ONLY)
    if any(view[key] is not None for key in absent):
        _invalid(f"{cfg['phase']} config requires null {list(absent)}")
    if hpo:
        for key in ("hpo_view_sha256", "plan_sha256", "search_space_sha256", "candidate_config_sha256"):
            _hex64(view[key], f"global.{key}")
        _safe_name(view["candidate_id"], "global.candidate_id")
        if view["hpo_view_file"] != _private_name("hpo-view", dataset["id"], view["hpo_view_sha256"]):
            _invalid("global.hpo_view_file must be global/<dataset>/hpo-view-<sha>.json")
    else:
        if view["freeze_kind"] not in FREEZE_KINDS:
            _invalid("global.freeze_kind must be hpo_d or external_fixed")
        _hex64(view["freeze_sha256"], "global.freeze_sha256")
        prefix = "freeze" if view["freeze_kind"] == "hpo_d" else "external-freeze"
        if view["freeze_file"] != _private_name(prefix, dataset["id"], view["freeze_sha256"]):
            _invalid(f"global.freeze_file must be global/<dataset>/{prefix}-<sha>.json")
        _int(view["fold"], "global.fold", 0, 4)

    model = cfg["model"]
    _closed(model, {"name", "weights", "weights_sha256", "num_outputs", "head", "init"}, "model")
    if model["name"] != "resnet50" or model["weights"] != "IMAGENET1K_V2" or model["init"] != "imagenet_fresh":
        _invalid("model must be a fresh ImageNet ResNet50 (IMAGENET1K_V2, init=imagenet_fresh)")
    _hex64(model["weights_sha256"], "model.weights_sha256")
    _int(model["num_outputs"], "model.num_outputs", 2)
    if model["head"] != "softmax":
        _fail("recipe_adapter_unsupported", "sigmoid head (multilabel) is slice S6")

    inputs = cfg["input"]
    _closed(inputs, {"size", "color"}, "input")
    _int(inputs["size"], "input.size", 32)
    if inputs["color"] not in ("grayscale_to_rgb", "rgb"):
        _invalid("input.color must be grayscale_to_rgb or rgb")

    if hpo:
        if cfg["augmentation"] != {"arm": "none", "ops": []}:
            _fail("hpo_augmentation_forbidden", "HPO jobs require augmentation exactly {arm: none, ops: []}")
    else:
        _augmentation(cfg["augmentation"])

    training = cfg["training"]
    _closed(training, _TRAINING_KEYS, "training")
    _closed(training["optimizer"], {"name", "lr", "weight_decay"}, "training.optimizer", "recipe_adapter_unsupported")
    _closed(training["scheduler"], {"name", "eta_min"}, "training.scheduler", "recipe_adapter_unsupported")
    _closed(training["loss"], {"name"}, "training.loss", "recipe_adapter_unsupported")
    # Re-run the registry on the trainer block so a hand-edited config cannot
    # smuggle an unsupported optimizer/scheduler/loss past the recipe mapping.
    _registry({"optimizer": {"name": training["optimizer"]["name"],
                             "parameters": {k: v for k, v in training["optimizer"].items() if k != "name"}},
               "scheduler": {"name": training["scheduler"]["name"],
                             "parameters": {k: v for k, v in training["scheduler"].items() if k != "name"}},
               "loss": {"name": training["loss"]["name"], "parameters": {}},
               "batch_size": training["batch_size"], "accumulation_steps": training["accumulation_steps"],
               "epochs": training["epochs"]})
    _int(training["seed"], "training.seed")
    for key, fixed in _fixed_training().items():
        if type(training[key]) is not type(fixed) or training[key] != fixed:
            _invalid(f"training.{key} is fixed by TRAINER_CONSTANTS_V1")
    if LOSSES_V1[training["loss"]["name"]] != model["head"]:
        _fail("recipe_adapter_unsupported", "loss and model head disagree")

    evaluation = cfg["evaluation"]
    _closed(evaluation, {"schema_version", "task", "classes", "positive_class", "endpoint", "score_kind",
                         "decision", "frozen"}, "evaluation")
    if type(evaluation["schema_version"]) is not int or evaluation["schema_version"] != 1 or evaluation["frozen"] is not True:
        _invalid("evaluation requires schema_version 1 and frozen=true")
    task = {"type": evaluation["task"], "classes": evaluation["classes"],
            "positive_class": evaluation["positive_class"], "endpoint": evaluation["endpoint"]}
    if task["type"] not in SCALAR_TASKS:
        _fail("recipe_adapter_unsupported", "multilabel global jobs are slice S6 (protocol v2)")
    if task["endpoint"] != SCALAR_TASKS[task["type"]]:
        _invalid("evaluation.endpoint does not match task")
    classes = task["classes"]
    if (type(classes) is not list or len(classes) != model["num_outputs"]
            or any(type(c) is not int or c < 0 for c in classes) or classes != sorted(set(classes))):
        _invalid("evaluation.classes must be ordered unique integers, one per model output")
    if task["type"] == "binary":
        if task["positive_class"] not in classes or type(task["positive_class"]) is not int or len(classes) != 2:
            _invalid("binary evaluation requires two classes and a positive_class among them")
    elif task["positive_class"] is not None:
        _invalid("positive_class applies only to binary tasks")
    if evaluation["score_kind"] != "softmax_probability":
        _invalid("scalar global tasks score softmax probabilities")
    _closed(evaluation["decision"], {"rule", "threshold", "comparison", "tie_break"}, "evaluation.decision")
    if task["type"] == "binary":
        expected_decision = _decision(task, {"rule": "fixed_threshold", "threshold_policy": "fixed_before_E",
                                             "thresholds": [BINARY_THRESHOLD_V1]})
    else:
        expected_decision = _decision(task, {"rule": "argmax", "threshold_policy": "not_applicable", "thresholds": []})
    if canonical_hash(evaluation["decision"]) != canonical_hash(expected_decision):
        _fail("recipe_adapter_unsupported", "evaluation.decision violates the global v1 decision constants")
    _json_check(cfg, "config")

    if recipe is not None:
        if canonical_hash(training) != canonical_hash(recipe_to_training(recipe, training["seed"])):
            _fail("recipe_mismatch", "training differs from recipe_to_training(recipe, seed)")
        model_identity = recipe.get("model")
        if (type(model_identity) is not dict or model_identity.get("weights_sha256") != model["weights_sha256"]
                or model_identity.get("name") != model["name"] or model_identity.get("weights") != model["weights"]):
            _fail("recipe_mismatch", "model identity differs from recipe")
        if canonical_hash(inputs) != canonical_hash(recipe.get("input")):
            _fail("recipe_mismatch", "input differs from recipe")
        prediction = recipe.get("prediction")
        if type(prediction) is not dict or canonical_hash(_decision(task, prediction)) != canonical_hash(evaluation["decision"]):
            _fail("recipe_mismatch", "evaluation decision differs from recipe prediction")
    return cfg


def render_config_bytes(cfg):
    """Deterministic on-disk bytes (sorted keys, ASCII, LF, trailing newline)."""
    return (json.dumps(cfg, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Pools, final fold views and freezes
# --------------------------------------------------------------------------

def _anchored(records, pools, expected_pools_sha256, *, global_task=False):
    if type(pools) is not dict or pools.get("sha256") != expected_pools_sha256:
        _fail("pools_anchor_mismatch", "pools document does not match the external anchor")
    try:
        audit = gd._anchored_pools(records, pools, expected_pools_sha256)
    except ValueError as error:
        code = "records_identity_mismatch" if "manifest identity" in str(error) else "pools_anchor_mismatch"
        _fail(code, str(error))
    if global_task:
        _global_task(pools["task"])
    return audit


def _seal(document):
    value = deepcopy(document)
    value.pop("sha256", None)
    value["sha256"] = canonical_hash(value)
    return value


def derive_final_view_doc(records, pools, fold, *, expected_pools_sha256, freeze_sha256):
    """Private ``splits.json`` body of a final job: train = D u (E \\ E_f), test = E_f.

    There is deliberately no selection partition. ``sha256`` is the job's
    ``split_hash``; it binds the pools, the freeze and the fold.
    """
    _anchored(records, pools, expected_pools_sha256)
    _hex64(freeze_sha256, "freeze_sha256")
    if type(fold) is not int or not 0 <= fold <= 4:
        _invalid("fold must be an integer in 0..4")
    base = gd.final_fold_view(records, pools, fold, expected_pools_sha256=expected_pools_sha256)
    groups = {row["id"]: row["group"] for row in records}
    train, test = base["final_train_ids"], base["test_ids"]
    if set(train) & set(test) or set(test) & set(pools["lists"]["D"]):
        _fail("D_in_test", "final view would test D or overlap train/test")
    return _seal({"schema_version": 1, "algorithm": FINAL_VIEW_ALGORITHM, "pools_sha256": expected_pools_sha256,
                  "freeze_sha256": freeze_sha256, "fold": fold, "train": list(train), "test": list(test),
                  "groups": {"train": sorted({groups[i] for i in train}), "test": sorted({groups[i] for i in test})}})


def _check_view_doc(view, cfg):
    _closed(view, {"schema_version", "algorithm", "pools_sha256", "freeze_sha256", "fold", "train", "test",
                   "groups", "sha256"}, "final view", "view_hash_mismatch")
    if view["sha256"] != canonical_hash({k: v for k, v in view.items() if k != "sha256"}):
        _fail("view_hash_mismatch", "final view checksum mismatch")
    if (view["algorithm"] != FINAL_VIEW_ALGORITHM or view["pools_sha256"] != cfg["global"]["pools_sha256"]
            or view["freeze_sha256"] != cfg["global"]["freeze_sha256"] or view["fold"] != cfg["global"]["fold"]
            or type(view["fold"]) is not int):
        _fail("view_hash_mismatch", "final view is not the view of this config")


def _utc(value, name, code):
    if type(value) is not str or not value.endswith("Z"):
        _fail(code, f"{name}: explicit UTC ISO timestamp ending in Z required")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(code, f"{name}: explicit UTC ISO timestamp required")
    if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
        _fail(code, f"{name}: UTC required")


def _global_task(task):
    """Pools task must use the global v1 decision constants (Q4; spec 6.9)."""
    if type(task) is not dict or task.get("type") not in SCALAR_TASKS:
        _fail("recipe_adapter_unsupported", "multilabel global jobs are slice S6 (protocol v2)")
    expected = "fixed_threshold" if task["type"] == "binary" else "argmax"
    if task.get("decision_rule") != expected:
        _fail("recipe_adapter_unsupported",
              f"global v1 {task['type']} tasks require decision_rule={expected} in the sealed pools")


def external_recipe_q2(pools, runtime, *, color):
    """Q2 recipe (decided 19/10/2026) as a v1 recipe; decision per task constants."""
    task = pools["task"]
    if task["type"] == "binary":
        prediction = {"rule": "fixed_threshold", "threshold_policy": "fixed_before_E", "thresholds": [BINARY_THRESHOLD_V1]}
    else:
        prediction = {"rule": "argmax", "threshold_policy": "not_applicable", "thresholds": []}
    return {"model": {"name": "resnet50", "weights": "IMAGENET1K_V2",
                      "weights_sha256": runtime["imagenet_weights_sha256"]},
            "input": {"size": 224, "color": color},
            "preprocessing_sha256": pools["provenance"]["preprocessing_sha256"],
            "optimizer": {"name": "adamw", "parameters": {"lr": 0.0001, "weight_decay": 0.0001}},
            "scheduler": {"name": "cosine_per_epoch", "parameters": {"eta_min": 0.0}},
            "batch_size": 16, "accumulation_steps": 1,
            "loss": {"name": "cross_entropy", "parameters": {}, "learned_on": "D_fit"},
            "epochs": 20, "prediction": prediction}


def _external_eligibility(pool_audit):
    reasons = list(pool_audit["eligibility"]["reasons"]) + ["external_recipe_not_tuned_on_D"]
    return {"contract_eligible": False, "scientific_eligible": False, "execution_verified": False,
            "reasons": reasons,
            "scientific_reasons": ["execution_unverified", "runtime_isolation_unverified",
                                   "chronology_unverified", "external_fixed_recipe"]}


_EXTERNAL_KEYS = {"schema_version", "protocol", "scientific_status", "dataset", "task", "pools_sha256", "recipe",
                  "source", "selection_scope", "runtime", "final_policy", "label", "frozen_at_utc",
                  "eligibility", "sha256"}


def build_external_freeze(records, pools, *, expected_pools_sha256, recipe, source, runtime, final_policy,
                          frozen_at_utc):
    """Unpublished external-freeze document (spec 6.8). Sealing/anchoring is S3."""
    pool_audit = _anchored(records, pools, expected_pools_sha256)
    document = _seal({"schema_version": 1, "protocol": EXTERNAL_FREEZE_PROTOCOL, "scientific_status": "provisional",
                      "dataset": deepcopy(pools["dataset"]), "task": deepcopy(pools["task"]),
                      "pools_sha256": expected_pools_sha256, "recipe": deepcopy(recipe), "source": deepcopy(source),
                      "selection_scope": {"data_used_for_selection": "none", "D_role": "final_training_only"},
                      "runtime": deepcopy(runtime), "final_policy": deepcopy(final_policy), "label": "exploratory",
                      "frozen_at_utc": frozen_at_utc, "eligibility": _external_eligibility(pool_audit)})
    audit_external_freeze(records, pools, document, expected_pools_sha256=expected_pools_sha256)
    return document


def audit_external_freeze(records, pools, document, *, expected_pools_sha256):
    pool_audit = _anchored(records, pools, expected_pools_sha256)
    _closed(document, _EXTERNAL_KEYS, "external freeze", "freeze_not_sealed")
    try:
        gd._checksum(document)
    except ValueError as error:
        _fail("freeze_not_sealed", str(error))
    if (document["schema_version"] != 1 or type(document["schema_version"]) is not int
            or document["protocol"] != EXTERNAL_FREEZE_PROTOCOL or document["scientific_status"] != "provisional"):
        _fail("freeze_not_sealed", "only provisional external freeze v1 supported")
    for key in ("dataset", "task"):
        if canonical_hash(document[key]) != canonical_hash(pools[key]):
            _fail("freeze_not_sealed", f"external freeze {key} differs from pools")
    if document["pools_sha256"] != expected_pools_sha256:
        _fail("pools_anchor_mismatch", "external freeze bound to other pools")
    source = document["source"]
    _closed(source, {"kind", "reference", "derived_from_this_dataset_scores", "notes"}, "source", "freeze_not_sealed")
    if source["derived_from_this_dataset_scores"] is not False:
        _fail("external_recipe_contaminated", "external recipe must not derive from this dataset's scores")
    if source["kind"] not in ("literature", "convention", "other_dataset"):
        _fail("freeze_not_sealed", "unknown external recipe source kind")
    _text(source["reference"], "source.reference")
    _text(source["notes"], "source.notes")
    if document["selection_scope"] != {"data_used_for_selection": "none", "D_role": "final_training_only"}:
        _fail("freeze_not_sealed", "external freeze uses no data for selection; D is final training only")
    if document["label"] != "exploratory":
        _fail("freeze_not_sealed", "external freeze v1 is always exploratory")
    _global_task(pools["task"])
    try:
        gd._runtime(document["runtime"])
        gd._final_policy(document["final_policy"])
        gd._recipe(document["recipe"], pools, document["runtime"])
    except ValueError as error:
        _fail("freeze_not_sealed", str(error))
    _registry(document["recipe"])
    _decision(pools["task"], document["recipe"]["prediction"])
    _utc(document["frozen_at_utc"], "frozen_at_utc", "freeze_not_sealed")
    if canonical_hash(document["eligibility"]) != canonical_hash(_external_eligibility(pool_audit)):
        _fail("freeze_not_sealed", "eligibility must equal derived state (all flags false)")
    return {"structural_valid": True, "execution_verified": False, "scientific_eligible": False,
            "freeze_sha256": document["sha256"], "eligibility": document["eligibility"]}


def _freeze_parts(records, pools, *, freeze, freeze_kind, expected_pools_sha256, expected_freeze_sha256,
                  evidence=None, expected_plan_sha256=None):
    """Audit a freeze and return (recipe, final_policy, runtime)."""
    if freeze_kind not in FREEZE_KINDS:
        _invalid("freeze_kind must be hpo_d or external_fixed")
    _hex64(expected_freeze_sha256, "expected_freeze_sha256")
    if type(freeze) is not dict or freeze.get("sha256") != expected_freeze_sha256:
        _fail("freeze_not_sealed", "freeze document does not match its external anchor")
    if freeze_kind == "external_fixed":
        audit_external_freeze(records, pools, freeze, expected_pools_sha256=expected_pools_sha256)
        return freeze["recipe"], freeze["final_policy"], freeze["runtime"]
    _anchored(records, pools, expected_pools_sha256)
    if evidence is None or expected_plan_sha256 is None:
        _fail("freeze_not_sealed", "hpo_d freeze requires its evidence and the anchored plan hash")
    try:
        gd.audit_recipe_freeze(records, pools, freeze, evidence, expected_pools_sha256=expected_pools_sha256,
                               expected_plan_sha256=expected_plan_sha256)
    except ValueError as error:
        _fail("freeze_not_sealed", str(error))
    hpo_seeds = set(freeze["selection_plan"]["seeds"])
    if hpo_seeds & set(freeze["final_policy"]["seeds"]):
        _fail("seed_reuse_forbidden", "final seeds must be disjoint from HPO seeds")
    return freeze["frozen_recipe"], freeze["final_policy"], freeze["runtime"]


def _check_evaluation_seed(pools):
    seeds = pools["partition_provenance"]["seeds"]
    if seeds["evaluation"] in (seeds["development"], seeds["selection"]):
        _fail("seed_reuse_forbidden", "E folds need a new evaluation seed distinct from D seeds (Q3)")


def _dataset_block(pools, dataset_source):
    _closed(dataset_source, {"adapter", "manifest", "manifest_sha256"}, "dataset_source")
    return {"id": pools["dataset"]["id"], "version": pools["dataset"]["version"],
            "adapter": dataset_source["adapter"], "manifest": dataset_source["manifest"],
            "manifest_sha256": dataset_source["manifest_sha256"], "require_image_hashes": True}


# --------------------------------------------------------------------------
# Config builders (pure) and materialization (filesystem)
# --------------------------------------------------------------------------

def build_hpo_configs(records, pools, *, expected_pools_sha256, dataset_source, plan, expected_plan_sha256,
                      search_space_sha256, hpo_view_sha256):
    """{filename: cfg} for every candidate x seed of an anchored v1 plan."""
    _anchored(records, pools, expected_pools_sha256, global_task=True)
    _hex64(search_space_sha256, "search_space_sha256")
    _hex64(hpo_view_sha256, "hpo_view_sha256")
    try:
        candidates = gd._plan(plan, pools, expected_plan_sha256)
    except ValueError as error:
        _fail("plan_invalid", str(error))
    dataset = _dataset_block(pools, dataset_source)
    configs = {}
    for candidate_id in sorted(candidates):
        candidate = candidates[candidate_id]
        _safe_name(candidate_id, "candidate id")
        recipe = candidate["recipe"]
        for seed in plan["seeds"]:
            cfg = {"schema_version": 1, "phase": "hpo", "scientific_status": "provisional",
                   "protocol_version": HPO_PROTOCOL, "dataset": deepcopy(dataset),
                   "global": {"pools_file": _private_name("pools", dataset["id"], expected_pools_sha256),
                              "pools_sha256": expected_pools_sha256, "view": "hpo",
                              "hpo_view_file": _private_name("hpo-view", dataset["id"], hpo_view_sha256),
                              "hpo_view_sha256": hpo_view_sha256, "plan_sha256": expected_plan_sha256,
                              "search_space_sha256": search_space_sha256, "candidate_id": candidate_id,
                              "candidate_config_sha256": canonical_hash(candidate), "freeze_kind": None,
                              "freeze_file": None, "freeze_sha256": None, "fold": None},
                   "model": _model_block(recipe, pools["task"]), "input": deepcopy(recipe["input"]),
                   "augmentation": {"arm": "none", "ops": []}, "training": recipe_to_training(recipe, seed),
                   "evaluation": _evaluation_block(pools["task"], recipe)}
            validate_global_config(cfg, recipe=recipe)
            configs[f"{candidate_id}-s{seed}.json"] = cfg
    return configs


def build_final_configs(records, pools, *, expected_pools_sha256, dataset_source, freeze, freeze_kind,
                        expected_freeze_sha256, augmentation_presets, seeds=None, evidence=None,
                        expected_plan_sha256=None):
    """{filename: cfg} for arms x 5 folds x seeds of a sealed freeze.

    ``seeds`` defaults to every final seed; a pilot passes an ordered subset
    (Q13: only seed[0], the incomplete matrix is declared by the campaign).
    """
    recipe, final_policy, _ = _freeze_parts(
        records, pools, freeze=freeze, freeze_kind=freeze_kind, expected_pools_sha256=expected_pools_sha256,
        expected_freeze_sha256=expected_freeze_sha256, evidence=evidence, expected_plan_sha256=expected_plan_sha256)
    _anchored(records, pools, expected_pools_sha256, global_task=True)
    _check_evaluation_seed(pools)
    policy_seeds = final_policy["seeds"]
    seeds = list(policy_seeds) if seeds is None else seeds
    if (type(seeds) is not list or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds) or not set(seeds) <= set(policy_seeds)
            or seeds != [s for s in policy_seeds if s in seeds]):
        _invalid("seeds must be a nonempty subset of final_policy.seeds in policy order")
    if type(augmentation_presets) is not dict or set(augmentation_presets) != set(ARMS):
        _invalid("augmentation_presets must define exactly the ten ARMS")
    if canonical_hash(augmentation_presets) != final_policy["augmentation_presets_sha256"]:
        _fail("freeze_not_sealed", "augmentation presets differ from final_policy.augmentation_presets_sha256")
    for arm in ARMS:
        _scan_forbidden(augmentation_presets[arm], f"augmentation_presets.{arm}")
        _augmentation(augmentation_presets[arm], arm)
    dataset = _dataset_block(pools, dataset_source)
    prefix = "freeze" if freeze_kind == "hpo_d" else "external-freeze"
    configs = {}
    for seed in seeds:
        for arm in ARMS:
            for fold in range(5):
                cfg = {"schema_version": 1, "phase": "final", "scientific_status": "provisional",
                       "protocol_version": FINAL_PROTOCOL, "dataset": deepcopy(dataset),
                       "global": {"pools_file": _private_name("pools", dataset["id"], expected_pools_sha256),
                                  "pools_sha256": expected_pools_sha256, "view": "final", "hpo_view_file": None,
                                  "hpo_view_sha256": None, "plan_sha256": None, "search_space_sha256": None,
                                  "candidate_id": None, "candidate_config_sha256": None,
                                  "freeze_kind": freeze_kind,
                                  "freeze_file": _private_name(prefix, dataset["id"], expected_freeze_sha256),
                                  "freeze_sha256": expected_freeze_sha256, "fold": fold},
                       "model": _model_block(recipe, pools["task"]), "input": deepcopy(recipe["input"]),
                       "augmentation": deepcopy(augmentation_presets[arm]),
                       "training": recipe_to_training(recipe, seed),
                       "evaluation": _evaluation_block(pools["task"], recipe)}
                validate_global_config(cfg, recipe=recipe)
                configs[f"{arm}-f{fold}-s{seed}.json"] = cfg
    return configs


def config_dir(dataset_id, phase):
    _safe_name(dataset_id, "dataset id")
    if phase not in ("hpo", "final"):
        _invalid("phase must be hpo or final")
    return f"configs/global/{dataset_id}/{phase}"


def _write_exclusive(target, data):
    """Create ``target`` with ``data`` or accept identical bytes; never overwrite."""
    if target.exists():
        if target.read_bytes() != data:
            _fail("materialization_conflict", f"{target.name} exists with different bytes")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != data:
                _fail("materialization_conflict", f"{target.name} concurrently created with different bytes")
    finally:
        Path(temporary).unlink(missing_ok=True)


def materialize_configs(phase, code_root, configs, *, pools_sha256, plan_sha256=None, freeze_sha256=None):
    """Write ``configs`` under ``configs/global/<ds>/<phase>/`` plus MANIFEST.json.

    Deterministic: regenerating identical inputs produces identical bytes and
    is idempotent; different bytes or stray configs are a conflict. Configs
    carry private file names and hashes, never record IDs or groups.
    """
    if type(configs) is not dict or not configs:
        _invalid("configs must be a nonempty {filename: cfg} mapping")
    _hex64(pools_sha256, "pools_sha256")
    if phase == "hpo":
        _hex64(plan_sha256, "plan_sha256")
        if freeze_sha256 is not None:
            _invalid("hpo manifest has no freeze")
    elif phase == "final":
        _hex64(freeze_sha256, "freeze_sha256")
        if plan_sha256 is not None:
            _invalid("final manifest has no plan")
    else:
        _invalid("phase must be hpo or final")
    datasets = set()
    rendered = {}
    for name in sorted(configs):
        cfg = validate_global_config(configs[name])
        _safe_name(name, "config file name")
        if not name.endswith(".json") or name == "MANIFEST.json":
            _invalid("config file names must be <name>.json and not MANIFEST.json")
        bound = cfg["global"]["plan_sha256"] if phase == "hpo" else cfg["global"]["freeze_sha256"]
        if (cfg["phase"] != phase or cfg["global"]["pools_sha256"] != pools_sha256
                or bound != (plan_sha256 if phase == "hpo" else freeze_sha256)):
            _invalid(f"{name} is not bound to this phase/pools/{'plan' if phase == 'hpo' else 'freeze'}")
        expected_name = (f"{cfg['global']['candidate_id']}-s{cfg['training']['seed']}.json" if phase == "hpo" else
                         f"{cfg['augmentation']['arm']}-f{cfg['global']['fold']}-s{cfg['training']['seed']}.json")
        if name != expected_name:
            _invalid(f"{name} must be named {expected_name}")
        datasets.add((cfg["dataset"]["id"], cfg["dataset"]["version"]))
        rendered[name] = render_config_bytes(cfg)
    if len(datasets) != 1:
        _invalid("all configs must belong to one dataset/version")
    (dataset_id, dataset_version), = datasets
    relative = config_dir(dataset_id, phase)
    target_dir = Path(code_root) / relative
    files = [{"path": f"{relative}/{name}", "sha256": _bytes_sha256(data)} for name, data in rendered.items()]
    manifest = _seal({"schema_version": 1, "protocol": MANIFEST_PROTOCOL, "phase": phase,
                      "dataset": {"id": dataset_id, "version": dataset_version}, "pools_sha256": pools_sha256,
                      "plan_sha256": plan_sha256, "freeze_sha256": freeze_sha256, "files": files})
    manifest_bytes = render_config_bytes(manifest)
    # Check every conflict before writing anything: no partial materialization.
    if target_dir.exists():
        allowed = set(rendered) | {"MANIFEST.json"}
        stray = sorted(p.name for p in target_dir.iterdir() if p.name not in allowed and not p.name.startswith("."))
        if stray:
            _fail("materialization_conflict", f"unexpected files in {relative}: {stray[:3]}")
        for name, data in list(rendered.items()) + [("MANIFEST.json", manifest_bytes)]:
            path = target_dir / name
            if path.exists() and path.read_bytes() != data:
                _fail("materialization_conflict", f"{name} exists with different bytes")
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, data in rendered.items():
        _write_exclusive(target_dir / name, data)
    _write_exclusive(target_dir / "MANIFEST.json", manifest_bytes)
    return manifest


# --------------------------------------------------------------------------
# Queue specs (IDENTITY_FIELDS unchanged)
# --------------------------------------------------------------------------

def _spec_common(cfg, config_path, config_sha256, code_commit):
    _commit(code_commit)
    _hex64(config_sha256, "config_sha256")
    if config_sha256 != _bytes_sha256(render_config_bytes(cfg)):
        _fail("recipe_mismatch", "config_sha256 is not the hash of the materialized config bytes")
    phase = cfg["phase"]
    expected = (f"{config_dir(cfg['dataset']['id'], phase)}/"
                + (f"{cfg['global']['candidate_id']}-s{cfg['training']['seed']}.json" if phase == "hpo" else
                   f"{cfg['augmentation']['arm']}-f{cfg['global']['fold']}-s{cfg['training']['seed']}.json"))
    if config_path != expected:
        _invalid(f"config path must be {expected}")
    return {"protocol_version": cfg["protocol_version"], "dataset": cfg["dataset"]["id"],
            "dataset_version": cfg["dataset"]["version"], "phase": phase, "train_seed": cfg["training"]["seed"],
            "arm": cfg["augmentation"]["arm"], "code_commit": code_commit, "config": config_path,
            "config_sha256": config_sha256, "dependencies": [], "compatibility": deepcopy(DEFAULT_COMPATIBILITY)}


def hpo_job_spec(cfg, *, config_path, config_sha256, code_commit):
    """Queue spec of an HPO job (spec 6.5); returns ``prepare_job`` output."""
    validate_global_config(cfg)
    if cfg["phase"] != "hpo":
        _invalid("hpo_job_spec requires phase=hpo")
    if cfg["augmentation"] != {"arm": "none", "ops": []}:
        _fail("hpo_augmentation_forbidden", "HPO jobs use arm none")
    spec = _spec_common(cfg, config_path, config_sha256, code_commit)
    view = cfg["global"]
    spec.update(split_hash=view["hpo_view_sha256"], recipe_hash=view["candidate_config_sha256"], fold=0,
                **{"global": {"pools_sha256": view["pools_sha256"], "plan_sha256": view["plan_sha256"],
                              "search_space_sha256": view["search_space_sha256"],
                              "candidate_id": view["candidate_id"], "freeze_sha256": None,
                              "campaign_sha256": None, "freeze_kind": None}})
    return prepare_job(spec)


def final_job_spec(cfg, view_doc, *, config_path, config_sha256, code_commit, campaign_sha256):
    """Queue spec of a final job; ``view_doc`` comes from ``derive_final_view_doc``."""
    validate_global_config(cfg)
    if cfg["phase"] != "final":
        _invalid("final_job_spec requires phase=final")
    _hex64(campaign_sha256, "campaign_sha256")
    _check_view_doc(view_doc, cfg)
    spec = _spec_common(cfg, config_path, config_sha256, code_commit)
    view = cfg["global"]
    spec.update(split_hash=view_doc["sha256"], recipe_hash=view["freeze_sha256"], fold=view["fold"],
                **{"global": {"pools_sha256": view["pools_sha256"], "plan_sha256": None,
                              "search_space_sha256": None, "candidate_id": None,
                              "freeze_sha256": view["freeze_sha256"], "campaign_sha256": campaign_sha256,
                              "freeze_kind": view["freeze_kind"]}})
    return prepare_job(spec)
