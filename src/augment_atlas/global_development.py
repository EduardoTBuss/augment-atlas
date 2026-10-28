"""Offline global D/E contracts. No image access, split generation, or training.

Checksums bind declarations; they cannot attest patient identity, chronology,
historical exposure, or the actual files opened by an evidence producer.

Caveat (``global_campaign.SELECTION_BIAS_CAVEAT``): reshuffling folds does not
remove selection bias if patients used to choose hyperparameters fall into a
test set (Bradshaw 2023, DOI 10.1148/ryai.220232); D therefore stays out of
every test.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from .contracts import ARMS, atomic_json, canonical_hash

PROTOCOL = "global-development-holdout-v1"
FREEZE_PROTOCOL = "global-recipe-freeze-v1"
# Evidence collected from the queue by the real HPO producer (slice S2, additive).
QUEUE_COLLECTED_KIND = "queue_collected_v1"
QUEUE_PRODUCER_ID = "augment-atlas-global-hpo-v1"
# Hash of the DECLARATION of the log policy (lacuna L8), never of real logs.
QUEUE_LOGS_DECLARATION = {"docker_log": "docker.log", "per_row": True}
FORCED_DEVELOPMENT_ALGORITHM = "global-dev-eval-v1+forced-d-v1"
POOL_NAMES = ("D", "D_fit", "D_select", "E")
_POOL_KEYS = {"schema_version", "protocol", "scientific_status", "dataset",
              "records_sha256", "source_sha256", "provenance", "grouping", "task",
              "partition_provenance", "lists", "evaluation_folds", "sha256"}


def _pool_keys(document):
    # The extension is inseparable from its algorithm identity. Legacy v1 stays closed.
    if type(document) is dict and ("forced_development" in document
                                  or "forced_development_groups" in document
                                  or (type(document.get("partition_provenance")) is dict
                                      and document["partition_provenance"].get("algorithm")
                                      == FORCED_DEVELOPMENT_ALGORITHM)):
        return _POOL_KEYS | {"forced_development", "forced_development_groups"}
    return _POOL_KEYS


def _object(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{name}: closed schema requires {sorted(keys)}")


def _text(value, name):
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name}: nonempty, trimmed string required")


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: integer >= {minimum} required")


def _hash(value, name):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name}: lowercase SHA256 required")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: finite number required")


def _json(value, name):
    """Reject NaN, non-JSON values and keys, including bool-as-integer tricks."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _number(value, name)
    elif type(value) is list:
        for item in value:
            _json(item, name)
    elif type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json(item, name)
    else:
        raise ValueError(f"{name}: only finite JSON data accepted")


def _checksum(document):
    _json(document, "document")
    _hash(document.get("sha256"), "sha256")
    if document["sha256"] != canonical_hash({k: v for k, v in document.items() if k != "sha256"}):
        raise ValueError("document checksum mismatch")


def _same(left, right):
    return canonical_hash(left) == canonical_hash(right)


def _signed(candidate):
    value = deepcopy(candidate)
    value.pop("sha256", None)
    _json(value, "candidate")
    value["sha256"] = canonical_hash(value)
    return value


def _records(records):
    if type(records) is not list or not records:
        raise ValueError("records: nonempty list required")
    ids, paths = set(), set()
    for row in records:
        _object(row, {"id", "path", "group", "label"}, "record")
        for key in ("id", "path", "group"):
            _text(row[key], f"record.{key}")
        _int(row["label"], "record.label")
        path = PurePosixPath(row["path"])
        if (path.is_absolute() or ".." in path.parts or ":" in row["path"]
                or "\\" in row["path"] or path.as_posix() != row["path"] or not path.parts):
            raise ValueError("record paths must be normalized, relative and confined")
        if row["id"] in ids or row["path"] in paths:
            raise ValueError("duplicate record identity or image path")
        ids.add(row["id"])
        paths.add(row["path"])
    return sorted(records, key=lambda row: row["id"])


def global_manifest_sha256(records, document):
    """Caller pins this identity BEFORE accepting lists or selection scores.

    Includes every metadata field, including partition provenance, plus the
    canonical records. Lists/folds have their separate persisted pools anchor.
    """
    rows = _records(records)
    keys = _pool_keys(document) - {"records_sha256", "lists", "evaluation_folds", "sha256"}
    if not keys <= set(document):
        raise ValueError("manifest metadata missing")
    return canonical_hash({"records": rows, "metadata": {k: document[k] for k in keys}})


def _task(task):
    _object(task, {"type", "classes", "positive_class", "endpoint", "decision_rule",
                   "missing_policy", "requires_patient_identity"}, "task")
    kind = task["type"]
    if kind == "multilabel":
        raise ValueError("multilabel has no supported offline record adapter in v1")
    endpoints = {"binary": "binary_auroc", "ordinal": "quadratic_weighted_kappa",
                 "multiclass": "macro_auroc_ovr"}
    if type(kind) is not str or kind not in endpoints or task["endpoint"] != endpoints[kind]:
        raise ValueError("unsupported task/endpoint")
    classes = task["classes"]
    if type(classes) is not list or len(classes) < 2:
        raise ValueError("task.classes requires predefined classes")
    for label in classes:
        _int(label, "task class")
    if classes != sorted(set(classes)):
        raise ValueError("task.classes must be ordered and unique")
    if kind == "binary":
        _int(task["positive_class"], "positive_class")
        if len(classes) != 2 or task["positive_class"] not in classes:
            raise ValueError("binary task requires two classes and a positive class")
    elif task["positive_class"] is not None:
        raise ValueError("positive_class applies only to binary tasks")
    if task["decision_rule"] not in ("scores_only", "argmax", "fixed_threshold"):
        raise ValueError("explicit decision_rule required")
    if kind == "ordinal" and task["decision_rule"] != "argmax":
        raise ValueError("ordinal QWK requires a predeclared class decision")
    if task["decision_rule"] == "fixed_threshold" and kind != "binary":
        raise ValueError("fixed_threshold is binary-only in v1")
    if task["missing_policy"] != "undefined_no_drop" or type(task["requires_patient_identity"]) is not bool:
        raise ValueError("explicit undefined_no_drop and patient policy required")


def _pool_metadata(document):
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    if document["protocol"] != PROTOCOL or document["scientific_status"] != "provisional":
        raise ValueError("only provisional offline global protocol supported")
    _object(document["dataset"], {"id", "version"}, "dataset")
    for key, value in document["dataset"].items():
        _text(value, f"dataset.{key}")
    for key in ("records_sha256", "source_sha256"):
        _hash(document[key], key)
    _object(document["provenance"], {"preprocessing_sha256", "labels_sha256",
                                     "duplicates_sha256", "exclusions_sha256"}, "provenance")
    for key, value in document["provenance"].items():
        _hash(value, key)
    grouping = document["grouping"]
    _object(grouping, {"namespace", "source", "identity_level", "linkage_status",
                       "evidence_sha256", "prior_exposure"}, "grouping")
    for key in ("namespace", "source"):
        _text(grouping[key], f"grouping.{key}")
    if grouping["identity_level"] not in ("group", "patient"):
        raise ValueError("unknown grouping identity level")
    if grouping["linkage_status"] not in ("unverified", "verified"):
        raise ValueError("unknown linkage status")
    if grouping["evidence_sha256"] is not None:
        _hash(grouping["evidence_sha256"], "linkage evidence")
    if grouping["linkage_status"] == "verified" and grouping["evidence_sha256"] is None:
        raise ValueError("verified linkage requires evidence hash")
    if grouping["identity_level"] == "patient" and grouping["linkage_status"] != "verified":
        raise ValueError("patient identity requires verified linkage evidence")
    if grouping["prior_exposure"] not in ("known", "unknown", "none_documented"):
        raise ValueError("explicit prior exposure required")
    if document["dataset"]["id"].lower() == "inbreast" and grouping["prior_exposure"] != "known":
        raise ValueError("INbreast pilot exposure is known in this protocol version")
    if "forced_development" in document:
        if (type(document["partition_provenance"]) is not dict
                or document["partition_provenance"].get("algorithm") != FORCED_DEVELOPMENT_ALGORITHM):
            raise ValueError("forced_development_invalid: forced groups require the forced-D algorithm")
        forced = document["forced_development"]
        _object(forced, {"sha256", "count", "groups_canonical_sha256"}, "forced_development")
        _hash(forced["sha256"], "forced_development.sha256")
        _hash(forced["groups_canonical_sha256"], "forced_development.groups_canonical_sha256")
        _int(forced["count"], "forced_development.count", minimum=1)
        groups = document["forced_development_groups"]
        validate_forced_development_groups(groups)
        if groups != sorted(groups) or len(groups) != forced["count"]:
            raise ValueError("forced_development_invalid: sorted groups must match the declared count")
        if canonical_hash(sorted(groups)) != forced["groups_canonical_sha256"]:
            raise ValueError("forced_development_hash_mismatch: private groups differ from their canonical SHA256 pin")
    _task(document["task"])
    provenance = document["partition_provenance"]
    legacy_conversion = (type(provenance) is dict
                         and provenance.get("algorithm") == "legacy-proposal-v1-converted")
    keys = {"algorithm", "version", "source", "generator_version", "seeds", "intended_fractions"}
    if legacy_conversion:
        keys.add("converted_from")
    _object(provenance, keys, "partition_provenance")
    if legacy_conversion:
        _object(provenance["converted_from"], {"protocol", "sha256"}, "converted_from")
        if (document["dataset"]["id"] != "inbreast" or provenance["version"] != "1"
                or provenance["converted_from"]["protocol"] != "global-development-proposal-v1"):
            raise ValueError("legacy conversion requires INbreast proposal-v1 provenance")
        _hash(provenance["converted_from"]["sha256"], "legacy file hash")
    for key in ("algorithm", "version", "source", "generator_version"):
        _text(provenance[key], key)
    _object(provenance["seeds"], {"development", "selection", "evaluation"}, "seeds")
    for key, value in provenance["seeds"].items():
        _int(value, key)
    _object(provenance["intended_fractions"], {"D", "D_fit_given_D", "E_test_given_E"}, "fractions")
    for key, value in provenance["intended_fractions"].items():
        if legacy_conversion and value == "not_preregistered_legacy":
            continue
        _number(value, key)
        if not 0 < value < 1:
            raise ValueError("intended fractions must be between zero and one")


def _ids(value, known, name):
    if type(value) is not list or not value or any(type(item) is not str for item in value):
        raise ValueError(f"{name}: nonempty ID list required")
    if value != sorted(set(value)) or not set(value) <= known:
        raise ValueError(f"{name}: IDs must be sorted, unique and known")
    return set(value)


def _disjoint(left, right, lookup, name):
    if left & right:
        raise ValueError(f"{name}: ID leakage")
    if {lookup[item]["group"] for item in left} & {lookup[item]["group"] for item in right}:
        raise ValueError(f"{name}: group leakage")


def _counts(ids, lookup, classes):
    rows = [lookup[item] for item in sorted(ids)]
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], set()).add(row["label"])
    absent = [label for label in classes if not any(row["label"] == label for row in rows)]
    return {"images": len(rows), "groups": len(groups),
            "class_counts": {str(c): sum(row["label"] == c for row in rows) for c in classes},
            "class_group_counts": {str(c): sum(c in labels for labels in groups.values()) for c in classes},
            "mixed_label_groups": sum(len(labels) > 1 for labels in groups.values()),
            "missing_classes": absent, "endpoint_feasible": not absent}


def validate_forced_development_groups(groups):
    """Validate private IDs without echoing any of them in an error."""
    if (type(groups) is not list or not groups
            or any(type(g) is not str or not g or g != g.strip() for g in groups)
            or len(set(groups)) != len(groups)):
        raise ValueError("forced_development_invalid: nonempty list of unique trimmed group IDs required")


def audit_global_pools(records, document, *, expected_manifest_sha256):
    """Return derived diagnostics; structural corruption raises ValueError."""
    rows = _records(records)
    _object(document, _pool_keys(document), "global pools")
    _pool_metadata(document)
    _checksum(document)
    _hash(expected_manifest_sha256, "expected_manifest_sha256")
    if (document["records_sha256"] != canonical_hash(rows)
            or global_manifest_sha256(rows, document) != expected_manifest_sha256):
        raise ValueError("externally anchored manifest identity mismatch")
    lookup = {row["id"]: row for row in rows}
    known = set(lookup)
    classes = document["task"]["classes"]
    if any(row["label"] not in classes for row in rows):
        raise ValueError("record label outside predefined class universe")
    _object(document["lists"], POOL_NAMES, "lists")
    pools = {name: _ids(document["lists"][name], known, name) for name in POOL_NAMES}
    for left, right in (("D", "E"), ("D_fit", "D_select")):
        _disjoint(pools[left], pools[right], lookup, f"{left}/{right}")
    if pools["D"] | pools["E"] != known:
        raise ValueError("D/E must cover every eligible record")
    if pools["D_fit"] | pools["D_select"] != pools["D"]:
        raise ValueError("D_fit/D_select must partition D")
    if "forced_development" in document:
        forced = set(document["forced_development_groups"])
        if not forced <= {row["group"] for row in rows}:
            raise ValueError("forced_development_unknown_group: forced group absent from manifest")
        if not forced <= {lookup[item]["group"] for item in pools["D"]}:
            raise ValueError("forced_development_not_in_d: every forced group must be in D")
    folds = document["evaluation_folds"]
    if type(folds) is not list or len(folds) != 5:
        raise ValueError("exactly five evaluation folds required")
    appearances, previous, diagnostics = Counter(), set(), {}
    for index, entry in enumerate(folds):
        _object(entry, {"fold", "test_ids"}, "evaluation fold")
        _int(entry["fold"], "fold")
        if entry["fold"] != index:
            raise ValueError("fold indices must be ordered exactly 0..4")
        test = _ids(entry["test_ids"], known, f"fold {index}")
        if not test <= pools["E"]:
            raise ValueError("evaluation tests must contain only E IDs")
        _disjoint(test, pools["D"], lookup, "D/test")
        _disjoint(test, previous, lookup, "evaluation folds")
        previous |= test
        appearances.update(test)
        diagnostics[f"E_test_{index}"] = _counts(test, lookup, classes)
    if appearances != Counter({item: 1 for item in pools["E"]}):
        raise ValueError("every E ID must appear in test exactly once")
    diagnostics.update({name: _counts(ids, lookup, classes) for name, ids in pools.items()})
    diagnostics["manifest"] = _counts(known, lookup, classes)
    reasons = [f"missing_classes:{name}" for name, counts in diagnostics.items() if counts["missing_classes"]]
    grouping = document["grouping"]
    if document["task"]["requires_patient_identity"] and grouping["identity_level"] != "patient":
        reasons.append("patient_identity_unproven")
    if grouping["prior_exposure"] != "none_documented":
        reasons.append("prior_exposure_" + grouping["prior_exposure"])
    return {"structural_valid": True, "pools_sha256": document["sha256"],
            "manifest_sha256": expected_manifest_sha256, "diagnostics": diagnostics,
            "eligibility": {"contract_eligible": not reasons, "reasons": reasons},
            "execution_verified": False}


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def _publish(output, document, auditor):
    """Publish complete bytes with an exclusive hard link; never os.replace target."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = _read(target)
        auditor(existing)
        if canonical_hash(existing) != canonical_hash(document):
            raise ValueError("no-clobber: conflicting persisted identity")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        atomic_json(temporary, document)
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _read(target)
            auditor(existing)
            if canonical_hash(existing) != canonical_hash(document):
                raise ValueError("no-clobber: concurrent conflicting identity")
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_name(temporary.name + ".tmp").unlink(missing_ok=True)


def seal_global_pools(records, candidate, *, expected_manifest_sha256, output):
    document = _signed(candidate)
    audit = lambda value: audit_global_pools(records, value, expected_manifest_sha256=expected_manifest_sha256)
    audit(document)
    _publish(output, document, audit)
    return document


def _anchored_pools(records, pools, expected_pools_sha256):
    _hash(expected_pools_sha256, "expected_pools_sha256")
    if type(pools) is not dict:
        raise ValueError("pools must be an object")
    if pools.get("sha256") != expected_pools_sha256:
        raise ValueError("externally anchored pools identity mismatch")
    # The trusted pools hash binds metadata AND the original manifest identity.
    return audit_global_pools(records, pools, expected_manifest_sha256=global_manifest_sha256(records, pools))


def final_fold_view(records, pools, fold, *, expected_pools_sha256):
    """Derive identities only. There is deliberately no E selection partition."""
    _anchored_pools(records, pools, expected_pools_sha256)
    _int(fold, "fold")
    if fold >= 5:
        raise ValueError("fold must be in 0..4")
    test = pools["evaluation_folds"][fold]["test_ids"]
    train = sorted(set(pools["lists"]["D"]) | (set(pools["lists"]["E"]) - set(test)))
    return {"pools_sha256": expected_pools_sha256, "fold": fold,
            "final_train_ids": train, "test_ids": list(test)}


def _producer(value):
    _object(value, {"id", "code_sha256", "environment_sha256", "manifest_sha256", "logs_sha256"}, "producer")
    _text(value["id"], "producer.id")
    for key in set(value) - {"id"}:
        _hash(value[key], key)


def _runtime(value):
    _object(value, {"commit", "environment_sha256", "imagenet_weights_sha256"}, "runtime")
    if type(value["commit"]) is not str or not re.fullmatch(r"[0-9a-f]{40}", value["commit"]):
        raise ValueError("full commit identity required")
    for key in ("environment_sha256", "imagenet_weights_sha256"):
        _hash(value[key], key)


def _seed_list(value, name, count=None):
    if type(value) is not list or not value or (count is not None and len(value) != count):
        raise ValueError(f"{name}: explicit seed list required")
    for item in value:
        _int(item, name)
    if len(set(value)) != len(value):
        raise ValueError(f"{name}: duplicate seed")


def _final_policy(policy):
    fixed = {"initialization": "fresh_imagenet_each_arm_fold_seed", "warm_start": False,
             "training_pool": "D_union_E_minus_test", "folds": list(range(5)),
             "arms": list(ARMS), "scheduler_duration": "fixed_epochs_recompute_steps_from_final_pool",
             "step_count": {"unit": "optimizer_update", "drop_last": False, "accumulation_tail": "step",
                            "per_epoch": "ceil(ceil(train_images/batch_size)/accumulation_steps)"},
             "early_stopping": False, "checkpoint_selection": "last_epoch", "selection_on_E": False}
    _object(policy, set(fixed) | {"seeds", "augmentation_presets_sha256"}, "final_policy")
    # Canonical comparison preserves bool/int type distinctions, including folds.
    if canonical_hash({k: policy[k] for k in fixed}) != canonical_hash(fixed):
        raise ValueError("fixed final policy violated")
    _seed_list(policy["seeds"], "final seeds", 3)
    _hash(policy["augmentation_presets_sha256"], "augmentation presets")


def _recipe(recipe, pools, runtime):
    _object(recipe, {"model", "input", "preprocessing_sha256", "optimizer", "scheduler", "batch_size",
                     "accumulation_steps", "loss", "epochs", "prediction"}, "recipe")
    _object(recipe["model"], {"name", "weights", "weights_sha256"}, "model")
    if recipe["model"] != {"name": "resnet50", "weights": "IMAGENET1K_V2",
                            "weights_sha256": runtime["imagenet_weights_sha256"]}:
        raise ValueError("recipe requires anchored ResNet50 ImageNet weights")
    _object(recipe["input"], {"size", "color"}, "input")
    _int(recipe["input"]["size"], "input size", 32)
    if recipe["input"]["color"] not in ("rgb", "grayscale_to_rgb"):
        raise ValueError("explicit input color required")
    if recipe["preprocessing_sha256"] != pools["provenance"]["preprocessing_sha256"]:
        raise ValueError("recipe preprocessing identity mismatch")
    for key in ("batch_size", "accumulation_steps", "epochs"):
        _int(recipe[key], key, 1)
    for key in ("optimizer", "scheduler"):
        _object(recipe[key], {"name", "parameters"}, key)
        _text(recipe[key]["name"], key)
        if type(recipe[key]["parameters"]) is not dict:
            raise ValueError(f"{key}.parameters must be an explicit object")
        _json(recipe[key]["parameters"], key)
    _object(recipe["loss"], {"name", "parameters", "learned_on"}, "loss")
    _text(recipe["loss"]["name"], "loss.name")
    if type(recipe["loss"]["parameters"]) is not dict or recipe["loss"]["learned_on"] != "D_fit":
        raise ValueError("loss parameters must be fixed and learned only on D_fit")
    _json(recipe["loss"]["parameters"], "loss parameters")
    prediction = recipe["prediction"]
    _object(prediction, {"rule", "threshold_policy", "thresholds"}, "prediction")
    if prediction["rule"] != pools["task"]["decision_rule"]:
        raise ValueError("prediction decision rule mismatch")
    if prediction["rule"] == "fixed_threshold":
        if prediction["threshold_policy"] != "fixed_before_E" or type(prediction["thresholds"]) is not list or len(prediction["thresholds"]) != 1:
            raise ValueError("complete fixed binary threshold required")
        _number(prediction["thresholds"][0], "threshold")
        if not 0 <= prediction["thresholds"][0] <= 1:
            raise ValueError("threshold outside [0,1]")
    elif prediction["threshold_policy"] != "not_applicable" or prediction["thresholds"] != []:
        raise ValueError("score/argmax prediction requires not_applicable thresholds")


def _plan(plan, pools, expected_plan_sha256):
    _object(plan, {"schema_version", "dataset", "pools_sha256", "task", "endpoint", "direction", "candidates",
                   "seeds", "ranking", "failure_policy", "epoch_policy", "threshold_policy", "producer", "runtime",
                   "final_policy"}, "selection_plan")
    _hash(expected_plan_sha256, "expected_plan_sha256")
    if canonical_hash(plan) != expected_plan_sha256:
        raise ValueError("externally anchored selection plan mismatch")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("plan schema_version must be integer 1")
    for key in ("dataset", "task"):
        if not _same(plan[key], pools[key]):
            raise ValueError(f"plan {key} mismatch")
    if plan["pools_sha256"] != pools["sha256"] or plan["endpoint"] != pools["task"]["endpoint"]:
        raise ValueError("plan pools/endpoint mismatch")
    if plan["direction"] != "maximize" or plan["ranking"] != "mean_then_candidate_id":
        raise ValueError("explicit supported deterministic ranking required")
    if plan["failure_policy"] not in ("require_all_defined", "exclude_candidate_on_any_nondefined"):
        raise ValueError("explicit failure policy required")
    if plan["epoch_policy"] != "fixed_in_candidate_before_scores" or plan["threshold_policy"] != "fixed_in_candidate_before_scores":
        raise ValueError("v1 requires epochs/thresholds fixed in candidates before scores")
    _seed_list(plan["seeds"], "selection seeds")
    _producer(plan["producer"])
    _runtime(plan["runtime"])
    if plan["producer"]["environment_sha256"] != plan["runtime"]["environment_sha256"]:
        raise ValueError("producer environment mismatch")
    _final_policy(plan["final_policy"])
    if type(plan["candidates"]) is not list or not plan["candidates"]:
        raise ValueError("complete candidates required")
    candidates = {}
    for candidate in plan["candidates"]:
        _object(candidate, {"id", "parameters", "recipe", "augmentation"}, "candidate")
        _text(candidate["id"], "candidate id")
        if candidate["id"] in candidates:
            raise ValueError("duplicate candidate")
        if type(candidate["parameters"]) is not dict:
            raise ValueError("candidate parameters must be explicit")
        _json(candidate["parameters"], "candidate parameters")
        _object(candidate["augmentation"], {"arm", "ops"}, "candidate augmentation")
        if candidate["augmentation"] != {"arm": "none", "ops": []}:
            raise ValueError("every selection candidate must use none augmentation with empty ops")
        _recipe(candidate["recipe"], pools, plan["runtime"])
        candidates[candidate["id"]] = candidate
    return candidates


def _scientific_reasons(evidence):
    """Reasons evidence cannot make a freeze scientifically eligible (always false)."""
    if evidence["kind"] == QUEUE_COLLECTED_KIND:
        return ["execution_unverified", "runtime_isolation_unverified", "chronology_unverified",
                QUEUE_COLLECTED_KIND]
    return ["no_integrated_evidence_producer", evidence["kind"]]


def _evidence(evidence, plan, candidates, pools):
    _object(evidence, {"schema_version", "kind", "producer", "dataset", "pools_sha256", "plan_sha256",
                       "task", "rows", "sha256"}, "selection_evidence")
    _checksum(evidence)
    if type(evidence["schema_version"]) is not int or evidence["schema_version"] != 1:
        raise ValueError("evidence schema_version must be integer 1")
    if evidence["kind"] not in ("synthetic_fixture", "external_unverified", QUEUE_COLLECTED_KIND):
        raise ValueError("no integrated real evidence producer exists in v1")
    for key in ("producer", "dataset", "task", "pools_sha256"):
        if not _same(evidence[key], plan[key]):
            raise ValueError(f"evidence {key} mismatch")
    if evidence["kind"] == QUEUE_COLLECTED_KIND and (
            plan["producer"]["id"] != QUEUE_PRODUCER_ID
            or plan["producer"]["logs_sha256"] != canonical_hash(QUEUE_LOGS_DECLARATION)):
        raise ValueError("queue_collected_v1 evidence requires the real HPO producer id and log-policy declaration")
    if evidence["plan_sha256"] != canonical_hash(plan):
        raise ValueError("evidence plan identity mismatch")
    if type(evidence["rows"]) is not list:
        raise ValueError("evidence rows must be a full ledger")
    expected = {(key, seed) for key in candidates for seed in plan["seeds"]}
    observed, scores, nondefined = set(), {key: {} for key in candidates}, []
    for row in evidence["rows"]:
        _object(row, {"candidate_id", "seed", "config_sha256", "train_ids", "select_ids", "epoch_ids",
                      "threshold_ids", "artifact_sha256", "log_sha256", "status", "metric", "reason"}, "ledger row")
        _text(row["candidate_id"], "ledger candidate")
        _int(row["seed"], "ledger seed")
        pair = (row["candidate_id"], row["seed"])
        if pair not in expected or pair in observed:
            raise ValueError("ledger has duplicate or unknown candidate/seed")
        observed.add(pair)
        candidate = candidates[pair[0]]
        if row["config_sha256"] != canonical_hash(candidate):
            raise ValueError("ledger candidate configuration mismatch")
        for key, pool in (("train_ids", "D_fit"), ("select_ids", "D_select"),
                          ("epoch_ids", "D_select"), ("threshold_ids", "D_select")):
            if row[key] != pools["lists"][pool]:
                raise ValueError(f"ledger {key} must use exactly {pool}, never E")
        _hash(row["log_sha256"], "ledger log reference")
        metric = row["metric"]
        _object(metric, {"endpoint", "state", "value", "reason"}, "ledger metric")
        if metric["endpoint"] != plan["endpoint"]:
            raise ValueError("ledger endpoint mismatch")
        if row["status"] == "completed" and metric["state"] == "defined":
            _number(metric["value"], "ledger score")
            lower = -1 if plan["endpoint"] == "quadratic_weighted_kappa" else 0
            if not lower <= metric["value"] <= 1 or metric["reason"] is not None or row["reason"] is not None:
                raise ValueError("defined score range/reason mismatch")
            _hash(row["artifact_sha256"], "result artifact")
            scores[pair[0]][pair[1]] = metric["value"]
        else:
            if row["status"] not in ("completed", "failed", "timeout", "missing"):
                raise ValueError("unknown ledger status")
            required_state = "undefined" if row["status"] == "completed" else "unavailable"
            if metric["state"] != required_state or metric["value"] is not None:
                raise ValueError("nondefined score must be explicit null, never zero")
            _text(metric["reason"], "nondefined metric reason")
            _text(row["reason"], "nondefined trial reason")
            if row["artifact_sha256"] is not None:
                _hash(row["artifact_sha256"], "nondefined artifact")
            nondefined.append({"candidate_id": pair[0], "seed": pair[1], "status": row["status"],
                               "reason": metric["reason"]})
    if observed != expected:
        raise ValueError("incomplete candidate x seed ledger; missing trials must be explicit rows")
    # A ledger is a candidate x seed mapping, not an ordered numerical stream.
    # Canonical seed order plus accurate summation prevents row permutations
    # from changing a near-tie and therefore the frozen recipe.
    means = {key: math.fsum(values[seed] for seed in sorted(values)) / len(values)
             for key, values in sorted(scores.items()) if len(values) == len(plan["seeds"])}
    if not means:
        raise ValueError("no fully defined candidate available for deterministic selection")
    winner = sorted(means, key=lambda key: (-means[key], key))[0]
    return winner, means, sorted(nondefined, key=lambda row: (row["candidate_id"], row["seed"]))


def audit_recipe_freeze(records, pools, freeze, evidence, *, expected_pools_sha256, expected_plan_sha256):
    """Audit declared evidence, never attest real HPO execution or image isolation."""
    pool_audit = _anchored_pools(records, pools, expected_pools_sha256)
    _object(freeze, {"schema_version", "protocol", "scientific_status", "dataset", "task", "pools_sha256",
                     "selection_plan", "selection_plan_sha256", "selection_evidence_sha256", "selected_candidate_id",
                     "selected_parameters", "frozen_recipe", "final_policy", "runtime", "selection_scope",
                     "frozen_at_utc", "eligibility", "sha256"}, "recipe freeze")
    _checksum(freeze)
    if type(freeze["schema_version"]) is not int or freeze["schema_version"] != 1 or freeze["protocol"] != FREEZE_PROTOCOL or freeze["scientific_status"] != "provisional":
        raise ValueError("only provisional offline recipe freeze v1 supported")
    for key in ("dataset", "task"):
        if not _same(freeze[key], pools[key]):
            raise ValueError(f"freeze {key} mismatch")
    if freeze["pools_sha256"] != expected_pools_sha256 or freeze["selection_plan_sha256"] != expected_plan_sha256:
        raise ValueError("freeze external anchor mismatch")
    plan = freeze["selection_plan"]
    candidates = _plan(plan, pools, expected_plan_sha256)
    if freeze["selection_evidence_sha256"] != evidence.get("sha256"):
        raise ValueError("freeze evidence identity mismatch")
    winner, means, nondefined = _evidence(evidence, plan, candidates, pools)
    if freeze["selected_candidate_id"] != winner:
        raise ValueError("selected candidate violates predeclared deterministic ranking")
    if (not _same(freeze["selected_parameters"], candidates[winner]["parameters"])
            or not _same(freeze["frozen_recipe"], candidates[winner]["recipe"])):
        raise ValueError("frozen recipe/parameters differ from selected candidate")
    if not _same(freeze["runtime"], plan["runtime"]) or not _same(freeze["final_policy"], plan["final_policy"]):
        raise ValueError("frozen runtime/final policy differs from anchored plan")
    expected_scope = {"train_ids": pools["lists"]["D_fit"], "select_ids": pools["lists"]["D_select"],
                      "epoch_ids": pools["lists"]["D_select"], "threshold_ids": pools["lists"]["D_select"],
                      "augmentation": {"arm": "none", "ops": []}, "producer": plan["producer"]}
    if not _same(freeze["selection_scope"], expected_scope):
        raise ValueError("selection_scope must bind exact D_fit/D_select lists and producer references")
    _text(freeze["frozen_at_utc"], "frozen_at_utc")
    try:
        timestamp = datetime.fromisoformat(freeze["frozen_at_utc"].replace("Z", "+00:00"))
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError("not UTC")
    except (ValueError, TypeError):
        raise ValueError("frozen_at_utc must be an explicit UTC ISO timestamp") from None
    reasons = list(pool_audit["eligibility"]["reasons"])
    if nondefined and plan["failure_policy"] == "require_all_defined":
        reasons.append("ledger_contains_nondefined_trials")
    eligibility = {"contract_eligible": not reasons, "scientific_eligible": False,
                   "reasons": reasons, "scientific_reasons": _scientific_reasons(evidence)}
    if not _same(freeze["eligibility"], eligibility):
        raise ValueError("eligibility must equal derived audit state")
    recipe = freeze["frozen_recipe"]
    final_steps = []
    for fold in pools["evaluation_folds"]:
        train_images = len(records) - len(fold["test_ids"])
        batches = (train_images + recipe["batch_size"] - 1) // recipe["batch_size"]
        per_epoch = (batches + recipe["accumulation_steps"] - 1) // recipe["accumulation_steps"]
        final_steps.append({"fold": fold["fold"], "train_images": train_images,
                            "optimizer_updates_per_epoch": per_epoch,
                            "total_optimizer_updates": per_epoch * recipe["epochs"]})
    return {"structural_valid": True, "execution_verified": False, "eligibility": eligibility,
            "selected_candidate_id": winner, "candidate_means": means, "nondefined_trials": nondefined,
            "pool_diagnostics": pool_audit["diagnostics"], "final_steps": final_steps,
            "freeze_sha256": freeze["sha256"]}


def seal_recipe_freeze(records, pools, candidate, evidence, *, expected_pools_sha256, expected_plan_sha256, output):
    document = deepcopy(candidate)
    pool_audit = _anchored_pools(records, pools, expected_pools_sha256)
    plan = document["selection_plan"]
    candidates = _plan(plan, pools, expected_plan_sha256)
    _, _, nondefined = _evidence(evidence, plan, candidates, pools)
    reasons = list(pool_audit["eligibility"]["reasons"])
    if nondefined and plan["failure_policy"] == "require_all_defined":
        reasons.append("ledger_contains_nondefined_trials")
    document["eligibility"] = {"contract_eligible": not reasons, "scientific_eligible": False,
                               "reasons": reasons, "scientific_reasons": _scientific_reasons(evidence)}
    document = _signed(document)
    audit = lambda value: audit_recipe_freeze(records, pools, value, evidence,
                                             expected_pools_sha256=expected_pools_sha256,
                                             expected_plan_sha256=expected_plan_sha256)
    audit(document)
    _publish(output, document, audit)
    return document
