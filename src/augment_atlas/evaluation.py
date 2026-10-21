"""Opt-in, private scalar classification evaluation without ML dependencies.

The caller supplies trusted config, persisted split and complete manifest records.
This module calculates clinical point estimates and checks artifact consistency;
neither a frozen declaration nor these checks establishes scientific eligibility
or proves that a model actually produced the stored probabilities.
See docs/clinical-evaluation-integration.md.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .clinical_metrics import (
    balanced_accuracy, binary_auroc, binary_average_precision, binary_predictions,
    macro_f1, macro_ovr_auroc, multiclass_predictions, quadratic_weighted_kappa,
)
from .contracts import canonical_hash, recipe_identity

SCORE_KIND = "softmax_probability"
PROBABILITY_SUM_TOLERANCE = 1e-6
PREDICTION_ARTIFACT = "predictions.json"
PRIMARY_METRICS = {"binary": "binary_auroc", "multiclass": "macro_ovr_auroc",
                   "ordinal": "quadratic_weighted_kappa"}


def validate_evaluation_config(spec: dict, num_classes: int) -> dict:
    """Validate an explicit, task-specific opt-in; never inject defaults."""
    if not isinstance(spec, dict):
        raise ValueError("evaluation must be an explicit object")
    task = spec.get("task")
    if task == "multilabel":
        raise ValueError("multilabel evaluation requires a multilabel trainer; scalar trainer rejects it")
    if not isinstance(task, str) or task not in PRIMARY_METRICS:
        raise ValueError("evaluation.task must be binary, multiclass or ordinal")
    common = {"schema_version", "task", "classes", "score_kind", "frozen"}
    specific = {"positive_class", "threshold", "comparison"} if task == "binary" else {"tie_break"}
    if set(spec) != common | specific:
        raise ValueError("evaluation requires exactly its explicit task-specific fields")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("evaluation.schema_version must be 1")
    if spec["frozen"] is not True:
        raise ValueError("evaluation.frozen=true is required as a declaration, not scientific approval")
    if type(spec["score_kind"]) is not str or spec["score_kind"] != SCORE_KIND:
        raise ValueError("evaluation.score_kind must be softmax_probability")
    classes = spec["classes"]
    if (type(num_classes) is not int or num_classes < 2 or not isinstance(classes, list)
            or len(classes) != num_classes):
        raise ValueError("evaluation.classes must have one explicit label per model output index")
    if any(type(label) not in (str, int) or (isinstance(label, str) and not label.strip()) for label in classes):
        raise ValueError("evaluation.classes must contain nonempty strings or integer labels, not bool")
    if len(set(classes)) != len(classes):
        raise ValueError("evaluation.classes must be distinct and ordered")
    if task == "binary":
        positive = spec["positive_class"]
        if num_classes != 2 or type(positive) not in (str, int) or positive not in classes:
            raise ValueError("binary evaluation requires two classes and an explicit positive_class in classes")
        threshold = spec["threshold"]
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError("evaluation.threshold must be finite in [0,1]")
        if type(spec["comparison"]) is not str or spec["comparison"] not in ("ge", "gt"):
            raise ValueError("evaluation.comparison must be ge or gt")
    elif type(spec["tie_break"]) is not str or spec["tie_break"] not in ("first", "error"):
        raise ValueError("evaluation.tie_break must be first or error")
    return spec


def evaluation_partition(cfg: dict) -> str:
    """Derive the final partition only from the existing evaluation scope."""
    scope = cfg["training"].get("evaluation_scope", "test")
    if scope not in ("test", "selection-only"):
        raise ValueError("unsupported evaluation_scope")
    return "selection" if scope == "selection-only" else "test"


def clinical_environment_block(cfg: dict) -> dict:
    """The exact environment.json declaration written by the trainer."""
    spec = cfg["evaluation"]
    return {"evaluation_sha256": canonical_hash(spec), "task": spec["task"],
            "score_kind": spec["score_kind"], "frozen_declaration_only": True,
            "scientifically_eligible": False}


def selected_epoch(history: list) -> tuple[int, float]:
    """Recompute the trainer rule: first epoch with strictly minimal selection loss.

    Only the persisted selection-loss journal is used; no test value is read.
    """
    if not isinstance(history, list) or not history:
        raise ValueError("selection history must be a nonempty epoch journal")
    best_epoch, best_loss = 0, math.inf
    for index, row in enumerate(history, start=1):
        selection = row.get("selection") if isinstance(row, dict) else None
        loss = selection.get("loss") if isinstance(selection, dict) else None
        if (type(row.get("epoch") if isinstance(row, dict) else None) is not int or row["epoch"] != index
                or type(loss) not in (int, float) or not math.isfinite(loss)):
            raise ValueError("selection history must contain consecutive epochs with finite selection loss")
        if loss < best_loss:
            best_epoch, best_loss = index, loss
    return best_epoch, best_loss


def partition_accuracy_bounds(artifact: dict) -> tuple[int, int]:
    """Bounds on argmax(logits)==target implied by stored softmax probabilities.

    Rounded softmax is monotone non-decreasing in the logit, so a strict
    probability maximum implies a strict logit maximum. Probability ties are
    ambiguous and widen the interval instead of guessing the logit order.
    Call only on an artifact already accepted by validate_prediction_artifact.
    """
    lower = upper = 0
    for row in artifact["rows"]:
        scores, target = row["scores"], row["target_index"]
        maximum = max(scores)
        if scores[target] == maximum:
            upper += 1
            lower += scores.count(maximum) == 1
    return lower, upper


def _context(cfg, splits, records, best_epoch, environment):
    from .contracts import validate_config
    from .data import audit_splits

    validate_config(cfg)
    if "evaluation" not in cfg:
        raise ValueError("clinical evaluation requires explicit evaluation configuration")
    if type(best_epoch) is not int or not 1 <= best_epoch <= cfg["training"]["epochs"]:
        raise ValueError("best_epoch must identify a configured completed epoch")
    if not isinstance(records, list) or not records:
        raise ValueError("expected manifest records must be a nonempty list")
    for record in records:
        if (not isinstance(record, dict) or any(not isinstance(record.get(key), str) or not record[key].strip()
                                               for key in ("id", "group"))
                or type(record.get("label")) is not int
                or not 0 <= record["label"] < cfg["model"]["num_classes"]):
            raise ValueError("expected manifest records need valid id/group/scalar class index")
    if not isinstance(splits, dict) or splits.get("sha256") != canonical_hash(
            {key: value for key, value in splits.items() if key != "sha256"}):
        raise ValueError("expected split checksum mismatch")
    expected = {"manifest_sha256": canonical_hash(sorted(records, key=lambda row: row["id"])),
                "dataset_version": cfg["dataset"]["version"],
                "split_seed": cfg["splits"]["split_seed"], "fold": cfg["splits"]["fold"],
                "algorithm": cfg["splits"]["version"]}
    if any(splits.get(key) != value for key, value in expected.items()):
        raise ValueError("expected split/config/manifest identity mismatch")
    audit_splits(records, splits["lists"])
    lookup = {record["id"]: record for record in records}
    expected_groups = {name: sorted({lookup[item]["group"] for item in ids})
                       for name, ids in splits["lists"].items()}
    if splits.get("groups") != expected_groups:
        raise ValueError("expected split group mapping mismatch")
    if cfg["splits"]["version"] == "stratified-group-outer-v1":
        from .scientific_splits import audit_scientific_split
        audit_scientific_split(records, splits)
    scope = cfg["training"].get("evaluation_scope", "test")
    if (not isinstance(environment, dict) or environment.get("config_sha256") != canonical_hash(cfg)
            or environment.get("splits_sha256") != splits["sha256"]
            or environment.get("evaluation_scope") != scope
            or environment.get("clinical_evaluation") != clinical_environment_block(cfg)):
        raise ValueError("environment does not declare this config/split/scope/clinical evaluation")
    partition = evaluation_partition(cfg)
    selected = [lookup[item] for item in splits["lists"][partition]]
    return cfg["evaluation"], partition, selected


def _probabilities(scores, n, width):
    if not isinstance(scores, list) or len(scores) != n:
        raise ValueError("probability rows must match the exact evaluation partition")
    normalized = []
    for row in scores:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("probability columns must match the fixed class order")
        if any(type(value) not in (float, int) or not 0 <= value <= 1 for value in row):
            raise ValueError("probabilities must be finite JSON numbers in [0,1]")
        values = [float(value) for value in row]
        if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=PROBABILITY_SUM_TOLERANCE):
            raise ValueError("softmax probabilities must sum to one within absolute tolerance 1e-6")
        normalized.append(values)
    return normalized


def build_prediction_artifact(cfg: dict, splits: dict, records: list[dict], scores: list[list],
                              *, best_epoch: int, environment: dict) -> tuple[dict, dict]:
    """Build private final rows and results from scores in persisted split order.

    records is the complete expected manifest, not self-declared prediction rows.
    environment is the exact environment.json document of this execution.
    Epoch selection remains outside this module and uses only selection loss.
    """
    spec, partition, selected = _context(cfg, splits, records, best_epoch, environment)
    classes = spec["classes"]
    scores = _probabilities(scores, len(selected), len(classes))
    truth = [classes[record["label"]] for record in selected]
    if spec["task"] == "binary":
        column = classes.index(spec["positive_class"])
        positive_scores = [row[column] for row in scores]
        predictions = binary_predictions(positive_scores, classes=classes, positive_label=spec["positive_class"],
                                         threshold=spec["threshold"], comparison=spec["comparison"])
        metrics = {
            "binary_auroc": binary_auroc(truth, positive_scores, classes=classes, positive_label=spec["positive_class"]),
            "binary_average_precision": binary_average_precision(
                truth, positive_scores, classes=classes, positive_label=spec["positive_class"]),
        }
    else:
        predictions = multiclass_predictions(scores, classes=classes, tie_break=spec["tie_break"])
        if spec["task"] == "ordinal":
            metrics = {"quadratic_weighted_kappa": quadratic_weighted_kappa(truth, predictions, classes=classes)}
        else:
            metrics = {"macro_ovr_auroc": macro_ovr_auroc(truth, scores, classes=classes)}
    metrics["macro_f1"] = macro_f1(truth, predictions, classes=classes)
    metrics["balanced_accuracy"] = balanced_accuracy(truth, predictions, classes=classes)
    identity = {"config_sha256": canonical_hash(cfg), "splits_sha256": splits["sha256"],
                "recipe_sha256": recipe_identity(cfg), "environment_sha256": canonical_hash(environment),
                "evaluation_sha256": canonical_hash(spec), "best_epoch": best_epoch,
                "partition": partition, "evaluation_scope": cfg["training"].get("evaluation_scope", "test"),
                "task": spec["task"], "classes": deepcopy(classes), "score_kind": SCORE_KIND}
    artifact = {"schema_version": 1, "artifact_type": "clinical_predictions", "visibility": "private",
                **identity, "evaluation": deepcopy(spec),
                "dataset": {"id": cfg["dataset"]["id"], "version": cfg["dataset"]["version"]},
                "rows": [{"id": record["id"], "group": record["group"], "target_index": record["label"],
                          "truth": target, "scores": score, "prediction": prediction}
                         for record, target, score, prediction in zip(selected, truth, scores, predictions)]}
    result = {"schema_version": 1, **identity, "samples": len(selected),
              "prediction_artifact": PREDICTION_ARTIFACT, "predictions_content_sha256": canonical_hash(artifact),
              "primary_metric": PRIMARY_METRICS[spec["task"]], "metrics": metrics,
              "selection_rule": "minimum_selection_loss", "frozen_declaration_only": True,
              "scientific_status": "provisional", "scientifically_eligible": False,
              "execution_verified": False}
    return artifact, result


def validate_prediction_artifact(cfg: dict, splits: dict, records: list[dict], artifact: dict,
                                 clinical_result: dict, *, expected_best_epoch: int,
                                 environment: dict) -> dict:
    """Read-only exact comparison against supplied trusted config/split/manifest.

    Rebuilds labels, decisions, coverage and metrics from stored probabilities.
    Caller must independently check file checksums and anchor expected inputs to
    its job identity. This does not prove model execution or patient identity:
    a forger who rewrites probabilities and every derived value consistently is
    not detectable here (execution_verified stays false).
    """
    if not isinstance(artifact, dict) or not isinstance(artifact.get("rows"), list):
        raise ValueError("clinical prediction artifact must contain rows")
    rows = artifact["rows"]
    if any(not isinstance(row, dict) or "scores" not in row for row in rows):
        raise ValueError("clinical prediction row must contain scores")
    expected_artifact, expected_result = build_prediction_artifact(
        cfg, splits, records, [row["scores"] for row in rows], best_epoch=expected_best_epoch,
        environment=environment)
    if canonical_hash(artifact) != canonical_hash(expected_artifact):
        raise ValueError("clinical prediction provenance/rows/decisions mismatch")
    if not isinstance(clinical_result, dict) or canonical_hash(clinical_result) != canonical_hash(expected_result):
        raise ValueError("clinical result metrics/provenance mismatch")
    return {"samples": expected_result["samples"], "partition": expected_result["partition"],
            "evaluation_sha256": expected_result["evaluation_sha256"],
            "predictions_content_sha256": expected_result["predictions_content_sha256"]}
