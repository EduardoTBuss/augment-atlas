"""Global D/E prediction artifact (slice S1, part A): build and validate the
private ``predictions.json`` of phase ``hpo`` (scope D_select) and ``final``
(scope E_f) for scalar binary/ordinal/multiclass tasks. Offline, stdlib only.

Fixed protocol rules (decided by Eduardo, not negotiable in this version):

* D is never a test set: a final artifact containing any D identity is
  ``D_in_test``. HPO only sees D_fit/D_select: an HPO artifact or hpo-view
  containing any E identity is ``hpo_view_contains_E``.
* A final artifact covers its E_f exactly once (``E_coverage_violation``); an
  HPO artifact covers D_select exactly once (``selection_coverage_violation``).
* Decisions are the sealed global v1 constants (Q4): binary fixed threshold
  0.5 with ``ge``; ordinal/multiclass argmax with ``tie_break="first"``.
  Metrics come from ``clinical_metrics``; an undefined metric is JSON null with
  a ``metric_undefined:<reason>`` reason, rows are never dropped.
* The host validates by RECOMPUTATION: truth comes from trusted records
  (ID -> group -> label) anchored by the pools hash, the evaluated partition
  from the re-derived/checked view; nothing declared by the artifact is
  trusted except its probability scores.
* Caveat (Bradshaw 2023, DOI 10.1148/ryai.220232): reshuffling folds does not
  remove selection bias if patients used to choose hyperparameters fall into
  a test set; that is why D stays out of every test.
* ``execution_verified`` and ``scientific_eligible`` are always false: a forger
  who rewrites scores and every derived value consistently is not detectable
  here, and model execution is not proven.

Multilabel is slice S6 (protocol v2) and is rejected. Errors are
``ValueError("<code>: <message>")`` with the codes of section 6.11 of
``docs/global-next-integration.md``.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

from . import global_jobs as gj
from .clinical_metrics import (
    _binary_metric, _macro, balanced_accuracy, binary_auroc, binary_average_precision,
    binary_predictions, macro_f1, macro_ovr_auroc, multiclass_predictions, quadratic_weighted_kappa,
)
from .contracts import canonical_hash
from .evaluation import PROBABILITY_SUM_TOLERANCE

ARTIFACT_TYPE = "global_predictions"
ARTIFACT_NAME = "predictions.json"
HPO_VIEW_PROTOCOL = "global-hpo-view-v1"
SCORE_KIND = "softmax_probability"
PARTITIONS = {"hpo": "D_select", "final": "E_test"}
SECONDARY_METRICS = ("macro_f1", "balanced_accuracy", "average_precision")

_ARTIFACT_KEYS = {"schema_version", "artifact_type", "visibility", "phase", "dataset", "task", "pools_sha256",
                  "view_sha256", "plan_sha256", "candidate_id", "freeze_sha256", "fold", "arm", "train_seed",
                  "config_sha256", "checkpoint_selection", "epoch", "epochs", "partition", "score_kind",
                  "decision", "rows", "metrics", "content_sha256"}
_IDENTITY_KEYS = tuple(sorted(_ARTIFACT_KEYS - {"rows", "metrics", "content_sha256", "epoch",
                                                "checkpoint_selection"}))
_ROW_KEYS = {"id", "group", "target", "scores", "prediction"}
_HPO_VIEW_KEYS = {"schema_version", "protocol", "scientific_status", "dataset", "task", "pools_sha256",
                  "records", "lists", "view_files_sha256", "sha256"}


def _fail(code, message):
    raise ValueError(f"{code}: {message}")


def _scalar_only(kind):
    if kind == "multilabel":
        _fail("recipe_adapter_unsupported", "multilabel global evaluation is slice S6 (protocol v2)")
    if kind not in gj.SCALAR_TASKS:
        _fail("predictions_invalid", "task must be binary, ordinal or multiclass")


def _content_hash(artifact):
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    try:
        return canonical_hash(body)
    except (TypeError, ValueError) as error:  # allow_nan=False refuses NaN/inf before hashing
        _fail("prediction_scores_invalid", f"artifact is not finite canonical JSON ({error})")


def config_sha256(cfg):
    """Hash of the materialized config bytes (the job spec's ``config_sha256``)."""
    return gj._bytes_sha256(gj.render_config_bytes(cfg))


# --------------------------------------------------------------------------
# Views: which identities the artifact must cover
# --------------------------------------------------------------------------

def check_hpo_view(view, cfg, records, pools):
    """Check an hpo-view (spec 6.3) against trusted records/pools; return D_select.

    The E check runs on the view's own content before its binding to the
    config, so a re-signed view that smuggles an E identity is reported as
    ``hpo_view_contains_E`` and not merely as a hash mismatch.
    """
    if type(view) is not dict or set(view) != _HPO_VIEW_KEYS:
        _fail("view_hash_mismatch", f"hpo view: closed schema requires exactly {sorted(_HPO_VIEW_KEYS)}")
    try:
        body_hash = canonical_hash({key: value for key, value in view.items() if key != "sha256"})
    except (TypeError, ValueError) as error:
        _fail("view_hash_mismatch", f"hpo view is not canonical JSON ({error})")
    if view["sha256"] != body_hash:
        _fail("view_hash_mismatch", "hpo view checksum mismatch")
    e_ids = set(pools["lists"]["E"])
    lookup = {row["id"]: row for row in records}
    e_groups = {lookup[item]["group"] for item in e_ids}
    view_records = view["records"] if type(view["records"]) is list else []
    lists = view["lists"] if type(view["lists"]) is dict else {}
    declared_ids = {row.get("id") for row in view_records if type(row) is dict}
    declared_groups = {row.get("group") for row in view_records if type(row) is dict}
    for name in ("train", "selection"):
        if type(lists.get(name)) is list:
            declared_ids |= {item for item in lists[name] if type(item) is str}
    if declared_ids & e_ids or declared_groups & e_groups:
        _fail("hpo_view_contains_E", "hpo view declares an E identity or group")
    if (view["sha256"] != cfg["global"]["hpo_view_sha256"] or view["pools_sha256"] != cfg["global"]["pools_sha256"]
            or view["schema_version"] != 1 or type(view["schema_version"]) is not int
            or view["protocol"] != HPO_VIEW_PROTOCOL or view["scientific_status"] != "provisional"):
        _fail("view_hash_mismatch", "hpo view is not the view of this config")
    if (canonical_hash(view["dataset"]) != canonical_hash(pools["dataset"])
            or canonical_hash(view["task"]) != canonical_hash(pools["task"])):
        _fail("view_hash_mismatch", "hpo view dataset/task differ from pools")
    d_ids = set(pools["lists"]["D"])
    expected_records = sorted((deepcopy(lookup[item]) for item in d_ids), key=lambda row: row["id"])
    if canonical_hash(view["records"]) != canonical_hash(expected_records):
        _fail("records_identity_mismatch", "hpo view records are not exactly the trusted D records")
    if canonical_hash(view["lists"]) != canonical_hash({"train": pools["lists"]["D_fit"],
                                                        "selection": pools["lists"]["D_select"]}):
        _fail("view_hash_mismatch", "hpo view lists must be exactly D_fit/D_select")
    gj._hex64(view["view_files_sha256"], "hpo view view_files_sha256")
    return list(pools["lists"]["D_select"])


def check_final_view(view, cfg, records, pools):
    """Check a final view against its re-derivation; return E_f (sorted IDs)."""
    gj._check_view_doc(view, cfg)
    if type(view["test"]) is not list or set(view["test"]) & set(pools["lists"]["D"]):
        _fail("D_in_test", "final view tests a D identity")
    expected = gj.derive_final_view_doc(records, pools, cfg["global"]["fold"],
                                        expected_pools_sha256=cfg["global"]["pools_sha256"],
                                        freeze_sha256=cfg["global"]["freeze_sha256"])
    if canonical_hash(view) != canonical_hash(expected):
        _fail("view_hash_mismatch", "final view differs from derive_final_view_doc(records, pools, fold)")
    return list(expected["test"])


def _context(cfg, view, records, pools):
    """Trusted inputs -> (evaluation spec, partition IDs, record lookup)."""
    if type(cfg) is dict and type(cfg.get("evaluation")) is dict:
        _scalar_only(cfg["evaluation"].get("task"))
    gj.validate_global_config(cfg)
    gj._anchored(records, pools, cfg["global"]["pools_sha256"], global_task=True)
    spec = cfg["evaluation"]
    task = pools["task"]
    if (task["type"] != spec["task"] or task["classes"] != spec["classes"]
            or task["positive_class"] != spec["positive_class"] or task["endpoint"] != spec["endpoint"]
            or cfg["dataset"]["id"] != pools["dataset"]["id"]
            or cfg["dataset"]["version"] != pools["dataset"]["version"]):
        _fail("global_config_invalid", "config dataset/evaluation task differ from the sealed pools")
    checker = check_hpo_view if cfg["phase"] == "hpo" else check_final_view
    partition = checker(view, cfg, records, pools)
    return spec, partition, {row["id"]: row for row in records}


# --------------------------------------------------------------------------
# Scores, decisions and metrics (all from clinical_metrics)
# --------------------------------------------------------------------------

def _probabilities(scores, n, width, *, strict):
    if type(scores) is not list or len(scores) != n:
        _fail("prediction_scores_invalid", "one probability row per evaluated identity is required")
    rows = []
    for row in scores:
        if type(row) is not list or len(row) != width:
            _fail("prediction_scores_invalid", f"probability rows must have exactly {width} columns")
        allowed = (float,) if strict else (float, int)
        if any(type(value) not in allowed or not math.isfinite(value) or not 0 <= value <= 1 for value in row):
            _fail("prediction_scores_invalid", "probabilities must be finite JSON floats in [0,1]")
        values = [float(value) for value in row]
        if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=PROBABILITY_SUM_TOLERANCE):
            _fail("prediction_scores_invalid", "softmax probabilities must sum to one within 1e-6")
        rows.append(values)
    return rows


def _metric(result, name=None):
    reason = result["reason"]
    return {"metric": name or result["metric"], "value": result["value"],
            "reason": None if reason is None else f"metric_undefined:{reason}",
            "coverage": deepcopy(result["coverage"]), "per_class": deepcopy(result.get("per_class"))}


def _macro_ovr_average_precision(truth, scores, classes):
    """Macro one-vs-rest AP composed from the clinical_metrics AP primitive."""
    per_class = [{"label": label, **_binary_metric([int(v == label) for v in truth],
                                                   [row[j] for row in scores], "binary_average_precision")}
                 for j, label in enumerate(classes)]
    return _macro("macro_ovr_average_precision", per_class, list(classes), len(truth))


def _decide_and_measure(spec, truth, scores):
    classes, decision = spec["classes"], spec["decision"]
    if spec["task"] == "binary":
        positive = spec["positive_class"]
        column = classes.index(positive)
        positive_scores = [row[column] for row in scores]
        predictions = binary_predictions(positive_scores, classes=classes, positive_label=positive,
                                         threshold=decision["threshold"], comparison=decision["comparison"])
        primary = binary_auroc(truth, positive_scores, classes=classes, positive_label=positive)
        average_precision = binary_average_precision(truth, positive_scores, classes=classes,
                                                     positive_label=positive)
    else:
        predictions = multiclass_predictions(scores, classes=classes, tie_break=decision["tie_break"])
        primary = (quadratic_weighted_kappa(truth, predictions, classes=classes) if spec["task"] == "ordinal"
                   else macro_ovr_auroc(truth, scores, classes=classes))
        average_precision = _macro_ovr_average_precision(truth, scores, classes)
    metrics = {"primary": _metric(primary, spec["endpoint"]),
               "secondary": {"macro_f1": _metric(macro_f1(truth, predictions, classes=classes)),
                             "balanced_accuracy": _metric(balanced_accuracy(truth, predictions, classes=classes)),
                             "average_precision": _metric(average_precision)}}
    return predictions, metrics


def _header(cfg, view, pools, spec):
    """Identity fields an artifact of this config/view must carry."""
    hpo = cfg["phase"] == "hpo"
    bound = cfg["global"]
    return {"schema_version": 1, "artifact_type": ARTIFACT_TYPE, "visibility": "private",
            "phase": cfg["phase"], "dataset": {"id": cfg["dataset"]["id"], "version": cfg["dataset"]["version"]},
            "task": deepcopy(pools["task"]), "pools_sha256": bound["pools_sha256"],
            "view_sha256": view["sha256"],
            "plan_sha256": bound["plan_sha256"] if hpo else None,
            "candidate_id": bound["candidate_id"] if hpo else None,
            "freeze_sha256": None if hpo else bound["freeze_sha256"],
            "fold": None if hpo else bound["fold"],
            "arm": cfg["augmentation"]["arm"], "train_seed": cfg["training"]["seed"],
            "config_sha256": config_sha256(cfg), "checkpoint_selection": "last_epoch",
            "epoch": cfg["training"]["epochs"], "epochs": cfg["training"]["epochs"],
            "partition": PARTITIONS[cfg["phase"]], "score_kind": SCORE_KIND,
            "decision": deepcopy(spec["decision"])}


def _build(cfg, view, pools, context, scores, *, strict):
    spec, partition, lookup = context
    scores = _probabilities(scores, len(partition), len(spec["classes"]), strict=strict)
    truth = [lookup[item]["label"] for item in partition]
    predictions, metrics = _decide_and_measure(spec, truth, scores)
    artifact = _header(cfg, view, pools, spec)
    artifact["rows"] = [{"id": item, "group": lookup[item]["group"], "target": target, "scores": score,
                         "prediction": prediction}
                        for item, target, score, prediction in zip(partition, truth, scores, predictions)]
    artifact["metrics"] = metrics
    artifact["content_sha256"] = _content_hash(artifact)
    return artifact


def build_global_predictions(cfg, view, records, scores, *, pools, epoch):
    """Private global ``predictions.json`` from last-epoch softmax scores.

    ``scores`` follows the evaluated partition sorted by ID: D_select (hpo) or
    E_f of ``cfg.global.fold`` (final). ``view`` is the hpo-view or the final
    view document; ``records``/``pools`` are the trusted manifest and the pools
    anchored by ``cfg.global.pools_sha256``.
    """
    context = _context(cfg, view, records, pools)
    if type(epoch) is not int or epoch != cfg["training"]["epochs"]:
        _fail("epoch_selection_forbidden", "global predictions come from the last epoch only (epoch == epochs)")
    return _build(cfg, view, pools, context, scores, strict=False)


# --------------------------------------------------------------------------
# Host validation by recomputation
# --------------------------------------------------------------------------

def _structure(artifact):
    if type(artifact) is not dict:
        _fail("predictions_invalid", "global predictions artifact must be an object")
    task = artifact.get("task")
    if type(task) is dict:
        _scalar_only(task.get("type"))
    if set(artifact) != _ARTIFACT_KEYS:
        _fail("predictions_invalid", f"closed schema requires exactly {sorted(_ARTIFACT_KEYS)}")
    if artifact["content_sha256"] != _content_hash(artifact):
        _fail("predictions_invalid", "content_sha256 mismatch")
    rows = artifact["rows"]
    if type(rows) is not list or any(type(row) is not dict or set(row) != _ROW_KEYS for row in rows):
        _fail("predictions_invalid", f"rows must be objects with exactly {sorted(_ROW_KEYS)}")
    if any(type(row["id"]) is not str for row in rows):
        _fail("predictions_invalid", "row ids must be strings")
    return rows


def _coverage(phase, ids, partition, pools):
    """Leakage first (stable codes), then exact 1x coverage, then order."""
    counted = Counter(ids)
    if phase == "final":
        if set(counted) & set(pools["lists"]["D"]):
            _fail("D_in_test", "final predictions contain a D identity")
        if counted != Counter(partition):
            _fail("E_coverage_violation", "final predictions must cover E_f exactly once")
    else:
        if set(counted) & set(pools["lists"]["E"]):
            _fail("hpo_view_contains_E", "HPO predictions contain an E identity")
        if counted != Counter(partition):
            _fail("selection_coverage_violation", "HPO predictions must cover D_select exactly once")
    if ids != partition:
        _fail("predictions_invalid", "rows must be sorted by ID")


def validate_global_predictions(cfg, view, records, artifact, *, pools):
    """Read-only recomputation of a global ``predictions.json`` (spec 6.6).

    Only the probability scores are taken from the artifact; identity, rows,
    truth, decisions and metrics are rebuilt from trusted inputs and compared
    canonically. File checksums/anchoring of the artifact bytes remain the
    caller's job. ``execution_verified``/``scientific_eligible`` stay false.
    """
    rows = _structure(artifact)
    context = _context(cfg, view, records, pools)
    spec, partition, lookup = context
    if (artifact["checkpoint_selection"] != "last_epoch" or type(artifact["epoch"]) is not int
            or artifact["epoch"] != cfg["training"]["epochs"] or artifact["epochs"] != artifact["epoch"]):
        _fail("epoch_selection_forbidden", "global predictions must be the last epoch (epoch == epochs)")
    header = _header(cfg, view, pools, spec)
    mismatched = [key for key in _IDENTITY_KEYS if canonical_hash(artifact[key]) != canonical_hash(header[key])]
    if mismatched:
        _fail("prediction_identity_mismatch", f"artifact bound to another campaign/config/view: {mismatched}")
    _coverage(cfg["phase"], [row["id"] for row in rows], partition, pools)
    for row in rows:
        record = lookup[row["id"]]
        if row["group"] != record["group"] or type(row["target"]) is not int or row["target"] != record["label"]:
            _fail("records_identity_mismatch", f"row {row['id']}: group/target differ from trusted records")
    expected = _build(cfg, view, pools, context, [row["scores"] for row in rows], strict=True)
    if any(type(row["prediction"]) is not int or row["prediction"] != want["prediction"]
           for row, want in zip(rows, expected["rows"])):
        _fail("decision_recompute_mismatch", "declared decisions differ from the sealed decision rule")
    if canonical_hash(artifact["metrics"]) != canonical_hash(expected["metrics"]):
        _fail("metric_recompute_mismatch", "declared metrics differ from recomputation from scores")
    if canonical_hash(artifact) != canonical_hash(expected):
        _fail("predictions_invalid", "artifact differs from its recomputation")
    primary = expected["metrics"]["primary"]
    return {"structural_valid": True, "phase": expected["phase"], "partition": expected["partition"],
            "samples": len(rows), "primary_metric": primary["metric"], "primary_value": primary["value"],
            "primary_reason": primary["reason"], "content_sha256": expected["content_sha256"],
            "execution_verified": False, "scientific_eligible": False}
