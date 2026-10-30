"""Global D/E final report (slice S4): private ``report.json``, ID-free public
summary, descriptive comparison of the two campaigns of one dataset and the
export of the article variables. Offline host logic only (stdlib).

Fixed protocol rules (decided by Eduardo, not negotiable in this version):

* D is never a test set. Every final ``predictions.json`` is re-validated by
  RECOMPUTATION (``global_evaluation.validate_global_predictions``) against the
  config/view re-derived from the audited campaign: a D identity is
  ``D_in_test``, a duplicated/absent E identity is ``E_coverage_violation`` and
  an artifact of another campaign is ``prediction_identity_mismatch``.
* Aggregation (Q5, ADR-09): primary = mean of the 5 folds per (arm, seed), then
  mean of the executed seeds; one undefined or missing fold makes the value
  ``null`` with a reason (no partial mean). Concatenated E per seed
  (``pooled_E``) is a sensitivity analysis only.
* Every declared cell of the matrix appears in the report (succeeded, failed,
  missing or not authorized by this campaign); none disappears.
* Without a sealed analysis plan the report is descriptive: no interval, test,
  contrast or multiplicity correction. Fold/seed standard deviations are
  descriptive dispersion of repeated measurements, not clinical replicates.
* Caveat (Bradshaw 2023, DOI 10.1148/ryai.220232): reshuffling folds does not
  remove selection bias if patients used to choose hyperparameters fall into a
  test set; that is why D stays out of every test. ``execution_verified`` and
  ``scientific_eligible`` are always false here.
* Neither the report nor anything derived from it carries patient, image or
  group identifiers; ``build_report`` scans its own output against the trusted
  records (``identifier_leak``).

Errors are ``ValueError("<code>: <message>")`` with the codes of section 6.11
of ``docs/global-next-integration.md`` (S4 additions marked).
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from . import global_campaign as gc
from . import global_development as gd
from . import global_evaluation as ge
from . import global_jobs as gj
from .contracts import ARMS, canonical_hash

REPORT_PROTOCOL = "global-final-report-v1"
PUBLIC_PROTOCOL = "global-final-report-public-v1"
COMPARISON_PROTOCOL = "global-campaign-comparison-v1"
ARTICLE_SCHEMA_VERSION = 1
FOLDS = (0, 1, 2, 3, 4)
SECONDARY_METRICS = ge.SECONDARY_METRICS
ENTRY_STATUSES = ("succeeded", "failed", "timeout", "interrupted")
COMPARISON_DIFFERENCE = "hpo_d_minus_external_fixed"
INBREAST_D_SELECT_WARNING = (
    "INbreast D_select has only about 4 groups / 4 positive groups (spec 1): the exploratory HPO choice is "
    "unstable and the comparison with the external fixed recipe is descriptive only.")
# The collector's failure reason is copied into the public summary: a short
# machine code (no spaces, slashes or dots), never free text. That it carries no
# hostname or identifier is a collector contract; build_report only refuses
# codes that embed a record identifier (see ``_scan_reasons``).
MIN_SCANNED_NEEDLE = 6
REASON_CODE = re.compile(r"[a-z0-9][a-z0-9_:-]{0,63}")
NO_INFERENCE_REASON = "no_inferential_interval_without_analysis_plan"
PLAN_NOT_EXECUTED_REASON = "analysis_plan_execution_not_implemented_in_s4"

# Fixed telemetry columns (null when absent), projected from the attempt's
# ``telemetry/summary.json`` (``train_telemetry`` schema below; read as data,
# not imported, so the report does not depend on the trainer module). Extra
# summary keys are ignored; only these scalars are carried (no UTC, no context).
TELEMETRY_SUMMARY_SCHEMA = "atlas.train_telemetry_summary.v1"
TELEMETRY_FIELDS = ("status", "sessions", "duration_s", "epochs_recorded", "train_images_processed", "train_s",
                    "throughput_img_s_mean", "vram_max_allocated_bytes", "vram_max_reserved_bytes", "gpu_temp_max_c",
                    "gpu_power_max_w", "gpu_mem_used_max_mib", "process_rss_max_bytes", "machine_samples",
                    "energy_kwh", "energy_coverage_fraction", "telemetry_errors", "invalid_lines")
_TELEMETRY_INTEGERS = {"sessions", "epochs_recorded", "train_images_processed", "vram_max_allocated_bytes",
                       "vram_max_reserved_bytes", "process_rss_max_bytes", "machine_samples", "telemetry_errors",
                       "invalid_lines"}
_TELEMETRY_STATUSES = {"completed", "interrupted", "failed"}

ARTICLE_FILES = ("hyperparameters.json", "seeds.json", "hashes.json", "metrics_by_fold.csv",
                 "metrics_summary.csv", "telemetry_summary.csv", "README.md")
COMPARISON_FILES = ("campaign_comparison.json", "campaign_comparison.csv")

_ENTRY_KEYS = {"campaign_sha256", "arm", "fold", "seed", "status", "reason", "predictions",
               "initial_state_sha256", "telemetry_summary"}
_REPORT_KEYS = {"schema_version", "protocol", "visibility", "scientific_status", "caveat", "caveats", "label",
                "label_reasons", "exploratory", "descriptive_only", "execution_verified", "scientific_eligible",
                "blockers", "dataset", "task", "recipe_source", "hashes", "evaluation", "arms", "folds", "seeds",
                "pools_summary", "recipe", "matrix", "per_seed", "per_arm", "paired_differences", "pairing",
                "telemetry", "chronology", "sha256"}
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _fail(code, message):
    raise ValueError(f"{code}: {message}")


def _hex64(value):
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _seal(document, key="sha256"):
    value = deepcopy(document)
    value.pop(key, None)
    value[key] = canonical_hash(value)
    return value


def _undefined(metric, reason):
    return {"metric": metric, "value": None, "reason": f"metric_undefined:{reason}", "coverage": None,
            "per_class": None}


# --------------------------------------------------------------------------
# Inputs: collected entries (one per final job of the authorized matrix)
# --------------------------------------------------------------------------

def _check_telemetry(value):
    """Project a ``telemetry/summary.json`` onto ``TELEMETRY_FIELDS`` (absent -> null)."""
    if value is None:
        return None
    if type(value) is not dict or value.get("schema") != TELEMETRY_SUMMARY_SCHEMA:
        _fail("telemetry_summary_invalid", f"telemetry_summary must be a {TELEMETRY_SUMMARY_SCHEMA} object")
    energy = value.get("energy")
    if energy is not None and type(energy) is not dict:
        _fail("telemetry_summary_invalid", "telemetry_summary.energy must be an object")
    source = dict(value, energy_coverage_fraction=(energy or {}).get("coverage_fraction"))
    out = {}
    for name in TELEMETRY_FIELDS:
        item = source.get(name)
        if item is not None:
            if name == "status":
                ok = type(item) is str and item in _TELEMETRY_STATUSES
            elif name in _TELEMETRY_INTEGERS:
                ok = type(item) is int and item >= 0
            else:
                ok = type(item) in (int, float) and math.isfinite(item) and item >= 0
            if not ok:
                _fail("telemetry_summary_invalid", f"telemetry_summary.{name} has an invalid value")
        out[name] = item
    return out


def _check_entry(entry):
    """Closed collected entry: identity of the cell, queue outcome and the private artifact."""
    if type(entry) is not dict or set(entry) != _ENTRY_KEYS:
        _fail("report_input_invalid", f"collected entries require exactly {sorted(_ENTRY_KEYS)}")
    if entry["status"] not in ENTRY_STATUSES:
        _fail("report_input_invalid", f"entry status must be one of {list(ENTRY_STATUSES)}")
    if entry["status"] == "succeeded":
        if type(entry["predictions"]) is not dict or entry["reason"] is not None:
            _fail("report_input_invalid", "a succeeded entry carries its predictions.json and no reason")
    elif (entry["predictions"] is not None or type(entry["reason"]) is not str
          or not REASON_CODE.fullmatch(entry["reason"])):
        _fail("report_input_invalid", "a failed/timeout/interrupted entry has null predictions and a reason code "
              f"matching {REASON_CODE.pattern} (no free text, paths or identifiers)")
    if entry["initial_state_sha256"] is not None and not _hex64(entry["initial_state_sha256"]):
        _fail("report_input_invalid", "initial_state_sha256 must be null or a lowercase hex64")
    _check_telemetry(entry["telemetry_summary"])


def _cell_of(entry, campaign):
    """(arm, fold, seed) of an entry bound to this campaign; foreign campaigns are refused."""
    if entry["campaign_sha256"] != campaign["sha256"]:
        _fail("prediction_identity_mismatch", "collected artifact belongs to another campaign")
    arm, fold, seed = entry["arm"], entry["fold"], entry["seed"]
    if arm not in ARMS or type(fold) is not int or fold not in FOLDS or type(seed) is not int:
        _fail("prediction_identity_mismatch", "collected entry arm/fold/seed are not in the campaign matrix")
    if seed not in campaign["seeds"]:
        _fail("prediction_identity_mismatch", "collected entry seed is not declared by the campaign")
    if seed not in campaign["matrix"]["authorized_seeds"]:
        _fail("campaign_invalid", "collected entry seed is outside the stage authorized by this campaign")
    return arm, fold, seed


# --------------------------------------------------------------------------
# Aggregation helpers (math.fsum, canonical order)
# --------------------------------------------------------------------------

def _sd(values):
    if len(values) < 2:
        return None
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _aggregate(items, unit):
    """``items`` = [(key, value|None, state)] with state in {defined, undefined, missing}.

    No partial mean: any missing or undefined unit makes the aggregate null.
    """
    missing = [key for key, _, state in items if state == "missing"]
    undefined = [key for key, _, state in items if state == "undefined"]
    values = [value for _, value, state in items if state == "defined"]
    if missing:
        value, reason = None, f"metric_undefined:missing_{unit}"
    elif undefined:
        value, reason = None, f"metric_undefined:undefined_in_{unit}"
    elif not items:
        value, reason = None, f"metric_undefined:no_{unit}"
    else:
        value, reason = math.fsum(values) / len(values), None
    return {"value": value, "reason": reason, "unit": unit, "n_expected": len(items), "n_defined": len(values),
            "missing": missing, "undefined": undefined, "sd": _sd(values) if value is not None else None}


def _state(metric):
    if metric is None:
        return None, "missing"
    return metric["value"], "defined" if metric["value"] is not None else "undefined"


def _interval(analysis_plan_sha256):
    return {"ci_low": None, "ci_high": None,
            "reason": NO_INFERENCE_REASON if analysis_plan_sha256 is None else PLAN_NOT_EXECUTED_REASON}


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------

def _pools_summary(records, pools):
    lookup = {row["id"]: row for row in records}
    classes = pools["task"]["classes"]

    def block(ids):
        rows = [lookup[item] for item in ids]
        return {"images": len(rows), "groups": len({row["group"] for row in rows}),
                "class_images": {f"class_{c}": sum(row["label"] == c for row in rows) for c in classes},
                "class_groups": {f"class_{c}": len({row["group"] for row in rows if row["label"] == c})
                                 for c in classes}}

    summary = {name: block(pools["lists"][name]) for name in gd.POOL_NAMES}
    summary["E_folds"] = [{"fold": fold["fold"], **block(fold["test_ids"])} for fold in pools["evaluation_folds"]]
    return summary


def _recipe_block(campaign, freeze, evidence):
    if campaign["freeze"]["kind"] == "external_fixed":
        return {"source": "external_fixed", "freeze_sha256": freeze["sha256"], "recipe": deepcopy(freeze["recipe"]),
                "external_source": deepcopy(freeze["source"]), "hpo": None}
    plan = freeze["selection_plan"]
    trials = {}
    for row in evidence["rows"]:
        trials.setdefault(row["candidate_id"], []).append(
            {"seed": row["seed"], "status": row["status"], "state": row["metric"]["state"],
             "value": row["metric"]["value"], "reason": row["metric"]["reason"]})
    candidates = []
    for candidate in plan["candidates"]:
        rows = sorted(trials.get(candidate["id"], []), key=lambda item: item["seed"])
        defined = [item["value"] for item in rows if item["state"] == "defined"]
        complete = len(defined) == len(plan["seeds"]) == len(rows)
        candidates.append({"id": candidate["id"], "parameters": deepcopy(candidate["parameters"]), "trials": rows,
                           "mean": math.fsum(defined) / len(defined) if complete else None,
                           "mean_reason": None if complete else "metric_undefined:not_all_trials_defined",
                           "selected": candidate["id"] == freeze["selected_candidate_id"]})
    selected = next(item for item in candidates if item["selected"])
    return {"source": "hpo_d", "freeze_sha256": freeze["sha256"], "recipe": deepcopy(freeze["frozen_recipe"]),
            "external_source": None,
            "hpo": {"label": "exploratory", "partition": "D_select", "endpoint": plan["endpoint"],
                    "direction": plan["direction"], "ranking": plan["ranking"],
                    "failure_policy": plan["failure_policy"], "seeds": list(plan["seeds"]),
                    "plan_sha256": campaign["freeze"]["plan_sha256"],
                    "evidence_sha256": campaign["freeze"]["evidence_sha256"],
                    "search_space_sha256": campaign["freeze"]["search_space_sha256"], "evidence_kind": evidence["kind"],
                    "candidates": candidates, "selected_candidate_id": freeze["selected_candidate_id"],
                    "selected_parameters": deepcopy(freeze["selected_parameters"]),
                    "selected_mean": selected["mean"]}}


def _hashes(campaign, pools):
    identity = campaign["code_identity"]
    return {"pools_sha256": campaign["pools_sha256"], "records_sha256": pools["records_sha256"],
            "campaign_sha256": campaign["sha256"], "freeze_sha256": campaign["freeze"]["sha256"],
            "plan_sha256": campaign["freeze"]["plan_sha256"], "evidence_sha256": campaign["freeze"]["evidence_sha256"],
            "search_space_sha256": campaign["freeze"]["search_space_sha256"],
            "configs_manifest_sha256": campaign["configs"]["manifest_sha256"],
            "augmentation_presets_sha256": campaign["augmentation_presets_sha256"],
            "analysis_plan_sha256": campaign["analysis_plan_sha256"],
            "code_commit": campaign["runtime"]["code_commit"], "freeze_commit": identity["freeze_commit"],
            "code_tree_sha256": identity["final_code_tree_sha256"], "image_id": campaign["runtime"]["image_id"],
            "environment_sha256": campaign["runtime"]["environment_sha256"],
            "imagenet_weights_sha256": campaign["runtime"]["imagenet_weights_sha256"]}


def _seeds_block(campaign, pools, freeze):
    authorized = list(campaign["matrix"]["authorized_seeds"])
    return {"final_policy": list(campaign["seeds"]), "execution_stages": deepcopy(campaign["execution_stages"]),
            "authorized": authorized, "not_authorized": [s for s in campaign["seeds"] if s not in authorized],
            "hpo": list(freeze["selection_plan"]["seeds"]) if campaign["freeze"]["kind"] == "hpo_d" else None,
            "partition": deepcopy(pools["partition_provenance"]["seeds"])}


def _chronology_block(chronology, campaign):
    if chronology is None:
        return {"audited": False}
    if (type(chronology) is not dict or chronology.get("protocol") != gc.CHRONOLOGY_PROTOCOL
            or chronology.get("campaign_sha256") != campaign["sha256"] or chronology.get("structural_valid") is not True
            or chronology.get("execution_verified") is not False or chronology.get("scientific_eligible") is not False):
        _fail("chronology_invalid", "chronology audit is not the audit_chronology result of this campaign")
    return {"audited": True, "head": chronology["head"], "C_freeze": chronology["C_freeze"],
            "C_campaign": chronology["C_campaign"], "final_jobs": chronology["final_jobs"],
            "other_campaign_final_jobs": chronology["other_campaign_final_jobs"],
            "warnings": list(chronology["warnings"])}


def _scan_identifiers(document, records, where):
    """Exact-match scan of every key/string against record IDs, groups and paths."""
    forbidden = set()
    for row in records:
        forbidden.update(value for value in (row.get("id"), row.get("group"), row.get("path")) if type(value) is str)

    def walk(value):
        if type(value) is dict:
            for key, item in value.items():
                if key in forbidden:
                    _fail("identifier_leak", f"{where} contains a record identifier as a key")
                walk(item)
        elif type(value) in (list, tuple):
            for item in value:
                walk(item)
        elif type(value) is str and value in forbidden:
            _fail("identifier_leak", f"{where} contains a record identifier")
    walk(document)


def _scan_reasons(report, records):
    """Failure reasons are collector text: refuse any that embeds a record ID, group or path (substring)."""
    needles = set()
    for row in records:
        for value in (row.get("id"), row.get("group"), row.get("path")):
            if type(value) is str:
                needles.update((value.lower(), Path(value).name.lower()))
    # Very short identifiers ("1", "17") would flag legitimate codes such as exit_17.
    needles = {needle for needle in needles if len(needle) >= MIN_SCANNED_NEEDLE}
    for cell in report["matrix"]["cells"]:
        reason = cell["reason"]
        if type(reason) is str and any(needle in reason.lower() for needle in needles):
            _fail("identifier_leak", "a failure reason embeds a record identifier or path")


# --------------------------------------------------------------------------
# build_report
# --------------------------------------------------------------------------

def _validated_cells(campaign, entries, records, pools, configs):
    """{(arm, fold, seed): cell} after recomputation of every collected artifact."""
    if type(entries) is not list:
        _fail("report_input_invalid", "collected_artifacts must be a list of entries")
    freeze_sha256 = campaign["freeze"]["sha256"]
    views = {fold: gj.derive_final_view_doc(records, pools, fold, expected_pools_sha256=campaign["pools_sha256"],
                                            freeze_sha256=freeze_sha256) for fold in FOLDS}
    cells = {}
    for entry in entries:
        _check_entry(entry)
        key = _cell_of(entry, campaign)
        arm, fold, seed = key
        if key in cells:
            if entry["status"] == "succeeded" and cells[key]["status"] == "succeeded":
                _fail("E_coverage_violation", f"two artifacts for {arm} fold {fold} seed {seed} cover E_f twice")
            _fail("report_input_invalid", f"duplicate collected entry for {arm} fold {fold} seed {seed}")
        cell = {"status": entry["status"], "reason": entry["reason"], "metrics": None, "content_sha256": None,
                "config_sha256": None, "n_samples": None, "rows": None,
                "initial_state_sha256": entry["initial_state_sha256"],
                "telemetry_summary": _check_telemetry(entry["telemetry_summary"])}
        if entry["status"] == "succeeded":
            cfg = configs[f"{arm}-f{fold}-s{seed}.json"]
            artifact = entry["predictions"]
            # Binding first, so a foreign campaign's artifact gets one stable code
            # whatever else differs (epochs, view, ...); then full recomputation.
            if (artifact.get("freeze_sha256") != freeze_sha256 or artifact.get("pools_sha256") != campaign["pools_sha256"]
                    or artifact.get("config_sha256") != ge.config_sha256(cfg)):
                _fail("prediction_identity_mismatch", f"{arm} fold {fold} seed {seed}: artifact is bound to another "
                                                      "campaign/freeze/config")
            result = ge.validate_global_predictions(cfg, views[fold], records, artifact, pools=pools)
            # validate_global_predictions proved metrics == recomputation from scores.
            cell.update(metrics=deepcopy(artifact["metrics"]), content_sha256=result["content_sha256"],
                        config_sha256=artifact["config_sha256"], n_samples=result["samples"],
                        rows=[(row["id"], row["scores"]) for row in artifact["rows"]], spec=cfg["evaluation"])
        cells[key] = cell
    return cells


def _check_e_coverage(cells, pools, authorized):
    """Per (arm, seed): union of fold rows covers E at most once, exactly once when complete."""
    e_ids, d_ids = set(pools["lists"]["E"]), set(pools["lists"]["D"])
    for arm in ARMS:
        for seed in authorized:
            seen = Counter()
            present = [fold for fold in FOLDS if cells.get((arm, fold, seed), {}).get("status") == "succeeded"]
            for fold in present:
                seen.update(item for item, _ in cells[(arm, fold, seed)]["rows"])
            if set(seen) & d_ids:
                _fail("D_in_test", f"{arm} seed {seed}: a D identity appears in the E predictions")
            if any(count != 1 for count in seen.values()) or not set(seen) <= e_ids:
                _fail("E_coverage_violation", f"{arm} seed {seed}: E is not covered at most once")
            if len(present) == len(FOLDS) and set(seen) != e_ids:
                _fail("E_coverage_violation", f"{arm} seed {seed}: the 5 folds do not cover E exactly once")


def _pooled(cells, arm, seed, records, primary_name):
    keys = [(arm, fold, seed) for fold in FOLDS]
    if any(cells.get(key, {}).get("status") != "succeeded" for key in keys):
        return {"primary": _undefined(primary_name, "missing_folds"),
                "secondary": {name: _undefined(name, "missing_folds") for name in SECONDARY_METRICS}}
    lookup = {row["id"]: row for row in records}
    rows = sorted((row for key in keys for row in cells[key]["rows"]), key=lambda row: row[0])
    spec = cells[keys[0]]["spec"]
    _, metrics = ge._decide_and_measure(spec, [lookup[item]["label"] for item, _ in rows],
                                        [list(scores) for _, scores in rows])
    return metrics


def _cell_metric(cells, key, name):
    cell = cells.get(key)
    if cell is None or cell["status"] != "succeeded":
        return None
    return cell["metrics"]["primary"] if name == "primary" else cell["metrics"]["secondary"][name]


def _per_seed(cells, authorized, records, campaign):
    primary_name = campaign["evaluation"]["primary"]
    out = []
    for arm in ARMS:
        for seed in authorized:
            def over_folds(name):
                return _aggregate([(fold, *_state(_cell_metric(cells, (arm, fold, seed), name))) for fold in FOLDS],
                                  "folds")
            out.append({"arm": arm, "seed": seed, "primary": {"metric": primary_name, **over_folds("primary")},
                        "secondary": {name: {"metric": name, **over_folds(name)} for name in SECONDARY_METRICS},
                        "pooled_E": _pooled(cells, arm, seed, records, primary_name)})
    return out


def _per_arm(per_seed, campaign):
    primary_name = campaign["evaluation"]["primary"]
    authorized = campaign["matrix"]["authorized_seeds"]
    interval = _interval(campaign["analysis_plan_sha256"])
    out = []
    for arm in ARMS:
        rows = [row for row in per_seed if row["arm"] == arm]

        def over_seeds(pick):
            return _aggregate([(row["seed"], *_state(pick(row))) for row in rows], "seeds")
        out.append({"arm": arm, "seeds": list(authorized), "all_stages_authorized": campaign["matrix"]["complete"],
                    "primary": {"metric": primary_name, "aggregation": "mean_over_folds_then_seeds",
                                **over_seeds(lambda row: row["primary"]), "interval": deepcopy(interval)},
                    "secondary": {name: {"metric": name, "aggregation": "mean_over_folds_then_seeds",
                                         **over_seeds(lambda row, n=name: row["secondary"][n]),
                                         "interval": deepcopy(interval)} for name in SECONDARY_METRICS},
                    "pooled_E": {"primary": {"metric": primary_name, "aggregation": "pooled_E_mean_over_seeds",
                                             **over_seeds(lambda row: row["pooled_E"]["primary"])},
                                 "secondary": {name: {"metric": name, "aggregation": "pooled_E_mean_over_seeds",
                                                      **over_seeds(lambda row, n=name: row["pooled_E"]["secondary"][n])}
                                               for name in SECONDARY_METRICS}}})
    return out


def _paired_differences(cells, authorized, campaign):
    """Primary arm - none per (seed, fold); descriptive only (no test, no multiplicity)."""
    cells_out, summary = [], []
    for arm in ARMS[1:]:
        per_seed = []
        for seed in authorized:
            items = []
            for fold in FOLDS:
                left = _state(_cell_metric(cells, (arm, fold, seed), "primary"))
                right = _state(_cell_metric(cells, ("none", fold, seed), "primary"))
                if "missing" in (left[1], right[1]):
                    value, state, reason = None, "missing", "metric_undefined:missing_cell"
                elif "undefined" in (left[1], right[1]):
                    value, state, reason = None, "undefined", "metric_undefined:undefined_cell"
                else:
                    value, state, reason = left[0] - right[0], "defined", None
                cells_out.append({"arm": arm, "reference": "none", "seed": seed, "fold": fold, "value": value,
                                  "reason": reason})
                items.append((fold, value, state))
            per_seed.append((seed, *_state_of(_aggregate(items, "folds"))))
        summary.append({"arm": arm, "reference": "none", "aggregation": "mean_over_folds_then_seeds",
                        **_aggregate(per_seed, "seeds"), "interval": _interval(campaign["analysis_plan_sha256"])})
    return {"metric": campaign["evaluation"]["primary"], "inference": "descriptive_only", "cells": cells_out,
            "summary": summary}


def _state_of(aggregate):
    return aggregate["value"], "defined" if aggregate["value"] is not None else "undefined"


def _pairing(cells, authorized):
    """initial_state_sha256 equal across the 10 arms of one (fold, seed), different across seeds."""
    checked, incomplete = 0, 0
    for fold in FOLDS:
        by_seed = {}
        for seed in authorized:
            states = [cells.get((arm, fold, seed), {}).get("initial_state_sha256") for arm in ARMS]
            known = {value for value in states if value is not None}
            if len(known) > 1:
                _fail("pairing_violation", f"fold {fold} seed {seed}: arms start from different initial states")
            if None in states:
                incomplete += 1
            else:
                checked += 1
            if known:
                by_seed[seed] = known.pop()
        if len(set(by_seed.values())) != len(by_seed):
            _fail("pairing_violation", f"fold {fold}: two seeds share one initial state")
    return {"status": "consistent" if not incomplete else "incomplete", "groups_checked": checked,
            "groups_incomplete": incomplete,
            "rule": "initial_state_sha256 equal across arms per (fold, seed) and distinct across seeds"}


def _matrix(cells, campaign):
    authorized = campaign["matrix"]["authorized_seeds"]
    out, counts = [], Counter()
    for seed in campaign["seeds"]:
        for arm in ARMS:
            for fold in FOLDS:
                cell = cells.get((arm, fold, seed))
                row = {"arm": arm, "fold": fold, "seed": seed, "status": None, "reason": None, "primary": None,
                       "secondary": None, "n_samples": None, "predictions_content_sha256": None,
                       "config_sha256": None}
                if seed not in authorized:
                    row.update(status="not_authorized", reason="stage_not_authorized_by_this_campaign")
                elif cell is None:
                    row.update(status="missing", reason="artifact_not_collected")
                elif cell["status"] != "succeeded":
                    row.update(status=cell["status"], reason=cell["reason"])
                else:
                    metrics = cell["metrics"]
                    row.update(status="succeeded", primary=deepcopy(metrics["primary"]),
                               secondary=deepcopy(metrics["secondary"]), n_samples=cell["n_samples"],
                               predictions_content_sha256=cell["content_sha256"], config_sha256=cell["config_sha256"])
                counts[row["status"]] += 1
                out.append(row)
    authorized_cells = campaign["matrix"]["authorized_cells"]
    return {"declared_cells": campaign["matrix"]["declared_cells"], "authorized_cells": authorized_cells,
            "succeeded_cells": counts["succeeded"],
            "failed_cells": counts["failed"] + counts["timeout"] + counts["interrupted"],
            "missing_cells": counts["missing"], "not_authorized_cells": counts["not_authorized"],
            "complete": counts["succeeded"] == campaign["matrix"]["declared_cells"],
            "authorized_complete": counts["succeeded"] == authorized_cells, "cells": out}


def build_report(campaign, collected_artifacts, pools, records, *, expected_campaign_sha256, freeze, dataset_source,
                 augmentation_presets, evidence=None, chronology=None, output=None,
                 stage_authorization=None, code_repo=None, scope="stage1"):
    """Private ``report.json`` of one sealed campaign (spec 6.10, Q5).

    ``collected_artifacts`` is a list of closed entries, one per final job:
    ``{campaign_sha256, arm, fold, seed, status, reason, predictions,
    initial_state_sha256, telemetry_summary}`` (``status`` in
    ``ENTRY_STATUSES``; ``predictions`` is the private ``predictions.json``
    for ``succeeded``, else null with a reason). The campaign is re-audited
    from its declared inputs, the final configs/views re-derived and every
    artifact recomputed; any contaminated artifact fails the whole report.
    ``chronology`` (optional) is the ``audit_chronology`` result. ``output``
    persists the report no-clobber.
    """
    gc.audit_campaign(records, pools, campaign, expected_campaign_sha256=expected_campaign_sha256, freeze=freeze,
                      dataset_source=dataset_source, augmentation_presets=augmentation_presets, evidence=evidence,
                      stage_authorization=stage_authorization, code_repo=code_repo)
    if scope not in ("stage1", "all_authorized"):
        _fail("report_invalid", "scope must be stage1 or all_authorized")
    if scope == "stage1" and isinstance(chronology, dict):
        chronology = chronology.get("stage1_chronology", chronology)
    if scope == "all_authorized" and stage_authorization is not None:
        # A reporting projection only; the sealed campaign above was audited intact.
        campaign = deepcopy(campaign)
        matrix = campaign["matrix"]
        matrix["authorized_seeds"] += stage_authorization["seeds"]
        matrix["authorized_cells"] = len(ARMS) * 5 * len(matrix["authorized_seeds"])
        matrix["complete"] = matrix["authorized_seeds"] == campaign["seeds"]
        for stage in campaign["execution_stages"]:
            if stage["stage"] == stage_authorization["stage"]:
                stage["status"] = "authorized_by_stage_authorization"
    kind = campaign["freeze"]["kind"]
    authorized = list(campaign["matrix"]["authorized_seeds"])
    configs = gj.build_final_configs(
        records, pools, expected_pools_sha256=campaign["pools_sha256"], dataset_source=dataset_source, freeze=freeze,
        freeze_kind=kind, expected_freeze_sha256=campaign["freeze"]["sha256"],
        augmentation_presets=augmentation_presets, seeds=authorized, evidence=evidence,
        expected_plan_sha256=campaign["freeze"]["plan_sha256"])
    cells = _validated_cells(campaign, collected_artifacts, records, pools, configs)
    _check_e_coverage(cells, pools, authorized)
    per_seed = _per_seed(cells, authorized, records, campaign)
    eligibility = campaign["eligibility"]
    report = {
        "schema_version": 1, "protocol": REPORT_PROTOCOL, "visibility": "private", "scientific_status": "provisional",
        "caveat": gc.SELECTION_BIAS_CAVEAT, "caveats": list(campaign["caveats"]), "label": campaign["label"],
        "label_reasons": list(campaign["label_reasons"]), "exploratory": campaign["label"] == "exploratory",
        "descriptive_only": True, "execution_verified": False, "scientific_eligible": False,
        "blockers": list(eligibility["blockers"]), "dataset": deepcopy(campaign["dataset"]),
        "task": deepcopy(campaign["task"]), "recipe_source": kind, "hashes": _hashes(campaign, pools),
        "evaluation": deepcopy(campaign["evaluation"]), "arms": list(ARMS), "folds": list(FOLDS),
        "seeds": _seeds_block(campaign, pools, freeze), "pools_summary": _pools_summary(records, pools),
        "recipe": _recipe_block(campaign, freeze, evidence), "matrix": _matrix(cells, campaign),
        "per_seed": per_seed, "per_arm": _per_arm(per_seed, campaign),
        "paired_differences": _paired_differences(cells, authorized, campaign),
        "pairing": _pairing(cells, authorized),
        "telemetry": [{"arm": arm, "fold": fold, "seed": seed, **cell["telemetry_summary"]}
                      for (arm, fold, seed), cell in sorted(cells.items(), key=lambda item: _cell_order(item[0]))
                      if cell["telemetry_summary"] is not None],
        "chronology": _chronology_block(chronology, campaign)}
    if scope == "all_authorized":
        report["scope"] = scope
        report["stage_authorization"] = (None if stage_authorization is None else
            {key: deepcopy(stage_authorization[key]) for key in
             ("sha256", "stage", "seeds", "execution_code_commit", "configs_manifest", "decided_on",
              "decided_after_primary_results", "rationale", "seed_interpretation")})
        if stage_authorization is not None and stage_authorization.get("cross_environment"):
            report["stage_authorization"].update(
                cross_environment=True, environment_sha256=stage_authorization["runtime"]["environment_sha256"],
                environment_label="different environment (declared)")
            for row in report["per_seed"]:
                if row["seed"] in stage_authorization["seeds"]:
                    row["environment"] = "different environment (declared)"
            for cell in report["matrix"]["cells"]:
                if cell["seed"] in stage_authorization["seeds"]:
                    cell["environment"] = "different environment (declared)"
    report = _seal(report)
    _scan_identifiers(report, records, "report")
    _scan_reasons(report, records)
    check_report(report)
    if output is not None:
        gd._publish(output, report, check_report)
    return report


def _cell_order(key):
    arm, fold, seed = key
    return seed, ARMS.index(arm), fold


def check_report(report):
    """Closed schema, checksum and fixed flags of a private report (``report_invalid``)."""
    expected_keys = _REPORT_KEYS | ({"scope", "stage_authorization"} if isinstance(report, dict) and report.get("scope") == "all_authorized" else set())
    if type(report) is not dict or set(report) != expected_keys:
        _fail("report_invalid", f"report must have exactly the {REPORT_PROTOCOL} keys")
    try:
        body = canonical_hash({key: value for key, value in report.items() if key != "sha256"})
    except (TypeError, ValueError) as error:
        _fail("report_invalid", f"report is not canonical JSON ({error})")
    if report["sha256"] != body:
        _fail("report_invalid", "report checksum mismatch")
    if (report["schema_version"] != 1 or report["protocol"] != REPORT_PROTOCOL or report["visibility"] != "private"
            or report["scientific_status"] != "provisional"):
        _fail("report_invalid", f"only provisional private {REPORT_PROTOCOL} is supported")
    if report["execution_verified"] is not False or report["scientific_eligible"] is not False:
        _fail("report_invalid", "execution_verified and scientific_eligible are always false")
    if report["caveat"] != gc.SELECTION_BIAS_CAVEAT or report["caveats"] != list(gc.CAMPAIGN_CAVEATS):
        _fail("report_invalid", "report header must carry the Bradshaw 2023 caveat and the fixed caveats")
    if report["recipe_source"] not in gj.FREEZE_KINDS or list(report["arms"]) != list(ARMS):
        _fail("report_invalid", "report recipe source/arms malformed")
    matrix = report["matrix"]
    cells = matrix.get("cells") if type(matrix) is dict else None
    if type(cells) is not list:
        _fail("report_invalid", "report matrix must carry a list of cells")
    for cell in cells:
        if type(cell) is not dict or "status" not in cell or "reason" not in cell:
            _fail("report_invalid", "a matrix cell needs a status and a reason")
        status, reason = cell["status"], cell["reason"]
        if status == "succeeded":
            ok = reason is None
        elif status == "missing":
            ok = reason == "artifact_not_collected"
        elif status == "not_authorized":
            ok = reason == "stage_not_authorized_by_this_campaign"
        elif type(status) is str and status in ENTRY_STATUSES:
            ok = type(reason) is str and REASON_CODE.fullmatch(reason) is not None
        else:
            _fail("report_invalid", "a matrix cell status is not part of the closed set")
        if not ok:
            _fail("report_invalid", "a matrix cell reason is not a closed code")
    return report


# --------------------------------------------------------------------------
# Public summary (no identifiers of any kind)
# --------------------------------------------------------------------------

def _value(metric):
    return {"value": metric["value"], "reason": metric["reason"]} if metric is not None else None


def _public_aggregate(aggregate):
    return {key: deepcopy(aggregate[key]) for key in ("metric", "value", "reason", "unit", "n_expected", "n_defined",
                                                      "missing", "undefined", "sd") if key in aggregate}


def public_summary(report):
    """Public summary of a private report: aggregates only, no IDs, groups, paths or job IDs."""
    check_report(report)
    hashes = report["hashes"]
    cells = [{"arm": c["arm"], "fold": c["fold"], "seed": c["seed"], "status": c["status"], "reason": c["reason"],
              "n_samples": c["n_samples"], "primary": _value(c["primary"]),
              "secondary": None if c["secondary"] is None else {n: _value(c["secondary"][n])
                                                                 for n in SECONDARY_METRICS}}
             for c in report["matrix"]["cells"]]
    summary = {
        "schema_version": 1, "protocol": PUBLIC_PROTOCOL, "visibility": "public",
        "scientific_status": "provisional", "caveat": report["caveat"], "caveats": list(report["caveats"]),
        "label": report["label"], "label_reasons": list(report["label_reasons"]), "exploratory": report["exploratory"],
        "descriptive_only": report["descriptive_only"], "execution_verified": False, "scientific_eligible": False,
        "blockers": list(report["blockers"]),
        "dataset": {"id": report["dataset"]["id"], "version": report["dataset"]["version"]},
        "task": {"type": report["task"]["type"], "endpoint": report["task"]["endpoint"],
                 "classes": list(report["task"]["classes"])},
        "recipe_source": report["recipe_source"],
        "hashes": {key: hashes[key] for key in ("pools_sha256", "campaign_sha256", "freeze_sha256", "code_commit",
                                                "image_id", "configs_manifest_sha256")},
        "report_sha256": report["sha256"], "evaluation": deepcopy(report["evaluation"]),
        "seeds": {"final_policy": list(report["seeds"]["final_policy"]),
                  "authorized": list(report["seeds"]["authorized"]),
                  "not_authorized": list(report["seeds"]["not_authorized"])},
        "pools_summary": deepcopy(report["pools_summary"]),
        "matrix": {**{key: report["matrix"][key] for key in report["matrix"] if key != "cells"}, "cells": cells},
        "per_arm": [{"arm": row["arm"], "primary": _public_aggregate(row["primary"]),
                     "secondary": {n: _public_aggregate(row["secondary"][n]) for n in SECONDARY_METRICS},
                     "pooled_E": {"primary": _public_aggregate(row["pooled_E"]["primary"]),
                                  "secondary": {n: _public_aggregate(row["pooled_E"]["secondary"][n])
                                                for n in SECONDARY_METRICS}}}
                    for row in report["per_arm"]],
        "paired_differences": [_public_aggregate(row) | {"arm": row["arm"], "reference": row["reference"]}
                               for row in report["paired_differences"]["summary"]],
        "pairing": {"status": report["pairing"]["status"]}}
    if report.get("scope") == "all_authorized":
        summary["scope"] = "all_authorized"
        authorization = report["stage_authorization"]
        summary["stage_authorization"] = (None if authorization is None else
            {key: deepcopy(authorization[key]) for key in
             ("sha256", "stage", "seeds", "execution_code_commit", "decided_on", "decided_after_primary_results")})
        if authorization is not None and authorization.get("cross_environment"):
            summary["stage_authorization"].update(
                cross_environment=True, environment_sha256=authorization["environment_sha256"],
                environment_label="different environment (declared)")
    return _seal(summary)


# --------------------------------------------------------------------------
# Comparison of the two campaigns of one dataset (descriptive only)
# --------------------------------------------------------------------------

def check_comparison(comparison):
    """Checksum and fixed flags of a comparison document (``campaign_comparison_invalid``)."""
    if type(comparison) is not dict or comparison.get("protocol") != COMPARISON_PROTOCOL:
        _fail("campaign_comparison_invalid", f"comparison must be a {COMPARISON_PROTOCOL} document")
    try:
        body = canonical_hash({k: v for k, v in comparison.items() if k != "sha256"})
    except (TypeError, ValueError) as error:
        _fail("campaign_comparison_invalid", f"comparison is not canonical JSON ({error})")
    if comparison.get("sha256") != body:
        _fail("campaign_comparison_invalid", "comparison checksum mismatch")
    if ("hypothesis_test" not in comparison or comparison.get("scientific_status") != "provisional"
            or comparison.get("execution_verified") is not False
            or comparison.get("scientific_eligible") is not False or comparison.get("superiority_claim") is not False
            or comparison.get("hypothesis_test") is not None or comparison.get("inference") != "descriptive_only"
            or comparison.get("visibility") != "private"):
        _fail("campaign_comparison_invalid", "comparison flags must be provisional/false/descriptive_only")
    return comparison


def compare_campaigns(report_a, report_b):
    """Per arm, primary of the HPO-exploratory campaign minus the external fixed recipe.

    Same dataset, pools, task, arms and folds are required; one report per
    recipe source. Descriptive only: no hypothesis test, no interval, no claim
    of superiority (spec 1, 10; D_select of INbreast has about 4 positive groups).
    """
    check_report(report_a)
    check_report(report_b)
    reports = {report_a["recipe_source"]: report_a, report_b["recipe_source"]: report_b}
    if set(reports) != {"external_fixed", "hpo_d"}:
        _fail("campaign_comparison_invalid", "compare exactly one external_fixed and one hpo_d campaign")
    external, hpo = reports["external_fixed"], reports["hpo_d"]
    for key in ("dataset", "task", "arms", "folds"):
        if canonical_hash(external[key]) != canonical_hash(hpo[key]):
            _fail("campaign_comparison_invalid", f"campaigns differ in {key}")
    if (external["hashes"]["pools_sha256"] != hpo["hashes"]["pools_sha256"]
            or external["hashes"]["records_sha256"] != hpo["hashes"]["records_sha256"]):
        _fail("campaign_comparison_invalid", "campaigns must share the same pools (same D and E folds)")
    if external["evaluation"]["primary"] != hpo["evaluation"]["primary"]:
        _fail("campaign_comparison_invalid", "campaigns use different primary metrics")
    common = [seed for seed in hpo["seeds"]["authorized"] if seed in external["seeds"]["authorized"]]
    index = {source: {(c["arm"], c["fold"], c["seed"]): c for c in report["matrix"]["cells"]}
             for source, report in reports.items()}
    arms = []
    for arm in ARMS:
        a = next(row for row in external["per_arm"] if row["arm"] == arm)["primary"]
        b = next(row for row in hpo["per_arm"] if row["arm"] == arm)["primary"]
        if a["value"] is None or b["value"] is None:
            difference, reason = None, "metric_undefined:primary_undefined_in_a_campaign"
        else:
            difference, reason = b["value"] - a["value"], None
        per_cell = []
        for seed in common:
            for fold in FOLDS:
                left = index["external_fixed"][(arm, fold, seed)]["primary"]
                right = index["hpo_d"][(arm, fold, seed)]["primary"]
                if left is None or right is None:
                    value, why = None, "metric_undefined:missing_cell"
                elif left["value"] is None or right["value"] is None:
                    value, why = None, "metric_undefined:undefined_cell"
                else:
                    value, why = right["value"] - left["value"], None
                per_cell.append({"seed": seed, "fold": fold, "external_fixed": _value(left), "hpo_d": _value(right),
                                 "difference": value, "reason": why})
        arms.append({"arm": arm, "external_fixed": {"value": a["value"], "reason": a["reason"]},
                     "hpo_d": {"value": b["value"], "reason": b["reason"]}, "difference": difference,
                     "reason": reason, "per_cell": per_cell})
    select = hpo["pools_summary"]["D_select"]
    warnings = [
        f"D_select used by the exploratory HPO has {select['images']} images in {select['groups']} groups "
        f"(groups per class: {json.dumps(select['class_groups'], sort_keys=True)}); a selection on so few groups is "
        "unstable.",
        INBREAST_D_SELECT_WARNING,
        "Both campaigns share pools, D and the same 5 E folds: fold/seed differences are paired repeated "
        "measurements on one partition, not independent clinical replicates.",
        "Descriptive only: no hypothesis test, interval or multiplicity correction; no superiority claim.",
    ]
    if set(external["seeds"]["authorized"]) != set(hpo["seeds"]["authorized"]):
        warnings.append("The campaigns authorize different seed sets: the per-arm difference compares each "
                        "campaign's own seed aggregate and is not paired; only per_cell is restricted to "
                        "common_seeds.")
    if not (external["matrix"]["complete"] and hpo["matrix"]["complete"]):
        warnings.append("At least one campaign matrix is incomplete (stage 1 only); seed variability is not "
                        "estimable.")
    comparison = {
        "schema_version": 1, "protocol": COMPARISON_PROTOCOL, "visibility": "private",
        "scientific_status": "provisional", "caveat": gc.SELECTION_BIAS_CAVEAT, "label": "exploratory",
        "execution_verified": False, "scientific_eligible": False,
        "dataset": deepcopy(external["dataset"]), "pools_sha256": external["hashes"]["pools_sha256"],
        "metric": external["evaluation"]["primary"], "aggregation": "mean_over_folds_then_seeds",
        "difference": COMPARISON_DIFFERENCE, "inference": "descriptive_only", "hypothesis_test": None,
        "superiority_claim": False, "common_seeds": common,
        "campaigns": {source: {"campaign_sha256": report["hashes"]["campaign_sha256"],
                               "freeze_sha256": report["hashes"]["freeze_sha256"], "report_sha256": report["sha256"],
                               "label": report["label"], "authorized_seeds": list(report["seeds"]["authorized"]),
                               "matrix_complete": report["matrix"]["complete"]}
                      for source, report in sorted(reports.items())},
        "arms": arms, "warnings": warnings}
    return _seal(comparison)


# --------------------------------------------------------------------------
# Article export (stable, versioned, ID-free, atomic and idempotent)
# --------------------------------------------------------------------------

def default_article_dir(campaign, root):
    """``<root>/<dataset>/<freeze_kind>-<campaign_sha256[:12]>``; the root is explicit and kept outside the repo."""
    if root is None or not str(root) or not Path(root).is_absolute():
        _fail("export_conflict", "an explicit absolute article root outside the repository is required")
    base = Path(root)
    dataset_id = campaign["dataset"]["id"]
    gj._safe_name(dataset_id, "dataset id")
    return base / dataset_id / f"{campaign['freeze']['kind']}-{campaign['sha256'][:12]}"


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _csv_cell(value):
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is float:
        return repr(value)
    return str(value)


def _csv_bytes(header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        if len(row) != len(header):
            raise AssertionError("CSV row width differs from its header")
        writer.writerow([_csv_cell(value) for value in row])
    return buffer.getvalue().encode("utf-8")


def _write_files(out_dir, files, overwrite):
    """Atomic per file (temp + fsync + replace); identical bytes are left untouched.

    Conflicts are checked for every file before anything is written.
    """
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    conflicts = []
    for name, data in files.items():
        path = target_dir / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            _fail("export_conflict", f"{name} exists and is not a regular file")
        if path.exists() and path.read_bytes() != data:
            conflicts.append(name)
    if conflicts and not overwrite:
        _fail("export_conflict", f"existing files differ (pass overwrite=True to replace): {conflicts}")
    written, unchanged = [], []
    for name, data in files.items():
        path = target_dir / name
        if path.exists() and path.read_bytes() == data:
            unchanged.append(name)
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}-", dir=target_dir)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        written.append(name)
    return {"out_dir": str(target_dir), "written": written, "unchanged": unchanged,
            "files": {name: gj._bytes_sha256(data) for name, data in files.items()}}


def _check_campaign_of(report, campaign):
    try:
        body = canonical_hash({key: value for key, value in campaign.items() if key != "sha256"})
    except (AttributeError, TypeError, ValueError):
        _fail("report_invalid", "campaign must be a sealed object")
    if campaign.get("sha256") != body or body != report["hashes"]["campaign_sha256"]:
        _fail("report_invalid", "campaign is not the sealed campaign of this report")


def _metric_columns(primary, secondary):
    values = [primary["metric"] if primary else None, primary["value"] if primary else None,
              primary["reason"] if primary else None]
    for name in SECONDARY_METRICS:
        metric = secondary[name] if secondary else None
        values += [metric["value"] if metric else None, metric["reason"] if metric else None]
    return values


def _metrics_by_fold(report):
    header = ["schema_version", "arm", "seed", "fold", "status", "reason", "primary_metric", "primary_value",
              "primary_reason"]
    for name in SECONDARY_METRICS:
        header += [name, f"{name}_reason"]
    header += ["n_samples", "predictions_content_sha256", "config_sha256"]
    rows = [[ARTICLE_SCHEMA_VERSION, c["arm"], c["seed"], c["fold"], c["status"], c["reason"],
             *_metric_columns(c["primary"], c["secondary"]), c["n_samples"], c["predictions_content_sha256"],
             c["config_sha256"]] for c in report["matrix"]["cells"]]
    return _csv_bytes(header, rows)


def _metrics_summary(report):
    header = ["schema_version", "arm", "seed", "aggregation", "metric", "value", "reason", "unit", "n_defined",
              "n_expected", "sd", "ci_low", "ci_high", "ci_reason"]
    interval = _interval(report["hashes"]["analysis_plan_sha256"])
    rows = []

    def add(arm, seed, aggregation, item, with_interval):
        ci = item.get("interval", interval) if with_interval else {"ci_low": None, "ci_high": None, "reason": None}
        rows.append([ARTICLE_SCHEMA_VERSION, arm, seed, aggregation, item["metric"], item["value"], item["reason"],
                     item.get("unit"), item.get("n_defined"), item.get("n_expected"), item.get("sd"),
                     ci["ci_low"], ci["ci_high"], ci["reason"]])

    def pooled(arm, seed, metric, name):
        # Column ``metric`` uses the report key (e.g. average_precision), as in the other rows.
        coverage = metric["coverage"] or {}
        rows.append([ARTICLE_SCHEMA_VERSION, arm, seed, "pooled_E", name, metric["value"],
                     metric["reason"], "samples", None, coverage.get("n_samples"), None, None, None, None])

    for arm in ARMS:
        for row in (r for r in report["per_seed"] if r["arm"] == arm):
            add(arm, row["seed"], "mean_over_folds", row["primary"], True)
            for name in SECONDARY_METRICS:
                add(arm, row["seed"], "mean_over_folds", row["secondary"][name], True)
            pooled(arm, row["seed"], row["pooled_E"]["primary"], row["primary"]["metric"])
            for name in SECONDARY_METRICS:
                pooled(arm, row["seed"], row["pooled_E"]["secondary"][name], name)
        summary = next(r for r in report["per_arm"] if r["arm"] == arm)
        add(arm, "all", "mean_over_folds_then_seeds", summary["primary"], True)
        for name in SECONDARY_METRICS:
            add(arm, "all", "mean_over_folds_then_seeds", summary["secondary"][name], True)
        add(arm, "all", "pooled_E_mean_over_seeds", summary["pooled_E"]["primary"], False)
        for name in SECONDARY_METRICS:
            add(arm, "all", "pooled_E_mean_over_seeds", summary["pooled_E"]["secondary"][name], False)
    metric = report["paired_differences"]["metric"]
    for row in report["paired_differences"]["summary"]:
        add(row["arm"], "all", "paired_difference_vs_none_mean_over_folds_then_seeds", dict(row, metric=metric), True)
    return _csv_bytes(header, rows)


def _telemetry_csv(report):
    header = ["schema_version", "arm", "seed", "fold", "cell_status", *TELEMETRY_FIELDS]
    telemetry = {(t["arm"], t["fold"], t["seed"]): t for t in report["telemetry"]}
    rows = []
    for cell in report["matrix"]["cells"]:
        if cell["status"] == "not_authorized":
            continue
        values = telemetry.get((cell["arm"], cell["fold"], cell["seed"]), {})
        rows.append([ARTICLE_SCHEMA_VERSION, cell["arm"], cell["seed"], cell["fold"], cell["status"],
                     *[values.get(name) for name in TELEMETRY_FIELDS]])
    return _csv_bytes(header, rows)


def _readme(report):
    matrix = report["matrix"]
    lines = [
        f"# Article variables: {report['dataset']['id']} {report['dataset']['version']} "
        f"({report['recipe_source']}, campaign {report['hashes']['campaign_sha256'][:12]})",
        "",
        "Generated by `augment_atlas.global_report.export_article_variables` "
        f"(schema_version {ARTICLE_SCHEMA_VERSION}, report protocol {REPORT_PROTOCOL}). Do not edit by hand: "
        "regenerating from the same report gives the same bytes.",
        "",
        f"Label: {report['label']} ({', '.join(report['label_reasons'])}). Scientific status: provisional; "
        "execution_verified=false; scientific_eligible=false. Descriptive only (no interval, test or multiplicity "
        "correction without a sealed analysis plan).",
        "",
        f"Caveat: {report['caveat']}",
        "",
        f"Matrix: {matrix['succeeded_cells']} of {matrix['declared_cells']} declared cells succeeded "
        f"({matrix['authorized_cells']} authorized by this campaign, {matrix['missing_cells']} missing, "
        f"{matrix['failed_cells']} failed, {matrix['not_authorized_cells']} not authorized).",
        "",
        "No patient, image or group identifiers are stored in these files. `null` marks an undefined or "
        "missing value; its reason column says which.",
        "",
        "| File | Content |",
        "|---|---|",
        "| `hyperparameters.json` | Frozen recipe (origin, values, trainer constants, decision rule); for `hpo_d`, "
        "the HPO candidates, their selection metric on D_select per seed and the selected candidate. |",
        "| `seeds.json` | Final seeds, execution stages (authorized or not), HPO seeds and pools partition seeds. |",
        "| `hashes.json` | Pools, records, freeze, campaign, plan/evidence/search space, configs manifest, code "
        "commit/tree, image, environment, weights, report and the SHA256 of every other exported file. |",
        "| `metrics_by_fold.csv` | One row per declared arm x seed x fold cell: status, primary and secondary "
        "metrics with reasons, sample count and artifact hashes. |",
        "| `metrics_summary.csv` | Mean over folds per seed, mean over folds then seeds (Q5 primary), pooled E "
        "(sensitivity), paired differences vs `none`; SD is descriptive dispersion, CI columns stay null. |",
        "| `telemetry_summary.csv` | Fixed telemetry columns per authorized cell; `null` when no summary. |",
        "| `README.md` | This file. |",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def export_article_variables(report, campaign, out_dir, *, records=None, overwrite=False):
    """Write the article variables of one campaign into ``out_dir`` (see ``ARTICLE_FILES``).

    Stable and versioned (``schema_version``), ID-free (optionally re-scanned
    against ``records``), atomic per file and idempotent: two runs on the same
    report give the same bytes. Existing files with different bytes are an
    ``export_conflict`` unless ``overwrite=True``.
    """
    check_report(report)
    _check_campaign_of(report, campaign)
    base = {"schema_version": ARTICLE_SCHEMA_VERSION, "dataset": deepcopy(report["dataset"]),
            "campaign_sha256": report["hashes"]["campaign_sha256"], "recipe_source": report["recipe_source"],
            "label": report["label"], "scientific_status": "provisional", "execution_verified": False,
            "scientific_eligible": False}
    recipe = report["recipe"]
    if recipe["source"] == "external_fixed":
        origin = {"kind": "external_fixed", "source": deepcopy(recipe["external_source"]),
                  "selection_scope": "no data used for selection; D is final training only"}
    else:
        origin = {"kind": "hpo_d", "plan_sha256": recipe["hpo"]["plan_sha256"],
                  "evidence_sha256": recipe["hpo"]["evidence_sha256"],
                  "search_space_sha256": recipe["hpo"]["search_space_sha256"], "selection_partition": "D_select"}
    hyperparameters = {**base, "artifact": "augment-atlas-article-hyperparameters",
                       "freeze_sha256": recipe["freeze_sha256"], "origin": origin,
                       "frozen_recipe": deepcopy(recipe["recipe"]),
                       "decision": deepcopy(report["evaluation"]["decision"]),
                       "trainer_constants": deepcopy(campaign["runtime"]["trainer_constants"]),
                       "hpo": deepcopy(recipe["hpo"])}
    seeds = {**base, "artifact": "augment-atlas-article-seeds", **deepcopy(report["seeds"]),
             "note": "train_seed of a cell equals its seed; the 10 arms share seeds and folds (paired design)."}
    files = {"hyperparameters.json": _json_bytes(hyperparameters), "seeds.json": _json_bytes(seeds),
             "metrics_by_fold.csv": _metrics_by_fold(report), "metrics_summary.csv": _metrics_summary(report),
             "telemetry_summary.csv": _telemetry_csv(report)}
    files["README.md"] = _readme(report)
    hashes = {**base, "artifact": "augment-atlas-article-hashes",
              "hashes": {**deepcopy(report["hashes"]), "report_sha256": report["sha256"],
                         "freeze_code_tree_sha256": campaign["code_identity"]["freeze_code_tree_sha256"],
                         "q6_policy": campaign["q6_policy"]},
              "files": {name: gj._bytes_sha256(data) for name, data in sorted(files.items())}}
    files["hashes.json"] = _json_bytes(hashes)
    files = {name: files[name] for name in ARTICLE_FILES}
    if records is not None:
        _scan_export(files, records)
    return _write_files(out_dir, files, overwrite)


def _scan_export(files, records):
    for name, data in files.items():
        text = data.decode("utf-8")
        if name.endswith(".json"):
            _scan_identifiers(json.loads(text), records, name)
        elif name.endswith(".csv"):
            _scan_identifiers(list(csv.reader(io.StringIO(text))), records, name)
        else:
            _scan_identifiers(re.split(r"[\s|`(),]+", text), records, name)


def export_comparison(comparison, out_dir, *, records=None, overwrite=False):
    """Write ``campaign_comparison.json`` and ``.csv`` (descriptive, ID-free) into ``out_dir``."""
    check_comparison(comparison)
    header = ["schema_version", "arm", "seed", "fold", "metric", "external_fixed", "external_fixed_reason", "hpo_d",
              "hpo_d_reason", "difference_hpo_d_minus_external_fixed", "reason", "inference"]
    rows = []
    for arm in comparison["arms"]:
        rows.append([ARTICLE_SCHEMA_VERSION, arm["arm"], "all", "all", comparison["metric"],
                     arm["external_fixed"]["value"], arm["external_fixed"]["reason"], arm["hpo_d"]["value"],
                     arm["hpo_d"]["reason"], arm["difference"], arm["reason"], comparison["inference"]])
        for cell in arm["per_cell"]:
            left, right = cell["external_fixed"] or {}, cell["hpo_d"] or {}
            rows.append([ARTICLE_SCHEMA_VERSION, arm["arm"], cell["seed"], cell["fold"], comparison["metric"],
                         left.get("value"), left.get("reason"), right.get("value"), right.get("reason"),
                         cell["difference"], cell["reason"], comparison["inference"]])
    files = {"campaign_comparison.json": _json_bytes(comparison),
             "campaign_comparison.csv": _csv_bytes(header, rows)}
    if records is not None:
        _scan_export(files, records)
    return _write_files(out_dir, files, overwrite)
