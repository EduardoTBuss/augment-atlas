"""Collector of finished global FINAL jobs for ``global_report.build_report`` (spec 6.10, S4).

``build_report`` takes closed entries, one per final job, "assembled by the
collector"; this module is that collector. Offline host logic (stdlib): it
reads the queue snapshot and the attempt directories, never images or Docker.

For every authorized cell (arm x fold x authorized seed) of an AUDITED
campaign the job is located by its recomputed identity (same config bytes,
view, code commit and campaign as ``seed_final``). Then:

* ``succeeded``: completion identity (phase, config, split), checksummed
  ``predictions.json`` and ``results.json`` (the source of
  ``initial_state_sha256``) and the declared clinical evaluation equal to the
  recomputation; the predictions go to the report, which re-validates them.
  ``telemetry/summary.json`` is best effort and outside the completion
  checksums: it is read as data and only its hash is recorded. ``settings``
  adds the worker's full nine-artifact validation;
* ``failed``/``cancelled`` -> ``failed``, ``interrupted`` -> ``interrupted``,
  a timeout recorded by the queue -> ``timeout`` (always with a reason);
* absent or not yet terminal: no entry, so the report shows the cell as
  ``missing`` (``artifact_not_collected``), never as zero.

A final job of this campaign outside the authorized cells, or a retried job
whose earlier attempt already had predictions, is ``collection_invalid``.

Caveat (``global_campaign.SELECTION_BIAS_CAVEAT``): reshuffling folds does not
remove selection bias if patients used to choose hyperparameters fall into a
test set (Bradshaw 2023, DOI 10.1148/ryai.220232); D therefore stays out of
every test.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import global_campaign as gc
from . import global_evaluation as ge
from . import global_hpo as gh
from . import global_jobs as gj
from .contracts import canonical_hash, file_hash
from .queue import validate_record

COLLECTION_PROTOCOL = "global-final-collection-v1"
_TERMINAL = ("succeeded", "failed", "cancelled", "interrupted")


def _fail(code, message):
    raise ValueError(f"{code}: {message}")


def expected_final_jobs(campaign, records, pools, *, freeze, dataset_source, augmentation_presets, evidence=None):
    """{job_id: (arm, fold, seed, cfg, view_doc, spec)} of the authorized cells of an audited campaign."""
    gc.audit_campaign(records, pools, campaign, expected_campaign_sha256=campaign.get("sha256"), freeze=freeze,
                      dataset_source=dataset_source, augmentation_presets=augmentation_presets, evidence=evidence)
    block = campaign["freeze"]
    configs = gj.build_final_configs(
        records, pools, expected_pools_sha256=campaign["pools_sha256"], dataset_source=dataset_source,
        freeze=freeze, freeze_kind=block["kind"], expected_freeze_sha256=block["sha256"],
        augmentation_presets=augmentation_presets, seeds=campaign["matrix"]["authorized_seeds"], evidence=evidence,
        expected_plan_sha256=block["plan_sha256"])
    directory = gj.config_dir(pools["dataset"]["id"], "final")
    views, expected = {}, {}
    for name in sorted(configs):
        cfg = configs[name]
        fold = cfg["global"]["fold"]
        if fold not in views:
            views[fold] = gj.derive_final_view_doc(records, pools, fold, expected_pools_sha256=campaign["pools_sha256"],
                                                   freeze_sha256=block["sha256"])
        spec = gj.final_job_spec(cfg, views[fold], config_path=f"{directory}/{name}",
                                 config_sha256=ge.config_sha256(cfg),
                                 code_commit=campaign["runtime"]["code_commit"], campaign_sha256=campaign["sha256"])
        expected[spec["job_id"]] = (cfg["augmentation"]["arm"], fold, cfg["training"]["seed"], cfg, views[fold], spec)
    return expected


def _read_success(out, cfg, view_doc, records, pools, record, settings):
    out = Path(out)
    if out.is_symlink() or not out.is_dir():
        _fail("global_artifacts_invalid", "attempt directory is missing")
    completed = gh._read_json(out / "completed.json", "completed.json")
    if type(completed) is not dict or "trainer_override" not in completed or completed["trainer_override"] is not None:
        _fail("trainer_override_forbidden", "completed.json records a trainer_override; toy/CPU attempts are never accepted")
    if (completed.get("status") != "completed" or completed.get("phase") != "final"
            or completed.get("config_sha256") != ge.config_sha256(cfg)
            or completed.get("splits_sha256") != view_doc["sha256"]):
        _fail("global_artifacts_invalid", "completion config/split/phase identity mismatch")
    provenance = gh._read_json(out / "runtime.json", "runtime.json")
    mount = provenance.get("data_mount") if type(provenance) is dict else None
    if type(mount) is not dict or mount.get("kind") != "data_root" or mount.get("destination") != "/data" \
            or mount.get("readonly") is not True:
        _fail("global_artifacts_invalid", "runtime.json does not record the read-only data root as /data")
    if settings is not None:
        from . import runtime
        runtime.validate_global_completion(settings, out, record["spec"])
    prediction_path = out / ge.ARTIFACT_NAME
    files = completed.get("artifacts")
    if (type(files) is not dict or prediction_path.is_symlink() or not prediction_path.is_file()
            or files.get(ge.ARTIFACT_NAME) != file_hash(prediction_path)):
        _fail("global_artifacts_invalid", "predictions.json is not covered by the completion checksums")
    results_path = out / "results.json"
    if (results_path.is_symlink() or not results_path.is_file()
            or files.get("results.json") != file_hash(results_path)):
        # initial_state_sha256 (the pairing proof) and the declared metric come from here.
        _fail("global_artifacts_invalid", "results.json is not covered by the completion checksums")
    artifact = gh._read_json(prediction_path, "predictions.json")
    summary = ge.validate_global_predictions(cfg, view_doc, records, artifact, pools=pools)
    results = gh._read_json(results_path, "results.json")
    declared = results.get("clinical_evaluation") if type(results) is dict else None
    recomputed = {"artifact": ge.ARTIFACT_NAME,
                  **{key: summary[key] for key in ("partition", "samples", "primary_metric", "primary_value",
                                                   "primary_reason", "content_sha256")}}
    try:
        same = canonical_hash(declared) == canonical_hash(recomputed)
    except (TypeError, ValueError):
        same = False
    if not same:
        _fail("metric_recompute_mismatch", "results.clinical_evaluation differs from the metric recomputed from the scores")
    state = results.get("initial_state_sha256")
    telemetry_path = out / "telemetry" / "summary.json"
    telemetry = None
    if telemetry_path.is_file() and not telemetry_path.is_symlink():
        telemetry = gh._read_json(telemetry_path, "telemetry/summary.json")
    return artifact, state, telemetry, {"completed_sha256": file_hash(out / "completed.json"),
                                        "predictions_sha256": file_hash(prediction_path),
                                        "results_sha256": file_hash(results_path),
                                        "telemetry_sha256": file_hash(telemetry_path) if telemetry else None}


def collect_final_entries(campaign, queue_snapshot, runs_root, *, records, pools, freeze, dataset_source,
                          augmentation_presets, evidence=None, settings=None):
    """Return (entries for ``build_report``, private collection manifest)."""
    expected = expected_final_jobs(campaign, records, pools, freeze=freeze, dataset_source=dataset_source,
                                   augmentation_presets=augmentation_presets, evidence=evidence)
    by_id = {}
    for record in queue_snapshot:
        gh._guard("collection_invalid", validate_record, record)
        if record["job_id"] in by_id:
            _fail("collection_invalid", "queue snapshot repeats a job")
        by_id[record["job_id"]] = record
    for job_id, record in by_id.items():
        bound = record["spec"].get("global")
        if (record["spec"].get("phase") == "final" and type(bound) is dict
                and bound.get("campaign_sha256") == campaign["sha256"] and job_id not in expected):
            _fail("collection_invalid", "the queue holds a final job of this campaign outside its authorized cells")
    entries, rows = [], []
    for job_id, (arm, fold, seed, cfg, view_doc, spec) in sorted(expected.items(), key=lambda item: item[1][:3]):
        record = by_id.get(job_id)
        state = None if record is None else record["state"]
        if record is not None:
            for key, value in spec.items():
                if key not in record["spec"] or canonical_hash(record["spec"][key]) != canonical_hash(value):
                    _fail("config_identity_mismatch", f"queue job {key} differs from the campaign's final config")
            gh._check_retries(record, runs_root)
        row = {"arm": arm, "fold": fold, "seed": seed, "job_id": job_id, "queue_state": state,
               "attempt_id": None if record is None else record["attempt_id"]}
        if state not in _TERMINAL:
            rows.append({**row, "status": "missing", "reason": "job_missing" if record is None
                         else f"job_not_terminal:{state}"})
            continue
        entry = {"campaign_sha256": campaign["sha256"], "arm": arm, "fold": fold, "seed": seed,
                 "predictions": None, "initial_state_sha256": None, "telemetry_summary": None}
        if state == "succeeded":
            out = Path(runs_root) / job_id / record["attempt_id"]
            artifact, initial, telemetry, shas = _read_success(out, cfg, view_doc, records, pools, record, settings)
            entry.update(status="succeeded", reason=None, predictions=artifact, initial_state_sha256=initial,
                         telemetry_summary=telemetry)
            rows.append({**row, "status": "succeeded", "reason": None, **shas})
        else:
            if state == "interrupted":
                status, reason = "interrupted", "job_interrupted"
            elif gh._timed_out(record):
                status, reason = "timeout", "job_timeout"
            else:
                status, reason = "failed", f"job_{state}"
            entry.update(status=status, reason=reason)
            rows.append({**row, "status": status, "reason": reason})
        entries.append(entry)
    manifest = gj._seal({"schema_version": 1, "protocol": COLLECTION_PROTOCOL, "campaign_sha256": campaign["sha256"],
                         "pools_sha256": campaign["pools_sha256"], "freeze_sha256": campaign["freeze"]["sha256"],
                         "host_validation": "minimal" if settings is None else "full",
                         "authorized_cells": len(expected), "collected": len(entries), "rows": deepcopy(rows)})
    return entries, manifest
