"""Global D/E campaign seal, public hash anchors and chronology audit (slice S3).

Offline host logic only: no images, training, Docker or servers. The private
``campaign.json`` (spec 6.9) seals everything that must be fixed before any
final job exists (decision constants, aggregation, arms/presets, code and image
identity, final configs manifest, analysis plan, label, caveats and the
declared execution matrix). Public anchors on the ``queue`` branch carry only
hashes; ``audit_chronology`` checks the order freeze -> campaign -> final jobs
on the first-parent history of that branch (spec 4.2 and 7.2).

Caveat (mandatory, see ``SELECTION_BIAS_CAVEAT``): reshuffling folds does not
remove selection bias if patients used to choose hyperparameters fall into a
test set (Bradshaw 2023, DOI 10.1148/ryai.220232); that is why D stays out of
every test. Anchors give auditability, not cryptographic proof: whoever
controls the repository could fabricate a history before publishing it, so
``execution_verified`` and ``scientific_eligible`` are always false here.

Structural errors are ``ValueError("<code>: <message>")`` with the stable codes
of section 6.11 of ``docs/global-next-integration.md`` (S3 additions marked).
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import PurePosixPath
import re
import subprocess
import tempfile

from . import global_development as gd
from . import global_jobs as gj
from .contracts import ARMS, canonical_hash
from .queue import (ANCHOR_CAVEAT, ANCHOR_PROTOCOL, QueueError, anchor_path, anchor_slot, parse_record,
                    validate_anchor)
from .schema import TERMINAL_STATES

SELECTION_BIAS_CAVEAT = (
    "Reshuffling folds does not remove selection bias if patients used to choose hyperparameters "
    "fall into a test set (Bradshaw 2023, DOI 10.1148/ryai.220232); D is therefore kept out of every test.")
BRADSHAW_CAVEAT_ID = "bradshaw_2023_shuffle_does_not_remove_selection_bias"
CAMPAIGN_CAVEATS = (BRADSHAW_CAVEAT_ID, "D_never_tested", "E_folds_not_cv_over_full_population",
                    "recipe_transferred_from_small_D_to_larger_pool")
CAMPAIGN_PROTOCOL = "global-final-campaign-v1"
CHRONOLOGY_PROTOCOL = "global-chronology-v1"
SECONDARY_METRICS = ("macro_f1", "balanced_accuracy", "average_precision")
AGGREGATION_V1 = {"primary": "mean_over_folds_then_seeds", "secondary": ["pooled_E_per_seed"]}
BASE_BLOCKERS = ("execution_unverified", "runtime_isolation_unverified", "chronology_unverified")

# Q6 ("same commit for HPO and final", decided 19/10/2026) cannot hold
# literally: final configs embed freeze_sha256, so they are committed after the
# freeze. Maestro interpretation, STILL PENDING Eduardo's confirmation: the
# same CODE identity (tree of src/ + scripts/ + docker/ + pyproject.toml and the
# pinned image), and only files under configs/global/** may change between the
# freeze commit and the final commit. The whole policy lives in
# ``_q6_same_code_identity_v1`` so it can be swapped if Eduardo decides
# differently (spec 6.9 "(S3)" and 12/Q6 mark it pending).
Q6_POLICY = "same_code_identity_v1"
Q6_POLICY_STATUS = "pending_eduardo_confirmation"
CODE_IDENTITY_PATHS = ("src", "scripts", "docker", "pyproject.toml")
Q6_ALLOWED_CHANGE_PREFIX = "configs/global/"

_CAMPAIGN_KEYS = {"schema_version", "protocol", "scientific_status", "dataset", "task", "pools_sha256", "freeze",
                  "evaluation", "arms", "augmentation_presets_sha256", "folds", "seeds", "execution_stages",
                  "matrix", "runtime", "code_identity", "q6_policy", "configs", "analysis_plan_sha256", "label",
                  "label_reasons", "caveats", "sealed_at_utc", "eligibility", "sha256"}
_IDENTITY_KEYS = {"policy", "policy_status", "freeze_commit", "code_commit", "image_id", "changed_paths",
                  "freeze_code_tree_sha256", "final_code_tree_sha256"}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def _fail(code, message):
    raise ValueError(f"{code}: {message}")


def _hex64(value, name, code="campaign_invalid"):
    if type(value) is not str or not _HEX64.fullmatch(value):
        _fail(code, f"{name} must be a lowercase hex64")


def _commit(value, name, code="code_identity_mismatch"):
    if type(value) is not str or not _HEX40.fullmatch(value):
        _fail(code, f"{name} must be a full 40-hex commit")


def _parse_utc(value, name, code="campaign_invalid"):
    gj._utc(value, name, code)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --------------------------------------------------------------------------
# External freeze (spec 6.8): S0 builds/audits, S3 seals (no-clobber) and anchors
# --------------------------------------------------------------------------

def audit_external_freeze(records, pools, document, *, expected_pools_sha256):
    """Reuse of the S0 auditor (``global_jobs.audit_external_freeze``)."""
    return gj.audit_external_freeze(records, pools, document, expected_pools_sha256=expected_pools_sha256)


def seal_external_freeze(records, pools, *, expected_pools_sha256, recipe, source, runtime, final_policy,
                         frozen_at_utc, output=None):
    """Build, audit and optionally persist (no-clobber) an external freeze.

    ``source.derived_from_this_dataset_scores`` must be false
    (``external_recipe_contaminated``); the label is always exploratory.
    Publish ``build_anchor("recipe_freeze", document)`` afterwards (C_freeze).
    """
    document = gj.build_external_freeze(records, pools, expected_pools_sha256=expected_pools_sha256, recipe=recipe,
                                        source=source, runtime=runtime, final_policy=final_policy,
                                        frozen_at_utc=frozen_at_utc)
    if output is not None:
        gd._publish(output, document, lambda value: audit_external_freeze(
            records, pools, value, expected_pools_sha256=expected_pools_sha256))
    return document


# --------------------------------------------------------------------------
# Public anchors (hashes only)
# --------------------------------------------------------------------------

def _freeze_source(document):
    """Recipe source of a sealed freeze: its protocol decides (never a free label)."""
    protocol = document.get("protocol")
    if protocol == gj.EXTERNAL_FREEZE_PROTOCOL:
        return "external_fixed"
    if protocol == gd.FREEZE_PROTOCOL:
        return "hpo_d"
    _fail("anchor_invalid", "recipe_freeze anchors take a sealed hpo_d or external freeze")


def build_anchor(kind, document, *, document_utc=None):
    """Public anchor of a sealed pools, freeze (external or hpo_d) or campaign document."""
    if type(document) is not dict:
        _fail("anchor_invalid", "document must be a sealed object")
    _hex64(document.get("sha256"), "document sha256", "anchor_invalid")
    if kind == "pools":
        if document_utc is None:
            _fail("anchor_invalid", "pools anchors need an explicit document_utc")
        fields = {"pools_sha256": document["sha256"], "freeze_sha256": None, "campaign_sha256": None,
                  "code_commit": None, "label": None, "recipe_source": None, "document_utc": document_utc}
    elif kind == "recipe_freeze":
        fields = {"pools_sha256": document["pools_sha256"], "freeze_sha256": document["sha256"],
                  "campaign_sha256": None, "code_commit": document["runtime"]["commit"], "label": None,
                  "recipe_source": _freeze_source(document),
                  "document_utc": document_utc or document["frozen_at_utc"]}
    elif kind == "campaign":
        fields = {"pools_sha256": document["pools_sha256"], "freeze_sha256": document["freeze"]["sha256"],
                  "campaign_sha256": document["sha256"], "code_commit": document["runtime"]["code_commit"],
                  "label": document["label"], "recipe_source": document["freeze"]["kind"],
                  "document_utc": document_utc or document["sealed_at_utc"]}
    elif kind == "stage_authorization":
        from . import stage_authorization as sa
        # Same closed schema as sa.audit: X6 optional keys are sealed into document_sha256.
        optional = {key for key in ("cross_environment", "seed_request") if key in document}
        if set(document) != sa.KEYS | optional or gj._seal(document) != document:
            _fail("anchor_invalid", "stage authorization must be sealed with the closed schema")
        fields = {"pools_sha256": document["pools_sha256"], "freeze_sha256": document["freeze"]["sha256"],
                  "campaign_sha256": document["campaign_sha256"], "code_commit": document["execution_code_commit"],
                  "label": None, "recipe_source": document["freeze"]["kind"],
                  "document_utc": document_utc or document["sealed_at_utc"]}
    else:
        _fail("anchor_invalid", "unknown anchor kind")
    anchor = {"schema_version": 1, "protocol": ANCHOR_PROTOCOL, "kind": kind,
              "dataset": {"id": document["dataset"]["id"], "version": document["dataset"]["version"]},
              "document_sha256": document["sha256"], "caveat": ANCHOR_CAVEAT, **fields}
    if kind == "stage_authorization":
        anchor["stage"] = document["stage"]
    anchor["sha256"] = canonical_hash(anchor)
    return validate_anchor(anchor)


# --------------------------------------------------------------------------
# Q6 code identity (isolated policy)
# --------------------------------------------------------------------------

def _git(repo, *args):
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        _fail("code_identity_mismatch", f"Git unavailable ({type(error).__name__})")
    return result


def code_tree_sha256(repo, commit):
    """Hash of the Git tree entries [[path, mode, blob], ...] of ``CODE_IDENTITY_PATHS`` at ``commit``."""
    _commit(commit, "commit")
    result = _git(repo, "ls-tree", "-r", "-z", "--full-tree", commit, "--", *CODE_IDENTITY_PATHS)
    if result.returncode:
        _fail("code_identity_mismatch", f"commit {commit[:12]} is not readable in the code repository")
    entries = []
    for item in result.stdout.decode("utf-8").split("\0"):
        if item:
            meta, path = item.split("\t", 1)
            mode, _, blob = meta.split(" ")
            entries.append([path, mode, blob])
    if not entries:
        _fail("code_identity_mismatch", "code identity tree (src/, scripts/, docker/) is empty")
    return canonical_hash(sorted(entries))


def derive_code_identity(repo, *, freeze_commit, code_commit, image_id):
    """Derive the Q6 evidence from the code repository (Git): changed paths and code trees."""
    _commit(freeze_commit, "freeze_commit")
    _commit(code_commit, "code_commit")
    if freeze_commit != code_commit:
        if _git(repo, "merge-base", "--is-ancestor", freeze_commit, code_commit).returncode:
            _fail("code_identity_mismatch", "the freeze commit is not an ancestor of the final code commit")
        diff = _git(repo, "diff", "--name-only", "--no-renames", "-z", freeze_commit, code_commit)
        if diff.returncode:
            _fail("code_identity_mismatch", "cannot diff the freeze and final commits")
        changed = sorted({p for p in diff.stdout.decode("utf-8").split("\0") if p})
    else:
        changed = []
    return {"policy": Q6_POLICY, "policy_status": Q6_POLICY_STATUS, "freeze_commit": freeze_commit,
            "code_commit": code_commit, "image_id": image_id, "changed_paths": changed,
            "freeze_code_tree_sha256": code_tree_sha256(repo, freeze_commit),
            "final_code_tree_sha256": code_tree_sha256(repo, code_commit)}


def _q6_same_code_identity_v1(identity):
    """Policy ``same_code_identity_v1`` (pending Eduardo): replace this function to change Q6."""
    for path in identity["changed_paths"]:
        pure = PurePosixPath(path)
        if (type(path) is not str or not path.startswith(Q6_ALLOWED_CHANGE_PREFIX) or pure.is_absolute()
                or ".." in pure.parts or "\\" in path):
            _fail("code_identity_mismatch",
                  f"only {Q6_ALLOWED_CHANGE_PREFIX}** may change between the freeze and final commits ({path!r})")
    if identity["freeze_code_tree_sha256"] != identity["final_code_tree_sha256"]:
        _fail("code_identity_mismatch", "src/ + scripts/ differ between the freeze and final commits")
    if identity["freeze_commit"] == identity["code_commit"] and identity["changed_paths"]:
        _fail("code_identity_mismatch", "identical commits cannot have changed paths")


Q6_POLICIES = {Q6_POLICY: _q6_same_code_identity_v1}


def audit_code_identity(identity, *, freeze_runtime, code_commit, image_id):
    """Closed Q6 evidence bound to the freeze runtime and the campaign runtime."""
    if type(identity) is not dict or set(identity) != _IDENTITY_KEYS:
        _fail("code_identity_mismatch", "code_identity must have exactly the Q6 evidence keys")
    if identity["policy"] not in Q6_POLICIES or identity["policy_status"] != Q6_POLICY_STATUS:
        _fail("code_identity_mismatch", f"only policy {Q6_POLICY} ({Q6_POLICY_STATUS}) is supported")
    _commit(identity["freeze_commit"], "freeze_commit")
    _commit(identity["code_commit"], "code_commit")
    _hex64(identity["freeze_code_tree_sha256"], "freeze_code_tree_sha256", "code_identity_mismatch")
    _hex64(identity["final_code_tree_sha256"], "final_code_tree_sha256", "code_identity_mismatch")
    changed = identity["changed_paths"]
    if type(changed) is not list or any(type(p) is not str or not p for p in changed) or changed != sorted(set(changed)):
        _fail("code_identity_mismatch", "changed_paths must be a sorted list of unique paths")
    if identity["freeze_commit"] != freeze_runtime["commit"]:
        _fail("code_identity_mismatch", "code_identity.freeze_commit differs from freeze.runtime.commit")
    if identity["code_commit"] != code_commit:
        _fail("code_identity_mismatch", "code_identity.code_commit differs from campaign runtime.code_commit")
    if type(image_id) is not str or not _IMAGE_ID.fullmatch(image_id) or identity["image_id"] != image_id:
        _fail("code_identity_mismatch", "the pinned image must be sha256:<hex64> and equal in runtime and evidence")
    Q6_POLICIES[identity["policy"]](identity)
    return identity


# --------------------------------------------------------------------------
# Campaign seal (spec 6.9)
# --------------------------------------------------------------------------

def _stages(value, seeds):
    if type(value) is not list or not value:
        _fail("campaign_invalid", "execution_stages must be a nonempty list")
    flat, stages = [], []
    for index, stage in enumerate(value, start=1):
        if type(stage) is not dict or set(stage) != {"stage", "seeds"}:
            _fail("campaign_invalid", "each execution stage is exactly {stage, seeds}")
        if stage["stage"] != index or type(stage["stage"]) is not int:
            _fail("campaign_invalid", "execution stages are numbered 1..n in order")
        if (type(stage["seeds"]) is not list or not stage["seeds"]
                or any(type(s) is not int for s in stage["seeds"])):
            _fail("campaign_invalid", "each stage declares a nonempty list of integer seeds")
        flat.extend(stage["seeds"])
        stages.append(stage)
    if flat != list(seeds):
        _fail("campaign_invalid", "execution stages must partition final_policy.seeds in policy order")
    cells = len(ARMS) * 5
    # v1: a campaign authorizes exactly stage 1; later stages stay declared and
    # explicitly missing (Q13 pilot: 1 seed = 50 of 150 cells).
    out = [{"stage": s["stage"], "seeds": list(s["seeds"]), "cells": cells * len(s["seeds"]),
            "status": "authorized" if s["stage"] == 1 else "declared_not_authorized"} for s in stages]
    authorized = out[0]
    missing = [{"stage": s["stage"], "seeds": s["seeds"], "cells": s["cells"],
                "reason": "stage_not_authorized_by_this_campaign"} for s in out[1:]]
    matrix = {"declared_cells": cells * len(seeds), "authorized_stage": 1, "authorized_seeds": authorized["seeds"],
              "authorized_cells": authorized["cells"], "missing_cells": sum(m["cells"] for m in missing),
              "missing": missing, "complete": not missing}
    return out, matrix


def _label(pools, freeze_kind, analysis_plan_sha256, label):
    grouping = pools["grouping"]
    reasons = []
    if grouping["prior_exposure"] != "none_documented":
        reasons.append("prior_exposure_" + grouping["prior_exposure"])
    if grouping["identity_level"] != "patient" or grouping["linkage_status"] != "verified":
        reasons.append("patient_identity_unproven")
    if analysis_plan_sha256 is None:
        reasons.append("analysis_plan_missing")
    if freeze_kind == "external_fixed" and grouping["prior_exposure"] != "none_documented":
        reasons.append("external_freeze_on_exposed_dataset")
    if "inbreast" in pools["dataset"]["id"].lower():
        reasons.append("inbreast_pilot_exploratory")
    if label not in ("exploratory", "confirmatory_candidate"):
        _fail("campaign_label_forbidden", "label must be exploratory or confirmatory_candidate")
    if label == "confirmatory_candidate" and reasons:
        _fail("campaign_label_forbidden", f"label must be exploratory: {', '.join(reasons)}")
    if label == "exploratory" and not reasons:
        reasons.append("declared_exploratory")
    return reasons


def _caveats(caveats):
    if type(caveats) is not list or BRADSHAW_CAVEAT_ID not in caveats:
        _fail("campaign_caveat_missing", f"caveats must include {BRADSHAW_CAVEAT_ID} (Bradshaw 2023, "
                                         "DOI 10.1148/ryai.220232)")
    if caveats != list(CAMPAIGN_CAVEATS):
        _fail("campaign_caveat_missing", "caveats must be exactly the fixed v1 set, in order")
    return list(caveats)


def _evaluation(task, recipe):
    return {"primary": task["endpoint"], "secondary": list(SECONDARY_METRICS),
            "decision": gj._decision(task, recipe["prediction"]), "multilabel": None,
            "undefined_policy": "propagate_no_drop", "aggregation": deepcopy(AGGREGATION_V1)}


def _final_manifest(configs, *, pools_sha256, freeze_sha256):
    """MANIFEST.json of the deterministic final materialization, computed off-tree."""
    with tempfile.TemporaryDirectory(prefix="atlas-campaign-manifest-") as directory:
        return gj.materialize_configs("final", directory, configs, pools_sha256=pools_sha256,
                                      freeze_sha256=freeze_sha256)


def _build_campaign(records, pools, *, expected_pools_sha256, freeze, freeze_kind, expected_freeze_sha256,
                    evidence, search_space_sha256, dataset_source, augmentation_presets, execution_stages,
                    code_commit, image_id, code_identity, analysis_plan_sha256, label, caveats, sealed_at_utc):
    recipe, final_policy, freeze_runtime = gj._freeze_parts(
        records, pools, freeze=freeze, freeze_kind=freeze_kind, expected_pools_sha256=expected_pools_sha256,
        expected_freeze_sha256=expected_freeze_sha256, evidence=evidence,
        expected_plan_sha256=freeze.get("selection_plan_sha256") if freeze_kind == "hpo_d" else None)
    gj._anchored(records, pools, expected_pools_sha256, global_task=True)
    gj._check_evaluation_seed(pools)
    if freeze_kind == "hpo_d":
        _hex64(search_space_sha256, "search_space_sha256")
        freeze_block = {"kind": "hpo_d", "sha256": freeze["sha256"], "plan_sha256": freeze["selection_plan_sha256"],
                        "evidence_sha256": freeze["selection_evidence_sha256"],
                        "search_space_sha256": search_space_sha256}
    else:
        if search_space_sha256 is not None or evidence is not None:
            _fail("campaign_invalid", "external_fixed freezes have no search space or evidence")
        freeze_block = {"kind": "external_fixed", "sha256": freeze["sha256"], "plan_sha256": None,
                        "evidence_sha256": None, "search_space_sha256": None}
    if analysis_plan_sha256 is not None:
        _hex64(analysis_plan_sha256, "analysis_plan_sha256")
    seeds = list(final_policy["seeds"])
    stages, matrix = _stages(execution_stages, seeds)
    _commit(code_commit, "runtime.code_commit")
    identity = audit_code_identity(code_identity, freeze_runtime=freeze_runtime, code_commit=code_commit,
                                   image_id=image_id)
    configs = gj.build_final_configs(
        records, pools, expected_pools_sha256=expected_pools_sha256, dataset_source=dataset_source, freeze=freeze,
        freeze_kind=freeze_kind, expected_freeze_sha256=expected_freeze_sha256,
        augmentation_presets=augmentation_presets, seeds=matrix["authorized_seeds"], evidence=evidence,
        expected_plan_sha256=freeze.get("selection_plan_sha256") if freeze_kind == "hpo_d" else None)
    manifest = _final_manifest(configs, pools_sha256=expected_pools_sha256, freeze_sha256=expected_freeze_sha256)
    if len(manifest["files"]) != len(ARMS) * 5 * len(matrix["authorized_seeds"]):
        _fail("campaign_invalid", "configs count must be arms x 5 folds x authorized seeds")
    reasons = _label(pools, freeze_kind, analysis_plan_sha256, label)
    _parse_utc(sealed_at_utc, "sealed_at_utc")
    document = {
        "schema_version": 1, "protocol": CAMPAIGN_PROTOCOL, "scientific_status": "provisional",
        "dataset": deepcopy(pools["dataset"]), "task": deepcopy(pools["task"]), "pools_sha256": expected_pools_sha256,
        "freeze": freeze_block, "evaluation": _evaluation(pools["task"], recipe), "arms": list(ARMS),
        "augmentation_presets_sha256": final_policy["augmentation_presets_sha256"], "folds": [0, 1, 2, 3, 4],
        "seeds": seeds, "execution_stages": stages, "matrix": matrix,
        "runtime": {"code_commit": code_commit, "image_id": image_id,
                    "environment_sha256": freeze_runtime["environment_sha256"],
                    "imagenet_weights_sha256": freeze_runtime["imagenet_weights_sha256"],
                    "trainer_constants": deepcopy(gj.TRAINER_CONSTANTS_V1)},
        "code_identity": deepcopy(identity), "q6_policy": identity["policy"],
        "configs": {"root": gj.config_dir(pools["dataset"]["id"], "final"), "manifest_sha256": manifest["sha256"],
                    "count": len(manifest["files"])},
        "analysis_plan_sha256": analysis_plan_sha256, "label": label, "label_reasons": reasons,
        "caveats": _caveats(caveats), "sealed_at_utc": sealed_at_utc,
        "eligibility": {"execution_verified": False, "scientific_eligible": False,
                        "blockers": list(BASE_BLOCKERS) + [r for r in reasons if r not in BASE_BLOCKERS]}}
    document["sha256"] = canonical_hash(document)
    return document, manifest


def _queue_final_jobs(queue_records, pools_sha256):
    if isinstance(queue_records, dict):
        queue_records = list(queue_records.values())
    if type(queue_records) is not list:
        _fail("campaign_invalid", "queue_records must be the current queue job list (GitQueue.list_jobs())")
    return [r for r in queue_records if r["spec"].get("phase") == "final" and isinstance(r["spec"].get("global"), dict)
            and r["spec"]["global"].get("pools_sha256") == pools_sha256]


def seal_campaign(records, pools, *, expected_pools_sha256, freeze, freeze_kind, expected_freeze_sha256,
                  dataset_source, augmentation_presets, execution_stages, code_commit, image_id, code_identity,
                  label, caveats, sealed_at_utc, queue_records, analysis_plan_sha256=None, evidence=None,
                  search_space_sha256=None, expected_manifest_sha256=None, output=None):
    """Seal ``campaign.json``; publish ``build_anchor("campaign", doc)`` next (C_campaign).

    Refuses if any final job of these pools already exists in the queue
    (``final_job_before_seal``). ``expected_manifest_sha256`` (the MANIFEST.json
    committed in ``code_commit``) must equal the deterministic recomputation.
    """
    if _queue_final_jobs(queue_records, expected_pools_sha256):
        _fail("final_job_before_seal", "a final job of these pools already exists in the queue")
    document, manifest = _build_campaign(
        records, pools, expected_pools_sha256=expected_pools_sha256, freeze=freeze, freeze_kind=freeze_kind,
        expected_freeze_sha256=expected_freeze_sha256, evidence=evidence, search_space_sha256=search_space_sha256,
        dataset_source=dataset_source, augmentation_presets=augmentation_presets,
        execution_stages=execution_stages, code_commit=code_commit, image_id=image_id,
        code_identity=code_identity, analysis_plan_sha256=analysis_plan_sha256, label=label, caveats=caveats,
        sealed_at_utc=sealed_at_utc)
    if expected_manifest_sha256 is not None and expected_manifest_sha256 != manifest["sha256"]:
        _fail("config_identity_mismatch", "committed final MANIFEST.json differs from the deterministic recomputation")
    if output is not None:
        gd._publish(output, document, lambda value: audit_campaign(
            records, pools, value, expected_campaign_sha256=document["sha256"], freeze=freeze, evidence=evidence,
            dataset_source=dataset_source, augmentation_presets=augmentation_presets))
    return document


def audit_campaign(records, pools, campaign, *, expected_campaign_sha256, freeze, dataset_source,
                   augmentation_presets, evidence=None, expected_manifest_sha256=None, code_repo=None,
                   stage_authorization=None):
    """Re-derive the campaign from its declared inputs; any difference is refused.

    ``code_repo`` (optional) re-derives the Q6 evidence from Git and requires it
    equal to the sealed ``code_identity``.
    """
    _hex64(expected_campaign_sha256, "expected_campaign_sha256", "campaign_anchor_missing")
    if type(campaign) is not dict or set(campaign) != _CAMPAIGN_KEYS:
        _fail("campaign_invalid", "campaign must have exactly the global-final-campaign-v1 keys")
    if campaign["sha256"] != expected_campaign_sha256:
        _fail("campaign_anchor_missing", "campaign document does not match its anchored hash")
    if campaign["sha256"] != canonical_hash({k: v for k, v in campaign.items() if k != "sha256"}):
        _fail("campaign_invalid", "campaign checksum mismatch")
    if (campaign["schema_version"] != 1 or campaign["protocol"] != CAMPAIGN_PROTOCOL
            or campaign["scientific_status"] != "provisional"):
        _fail("campaign_invalid", "only provisional global-final-campaign-v1 is supported")
    eligibility = campaign["eligibility"]
    if (type(eligibility) is not dict or eligibility.get("execution_verified") is not False
            or eligibility.get("scientific_eligible") is not False):
        _fail("campaign_invalid", "execution_verified and scientific_eligible are always false")
    _caveats(campaign["caveats"])
    stages = campaign["execution_stages"]
    if type(stages) is not list or any(type(s) is not dict or "stage" not in s or "seeds" not in s for s in stages):
        _fail("campaign_invalid", "execution_stages malformed")
    freeze_block = campaign["freeze"]
    if type(freeze_block) is not dict or freeze_block.get("kind") not in gj.FREEZE_KINDS:
        _fail("campaign_invalid", "freeze block malformed")
    rebuilt, manifest = _build_campaign(
        records, pools, expected_pools_sha256=campaign["pools_sha256"], freeze=freeze,
        freeze_kind=freeze_block["kind"], expected_freeze_sha256=freeze_block["sha256"], evidence=evidence,
        search_space_sha256=freeze_block.get("search_space_sha256"), dataset_source=dataset_source,
        augmentation_presets=augmentation_presets,
        execution_stages=[{"stage": s["stage"], "seeds": s["seeds"]} for s in stages],
        code_commit=campaign["runtime"]["code_commit"], image_id=campaign["runtime"]["image_id"],
        code_identity=campaign["code_identity"], analysis_plan_sha256=campaign["analysis_plan_sha256"],
        label=campaign["label"], caveats=campaign["caveats"], sealed_at_utc=campaign["sealed_at_utc"])
    if rebuilt["sha256"] != campaign["sha256"]:
        _fail("campaign_invalid", "campaign is not the deterministic seal of its declared inputs")
    if expected_manifest_sha256 is not None and expected_manifest_sha256 != manifest["sha256"]:
        _fail("config_identity_mismatch", "final MANIFEST.json differs from the sealed campaign configs")
    if code_repo is not None:
        derived = derive_code_identity(code_repo, freeze_commit=campaign["code_identity"]["freeze_commit"],
                                       code_commit=campaign["runtime"]["code_commit"],
                                       image_id=campaign["runtime"]["image_id"])
        if derived != campaign["code_identity"]:
            _fail("code_identity_mismatch", "Git-derived Q6 evidence differs from the sealed code_identity")
    if stage_authorization is not None:
        from . import stage_authorization as sa
        sa.audit(stage_authorization, campaign, code_repo=code_repo)
    return {"structural_valid": True, "execution_verified": False, "scientific_eligible": False,
            "campaign_sha256": campaign["sha256"], "label": campaign["label"],
            "blockers": list(eligibility["blockers"]), "matrix": deepcopy(campaign["matrix"]),
            "authorized_seeds": list(campaign["matrix"]["authorized_seeds"]),
            "manifest_sha256": manifest["sha256"], "q6_policy": campaign["q6_policy"],
            "q6_policy_status": Q6_POLICY_STATUS, "caveat": SELECTION_BIAS_CAVEAT}


# --------------------------------------------------------------------------
# Chronology (spec 7.2)
# --------------------------------------------------------------------------

_RAW = re.compile(r":(\d{6}) (\d{6}) ([0-9a-f]{40}) ([0-9a-f]{40}) ([A-Z])\t(.+)")


def _history(queue, head, dataset_id):
    """[(commit, commit_time, {path: (status, blob)})] oldest first, in ONE Git call.

    Only ``jobs/`` and ``anchors/<dataset_id>/`` changes are kept. The queue
    only creates single-parent commits, so any merge (or a second root) makes
    the history non-auditable.
    """
    raw = queue._git_bytes("-c", "core.quotePath=false", "log", "--reverse", "--root", "--raw", "--no-renames",
                           "--no-abbrev", "--format=%x00%H %ct %P", head).decode("utf-8")
    prefix = f"anchors/{dataset_id}/"
    commits = []
    for line in raw.split("\n"):
        if not line:
            continue
        if line.startswith("\0"):
            parts = line[1:].split()
            if len(parts) > 3:
                _fail("chronology_invalid", "queue history must be linear (no merge commits)")
            commit, stamp = parts[0], int(parts[1])
            parent = parts[2] if len(parts) == 3 else None
            if (commits and parent != commits[-1][0]) or (not commits and parent is not None):
                _fail("chronology_invalid", "queue history is not a single first-parent chain")
            commits.append((commit, stamp, {}))
            continue
        match = _RAW.fullmatch(line)
        if not match or not commits:
            _fail("chronology_invalid", "unparseable queue history")
        status, path = match.group(5), match.group(6)
        if path.startswith("jobs/") or path.startswith(prefix):
            commits[-1][2][path] = (status, match.group(4))
    if not commits or commits[-1][0] != head:
        _fail("chronology_invalid", "queue history does not end at the audited HEAD")
    return commits


def _blobs(queue, shas):
    """{sha: bytes} for many blobs in ONE ``git cat-file --batch`` call."""
    shas = sorted(set(shas))
    if not shas:
        return {}
    out = queue._git_bytes("cat-file", "--batch", input=("\n".join(shas) + "\n").encode("ascii"))
    blobs, position = {}, 0
    for sha in shas:
        end = out.index(b"\n", position)
        header = out[position:end].decode("ascii").split()
        if len(header) != 3 or header[0] != sha or header[1] != "blob":
            _fail("chronology_invalid", f"queue object {sha[:12]} is missing or not a blob")
        size = int(header[2])
        blobs[sha] = out[end + 1:end + 1 + size]
        position = end + 2 + size
    return blobs


def _job_record(data):
    # Legacy job records are read with universal newlines (``GitQueue._git`` text mode).
    try:
        return parse_record(data.decode("utf-8").replace("\r\n", "\n"))
    except (UnicodeDecodeError, QueueError) as error:
        _fail("chronology_invalid", f"unreadable job record in the queue history ({error})")


def _stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def audit_chronology(queue, campaign, *, freeze=None, recorded_campaign_commit=None, head=None,
                     stage_authorization=None, code_repo=None):
    """Audit the order freeze -> campaign -> final jobs of ONE campaign on the ``queue`` branch (spec 7.2).

    ``queue`` is a ``GitQueue`` (local checkout + remote). Several campaigns
    may share one dataset/pools (INbreast, 20/10/2026: ``external_fixed`` and
    an exploratory ``hpo_d``); each is identified by its own anchors (slot
    ``(kind, pools_sha256, recipe_source)``) and audited separately:

    1. linear single-parent history up to ``head``; anchor files never change
       or disappear; at most one anchor document per slot;
    2. ``C_freeze`` (first commit with this freeze anchor) strictly before
       ``C_campaign``; ``C_campaign`` equal to the commit recorded at seal time;
    3. no final job of these pools (of ANY campaign) before or at ``C_campaign``;
       each final job first appears after the anchor of the campaign it is
       bound to; this campaign's final jobs match its freeze, code commit and
       authorized stage seeds;
    4. ``hpo_d``: no HPO job of these pools created or changed at/after
       ``C_freeze``, all earlier ones terminal and bound to the sealed
       plan/search space; ``external_fixed``: every HPO job of these pools
       first appears strictly after ``C_freeze`` (the external recipe was
       anchored before any HPO score of the dataset existed);
    5. timestamp inversions are warnings (host clocks), never approval.

    Anchors give auditability, not proof: the flags stay false.
    """
    if type(campaign) is not dict or set(campaign) != _CAMPAIGN_KEYS:
        _fail("campaign_invalid", "campaign must have exactly the global-final-campaign-v1 keys")
    dataset, pools_sha256 = campaign["dataset"], campaign["pools_sha256"]
    dataset_id, source = dataset["id"], campaign["freeze"]["kind"]
    freeze_sha256, campaign_sha256 = campaign["freeze"]["sha256"], campaign["sha256"]
    if freeze is not None and (type(freeze) is not dict or freeze.get("sha256") != freeze_sha256):
        _fail("freeze_not_sealed", "freeze document is not the campaign freeze")
    head = head or queue._fetch_head()
    if head is None:
        _fail("campaign_anchor_missing", "queue branch is empty")
    commits = _history(queue, head, dataset_id)

    # Replay the first-parent history: creation index of anchors, blob sequence of jobs.
    anchor_index, job_versions = {}, {}
    for index, (_, _, changes) in enumerate(commits):
        for path, (status, blob) in sorted(changes.items()):
            if path.startswith("anchors/"):
                if path in anchor_index or status != "A":
                    _fail("anchor_conflict", f"anchor {path} changed or disappeared after creation")
                anchor_index[path] = (index, blob)
            elif status not in ("A", "M"):
                _fail("chronology_invalid", f"job record {path} was deleted or retyped (the queue never does that)")
            else:
                job_versions.setdefault(path, []).append((index, blob))
    first = _blobs(queue, [blob for _, blob in anchor_index.values()]
                   + [blob for versions in job_versions.values() for blob in (versions[0][1], versions[-1][1])])
    anchors = {}
    for path, (index, blob) in anchor_index.items():
        try:
            anchor = validate_anchor(json.loads(first[blob].decode("utf-8")))
        except (UnicodeDecodeError, ValueError) as error:
            _fail("anchor_invalid", f"invalid anchor {path}: {error}")
        if anchor["dataset"]["id"] != dataset_id or path != anchor_path(dataset_id, anchor["kind"],
                                                                          anchor["document_sha256"]):
            _fail("anchor_invalid", f"anchor path does not match its content: {path}")
        anchors[path] = (index, anchor)
    slots = {}
    for path, (_, anchor) in sorted(anchors.items()):
        if slots.setdefault(anchor_slot(anchor), anchor["document_sha256"]) != anchor["document_sha256"]:
            _fail("anchor_conflict", f"two {anchor['kind']} documents for one pools/recipe_source slot ({path})")

    freeze_path = anchor_path(dataset_id, "recipe_freeze", freeze_sha256)
    campaign_path = anchor_path(dataset_id, "campaign", campaign_sha256)
    if freeze_path not in anchors:
        _fail("freeze_anchor_missing", "freeze anchor is not in the queue history")
    if campaign_path not in anchors:
        _fail("campaign_anchor_missing", "campaign anchor is not in the queue history")
    idx_freeze, freeze_anchor = anchors[freeze_path]
    idx_campaign, campaign_anchor = anchors[campaign_path]
    if idx_freeze >= idx_campaign:
        _fail("chronology_invalid", "C_freeze must be a strict ancestor of C_campaign")
    if (freeze_anchor["pools_sha256"] != pools_sha256 or freeze_anchor["recipe_source"] != source
            or freeze_anchor["dataset"] != dataset or campaign_anchor["pools_sha256"] != pools_sha256
            or campaign_anchor["freeze_sha256"] != freeze_sha256 or campaign_anchor["recipe_source"] != source
            or campaign_anchor["label"] != campaign["label"] or campaign_anchor["dataset"] != dataset
            or campaign_anchor["code_commit"] != campaign["runtime"]["code_commit"]):
        _fail("anchor_conflict", "anchors are not bound to this campaign/freeze/pools/recipe source")
    c_freeze, c_campaign = commits[idx_freeze], commits[idx_campaign]
    if recorded_campaign_commit is not None and recorded_campaign_commit != c_campaign[0]:
        _fail("chronology_invalid", "C_campaign differs from the commit recorded at seal time (history rewritten?)")

    campaign_anchors = {anchor["campaign_sha256"]: (index, anchor) for index, anchor in anchors.values()
                        if anchor["kind"] == "campaign"}
    authorized = set(campaign["matrix"]["authorized_seeds"])
    idx_stage = None
    if stage_authorization is not None:
        from . import stage_authorization as sa
        sa.audit(stage_authorization, campaign, code_repo=code_repo)
        stage_path = anchor_path(dataset_id, "stage_authorization", stage_authorization["sha256"])
        if stage_path not in anchors:
            _fail("campaign_anchor_missing", "stage authorization is not anchored")
        idx_stage, stage_anchor = anchors[stage_path]
        if idx_stage <= idx_campaign or stage_anchor != build_anchor("stage_authorization", stage_authorization):
            _fail("campaign_invalid", "stage anchor must follow and bind this campaign")
    finals, others, hpo = [], 0, []
    for path, versions in sorted(job_versions.items()):
        record, latest = _job_record(first[versions[0][1]]), _job_record(first[versions[-1][1]])
        if path != f"jobs/{record['job_id']}.md" or latest["job_id"] != record["job_id"]:
            _fail("chronology_invalid", "job path does not match its structured identity")
        if latest["spec"] != record["spec"]:
            _fail("chronology_invalid", f"job {record['job_id'][:12]} spec was rewritten (specs are immutable)")
        spec = record["spec"]
        bound = spec.get("global") if isinstance(spec.get("global"), dict) else None
        if bound is None or bound.get("pools_sha256") != pools_sha256 or spec.get("phase") not in ("hpo", "final"):
            continue
        created = versions[0][0]
        if spec["phase"] == "final":
            if created <= idx_campaign:
                _fail("final_job_before_seal", f"final job {record['job_id'][:12]} appears at or before C_campaign")
            own = campaign_anchors.get(bound.get("campaign_sha256"))
            if own is None or own[0] >= created or own[1]["freeze_sha256"] != bound.get("freeze_sha256"):
                _fail("campaign_anchor_missing", f"final job {record['job_id'][:12]} is not preceded by the anchor "
                                                 "of the campaign it is bound to")
            if bound["campaign_sha256"] != campaign_sha256:
                others += 1
                continue
            if (bound.get("freeze_kind") != source or spec["recipe_hash"] != freeze_sha256
                    or (spec["dataset"], spec["dataset_version"]) != (dataset_id, dataset["version"])):
                _fail("campaign_anchor_missing", f"final job {record['job_id'][:12]} is not bound to this freeze")
            if stage_authorization is None:
                if spec["code_commit"] != campaign["runtime"]["code_commit"]:
                    _fail("code_identity_mismatch", "final job code_commit differs from the campaign")
                if spec["train_seed"] not in authorized:
                    _fail("campaign_invalid", "seed is outside the authorized stage")
            else:
                sa.job_binding(spec, campaign, stage_authorization, anchor_commit=commits[idx_stage][0])
                if spec["train_seed"] in stage_authorization["seeds"] and created <= idx_stage:
                    _fail("campaign_invalid", "later-stage job appears before or at its authorization anchor")
            finals.append((path, versions))
        elif source == "hpo_d":
            if created >= idx_freeze or any(index >= idx_freeze for index, _ in versions):
                _fail("hpo_job_after_freeze", f"HPO job {record['job_id'][:12]} created/changed at or after C_freeze")
            if (bound.get("plan_sha256") != campaign["freeze"]["plan_sha256"]
                    or bound.get("search_space_sha256") != campaign["freeze"]["search_space_sha256"]):
                _fail("search_space_mismatch", f"HPO job {record['job_id'][:12]} belongs to another plan/search space")
            before = [blob for index, blob in versions if index < idx_freeze]
            hpo.append((record["job_id"], before[-1]))
        else:
            if created <= idx_freeze:
                _fail("external_recipe_contaminated", f"HPO job {record['job_id'][:12]} of these pools exists at or "
                                                      "before the external C_freeze")
            hpo.append((record["job_id"], None))
    late = _blobs(queue, [blob for _, blob in hpo if blob])
    for job_id, blob in hpo:
        if blob and _job_record(late[blob])["state"] not in TERMINAL_STATES:
            _fail("hpo_job_after_freeze", f"HPO job {job_id[:12]} was not terminal before C_freeze")
    claims = sorted(event["at"] for _, versions in finals
                    for event in _job_record(first[versions[-1][1]])["history"] if event.get("kind") == "claim")

    timeline = []
    if freeze is not None:
        _parse_utc(freeze["frozen_at_utc"], "frozen_at_utc", "freeze_not_sealed")
        timeline.append(("freeze.frozen_at_utc", _stamp(freeze["frozen_at_utc"])))
    timeline.append(("C_freeze.commit_time", c_freeze[1]))
    timeline.append(("campaign.sealed_at_utc", _parse_utc(campaign["sealed_at_utc"], "sealed_at_utc").timestamp()))
    timeline.append(("C_campaign.commit_time", c_campaign[1]))
    if claims:
        timeline.append(("first_final_claim", _stamp(claims[0])))
    warnings = []
    for (left, a), (right, b) in zip(timeline, timeline[1:]):
        # Git commit times have 1 s resolution; tolerate sub-second truncation.
        if a > b + 1:
            warnings.append(f"timestamp_inversion: {left} > {right} (host clocks; supporting evidence only)")
    report = {"protocol": CHRONOLOGY_PROTOCOL, "dataset": deepcopy(dataset), "pools_sha256": pools_sha256,
            "recipe_source": source, "freeze_sha256": freeze_sha256, "campaign_sha256": campaign_sha256,
            "head": head, "C_freeze": c_freeze[0], "C_campaign": c_campaign[0], "commits": len(commits),
            "hpo_jobs": len(hpo), "final_jobs": len(finals), "other_campaign_final_jobs": others,
            "first_final_claim_utc": claims[0] if claims else None, "warnings": warnings,
            "structural_valid": True, "execution_verified": False, "scientific_eligible": False,
            "caveat": SELECTION_BIAS_CAVEAT}
    if stage_authorization is not None:
        # Primary snapshot is the parent of C_authorization. Its historic report bytes stay valid.
        primary_head = commits[idx_stage - 1][0]
        report["stage1_head"] = primary_head
        report["stage1_chronology"] = audit_chronology(queue, campaign, freeze=freeze,
            recorded_campaign_commit=recorded_campaign_commit, head=primary_head)
        report["stage_authorization_sha256"] = stage_authorization["sha256"]
        report["C_stage_authorization"] = commits[idx_stage][0]
    return report
