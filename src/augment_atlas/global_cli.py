"""Operator CLI of the global D/E protocol (spec section 13).

Thin glue over the audited library: every subcommand loads its inputs from the
private store ``<data-root>/global/<dataset>/<kind>-<sha256>.json`` (the
layout the worker reads, spec section 5), pinned by hash on the command line,
re-audits them, and writes new documents with the no-clobber publisher.
Standard output carries only hashes, counts, commits and private file paths;
it is scanned for record IDs and image paths before printing.

Covered: records/pools identity, the D/E pools generator ``global-dev-eval-v1``
(``make-pools``, binary tasks), the D-only hpo-view and D-view; the external
fixed recipe freeze (campaign (a)); the HPO path (search space, plan, HPO
configs, seeding, evidence collection, recipe freeze; campaign (b)); final
config materialization, campaign seal, public anchors, ``seed_final``,
chronology, collection of finished final jobs, the private report, the public
summary, the article export and the two-campaign comparison.

HPO commit identity (decision B1, Eduardo 22/10/2026): the plan is built at a
clean commit P (``plan.runtime.commit``); the HPO configs embed ``plan_sha256``
and are therefore committed later, at X. ``seed-hpo`` and ``collect-hpo``
require Q6 evidence from P to X (``same_code_identity_v1``: ancestor, only
``configs/global/**`` changed, same code tree, pinned image).

Nothing here trains, opens images beyond hashing, or talks to Docker.

Caveat (``global_campaign.SELECTION_BIAS_CAVEAT``): reshuffling folds does not
remove selection bias if patients used to choose hyperparameters fall into a
test set (Bradshaw 2023, DOI 10.1148/ryai.220232); D therefore stays out of
every test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from . import global_campaign as gc
from . import global_collect as gcol
from . import global_development as gd
from . import global_hpo as gh
from . import global_jobs as gj
from . import global_report as gr
from . import global_splits as gs
from . import legacy_pools as lp
from .contracts import ARMS, canonical_hash, file_hash
from .data import load_records
from .queue import GitQueue, QueueError

PRESETS_DIR = "configs/smoke/candidate-v2"
FREEZE_PREFIX = {"external_fixed": "external-freeze", "hpo_d": "freeze"}
# Fixed part of final_policy (validated again by global_development._final_policy).
FINAL_POLICY_FIXED = {
    "initialization": "fresh_imagenet_each_arm_fold_seed", "warm_start": False,
    "training_pool": "D_union_E_minus_test", "folds": [0, 1, 2, 3, 4], "arms": list(ARMS),
    "scheduler_duration": "fixed_epochs_recompute_steps_from_final_pool",
    "step_count": {"unit": "optimizer_update", "drop_last": False, "accumulation_tail": "step",
                   "per_epoch": "ceil(ceil(train_images/batch_size)/accumulation_steps)"},
    "early_stopping": False, "checkpoint_selection": "last_epoch", "selection_on_E": False}


_LOADED_RECORDS = []  # records of this invocation, for redacting error text


def _fail(code, message):
    raise ValueError(f"{code}: {message}")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def _read(path, what):
    try:
        return gd._read(Path(path))
    except FileNotFoundError:
        _fail("cli_input_missing", f"{what} not found")
    except (OSError, ValueError, RecursionError) as error:
        _fail("cli_input_invalid", f"{what} unreadable ({type(error).__name__})")


def _private(args, kind, sha):
    gj._hex64(sha, f"{kind} sha256")
    return Path(args.data_root).resolve().joinpath(*gj._private_name(kind, args.dataset, sha).split("/"))


def _document(args, kind, sha):
    """A sealed private document whose own ``sha256`` equals the pin."""
    document = _read(_private(args, kind, sha), f"private {kind} document")
    if type(document) is not dict or document.get("sha256") != sha:
        _fail("cli_input_invalid", f"private {kind} document does not carry the pinned sha256")
    return document


def _source(args):
    source = _read(args.source, "dataset source")
    if type(source) is not dict or set(source) != {"adapter", "manifest", "manifest_sha256"}:
        _fail("cli_input_invalid", "--source must be exactly {adapter, manifest, manifest_sha256}")
    return source


def _records(args, num_classes):
    """Host records exactly as the worker derives them (runtime.load_global_trusted_inputs)."""
    source = _source(args)
    dataset = {"id": args.dataset, **source, "require_image_hashes": True}
    loaded = load_records({"dataset": dataset, "model": {"num_classes": num_classes}}, check_files=True,
                          root=Path(args.data_root).resolve())
    records = sorted(({k: row[k] for k in ("id", "path", "group", "label")} for row in loaded),
                     key=lambda row: row["id"])
    _LOADED_RECORDS[:] = records
    return source, records, {row["path"]: row["sha256"] for row in loaded}


def _pools(args):
    pools = _document(args, "pools", args.pools_sha256)
    if pools.get("dataset", {}).get("id") != args.dataset:
        _fail("cli_input_invalid", "pools belong to another dataset")
    source, records, images = _records(args, len(pools["task"]["classes"]))
    return source, records, images, pools


def _git(code, *argv):
    result = subprocess.run(["git", "-C", str(code), *argv], capture_output=True, text=True, timeout=60)
    if result.returncode:
        _fail("code_identity_mismatch", f"git {argv[0]} failed in the code checkout")
    return result.stdout.strip()


def _clean_head(code):
    if _git(code, "status", "--porcelain"):
        _fail("code_identity_mismatch", "the code checkout must be clean (commit the materialized configs first)")
    return _git(code, "rev-parse", "HEAD")


def _presets(code):
    presets = {}
    for arm in ARMS:
        cfg = _read(Path(code).resolve() / PRESETS_DIR / f"{arm}.json", f"preset {arm}")
        presets[arm] = cfg["augmentation"]
    return presets


def _final_policy(code, seeds):
    return {**json.loads(json.dumps(FINAL_POLICY_FIXED)), "seeds": list(seeds),
            "augmentation_presets_sha256": canonical_hash(_presets(code))}


def _utc(text, name):
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        moment = None
    if moment is None or moment.utcoffset() != timezone.utc.utcoffset(moment):
        _fail("cli_input_invalid", f"{name} must be an explicit UTC ISO timestamp")
    return text


def _seeds(text):
    try:
        return [int(part) for part in text.split(",")]
    except ValueError:
        _fail("cli_input_invalid", "seeds must be comma-separated integers")


def _stages(text):
    """``11;13,19`` -> [{stage 1: [11]}, {stage 2: [13, 19]}]."""
    return [{"stage": index, "seeds": _seeds(part)} for index, part in enumerate(text.split(";"), start=1)]


def _freeze(args, records, pools):
    freeze = _document(args, FREEZE_PREFIX[args.freeze_kind], args.freeze_sha256)
    evidence = None
    if args.freeze_kind == "hpo_d":
        evidence = _document(args, "evidence", freeze.get("selection_evidence_sha256"))
    gj._freeze_parts(records, pools, freeze=freeze, freeze_kind=args.freeze_kind,
                     expected_pools_sha256=pools["sha256"], expected_freeze_sha256=args.freeze_sha256,
                     evidence=evidence,
                     expected_plan_sha256=freeze.get("selection_plan_sha256") if args.freeze_kind == "hpo_d" else None)
    return freeze, evidence


def _queue(args):
    return GitQueue(args.checkout, args.remote, args.branch)


def _committed_manifest(code, dataset):
    path = Path(code).resolve() / gj.config_dir(dataset, "final") / "MANIFEST.json"
    return _read(path, "committed final MANIFEST.json")


def _campaign(args):
    """Campaign + its freeze/evidence/records; audited against the committed code checkout."""
    source, records, images, pools = _pools(args)
    campaign = _document(args, "campaign", args.campaign_sha256)
    if campaign.get("pools_sha256") != pools["sha256"]:
        _fail("campaign_invalid", "campaign is bound to other pools")
    args.freeze_kind, args.freeze_sha256 = campaign["freeze"]["kind"], campaign["freeze"]["sha256"]
    freeze, evidence = _freeze(args, records, pools)
    presets = _presets(args.code)
    gc.audit_campaign(records, pools, campaign, expected_campaign_sha256=args.campaign_sha256, freeze=freeze,
                      dataset_source=source, augmentation_presets=presets, evidence=evidence,
                      expected_manifest_sha256=_committed_manifest(args.code, args.dataset).get("sha256"),
                      code_repo=Path(args.code).resolve())
    return source, records, pools, campaign, freeze, evidence, presets


def _plan(args):
    """The anchored plan (it has no own ``sha256`` field: its pin is ``canonical_hash(plan)``)."""
    plan = _read(_private(args, "plan", args.plan_sha256), "private plan document")
    if type(plan) is not dict or gh.plan_hash(plan) != args.plan_sha256:
        _fail("cli_input_invalid", "private plan document does not hash to the pinned plan_sha256")
    return plan


def _hpo_inputs(args):
    """Pools, records, images, search space, plan and hpo-view of one HPO search, re-audited."""
    source, records, images, pools = _pools(args)
    space = _document(args, "search-space", args.search_space_sha256)
    plan = _plan(args)
    view = _document(args, "hpo-view", args.view_sha256)
    gh.validate_search_space(space, pools=pools, runtime=plan.get("runtime"))
    gh.audit_plan_from_search_space(plan, space)
    gh.audit_hpo_view(view, records, pools, expected_pools_sha256=pools["sha256"], image_sha256=images)
    if plan["producer"]["manifest_sha256"] != view["view_files_sha256"]:
        _fail("view_hash_mismatch", "plan.producer.manifest_sha256 is not the view_files_sha256 of this D-view")
    return source, records, images, pools, space, plan, view


def _runtime_decl(args):
    """{commit: clean HEAD, environment_sha256, imagenet_weights_sha256} of the pinned code and image."""
    probe = _read(args.environment_probe, "environment probe")
    return {"commit": _clean_head(args.code), "environment_sha256": gh.environment_sha256(args.image_id, probe),
            "imagenet_weights_sha256": args.weights_sha256}


def _check_environment(args, runtime):
    """Decision Q6 (same immutable image): the image + probe must reproduce the recorded environment."""
    probe = _read(args.environment_probe, "environment probe")
    if gh.environment_sha256(args.image_id, probe) != runtime["environment_sha256"]:
        _fail("code_identity_mismatch", "--image-id/--environment-probe do not reproduce the environment_sha256 "
                                        "recorded by the plan/freeze (another image or probe)")


def _hpo_identity(args, plan, head=None):
    """Q6 evidence from plan.runtime.commit to the commit holding the HPO configs (decision B1).

    ``seed-hpo`` uses the clean HEAD; ``collect-hpo`` names the configs commit that seed-hpo printed.
    """
    head = head or _clean_head(args.code)
    identity = gc.derive_code_identity(Path(args.code).resolve(), freeze_commit=plan["runtime"]["commit"],
                                       code_commit=head, image_id=args.image_id)
    return gc.audit_code_identity(identity, freeze_runtime=plan["runtime"], code_commit=head, image_id=args.image_id)


def _hpo_configs(args, source, records, pools, space, plan, view):
    configs = gj.build_hpo_configs(records, pools, expected_pools_sha256=pools["sha256"], dataset_source=source,
                                   plan=plan, expected_plan_sha256=args.plan_sha256,
                                   search_space_sha256=space["sha256"], hpo_view_sha256=view["sha256"])
    return configs, gj.config_dir(args.dataset, "hpo")


def _anchors(queue, dataset):
    from .queue import validate_anchor
    head = queue._fetch_head()
    return [validate_anchor(json.loads(queue.read_path(head, path))) for path in queue.list_anchors(head, dataset)]


def _receipt(args, kind, anchor, published):
    receipt = {"schema_version": 1, "kind": kind, "anchor_sha256": anchor["sha256"],
               "document_sha256": anchor["document_sha256"], "path": published["path"],
               "commit": published["commit"]}
    receipt["sha256"] = canonical_hash(receipt)
    gd._publish(_private(args, f"anchor-{kind}", anchor["document_sha256"]), receipt, lambda value: None)
    return receipt


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _emit(value, records):
    """Print JSON after two scans: exact IDs/groups/paths anywhere, and ID/path substrings."""
    gr._scan_identifiers(value, records, "CLI output")
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    for row in records:
        if row["id"] in text or row["path"] in text:
            _fail("identifier_leak", "refusing to print a record ID or image path")
    sys.stdout.write(text + "\n")
    return 0


def _redact(message):
    """Library error text may name a file of the view; keep only the code if it holds an identifier."""
    for row in _LOADED_RECORDS:
        if any(type(v) is str and v in message for v in (row["id"], row["path"], row["group"])):
            return message.split(":", 1)[0] + ": (message withheld: it names a record identifier)"
    return message


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_identity(args):
    if args.pools_sha256:
        _, records, _, pools = _pools(args)
        extra = {"global_manifest_sha256": gd.global_manifest_sha256(records, pools),
                 "pools_records_match": pools.get("records_sha256") == canonical_hash(records)}
    else:
        _, records, _ = _records(args, args.num_classes)
        extra = {}
    return _emit({"records": len(records), "records_sha256": canonical_hash(records), **extra}, records)


def _acquisition_spec(args, amendment):
    """Spec recomputed from a verified acquisition root bound to the amendment's config identity."""
    root = Path(args.acquisition_root).resolve()
    ready = _read(root / "manifest" / "READY.json", "READY.json")
    private_path = root / "manifest" / "manifest-private.json"
    if type(ready) is not dict or ready.get("files", {}).get("manifest-private.json") != file_hash(private_path):
        _fail("split_spec_invalid", "manifest-private.json differs from the hash READY.json records")
    if type(amendment) is not dict or type(amendment.get("amends")) is not dict or (
            ready.get("config_identity_sha256") != amendment["amends"].get("config_identity_sha256")):
        _fail("split_spec_invalid", "the acquisition was verified under another config identity than the amendment's")
    if amendment.get("dataset") != args.dataset:
        _fail("split_spec_invalid", "the split amendment belongs to another dataset")
    spec = gs.spec_from_acquisition(ready, _read(private_path, "manifest-private.json"), amendment,
                                    dataset_id=args.dataset)
    return ready, spec


def _tracked(args, path, what):
    """(document, sha256 of its bytes) as committed at HEAD of --code AND identical to origin/main.

    Read with ``git show HEAD:<path>`` (not from disk: skip-worktree edits are invisible to
    ``git status``), and the blob must equal the one on ``origin/main`` (run ``git fetch origin``
    first), so the pre-registered bytes are the reviewed, merged ones rather than a local or
    feature-branch commit. Local refs are operator-controlled: this guards against mistakes;
    the anchored pools record the seeds, so a reviewer can compare them with the public file.
    """
    import hashlib
    root = Path(args.code).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root):
        _fail("split_spec_invalid", f"{what} must be the committed file inside --code")
    relative = target.relative_to(root).as_posix()
    blobs = [subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "-q", f"{ref}:{relative}"],
                            capture_output=True, text=True, timeout=60).stdout.strip() for ref in ("HEAD", "origin/main")]
    if not blobs[0] or blobs[0] != blobs[1]:
        _fail("split_spec_invalid", f"{what} at HEAD is not the published version on origin/main "
                                    "(git fetch origin; use the merged pre-registration)")
    shown = subprocess.run(["git", "-C", str(root), "show", "HEAD:" + relative],
                           capture_output=True, timeout=60)
    if shown.returncode:
        _fail("split_spec_invalid", f"{what} is not committed at HEAD of the code repository")
    try:
        return json.loads(shown.stdout.decode("utf-8")), hashlib.sha256(shown.stdout).hexdigest()
    except (UnicodeDecodeError, ValueError):
        _fail("split_spec_invalid", f"{what} is not valid JSON at HEAD")


def cmd_split_spec(args):
    """Split spec from a verified acquisition root (READY + manifest-private), hashes only."""
    amendment = _read(args.split_amendment, "split amendment")
    ready, spec = _acquisition_spec(args, amendment)
    out = Path(args.out)
    if out.exists():
        _fail("cli_input_invalid", "--out exists; split specs are never overwritten")
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _emit({"out": str(out), "spec_sha256": canonical_hash(spec), "seeds": spec["seeds"],
                  "dataset": spec["dataset"], "manifest_csv_sha256": ready["files"].get("manifest.csv")}, [])


def cmd_make_pools(args):
    spec = _read(args.spec, "split spec")
    _clean_head(args.code)  # the amendment and the config are read as committed, unmodified files
    config, config_sha = _tracked(args, args.acquisition_config, "acquisition config")
    amendment, amendment_sha = _tracked(args, args.split_amendment, "split amendment")
    if type(spec) is not dict or type(spec.get("dataset")) is not dict or spec["dataset"].get("id") != args.dataset:
        _fail("cli_input_invalid", "split spec belongs to another dataset")
    if type(config) is not dict:
        _fail("cli_input_invalid", "acquisition config must be a JSON object")
    gs.check_amendment(spec, amendment, config)
    ready, expected = _acquisition_spec(args, amendment)
    if canonical_hash(spec) != canonical_hash(expected):
        _fail("split_spec_invalid", "the spec differs from the one derived from the verified acquisition root")
    source = _source(args)
    manifest = (Path(args.data_root).resolve() / source["manifest"]).resolve()
    if (source["manifest_sha256"] != ready["files"].get("manifest.csv")
            or manifest != (Path(args.acquisition_root).resolve() / "manifest" / "manifest.csv")):
        _fail("split_spec_invalid", "--source is not the manifest.csv that READY.json verified")
    _, records, _ = _records(args, len(spec["task"]["classes"]))
    candidate, report = gs.generate_dev_eval_pools(records, spec)
    path = _private(args, "pools-candidate", report["candidate_sha256"])
    gd._publish(path, candidate, lambda value: None)
    return _emit({**report, "path": str(path), "seeds": spec["seeds"],
                  "spec_file_sha256": file_hash(Path(args.spec)),
                  "acquisition_config_sha256": config_sha, "split_amendment_sha256": amendment_sha,
                  "code_commit": _git(args.code, "rev-parse", "HEAD"),
                  "next": "approve the hashes, then import-pools --file <path> --expected-manifest "
                          "<global_manifest_sha256>"}, records)


def cmd_convert_legacy_pools(args):
    if args.dataset != "inbreast":
        _fail("legacy_conversion_invalid", "conversion is restricted to the INbreast 16/10 pools")
    if _source(args)["adapter"] != "inbreast_csv":
        _fail("legacy_conversion_invalid", "conversion requires the inbreast_csv manifest adapter")
    try:
        source, records, _ = _records(args, 2)
    except ValueError:
        _fail("legacy_conversion_invalid", "manifest validation failed (message withheld: it may name a record identifier)")
    try:
        legacy_bytes = Path(args.legacy).read_bytes()
    except OSError:
        _fail("cli_input_missing", "legacy pools file unreadable")
    candidate, audit = lp.convert_legacy_pools(records, legacy_bytes, source=source)
    path = Path(args.out)
    gd._publish(path, candidate, lambda value: gd.audit_global_pools(
        records, gd._signed(value), expected_manifest_sha256=audit["manifest_sha256"]))
    return _emit({"path": str(path), "candidate_sha256": audit["pools_sha256"],
                  "global_manifest_sha256": audit["manifest_sha256"], "seeds": candidate["partition_provenance"]["seeds"],
                  "converted_from": candidate["partition_provenance"]["converted_from"],
                  "diagnostics": audit["diagnostics"], "eligibility": audit["eligibility"]}, records)


def cmd_import_pools(args):
    candidate = _read(args.file, "pools candidate")
    if type(candidate) is not dict or candidate.get("dataset", {}).get("id") != args.dataset:
        _fail("cli_input_invalid", "pools candidate belongs to another dataset")
    _, records, _ = _records(args, len(candidate["task"]["classes"]))
    sha = gd._signed(candidate)["sha256"]
    path = _private(args, "pools", sha)
    pools = gd.seal_global_pools(records, candidate, expected_manifest_sha256=args.expected_manifest, output=path)
    audit = gd.audit_global_pools(records, pools, expected_manifest_sha256=args.expected_manifest)
    return _emit({"pools_sha256": pools["sha256"], "path": str(path), "eligibility": audit["eligibility"]}, records)


def cmd_audit_pools(args):
    _, records, _, pools = _pools(args)
    return _emit(gd.audit_global_pools(records, pools, expected_manifest_sha256=args.expected_manifest), records)


def cmd_hpo_view(args):
    _, records, images, pools = _pools(args)
    view = gh.derive_hpo_view(records, pools, expected_pools_sha256=pools["sha256"], image_sha256=images)
    path = Path(args.data_root).resolve().joinpath(*gh.hpo_view_file(view).split("/"))
    gd._publish(path, view, lambda value: gh.audit_hpo_view(value, records, pools,
                                                             expected_pools_sha256=pools["sha256"],
                                                             image_sha256=images))
    return _emit({"hpo_view_sha256": view["sha256"], "view_files_sha256": view["view_files_sha256"],
                  "path": str(path)}, records)


def cmd_d_view(args):
    _, records, images, pools = _pools(args)
    view = _document(args, "hpo-view", args.view_sha256)
    gh.audit_hpo_view(view, records, pools, expected_pools_sha256=pools["sha256"], image_sha256=images)
    result = gh.materialize_d_view(Path(args.data_root).resolve(), Path(args.views_root).resolve(), view,
                                   records=records, pools=pools, image_sha256=images)
    return _emit({"dest": result["dest"], "files": result["files"], "methods": result["methods"],
                  "view_files_sha256": view["view_files_sha256"]}, records)


def cmd_seal_external(args):
    _, records, _, pools = _pools(args)
    probe = _read(args.environment_probe, "environment probe")
    runtime = {"commit": _clean_head(args.code), "environment_sha256": gh.environment_sha256(args.image_id, probe),
               "imagenet_weights_sha256": args.weights_sha256}
    recipe = gj.external_recipe_q2(pools, runtime, color=args.color)
    common = dict(expected_pools_sha256=pools["sha256"], recipe=recipe, source=gj.Q2_EXTERNAL_SOURCE,
                  runtime=runtime, final_policy=_final_policy(args.code, _seeds(args.final_seeds)),
                  frozen_at_utc=_utc(args.frozen_at, "--frozen-at"))
    document = gc.seal_external_freeze(records, pools, **common)
    path = _private(args, "external-freeze", document["sha256"])
    gc.seal_external_freeze(records, pools, **common, output=path)
    return _emit({"freeze_sha256": document["sha256"], "freeze_kind": "external_fixed", "path": str(path),
                  "runtime_commit": runtime["commit"]}, records)


def cmd_search_space(args):
    _, records, _, pools = _pools(args)
    spec = _read(args.spec, "search space spec")
    keys = {"grid", "default", "seeds", "budget", "failure_policy"}
    if type(spec) is not dict or set(spec) != keys:
        _fail("cli_input_invalid", f"search space spec must be exactly {sorted(keys)}")
    base = gj.external_recipe_q2(pools, _runtime_decl(args), color=args.color)  # c00 = the Q2 default (Q7)
    space = gh.build_search_space(pools, base_recipe=base, grid=spec["grid"], default=spec["default"],
                                  seeds=spec["seeds"], budget=spec["budget"], failure_policy=spec["failure_policy"])
    path = _private(args, "search-space", space["sha256"])
    gd._publish(path, space, lambda value: gh.validate_search_space(value, pools=pools))
    return _emit({"search_space_sha256": space["sha256"], "path": str(path),
                  "candidates": len(gh.expand_candidates(space)), "seeds": space["seeds"]}, records)


def cmd_build_plan(args):
    _, records, images, pools = _pools(args)
    space = _document(args, "search-space", args.search_space_sha256)
    view = _document(args, "hpo-view", args.view_sha256)
    gh.audit_hpo_view(view, records, pools, expected_pools_sha256=pools["sha256"], image_sha256=images)
    runtime = _runtime_decl(args)
    producer = gh.make_producer(code_sha256=gh.code_tree_sha256(Path(args.code).resolve() / "src"),
                                environment_sha256=runtime["environment_sha256"],
                                view_files_sha256=view["view_files_sha256"])
    plan = gh.build_selection_plan(space, pools, producer=producer, runtime=runtime,
                                   final_policy=_final_policy(args.code, _seeds(args.final_seeds)))
    sha = gh.plan_hash(plan)
    path = _private(args, "plan", sha)
    gd._publish(path, plan, lambda value: gh.audit_plan_from_search_space(value, space))
    return _emit({"plan_sha256": sha, "path": str(path), "plan_commit": runtime["commit"],
                  "candidates": len(plan["candidates"]), "seeds": plan["seeds"],
                  "next": "materialize-hpo, commit configs/global/<ds>/hpo/, then seed-hpo on that commit"}, records)


def cmd_materialize_hpo(args):
    source, records, _, pools, space, plan, view = _hpo_inputs(args)
    configs, directory = _hpo_configs(args, source, records, pools, space, plan, view)
    manifest = gj.materialize_configs("hpo", Path(args.code).resolve(), configs, pools_sha256=pools["sha256"],
                                      plan_sha256=args.plan_sha256)
    return _emit({"manifest_sha256": manifest["sha256"], "configs": len(manifest["files"]), "directory": directory,
                  "plan_commit": plan["runtime"]["commit"],
                  "next": "commit the directory (only configs/global/** may change), then seed-hpo"}, records)


def cmd_seed_hpo(args):
    source, records, _, pools, space, plan, view = _hpo_inputs(args)
    _check_environment(args, plan["runtime"])
    identity = _hpo_identity(args, plan)
    queue = _queue(args)
    for record in queue.list_jobs():
        bound = record["spec"].get("global")
        if (record["spec"].get("phase") == "hpo" and type(bound) is dict and bound.get("pools_sha256") == pools["sha256"]
                and record["spec"]["code_commit"] != identity["code_commit"]):
            _fail("search_space_mismatch", "HPO jobs of these pools already exist at another configs commit")
    anchors = [a for a in _anchors(queue, args.dataset) if a["pools_sha256"] == pools["sha256"]]
    if any(a["kind"] == "recipe_freeze" and a["recipe_source"] == "hpo_d" for a in anchors):
        _fail("hpo_job_after_freeze", "an hpo_d recipe freeze of these pools is already anchored")
    if pools["grouping"]["prior_exposure"] != "none_documented" and not any(
            a["kind"] == "campaign" and a["recipe_source"] == "external_fixed" for a in anchors):
        # Decision 7 (22/10/2026): on an exposed dataset (INbreast) the external fixed recipe campaign is
        # sealed and anchored before any HPO job of these pools exists.
        _fail("external_recipe_contaminated", "exposed dataset: anchor the external_fixed campaign before HPO jobs")
    configs, directory = _hpo_configs(args, source, records, pools, space, plan, view)
    root = Path(args.code).resolve()
    specs = []
    for name in sorted(configs):
        sha = gj._bytes_sha256(gj.render_config_bytes(configs[name]))
        committed = root / directory / name
        if not committed.is_file() or file_hash(committed) != sha:
            _fail("config_identity_mismatch", "committed HPO configs differ from the plan's materialization")
        specs.append(gj.hpo_job_spec(configs[name], config_path=f"{directory}/{name}", config_sha256=sha,
                                     code_commit=identity["code_commit"]))
    jobs = queue.seed(specs)
    path = _private(args, "hpo-code-identity", canonical_hash(identity))
    gd._publish(path, {**identity, "sha256": canonical_hash(identity)}, lambda value: None)
    return _emit({"jobs": len(jobs), "code_commit": identity["code_commit"], "plan_commit": identity["freeze_commit"],
                  "changed_paths": len(identity["changed_paths"]), "code_identity_sha256": canonical_hash(identity),
                  "worker_settings": {"phase": "hpo", "views": "<views root of d-view>"}}, records)


def _validation(args):
    if (args.host_validation == "full") != bool(args.settings):
        _fail("cli_input_invalid", "--host-validation full needs --settings; minimal must not get one")
    return _read(args.settings, "worker settings") if args.settings else None


def _collect(args):
    """Evidence of the CURRENT queue state; refuses while any job of the plan is missing or not terminal."""
    source, records, images, pools, space, plan, view = _hpo_inputs(args)
    _check_environment(args, plan["runtime"])
    identity = _hpo_identity(args, plan, head=args.configs_commit)
    stored = _read(_private(args, "hpo-code-identity", canonical_hash(identity)), "seed-time HPO code identity")
    if {k: v for k, v in stored.items() if k != "sha256"} != identity:
        _fail("code_identity_mismatch", "the Q6 evidence differs from the one recorded when the HPO jobs were seeded")
    evidence, manifest = gh.collect_hpo_evidence(
        space, plan, args.plan_sha256, view, _queue(args).list_jobs(), Path(args.runs).resolve(), records=records,
        pools=pools, dataset_source=source, image_sha256=images, views_root=Path(args.views_root).resolve(),
        settings=_validation(args), code_identity=identity, code_repo=Path(args.code).resolve())
    waiting = [row for row in manifest["rows"] if row["status"] == "missing"]
    if waiting:
        # A partial snapshot would let a freeze be sealed from evidence chosen after seeing other scores.
        _fail("collection_incomplete", f"{len(waiting)} HPO job(s) are absent or not terminal; collect when all are")
    # The FIRST complete collection of a plan is binding: a later retry/resume chosen after seeing its
    # scores changes the evidence and is refused (retry infrastructure failures BEFORE collecting).
    pin_path = _private(args, "hpo-evidence-pin", args.plan_sha256)
    pin = {"plan_sha256": args.plan_sha256, "evidence_sha256": evidence["sha256"],
           "code_identity_sha256": canonical_hash(identity)}
    if pin_path.exists():
        pinned = _read(pin_path, "HPO evidence pin")
        if type(pinned) is not dict or pinned.get("sha256") != canonical_hash(
                {k: v for k, v in pinned.items() if k != "sha256"}):
            _fail("collection_invalid", "the HPO evidence pin fails its checksum")
        if {k: pinned.get(k) for k in pin} != pin:
            _fail("collection_invalid", "the evidence of this plan was pinned by its first complete collection; "
                                        "the HPO jobs changed afterwards (retry/resume after scores were visible)")
    else:
        gd._publish(pin_path, {**pin, "sha256": canonical_hash(pin)}, lambda value: None)
    return records, pools, plan, evidence, manifest


def cmd_collect_hpo(args):
    records, _, _, evidence, manifest = _collect(args)
    evidence_path = _private(args, "evidence", evidence["sha256"])
    gd._publish(evidence_path, evidence, lambda value: None)
    gd._publish(_private(args, "hpo-collection", manifest["sha256"]), manifest, lambda value: None)
    counts = {}
    for row in evidence["rows"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return _emit({"evidence_sha256": evidence["sha256"], "path": str(evidence_path),
                  "collection_sha256": manifest["sha256"], "status_counts": counts,
                  "gpu_minutes_counted": manifest["gpu_minutes_counted"], "gpu_minutes_cap": manifest["gpu_minutes_cap"]},
                 records)


def cmd_seal_freeze(args):
    import tempfile
    records, pools, plan, evidence, _ = _collect(args)
    if evidence["sha256"] != args.evidence_sha256:
        # Only the evidence of the current, complete queue state can be frozen (no choice among snapshots).
        _fail("collection_invalid", "--evidence-sha256 is not the evidence recomputed from the current queue")
    if _document(args, "evidence", args.evidence_sha256) != evidence:
        _fail("collection_invalid", "the stored evidence differs from the recomputation")
    candidates = gd._plan(plan, pools, args.plan_sha256)
    winner, means, _ = gd._evidence(evidence, plan, candidates, pools)  # predeclared ranking, never chosen here
    candidate = {"schema_version": 1, "protocol": gd.FREEZE_PROTOCOL, "scientific_status": "provisional",
                 "dataset": plan["dataset"], "task": plan["task"], "pools_sha256": pools["sha256"],
                 "selection_plan": plan, "selection_plan_sha256": args.plan_sha256,
                 "selection_evidence_sha256": evidence["sha256"], "selected_candidate_id": winner,
                 "selected_parameters": candidates[winner]["parameters"], "frozen_recipe": candidates[winner]["recipe"],
                 "final_policy": plan["final_policy"], "runtime": plan["runtime"],
                 "selection_scope": {"train_ids": pools["lists"]["D_fit"], "select_ids": pools["lists"]["D_select"],
                                     "epoch_ids": pools["lists"]["D_select"],
                                     "threshold_ids": pools["lists"]["D_select"],
                                     "augmentation": {"arm": "none", "ops": []}, "producer": plan["producer"]},
                 "frozen_at_utc": _utc(args.frozen_at, "--frozen-at")}
    with tempfile.TemporaryDirectory(prefix="atlas-freeze-") as directory:
        freeze = gd.seal_recipe_freeze(records, pools, candidate, evidence, expected_pools_sha256=pools["sha256"],
                                       expected_plan_sha256=args.plan_sha256, output=Path(directory) / "freeze.json")
    path = _private(args, "freeze", freeze["sha256"])
    gd._publish(path, freeze, lambda value: gd.audit_recipe_freeze(
        records, pools, value, evidence, expected_pools_sha256=pools["sha256"], expected_plan_sha256=args.plan_sha256))
    return _emit({"freeze_sha256": freeze["sha256"], "freeze_kind": "hpo_d", "path": str(path),
                  "selected_candidate_id": winner, "candidates_defined": len(means)}, records)


def cmd_materialize_final(args):
    source, records, _, pools = _pools(args)
    freeze, evidence = _freeze(args, records, pools)
    _, matrix = gc._stages(_stages(args.stages), freeze["final_policy"]["seeds"])
    configs = gj.build_final_configs(
        records, pools, expected_pools_sha256=pools["sha256"], dataset_source=source, freeze=freeze,
        freeze_kind=args.freeze_kind, expected_freeze_sha256=args.freeze_sha256,
        augmentation_presets=_presets(args.code), seeds=matrix["authorized_seeds"], evidence=evidence,
        expected_plan_sha256=freeze.get("selection_plan_sha256") if args.freeze_kind == "hpo_d" else None)
    manifest = gj.materialize_configs("final", Path(args.code).resolve(), configs, pools_sha256=pools["sha256"],
                                      freeze_sha256=args.freeze_sha256)
    return _emit({"manifest_sha256": manifest["sha256"], "configs": len(manifest["files"]),
                  "directory": gj.config_dir(args.dataset, "final"), "authorized_seeds": matrix["authorized_seeds"],
                  "next": "commit the directory, then seal-campaign on that clean commit"}, records)


def cmd_seal_campaign(args):
    source, records, _, pools = _pools(args)
    freeze, evidence = _freeze(args, records, pools)
    _check_environment(args, freeze["runtime"])
    code_commit = _clean_head(args.code)
    identity = gc.derive_code_identity(Path(args.code).resolve(), freeze_commit=freeze["runtime"]["commit"],
                                       code_commit=code_commit, image_id=args.image_id)
    common = dict(expected_pools_sha256=pools["sha256"], freeze=freeze, freeze_kind=args.freeze_kind,
                  expected_freeze_sha256=args.freeze_sha256, dataset_source=source,
                  augmentation_presets=_presets(args.code), execution_stages=_stages(args.stages),
                  code_commit=code_commit, image_id=args.image_id, code_identity=identity, label=args.label,
                  caveats=list(gc.CAMPAIGN_CAVEATS), sealed_at_utc=_utc(args.sealed_at, "--sealed-at"),
                  queue_records=_queue(args).list_jobs(), analysis_plan_sha256=args.analysis_plan_sha256,
                  evidence=evidence, search_space_sha256=args.search_space_sha256,
                  expected_manifest_sha256=_committed_manifest(args.code, args.dataset).get("sha256"))
    document = gc.seal_campaign(records, pools, **common)
    path = _private(args, "campaign", document["sha256"])
    gc.seal_campaign(records, pools, **common, output=path)
    return _emit({"campaign_sha256": document["sha256"], "path": str(path), "code_commit": code_commit,
                  "label": document["label"], "matrix": document["matrix"]}, records)


def cmd_anchor(args):
    source, records, _, pools = _pools(args)
    if args.kind == "pools":
        if not args.expected_manifest:
            _fail("cli_input_invalid", "a pools anchor needs --expected-manifest")
        gd.audit_global_pools(records, pools, expected_manifest_sha256=args.expected_manifest)
        document = pools
        anchor = gc.build_anchor("pools", pools, document_utc=_utc(args.document_utc or "", "--document-utc"))
    elif args.kind == "recipe_freeze":
        args.freeze_sha256 = args.document_sha256
        document, _ = _freeze(args, records, pools)
        if args.freeze_kind == "hpo_d":
            # Only the freeze of the pinned, current evidence can become C_freeze (audit B-1).
            missing = [f for f in ("search_space_sha256", "plan_sha256", "view_sha256", "image_id",
                                   "environment_probe", "runs", "views_root", "configs_commit")
                       if not getattr(args, f, None)]
            if missing:
                _fail("cli_input_invalid", "an hpo_d freeze anchor needs the collection arguments: "
                                           + ", ".join("--" + f.replace("_", "-") for f in missing))
            if args.plan_sha256 != document["selection_plan_sha256"]:
                _fail("collection_invalid", "--plan-sha256 is not the plan of this freeze")
            _, _, _, evidence, _ = _collect(args)
            if evidence["sha256"] != document["selection_evidence_sha256"]:
                _fail("collection_invalid", "this freeze was not sealed from the evidence of the current queue")
        anchor = gc.build_anchor("recipe_freeze", document)
    else:
        args.campaign_sha256 = args.document_sha256
        _, records, pools, document, _, _, _ = _campaign(args)
        anchor = gc.build_anchor("campaign", document)
    if document["sha256"] != args.document_sha256:
        _fail("cli_input_invalid", "--document-sha256 does not name the audited document")
    published = _queue(args).publish_anchor(anchor)
    receipt = _receipt(args, args.kind, anchor, published)
    return _emit({"kind": args.kind, "document_sha256": anchor["document_sha256"], "path": published["path"],
                  "commit": published["commit"], "receipt_sha256": receipt["sha256"]}, records)


def cmd_seed_final(args):
    source, records, pools, campaign, freeze, evidence, presets = _campaign(args)
    if _clean_head(args.code) != campaign["runtime"]["code_commit"]:
        _fail("code_identity_mismatch", "the code checkout HEAD is not the sealed campaign code_commit")
    if pools["grouping"]["prior_exposure"] != "none_documented":
        # Decision 7: on an exposed dataset (INbreast) both exploratory campaigns are sealed and anchored
        # before the first final job of these pools.
        sources = {a["recipe_source"] for a in _anchors(_queue(args), args.dataset)
                   if a["kind"] == "campaign" and a["pools_sha256"] == pools["sha256"]}
        if not {"external_fixed", "hpo_d"} <= sources:
            _fail("final_job_before_seal", "exposed dataset: anchor both campaigns (external_fixed, hpo_d) first")
    expected = gcol.expected_final_jobs(campaign, records, pools, freeze=freeze, dataset_source=source,
                                        augmentation_presets=presets, evidence=evidence)
    specs = [item[5] for _, item in sorted(expected.items(), key=lambda pair: pair[1][:3])]
    jobs = _queue(args).seed_final(specs, campaign["sha256"])
    return _emit({"campaign_sha256": campaign["sha256"], "jobs": len(jobs),
                  "authorized_seeds": campaign["matrix"]["authorized_seeds"]}, records)


def _recorded_commit(args):
    if args.recorded_commit:
        return args.recorded_commit
    receipt = _read(_private(args, "anchor-campaign", args.campaign_sha256), "campaign anchor receipt")
    return receipt["commit"]


def cmd_audit_chronology(args):
    _, records, _, campaign, freeze, _, _ = _campaign(args)
    report = gc.audit_chronology(_queue(args), campaign, freeze=freeze,
                                 recorded_campaign_commit=_recorded_commit(args))
    return _emit(report, records)


def cmd_report(args):
    source, records, pools, campaign, freeze, evidence, presets = _campaign(args)
    queue = _queue(args)
    settings = _validation(args)
    entries, manifest = gcol.collect_final_entries(campaign, queue.list_jobs(), Path(args.runs).resolve(),
                                                   records=records, pools=pools, freeze=freeze,
                                                   dataset_source=source, augmentation_presets=presets,
                                                   evidence=evidence, settings=settings)
    gd._publish(_private(args, "final-collection", manifest["sha256"]), manifest, lambda value: None)
    chronology = gc.audit_chronology(queue, campaign, freeze=freeze, recorded_campaign_commit=_recorded_commit(args))
    common = dict(expected_campaign_sha256=campaign["sha256"], freeze=freeze, dataset_source=source,
                  augmentation_presets=presets, evidence=evidence, chronology=chronology)
    report = gr.build_report(campaign, entries, pools, records, **common)
    path = _private(args, "report", report["sha256"])
    gr.build_report(campaign, entries, pools, records, **common, output=path)
    counts = {}
    for row in manifest["rows"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return _emit({"report_sha256": report["sha256"], "path": str(path), "collection_sha256": manifest["sha256"],
                  "authorized_cells": manifest["authorized_cells"], "status_counts": counts}, records)


def _report(args, sha):
    report = _document(args, "report", sha)
    gr.check_report(report)
    if report["hashes"]["pools_sha256"] != args.pools_sha256 or report["dataset"]["id"] != args.dataset:
        _fail("report_invalid", "report belongs to other pools or another dataset")
    return report


def cmd_summary(args):
    _, records, _, _ = _pools(args)
    report = _report(args, args.report_sha256)
    return _emit(gr.public_summary(report), records)


def cmd_export_article(args):
    _, records, _, _ = _pools(args)
    report = _report(args, args.report_sha256)
    campaign = _document(args, "campaign", report["hashes"]["campaign_sha256"])
    out = Path(args.out).resolve() if args.out else Path(gr.default_article_dir(campaign, args.article_root))
    written = gr.export_article_variables(report, campaign, out, records=records, overwrite=args.overwrite)
    return _emit({"out": str(out), "result": written}, records)


def cmd_compare(args):
    _, records, _, _ = _pools(args)
    reports = []
    for sha in (args.report_a, args.report_b):
        reports.append(_report(args, sha))
    comparison = gr.compare_campaigns(*reports)
    written = gr.export_comparison(comparison, Path(args.out).resolve(), records=records, overwrite=args.overwrite)
    return _emit({"comparison_sha256": comparison.get("sha256"), "out": str(Path(args.out).resolve()),
                  "result": written}, records)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

def _parser():
    p = argparse.ArgumentParser(prog="python -m augment_atlas.global_cli",
                                description="Global D/E protocol operator CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def command(name, function, help_text, pools=True):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--data-root", required=True, help="private host data root (the worker's settings.data)")
        s.add_argument("--source", required=True, help="JSON {adapter, manifest, manifest_sha256} of the manifest")
        s.add_argument("--dataset", required=True, help="dataset id (pools.dataset.id)")
        if pools:
            s.add_argument("--pools-sha256", required=True)
        s.set_defaults(function=function)
        return s

    def queue(s):
        s.add_argument("--remote", required=True); s.add_argument("--checkout", required=True)
        s.add_argument("--branch", default="queue")

    def code(s):
        s.add_argument("--code", required=True, help="pinned code checkout (clean)")

    s = command("identity", cmd_identity, "record count and hashes; with --pools-sha256 the global manifest hash",
                pools=False)
    s.add_argument("--pools-sha256"); s.add_argument("--num-classes", type=int, default=2)
    s = command("make-pools", cmd_make_pools, "generate D/E pools (global-dev-eval-v1) as a private candidate",
                pools=False)
    s.add_argument("--spec", required=True, help="split spec JSON (seeds, fractions, thresholds, metadata)")
    s.add_argument("--acquisition-config", required=True,
                   help="pre-registered config whose splits block the spec must reproduce")
    s.add_argument("--split-amendment", required=True,
                   help="committed pre-registration amendment (explicit seeds, S2 rules) the spec must reproduce")
    s.add_argument("--acquisition-root", required=True, help="the verified acquisition root (READY.json present)")
    code(s)
    s = command("split-spec", cmd_split_spec, "write the split spec of a verified acquisition (hashes only)",
                pools=False)
    s.add_argument("--acquisition-root", required=True, help="the verified acquisition root (READY.json present)")
    s.add_argument("--split-amendment", required=True); s.add_argument("--out", required=True)
    s = command("convert-legacy-pools", cmd_convert_legacy_pools,
                "convert fixed INbreast proposal-v1 lists to a private v1 candidate (no resampling)", pools=False)
    s.add_argument("--legacy", required=True); s.add_argument("--out", required=True)
    s = command("import-pools", cmd_import_pools, "seal an approved pools document into the private store",
                pools=False)
    s.add_argument("--file", required=True); s.add_argument("--expected-manifest", required=True)
    s = command("audit-pools", cmd_audit_pools, "audit sealed pools against the host manifest")
    s.add_argument("--expected-manifest", required=True)
    command("hpo-view", cmd_hpo_view, "derive and store the D-only hpo-view")
    s = command("d-view", cmd_d_view, "materialize views/<pools>/D (hardlinks of D only)")
    s.add_argument("--view-sha256", required=True); s.add_argument("--views-root", required=True)
    s = command("seal-external", cmd_seal_external, "seal the Q2 external fixed recipe freeze (no HPO)")
    code(s)
    s.add_argument("--image-id", required=True); s.add_argument("--environment-probe", required=True)
    s.add_argument("--weights-sha256", required=True); s.add_argument("--final-seeds", required=True)
    s.add_argument("--color", required=True, choices=["grayscale_to_rgb", "rgb"])
    s.add_argument("--frozen-at", required=True, help="explicit UTC, e.g. 2026-10-23T12:00:00Z")
    def runtime_args(s):
        code(s)
        s.add_argument("--image-id", required=True); s.add_argument("--environment-probe", required=True)
        s.add_argument("--weights-sha256", required=True)

    def hpo_pins(s):
        s.add_argument("--search-space-sha256", required=True); s.add_argument("--plan-sha256", required=True)
        s.add_argument("--view-sha256", required=True)

    s = command("search-space", cmd_search_space, "seal the HPO search space (base recipe = Q2 default, c00)")
    runtime_args(s)
    s.add_argument("--spec", required=True, help="JSON {grid, default, seeds, budget, failure_policy}")
    s.add_argument("--color", required=True, choices=["grayscale_to_rgb", "rgb"])
    s = command("build-plan", cmd_build_plan, "build and store the selection plan at the clean HEAD (commit P)")
    runtime_args(s)
    s.add_argument("--search-space-sha256", required=True); s.add_argument("--view-sha256", required=True)
    s.add_argument("--final-seeds", required=True)
    s = command("materialize-hpo", cmd_materialize_hpo, "write the HPO configs + MANIFEST of an anchored plan")
    code(s); hpo_pins(s)
    s = command("seed-hpo", cmd_seed_hpo, "seed the HPO jobs at the configs commit (Q6 evidence from P)")
    code(s); hpo_pins(s); queue(s)
    s.add_argument("--image-id", required=True); s.add_argument("--environment-probe", required=True)

    def collection_args(s):
        code(s); hpo_pins(s); queue(s)
        s.add_argument("--image-id", required=True); s.add_argument("--environment-probe", required=True)
        s.add_argument("--runs", required=True); s.add_argument("--views-root", required=True)
        s.add_argument("--configs-commit", required=True, help="the code_commit printed by seed-hpo")
        s.add_argument("--settings", help="worker settings: full nine-artifact validation (default mode)")
        s.add_argument("--host-validation", choices=["full", "minimal"], default="full")

    s = command("collect-hpo", cmd_collect_hpo, "collect the finished HPO jobs into queue_collected_v1 evidence")
    collection_args(s)
    s = command("seal-freeze", cmd_seal_freeze,
                "seal the hpo_d freeze from the evidence of the current queue (winner by the predeclared ranking)")
    collection_args(s)
    s.add_argument("--evidence-sha256", required=True); s.add_argument("--frozen-at", required=True)
    for name, function, text in (("materialize-final", cmd_materialize_final, "write the final configs + MANIFEST"),
                                 ("seal-campaign", cmd_seal_campaign, "seal campaign.json on the configs commit")):
        s = command(name, function, text)
        code(s)
        s.add_argument("--freeze-sha256", required=True)
        s.add_argument("--freeze-kind", required=True, choices=sorted(FREEZE_PREFIX))
        s.add_argument("--stages", required=True, help="seeds per stage, e.g. '11;13,19' (stage 1 is authorized)")
        if name == "seal-campaign":
            queue(s)
            s.add_argument("--image-id", required=True); s.add_argument("--environment-probe", required=True)
            s.add_argument("--sealed-at", required=True)
            s.add_argument("--label", default="exploratory", choices=["exploratory", "confirmatory_candidate"])
            s.add_argument("--analysis-plan-sha256"); s.add_argument("--search-space-sha256")
    s = command("anchor", cmd_anchor, "publish a public hash anchor on the queue branch")
    code(s); queue(s)
    s.add_argument("--kind", required=True, choices=["pools", "recipe_freeze", "campaign"])
    s.add_argument("--document-sha256", required=True)
    s.add_argument("--freeze-kind", choices=sorted(FREEZE_PREFIX), default="external_fixed")
    s.add_argument("--expected-manifest"); s.add_argument("--document-utc")
    for name in ("--search-space-sha256", "--plan-sha256", "--view-sha256", "--image-id", "--environment-probe",
                 "--runs", "--views-root", "--configs-commit", "--settings"):
        s.add_argument(name, help="hpo_d freeze anchors re-run the HPO collection (same arguments as collect-hpo)")
    s.add_argument("--host-validation", choices=["full", "minimal"], default="full")
    for name, function, text in (("seed-final", cmd_seed_final, "seed the authorized final jobs"),
                                 ("audit-chronology", cmd_audit_chronology, "audit freeze -> campaign -> finals"),
                                 ("report", cmd_report, "collect finished final jobs and build the private report")):
        s = command(name, function, text)
        code(s); queue(s)
        s.add_argument("--campaign-sha256", required=True)
        if name != "seed-final":
            s.add_argument("--recorded-commit", help="C_campaign recorded at sealing (default: the anchor receipt)")
        if name == "report":
            s.add_argument("--runs", required=True)
            s.add_argument("--settings", help="worker settings: full nine-artifact validation (default mode)")
            s.add_argument("--host-validation", choices=["full", "minimal"], default="full",
                           help="minimal skips the worker re-validation; it must be chosen explicitly")
    s = command("summary", cmd_summary, "public, ID-free summary of a private report")
    s.add_argument("--report-sha256", required=True)
    s = command("export-article", cmd_export_article, "export the article variables of one report")
    s.add_argument("--report-sha256", required=True); s.add_argument("--out")
    s.add_argument("--article-root"); s.add_argument("--overwrite", action="store_true")
    s = command("compare", cmd_compare, "descriptive comparison of the two campaigns of one dataset")
    s.add_argument("--report-a", required=True); s.add_argument("--report-b", required=True)
    s.add_argument("--out", required=True); s.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "export-article" and not (args.out or args.article_root):
        _parser().error("export-article needs --out or --article-root")
    try:
        return args.function(args)
    except (ValueError, OSError, QueueError, KeyError, TypeError) as error:
        message = str(error) if isinstance(error, (ValueError, QueueError)) else f"cli_error: {type(error).__name__}"
        sys.stderr.write(json.dumps({"error": _redact(message)}) + "\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
