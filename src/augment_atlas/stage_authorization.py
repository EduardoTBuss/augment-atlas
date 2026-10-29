"""Sealed later-stage supplement; original campaign and primary configs stay immutable.

Bradshaw 2023, DOI 10.1148/ryai.220232: reshuffling folds does not
remove selection bias. This authorization grants sensitivity execution only.
"""
from copy import deepcopy
from datetime import date
from functools import lru_cache
import json

from . import global_campaign as gc
from . import global_jobs as gj
from .contracts import ARMS, canonical_hash

PROTOCOL = "global-stage-authorization-v1"
POLICY = "additive_stage_configs_same_repository_tree_v1"
SEED_NOTE = ('SAP sealed item 5: "training seed 1" means the first declared training seed (11); '
             'the sealed SAP is unchanged. Later seeds are sensitivity only.')
KEYS = {"schema_version", "protocol", "campaign_sha256", "dataset", "pools_sha256", "freeze",
        "analysis_plan_sha256", "runtime", "stage1_configs", "stage", "seeds", "configs_manifest",
        "execution_code_commit", "identity_policy", "unchanged_tree_sha256", "decided_on",
        "decided_after_primary_results", "rationale", "seed_interpretation", "sealed_at_utc", "sha256"}


def directory(campaign, stage):
    return gj.config_dir(campaign["dataset"]["id"], "final") + f"/stage-{stage}"


def stage_seeds(campaign, stage):
    if type(stage) is not int or not 2 <= stage <= len(campaign["execution_stages"]):
        gc._fail("campaign_invalid", "authorize only a declared later stage (index >= 2)")
    return list(campaign["execution_stages"][stage - 1]["seeds"])


@lru_cache(maxsize=8192)
def _blob(repo, commit, path):
    result = gc._git(repo, "show", f"{commit}:{path}")
    if result.returncode:
        gc._fail("config_identity_mismatch", f"pinned config object is unreadable: {path}")
    return result.stdout


def _entries(repo, commit):
    gc._commit(commit, "execution_code_commit")
    result = gc._git(repo, "ls-tree", "-r", "-z", "--full-tree", commit)
    if result.returncode:
        gc._fail("code_identity_mismatch", "execution commit is unreadable")
    entries = {}
    for item in result.stdout.decode("utf-8").split("\0"):
        if item:
            meta, path = item.split("\t", 1)
            mode, kind, sha = meta.split()
            entries[path] = (mode, kind, sha)
    return entries


def configs(campaign, stage, code_repo):
    """Clone the sealed primary cell for each declared seed; only training.seed changes."""
    seeds = stage_seeds(campaign, stage)
    commit = campaign["runtime"]["code_commit"]
    path = campaign["configs"]["root"] + "/MANIFEST.json"
    data = _blob(code_repo, commit, path)
    manifest = json.loads(data)
    if (manifest.get("sha256") != campaign["configs"]["manifest_sha256"]
            or gj._seal(manifest) != manifest or gj.render_config_bytes(manifest) != data):
        gc._fail("config_identity_mismatch", "original manifest does not match the sealed campaign")
    primary = campaign["matrix"]["authorized_seeds"][0]
    output = {}
    for arm in ARMS:
        for fold in campaign["folds"]:
            name = f"{arm}-f{fold}-s{primary}.json"
            path = campaign["configs"]["root"] + "/" + name
            data = _blob(code_repo, commit, path)
            if {"path": path, "sha256": gj._bytes_sha256(data)} not in manifest["files"]:
                gc._fail("config_identity_mismatch", "primary config is not pinned by the original manifest")
            template = json.loads(data)
            gj.validate_global_config(template)
            if gj.render_config_bytes(template) != data:
                gc._fail("config_identity_mismatch", "primary config rendering differs")
            for seed in seeds:
                cfg = deepcopy(template)
                cfg["training"]["seed"] = seed
                output[f"{arm}-f{fold}-s{seed}.json"] = cfg
    return output


def manifest(campaign, stage, stage_configs):
    relative = directory(campaign, stage)
    return gj._seal({"schema_version": 1, "protocol": gj.MANIFEST_PROTOCOL, "phase": "final",
                     "dataset": deepcopy(campaign["dataset"]), "pools_sha256": campaign["pools_sha256"],
                     "plan_sha256": None, "freeze_sha256": campaign["freeze"]["sha256"],
                     "files": [{"path": relative + "/" + name,
                                "sha256": gj._bytes_sha256(gj.render_config_bytes(stage_configs[name]))}
                               for name in sorted(stage_configs)]})


def identity(campaign, stage, execution_code_commit, code_repo, stage_configs, stage_manifest):
    """Compare ALL Git entries (including tests/docs); only this supplement may be added."""
    original = campaign["runtime"]["code_commit"]
    old, new = _entries(code_repo, original), _entries(code_repo, execution_code_commit)
    if gc._git(code_repo, "merge-base", "--is-ancestor", original, execution_code_commit).returncode:
        gc._fail("code_identity_mismatch", "campaign commit must be an ancestor of execution commit")
    if any(new.get(path) != entry for path, entry in old.items()):
        gc._fail("code_identity_mismatch", "an existing repository entry changed (including primary configs)")
    relative = directory(campaign, stage)
    expected = {relative + "/" + name: gj.render_config_bytes(cfg) for name, cfg in stage_configs.items()}
    expected[relative + "/MANIFEST.json"] = gj.render_config_bytes(stage_manifest)
    if set(new) - set(old) != set(expected):
        gc._fail("code_identity_mismatch", "only the exact authorized stage config files may be added")
    for path, data in expected.items():
        if new[path][0:2] != ("100644", "blob") or _blob(code_repo, execution_code_commit, path) != data:
            gc._fail("config_identity_mismatch", "execution commit differs from deterministic stage configs")
    return canonical_hash(sorted([path, *entry] for path, entry in old.items()
                                 if not path.startswith("configs/global/")))


def authorize_stage(campaign, *, stage, execution_code_commit, code_repo, image_id, environment_sha256,
                    decided_on, decided_after_primary_results, rationale, sealed_at_utc, output=None):
    if (type(campaign) is not dict or set(campaign) != gc._CAMPAIGN_KEYS
            or gj._seal(campaign) != campaign):
        gc._fail("campaign_invalid", "a sealed original campaign is required")
    seeds = stage_seeds(campaign, stage)
    if image_id != campaign["runtime"]["image_id"] or environment_sha256 != campaign["runtime"]["environment_sha256"]:
        gc._fail("code_identity_mismatch", "stage image/probe environment must equal the original campaign")
    if type(decided_after_primary_results) is not bool:
        gc._fail("campaign_invalid", "decided_after_primary_results is a mandatory boolean")
    try:
        day = date.fromisoformat(decided_on) if type(decided_on) is str else None
    except ValueError:
        day = None
    sealed = gc._parse_utc(sealed_at_utc, "sealed_at_utc")
    if (day is None or day.isoformat() != decided_on or day < gc._parse_utc(campaign["sealed_at_utc"], "campaign seal").date()
            or day > sealed.date() or sealed < gc._parse_utc(campaign["sealed_at_utc"], "campaign seal")):
        gc._fail("campaign_invalid", "decision date/seal must not precede campaign or follow authorization seal")
    if type(rationale) is not str or not rationale.strip():
        gc._fail("campaign_invalid", "rationale is mandatory nonempty text")
    stage_configs = configs(campaign, stage, code_repo)
    stage_manifest = manifest(campaign, stage, stage_configs)
    tree = identity(campaign, stage, execution_code_commit, code_repo, stage_configs, stage_manifest)
    document = gj._seal({"schema_version": 1, "protocol": PROTOCOL, "campaign_sha256": campaign["sha256"],
                        "dataset": deepcopy(campaign["dataset"]), "pools_sha256": campaign["pools_sha256"],
                        "freeze": deepcopy(campaign["freeze"]), "analysis_plan_sha256": campaign["analysis_plan_sha256"],
                        "runtime": deepcopy(campaign["runtime"]), "stage1_configs": deepcopy(campaign["configs"]),
                        "stage": stage, "seeds": seeds, "configs_manifest": stage_manifest,
                        "execution_code_commit": execution_code_commit, "identity_policy": POLICY,
                        "unchanged_tree_sha256": tree, "decided_on": decided_on,
                        "decided_after_primary_results": decided_after_primary_results,
                        "rationale": rationale, "seed_interpretation": SEED_NOTE, "sealed_at_utc": sealed_at_utc})
    if output is not None:
        from .global_development import _publish
        _publish(output, document, lambda value: audit(value, campaign, code_repo=code_repo))
    return document


def audit(document, campaign, *, code_repo):
    if type(document) is not dict or set(document) != KEYS or gj._seal(document) != document:
        gc._fail("campaign_invalid", "authorization requires the closed sealed stage schema")
    if code_repo is None:
        gc._fail("code_identity_mismatch", "stage authorization requires the code repository for Git verification")
    rebuilt = authorize_stage(campaign, stage=document["stage"], execution_code_commit=document["execution_code_commit"],
                              code_repo=code_repo, image_id=document["runtime"].get("image_id"),
                              environment_sha256=document["runtime"].get("environment_sha256"),
                              decided_on=document["decided_on"], decided_after_primary_results=document["decided_after_primary_results"],
                              rationale=document["rationale"], sealed_at_utc=document["sealed_at_utc"])
    if rebuilt != document:
        gc._fail("campaign_invalid", "authorization differs from deterministic campaign/stage/Git identity")
    return document


def materialize(campaign, stage, code_repo, *, authorization=None):
    """Prepare additions before sealing, or verify/materialize the pinned supplement idempotently."""
    if authorization is not None:
        audit(authorization, campaign, code_repo=code_repo)
        if stage != authorization["stage"]:
            gc._fail("campaign_invalid", "materialization stage differs from authorization")
    stage_configs = configs(campaign, stage, code_repo)
    return gj.materialize_configs("final", code_repo, stage_configs, pools_sha256=campaign["pools_sha256"],
                                  freeze_sha256=campaign["freeze"]["sha256"], stage=stage)


def job_binding(spec, campaign, document):
    """Check stage and primary identities without changing the immutable job schema."""
    bound = spec.get("global")
    expected = {"campaign_sha256": campaign["sha256"], "pools_sha256": campaign["pools_sha256"],
                "freeze_sha256": campaign["freeze"]["sha256"], "freeze_kind": campaign["freeze"]["kind"]}
    if (type(bound) is not dict or any(bound.get(key) != value for key, value in expected.items())
            or spec.get("recipe_hash") != campaign["freeze"]["sha256"]
            or (spec.get("dataset"), spec.get("dataset_version")) !=
               (campaign["dataset"]["id"], campaign["dataset"]["version"])):
        gc._fail("campaign_anchor_missing", "job is not bound to this authorized campaign/pools/freeze")
    seed = spec.get("train_seed")
    if seed in campaign["matrix"]["authorized_seeds"]:
        commit, relative = campaign["runtime"]["code_commit"], campaign["configs"]["root"]
    elif document is not None and seed in document["seeds"]:
        commit, relative = document["execution_code_commit"], directory(campaign, document["stage"])
    else:
        gc._fail("campaign_invalid", "seed is outside the authorized stage")
    if spec.get("code_commit") != commit:
        gc._fail("code_identity_mismatch", "job commit differs from its authorized stage")
    expected_path = relative + f"/{spec['arm']}-f{spec['fold']}-s{seed}.json"
    if spec.get("config") != expected_path:
        gc._fail("config_identity_mismatch", "job path differs from its authorized stage")
    if document is not None and seed in document["seeds"]:
        if {"path": expected_path, "sha256": spec.get("config_sha256")} not in document["configs_manifest"]["files"]:
            gc._fail("config_identity_mismatch", "job config is not pinned by stage authorization")
