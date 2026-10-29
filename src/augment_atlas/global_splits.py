"""D/E pools generator ``global-dev-eval-v1`` (dataset-acquisition-plan section 5).

Stdlib only, deterministic and seed-pinned: no sklearn, no retry. The unit is
the group of the manifest (patient, merged with duplicates by the acquisition
tool). The output is an UNSIGNED v1 pools candidate for
``global_development.seal_global_pools`` (the CLI ``make-pools`` then
``import-pools``); it holds record IDs and stays private.

Algorithm (binary tasks; other task types have no specified stratification yet):

1. Strata: ``positive`` (the group has an image of ``task.positive_class``) x
   the quartile bucket of the group's image count. Quartile cut points are the
   nearest-rank values ``sizes[ceil(k*n/4) - 1]`` (k = 1, 2, 3) of the sorted
   sizes of ALL groups; the bucket is the number of cut points strictly below
   the size. D_select reuses these strata restricted to D (no new quartiles).
2. D: in each stratum, groups ordered by
   ``canonical_hash([algorithm, seeds.development, "DE", group])``; the first
   ``floor(f_D * n + 1/2)`` go to D (half-up rounding in exact decimal
   arithmetic, not banker's ``round``; for f_sel = 0.30 the two differ when
   ``n`` is 15, 35, 55, ... in a stratum).
3. D_select: the same inside D with stage ``"D-select"``, ``seeds.selection``
   and ``f_sel``; the rest of D is D_fit.
4. E -> 5 folds: strata in order, groups in each ordered by stage
   ``"E-folds"`` with ``seeds.evaluation``, assigned round-robin 0..4 with ONE
   counter that continues across strata (fold sizes differ by at most one
   group; a restart per stratum would bias fold 0).
5. Pass/fail audits, no seed retry and no fallback: coverage exactly once,
   groups never split, both classes in D_fit, D_select and every E fold, and
   the pre-registered thresholds (positive groups in D_select and in each E
   fold). A failure raises ``split_audit_failed`` and must be escalated.

Optional forced-D extension (disabled by default): a private JSON list pinned by
its file SHA256 reserves D places BEFORE the per-stratum draw. Quotas keep the
same half-up rule over ALL groups; excess forced groups fail without fallback.
D_select is still drawn from all of D, including forced groups, so they may land
in D_fit or D_select. The distinct algorithm identity is used for every stage's
hash ordering. Private IDs are persisted for re-audit; public reports contain
only their file SHA256 and count. With no option, legacy v1 bytes are unchanged.

The evaluation seed must differ from the development and selection seeds (Q3,
``seed_reuse_forbidden``).

Caveat (``global_campaign.SELECTION_BIAS_CAVEAT``): reshuffling folds does not
remove selection bias if patients used to choose hyperparameters fall into a
test set (Bradshaw 2023, DOI 10.1148/ryai.220232); D therefore stays out of
every test.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from . import global_development as gd
from .contracts import canonical_hash

ALGORITHM = "global-dev-eval-v1"
FORCED_ALGORITHM = gd.FORCED_DEVELOPMENT_ALGORITHM
VERSION = "1"
GENERATOR_VERSION = "augment_atlas.global_splits v1"
FOLDS = 5
_SPEC_KEYS = {"dataset", "task", "grouping", "provenance", "source_sha256", "seeds", "fractions", "thresholds"}


def _fail(code, message):
    raise ValueError(f"{code}: {message}")


def check_preregistered(spec, splits):
    """The spec must reproduce the pre-registered ``splits`` block of the acquisition config.

    Fractions and thresholds cannot be relaxed at split time; seeds are not
    compared because the config pins one ``split_seed`` while the pools schema
    needs three (a pending amendment), so they are printed for the record.
    """
    _check_spec(spec)
    expected = {"algorithm": ALGORITHM, "eval_folds": FOLDS}
    if type(splits) is not dict or any(splits.get(k) != v for k, v in expected.items()):
        _fail("split_spec_invalid", "acquisition config splits must declare global-dev-eval-v1 with 5 folds")
    pairs = ((spec["fractions"]["D"], splits.get("dev_fraction"), "D fraction"),
             (spec["fractions"]["D_select_given_D"], splits.get("select_fraction"), "D_select fraction"),
             (spec["thresholds"]["min_select_positive_groups"], splits.get("min_select_positive_groups"),
              "min_select_positive_groups"),
             (spec["thresholds"]["min_eval_fold_positive_groups"], splits.get("min_eval_fold_positive_groups"),
              "min_eval_fold_positive_groups"))
    for got, pinned, name in pairs:
        if type(pinned) not in (int, float) or Fraction(str(got)) != Fraction(str(pinned)):
            _fail("split_spec_invalid", f"{name} differs from the pre-registered acquisition config")


AMENDMENT_SCHEMA = "augment-atlas/split-amendment/v1"
_AMENDMENT_KEYS = {"schema", "dataset", "amends", "decided_by", "decided_on", "before_any_score", "algorithm",
                   "eval_folds", "seeds", "fractions", "thresholds", "rules", "seed_rationale", "no_reseed",
                   "supersedes"}


def check_amendment(spec, amendment, acquisition_raw):
    """The spec must reproduce a committed pre-registration amendment of the acquisition ``splits`` block.

    The amendment pins the acquisition config identity (unchanged, so the existing
    download ledger stays valid), declares the three seeds (Q3) and the S2 rules,
    and must agree with the fractions/thresholds of the config itself.
    """
    from .acquisition.config import identity_of
    if type(amendment) is not dict or set(amendment) != _AMENDMENT_KEYS or amendment["schema"] != AMENDMENT_SCHEMA:
        _fail("split_spec_invalid", "split amendment must be a closed augment-atlas/split-amendment/v1 document")
    if amendment["before_any_score"] is not True or amendment["algorithm"] != ALGORITHM or amendment["eval_folds"] != FOLDS:
        _fail("split_spec_invalid", "split amendment must be global-dev-eval-v1, 5 folds, declared before any score")
    if type(acquisition_raw) is not dict or amendment["amends"].get("config_identity_sha256") != identity_of(
            acquisition_raw):
        _fail("split_spec_invalid", "split amendment is bound to another acquisition config identity")
    if "splits" in acquisition_raw:
        check_preregistered(spec, acquisition_raw["splits"])
    else:
        # No splits block (rsna2018-v1): the amendment is the sole split pre-registration (its
        # ``supersedes`` says so) and pins the fractions/thresholds compared below.
        _check_spec(spec)
    if amendment["dataset"] != spec["dataset"].get("id"):
        _fail("split_spec_invalid", "the split amendment belongs to another dataset")
    for key in ("seeds", "fractions", "thresholds"):
        if canonical_hash(spec[key]) != canonical_hash(amendment[key]):
            _fail("split_spec_invalid", f"spec {key} differ from the pre-registered split amendment")


def spec_from_acquisition(ready, private, amendment, *, dataset_id):
    """Split spec of a verified acquisition (READY + manifest-private): hashes only, no identifiers.

    ``source_sha256`` = canonical hash of the release sources (ETag/size, central
    directory and CSV hashes); provenance hashes cover the preprocessing block,
    the GroundTruth CSVs, the duplicate pairs/policy and the exclusions. The
    grouping is the patient/duplicate union-find, declared at ``group`` level
    with unverified linkage (no external patient-identity evidence).
    """
    if type(ready) is not dict or ready.get("schema") != "augment-atlas/dataset-ready/v1":
        _fail("split_spec_invalid", "READY.json is not an augment-atlas/dataset-ready/v1 document")
    if type(private) is not dict or private.get("schema") != "augment-atlas/dataset-manifest-private/v1":
        _fail("split_spec_invalid", "manifest-private.json is not a dataset-manifest-private/v1 document")
    if type(private.get("policy")) is dict and private["policy"].get("group") == RSNA_PATIENT_GROUP:
        return _rsna_spec(private, amendment, dataset_id=dataset_id)
    sources = private["acquisition"]["sources"]
    labels = [s for s in sources if s["role"] in ("ground_truth_v1", "ground_truth_v2")]
    excluded = [{"image": r["image"], "status": r["status"], "reason": r["reason"]}
                for r in private["records"] if r["status"] != "included"]
    return {"dataset": {"id": dataset_id, "version": private["preprocessing"]["id"]},
            "task": {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
                     "decision_rule": "fixed_threshold", "missing_policy": "undefined_no_drop",
                     "requires_patient_identity": True},
            "grouping": {"namespace": f"{private['dataset']['name']} patient_id merged by duplicate pairs",
                         "source": private["groups"]["algorithm"], "identity_level": "group",
                         "linkage_status": "unverified", "evidence_sha256": None,
                         "prior_exposure": "none_documented"},
            "provenance": {"preprocessing_sha256": canonical_hash(private["preprocessing"]),
                           "labels_sha256": canonical_hash(labels),
                           "duplicates_sha256": canonical_hash(private["duplicates"]),
                           "exclusions_sha256": canonical_hash(excluded)},
            "source_sha256": canonical_hash(sources),
            "seeds": deepcopy(amendment["seeds"]), "fractions": deepcopy(amendment["fractions"]),
            "thresholds": deepcopy(amendment["thresholds"])}


RSNA_PATIENT_GROUP = "nih-img-id-eight-digit-patient-prefix"


def _rsna_spec(private, amendment, *, dataset_id):
    """rsna2018 layout (acquisition/rsna2018.py): ``sources`` keyed by role, no union-find ``groups``.

    The group is the NIH patient (eight-digit ``img_id`` prefix) read from the SHA256-pinned
    mappings file, so identity is patient-level with that file + rule as linkage evidence;
    pixel duplicates are excluded, never merged across patients.
    """
    sources, policy = private["acquisition"]["sources"], private["policy"]
    excluded = [{"image": r["SOPInstanceUID"], "status": r["status"], "reason": r["reason"], "reasons": r["reasons"]}
                for r in private["records"] if r["status"] != "included"]
    return {"dataset": {"id": dataset_id, "version": private["preprocessing"]["id"]},
            "task": {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
                     "decision_rule": "fixed_threshold", "missing_policy": "undefined_no_drop",
                     "requires_patient_identity": True},
            "grouping": {"namespace": f"{private['dataset']['name']} NIH patient (img_id eight-digit prefix)",
                         "source": policy["group"], "identity_level": "patient", "linkage_status": "verified",
                         "evidence_sha256": canonical_hash({"group": policy["group"], "mappings": sources["mappings"]}),
                         "prior_exposure": "none_documented"},
            "provenance": {"preprocessing_sha256": canonical_hash(private["preprocessing"]),
                           "labels_sha256": canonical_hash({"annotations": sources["annotations"],
                                                            **{k: policy[k] for k in ("label_group", "labels",
                                                                                      "missing_label")}}),
                           "duplicates_sha256": canonical_hash(private["duplicates"]),
                           "exclusions_sha256": canonical_hash(excluded)},
            "source_sha256": canonical_hash(sources),
            "seeds": deepcopy(amendment["seeds"]), "fractions": deepcopy(amendment["fractions"]),
            "thresholds": deepcopy(amendment["thresholds"])}


def _check_spec(spec):
    if type(spec) is not dict or set(spec) != _SPEC_KEYS:
        _fail("split_spec_invalid", f"split spec must have exactly {sorted(_SPEC_KEYS)}")
    if type(spec["dataset"]) is not dict or type(spec["task"]) is not dict:
        _fail("split_spec_invalid", "dataset and task must be objects")
    seeds = spec["seeds"]
    if type(seeds) is not dict or set(seeds) != {"development", "selection", "evaluation"} \
            or any(type(v) is not int for v in seeds.values()):
        _fail("split_spec_invalid", "seeds must be integers {development, selection, evaluation}")
    if seeds["evaluation"] in (seeds["development"], seeds["selection"]):
        _fail("seed_reuse_forbidden", "E folds need an evaluation seed distinct from the D seeds (Q3)")
    fractions = spec["fractions"]
    if type(fractions) is not dict or set(fractions) != {"D", "D_select_given_D"}:
        _fail("split_spec_invalid", "fractions must be {D, D_select_given_D}")
    for key, value in fractions.items():
        if type(value) not in (int, float) or not 0 < value < 1:
            _fail("split_spec_invalid", f"fraction {key} must be strictly between 0 and 1")
    thresholds = spec["thresholds"]
    if type(thresholds) is not dict or set(thresholds) != {"min_select_positive_groups",
                                                           "min_eval_fold_positive_groups"} \
            or any(type(v) is not int or v < 1 for v in thresholds.values()):
        _fail("split_spec_invalid", "thresholds must be positive integers "
                                    "{min_select_positive_groups, min_eval_fold_positive_groups}")
    task = spec["task"]
    if type(task) is not dict or task.get("type") != "binary":
        _fail("split_spec_invalid", "global-dev-eval-v1 stratification is specified for binary tasks only")


def _quartile_bucket(sizes):
    ordered = sorted(sizes)
    n = len(ordered)
    cuts = [ordered[math.ceil(k * n / 4) - 1] for k in (1, 2, 3)]
    return lambda size: sum(size > cut for cut in cuts)


def _half_up_count(fraction, n):
    """floor(f * n + 1/2) in exact decimal arithmetic (0.35 * 90 is 31.5, not 31.4999...)."""
    return math.floor(Fraction(str(fraction)) * n + Fraction(1, 2))


def _key(seed, stage, group, algorithm=ALGORITHM):
    return canonical_hash([algorithm, seed, stage, group])


def _strata(groups, members, positive_class):
    """{(positive, bucket): [group, ...]} for the given groups."""
    bucket = _quartile_bucket([len(members[g]) for g in groups])
    strata = {}
    for group in groups:
        positive = int(any(row["label"] == positive_class for row in members[group]))
        strata.setdefault((positive, bucket(len(members[group]))), []).append(group)
    return dict(sorted(strata.items()))


def _take(strata, fraction, seed, stage, *, forced=(), algorithm=ALGORITHM):
    chosen = set()
    for groups in strata.values():
        pinned = set(groups).intersection(forced)
        quota = _half_up_count(fraction, len(groups))
        if len(pinned) > quota:
            _fail("forced_development_quota_exceeded",
                  f"forced groups in a stratum ({len(pinned)}) exceed D quota ({quota}); no fallback")
        ordered = sorted((g for g in groups if g not in pinned),
                         key=lambda g: (_key(seed, stage, g, algorithm), g))
        chosen.update(pinned)
        chosen.update(ordered[:quota - len(pinned)])
    return chosen


def _ids(groups, members):
    return sorted(row["id"] for group in groups for row in members[group])


def load_forced_development(path, expected_sha256):
    """Read/hash the SAME private file bytes once; never echo IDs in errors."""
    try:
        gd._hash(expected_sha256, "force-development-sha256")
    except ValueError:
        _fail("forced_development_invalid", "a lowercase 64-hex file SHA256 is required")
    try:
        raw = Path(path).read_bytes()
    except OSError:
        _fail("forced_development_invalid", "private forced groups file is unreadable")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail("forced_development_hash_mismatch", "private forced groups file differs from its SHA256 pin")
    try:
        groups = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        _fail("forced_development_invalid", "private forced groups file must be a JSON list of group IDs")
    gd.validate_forced_development_groups(groups)
    return groups


def generate_dev_eval_pools(records, spec, *, forced_development_groups=None, forced_development_sha256=None):
    """Return (unsigned pools candidate, split report). Raises on any failed audit.

    Optional private IDs must come from ``load_forced_development``: the caller
    verifies the raw file hash there, then supplies those IDs and that pin here.
    """
    _check_spec(spec)
    forced = set()
    algorithm = ALGORITHM
    if forced_development_groups is not None or forced_development_sha256 is not None:
        gd.validate_forced_development_groups(forced_development_groups)
        try:
            gd._hash(forced_development_sha256, "forced_development.sha256")
        except ValueError:
            _fail("forced_development_invalid", "forced group IDs require their file SHA256")
        forced = set(forced_development_groups)
        algorithm = FORCED_ALGORITHM
    rows = gd._records(deepcopy(records))
    task = spec["task"]
    positive_class = task["positive_class"]
    members = {}
    for row in rows:
        if row["label"] not in task["classes"]:
            _fail("split_spec_invalid", "record label outside the task classes")
        members.setdefault(row["group"], []).append(row)
    if not forced <= set(members):
        _fail("forced_development_unknown_group", "forced group absent from manifest")
    groups = sorted(members)
    seeds, fractions = spec["seeds"], spec["fractions"]
    strata = _strata(groups, members, positive_class)
    d_groups = _take(strata, fractions["D"], seeds["development"], "DE",
                     forced=forced, algorithm=algorithm)
    d_strata = {key: [g for g in value if g in d_groups] for key, value in strata.items()}
    select_groups = _take(d_strata, fractions["D_select_given_D"], seeds["selection"], "D-select",
                          algorithm=algorithm)
    fold_of, counter = {}, 0
    for value in strata.values():
        for group in sorted((g for g in value if g not in d_groups), key=lambda g: (_key(seeds["evaluation"],
                                                                                      "E-folds", g, algorithm), g)):
            fold_of[group] = counter % FOLDS
            counter += 1
    lists = {"D": _ids(d_groups, members), "D_fit": _ids(d_groups - select_groups, members),
             "D_select": _ids(select_groups, members), "E": _ids(set(fold_of), members)}
    folds = [{"fold": f, "test_ids": _ids({g for g, k in fold_of.items() if k == f}, members)} for f in range(FOLDS)]
    candidate = {
        "schema_version": 1, "protocol": gd.PROTOCOL, "scientific_status": "provisional",
        "dataset": deepcopy(spec["dataset"]), "records_sha256": canonical_hash(rows),
        "source_sha256": spec["source_sha256"], "provenance": deepcopy(spec["provenance"]),
        "grouping": deepcopy(spec["grouping"]), "task": deepcopy(task),
        "partition_provenance": {"algorithm": algorithm, "version": VERSION,
                                 "source": "stdlib hash-ordered stratified groups (dataset-acquisition-plan 5)",
                                 "generator_version": GENERATOR_VERSION, "seeds": deepcopy(seeds),
                                 "intended_fractions": {"D": fractions["D"],
                                                        "D_fit_given_D": 1 - fractions["D_select_given_D"],
                                                        "E_test_given_E": 1 / FOLDS}},
        "lists": lists, "evaluation_folds": folds}
    if forced:
        candidate["forced_development"] = {"sha256": forced_development_sha256, "count": len(forced),
                                           "groups_canonical_sha256": canonical_hash(sorted(forced))}
        # Private audit evidence; excluded from the public split report.
        candidate["forced_development_groups"] = sorted(forced)
    report = _audit(candidate, rows, members, positive_class, spec["thresholds"])
    return candidate, report


def _positive_groups(ids, lookup, positive_class):
    return len({lookup[i]["group"] for i in ids if lookup[i]["label"] == positive_class})


def _audit(candidate, rows, members, positive_class, thresholds):
    """Structural audit (the v1 auditor on a signed copy) plus the pre-registered pass/fail rules."""
    signed = gd._signed(candidate)
    try:
        audit = gd.audit_global_pools(rows, signed,
                                      expected_manifest_sha256=gd.global_manifest_sha256(rows, candidate))
    except ValueError as error:
        _fail("split_audit_failed", f"structural audit: {error}")
    lookup = {row["id"]: row for row in rows}
    pools = {"D_fit": candidate["lists"]["D_fit"], "D_select": candidate["lists"]["D_select"],
             **{f"E_test_{f['fold']}": f["test_ids"] for f in candidate["evaluation_folds"]}}
    failures = []
    for name, ids in pools.items():
        if audit["diagnostics"][name]["missing_classes"]:
            failures.append(f"{name} lacks a class")
    selected = _positive_groups(pools["D_select"], lookup, positive_class)
    if selected < thresholds["min_select_positive_groups"]:
        failures.append(f"D_select has {selected} positive groups (< {thresholds['min_select_positive_groups']})")
    for fold in range(FOLDS):
        count = _positive_groups(pools[f"E_test_{fold}"], lookup, positive_class)
        if count < thresholds["min_eval_fold_positive_groups"]:
            failures.append(f"E_test_{fold} has {count} positive groups "
                            f"(< {thresholds['min_eval_fold_positive_groups']})")
    if failures:
        _fail("split_audit_failed", "; ".join(failures) + " (no retry, no fallback: escalate)")
    summary = {name: {"images": audit["diagnostics"][name]["images"], "groups": audit["diagnostics"][name]["groups"],
                      "positive_groups": _positive_groups(candidate["lists"][name] if name in candidate["lists"]
                                                          else pools[name], lookup, positive_class),
                      "class_counts": audit["diagnostics"][name]["class_counts"]}
               for name in ("D", "D_fit", "D_select", "E", *[f"E_test_{f}" for f in range(FOLDS)])}
    return {"algorithm": candidate["partition_provenance"]["algorithm"],
            **({"forced_development": deepcopy(candidate["forced_development"])}
               if "forced_development" in candidate else {}),
            "groups": len(members), "records": len(rows), "pools": summary,
            "global_manifest_sha256": gd.global_manifest_sha256(rows, candidate),
            "candidate_sha256": signed["sha256"], "eligibility": audit["eligibility"]}
