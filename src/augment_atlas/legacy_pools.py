"""Convert fixed INbreast proposal lists to v1; no sampling or split generation.

The legacy schema has no intended fractions or preregistered thresholds. Missing
provenance is hashed as an explicit declaration of missing evidence, never as
an assertion about preprocessing, duplicate detection or exclusions.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from . import global_development as gd
from .contracts import canonical_hash

ALGORITHM = "legacy-proposal-v1-converted"
LEGACY_PROTOCOL = "global-development-proposal-v1"
NOT_PREREGISTERED = "not_preregistered_legacy"
_LEGACY_KEYS = {"protocol", "scientific_status", "sklearn", "seeds", "source_fold_sha256",
                "development", "search_train", "search_selection", "evaluation", "folds"}


def _fail(message):
    # Never interpolate legacy values, record IDs, paths or group identifiers.
    raise ValueError("legacy_conversion_invalid: " + message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate JSON object key (message withheld: it may name a record identifier)")
        result[key] = value
    return result


def convert_legacy_pools(records, legacy_bytes, *, source):
    """Return an unsigned v1 candidate and audit; SHA-256 binds the parsed bytes.

    Only the described proposal-v1 format and the inbreast_csv source are
    accepted. Sorting canonicalizes lists without changing memberships. All
    image paths must match manifest paths exactly; no basename or alias matching.
    """
    try:
        legacy = json.loads(legacy_bytes, object_pairs_hook=_unique)
    except (UnicodeError, ValueError, RecursionError):
        _fail("legacy JSON unreadable (message withheld: it may name a record identifier)")
    gd._json(legacy, "legacy document")
    gd._object(legacy, _LEGACY_KEYS, "legacy document")
    if legacy["protocol"] != LEGACY_PROTOCOL or legacy["scientific_status"] != "provisional-feasibility-only":
        _fail("unsupported legacy protocol or scientific status")
    gd._text(legacy["sklearn"], "legacy sklearn")
    gd._hash(legacy["source_fold_sha256"], "legacy source fold hash")
    gd._object(legacy["seeds"], {"development", "selection", "evaluation"}, "legacy seeds")
    for value in legacy["seeds"].values():
        gd._int(value, "legacy seed")
    if legacy["seeds"] != {"development": 17, "selection": 29, "evaluation": 31}:
        _fail("INbreast 16/10 pools require seeds 17/29/31; no reseeding allowed")
    gd._object(source, {"adapter", "manifest", "manifest_sha256"}, "dataset source")
    if source["adapter"] != "inbreast_csv":
        _fail("conversion requires the inbreast_csv manifest adapter")
    gd._hash(source["manifest_sha256"], "source manifest hash")
    rows = gd._records(deepcopy(records))
    by_path = {row["path"]: row["id"] for row in rows}

    def ids(value):
        if type(value) is not list or not value or any(type(item) is not str for item in value):
            _fail("each pool and fold list must be a nonempty image path list")
        if len(value) != len(set(value)):
            _fail("duplicate image in a legacy list")
        if not set(value) <= by_path.keys():
            _fail("unknown image path (message withheld: it names a record identifier)")
        return sorted(by_path[path] for path in value)

    lists = {name: ids(legacy[key]) for name, key in (
        ("D", "development"), ("D_fit", "search_train"),
        ("D_select", "search_selection"), ("E", "evaluation"))}
    if type(legacy["folds"]) is not list or len(legacy["folds"]) != 5:
        _fail("exactly five legacy folds required")
    folds = []
    for index, fold in enumerate(legacy["folds"]):
        gd._object(fold, {"fold", "train", "test", "diagnostics"}, "legacy fold")
        if type(fold["fold"]) is not int or fold["fold"] != index:
            _fail("fold indices must be ordered exactly 0..4")
        if type(fold["diagnostics"]) is not dict:
            _fail("fold diagnostics must be an object (not trusted as audit evidence)")
        test, train = ids(fold["test"]), ids(fold["train"])
        if set(train) != set(lists["D"]) | (set(lists["E"]) - set(test)):
            _fail("fold.train must equal D union (E minus test)")
        folds.append({"fold": index, "test_ids": test})

    sha = hashlib.sha256(legacy_bytes).hexdigest()
    missing = lambda aspect: canonical_hash({"status": "not_documented_legacy", "aspect": aspect,
                                             "legacy_sha256": sha})
    candidate = {
        "schema_version": 1, "protocol": gd.PROTOCOL, "scientific_status": "provisional",
        "dataset": {"id": "inbreast", "version": ALGORITHM},
        "records_sha256": canonical_hash(rows), "source_sha256": source["manifest_sha256"],
        "provenance": {"preprocessing_sha256": missing("preprocessing"),
                       "labels_sha256": canonical_hash([{"path": r["path"], "label": r["label"]} for r in rows]),
                       "duplicates_sha256": missing("duplicates"), "exclusions_sha256": missing("exclusions")},
        "grouping": {"namespace": "inbreast manifest group_key", "source": "inbreast_csv.group_key",
                     "identity_level": "group", "linkage_status": "unverified", "evidence_sha256": None,
                     "prior_exposure": "known"},
        "task": {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
                 "decision_rule": "fixed_threshold", "missing_policy": "undefined_no_drop",
                 "requires_patient_identity": True},
        "partition_provenance": {
            "algorithm": ALGORITHM, "version": "1", "source": "fixed legacy lists; no resampling; thresholds not_preregistered_legacy",
            "generator_version": "augment_atlas.legacy_pools v1 (converter, no generator)",
            "seeds": deepcopy(legacy["seeds"]),
            "converted_from": {"protocol": LEGACY_PROTOCOL, "sha256": sha},
            "intended_fractions": {key: NOT_PREREGISTERED for key in ("D", "D_fit_given_D", "E_test_given_E")}},
        "lists": lists, "evaluation_folds": folds}
    manifest = gd.global_manifest_sha256(rows, candidate)
    # This audits exact coverage, D_fit/D_select, E folds and whole-group closure.
    audit = gd.audit_global_pools(rows, gd._signed(candidate), expected_manifest_sha256=manifest)
    return candidate, audit
