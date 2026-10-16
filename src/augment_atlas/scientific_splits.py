"""Prepare private, group-disjoint outer folds without training or HPO.

Five outer folds each have ONE inner selection holdout: the first split of a
five-way StratifiedGroupKFold on the outer development pool. This is not full
inner cross-validation. Group identifiers are supplied, never inferred to be
patients or assigned a single label. Importing and auditing need only stdlib;
generation imports scikit-learn lazily. Persisted lists, not regeneration,
define subsequent runs even when the installed sklearn version changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath

from .contracts import atomic_json, canonical_hash
from .data import audit_splits

ALGORITHM = "stratified-group-outer-v1"
PARTITIONS = ("train", "selection", "test")
SELECTION_PROTOCOL = "single-holdout-within-outer-train"
FOLDS = 5


def _integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _records(records: list[dict]) -> list[dict]:
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")
    ids, paths = set(), set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("each record must be an object")
        for field in ("id", "path", "group"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"record {field} must be a nonempty string")
        _integer(row.get("label"), "label")
        path = PurePosixPath(row["path"].replace("\\", "/"))
        if not path.parts or path.is_absolute() or ":" in str(path) or ".." in path.parts:
            raise ValueError("record paths must be relative and confined")
        if row["id"] in ids or path.as_posix() in paths:
            raise ValueError("duplicate record identity or image path")
        ids.add(row["id"])
        paths.add(path.as_posix())
    return sorted(records, key=lambda row: row["id"])


def split_diagnostics(records: list[dict]) -> dict:
    """Counts use image labels, including groups containing multiple labels."""
    rows = _records(records)
    labels = sorted({row["label"] for row in rows})
    group_labels: dict[str, set[int]] = {}
    for row in rows:
        group_labels.setdefault(row["group"], set()).add(row["label"])
    return {
        "images": len(rows), "groups": len(group_labels),
        "class_counts": {str(label): sum(row["label"] == label for row in rows) for label in labels},
        "class_group_counts": {str(label): sum(label in values for values in group_labels.values()) for label in labels},
        "mixed_label_groups": sum(len(values) > 1 for values in group_labels.values()),
    }


def _seed(split_seed: int, stage: str, fold: int | None = None) -> int:
    return int(canonical_hash([ALGORITHM, split_seed, stage, fold])[:8], 16)


def _feasibility(records: list[dict], *, stage: str, class_groups: int) -> dict:
    counts = split_diagnostics(records)
    if (len(counts["class_counts"]) < 2 or counts["groups"] < FOLDS
            or min(counts["class_group_counts"].values()) < class_groups):
        raise ValueError(f"{stage} is infeasible for fixed five-fold protocol: {json.dumps(counts, sort_keys=True)}")
    return counts


def _partition_diagnostics(rows: list[dict], lists: dict) -> dict:
    lookup = {row["id"]: row for row in rows}
    return {name: split_diagnostics([lookup[item] for item in lists[name]]) for name in PARTITIONS}


def audit_scientific_split(records: list[dict], document: dict) -> None:
    """Validate one persisted fold without importing sklearn or training code."""
    rows = _records(records)
    if not isinstance(document, dict):
        raise ValueError("split document must be an object")
    _integer(document.get("split_seed"), "split_seed")
    _integer(document.get("fold"), "fold")
    if document["fold"] >= FOLDS:
        raise ValueError("scientific fold must be in 0..4")
    if not isinstance(document.get("dataset_version"), str) or not document["dataset_version"]:
        raise ValueError("dataset_version is required")
    expected = {
        "algorithm": ALGORITHM, "scientific_status": "provisional",
        "manifest_sha256": canonical_hash(rows), "outer_folds": FOLDS,
        "outer_seed": _seed(document["split_seed"], "outer"),
        "selection_protocol": SELECTION_PROTOCOL, "selection_folds": FOLDS,
        "selection_fold": 0, "selection_seed": _seed(document["split_seed"], "selection", document["fold"]),
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError("scientific split identity or fixed holdout protocol mismatch")
    if document.get("sha256") != canonical_hash({key: value for key, value in document.items() if key != "sha256"}):
        raise ValueError("persisted split checksum mismatch")
    lists = document.get("lists")
    if not isinstance(lists, dict) or any(not isinstance(lists.get(name), list) for name in PARTITIONS):
        raise ValueError("split lists must contain train, selection, test arrays")
    if any(any(not isinstance(item, str) for item in lists[name]) for name in PARTITIONS):
        raise ValueError("split identifiers must be strings")
    audit_splits(rows, lists)
    lookup = {row["id"]: row for row in rows}
    groups = {name: sorted({lookup[item]["group"] for item in lists[name]}) for name in PARTITIONS}
    if document.get("groups") != groups:
        raise ValueError("persisted split group lists mismatch")
    classes = {row["label"] for row in rows}
    if len(classes) < 2 or any({lookup[item]["label"] for item in lists[name]} != classes for name in PARTITIONS):
        raise ValueError("every partition must contain all manifest classes; fixed seed is not retried")
    diagnostics = {"manifest": split_diagnostics(rows), "partitions": _partition_diagnostics(rows, lists)}
    if document.get("diagnostics") != diagnostics:
        raise ValueError("persisted split diagnostics mismatch")
    if not isinstance(document.get("sklearn_version"), str) or not document["sklearn_version"]:
        raise ValueError("sklearn generation version is required")


def audit_scientific_folds(records: list[dict], documents: list[dict]) -> None:
    """Require exactly one external test appearance per image across five folds."""
    rows = _records(records)
    if not isinstance(documents, list) or len(documents) != FOLDS:
        raise ValueError("exactly five outer fold documents are required")
    for document in documents:
        audit_scientific_split(rows, document)
    if {document["fold"] for document in documents} != set(range(FOLDS)):
        raise ValueError("outer fold indices must be exactly 0..4")
    for field in ("manifest_sha256", "dataset_version", "split_seed", "algorithm", "sklearn_version"):
        if len({document[field] for document in documents}) != 1:
            raise ValueError("outer fold identity mismatch")
    appearances = Counter(item for document in documents for item in document["lists"]["test"])
    if appearances != Counter({row["id"]: 1 for row in rows}):
        raise ValueError("every image must occur in external test exactly once across five folds")


def prepare_scientific_splits(records: list[dict], *, dataset_version: str,
                              split_seed: int, outer_folds: int = FOLDS) -> list[dict]:
    """Generate five candidate fold documents, failing rather than retrying seeds.

    No hyperparameters are selected here. Labels and groups are supplied per
    image, and the protected outer test is never used to choose an inner split.
    """
    rows = _records(records)
    _integer(split_seed, "split_seed")
    if isinstance(outer_folds, bool) or not isinstance(outer_folds, int) or outer_folds != FOLDS:
        raise ValueError("stratified-group-outer-v1 requires exactly five outer folds")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("dataset_version is required")
    manifest_counts = _feasibility(rows, stage="outer split", class_groups=FOLDS)
    try:
        import sklearn
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise RuntimeError("generating scientific splits requires scikit-learn; auditing persisted lists does not") from exc
    outer_seed = _seed(split_seed, "outer")
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=outer_seed)
    documents = []
    for fold, (development, test) in enumerate(splitter.split(
            [[0]] * len(rows), [row["label"] for row in rows], [row["group"] for row in rows])):
        pool = [rows[int(index)] for index in development]
        _feasibility(pool, stage=f"outer fold {fold} selection holdout", class_groups=2)
        selection_seed = _seed(split_seed, "selection", fold)
        selection_splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=selection_seed)
        train, selection = next(selection_splitter.split(
            [[0]] * len(pool), [row["label"] for row in pool], [row["group"] for row in pool]))
        lists = {"train": sorted(pool[int(index)]["id"] for index in train),
                 "selection": sorted(pool[int(index)]["id"] for index in selection),
                 "test": sorted(rows[int(index)]["id"] for index in test)}
        lookup = {row["id"]: row for row in rows}
        document = {
            "manifest_sha256": canonical_hash(rows), "dataset_version": dataset_version,
            "split_seed": split_seed, "fold": fold, "algorithm": ALGORITHM,
            "scientific_status": "provisional", "outer_folds": FOLDS, "outer_seed": outer_seed,
            "selection_protocol": SELECTION_PROTOCOL, "selection_folds": FOLDS,
            "selection_fold": 0, "selection_seed": selection_seed,
            "sklearn_version": sklearn.__version__, "lists": lists,
            "groups": {name: sorted({lookup[item]["group"] for item in lists[name]}) for name in PARTITIONS},
            "diagnostics": {"manifest": manifest_counts, "partitions": _partition_diagnostics(rows, lists)},
        }
        document["sha256"] = canonical_hash(document)
        audit_scientific_split(rows, document)
        documents.append(document)
    audit_scientific_folds(rows, documents)
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True, help="private JSON array of id/path/group/label records")
    parser.add_argument("--output", required=True, help="private output directory; contains image and group identifiers")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--outer-folds", default=FOLDS, type=int, choices=[FOLDS])
    args = parser.parse_args()
    records = json.loads(Path(args.manifest_json).read_text(encoding="utf-8-sig"))
    output = Path(args.output)
    targets = [output / f"fold-{fold}.json" for fold in range(FOLDS)]
    present = [target.exists() for target in targets]
    if any(present):
        if not all(present):
            raise ValueError("partial persisted fold set; refusing regeneration or overwrite")
        documents = [json.loads(target.read_text(encoding="utf-8")) for target in targets]
        audit_scientific_folds(records, documents)
        if any(doc["dataset_version"] != args.dataset_version or doc["split_seed"] != args.split_seed
               for doc in documents):
            raise ValueError("persisted fold set does not match dataset version or seed")
    else:
        documents = prepare_scientific_splits(records, dataset_version=args.dataset_version,
                                             split_seed=args.split_seed, outer_folds=args.outer_folds)
        for target, document in zip(targets, documents):
            atomic_json(target, document)
    print(json.dumps({"algorithm": ALGORITHM, "outer_folds": FOLDS,
                      "selection_protocol": SELECTION_PROTOCOL,
                      "sha256": [document["sha256"] for document in documents]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
