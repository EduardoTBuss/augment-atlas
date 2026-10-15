"""Private classification manifests and persistent group-disjoint partitions.

Imports are stdlib-only; image/torch dependencies are loaded by the trainer.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path, PurePosixPath

from .contracts import atomic_json, canonical_hash, file_hash


def data_root(cfg: dict) -> Path:
    return Path(os.environ.get("ATLAS_DATA_ROOT") or cfg["dataset"].get("root", "/data")).resolve()


def confined_path(root: Path, relative: str) -> Path:
    # Also reject Windows-looking paths when validating from Linux.
    clean = relative.replace("\\", "/")
    if not clean or PurePosixPath(clean).is_absolute() or ":" in clean or ".." in PurePosixPath(clean).parts:
        raise ValueError("dataset paths must be relative and confined to data root")
    result = (root / clean).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("dataset path escapes root")
    return result


def load_records(cfg: dict, check_files: bool = True) -> list[dict]:
    dataset = cfg["dataset"]
    if dataset["adapter"] == "synthetic":
        return [{"id": f"synthetic-{i:04d}", "path": f"synthetic-{i:04d}",
                 "group": f"synthetic-group-{i:04d}", "label": i % cfg["model"]["num_classes"]}
                for i in range(dataset.get("synthetic_groups", 12))]
    root = data_root(cfg)
    manifest = confined_path(root, dataset["manifest"])
    if dataset.get("manifest_sha256") and file_hash(manifest) != dataset["manifest_sha256"]:
        raise ValueError("private manifest checksum mismatch")
    names = ("processed_image_filename", "group_key", "classification") if dataset["adapter"] == "inbreast_csv" else ("path", "group", "label")
    records = []
    seen_paths = set()
    seen_physical_files = set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(names).issubset(reader.fieldnames or []):
            raise ValueError("manifest lacks required classification columns")
        for row in reader:
            path, group, label_text = (row[name].strip() for name in names)
            prefix = dataset.get("image_dir", "")
            path = str(PurePosixPath(prefix) / path) if prefix else path
            image = confined_path(root, path)
            if check_files and not image.is_file():
                raise ValueError("private manifest references a missing image")
            # Identity follows the resolved file, not its spelling in the CSV.
            # This collapses ./ aliases and in-root symlinks before partitioning.
            normalized_path = image.relative_to(root).as_posix()
            if normalized_path in seen_paths:
                raise ValueError("manifest must contain unique physical image records")
            seen_paths.add(normalized_path)
            # Distinct resolved paths can still be hard links to the same file.
            # Inodes remain local validation inputs, never portable manifest IDs.
            if image.exists():
                metadata = image.stat()
                physical_id = (metadata.st_dev, metadata.st_ino)
                if metadata.st_ino:
                    if physical_id in seen_physical_files:
                        raise ValueError("manifest must contain unique physical image records")
                    seen_physical_files.add(physical_id)
            expected_image_hash = row.get("sha256", "").strip()
            if dataset.get("require_image_hashes") and not expected_image_hash:
                raise ValueError("manifest image checksum is required")
            if expected_image_hash and check_files and file_hash(image) != expected_image_hash:
                raise ValueError("private image checksum mismatch")
            if not group:
                raise ValueError("empty group identifier")
            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError("classification labels must be integer class indices") from exc
            if not 0 <= label < cfg["model"]["num_classes"]:
                raise ValueError("classification label outside configured classes")
            record = {"id": normalized_path, "path": normalized_path, "group": group, "label": label}
            if expected_image_hash:
                record["sha256"] = expected_image_hash
            records.append(record)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("manifest must contain unique nonempty image records")
    return records


def audit_splits(records: list[dict], splits: dict[str, list[str]]) -> None:
    if set(splits) != {"train", "selection", "test"}:
        raise ValueError("splits must contain train, selection, test")
    lookup = {row["id"]: row for row in records}
    if len(lookup) != len(records):
        raise ValueError("duplicate record identifiers")
    observed, groups = set(), set()
    for name in ("train", "selection", "test"):
        ids = splits[name]
        if not ids or len(ids) != len(set(ids)) or set(ids) - lookup.keys() or observed.intersection(ids):
            raise ValueError("split is empty, duplicated, unknown or overlapping")
        current = {lookup[i]["group"] for i in ids}
        if groups.intersection(current):
            raise ValueError("group leakage across train/selection/test")
        observed.update(ids)
        groups.update(current)
    if observed != set(lookup):
        raise ValueError("split lists must account for every manifest image")


def prepare_splits(cfg: dict, records: list[dict], output: str | Path) -> dict:
    """Persist actual lists; never regenerate on resume/library changes.

    group-hash-v1 is an unstratified 60/20/20 technical partition, not the
    future five-fold scientific evaluation. Supplied lists take precedence.
    """
    output = Path(output)
    target = output / "splits.json"
    manifest_hash = canonical_hash(sorted(records, key=lambda r: r["id"]))
    spec = cfg["splits"]
    identity = {"manifest_sha256": manifest_hash, "dataset_version": cfg["dataset"]["version"],
                "split_seed": spec["split_seed"], "fold": spec["fold"], "algorithm": spec["version"]}
    source = cfg["dataset"].get("split_file")
    if target.exists() or source:
        source_path = target if target.exists() else confined_path(data_root(cfg), source)
        document = json.loads(source_path.read_text(encoding="utf-8"))
        if any(document.get(key) != value for key, value in identity.items()):
            raise ValueError("persisted splits do not match manifest/version/seed/fold")
        if document.get("sha256") != canonical_hash({k: v for k, v in document.items() if k != "sha256"}):
            raise ValueError("persisted split checksum mismatch")
    else:
        groups = sorted({row["group"] for row in records}, key=lambda group: canonical_hash([spec["split_seed"], spec["fold"], group]))
        if len(groups) < 6:
            raise ValueError("technical split requires at least six independent groups")
        n_selection = max(1, len(groups) // 5)
        assigned = {group: ("selection" if i < n_selection else "test" if i < 2 * n_selection else "train") for i, group in enumerate(groups)}
        lists = {name: sorted(row["id"] for row in records if assigned[row["group"]] == name) for name in ("train", "selection", "test")}
        document = {**identity, "scientific_status": "provisional", "lists": lists,
                    "groups": {name: sorted({row["group"] for row in records if row["id"] in ids}) for name, ids in lists.items()}}
        document["sha256"] = canonical_hash(document)
    audit_splits(records, document["lists"])
    expected_groups = {name: sorted({row["group"] for row in records if row["id"] in ids}) for name, ids in document["lists"].items()}
    if document.get("groups") != expected_groups:
        raise ValueError("persisted split group lists mismatch")
    if not target.exists():
        atomic_json(target, document)
    return document
