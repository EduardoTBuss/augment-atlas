"""Offline, restartable acquisition of the adjudicated RSNA Pneumonia 2018 release.

python -m augment_atlas.acquisition.rsna2018 --raw RAW --out ROOT --config JSON [verify]
The shared ISIC ledger, lock, atomic writers and divergence/privacy utilities are reused.
Only the parent writes; at most two decoded images are in flight.
"""
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import signal
from struct import error as struct_error
import sys
import zipfile

from ..contracts import atomic_json, canonical_hash, file_hash
from .config import ConfigError
from .derive import decoder_versions, PillowUnavailable
from .diskguard import ProcessLock
from .errors import AcquisitionError, DivergenceError, Incomplete, IntegrityError, Interrupted
from .isic2020 import apply_acceptance, divergence, pending
from .ledger import Ledger
from .pipeline import _atomic_bytes, _csv_bytes, _fsync_dir
from .summary import privacy_lint, public_divergences
from .zipcd import CDEntry
from .rsna_dicom import derive

SCHEMA = "augment-atlas/rsna2018-acquisition-config/v1"
PREPROCESSING = {"id": "rsna2018-derived-512-v1", "source_size": [1024, 1024],
                 "size": [512, 512], "mode": "L", "bits": 8, "format": "PNG",
                 "resample": "BILINEAR", "windowing": False, "keep_metadata": False}
POLICY = {"group": "nih-img-id-eight-digit-patient-prefix",
          "label_group": "Calculated",
          "labels": {"Lung Opacity": 1, "No Lung Opacity / Not Normal": 0, "Normal": 0},
          "missing_label": "exclude:no_consensus_label",
          "duplicates": "exclude-all-decoded-pixel-sha256"}
COUNT_KEYS = {"images", "labeled_images", "no_consensus_label", "pixel_duplicate_groups",
              "pixel_duplicate_images", "pixel_duplicate_labeled_images",
              "images_included", "patients_included"}
ROLES = {"images_zip", "mappings", "annotations"}
NIH = re.compile(r"^[0-9]{8}_[0-9]{3}\.png$")
UID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_DICOM_BYTES = 64 * 1024 * 1024


def _require(ok, message):
    if not ok:
        raise ConfigError(message)


def load_config(path):
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError("cannot read RSNA config") from exc
    _require(isinstance(cfg, dict) and cfg.get("schema") == SCHEMA, "invalid RSNA config schema")
    _require(cfg.get("dataset", {}).get("name") == "rsna2018", "dataset.name must be rsna2018")
    _require(isinstance(cfg["dataset"].get("license"), str) and cfg["dataset"]["license"],
             "dataset.license required")
    _require(cfg.get("preprocessing") == PREPROCESSING, "preprocessing must match approved 512-v1")
    _require(cfg.get("policy") == POLICY, "policy must match approved NIH/Calculated/exclude-all decisions")
    _require(isinstance(cfg.get("sources"), dict) and set(cfg["sources"]) == ROLES, "invalid source roles")
    for source in cfg["sources"].values():
        _require(isinstance(source, dict), "source must be an object")
        _require(isinstance(source.get("filename"), str) and
                 re.fullmatch(r"[A-Za-z0-9_.-]+", source["filename"]) and
                 source["filename"] not in (".", ".."), "source requires a plain filename")
        _require(isinstance(source.get("sha256"), str) and
                 re.fullmatch(r"[a-f0-9]{64}", source["sha256"]), "source requires pinned SHA256")
    _require(len({s["filename"] for s in cfg["sources"].values()}) == 3, "source filenames must differ")
    _require(isinstance(cfg.get("expected"), dict) and set(cfg["expected"]) == COUNT_KEYS,
             "expected must contain all RSNA count fields")
    for count in cfg["expected"].values():
        _require(type(count) is int and count >= 0, "expected counts must be nonnegative integers")
    privacy = cfg.get("privacy", {}).get("patterns")
    _require(isinstance(privacy, list) and privacy and all(isinstance(x, str) and x for x in privacy),
             "privacy.patterns required")
    try:
        for pattern in privacy:
            re.compile(pattern)
    except re.error as exc:
        raise ConfigError("invalid privacy regex") from exc
    accepted = cfg.get("accepted_divergences", [])
    _require(isinstance(accepted, list), "accepted_divergences must be a list")
    for item in accepted:
        _require(isinstance(item, dict) and item.get("field") in {"counts." + x for x in COUNT_KEYS}
                 and type(item.get("observed")) is int
                 and isinstance(item.get("resolution"), str) and item["resolution"].strip()
                 and isinstance(item.get("decided_by"), str) and item["decided_by"].strip(),
                 "acceptance requires a count field, observed, resolution and decided_by")
    return cfg


def identity(cfg):
    return canonical_hash({k: v for k, v in cfg.items() if k != "accepted_divergences"})


def _sources(raw, cfg):
    sources = {}
    for role, spec in sorted(cfg["sources"].items()):
        path = Path(raw) / spec["filename"]
        if not path.is_file():
            raise Incomplete("a pinned local source is missing")
        digest = file_hash(path)
        if digest != spec["sha256"]:
            # Hash changes are hard failures, never accept-able via count overrides.
            raise DivergenceError("source SHA256 mismatch for " + role)
        sources[role] = {"filename": spec["filename"], "sha256": digest, "bytes": path.stat().st_size}
    return sources


def reconcile(mapping, annotations):
    """Strict mapping bijection and Calculated labels, with Study UID fallback."""
    if not isinstance(mapping, list) or not mapping:
        raise IntegrityError("mapping must be a nonempty list")
    records, img_ids, subset_ids, studies = {}, set(), set(), defaultdict(list)
    for row in mapping:
        if not isinstance(row, dict) or not NIH.fullmatch(str(row.get("img_id", ""))):
            raise IntegrityError("invalid mapping NIH img_id")
        sop, study, subset = (row.get(k) for k in ("SOPInstanceUID", "StudyInstanceUID", "subset_img_id"))
        if not isinstance(sop, str) or not UID.fullmatch(sop) or not isinstance(study, str) or not UID.fullmatch(study):
            raise IntegrityError("invalid mapping SOP/Study UID")
        if not isinstance(subset, str) or not subset or sop in records or row["img_id"] in img_ids or subset in subset_ids:
            raise IntegrityError("mapping image identities must be unique")
        records[sop] = {**row, "group": row["img_id"][:8], "label": None}
        img_ids.add(row["img_id"])
        subset_ids.add(subset)
        studies[study].append(sop)
    label_index = {}
    for group in annotations["labelGroups"]:
        for label in group["labels"]:
            if label["id"] in label_index:
                raise IntegrityError("repeated annotation label id")
            label_index[label["id"]] = (group["name"].strip(), label["name"].strip())
    datasets = annotations["datasets"]
    if len(datasets) != 1:
        raise IntegrityError("expected one MD.ai dataset")
    dataset = datasets[0]
    ann_studies = [s["StudyInstanceUID"] for s in dataset["studies"]]
    if len(set(ann_studies)) != len(ann_studies) or set(ann_studies) != set(studies):
        raise IntegrityError("annotation/mapping study coverage differs")
    labels, fallback = defaultdict(set), 0
    for ann in dataset["annotations"]:
        group, name = label_index[ann["labelId"]]
        sop = ann.get("SOPInstanceUID")
        study = ann.get("StudyInstanceUID")
        if not sop:
            candidates = studies.get(study, [])
            if len(candidates) != 1:
                raise IntegrityError("annotation Study UID does not resolve uniquely")
            sop = candidates[0]
            fallback += 1
        if sop not in records or (study and study != records[sop]["StudyInstanceUID"]):
            raise IntegrityError("annotation does not resolve to its mapping image")
        if group == "Calculated":
            if name not in POLICY["labels"]:
                raise IntegrityError("unexpected Calculated label")
            labels[sop].add(name)
    for sop, names in labels.items():
        if len(names) != 1:
            raise IntegrityError("conflicting Calculated labels require review")
        records[sop]["label"] = POLICY["labels"][next(iter(names))]
    return records, fallback


def _members(archive, records):
    entries, members, seen = [], [], set()
    for idx, info in enumerate(sorted(archive.infolist(), key=lambda i: i.header_offset)):
        name = PurePosixPath(info.filename)
        if "\\" in info.filename or name.is_absolute() or ".." in name.parts:
            raise IntegrityError("unsafe ZIP member name")
        sop = None if info.is_dir() else name.stem
        if not info.is_dir():
            if name.suffix != ".dcm" or sop not in records or sop in seen:
                raise IntegrityError("ZIP/mapping image coverage differs or repeats")
            if info.file_size > MAX_DICOM_BYTES or info.flag_bits & 1:
                raise IntegrityError("oversized/encrypted DICOM ZIP member")
            seen.add(sop)
        entries.append(CDEntry(index=idx, name=info.filename, raw_name=info.filename.encode(),
                               method=info.compress_type, flags=info.flag_bits, crc32=info.CRC,
                               compressed_size=info.compress_size, uncompressed_size=info.file_size,
                               local_header_offset=info.header_offset, is_dir=info.is_dir()))
        members.append({"index": idx, "position": idx, "kind": "dir" if info.is_dir() else "image", "image": sop})
    if seen != set(records):
        raise IntegrityError("ZIP/mapping image coverage differs")
    return entries, members


def _relative(record):
    return "derived-512-v1/images/" + record["img_id"]


def _complete(ledger, row, info, body, out, records):
    rec = records[row["image"]]
    if (info["original"]["SOPInstanceUID"] != row["image"] or
            info["original"]["StudyInstanceUID"] != rec["StudyInstanceUID"]):
        raise IntegrityError("DICOM UIDs disagree with mapping/member")
    path = out / _relative(rec)
    _atomic_bytes(path, body)
    _fsync_dir(path.parent)
    ledger.mark_derived(row["idx"], info)


def _run(archive, entries, ledger, out, records, workers):
    jobs = deque()
    context = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
               if workers else nullcontext(None))
    # All jobs fit in a bounded window; excluded images also need a decoded pixel hash.
    with context as pool:
        def drain():
            row, future = jobs.popleft()
            info, body = future.result()
            _complete(ledger, row, info, body, out, records)
        for row in ledger.members(kind="image"):
            if row["state"] == "DERIVED":
                continue
            if row["state"] not in ("PENDING", "FETCHED"):
                raise IntegrityError("unexpected RSNA ledger state")
            data = archive.read(entries[row["idx"]].name)  # zipfile checks CRC32
            digest = hashlib.sha256(data).hexdigest()
            if row["state"] == "PENDING":
                ledger.mark_fetched(row["idx"], digest)
            elif digest != row["sha256_original"]:
                raise IntegrityError("FETCHED original differs from ledger")
            if pool:
                jobs.append((row, pool.submit(derive, data)))
                if len(jobs) >= workers:
                    drain()
            else:
                info, body = derive(data)
                _complete(ledger, row, info, body, out, records)
        while jobs:
            drain()


def _verify(cfg, out, ledger, records, sources, fallback):
    if ledger.counts()["PENDING"] or ledger.counts()["FETCHED"]:
        raise Incomplete("acquisition incomplete: resume without verify")
    rows = {row["image"]: row for row in ledger.members(kind="image")}
    if set(rows) != set(records) or any(row["state"] != "DERIVED" for row in rows.values()):
        raise IntegrityError("RSNA ledger image coverage/state differs")
    by_hash, infos = defaultdict(list), {}
    for sop, row in rows.items():
        info = json.loads(row["info"])
        path = out / _relative(records[sop])
        if not path.is_file() or path.stat().st_size != info["derived"]["bytes"] or file_hash(path) != info["derived"]["sha256"]:
            raise IntegrityError("derived PNG missing or differs from ledger")
        if (info["original"]["SOPInstanceUID"] != sop or
                info["original"]["StudyInstanceUID"] != records[sop]["StudyInstanceUID"]):
            raise IntegrityError("ledger DICOM identity mismatch")
        infos[sop] = info
        by_hash[info["pixel_sha256"]].append(sop)
    expected_files = {rec["img_id"] for rec in records.values()}
    actual_files = {path.name for path in (out / "derived-512-v1/images").iterdir()}
    if actual_files != expected_files:
        raise IntegrityError("unexpected files in derived image directory")
    duplicate_groups = [sorted(sops) for sops in by_hash.values() if len(sops) > 1]
    duplicates = {sop for sops in duplicate_groups for sop in sops}
    included, private, manifest_rows, identifiers = [], [], [], set()
    for sop, rec in sorted(records.items()):
        info = infos[sop]
        reasons = []
        if rec["label"] is None:
            reasons.append("no_consensus_label")
        if sop in duplicates:
            reasons.append("pixel_duplicate")
        status = "excluded" if reasons else "included"
        path = _relative(rec)
        private.append({**rec, "status": status, "reason": reasons[0] if reasons else None,
                        "reasons": reasons, "ViewPosition": info["original"]["ViewPosition"],
                        "original": {"sha256": rows[sop]["sha256_original"],
                                     "pixel_sha256": info["pixel_sha256"]},
                        "derived": {**info["derived"], "path": path}})
        identifiers.update({sop, rec["StudyInstanceUID"], rec["subset_img_id"], rec["img_id"],
                            rec["group"], Path(rec["img_id"]).stem})
        if status == "included":
            included.append({"id": path, "path": path, "group": rec["group"], "label": rec["label"]})
            manifest_rows.append((path, rec["group"], rec["label"], info["derived"]["sha256"]))
    labeled = sum(rec["label"] is not None for rec in records.values())
    counts = {"images": len(records), "labeled_images": labeled,
              "no_consensus_label": len(records) - labeled,
              "pixel_duplicate_groups": len(duplicate_groups), "pixel_duplicate_images": len(duplicates),
              "pixel_duplicate_labeled_images": sum(records[sop]["label"] is not None for sop in duplicates),
              "images_included": len(included), "patients_included": len({r["group"] for r in included})}
    divergences = [divergence("counts." + key, expected, counts[key], "rsna2018")
                   for key, expected in sorted(cfg["expected"].items()) if counts[key] != expected]
    apply_acceptance(divergences, cfg.get("accepted_divergences", []))
    manifest = out / "manifest"
    atomic_json(manifest / "divergences.json", {"schema": "augment-atlas/acquisition-divergences/v1",
                                               "stage": "verify", "divergences": divergences})
    if pending(divergences):
        raise DivergenceError("RSNA count divergence; see private manifest/divergences.json")
    records_doc = {"schema": "augment-atlas/records/v1", "records": included}
    records_doc["canonical_hash"] = canonical_hash(records_doc)
    manifest_body = _csv_bytes(manifest_rows, ("path", "group", "label", "sha256"))
    private_doc = {"schema": "augment-atlas/dataset-manifest-private/v1", "dataset": cfg["dataset"],
                   "preprocessing": cfg["preprocessing"], "policy": cfg["policy"],
                   "expected": cfg["expected"], "observed_counts": counts, "divergences": divergences,
                   "acquisition": {"config_sha256": canonical_hash(cfg), "config_identity_sha256": identity(cfg),
                                   "sources": sources, "decoder": decoder_versions(),
                                   "annotations_resolved_by_study": fallback},
                   "duplicates": {"policy": POLICY["duplicates"], "groups": sorted(duplicate_groups)},
                   "records": private}
    private_doc["sha256"] = canonical_hash(private_doc)
    _atomic_bytes(manifest / "manifest.csv", manifest_body)
    atomic_json(manifest / "records.json", records_doc)
    atomic_json(manifest / "manifest-private.json", private_doc)
    files = {name: file_hash(manifest / name) for name in
             ("manifest.csv", "records.json", "manifest-private.json", "divergences.json")}
    document = {"schema": "augment-atlas/dataset-summary-public/v1",
                "dataset": cfg["dataset"]["name"], "license": cfg["dataset"]["license"],
                "preprocessing_id": PREPROCESSING["id"], "config_sha256": canonical_hash(cfg),
                "config_identity_sha256": identity(cfg), "counts": counts,
                "class_images": dict(sorted(Counter(str(r["label"]) for r in included).items())),
                "source_hashes": {role: spec["sha256"] for role, spec in sources.items()},
                "hashes": files, "divergences": public_divergences(divergences)}
    # Mandatory patterns cannot be removed by an operational config edit.
    patterns = [r"[0-9]{8}_[0-9]{3}\.png",
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                r"\b[0-9]+(?:\.[0-9]+){3,}\b", *cfg["privacy"]["patterns"]]
    lint = privacy_lint(json.dumps(document, sort_keys=True), patterns, identifiers)
    if lint["pattern_hits"] or lint["id_hits"]:
        raise IntegrityError("RSNA public summary failed privacy lint")
    document["privacy_lint"] = lint
    document["sha256"] = canonical_hash(document)
    # Include the lint metadata and final digest in the last privacy check too.
    final_lint = privacy_lint(json.dumps(document, sort_keys=True), patterns, identifiers)
    if final_lint["pattern_hits"] or final_lint["id_hits"]:
        raise IntegrityError("RSNA public summary failed final privacy lint")
    public_path = out / "public/rsna2018-summary-public.json"
    atomic_json(public_path, document)
    _fsync_dir(manifest)
    _fsync_dir(public_path.parent)
    # Publication marker LAST, after successful lint and all artifact renames.
    atomic_json(manifest / "READY.json", {
        "schema": "augment-atlas/dataset-ready/v1", "dataset": "rsna2018",
        "preprocessing_id": PREPROCESSING["id"], "config_sha256": canonical_hash(cfg),
        "config_identity_sha256": identity(cfg), "files": files,
        "public_summary_sha256": file_hash(public_path)})
    _fsync_dir(manifest)
    return document


def acquire(raw, out, config, *, verify=False, workers=0):
    if type(workers) is not int or not 0 <= workers <= 2:
        raise ConfigError("workers must be 0 (inline), 1 or 2")
    cfg = load_config(config)
    out = Path(out).absolute()
    state, manifest = out / "acquisition-v1", out / "manifest"
    if verify and not (state / "ledger.sqlite").exists():
        raise Incomplete("no ledger: run acquisition first")
    with ProcessLock(state / "lock"):
        # A failed re-verification cannot leave an old validity marker/public summary.
        (manifest / "READY.json").unlink(missing_ok=True)
        (out / "public/rsna2018-summary-public.json").unlink(missing_ok=True)
        sources = _sources(raw, cfg)
        source_path = lambda role: Path(raw) / cfg["sources"][role]["filename"]
        records, fallback = reconcile(json.loads(source_path("mappings").read_text(encoding="utf-8")),
                                      json.loads(source_path("annotations").read_text(encoding="utf-8")))
        for path in (out, state, manifest, out / "derived-512-v1/images", out / "public"):
            path.mkdir(parents=True, exist_ok=True)
            if os.name == "posix":
                os.chmod(path, 0o700)
        with zipfile.ZipFile(source_path("images_zip")) as archive, Ledger(state / "ledger.sqlite") as ledger:
            entries, members = _members(archive, records)
            pins = {"dataset": "rsna2018", "config_identity_sha256": identity(cfg),
                    "source_hashes": canonical_hash(sources), "decoder": canonical_hash(decoder_versions())}
            if verify and any(ledger.meta().get(key) != str(value) for key, value in pins.items()):
                raise DivergenceError("verify identity differs from pinned ledger")
            ledger.pin(pins)
            ledger.init_members(entries, members, {s for s, r in records.items() if r["label"] is not None})
            atomic_json(state / "config.json", cfg)
            if not verify:
                session = ledger.start_session("rsna2018-full", None)
                try:
                    _run(archive, entries, ledger, out, records, workers)
                except BaseException as exc:
                    ledger.end_session(session, None, getattr(exc, "exit_code", 2), type(exc).__name__)
                    raise
                else:
                    ledger.end_session(session, None, 0)
                # Recover only this pipeline's atomic-write leftovers after a killed parent.
                for rec in records.values():
                    path = (out / _relative(rec)).with_suffix(".png.tmp")
                    path.unlink(missing_ok=True)
            return _verify(cfg, out, ledger, records, sources, fallback)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("mode", nargs="?", choices=("verify",))
    args = parser.parse_args(argv)
    old_handlers = {}
    def stop(signum, frame):
        raise Interrupted("RSNA acquisition interrupted; repeat acquisition to resume")
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, stop)
        result = acquire(args.raw, args.out, args.config, verify=args.mode == "verify", workers=args.workers)
        print(json.dumps({"counts": result["counts"], "sha256": result["sha256"]}, sort_keys=True))
        return 0
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 64
    except AcquisitionError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, KeyError, TypeError, struct_error, zipfile.BadZipFile, PillowUnavailable) as exc:
        # No source bytes or raw identifiers in the CLI diagnostic.
        print("RSNA acquisition failed: " + type(exc).__name__, file=sys.stderr)
        return 2
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

if __name__ == "__main__":
    raise SystemExit(main())
