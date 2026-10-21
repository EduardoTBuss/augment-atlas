"""ISIC 2020 reconciliation: labels (GT v2, checked 1:1 against GT v1), duplicate pairs,
ZIP member classification and ZIP<->CSV cross-check. Pure functions over parsed data.

Two failure grades:
* hard (DivergenceError, exit 3 always): the documents cannot be used at all
  (duplicate or malformed image names, empty patient_id, non-binary target);
* soft divergences (dicts): an observed count differs from the documented one or
  from the other document. They are recorded with private examples, block READY
  until accepted in the config, and stop strict runs with exit 3. Never silenced.
"""

from collections import Counter
import itertools

from .errors import DivergenceError, IntegrityError

EXAMPLES_MAX = 10


def divergence(field, documented, observed, source, examples=()):
    return {"field": field, "documented": documented, "observed": observed, "source": source,
            "examples": sorted(examples)[:EXAMPLES_MAX], "accepted": False, "resolution": None, "decided_by": None}


def apply_acceptance(divergences, accepted):
    """Mark a divergence accepted iff the config records a decision for the same field AND observed value."""
    decisions = {(item["field"], repr(item["observed"])): item for item in accepted}
    for item in divergences:
        decision = decisions.get((item["field"], repr(item["observed"])))
        if decision is not None:
            item.update(accepted=True, resolution=decision["resolution"], decided_by=decision["decided_by"])
    return divergences


def pending(divergences):
    return [item for item in divergences if not item["accepted"]]


def _check_count(divergences, field, documented, observed, source):
    if documented != observed:
        divergences.append(divergence(field, documented, observed, source))


def _index_rows(rows, name_regex, *, source):
    """Hard schema checks shared by v1 and v2; return ``{image_name: row}``."""
    index = {}
    for row in rows:
        image = row["image_name"]
        if not name_regex.fullmatch(image):
            raise DivergenceError(f"{source}: image_name does not match names.image_name_regex (row {len(index) + 2})")
        if image in index:
            raise DivergenceError(f"{source}: repeated image_name (row {len(index) + 2})")
        if not row["patient_id"].strip() or row["patient_id"] != row["patient_id"].strip():
            raise DivergenceError(f"{source}: empty or padded patient_id (row {len(index) + 2}); never imputed")
        if row["target"] not in ("0", "1"):
            raise DivergenceError(f"{source}: target must be '0' or '1' (row {len(index) + 2})")
        index[image] = row
    return index


def analyze_labels(v2_rows, v1_rows, dup_rows, cfg):
    """Return ``(records, pairs, observed, divergences)``.

    ``records``: image -> {patient_id, lesion_id, diagnosis, benign_malignant, label} from GT v2.
    ``pairs``: sorted unique official duplicate pairs ``(a, b)`` with ``a < b``.
    """
    regex = cfg.image_name_regex
    documented = cfg.documented
    divergences = []
    v2 = _index_rows(v2_rows, regex, source="ground_truth_v2")
    v1 = _index_rows(v1_rows, regex, source="ground_truth_v1")
    records = {}
    for image, row in v2.items():
        records[image] = {"patient_id": row["patient_id"], "lesion_id": row["lesion_id"], "diagnosis": row["diagnosis"],
                          "benign_malignant": row["benign_malignant"], "label": int(row["target"])}

    # v1 is only a 1:1 cross-check of image_name / patient_id / target.
    only_v1, only_v2 = set(v1) - set(v2), set(v2) - set(v1)
    patient_mismatch = [image for image in set(v1) & set(v2) if v1[image]["patient_id"] != v2[image]["patient_id"]]
    target_mismatch = [image for image in set(v1) & set(v2) if v1[image]["target"] != v2[image]["target"]]
    for field, items in (("ground_truth_v1_vs_v2.only_in_v1", only_v1), ("ground_truth_v1_vs_v2.only_in_v2", only_v2),
                         ("ground_truth_v1_vs_v2.patient_id_mismatch", patient_mismatch),
                         ("ground_truth_v1_vs_v2.target_mismatch", target_mismatch)):
        if items:
            divergences.append(divergence(field, 0, len(items), "ground_truth_v1", items))

    # label coherence: target==1 <=> diagnosis=="melanoma" <=> benign_malignant=="malignant"
    incoherent_dx = [i for i, r in records.items() if (r["label"] == 1) != (r["diagnosis"] == "melanoma")]
    incoherent_bm = [i for i, r in records.items() if (r["label"] == 1) != (r["benign_malignant"] == "malignant")]
    if incoherent_dx:
        divergences.append(divergence("labels.target_vs_diagnosis_melanoma", 0, len(incoherent_dx), "ground_truth_v2", incoherent_dx))
    if incoherent_bm:
        divergences.append(divergence("labels.target_vs_benign_malignant", 0, len(incoherent_bm), "ground_truth_v2", incoherent_bm))

    patients = {r["patient_id"] for r in records.values()}
    positives = sum(r["label"] for r in records.values())
    lesions = {r["lesion_id"] for r in records.values() if r["lesion_id"].strip()}
    _check_count(divergences, "counts.images_csv", documented["images"], len(records), "ground_truth_v2")
    _check_count(divergences, "counts.patients", documented["patients"], len(patients), "ground_truth_v2")
    _check_count(divergences, "counts.melanoma_images", documented["melanoma_images"], positives, "ground_truth_v2")

    # duplicates: 425 rows of disjoint pairs (plan 3.1.2)
    pairs, self_pairs, repeated, unknown = set(), [], 0, set()
    occurrences = Counter()
    for row in dup_rows:
        a, b = row["image_name_1"], row["image_name_2"]
        for name in (a, b):
            if not regex.fullmatch(name):
                raise DivergenceError(f"duplicates: name does not match names.image_name_regex (row {len(pairs) + 2})")
            if name not in records:
                unknown.add(name)
        if a == b:
            self_pairs.append(a)
            continue
        pair = (min(a, b), max(a, b))
        if pair in pairs:
            repeated += 1
            continue
        pairs.add(pair)
        occurrences.update(pair)
    distinct = set(occurrences)
    shared = [name for name, count in occurrences.items() if count > 1]
    _check_count(divergences, "counts.duplicate_rows", documented["duplicate_pairs"], len(dup_rows), "duplicates")
    _check_count(divergences, "counts.duplicate_pairs", documented["duplicate_pairs"], len(pairs), "duplicates")
    _check_count(divergences, "counts.duplicate_distinct_images", documented["duplicate_distinct_images"], len(distinct), "duplicates")
    if self_pairs:
        divergences.append(divergence("duplicates.self_pairs", 0, len(self_pairs), "duplicates", self_pairs))
    if repeated:
        divergences.append(divergence("duplicates.repeated_pairs", 0, repeated, "duplicates"))
    if shared:
        divergences.append(divergence("duplicates.pairs_not_disjoint", 0, len(shared), "duplicates", shared))
    if unknown:
        divergences.append(divergence("duplicates.names_not_in_ground_truth", 0, len(unknown), "duplicates", unknown))

    observed = {"csv_rows": len(records), "csv_rows_v1": len(v1), "patients": len(patients), "melanoma_images": positives,
                "lesions": len(lesions), "lesion_id_empty": sum(1 for r in records.values() if not r["lesion_id"].strip()),
                "duplicate_rows_official": len(dup_rows), "duplicate_pairs_official": len(pairs),
                "duplicate_distinct_images_official": len(distinct)}
    return records, sorted(pairs), observed, divergences


def classify_members(entries, cfg):
    """Map CD entries to ``{"index", "position", "kind", "image"}``; names never become paths.

    kind: ``dir`` (directory entry), ``junk`` (ignored_member_regexes, e.g. __MACOSX/.DS_Store),
    ``image`` (member_regex matched; ``image`` is the reconstructed name), ``unexpected``.
    """
    member_regex, image_regex, ignored = cfg.member_regex, cfg.image_name_regex, cfg.ignored_member_regexes
    out, seen = [], {}
    for position, entry in enumerate(entries):
        kind, image = "unexpected", None
        if entry.is_dir:
            kind = "dir"
        elif any(pattern.search(entry.name) for pattern in ignored):
            kind = "junk"
        else:
            match = member_regex.fullmatch(entry.name)
            if match and image_regex.fullmatch(match.group("image")):
                kind, image = "image", match.group("image")
                if image in seen:
                    raise IntegrityError(f"two ZIP members map to the same image name (CD entries {seen[image]} and {entry.index})")
                seen[image] = entry.index
        out.append({"index": entry.index, "position": position, "kind": kind, "image": image})
    return out


def cross_check(members, records, cfg):
    """ZIP<->CSV bijection and member counts; returns ``(observed, divergences)``."""
    images = {m["image"] for m in members if m["kind"] == "image"}
    kinds = Counter(m["kind"] for m in members)
    divergences = []
    without_label = images - set(records)
    without_image = set(records) - images
    _check_count(divergences, "counts.zip_jpg_members", cfg.documented["images"], len(images), "images_zip")
    if without_label:
        divergences.append(divergence("zip_csv.images_without_label", 0, len(without_label), "images_zip", without_label))
    if without_image:
        divergences.append(divergence("zip_csv.labels_without_image", 0, len(without_image), "ground_truth_v2", without_image))
    if kinds["unexpected"]:
        divergences.append(divergence("zip.unexpected_members", 0, kinds["unexpected"], "images_zip"))
    observed = {"zip_entries": len(members), "zip_jpg_members": len(images), "zip_dirs": kinds["dir"],
                "zip_junk_members": kinds["junk"], "zip_unexpected_members": kinds["unexpected"],
                "zip_images_without_label": len(without_label), "csv_labels_without_image": len(without_image)}
    return observed, divergences


def exact_hash_check(official_pairs, sha_by_image):
    """Official pairs must be byte-identical originals and vice versa (plan 3.5).

    Only images with a known original SHA256 take part. Returns ``(exact_pairs, divergences)``.
    """
    by_hash = {}
    for image, sha in sha_by_image.items():
        by_hash.setdefault(sha, []).append(image)
    exact = set()
    for images in by_hash.values():
        for a, b in itertools.combinations(sorted(images), 2):
            exact.add((a, b))
    official = {pair for pair in official_pairs if pair[0] in sha_by_image and pair[1] in sha_by_image}
    not_identical = sorted(official - exact)
    not_official = sorted(exact - set(official_pairs))
    divergences = []
    if not_identical:
        divergences.append(divergence("duplicates.official_pair_not_byte_identical", 0, len(not_identical), "duplicates",
                                      [a for a, _b in not_identical]))
    if not_official:
        divergences.append(divergence("duplicates.identical_originals_not_listed", 0, len(not_official), "images_zip",
                                      [a for a, _b in not_official]))
    return sorted(exact), divergences


__all__ = ["analyze_labels", "apply_acceptance", "classify_members", "cross_check", "divergence", "exact_hash_check", "pending"]
