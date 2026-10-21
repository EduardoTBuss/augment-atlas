"""Public aggregate summary (plan 4.2) and its privacy lint.

The public document carries counts, quantiles and hashes only: no image, patient,
lesion or group identifier, no member name, no path, no per-group list. Before it is
written, the serialized text is scanned for (a) the configured ID regexes and (b) every
literal identifier known to the run (images, patients, lesions, groups, member names).
Any hit refuses to write it (IntegrityError, exit 2).
"""

import json
import re

from ..contracts import canonical_hash
from .errors import IntegrityError

SCHEMA = "augment-atlas/dataset-summary-public/v1"


MIN_LITERAL_ID_LEN = 6  # shorter literals would match by accident (e.g. inside hashes); regexes still apply


def privacy_lint(text, patterns, identifiers):
    pattern_hits = sum(len(re.findall(pattern, text)) for pattern in patterns)
    id_hits = sum(1 for item in identifiers if item and len(item) >= MIN_LITERAL_ID_LEN and item in text)
    return {"pattern_hits": pattern_hits, "id_hits": id_hits, "patterns": list(patterns)}


def quantiles(values, points=(10, 50, 90)):
    """Nearest-rank quantiles (deterministic, no interpolation)."""
    ordered = sorted(values)
    if not ordered:
        return {f"p{p}": None for p in points}
    out = {}
    for p in points:
        rank = max(1, -(-p * len(ordered) // 100))  # ceil(p/100 * n)
        out[f"p{p}"] = ordered[rank - 1]
    return out


def public_divergences(divergences):
    return [{"field": d["field"], "documented": d["documented"], "observed": d["observed"], "accepted": d["accepted"]}
            for d in divergences]


def build_summary(*, cfg, tool_commit, docker_image_digest, zip_source, hashes, included, excluded_count, groups_stats,
                  duplicates, divergences, derived_bytes_total, splits_counts, identifiers):
    """``included``: list of {"group", "label"} for included images (never serialized)."""
    by_group = {}
    for record in included:
        n, pos = by_group.get(record["group"], (0, 0))
        by_group[record["group"]] = (n + 1, pos + int(record["label"] == 1))
    class_images = {"0": sum(1 for r in included if r["label"] == 0), "1": sum(1 for r in included if r["label"] == 1)}
    class_groups = {"0": sum(1 for n, pos in by_group.values() if pos == 0),
                    "1": sum(1 for n, pos in by_group.values() if pos > 0)}
    mixed = sum(1 for n, pos in by_group.values() if 0 < pos < n)
    document = {
        "schema": SCHEMA,
        "dataset": cfg.dataset["name"], "release_doi": cfg.dataset.get("doi"), "license": cfg.dataset.get("license"),
        "preprocessing_id": cfg.preprocessing["id"],
        "tool_commit": tool_commit, "docker_image_digest": docker_image_digest,
        "config_sha256": cfg.sha256, "config_identity_sha256": cfg.identity_sha256,
        "source_identity": zip_source,
        "hashes": hashes,
        "counts": {"images_included": len(included), "images_excluded": excluded_count,
                   "groups": len(by_group), "patients": groups_stats["patients"],
                   "class_images": class_images, "class_groups": class_groups, "mixed_label_groups": mixed,
                   "images_per_group_quantiles": quantiles([n for n, _pos in by_group.values()])},
        "duplicates": duplicates,
        "divergences": public_divergences(divergences),
        "derived_bytes_total": derived_bytes_total,
        "splits": splits_counts,
    }
    text = json.dumps(document, sort_keys=True)
    lint = privacy_lint(text, cfg.privacy_patterns, identifiers)
    document["privacy_lint"] = lint
    if lint["pattern_hits"] or lint["id_hits"]:
        raise IntegrityError(f"public summary failed the privacy lint ({lint['pattern_hits']} pattern hits, "
                             f"{lint['id_hits']} identifier hits); not written")
    document["sha256"] = canonical_hash(document)
    return document


__all__ = ["SCHEMA", "build_summary", "privacy_lint", "public_divergences", "quantiles"]
