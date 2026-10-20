"""Groups = connected components of patient_id joined by duplicate pairs (plan 3.5, ADR-A5),
and the duplicate policy ``keep-min-image-name-v1``.
"""

import hashlib

from .errors import IntegrityError


class UnionFind:
    """Union-find with path halving; deterministic (smaller key becomes root on ties)."""

    def __init__(self, items=()):
        self.parent = {}
        for item in items:
            self.add(item)

    def add(self, item):
        self.parent.setdefault(item, item)

    def find(self, item):
        self.add(item)
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        root, child = (ra, rb) if ra < rb else (rb, ra)
        self.parent[child] = root
        return root

    def components(self):
        out = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return sorted(sorted(members) for members in out.values())


def group_id(prefix, patient_ids):
    digest = hashlib.sha256("\n".join(sorted(patient_ids)).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:16]}"


def build_groups(records, pairs, prefix):
    """Return ``(patient_to_group, image_to_group, stats)``.

    ``records``: image -> {"patient_id", ...}; ``pairs``: iterable of (image_a, image_b)
    (official and exact-hash pairs). A pair across patients merges both patients.
    """
    uf = UnionFind(sorted({r["patient_id"] for r in records.values()}))
    cross = 0
    for a, b in pairs:
        if a not in records or b not in records:
            continue
        pa, pb = records[a]["patient_id"], records[b]["patient_id"]
        if pa != pb:
            cross += 1
        uf.union(pa, pb)
    patient_to_group = {}
    for patients in uf.components():
        gid = group_id(prefix, patients)
        for patient in patients:
            patient_to_group[patient] = gid
    image_to_group = {image: patient_to_group[r["patient_id"]] for image, r in records.items()}
    if len(set(patient_to_group.values())) != len(uf.components()):
        raise IntegrityError("group id collision on the 64-bit prefix; needs a decision, never silent")
    stats = {"patients": len(patient_to_group), "groups": len(set(patient_to_group.values())), "cross_patient_pairs": cross}
    return patient_to_group, image_to_group, stats


def duplicate_policy(records, pairs):
    """keep-min-image-name-v1: one representative (smallest image_name) per duplicate component;
    if the component's labels conflict, exclude every member. Returns ``{image: reason}``.
    """
    uf = UnionFind()
    for a, b in pairs:
        if a in records and b in records:
            uf.union(a, b)
    excluded = {}
    for component in uf.components():
        if len(component) < 2:
            continue
        labels = {records[image]["label"] for image in component}
        if len(labels) > 1:
            for image in component:
                excluded[image] = "duplicate-label-conflict"
            continue
        keep = component[0]
        for image in component[1:]:
            excluded[image] = f"duplicate-of:{keep}"
    return excluded


__all__ = ["UnionFind", "build_groups", "duplicate_policy", "group_id"]
