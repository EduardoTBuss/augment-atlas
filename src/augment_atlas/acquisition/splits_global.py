"""``global-dev-eval-v1`` (plan section 5): deterministic group-level D/E pools, stdlib only.

1. Unit = group (plan 3.5). Stratum = (group has a positive image) x (quartile of images per group).
2. D/E: inside each stratum, sort groups by sha256("alg|seed|DE|group"); the first
   round_half_up(dev_fraction * n) go to D.
3. D_fit/D_select: same inside D with stage "D-select" and ``select_fraction``.
4. E -> folds: inside each stratum sort by stage "E-folds" and deal round-robin.
   Interpretation (documented): the dealer position carries over between strata
   (strata visited in sorted order) so fold sizes stay within one group per stratum
   boundary instead of always favouring fold 0. Still deterministic and per stratum.
5. Audits are pass/fail with pre-registered thresholds; there is no seed retry.

Quartile rule (documented): thresholds t_k = sizes_sorted[(k * n) // 4], k = 1..3;
quartile(size) = number of thresholds strictly below ``size``.
"""

import hashlib

from ..contracts import canonical_hash

ALGORITHM = "global-dev-eval-v1"


def round_half_up(x_num, x_den=1):
    """round(x) with halves up, for non-negative rationals given as num/den."""
    return (2 * x_num + x_den) // (2 * x_den)


def _fraction_count(fraction, n):
    # exact enough for fractions written with few decimals in the config
    num = round(fraction * 1_000_000)
    return round_half_up(num * n, 1_000_000)


def _key(seed, stage, group):
    return hashlib.sha256(f"{ALGORITHM}|{seed}|{stage}|{group}".encode("utf-8")).hexdigest()


def group_table(records):
    """records: iterable of {"group", "label"} (included images only) -> {group: (n_images, n_positive)}."""
    table = {}
    for record in records:
        n, pos = table.get(record["group"], (0, 0))
        table[record["group"]] = (n + 1, pos + int(record["label"] == 1))
    return table


def strata(table):
    sizes = sorted(n for n, _pos in table.values())
    count = len(sizes)
    thresholds = [sizes[(k * count) // 4] for k in (1, 2, 3)] if count else []
    out = {}
    for group, (n, pos) in table.items():
        quartile = sum(1 for t in thresholds if n > t)
        out.setdefault((int(pos > 0), quartile), []).append(group)
    return {key: sorted(groups) for key, groups in sorted(out.items())}, thresholds


def make_pools(records, params):
    seed = params["split_seed"]
    table = group_table(records)
    stratified, thresholds = strata(table)
    pools = {"D_fit": [], "D_select": [], "E": []}
    folds = [[] for _ in range(params["eval_folds"])]
    dealer = 0
    for key, groups in stratified.items():
        ordered = sorted(groups, key=lambda g: _key(seed, "DE", g))
        n_dev = _fraction_count(params["dev_fraction"], len(ordered))
        dev, evaluation = ordered[:n_dev], ordered[n_dev:]
        dev_ordered = sorted(dev, key=lambda g: _key(seed, "D-select", g))
        n_select = _fraction_count(params["select_fraction"], len(dev_ordered))
        pools["D_select"] += dev_ordered[:n_select]
        pools["D_fit"] += dev_ordered[n_select:]
        pools["E"] += evaluation
        for group in sorted(evaluation, key=lambda g: _key(seed, "E-folds", g)):
            folds[dealer % len(folds)].append(group)
            dealer += 1
    pools = {name: sorted(groups) for name, groups in pools.items()}
    folds = {f"E_{index}": sorted(groups) for index, groups in enumerate(folds)}
    return pools, folds, {"quartile_thresholds": thresholds, "strata": {f"pos{k[0]}-q{k[1]}": len(v) for k, v in stratified.items()}}


def audit(records, pools, folds, params):
    """Pre-registered pass/fail audits; returns ``(passed, report)`` with counts only."""
    table = group_table(records)
    partition = {"D_fit": pools["D_fit"], "D_select": pools["D_select"], **folds}
    seen, failures = {}, []
    for name, groups in partition.items():
        for group in groups:
            if group in seen:
                failures.append(f"group in both {seen[group]} and {name}")
            seen[group] = name
    if set(seen) != set(table):
        failures.append(f"coverage: {len(set(table) - set(seen))} groups unassigned, {len(set(seen) - set(table))} unknown")
    if set(pools["E"]) != {g for name, groups in folds.items() for g in groups}:
        failures.append("E pool differs from the union of its folds")
    counts = {}
    for name, groups in {**partition, "E": pools["E"]}.items():
        images = sum(table[g][0] for g in groups if g in table)
        positives = sum(table[g][1] for g in groups if g in table)
        positive_groups = sum(1 for g in groups if g in table and table[g][1] > 0)
        counts[name] = {"groups": len(groups), "images": images, "positive_images": positives,
                        "positive_groups": positive_groups}
        if positives == 0 or positives == images:
            failures.append(f"{name} lacks one of the classes")
    if counts["D_select"]["positive_groups"] < params["min_select_positive_groups"]:
        failures.append(f"D_select has {counts['D_select']['positive_groups']} positive groups "
                        f"< {params['min_select_positive_groups']}")
    for name in folds:
        if counts[name]["positive_groups"] < params["min_eval_fold_positive_groups"]:
            failures.append(f"{name} has {counts[name]['positive_groups']} positive groups "
                            f"< {params['min_eval_fold_positive_groups']}")
    return not failures, {"counts": counts, "failures": failures}


def build(records, params):
    pools, folds, info = make_pools(records, params)
    passed, report = audit(records, pools, folds, params)
    header = {"algorithm": ALGORITHM, "split_seed": params["split_seed"], "dev_fraction": params["dev_fraction"],
              "select_fraction": params["select_fraction"], "eval_folds": params["eval_folds"]}
    pools_doc = {**header, "pools": pools}
    pools_doc["canonical_hash"] = canonical_hash(pools_doc)
    folds_doc = {**header, "folds": folds}
    folds_doc["canonical_hash"] = canonical_hash(folds_doc)
    audit_doc = {**header, "passed": passed, **report, **info,
                 "thresholds": {"min_select_positive_groups": params["min_select_positive_groups"],
                                "min_eval_fold_positive_groups": params["min_eval_fold_positive_groups"]},
                 "pools_sha256": pools_doc["canonical_hash"], "folds_sha256": folds_doc["canonical_hash"]}
    return pools_doc, folds_doc, audit_doc


__all__ = ["ALGORITHM", "audit", "build", "group_table", "make_pools", "round_half_up", "strata"]
