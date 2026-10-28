"""Offline, image-level analysis of one dataset under SAP v1 option A.

No training, selection, adaptive resampling, real-data access or network I/O.
See docs/analysis-sap-v1.md for inputs, contrasts, trust and seed boundaries.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from . import clinical_metrics as cm
from .contracts import ARMS

FOLDS = tuple(range(5))
TRAIN_SEEDS = (1, 2, 3)
PROTOCOL = "augment-atlas-analysis-sap-v1"
CSV_FIELDS = {"dataset", "arm", "fold", "seed", "image", "group", "label", "score"}


def contrast_specs():
    """Coefficients operate on metrics, never on an ensemble of scores."""
    specs = []
    for family, primary, secondary in (("gamma", "H1", "S1"), ("clahe", "H2", "S2")):
        specs.append((primary, {f"basic_{family}_light": .5,
                                f"basic_{family}_heavy": .5, "basic": -1.}))
        specs.append((secondary, {f"{family}_light": .5,
                                  f"{family}_heavy": .5, "none": -1.}))
    specs.append(("S3", {"basic": 1., "none": -1.}))
    for arm in ARMS[2:]:
        reference = "basic" if arm.startswith("basic_") else "none"
        specs.append((f"S4_{arm}", {arm: 1., reference: -1.}))
    for family in ("gamma", "clahe"):
        specs.append((f"S5_{family}", {f"basic_{family}_light": .5,
                                      f"basic_{family}_heavy": .5, "basic": -1.,
                                      f"{family}_light": -.5,
                                      f"{family}_heavy": -.5, "none": 1.}))
    return sorted(specs, key=lambda item: (not item[0].startswith("H"), item[0]))


def holm_adjust(pvalues):
    """Holm m=2 even when one primary is unavailable (treated as p=1)."""
    if set(pvalues) != {"H1", "H2"}:
        raise ValueError("Holm requires exactly H1 and H2")
    for p in pvalues.values():
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("invalid p-value")
    order = sorted(pvalues, key=lambda name: (pvalues[name] if pvalues[name] is not None else 1., name))
    result, previous = {}, 0.
    for index, name in enumerate(order):
        p = 1. if pvalues[name] is None else pvalues[name]
        previous = min(1., max(previous, (2 - index) * p))
        result[name] = previous if pvalues[name] is not None else None
    return result


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_predictions(path, dataset):
    """Read native final artifacts or one normalized, binary-positive-score CSV.

    Native integrity is checked, but upstream campaign auditing is still required:
    predictions alone cannot establish D/E isolation or complete trusted E coverage.
    """
    path = Path(path)
    files = sorted(path.rglob("predictions.json")) if path.is_dir() else [path]
    if not files:
        raise ValueError("no predictions.json files")
    rows, bindings, sources = [], set(), set()
    for file in files:
        if file.suffix.lower() == ".csv":
            with file.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if not CSV_FIELDS <= set(reader.fieldnames or []):
                    raise ValueError("CSV requires " + ",".join(sorted(CSV_FIELDS)))
                for row in reader:
                    if row.get("phase", "final") != "final" or row.get("partition", "E_test") != "E_test":
                        raise ValueError("isolation_violation: only final/E_test is allowed")
                    if row.get("recipe_source"):
                        sources.add(row["recipe_source"])
                    rows.append({**{key: row[key] for key in ("dataset", "arm", "image", "group")},
                                 "fold": int(row["fold"]), "seed": int(row["seed"]),
                                 "label": int(row["label"]), "score": float(row["score"])})
        elif file.suffix.lower() == ".json":
            from .global_evaluation import _structure
            artifact = json.loads(file.read_text(encoding="utf-8"))
            _structure(artifact)
            if (artifact["artifact_type"] != "global_predictions" or artifact["phase"] != "final"
                    or artifact["partition"] != "E_test"):
                raise ValueError("isolation_violation: only final/E_test is allowed")
            if (artifact["checkpoint_selection"] != "last_epoch"
                    or artifact["epoch"] != 20 or artifact["epochs"] != 20):
                raise ValueError("SAP requires fixed last epoch 20")
            task = artifact["task"]
            classes, positive = task["classes"], task["positive_class"]
            if task["type"] != "binary" or len(classes) != 2 or positive not in classes:
                raise ValueError("SAP requires an explicit binary positive class")
            if artifact["score_kind"] != "softmax_probability":
                raise ValueError("native artifacts require softmax probabilities")
            bindings.add(json.dumps({key: artifact[key] for key in
                                    ("dataset", "task", "freeze_sha256", "pools_sha256")}, sort_keys=True))
            positive_index = classes.index(positive)
            for row in artifact["rows"]:
                scores = row["scores"]
                if (row["target"] not in classes or len(scores) != 2
                        or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                               for p in scores) or not math.isclose(sum(scores), 1., abs_tol=1e-6)):
                    raise ValueError("invalid native binary probabilities/target")
                rows.append({"dataset": artifact["dataset"]["id"], "arm": artifact["arm"],
                             "fold": artifact["fold"], "seed": artifact["train_seed"],
                             "image": row["id"], "group": row["group"],
                             "label": int(row["target"] == positive), "score": scores[positive_index]})
        else:
            raise ValueError("expected a directory, CSV or native predictions.json")
    if len(bindings) > 1:
        raise ValueError("mixed campaigns, datasets, tasks or recipes are forbidden")
    if len(sources) > 1:
        raise ValueError("mixed recipe sources are forbidden")
    if any(row["dataset"] != dataset for row in rows):
        raise ValueError("each dataset is a separate study; dataset mismatch")
    binding = json.loads(next(iter(bindings))) if bindings else None
    metadata = {"inputs": [{"index": i, "sha256": _hash(file), "bytes": file.stat().st_size}
                           for i, file in enumerate(files)],
                "recipe_source": next(iter(sources), None),
                "campaign_binding": binding,
                "freeze_sha256": binding["freeze_sha256"] if binding else None,
                "pools_sha256": binding["pools_sha256"] if binding else None}
    return rows, metadata


@dataclass
class RankPlan:
    """Score tie blocks, with each image mapped to its sampling group."""
    group_order: np.ndarray
    positive_order: np.ndarray
    starts: np.ndarray


def rank_plan(labels, scores, groups):
    order = np.argsort(scores, kind="stable")
    sorted_scores = np.asarray(scores)[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    return RankPlan(np.asarray(groups)[order], np.asarray(labels, dtype=np.float64)[order], starts)


def weighted_metrics(plan, group_counts):
    """Batched Mann-Whitney midrank AUROC and tied-threshold AP.

    Multiplicities weight images; rows are never replicated. NaN denotes an
    undefined bootstrap endpoint and is handled explicitly by the caller.
    """
    weights = group_counts[:, plan.group_order]
    p = np.add.reduceat(weights * plan.positive_order, plan.starts, axis=1)
    n = np.add.reduceat(weights * (1. - plan.positive_order), plan.starts, axis=1)
    positives, negatives = p.sum(axis=1), n.sum(axis=1)
    twice_wins = (p * (2 * np.cumsum(n, axis=1) - n)).sum(axis=1)
    auc = np.full(len(group_counts), np.nan)
    np.divide(twice_wins, 2 * positives * negatives, out=auc,
              where=(positives > 0) & (negatives > 0))
    # Ascending tie blocks: sum over thresholds using counts at or above each.
    tp = positives[:, None] - np.cumsum(p, axis=1) + p
    seen = (positives + negatives)[:, None] - np.cumsum(p + n, axis=1) + p + n
    precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    ap = np.full(len(group_counts), np.nan)
    np.divide((p * precision).sum(axis=1), positives, out=ap, where=positives > 0)
    return np.column_stack((auc, ap))


def _prepare(rows, dataset, training_seeds=TRAIN_SEEDS):
    cells, truth, group_folds = {}, {}, {}
    for row in rows:
        if row.get("dataset") != dataset or row.get("arm") not in ARMS:
            raise ValueError("dataset/arm mismatch")
        if (type(row.get("fold")) is not int or row["fold"] not in FOLDS
                or type(row.get("seed")) is not int or row["seed"] not in training_seeds
                or type(row.get("label")) is not int or row["label"] not in (0, 1)):
            raise ValueError("fold must be 0..4, seed must be predeclared, binary label 0/1")
        if any(type(row.get(key)) is not str or not row[key].strip() for key in ("image", "group")):
            raise ValueError("image and group must be nonempty strings")
        if type(row.get("score")) not in (int, float):
            raise ValueError("score must be numeric")
        identity = (row["fold"], row["group"], row["label"])
        if row["image"] in truth and truth[row["image"]] != identity:
            raise ValueError("isolation_violation: inconsistent image truth/fold/group")
        truth[row["image"]] = identity
        if row["group"] in group_folds and group_folds[row["group"]] != row["fold"]:
            raise ValueError("isolation_violation: group crosses folds")
        group_folds[row["group"]] = row["fold"]
        cell = cells.setdefault((row["arm"], row["fold"], row["seed"]), {})
        if row["image"] in cell:
            raise ValueError("duplicate prediction (retry attempts must be resolved upstream)")
        cell[row["image"]] = row["score"]
    folds, points, plans = {}, {}, {}
    for fold in FOLDS:
        ids = sorted(image for image in truth if truth[image][0] == fold)
        groups = sorted({truth[image][1] for image in ids})
        group_index = {group: i for i, group in enumerate(groups)}
        image_groups = np.array([group_index[truth[image][1]] for image in ids], dtype=int)
        labels = [truth[image][2] for image in ids]
        status = np.zeros(len(groups), dtype=bool)
        for g, label in zip(image_groups, labels):
            status[g] |= bool(label)
        folds[fold] = {"ids": ids, "labels": labels, "groups": image_groups,
                       "n_groups": len(groups), "strata": (np.flatnonzero(~status), np.flatnonzero(status))}
        for arm in ARMS:
            for train_seed in training_seeds:
                key = (arm, fold, train_seed)
                cell = cells.get(key)
                reason = ("missing_fold" if not cell else "unpaired_images" if set(cell) != set(ids)
                          else "nonfinite_scores" if any(not math.isfinite(v) for v in cell.values()) else None)
                if reason is None:
                    scores = [cell[image] for image in ids]
                    auc = cm.binary_auroc(labels, scores, classes=[0, 1], positive_label=1)
                    ap = cm.binary_average_precision(labels, scores, classes=[0, 1], positive_label=1)
                    reason = auc["reason"] or ap["reason"]
                if reason:
                    points[key] = {"status": "not_evaluable", "reason": reason}
                    continue
                decisions = [int(score >= .5) for score in scores]
                positives = sum(labels)
                negatives = len(labels) - positives
                points[key] = {"status": "ok", "reason": None, "auroc": auc["value"], "ap": ap["value"],
                               "macro_f1": cm.macro_f1(labels, decisions, classes=[0, 1])["value"],
                               "balanced_accuracy": cm.balanced_accuracy(labels, decisions, classes=[0, 1])["value"],
                               "sensitivity": sum(y and p for y, p in zip(labels, decisions)) / positives,
                               "specificity": sum(not y and not p for y, p in zip(labels, decisions)) / negatives}
                plans[key] = rank_plan(labels, scores, image_groups)
    return cells, folds, points, plans


def _bootstrap(folds, plans, seeds, *, B, seed, block_size):
    result = {(arm, train_seed): np.zeros((B, 2)) for arm in ARMS for train_seed in seeds}
    for fold in FOLDS:
        info = folds[fold]
        # Independent deterministic streams per fold/status; block size cannot
        # change which random draws belong to either stratum.
        rngs = [np.random.default_rng(np.random.SeedSequence([seed, fold, status])) for status in (0, 1)]
        active = [(arm, train_seed, plans[(arm, fold, train_seed)]) for arm in ARMS for train_seed in seeds
                  if (arm, fold, train_seed) in plans]
        for start in range(0, B, block_size):
            stop = min(start + block_size, B)
            counts = np.zeros((stop - start, info["n_groups"]), dtype=np.int64)
            for rng, indices in zip(rngs, info["strata"]):
                n = len(indices)
                if n:
                    counts[:, indices] = rng.multinomial(n, np.full(n, 1. / n), size=stop - start)
            for arm, train_seed, plan in active:
                result[(arm, train_seed)][start:stop] += weighted_metrics(plan, counts) / 5.
    return result


def _interval(estimate, draws, *, primary, inference):
    out = {"estimate": estimate, "ci95": None}
    if primary:
        out.update(ci97_5=None, p_boot=None)
    if not inference:
        out.update(status="descriptive_only", reason="inbreast_hpo_d_descriptive_only")
        return out
    undefined = int(np.count_nonzero(~np.isfinite(draws)))
    out["undefined_bootstrap_replicates"] = undefined
    if undefined:
        out.update(status="not_evaluable", reason="undefined_bootstrap_replicates")
        return out
    out.update(status="ok", reason=None, ci95=np.quantile(draws, [.025, .975]).tolist())
    if primary:
        out["ci97_5"] = np.quantile(draws, [.0125, .9875]).tolist()
        count = min(np.count_nonzero(draws <= 0), np.count_nonzero(draws >= 0))
        out["p_boot"] = min(1., 2. * (int(count) + 1) / (len(draws) + 1))
    return out


def _summarize(points, bootstrap, seeds, *, inference):
    arms, values, draws = [], {}, {}
    metrics = ("auroc", "ap", "macro_f1", "balanced_accuracy", "sensitivity", "specificity")
    for arm in ARMS:
        entries = [{"fold": fold, "seed": train_seed, **points[(arm, fold, train_seed)]}
                   for train_seed in seeds for fold in FOLDS]
        invalid = [entry for entry in entries if entry["status"] != "ok"]
        if invalid:
            arms.append({"arm": arm, "status": "not_evaluable", "reason": "missing_or_undefined_folds",
                         "metrics": None, "per_fold": entries})
            continue
        # Same canonical math.fsum ordering as global_report._aggregate.
        means = {metric: math.fsum(math.fsum(points[(arm, fold, train_seed)][metric]
                                             for fold in FOLDS) / 5. for train_seed in seeds) / len(seeds)
                 for metric in metrics}
        values[arm] = means
        if inference:
            draws[arm] = sum((bootstrap[(arm, train_seed)] for train_seed in seeds)) / len(seeds)
        arms.append({"arm": arm, "status": "ok", "reason": None, "metrics": means, "per_fold": entries})
    contrasts = []
    for name, coefficients in contrast_specs():
        primary = name in ("H1", "H2")
        item = {"id": name, "coefficients": coefficients, "primary": primary}
        missing = [arm for arm in coefficients if arm not in values]
        if missing:
            item.update(status="not_evaluable", reason="missing_or_undefined_folds", unavailable_arms=missing,
                        auroc={"estimate": None, "ci95": None}, ap={"estimate": None, "ci95": None})
            if primary:
                item["auroc"].update(ci97_5=None, p_boot=None)
        else:
            for column, metric in enumerate(("auroc", "ap")):
                estimate = math.fsum(coef * values[arm][metric] for arm, coef in coefficients.items())
                samples = (sum(coef * draws[arm][:, column] for arm, coef in coefficients.items())
                           if inference else None)
                item[metric] = _interval(estimate, samples, primary=primary and metric == "auroc", inference=inference)
            item.update(status=item["auroc"]["status"], reason=item["auroc"]["reason"])
        contrasts.append(item)
    adjusted = holm_adjust({item["id"]: item["auroc"]["p_boot"] for item in contrasts if item["primary"]})
    for item in contrasts:
        if item["primary"]:
            p = adjusted[item["id"]]
            item["auroc"].update(p_holm=p, reject_holm_0_05=None if p is None else p <= .05)
    return {"status": "ok" if all(item["status"] in ("ok", "descriptive_only") for item in contrasts[:2])
            else "not_evaluable", "training_seeds": list(seeds), "arms": arms, "contrasts": contrasts}


def _pooled(cells, folds, points, seeds):
    output = []
    for arm in ARMS:
        for train_seed in seeds:
            if any(points[(arm, fold, train_seed)]["status"] != "ok" for fold in FOLDS):
                output.append({"arm": arm, "seed": train_seed, "status": "not_evaluable",
                               "reason": "missing_or_undefined_folds", "auroc": None, "ap": None})
                continue
            labels = [label for fold in FOLDS for label in folds[fold]["labels"]]
            scores = [cells[(arm, fold, train_seed)][image] for fold in FOLDS for image in folds[fold]["ids"]]
            output.append({"arm": arm, "seed": train_seed, "status": "ok", "reason": None,
                           "auroc": cm.binary_auroc(labels, scores, classes=[0, 1], positive_label=1)["value"],
                           "ap": cm.binary_average_precision(labels, scores, classes=[0, 1], positive_label=1)["value"]})
    return {"aggregation": "concatenated_E_per_seed_no_ensemble", "inference": "descriptive_only", "arms": output}


def analyze(rows, *, dataset, seed=20261028, B=10000, block_size=64, recipe_source=None,
            training_seeds=TRAIN_SEEDS):
    """Analyze one dataset. First predeclared training seed is primary.

    seed is exclusively the bootstrap RNG seed. It never selects training seeds.
    """
    if type(B) is not int or B < 1 or type(block_size) is not int or block_size < 1:
        raise ValueError("B and block_size must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if recipe_source not in (None, "external_fixed", "hpo_d"):
        raise ValueError("recipe_source must be external_fixed or hpo_d")
    training_seeds = tuple(training_seeds)
    if (len(training_seeds) != 3 or len(set(training_seeds)) != 3
            or any(type(value) is not int or value < 0 for value in training_seeds)):
        raise ValueError("declare exactly three distinct nonnegative training seeds in SAP order")
    inbreast = "inbreast" in dataset.lower()
    if inbreast and recipe_source is None:
        raise ValueError("INbreast requires explicit recipe_source before analysis")
    inference = not (inbreast and recipe_source == "hpo_d")
    cells, folds, points, plans = _prepare(rows, dataset, training_seeds)
    complete_three = all(points[(arm, fold, train_seed)]["status"] == "ok"
                         for arm in ARMS for fold in FOLDS for train_seed in training_seeds)
    seeds = training_seeds if complete_three else (training_seeds[0],)
    bootstrap = _bootstrap(folds, plans, seeds, B=B, seed=seed, block_size=block_size) if inference else {}
    primary = _summarize(points, bootstrap, (training_seeds[0],), inference=inference)
    sensitivity = (_summarize(points, bootstrap, training_seeds, inference=inference) if complete_three else
                   {"status": "not_evaluable", "reason": "requires_complete_10_arms_5_folds_3_seeds"})
    coverage = [{"fold": fold, "images": len(info["ids"]), "groups": info["n_groups"],
                 "positive_groups": len(info["strata"][1])} for fold, info in folds.items()]
    warnings = (["coverage below nominal measured in simulation with <8 positive groups per fold"]
                if any(fold["positive_groups"] < 8 for fold in coverage) else [])
    return {"protocol": PROTOCOL, "dataset": dataset, "exploratory": True,
            "analysis_scope": "primary_external_fixed" if inbreast and inference else
                              "descriptive_hpo_d" if not inference else "primary",
            "recipe_source": recipe_source, "declared_training_seeds": list(training_seeds), "aggregation": "mean_5_folds_then_mean_training_seeds",
            "bootstrap": {"B": B, "seed": seed, "block_size": block_size,
                          "sampling": "paired_groups_within_fold_by_group_has_positive",
                          "undefined_policy": "not_evaluable_no_retry_no_drop",
                          "interval": "percentile", "holm_m": 2},
            "primary": primary, "sensitivities": {"three_seed_mean": sensitivity,
                                                    "pooled_E": _pooled(cells, folds, points, seeds)},
            "coverage": coverage, "warnings": warnings,
            "execution_verified": False, "scientific_eligible": False}


def write_outputs(report, out):
    """Manifest is written last; identifiers and raw scores never enter outputs."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Remove a stale completion marker before replacing the individual outputs.
    (out / "manifest.json").unlink(missing_ok=True)
    (out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "table.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("analysis", "kind", "id", "metric", "status", "reason", "estimate", "ci95_low", "ci95_high",
                  "ci97_5_low", "ci97_5_high", "p_boot", "p_holm")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, analysis in (("primary", report["primary"]),
                                ("three_seed_mean", report["sensitivities"]["three_seed_mean"])):
            if "arms" not in analysis:
                writer.writerow(dict(analysis=label, kind="analysis", status=analysis["status"], reason=analysis["reason"]))
                continue
            for arm in analysis["arms"]:
                for metric in ("auroc", "ap", "macro_f1", "balanced_accuracy", "sensitivity", "specificity"):
                    writer.writerow(dict(analysis=label, kind="arm", id=arm["arm"], metric=metric,
                                         status=arm["status"], reason=arm["reason"],
                                         estimate=(arm["metrics"] or {}).get(metric)))
            for item in analysis["contrasts"]:
                for metric in ("auroc", "ap"):
                    value = item[metric]
                    ci95, ci97 = value.get("ci95") or [None, None], value.get("ci97_5") or [None, None]
                    writer.writerow(dict(analysis=label, kind="contrast", id=item["id"], metric=metric,
                                         status=value.get("status", item["status"]), reason=value.get("reason", item["reason"]),
                                         estimate=value["estimate"], ci95_low=ci95[0], ci95_high=ci95[1],
                                         ci97_5_low=ci97[0], ci97_5_high=ci97[1],
                                         p_boot=value.get("p_boot"), p_holm=value.get("p_holm")))
    items = report["primary"]["contrasts"]
    fig, ax = plt.subplots(figsize=(3.35, 5.5), layout="constrained")
    for index, item in enumerate(items):
        value = item["auroc"]
        if value["estimate"] is not None:
            ax.plot(value["estimate"], index, "o", color="#24587a", markersize=3)
            if value.get("ci95"):
                ax.plot(value["ci95"], [index, index], color="#24587a", linewidth=1)
            if value.get("ci97_5"):
                ax.plot(value["ci97_5"], [index, index], color="#24587a", linewidth=.5, alpha=.6)
    labels = []
    for item in items:
        label = item["id"].replace("S4_basic_", "S4 b+").replace("S4_", "S4 ").replace("S5_", "S5 ")
        label = label.replace("_light", " L").replace("_heavy", " H")
        if item["status"] == "not_evaluable":
            label += " [NE]"
        labels.append(label)
    ax.set_yticks(range(len(items)), labels, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color=".5", linestyle="--", linewidth=.6)
    ax.set_xlabel("Absolute delta AUROC (95% CI)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    fig.savefig(out / "forest.png", dpi=300)
    fig.savefig(out / "forest.pdf")
    plt.close(fig)
    manifest = {"protocol": PROTOCOL, "source_sha256": report["provenance"]["source_sha256"],
                **{key: report["provenance"][key] for key in
                   ("inputs", "freeze_sha256", "pools_sha256", "recipe_source", "numpy_version")},
                "outputs": {name: _hash(out / name) for name in ("analysis.json", "table.csv", "forest.pdf", "forest.png")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, help="bootstrap RNG seed, not training seed")
    parser.add_argument("--B", default=10000, type=int)
    parser.add_argument("--block-size", default=64, type=int)
    parser.add_argument("--training-seeds", nargs=3, default=TRAIN_SEEDS, type=int,
                        metavar=("PRIMARY", "SENS2", "SENS3"), help="predeclared SAP order; default 1 2 3")
    parser.add_argument("--recipe-source", choices=("external_fixed", "hpo_d"))
    args = parser.parse_args(argv)
    try:
        # Fingerprint code and inputs before analysis, without asserting SAP seal.
        source_hash = _hash(__file__)
        rows, metadata = read_predictions(args.predictions, args.dataset)
        source = args.recipe_source or metadata["recipe_source"]
        if args.recipe_source and metadata["recipe_source"] and args.recipe_source != metadata["recipe_source"]:
            raise ValueError("conflicting recipe_source declarations")
        report = analyze(rows, dataset=args.dataset, seed=args.seed, B=args.B,
                         block_size=args.block_size, recipe_source=source, training_seeds=args.training_seeds)
        report["provenance"] = {**metadata, "recipe_source": source, "source_sha256": source_hash,
                                "clinical_metrics_sha256": _hash(cm.__file__), "numpy_version": np.__version__}
        write_outputs(report, args.out)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"analysis_sap_v1: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
