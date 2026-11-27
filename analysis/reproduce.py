"""Regenerate study summaries from the released aggregate inputs."""

import argparse
import csv
from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np

from src.augment_atlas.analysis_sap_v1 import contrast_specs, holm_adjust

DATASETS = {"inbreast": "INbreast", "isic2020": "ISIC 2020", "rsna2018": "RSNA 2018"}
FAMILIES = {"gamma": "H1", "clahe": "H2"}
ARMS = {
    "none": "None", "gamma_light": "Gamma light", "gamma_heavy": "Gamma heavy",
    "clahe_light": "CLAHE light", "clahe_heavy": "CLAHE heavy", "basic": "Basic",
    "basic_gamma_light": "Basic + gamma light", "basic_gamma_heavy": "Basic + gamma heavy",
    "basic_clahe_light": "Basic + CLAHE light", "basic_clahe_heavy": "Basic + CLAHE heavy",
}
FOLDS = tuple(range(5))
SEEDS = (11, 13, 19)
METRICS = ("auroc", "ap")
TREATED = tuple(arm for arm in ARMS if arm.startswith("basic_"))
MODES = {"seed11": SEEDS[:1], "three_seed_mean": SEEDS}
ARCHIVED = "from archived SAP outputs"
RECOMPUTED = "from per-fit aggregates"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def complete_index(path, fields, keys):
    rows = read_csv(path)
    index = {tuple(row[field] for field in fields): row for row in rows}
    expected = {tuple(map(str, key)) for key in keys}
    if len(rows) != len(index) or index.keys() != expected:
        raise ValueError(f"{path.name} does not contain the complete unique study grid")
    return index


def probability(value):
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"expected a score or probability in [0, 1], got {value}")
    return result


def verify_close(value, reference, label):
    if not math.isclose(value, reference, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"{label}: {value} differs from {reference}")


def load_inputs(directory):
    fits = complete_index(directory / "metrics.csv", ("dataset", "arm", "fold", "seed"),
                          product(DATASETS, ARMS, FOLDS, SEEDS))
    coverage = complete_index(directory / "coverage.csv", ("dataset", "fold"),
                              product(DATASETS, FOLDS))
    pooled = complete_index(directory / "pooled-metrics.csv", ("dataset", "arm", "seed"),
                            product(DATASETS, ARMS, SEEDS))
    archive = complete_index(directory / "conditional-contrasts.csv",
                             ("dataset", "analysis", "contrast", "metric"),
                             product(DATASETS, MODES, dict(contrast_specs()), METRICS))
    for (dataset, arm, fold, _), row in fits.items():
        images, positives, distinct, constant, loss_epochs = (
            int(row[name]) for name in ("n_images", "n_positive", "n_distinct", "constant_score",
                                       "epochs_loss_gt_5")
        )
        if (images != int(coverage[dataset, fold]["images"])
                or not 0 < positives < images or not 1 <= distinct <= images
                or constant != int(distinct == 1) or loss_epochs < 0):
            raise ValueError(f"invalid fit counts for {dataset}/{arm}/fold {fold}")
        for metric in METRICS:
            probability(row[metric])
        if constant:
            verify_close(float(row["auroc"]), .5, "constant-score AUROC")
            verify_close(float(row["ap"]), positives / images, "constant-score AP")
    for row in coverage.values():
        if not 0 < int(row["positive_groups"]) <= int(row["groups"]) <= int(row["images"]):
            raise ValueError("invalid fold coverage counts")
    for row in pooled.values():
        for metric in METRICS:
            probability(row[metric])
    return fits, list(coverage.values()), pooled, archive


def factor_components(values, factors):
    """Orthogonal marginal projections on a complete balanced grid."""
    axes = tuple(range(values.ndim))
    center = values.mean()
    effects = {}
    for size in range(1, values.ndim + 1):
        for subset in combinations(axes, size):
            omitted = tuple(axis for axis in axes if axis not in subset)
            effect = np.broadcast_to(values.mean(axis=omitted, keepdims=True), values.shape)
            effect = effect - center
            for smaller, component in effects.items():
                if set(smaller) < set(subset):
                    effect -= component
            effects[subset] = effect
    total = float(np.square(values - center).sum())
    terms = {}
    for subset, effect in effects.items():
        name = ":".join(factors[axis] for axis in subset)
        ss = float(np.square(effect).sum())
        terms[name] = dict(ss=ss, df=math.prod(values.shape[axis] - 1 for axis in subset),
                           share_pct=100 * ss / total if total > 1e-24 else None)
    verify_close(sum(term["ss"] for term in terms.values()), total, "centered SS decomposition")
    return dict(n=values.size, total=total, terms=terms), {
        ":".join(factors[axis] for axis in subset): component
        for subset, component in effects.items()
    }


def rms(values):
    return float(np.sqrt(np.mean(values ** 2)))


def describe(values):
    return dict(n=values.size, mean=float(values.mean()), sd=float(values.std(ddof=1)),
                min=float(values.min()), max=float(values.max()),
                positive=int(np.count_nonzero(values > 0)), negative=int(np.count_nonzero(values < 0)))


def paired_summary(values, flags):
    clean = values[~flags]
    decomposition, _ = factor_components(values, ("fold", "setting"))
    return dict(
        **describe(values), matrix=values.tolist(), constant_score=flags.tolist(), ss=decomposition,
        setting_means={str(seed): float(values[:, col].mean()) for col, seed in enumerate(SEEDS)},
        setting_positive={str(seed): int(np.count_nonzero(values[:, col] > 0))
                          for col, seed in enumerate(SEEDS)},
        leave_one_fold_out=[float(np.delete(values, fold, axis=0).mean()) for fold in FOLDS],
        seed11_leave_one_fold_out=[float(np.delete(values[:, 0], fold).mean()) for fold in FOLDS],
        without_constant=dict(n=clean.size, mean=float(clean.mean()) if clean.size else None,
                              positive=int(np.count_nonzero(clean > 0))),
    )


def repeatability(values):
    first, second = values[..., 1], values[..., 2]
    difference = second - first
    return dict(n=difference.size, agreement=int(np.count_nonzero(np.sign(first) == np.sign(second))),
                ties=int(np.count_nonzero((first == 0) | (second == 0))),
                median_change=float(np.median(np.abs(difference))), rms_change=rms(difference),
                within_fold_sd=rms(difference) / math.sqrt(2))


def ss_identity(treated, basic, paired):
    treated_ss = float(np.square(treated).sum())
    basic_ss = float(np.square(basic).sum())
    cross = -2 * float((treated * basic).sum())
    paired_ss = float(np.square(paired).sum())
    verify_close(treated_ss + basic_ss + cross, paired_ss, "baseline subtraction SS identity")
    return dict(treated=treated_ss, basic=basic_ss, cross=cross, paired=paired_ss)


def diagnose(values, contrasts):
    treated = np.stack([values[arm] for arm in TREATED])
    basic = np.broadcast_to(values["basic"], treated.shape)
    paired = treated - basic
    axes = ("policy", "fold", "setting")
    absolute, tx = factor_components(treated, axes)
    relative, delta = factor_components(paired, axes)
    _, base = factor_components(basic, axes)
    levels = ("centered_total", "fold:setting")
    return dict(
        correlation=float(np.corrcoef(treated.ravel(), basic.ravel())[0, 1]),
        treated=absolute, paired=relative,
        identities={
            levels[0]: ss_identity(treated - treated.mean(), basic - basic.mean(), paired - paired.mean()),
            levels[1]: ss_identity(tx[levels[1]], base[levels[1]], delta[levels[1]]),
        },
        repeatability={arm: repeatability(values[arm]) for arm in ARMS},
        family_repeatability={family: repeatability(contrasts[contrast])
                              for family, contrast in FAMILIES.items()},
        all_arm_repeatability=repeatability(np.stack(list(values.values()))),
        basic_and_treated_repeatability=repeatability(np.stack([values["basic"], *treated])),
    )


def regenerate(input_dir):
    fits, coverage, pooled, archive = load_inputs(input_dir)
    results = {}
    for dataset in DATASETS:
        def grid(arm, field, convert):
            return np.array([[convert(fits[dataset, arm, str(fold), str(seed)][field])
                              for seed in SEEDS] for fold in FOLDS])

        constant = {arm: grid(arm, "constant_score", lambda v: v == "1") for arm in ARMS}
        data = dict(
            fits=len(ARMS) * len(FOLDS) * len(SEEDS),
            constant_fits=[dict(arm=arm, fold=int(fold), seed=SEEDS[col])
                           for arm, flags in constant.items()
                           for fold, col in zip(*np.nonzero(flags), strict=True)],
            loss_above_five=sum(int(np.count_nonzero(grid(arm, "epochs_loss_gt_5", int))) for arm in ARMS),
            metrics={}, pooled={},
        )
        for metric in METRICS:
            values = {arm: grid(arm, metric, float) for arm in ARMS}
            specs = dict(contrast_specs())
            gaps = {f"{family}_heavy_minus_light": {f"basic_{family}_heavy": 1,
                                                   f"basic_{family}_light": -1}
                    for family in FAMILIES}
            complete_policies = {f"{family}_complete_minus_none":
                                 {f"basic_{family}_light": .5, f"basic_{family}_heavy": .5, "none": -1}
                                 for family in FAMILIES}
            contrasts = {key: sum(weight * values[arm] for arm, weight in weights.items())
                         for key, weights in (specs | gaps | complete_policies).items()}
            summaries = {key: paired_summary(contrasts[key], np.logical_or.reduce(
                [constant[arm] for arm in weights])) for key, weights in (specs | gaps | complete_policies).items()}
            data["metrics"][metric] = dict(
                arms={arm: dict(**describe(values[arm]), constant=int(constant[arm].sum())) for arm in ARMS},
                contrasts=summaries,
                **diagnose(values, contrasts),
            )
            for family, key in FAMILIES.items():
                for seed in SEEDS:
                    value = sum(weight * float(pooled[dataset, arm, str(seed)][metric])
                                for arm, weight in specs[key].items())
                    data["pooled"].setdefault(metric, {}).setdefault(family, {})[str(seed)] = value
            for key in specs:
                for mode in MODES:
                    row = archive[dataset, mode, key, metric]
                    expected = summaries[key]["setting_means"]["11"] if mode == "seed11" else summaries[key]["mean"]
                    verify_close(expected, float(row["estimate"]), f"{dataset}/{mode}/{key}/{metric}")
        primary = [archive[dataset, "seed11", key, "auroc"] for key in FAMILIES.values()]
        holm = holm_adjust({row["contrast"]: probability(row["p_boot"]) for row in primary})
        for row in primary:
            verify_close(holm[row["contrast"]], probability(row["p_holm"]), "primary Holm adjustment")
        results[dataset] = data
    return dict(datasets=results, coverage=coverage, conditional_contrasts=list(archive.values()),
                pooled_metrics=list(pooled.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("build/study-results"))
    args = parser.parse_args()
    results = regenerate(args.input_dir)
    from .report import write_reports

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    write_reports(results, args.out_dir)
    print(f"Regenerated study outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
