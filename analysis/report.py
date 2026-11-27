"""Display the regenerated tables and figures with their numerical precision."""

from dataclasses import dataclass
import math

from .reproduce import ARCHIVED, ARMS, DATASETS, FAMILIES, FOLDS, RECOMPUTED, SEEDS, write_csv


@dataclass(frozen=True)
class Number:
    value: float | None
    digits: int = 0
    source: str = RECOMPUTED
    relation: str = ""

    @property
    def display(self):
        if self.value is None:
            return ""
        return f"{self.relation}{format(self.value, f'.{self.digits}f')}"


def tables(results):
    a = results["datasets"]
    archive = {(r["dataset"], r["analysis"], r["contrast"], r["metric"]): r
               for r in results["conditional_contrasts"]}
    table = {name: {} for name in ("auroc", "planned", "grid", "dose", "repeatability")}
    for arm in ARMS:
        table["auroc"][arm] = {}
        for ds in DATASETS:
            stats = a[ds]["metrics"]["auroc"]["arms"][arm]
            table["auroc"][arm][ds] = (Number(stats["mean"], 4), Number(stats["sd"], 4),
                                        Number(stats["constant"]))
    for ds, data in a.items():
        auc = data["metrics"]["auroc"]
        for family, cid in FAMILIES.items():
            def interval(mode, prefix):
                row = archive[ds, mode, cid, "auroc"]
                estimate = auc["contrasts"][cid]["setting_means"]["11"] if mode == "seed11" else auc["contrasts"][cid]["mean"]
                return (Number(estimate, 4),
                        Number(float(row[f"{prefix}_lo"]), 4, ARCHIVED),
                        Number(float(row[f"{prefix}_hi"]), 4, ARCHIVED))

            p = float(archive[ds, "seed11", cid, "auroc"]["p_holm"])
            p_field = Number(.001, 3, ARCHIVED, "<") if p < .001 else Number(p, 3, ARCHIVED)
            table["planned"][ds, family] = {
                "primary": interval("seed11", "ci975"),
                "holm_p": (p_field,),
                "three_seed_mean": interval("three_seed_mean", "ci95"),
            }
            stats = auc["contrasts"][cid]
            terms = stats["ss"]["terms"]
            table["dose"][ds, family] = {
                "light": (Number(auc["contrasts"][f"S4_basic_{family}_light"]["mean"], 4),),
                "heavy": (Number(auc["contrasts"][f"S4_basic_{family}_heavy"]["mean"], 4),),
                "sd": (Number(stats["sd"], 4),),
                "positive": (Number(stats["positive"]), Number(stats["n"])),
                **{term: (Number(terms[term]["share_pct"], 1),)
                   for term in ("fold", "setting", "fold:setting")},
            }
        table["grid"][ds] = {
            "correlation": (Number(auc["correlation"], 3),),
            "treated_ss": (Number(auc["treated"]["total"], 5),),
            "paired_ss": (Number(auc["paired"]["total"], 5),),
            "fold_share": tuple(Number(auc[key]["terms"]["fold"]["share_pct"], 1)
                                for key in ("treated", "paired")),
            "rms": (Number(auc["basic_and_treated_repeatability"]["rms_change"], 4),),
        }
        repeat = auc["family_repeatability"]
        table["repeatability"][ds] = {
            "unchanged_policy_median": (Number(auc["all_arm_repeatability"]["median_change"], 4),),
            **{key: numbers for family, stats in repeat.items() for key, numbers in (
                (f"{family}_agreement", (Number(stats["agreement"]), Number(stats["n"]))),
                (f"{family}_median", (Number(stats["median_change"], 4),)),
            )},
        }
    return table


def display_cell(column, numbers):
    values = [number.display for number in numbers]
    if column in DATASETS:
        suffix = f" [constant: {values[2]}]" if numbers[2].value else ""
        return f"{values[0]} ± {values[1]}{suffix}"
    if column in ("primary", "three_seed_mean"):
        return f"{values[0]} [{values[1]}, {values[2]}]"
    if column == "fold_share":
        return " → ".join(values)
    return "/".join(values)


def figure_rows(results):
    paired, shares = [], []
    for dataset, data in results["datasets"].items():
        auc = data["metrics"]["auroc"]
        for family, cid in FAMILIES.items():
            contrast = auc["contrasts"][cid]
            digits = 2 if dataset == "rsna2018" else 1
            for fold in FOLDS:
                for col, seed in enumerate(SEEDS):
                    value = 100 * contrast["matrix"][fold][col]
                    label = f"{value:+.{digits}f}" if value and (digits == 2 or abs(value) >= .05) else f"{0:.{digits}f}"
                    paired.append(dict(dataset=dataset, family=family, fold=fold, seed=seed, auroc_points=value,
                                       label=label, constituent_constant=contrast["constant_score"][fold][col]))
        for kind in ("treated", "paired"):
            stats = auc[kind]
            for term, value in stats["terms"].items():
                shares.append(dict(dataset=dataset, analysis=kind, term=term, **value,
                                   total=stats["total"], n=stats["n"], label=f'{value["share_pct"]:.1f}'))
    return paired, shares


def plot(paired, shares, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    import numpy as np

    def save(fig, name):
        fig.savefig(directory / f"{name}.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
        fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9, 4.7), layout="constrained")
    for col, (dataset, title) in enumerate(DATASETS.items()):
        records = [row for row in paired if row["dataset"] == dataset]
        maximum = max(abs(row["auroc_points"]) for row in records)
        step = 5 if maximum > 5 else .5
        limit = math.ceil(maximum / step) * step
        for row_index, family in enumerate(FAMILIES):
            ax = axes[row_index, col]
            panel = [row for row in records if row["family"] == family]
            matrix = np.array([row["auroc_points"] for row in panel]).reshape(len(FOLDS), len(SEEDS))
            image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
            for cell in panel:
                fold, seed_index = cell["fold"], SEEDS.index(cell["seed"])
                ax.text(seed_index, fold, cell["label"], ha="center", va="center", fontsize=9,
                        color="white" if abs(cell["auroc_points"]) > .65 * limit else "black")
                if cell["constituent_constant"]:
                    ax.add_patch(Rectangle((seed_index - .48, fold - .48), .96, .96,
                                           fill=False, edgecolor="black", linestyle=":", linewidth=1.5))
            ax.set_xticks(range(len(SEEDS)), labels=SEEDS)
            ax.set_yticks(FOLDS)
            ax.set(xlabel="Seed" if row_index else "", title=title if not row_index else "",
                   ylabel=f"{'Gamma' if family == 'gamma' else 'CLAHE'}\nFold" if not col else "")
        fig.colorbar(image, ax=axes[:, col], ticks=(-limit, 0, limit), shrink=.8, label="AUROC points")
    save(fig, "paired-effects")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.subplots_adjust(left=.20, right=.87, bottom=.27)
    palette = ("#0072B2", "#E69F00", "#CC79A7", "#009E73", "#56B4E9", "#6A3D9A", "#888888")
    labels = ("Policy", "Fold", "Setting", "Policy × fold", "Policy × setting",
              "Fold × setting", "Policy × fold × setting")
    ticks, bars = [], []
    for index, (dataset, title) in enumerate(DATASETS.items()):
        for offset, kind in enumerate(("treated", "paired")):
            y = index * 2.6 + offset
            ticks.append(y)
            bars.append(f"{title}  {'AUROC' if kind == 'treated' else 'ΔAUROC'}")
            records = [row for row in shares if row["dataset"] == dataset and row["analysis"] == kind]
            left = 0
            for row, color in zip(records, palette, strict=True):
                share = row["share_pct"]
                ax.barh(y, share, left=left, color=color, edgecolor="white", height=.82)
                if share >= 5:
                    ax.text(left + share / 2, y, row["label"], ha="center", va="center",
                            fontsize=8, color="white" if color in (palette[0], palette[5]) else "black")
                left += share
            digits = 6 if dataset == "rsna2018" else 5
            ax.text(102, y, f'{records[0]["total"]:.{digits}f}', va="center", fontsize=8)
    ax.set_yticks(ticks, labels=bars)
    ax.set(xlim=(0, 100), xlabel="Share of centered observed SS (%)")
    ax.invert_yaxis()
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend([Patch(facecolor=color) for color in palette], labels,
               loc="lower center", ncol=4, frameon=False, fontsize=8)
    save(fig, "factor-shares")


def write_reports(results, directory):
    for table, rows in tables(results).items():
        output = []
        for key, cells in rows.items():
            if isinstance(key, tuple):
                row = dict(dataset=DATASETS[key[0]], family="Gamma" if key[1] == "gamma" else "CLAHE")
            elif table == "auroc":
                row = dict(arm=ARMS[key])
            else:
                row = dict(dataset=DATASETS[key])
            row.update({DATASETS.get(column, column): display_cell(column, numbers)
                        for column, numbers in cells.items()})
            output.append(row)
        write_csv(directory / f"table-{table}.csv", output)
    paired, shares = figure_rows(results)
    write_csv(directory / "paired-effects-labels.csv", paired)
    write_csv(directory / "factor-shares.csv", shares)
    plot(paired, shares, directory)
