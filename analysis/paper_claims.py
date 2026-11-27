"""Numeric provenance for the Results and Discussion paragraphs."""

from .report import Number
from .reproduce import ARCHIVED, ARMS, DATASETS, FAMILIES, FOLDS, SEEDS, TREATED

DEVELOPMENT = "development and recipe settings outside the final-fit aggregation"


def prose_claims(results):
    data = results["datasets"]
    auc = {dataset: item["metrics"]["auroc"] for dataset, item in data.items()}
    inb, isic, rsna = (auc[dataset]["contrasts"] for dataset in DATASETS)
    inb_gamma, inb_clahe = inb["H1"], inb["H2"]
    isic_gamma, isic_clahe = isic["H1"], isic["H2"]
    rsna_gamma, rsna_clahe = rsna["H1"], rsna["H2"]
    archive = {(r["dataset"], r["analysis"], r["contrast"], r["metric"]): r
               for r in results["conditional_contrasts"]}
    fits = data["isic2020"]["constant_fits"]
    basic_collapse = next(row for row in fits if row["arm"] == "basic")
    clahe_collapse = next(row for row in fits if row["arm"] == "basic_clahe_light")
    fold, seed = basic_collapse["fold"], str(basic_collapse["seed"])
    isic_heavy = isic["S4_basic_gamma_heavy"]
    isic_clean = isic_heavy["without_constant"]
    light, heavy = (inb[f"S4_basic_gamma_{dose}"] for dose in ("light", "heavy"))
    gap = inb["gamma_heavy_minus_light"]
    inb_clahe_repeat = auc["inbreast"]["family_repeatability"]["clahe"]
    rsna_clahe_ap = data["rsna2018"]["metrics"]["ap"]["contrasts"]["H2"]
    isic_clahe_ap = data["isic2020"]["metrics"]["ap"]["contrasts"]["H2"]
    same_hw = SEEDS[1:]

    def n(value, digits=0):
        return Number(value, digits)

    def mean(stats):
        return n(stats["mean"], 4)

    def primary(stats):
        return n(stats["setting_means"]["11"], 4)

    def setting_means(stats):
        return [n(stats["setting_means"][str(seed)], 4) for seed in SEEDS]

    def ci(dataset, contrast, mode="three_seed_mean", metric="auroc", digits=4, level="ci95"):
        row = archive[dataset, mode, contrast, metric]
        return [Number(float(row[f"{level}_{end}"]), digits, ARCHIVED) for end in ("lo", "hi")]

    def zero_if(condition, source=None):
        return Number(0 if condition else 1, source=source) if source else n(0 if condition else 1)

    def includes_zero(interval):
        return zero_if(interval[0].value <= 0 <= interval[1].value, ARCHIVED)

    def p_holm(dataset, contrast):
        value = float(archive[dataset, "seed11", contrast, "auroc"]["p_holm"])
        return Number(.001, 3, ARCHIVED, "<") if value < .001 else Number(value, 3, ARCHIVED)

    def primary_ci_excludes_zero(dataset, contrast):
        row = archive[dataset, "seed11", contrast, "auroc"]
        return float(row["ci975_lo"]) * float(row["ci975_hi"]) > 0

    treated_collapse = sum(row["arm"] in ("basic", *TREATED) for row in fits)
    per_arm = auc["isic2020"]["arms"]
    constant_arms = [arm for arm, stats in per_arm.items() if stats["constant"]]
    nonconstant_arms = [arm for arm, stats in per_arm.items() if not stats["constant"]]
    sd_ratio = min(per_arm[arm]["sd"] for arm in constant_arms) / max(
        per_arm[arm]["sd"] for arm in nonconstant_arms)
    grid_n = len(FOLDS) * len(SEEDS)
    fold_df = len(FOLDS) - 1
    setting_df = len(SEEDS) - 1
    interaction_df = fold_df * setting_df
    inb_gamma_ci = ci("inbreast", "H1")
    rsna_cis = [ci("rsna2018", family, mode, level=level) for family in FAMILIES.values()
                for mode, level in (("seed11", "ci975"), ("three_seed_mean", "ci95"))]
    heavy_primary_ci = ci("inbreast", "S4_basic_gamma_heavy", "seed11")
    heavy_all_ci = ci("inbreast", "S4_basic_gamma_heavy")
    light_all_ci = ci("inbreast", "S4_basic_gamma_light")
    small = [r for values in auc.values() for r in values["family_repeatability"].values()]
    family_shares = [stats["ss"]["terms"]["fold:setting"]["share_pct"] for values in auc.values()
                     for key, stats in values["contrasts"].items() if key in FAMILIES.values()]
    ratio = {dataset: item["paired"]["total"] / item["treated"]["total"] for dataset, item in auc.items()}

    yield "Basic augmentation averaged", (
        *[n(data[dataset]["metrics"][metric]["arms"]["basic"]["mean"], 3)
          for metric in ("auroc", "ap") for dataset in DATASETS],
        n(grid_n), *[n(item["arms"]["basic"]["sd"], 4) for item in auc.values()],
    )
    yield r"On ISIC, \w+ of the \w+ policies", (
        n(len(constant_arms)), n(len(ARMS)), n(min(per_arm[arm]["constant"] for arm in constant_arms)),
        n(3 if sd_ratio > 3 else sd_ratio), n(per_arm["gamma_light"]["sd"], 4),
        n(per_arm["basic_gamma_heavy"]["sd"], 4),
    )
    yield "At the primary setting, both INbreast", (
        primary(inb_gamma), p_holm("inbreast", "H1"), primary(inb_clahe), p_holm("inbreast", "H2"),
        Number(97.5, 1, ARCHIVED),
        zero_if(all(primary_ci_excludes_zero("inbreast", key) for key in FAMILIES.values()), ARCHIVED),
        primary(isic_gamma), primary(isic_clahe),
        max((p_holm("isic2020", key) for key in FAMILIES.values()), key=lambda item: item.value),
        n(fold), n(len(rsna_cis)), zero_if(all(includes_zero(interval).value == 0 for interval in rsna_cis), ARCHIVED),
    )
    yield "Averaging across settings changed", (
        mean(inb_gamma), *inb_gamma_ci, mean(inb_clahe),
        zero_if(ci("inbreast", "H2")[1].value < 0, ARCHIVED), mean(isic_gamma),
        zero_if(ci("isic2020", "H1")[0].value > 0, ARCHIVED), mean(isic_clahe),
        includes_zero(ci("isic2020", "H2")),
    )
    yield "Pooling held-out predictions", (
        *[Number(data[dataset]["pooled"]["auroc"][family]["11"], 4, ARCHIVED)
          for dataset in DATASETS for family in FAMILIES], n(len(FOLDS)),
    )
    yield "The fold-by-setting interaction had the largest share", (
        *[n(item["paired"]["terms"]["fold:setting"]["share_pct"], 1) for item in auc.values()],
        n(int("basic" in ARMS)), n(len(TREATED)), n(fold_df), n(interaction_df),
    )
    yield "INbreast gamma ranged", (
        n(inb_gamma["min"], 4), n(inb_gamma["max"], 4), n(inb_gamma["positive"]), n(inb_gamma["n"]),
        *map(n, SEEDS), *setting_means(inb_gamma), n(inb_clahe["negative"]), n(inb_clahe["n"]),
        n(len(SEEDS)), *setting_means(inb_clahe), zero_if(min(rsna_gamma["leave_one_fold_out"]) > 0),
        includes_zero(ci("rsna2018", "H1")), *setting_means(rsna_gamma),
    )
    yield r"Across the \w+ seeds on the same GPU", (
        n(len(same_hw)), n(min(r["n"] - r["agreement"] for r in small)),
    )
    yield "At the family level, fold-by-setting", (
        n(min(family_shares), 1), n(max(family_shares), 1), n(100 * interaction_df / (grid_n - 1), 1),
    )
    yield "The prespecified secondary comparisons", (
        n(2), n(len(SEEDS)), mean(light), n(light["positive"]), n(light["n"]), mean(heavy), n(heavy["positive"]),
        primary(light), primary(heavy), includes_zero(heavy_primary_ci), *heavy_primary_ci, *heavy_all_ci,
        n(len(SEEDS)), zero_if(light_all_ci[1].value < 0, ARCHIVED), *light_all_ci,
    )
    yield "In a direct post-hoc comparison", (
        n(gap["positive"]), n(gap["n"]), mean(gap), n(95),
        Number(None, source="direct dose bootstrap absent from supplied archive"),
        Number(None, source="direct dose bootstrap absent from supplied archive"),
        n(SEEDS[0]), primary(gap),
        Number(None, source="direct dose bootstrap absent from supplied archive"),
        Number(None, source="direct dose bootstrap absent from supplied archive"),
        n(gap["setting_positive"]["11"]), n(len(FOLDS)), mean(inb["S4_basic_clahe_light"]),
        mean(inb["S4_basic_clahe_heavy"]), n(inb_clahe_repeat["agreement"]), n(inb_clahe_repeat["n"]),
        *map(n, same_hw),
    )
    yield "On RSNA, light CLAHE averaged", (
        mean(rsna["S4_basic_clahe_light"]), *ci("rsna2018", "S4_basic_clahe_light"),
        mean(rsna["S4_basic_clahe_heavy"]),
    )
    yield r"\w+ of the \d+ ISIC fits", (
        n(len(fits)), n(data["isic2020"]["fits"]), n(len(constant_arms)), n(treated_collapse), n(len(fits)),
        n(1 + len(TREATED)), *map(n, SEEDS), *[n(sum(row["seed"] == s for row in fits)) for s in SEEDS],
        n(5), n(data["isic2020"]["loss_above_five"]), n(data["isic2020"]["fits"]),
    )
    omitted_n = len(FOLDS) - 1
    yield "Basic collapsed in fold", (
        n(fold), n(int(seed)), n(isic_gamma["matrix"][fold][0], 4), n(isic_clahe["matrix"][fold][0], 4),
        primary(isic_gamma), n(isic_gamma["seed11_leave_one_fold_out"][fold], 4),
        primary(isic_clahe), n(isic_clahe["seed11_leave_one_fold_out"][fold], 4), n(omitted_n),
        n(clahe_collapse["fold"]), n(clahe_collapse["seed"]),
        n(isic_clahe["matrix"][clahe_collapse["fold"]][0], 4), n(omitted_n),
    )
    yield "Heavy gamma was positive", (
        n(isic_heavy["positive"]), n(isic_heavy["n"]), mean(isic_heavy), n(isic_clean["n"]),
        n(isic_clean["positive"]), mean(isic_clean), n(3),
        mean(isic_gamma["without_constant"]), n(isic_gamma["without_constant"]["n"]),
        mean(isic_clahe["without_constant"]), n(isic_clahe["without_constant"]["n"]),
        n(len(FAMILIES)), n(int(seed)), n(len(FOLDS)), primary(isic_clahe), primary(isic_clahe_ap), n(fold),
    )
    yield "For CLAHE versus basic", (
        n(len(SEEDS)), n(rsna_clahe_ap["mean"], 5), *ci("rsna2018", "H2", metric="ap", digits=5),
        n(rsna_clahe_ap["positive"]), n(rsna_clahe_ap["n"]), n(rsna["S3"]["negative"]), n(rsna["S3"]["n"]),
        mean(rsna["S3"]), *ci("rsna2018", "S3"),
    )
    yield "Against basic, adding gamma or CLAHE", (
        n(len(SEEDS)), mean(rsna_gamma), mean(rsna_clahe), mean(rsna["gamma_complete_minus_none"]),
        mean(rsna["clahe_complete_minus_none"]),
    )
    yield "Subtraction reduces dispersion", (
        n(1), n(auc["rsna2018"]["correlation"], 3), n(100 * (1 - ratio["rsna2018"])),
        n(auc["inbreast"]["correlation"], 3), n(auc["isic2020"]["correlation"], 3),
        n(100 * (ratio["inbreast"] - 1)),
    )
    yield "The same arithmetic applies", (n(auc["inbreast"]["treated"]["n"]),)
    yield "On INbreast the treated term", (
        *[n(auc[dataset]["identities"]["fold:setting"][term],
             6 if dataset == "rsna2018" else 0 if dataset == "isic2020" and term == "cross" else 5)
          for dataset in DATASETS for term in ("treated", "basic", "cross", "paired")],
        n(len(DATASETS)),
    )
    yield "On ISIC, the failed baseline enters", (
        n(sum(row["arm"] == "basic" and row["seed"] == SEEDS[0] for row in fits)),
        n(len(FOLDS)), n(len(FAMILIES)),
    )
    yield r"On INbreast, the \w+ gamma doses", (
        n(2), includes_zero(heavy_all_ci), n(inb_clahe_repeat["agreement"]), n(inb_clahe_repeat["n"]),
    )
    max_change = max(r["median_change"] for r in auc["rsna2018"]["family_repeatability"].values())
    yield "On RSNA, adding a transform", (n(.009 if max_change < .009 else max_change, 3),)
    # Learning-rate selection used development runs outside the final study matrix.
    yield "The horizontal flip mirrors", (
        *([Number(None, source=DEVELOPMENT)] * 9),
        n(data["isic2020"]["fits"]),
        *([Number(None, source=DEVELOPMENT)] * 3),
    )
