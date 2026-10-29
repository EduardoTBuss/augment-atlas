"""Render separate SAP v1 studies as paper artifacts; no statistical computation.

See docs/paper-results.md. Input artifacts must have been audited upstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

from . import analysis_sap_v1 as sap
from . import clinical_metrics as cm

PROTOCOL = "augment-atlas-paper-results-v1"
COLUMN_WIDTH_INCHES = 3.35
SEALED_SAP_SHA256 = "bb3bb94a76705f2499f3d32c3778a0c3d212abc314d92ec08678d6ea1d2c2458"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_sap_seal():
    digest = _sha(sap.__file__)
    if digest != SEALED_SAP_SHA256:
        raise ValueError("analysis_sap_v1.py does not match the sealed SAP SHA-256")
    return digest


def prediction_completeness(rows, report):
    """Count supplied cells; use emitted fold statuses where available.

    Missing sensitivity output is covered by structural checks on validated rows,
    without computing metrics, intervals, or additional SAP analyses.
    """
    universe, labels, cells = {}, {}, {}
    for row in rows:
        fold = row["fold"]
        universe.setdefault(fold, set()).add(row["image"])
        labels.setdefault(fold, set()).add(row["label"])
        cells.setdefault((row["arm"], row["seed"], fold), {})[row["image"]] = row["score"]
    emitted = {}
    for analysis in (report["primary"], report["sensitivities"]["three_seed_mean"]):
        for arm in analysis.get("arms", []):
            for entry in arm["per_fold"]:
                emitted[arm["arm"], entry["seed"], entry["fold"]] = entry
    output = []
    for arm in report["primary"]["arms"]:
        for seed in report["declared_training_seeds"]:
            folds = []
            for fold in sap.FOLDS:
                key = (arm["arm"], seed, fold)
                cell = cells.get(key, {})
                expected = universe.get(fold, set())
                entry = emitted.get(key)
                if entry is not None:
                    status, reason = entry["status"], entry["reason"]
                else:
                    reason = ("missing_fold" if not cell else
                              "unpaired_images" if set(cell) != expected else
                              "nonfinite_scores" if any(not math.isfinite(v) for v in cell.values()) else
                              "no_positive_examples" if 1 not in labels.get(fold, set()) else
                              "no_negative_examples" if 0 not in labels.get(fold, set()) else None)
                    status = "not_evaluable" if reason else "ok"
                folds.append({"fold": fold, "observed_images": len(cell),
                              "expected_images": len(expected), "status": status, "reason": reason})
            output.append({"arm": arm["arm"], "training_seed": seed,
                           "observed_folds": sum(item["observed_images"] > 0 for item in folds),
                           "evaluable_folds": sum(item["status"] == "ok" for item in folds),
                           "folds": folds})
    return output


def _assignment(value):
    dataset, sep, source = value.partition("=")
    if not sep or not dataset.strip() or not source.strip():
        raise ValueError("expected DATASET=VALUE")
    if any(ord(char) < 32 for char in dataset):
        raise ValueError("dataset must not contain control characters")
    return dataset, source


def _mapping(values, *, label):
    result = {}
    for value in values:
        dataset, source = _assignment(value)
        if dataset in result:
            raise ValueError(f"duplicate {label} for dataset {dataset}")
        result[dataset] = source
    return result


def _slug(dataset):
    # Digest prevents collisions after sanitization; no input path enters output.
    stem = re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")[:40] or "dataset"
    return stem + "-" + hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:10]


def _tex(value):
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
               "<": r"\textless{}", ">": r"\textgreater{}"}
    return "".join(escapes.get(char, char) for char in str(value))


def _number(value):
    return "--" if value is None else f"{value:.4f}"


def _effect(value, interval="ci95"):
    estimate, ci = value.get("estimate"), value.get(interval)
    result = _number(estimate)
    if ci is not None:
        result += f" [{ci[0]:.4f}, {ci[1]:.4f}]"
    if value.get("status") == "not_evaluable":
        result += " (NE)"
    elif value.get("status") == "descriptive_only":
        result += " (D)"
    return result


def dataset_table(report):
    """Only format full-precision values returned by SAP; never recompute them."""
    dataset = _tex(report["dataset"])
    primary = report["primary"]
    completeness = {(item["arm"], item["training_seed"]): item for item in report["completeness"]}
    primary_seed = primary["training_seeds"][0]
    lines = [r"\begin{table*}[t]", r"\centering\scriptsize",
             r"\caption{" + dataset + r": primary-seed fold-macro AUROC points and prespecified SAP contrasts; "
             r"intervals apply only to contrasts. Brackets denote paired group-bootstrap 95\% percentile intervals. H1/H2 also show "
             r"97.5\% marginal intervals and within-dataset Holm-adjusted $p$ (two hypotheses).}",
             r"\label{tab:paper-" + _slug(report["dataset"]) + "}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrr}", r"\hline",
             r"Arm & AUROC (point) & Folds O/E \\",
             r"\hline"]
    for arm in primary["arms"]:
        name = arm["arm"]
        counts = completeness[name, primary_seed]
        lines.append(" & ".join((_tex(name), _effect({"estimate": (arm["metrics"] or {}).get("auroc"),
                                           "status": arm["status"]}),
                                  f'{counts["observed_folds"]}/{counts["evaluable_folds"]}')) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\par\smallskip",
              r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llll}", r"\hline",
              r"Contrast & $\Delta$AUROC [95\% CI] & 97.5\% marginal CI & Holm $p$ \\", r"\hline"]
    for item in primary["contrasts"]:
        value = item["auroc"]
        ci = value.get("ci97_5")
        ci_text = "--" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        p = value.get("p_holm")
        lines.append(" & ".join((_tex(item["id"]), _effect({**value, "status": item["status"]}),
                                  ci_text, "--" if p is None else f"{p:.4g}")) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\par\smallskip",
              r"\begin{tabular}{rrrr}", r"\hline",
              r"Fold & Images & Groups & Positive groups \\", r"\hline"]
    for fold in report["coverage"]:
        lines.append(" & ".join(str(fold[key]) for key in
                                  ("fold", "images", "groups", "positive_groups")) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\par\smallskip",
              r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrr}", r"\hline",
              "Arm & " + " & ".join(f"Seed {seed} O/E" for seed in report["declared_training_seeds"]) + r" \\",
              r"\hline"]
    for arm in primary["arms"]:
        cells = [completeness[arm["arm"], seed] for seed in report["declared_training_seeds"]]
        lines.append(_tex(arm["arm"]) + " & " + " & ".join(
            f'{cell["observed_folds"]}/{cell["evaluable_folds"]}' for cell in cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\par\smallskip",
              r"\parbox{\linewidth}{\raggedright "
              + f"Primary training seed: {primary_seed}. "
              + "O/E: observed/evaluable folds out of five; observed folds may have incomplete images. "
              + "Counts describe the supplied E prediction universe, not audited D/E eligibility. "
              + "NE: not evaluable; D: descriptive only (no inference). "
              + "Positive groups contain at least one positive image. "
              + "Three-seed sensitivity status: "
              + _tex(report["sensitivities"]["three_seed_mean"]["status"]) + ". "
              + "Analysis scope: " + _tex(report["analysis_scope"]) + ".}"]
    # Retain machine reasons, including unavailable inference with a defined point.
    reasons = sorted({value["reason"] for value in
                      primary["arms"] + primary["contrasts"] if value.get("reason")})
    for reason in reasons:
        lines.append(r"\par Reason: " + _tex(reason) + ".")
    for warning in report["warnings"]:
        lines.append(r"\par SAP warning: " + _tex(warning) + ".")
    lines += [r"\end{table*}", ""]
    return "\n".join(lines)


def forest_figure(reports):
    """One vertical panel per separate study at fixed paper-column width."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(len(reports), 1, figsize=(COLUMN_WIDTH_INCHES, 2.5 * len(reports)),
                             squeeze=False, layout="constrained")
    for ax, report in zip(axes[:, 0], reports):
        items = report["primary"]["contrasts"]
        labels = []
        for index, item in enumerate(items):
            value = item["auroc"]
            # A panel label is not an embedded plot title; the caption describes inference.
            label = item["id"].replace("S4_basic_", "S4 b+").replace("S4_", "S4 ").replace("S5_", "S5 ")
            label = label.replace("_light", " L").replace("_heavy", " H")
            if item["status"] == "not_evaluable":
                label += " [NE]"
                ax.plot(.98, index, marker="x", color=".35", markersize=3,
                        transform=ax.get_yaxis_transform())
            elif item["status"] == "descriptive_only":
                label += " [D]"
            labels.append(label)
            if value["estimate"] is not None:
                if value.get("ci97_5") is not None:
                    ax.plot(value["ci97_5"], [index, index], color="#24587a", linewidth=.7)
                if value.get("ci95") is not None:
                    ax.plot(value["ci95"], [index, index], color="#24587a", linewidth=2)
                ax.plot(value["estimate"], index, "o", color="#24587a", markersize=3)
        ax.set_yticks(range(len(items)), labels, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color=".5", linestyle="--", linewidth=.6)
        ax.set_xlabel("Absolute delta AUROC", fontsize=8)
        ax.set_ylabel(report["dataset"], fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.margins(x=.12, y=.05)
    return fig


def _snippet(reports):
    lines = ["% Requires graphicx. Set this macro before input when using a subdirectory:",
             r"% \newcommand{\PaperResultsPath}{path/to/generated/results}",
             r"\providecommand{\PaperResultsPath}{.}",
             "The following tables contain separate dataset summaries from SAP v1. "
             "They list primary-seed fold-macro AUROC points by arm, H1/H2 and S1--S5 "
             "with 95\\% contrast intervals, H1/H2 with 97.5\\% marginal intervals and Holm-adjusted "
             "$p$, fold group and positive counts, and prediction-matrix completeness."]
    for report in reports:
        lines.append(r"\input{\PaperResultsPath/table-" + _slug(report["dataset"]) + ".tex}")
    lines += [r"\input{\PaperResultsPath/forest.tex}",
              "SAP v1 is exploratory. Intervals do not establish equivalence or absence of an effect. "
              "Datasets remain separate studies; no cross-dataset pooled effect or ranking is reported. "
              "The primary analysis uses the first predeclared training seed; the three-seed mean "
              "is a sensitivity only when all arms and folds are evaluable. "
              "Undefined bootstrap replicates make inference not evaluable without retry or omission. "
              "Folds with fewer than eight positive groups carry the SAP coverage warning. "
              "INbreast with an HPO-D recipe is descriptive only. "
              "Prediction artifacts alone do not verify complete trusted E coverage, D/E isolation, "
              "eligibility, SAP sealing chronology, or model execution. "
              "Provenance hashes record bindings and code, not approval; "
              r"\texttt{execution\_verified} and \texttt{scientific\_eligible} remain false.", ""]
    return "\n\n".join(lines)


FOREST_TEX = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{\PaperResultsPath/forest.pdf}
\caption{Separate dataset panels of primary-seed paired absolute $\Delta$AUROC from SAP v1.
Points show estimates; thick lines are 95\% percentile intervals, and thin lines are
97.5\% marginal intervals for H1/H2. The dashed line marks zero. NE denotes unavailable
inference (a defined point may remain); D denotes descriptive-only estimates without
intervals. H1/H2 precede exploratory contrasts. No pooled cross-dataset effect is shown.}
\label{fig:paper-results-forest}
\end{figure}
"""


def generate(campaigns, out, *, seed, B=10000, recipe_sources=None,
             training_seeds=sap.TRAIN_SEEDS, block_size=64):
    """Read and analyze all inputs before replacing output; publish manifest last."""
    if not campaigns:
        raise ValueError("at least one campaign is required")
    recipe_sources = recipe_sources or {}
    if set(recipe_sources) - set(campaigns):
        raise ValueError("recipe-source dataset has no campaign")
    reports = []
    source_hash = _check_sap_seal()
    for dataset, predictions in sorted(campaigns.items()):
        rows, metadata = sap.read_predictions(predictions, dataset)
        source = recipe_sources.get(dataset) or metadata["recipe_source"]
        if recipe_sources.get(dataset) and metadata["recipe_source"] and source != metadata["recipe_source"]:
            raise ValueError("conflicting recipe_source declarations")
        report = sap.analyze(rows, dataset=dataset, seed=seed, B=B, block_size=block_size,
                             recipe_source=source, training_seeds=training_seeds)
        report["completeness"] = prediction_completeness(rows, report)
        report["provenance"] = {**metadata, "recipe_source": source, "source_sha256": source_hash,
                                "clinical_metrics_sha256": _sha(cm.__file__),
                                "numpy_version": np.__version__}
        reports.append(report)
    bundle = {"protocol": PROTOCOL, "sap_protocol": sap.PROTOCOL,
              "provenance": {"sap_source_sha256": source_hash,
                             "generator_sha256": _sha(__file__),
                             "clinical_metrics_sha256": _sha(cm.__file__),
                             "numpy_version": np.__version__},
              "datasets": reports, "execution_verified": False, "scientific_eligible": False}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper-results-", dir=out.parent) as temporary:
        stage = Path(temporary)
        for report in reports:
            (stage / f'table-{_slug(report["dataset"])}.tex').write_text(dataset_table(report), encoding="utf-8")
        (stage / "results.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (stage / "results.tex").write_text(_snippet(reports), encoding="utf-8")
        (stage / "forest.tex").write_text(FOREST_TEX, encoding="utf-8")
        figure = forest_figure(reports)
        try:
            figure.savefig(stage / "forest.pdf", metadata={"CreationDate": None, "ModDate": None})
            figure.savefig(stage / "forest.png", dpi=300)
        finally:
            from matplotlib import pyplot as plt
            plt.close(figure)
        names = sorted(path.name for path in stage.iterdir())
        manifest = {"protocol": PROTOCOL, "provenance": bundle["provenance"],
                    "outputs": {name: _sha(stage / name) for name in names}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").unlink(missing_ok=True)
        for name in names + ["manifest.json"]:
            os.replace(stage / name, out / name)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", required=True, metavar="DATASET=PREDICTIONS")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="bootstrap RNG seed, not training seed")
    parser.add_argument("--B", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--training-seeds", type=int, nargs=3, default=sap.TRAIN_SEEDS,
                        metavar=("PRIMARY", "SENS2", "SENS3"))
    parser.add_argument("--recipe-source", action="append", default=[], metavar="DATASET=KIND",
                        help="per-dataset external_fixed or hpo_d; required for native INbreast")
    args = parser.parse_args(argv)
    try:
        generate(_mapping(args.campaign, label="campaign"), args.out, seed=args.seed, B=args.B,
                 block_size=args.block_size, training_seeds=args.training_seeds,
                 recipe_sources=_mapping(args.recipe_source, label="recipe source"))
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"paper_results: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
