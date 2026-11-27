"""Compare a manuscript's result numbers with regenerated aggregate outputs."""

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import re

from .paper_claims import DEVELOPMENT, prose_claims
from .report import Number, tables
from .reproduce import ARMS, DATASETS, FAMILIES, FOLDS, SEEDS, TREATED, read_csv, write_csv

WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split(),
    range(16), strict=True,
))
NUMBERS = re.compile(r"(?P<relation>[<>]?)\s*(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+))"
                     r"|\b(?P<word>" + "|".join(WORDS) + r")\b", re.IGNORECASE)
DATASET_IDS = {label: key for key, label in DATASETS.items()}
ARM_IDS = {label: key for key, label in ARMS.items()}
REFERENCE = "archived descriptive figure data"


def numbers(text, include_words=False):
    text = re.sub(r"\\(?:ref|cite)\{[^}]+\}|\bRQ[123]\b", "", text)
    text = re.sub(r"(?<=\d)--(?=\d)", " ", text)
    text = text.replace("$", "").replace(r"\to", " ").replace("−", "-")
    for match in NUMBERS.finditer(text):
        word = match["word"]
        if word is None:
            yield match["relation"], Decimal(match["number"]), match[0].strip()
        elif include_words:
            yield "", Decimal(WORDS[word.lower()]), word


class Audit:
    def __init__(self):
        self.rows = []

    def record(self, location, observed, regenerated, source, status):
        self.rows.append(dict(location=location, manuscript=observed, regenerated=regenerated,
                              source=source, status=status))

    def compare(self, location, actual, expected):
        observed = "" if actual is None else actual[2]
        if expected.value is None:
            status = "out of scope" if expected.source == DEVELOPMENT else "unverified"
        else:
            status = "mismatch"
            relation = expected.relation if expected.relation in ("<", ">") else ""
            signed_zero = expected.relation != "+" or observed.startswith("+")
            if actual and signed_zero and actual[:2] == (relation, Decimal(expected.display.lstrip("<>"))):
                status = "match"
        self.record(location, observed, expected.display, expected.source, status)

    def sequence(self, location, actual, expected):
        for index in range(max(len(actual), len(expected))):
            key = f"{location}/{index + 1}"
            if index < len(expected):
                self.compare(key, actual[index] if index < len(actual) else None, expected[index])
            else:
                self.record(key, actual[index][2], "", "unmapped numeric token", "mismatch")

    def figure(self, location, actual, expected):
        self.record(location, actual, expected, REFERENCE, "match" if actual == expected else "mismatch")


def table_rows(tex, name):
    body = re.search(r"\\label\{tab:" + name + r"\}.*?\\midrule(.*?)\\bottomrule", tex, re.S)
    if not body:
        raise ValueError(f"table tab:{name} is missing or its structure has changed")
    dataset = None
    rows = {}
    for line in body[1].splitlines():
        heading = re.search(r"\\textit\{([^}]+)\}", line)
        if heading:
            dataset = DATASET_IDS.get(heading[1], dataset)
        if "&" not in line:
            continue
        columns = [part.strip() for part in line.strip().removesuffix(r"\\").split("&")]
        first = columns.pop(0)
        if name == "planned":
            key = (dataset, first.lower())
        elif name == "dose":
            dataset = DATASET_IDS[first] if first else dataset
            key = (dataset, columns.pop(0).lower())
        elif name == "auroc":
            key = ARM_IDS[first]
        else:
            key = DATASET_IDS[first]
        if key in rows:
            raise ValueError(f"duplicate row in tab:{name}: {key}")
        rows[key] = columns
    return rows


def caption(tex, label):
    match = re.search(r"\\caption\{([^\n]+)\}\s*\\label\{" + label + r"\}", tex)
    return match[1] if match else ""


def check_tables(audit, tex, results):
    n_grid = len(FOLDS) * len(SEEDS)
    n_interaction = (len(FOLDS) - 1) * (len(SEEDS) - 1)
    metadata = {
        "auroc": [len(ARMS), n_grid, len(FOLDS), len(SEEDS), n_grid - 1, .5],
        "planned": [len(FOLDS), SEEDS[0], 97.5, len(FAMILIES), n_grid, 95],
        "grid": [len(TREATED) * n_grid, *SEEDS[1:]],
        "dose": [n_grid, n_grid - 1, 0, len(FOLDS), len(SEEDS), n_interaction, n_grid - 1,
                 round(100 * n_interaction / (n_grid - 1), 1)],
        "repeatability": [*SEEDS[1:], len(ARMS), len(FOLDS)],
    }
    for name, expected_rows in tables(results).items():
        label = f"tab:{name}"
        text = re.sub(r"\bRTX A\d+\b", "", caption(tex, label))
        expected_caption = [Number(value, int(isinstance(value, float))) for value in metadata[name]]
        if name == "dose":
            expected_caption[2] = Number(0, relation=">")
        audit.sequence(f"{label}/caption", list(numbers(text, include_words=True)), expected_caption)
        actual_rows = table_rows(tex, name)
        for key, cells in expected_rows.items():
            location = f"tab:{name}/{'/'.join(key) if isinstance(key, tuple) else key}"
            actual_cells = actual_rows.pop(key, [])
            if len(actual_cells) != len(cells):
                audit.record(location, str(len(actual_cells)), str(len(cells)), "column count", "mismatch")
            for index, (column, expected) in enumerate(cells.items()):
                actual = actual_cells[index] if index < len(actual_cells) else ""
                tokens = list(numbers(actual))
                if name == "auroc" and len(tokens) == 2:
                    # An omitted superscript displays a zero constant-fit count.
                    tokens.append(("", Decimal(0), "0 (implicit)"))
                audit.sequence(f"{location}/{column}", tokens, expected)
        for key in actual_rows:
            audit.record(f"tab:{name}/{key}", str(actual_rows[key]), "", "unexpected row", "mismatch")


def check_prose(audit, tex, results):
    for anchor, expected in prose_claims(results):
        lines = [(index, line) for index, line in enumerate(tex.splitlines(), 1)
                 if re.match(anchor, line)]
        if len(lines) != 1:
            audit.record(f"prose/{anchor}", str(len(lines)), "1", "paragraph anchor", "mismatch")
            continue
        index, line = lines[0]
        audit.sequence(f"tex:{index}", list(numbers(line, include_words=True)), expected)


def check_figures(audit, tex, regenerated, reference_dir):
    audit.sequence("fig:paired/caption", list(numbers(caption(tex, "fig:paired"), include_words=True)), (
        Number(100), Number(0), Number(4), Number(2), Number(3), Number(1),
        Number(1), Number(2), Number(0, 2, relation="+"), Number(.005, 3),
    ))
    audit.sequence("fig:factors/caption", list(numbers(caption(tex, "fig:factors"), include_words=True)),
                   (Number(4), Number(60), Number(4), Number(2), Number(8), Number(3)))
    # The manuscript contains includegraphics links. The comparison below uses the numeric archive.
    for figure in ("paired", "factors"):
        audit.record(f"fig:{figure}/manuscript rendering", "", "",
                     "rendered figure values are external to the manuscript source", "unverified")

    if reference_dir is None:
        audit.record("fig:paired/labels", "", "90 labels", "provide archived descriptive figure data",
                     "unverified")
        audit.record("fig:factors/shares", "", "42 shares and 6 totals", "provide archived descriptive figure data",
                     "unverified")
        return

    paired_ref = {(r["dataset"], r["contrast"], r["fold"], r["seed"]): r for r in
                  read_csv(reference_dir / "paired-cells.csv")
                  if r["metric"] == "auroc" and r["contrast"] in FAMILIES}
    shares_ref = {(r["dataset"], r["analysis"], r["term"]): r for r in
                  read_csv(reference_dir / "factor-shares.csv")
                  if r["metric"] == "auroc" and r["analysis"] in ("matched_treatment_absolute", "four_dose_paired")}
    if len(paired_ref) != 90 or len(shares_ref) != 42:
        raise ValueError("expected archived figure data with 90 cells and 42 shares")
    paired = read_csv(regenerated / "paired-effects-labels.csv")
    shares = read_csv(regenerated / "factor-shares.csv")
    if len(paired) != 90 or len(shares) != 42:
        raise ValueError("expected regenerated figure data with 90 cells and 42 shares")
    for cell in paired:
        key = tuple(cell[field] for field in ("dataset", "family", "fold", "seed"))
        old = paired_ref.pop(key)
        value = 100 * float(old["delta"])
        digits = 2 if key[0] == "rsna2018" else 1
        expected = f"{0:.{digits}f}" if value == 0 or (digits == 1 and abs(value) < .05) else f"{value:+.{digits}f}"
        location = "fig:paired/" + "/".join(key)
        audit.figure(location, expected, cell["label"])
        if not math.isclose(value, float(cell["auroc_points"]), rel_tol=1e-11, abs_tol=1e-12):
            audit.figure(f"{location}/full_precision", str(value), cell["auroc_points"])
        flag = str(old["constituent_collapse"] == "1")
        if flag != cell["constituent_constant"]:
            audit.figure(f"{location}/constant_score", flag, cell["constituent_constant"])
    for row in shares:
        mode = "matched_treatment_absolute" if row["analysis"] == "treated" else "four_dose_paired"
        term = row["term"].replace("policy", "dose").replace("setting", "seed")
        old = shares_ref.pop((row["dataset"], mode, term))
        key = f"fig:factors/{row['dataset']}/{row['analysis']}"
        audit.figure(f"{key}/{row['term']}", f'{float(old["share_pct"]):.1f}', row["label"])
        if row["term"] == "policy":
            digits = 6 if row["dataset"] == "rsna2018" else 5
            audit.figure(f"{key}/total", f'{float(old["total_centered_ss"]):.{digits}f}',
                         f'{float(row["total"]):.{digits}f}')
        if not math.isclose(float(old["ss"]), float(row["ss"]), rel_tol=1e-11, abs_tol=1e-12):
            audit.figure(f"{key}/{row['term']}/full_precision_ss", old["ss"], row["ss"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--generated", type=Path, default=Path("build/study-results"))
    parser.add_argument("--reference-analysis", type=Path,
                        help="independent archive containing paired-cells.csv and factor-shares.csv")
    parser.add_argument("--report", type=Path, help="write a CSV record for every comparison")
    args = parser.parse_args()
    tex = args.manuscript.read_text(encoding="utf-8")
    results = json.loads((args.generated / "results.json").read_text(encoding="utf-8"))
    audit = Audit()
    check_tables(audit, tex, results)
    check_prose(audit, tex, results)
    check_figures(audit, tex, args.generated, args.reference_analysis)
    if args.report:
        write_csv(args.report, audit.rows)
    counts = {status: sum(row["status"] == status for row in audit.rows)
              for status in ("match", "mismatch", "unverified", "out of scope")}
    print(", ".join(f"{count} {status}" for status, count in counts.items()))
    print(f'{sum(row["source"] == "from archived SAP outputs" for row in audit.rows)} values from archived SAP outputs')
    for row in audit.rows:
        if row["status"] in ("mismatch", "unverified"):
            print(f'{row["status"]}: {row["location"]}: observed {row["manuscript"]!r}, '
                  f'regenerated {row["regenerated"]!r} ({row["source"]})')
    return 1 if counts["mismatch"] else 2 if counts["unverified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
