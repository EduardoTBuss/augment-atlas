import csv
import json
from pathlib import Path
import subprocess
import sys


def test_study_regeneration_command(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "analysis.reproduce", "--out-dir", str(tmp_path)],
                   cwd=root, check=True, capture_output=True, text=True)
    summaries = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert sum(dataset["fits"] for dataset in summaries["datasets"].values()) == 450
    assert len(summaries["conditional_contrasts"]) == 180
    assert any(row["contrast"] == "S5_gamma" for row in summaries["conditional_contrasts"])

    with (tmp_path / "paired-effects-labels.csv").open(encoding="utf-8", newline="") as stream:
        cells = list(csv.DictReader(stream))
    assert len(cells) == 90
    assert any(row["dataset"] == "rsna2018" and row["label"] == "+0.00" for row in cells)
    for table in ("auroc", "planned", "grid", "dose", "repeatability"):
        assert (tmp_path / f"table-{table}.csv").stat().st_size > 0
    for stem in ("paired-effects", "factor-shares"):
        assert (tmp_path / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
