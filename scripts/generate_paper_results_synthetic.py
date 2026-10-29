"""Materialize native SYNTHETIC paper-report fixtures; never opens images.

Uses the same fixture factory as test_paper_results and test_analysis_sap_v1.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from augment_atlas.paper_results import _check_sap_seal
from test_paper_results import write_studies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Fixtures must be generated against the public sealed SAP implementation.
    try:
        _check_sap_seal()
    except ValueError as error:
        parser.error(str(error))
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("fixture directory must be new or empty (refuse mixing campaigns)")
    for dataset, path in write_studies(args.out).items():
        print(f"--campaign {dataset}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
