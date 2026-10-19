"""Real CPU toy training: legacy (8 artifacts) vs clinical opt-in (9 artifacts).

Runs the actual trainer CLI (``python -m augment_atlas.train``) with a real
random-initialised ResNet-50 on the synthetic adapter (hash-generated images,
no data files, no downloads), then validates each attempt through the host
runtime entry ``validate_completion`` using trusted host inputs. Prints one JSON
summary to stdout and exits nonzero if any check fails. Synthetic only; the
metric values are meaningless and carry no clinical or scientific claim.

Intended for the pinned Linux container; see scripts/toy-train-cpu-clinical.sh.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))

from augment_atlas import runtime  # noqa: E402
from augment_atlas.contracts import atomic_json, canonical_hash, load_config, recipe_identity  # noqa: E402
from augment_atlas.data import load_records, prepare_splits  # noqa: E402
from augment_atlas.storage import digest  # noqa: E402

BINARY = dict(schema_version=1, task="binary", classes=[0, 1], score_kind="softmax_probability",
              frozen=True, positive_class=1, threshold=0.5, comparison="ge")


def run_suite() -> dict:
    """Full unittest discovery with the container CPU gate enabled."""
    env = dict(os.environ, ATLAS_CONTAINER_CPU_TESTS="1", PYTHONPATH=str(CODE / "src"))
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                          cwd=CODE, env=env, capture_output=True, text=True)
    text = proc.stderr
    ran = re.search(r"^Ran (\d+) tests? in", text, re.M)
    status = re.search(r"^(OK|FAILED)(?: \((.*)\))?$", text, re.M)
    details = dict(item.split("=") for item in status.group(2).split(", ")) if status and status.group(2) else {}
    return {"returncode": proc.returncode, "ran": int(ran.group(1)) if ran else None,
            "status": status.group(1) if status else "unparsed",
            "skipped": int(details.get("skipped", 0)), "failures": int(details.get("failures", 0)),
            "errors": int(details.get("errors", 0)), "seconds": round(time.monotonic() - started, 3),
            "tail": text.strip().splitlines()[-3:] if proc.returncode else []}


def toy_config(clinical: bool) -> dict:
    cfg = load_config(CODE / "configs/smoke/synthetic.json")
    cfg["training"].update(device="cpu", workers=0, epochs=2)
    if clinical:
        cfg["evaluation"] = dict(BINARY)
    return cfg


def code_commit() -> str:
    if os.environ.get("ATLAS_CODE_COMMIT"):
        return os.environ["ATLAS_CODE_COMMIT"]
    try:
        return subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "toy-uncommitted-worktree"


def train_and_validate(name: str, clinical: bool, work: Path, commit: str) -> dict:
    cfg = toy_config(clinical)
    relative = f"configs/smoke/toy-{name}.json"
    code_dir, data_dir, out = work / "code", work / "data", work / "runs" / name
    atomic_json(code_dir / relative, cfg)
    data_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(CODE / "src"), ATLAS_CODE_COMMIT=commit,
               ATLAS_IMAGE_ID=os.environ.get("ATLAS_IMAGE_ID", "unavailable"),
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env.pop("ATLAS_DATA_ROOT", None)
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-m", "augment_atlas.train", "--config", str(code_dir / relative),
                           "--output", str(out)], cwd=CODE, env=env, capture_output=True, text=True)
    report = {"name": name, "clinical_opt_in": clinical, "train_exit_code": proc.returncode,
              "train_seconds": round(time.monotonic() - started, 3)}
    if proc.returncode:
        report["train_stderr_tail"] = proc.stderr.strip().splitlines()[-5:]
        report["passed"] = False
        return report
    # Spec built like ``cli seed`` from host-trusted inputs, never from the attempt output.
    records = load_records(cfg)
    with tempfile.TemporaryDirectory(prefix="atlas-toy-split-") as directory:
        splits = prepare_splits(cfg, records, directory)
    spec = {"config": relative, "config_sha256": digest(code_dir / relative), "split_hash": splits["sha256"],
            "recipe_hash": recipe_identity(cfg), "train_seed": cfg["training"]["seed"],
            "arm": cfg["augmentation"]["arm"], "code_commit": commit}
    settings = {"code": str(code_dir), "data": str(data_dir)}
    completed = json.loads((out / "completed.json").read_text(encoding="utf-8"))
    report["completed_artifacts"] = sorted(completed["artifacts"])
    try:
        summary = runtime.validate_completion(settings, out, spec)
        report["runtime_validation"] = "accepted"
        report["artifact_count"] = summary["artifact_count"]
    except ValueError as exc:
        report["runtime_validation"] = f"rejected: {exc}"
        report["passed"] = False
        return report
    expected = 9 if clinical else 8
    checks = {"artifact_count": report["artifact_count"] == expected,
              "completed_lists_expected": len(completed["artifacts"]) == expected,
              "predictions_presence_matches_opt_in": (out / "predictions.json").exists() == clinical}
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    report["best_epoch"] = results["best_epoch"]
    if clinical:
        try:
            runtime.validate_artifacts(out, spec)
            checks["fails_closed_without_trusted_inputs"] = False
        except ValueError:
            checks["fails_closed_without_trusted_inputs"] = True
        result = results["clinical_evaluation"]
        primary = result["metrics"][result["primary_metric"]]
        report["clinical"] = {"primary_metric": result["primary_metric"], "value": primary["value"],
                              "reason": primary["reason"], "samples": result["samples"],
                              "partition": result["partition"],
                              "scientifically_eligible": result["scientifically_eligible"],
                              "execution_verified": result["execution_verified"],
                              "predictions_content_sha256": result["predictions_content_sha256"],
                              "environment_sha256": result["environment_sha256"],
                              "recipe_sha256": result["recipe_sha256"]}
        checks["not_eligible"] = result["scientifically_eligible"] is False and result["execution_verified"] is False
    else:
        checks["no_clinical_result"] = "clinical_evaluation" not in results
    report["checks"] = checks
    report["completed_sha256"] = digest(out / "completed.json")
    report["passed"] = all(checks.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="only run the two toy trainings")
    parser.add_argument("--work", help="scratch directory (default: a new temporary directory)")
    args = parser.parse_args()
    import torch
    import torchvision
    summary = {"schema_version": 1, "kind": "toy-train-cpu-clinical", "synthetic_only": True,
               "python": platform.python_version(), "torch": torch.__version__,
               "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
               "image_id": os.environ.get("ATLAS_IMAGE_ID", "unavailable")}
    summary["code_commit"] = commit = code_commit()
    if not args.skip_tests:
        summary["test_suite"] = run_suite()
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="atlas-toy-clinical-"))
    summary["runs"] = [train_and_validate("legacy", False, work, commit),
                       train_and_validate("clinical", True, work, commit)]
    runs_passed = all(run["passed"] for run in summary["runs"])
    tests_passed = args.skip_tests or summary["test_suite"]["returncode"] == 0
    summary["passed"] = bool(runs_passed and tests_passed)
    summary["summary_sha256"] = canonical_hash({k: v for k, v in summary.items() if k != "summary_sha256"})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
