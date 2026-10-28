"""Synthetic ISIC-shape performance evidence. No real data; no network.

Run from repository root with PYTHONPATH=src:
python scripts/benchmark_analysis_sap_v1.py --out outputs/sap-v1-synthetic
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
import matplotlib

from augment_atlas import analysis_sap_v1 as sap
from augment_atlas.contracts import ARMS


def git(*args):
    run = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return {"command": ["git", *args], "exit_status": run.returncode, "stdout": run.stdout.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--B", nargs="+", type=int, default=[128, 256])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "benchmark.json"
    manifest = {"project": "AugmentAtlas", "experiment": "C5-sap-v1-synthetic-isic-shape",
                "task_id": "20261028-162617-software-implementation-engineer",
                "commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "dirty_worktree": git("status", "--porcelain"),
                "command": [sys.executable, *sys.argv], "working_directory": str(Path.cwd()),
                "started_utc": datetime.now(timezone.utc).isoformat(), "ended_utc": None, "exit_status": None,
                "environment": {"os": platform.platform(), "architecture": platform.machine(),
                                "cpu": platform.processor(), "python": sys.version,
                                "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "dataset": {"name": "synthetic_isic_shape", "version": 1, "source": "generated_in_memory",
                            "images": 26000, "folds": 5, "arms": 10, "training_seeds": [1],
                            "prediction_rows": 260000, "groups": 13000, "images_per_group": 2,
                            "positive_group_fraction": .02, "preprocessing": "none; independent group/image normals"},
                "parameters": {"generator_seed": 20261028, "bootstrap_seed": 7331,
                               "B_measured": args.B, "target_B": 10000, "block_size": 64,
                               "warmup_B": 16, "repetitions_per_B": 1, "timeout": "external task budget"},
                "metric": "elapsed seconds for paired weighted AUROC+AP across all 10 arms, macro 5 folds",
                "timing_boundary": "bootstrap only; input generation, preparation, pooled points, plotting excluded",
                "aggregation": "same group draws across arms; five-fold macro mean",
                "source_sha256": sap._hash(sap.__file__), "benchmark_source_sha256": sap._hash(__file__),
                "runs": [], "limitations": ["Synthetic only; one timing repetition per B, fixed CPU/environment.",
                                             "Extrapolation is linear in B; excludes I/O and plotting.",
                                             "This machine is not the target notebook; hardware transfer is unverified."]}
    def save():
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()  # Planned matrix is durable before work starts.
    try:
        start = time.perf_counter()
        rng = np.random.default_rng(20261028)
        rows = []
        for fold in sap.FOLDS:
            for group in range(2600):
                label = int(group % 50 == 0)
                shared = rng.normal()
                for image in range(2):
                    z = .7 * shared + .7 * rng.normal()
                    for j, arm in enumerate(ARMS):
                        score = float(1 / (1 + np.exp(-(z + (1. + .03 * j) * label + .1 * rng.normal()))))
                        rows.append(dict(dataset="synthetic", arm=arm, fold=fold, seed=1,
                                         image=f"{fold}-{group}-{image}", group=f"{fold}-{group}", label=label, score=score))
        manifest["generation_seconds"] = time.perf_counter() - start
        manifest["dataset"]["generated_rows_sha256"] = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        start = time.perf_counter()
        _, folds, points, plans = sap._prepare(rows, "synthetic")
        if len(plans) != 50:
            raise RuntimeError("expected 50 evaluable fold-arm cells")
        manifest["preparation_seconds"] = time.perf_counter() - start
        sap._bootstrap(folds, plans, (1,), B=16, seed=7331, block_size=64)
        for B in args.B:
            start = time.perf_counter()
            boot = sap._bootstrap(folds, plans, (1,), B=B, seed=7331, block_size=64)
            elapsed = time.perf_counter() - start
            raw = np.stack([boot[(arm, 1)] for arm in ARMS])
            raw_path = args.out / f"bootstrap_B{B}.npy"
            np.save(raw_path, raw, allow_pickle=False)
            manifest["runs"].append({"B": B, "seconds": elapsed, "estimated_seconds_B10000": elapsed * 10000 / B,
                                     "raw_output": raw_path.name, "sha256": sap._hash(raw_path),
                                     "shape": list(raw.shape), "arm_order": list(ARMS), "metric_order": ["auroc", "ap"],
                                     "finite": bool(np.all(np.isfinite(raw))), "status": "completed"})
            save()
            print(json.dumps(manifest["runs"][-1]), flush=True)
        manifest["estimated_seconds_B10000_range"] = [min(r["estimated_seconds_B10000"] for r in manifest["runs"]),
                                                       max(r["estimated_seconds_B10000"] for r in manifest["runs"])]
        manifest["working_array_memory_note"] = "Block arrays scale with 64 x 5200 images, not 10000 x 26000 x 10; Python input memory also matters. Peak RSS unmeasured."
        manifest["exit_status"] = 0
    except Exception as error:
        manifest["exit_status"] = 1
        manifest["failure"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
