"""Versioned smoke contracts. This module deliberately has no ML dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

ARMS = ("none", "basic", "basic_gamma_light", "basic_gamma_heavy",
        "basic_clahe_light", "basic_clahe_heavy")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value: Any, name: str, minimum: float = 0, maximum: float = math.inf) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")


def validate_config(cfg: dict) -> dict:
    if cfg.get("schema_version") != 1 or cfg.get("phase") != "smoke" or cfg.get("scientific_status") != "provisional":
        raise ValueError("v1 accepts only explicit phase=smoke, scientific_status=provisional")
    for name in ("protocol_version",):
        if not isinstance(cfg.get(name), str) or not cfg[name]:
            raise ValueError(f"missing {name}")
    model, training = cfg["model"], cfg["training"]
    if model.get("name") != "resnet50" or model.get("weights") not in ("none", "IMAGENET1K_V2"):
        raise ValueError("only ResNet-50 with none or cached IMAGENET1K_V2 weights is supported")
    _integer(model["num_classes"], "num_classes", 2)
    if cfg["input"].get("color") not in ("grayscale_to_rgb", "rgb"):
        raise ValueError("input.color must explicitly describe channel semantics")
    _integer(cfg["input"]["size"], "input.size", 32)
    for name in ("seed", "workers"):
        _integer(training[name], name)
    for name in ("epochs", "batch_size", "accumulation_steps"):
        _integer(training[name], name, 1)
    _number(training["lr"], "lr", 1e-12)
    _number(training["weight_decay"], "weight_decay")
    if training.get("device") not in ("cuda", "cpu") or training.get("deterministic") is not True:
        raise ValueError("explicit cpu/cuda and deterministic=true are required")
    # Disable unimplemented knobs rather than silently pretending to support them.
    if training.get("autocast", False) or training.get("tf32", False):
        raise ValueError("FP32 smoke requires autocast=false and tf32=false")
    dataset, split = cfg["dataset"], cfg["splits"]
    if dataset.get("adapter") not in ("synthetic", "manifest_csv", "inbreast_csv"):
        raise ValueError("unsupported dataset adapter")
    for name in ("id", "version"):
        if not isinstance(dataset.get(name), str) or not dataset[name]:
            raise ValueError(f"dataset.{name} is required")
    if dataset["adapter"] != "synthetic" and not dataset.get("manifest"):
        raise ValueError("private dataset manifest is required")
    if dataset["adapter"] == "synthetic":
        _integer(dataset.get("synthetic_groups", 12), "synthetic_groups", 6)
    _integer(split["split_seed"], "split_seed")
    _integer(split["fold"], "fold")
    if split.get("version") != "group-hash-v1":
        raise ValueError("unknown split algorithm version")
    recipe = cfg["recipe"]
    if recipe.get("frozen") is not True or not recipe.get("id") or not recipe.get("version"):
        raise ValueError("an explicitly frozen, versioned technical recipe is required")
    hpo = recipe.get("hpo", {})
    if hpo.get("status") not in ("pending", "external") or hpo.get("protocol") not in ("undecided", "global-development", "within-outer-fold") or not hpo.get("origin"):
        raise ValueError("HPO origin/status/protocol must be recorded; no implicit final decision")
    if hpo["status"] == "external" and (hpo["protocol"] == "undecided" or not hpo.get("artifact_sha256")):
        raise ValueError("external HPO needs an explicit protocol and artifact hash")
    dependencies = recipe.get("dependencies")
    if not isinstance(dependencies, list) or any(not isinstance(item, str) or not item for item in dependencies):
        raise ValueError("recipe.dependencies must be a list of immutable stage identifiers")
    aug = cfg["augmentation"]
    if aug.get("arm") not in ARMS:
        raise ValueError("unknown augmentation arm")
    ops = aug.get("ops")
    expected = [] if aug["arm"] == "none" else ["basic"]
    if "gamma" in aug["arm"]:
        expected.append("gamma")
    if "clahe" in aug["arm"]:
        expected.append("clahe")
    if not isinstance(ops, list) or [op.get("op") for op in ops] != expected:
        raise ValueError("augmentation operations/order must match the declared arm")
    for op in ops:
        if op["op"] == "basic":
            for key in ("flip_p", "photometric_p"):
                _number(op[key], key, 0, 1)
            for key in ("brightness", "contrast"):
                values = op[key]
                if len(values) != 2:
                    raise ValueError(f"{key} requires two bounds")
                _number(values[0], key, 0.001)
                _number(values[1], key, values[0])
        else:
            _number(op["p"], "augmentation.p", 0, 1)
            if op["op"] == "gamma":
                if len(op["range"]) != 2:
                    raise ValueError("gamma.range requires two bounds")
                _number(op["range"][0], "gamma minimum", 0.001)
                _number(op["range"][1], "gamma maximum", op["range"][0])
            else:
                _number(op["clip_limit"], "CLAHE clip_limit", 1e-12, 1)
                _number(op["window_fraction"], "CLAHE window_fraction", 1e-12, 1)
                if cfg["input"]["color"] != "grayscale_to_rgb":
                    raise ValueError("v1 CLAHE is explicitly grayscale-only")
    return cfg


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(json.load(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps({"valid": True, "config_sha256": canonical_hash(cfg), "phase": cfg["phase"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
