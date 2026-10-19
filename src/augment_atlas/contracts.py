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
        "basic_clahe_light", "basic_clahe_heavy", "gamma_light", "gamma_heavy",
        "clahe_light", "clahe_heavy")
CANDIDATE_BACKEND = "candidate-opencv-v2"
LEGACY_BACKEND = "legacy-pilot-v1"
BASE_REQUIRED_ARTIFACTS = ("checkpoint.pt", "best.pt", "splits.json", "config.json",
                           "environment.json", "metrics.json", "results.json", "progress.json")


def required_artifacts(cfg: dict) -> tuple[str, ...]:
    """Legacy completion stays at eight files; clinical opt-in adds private rows."""
    return BASE_REQUIRED_ARTIFACTS + (("predictions.json",) if "evaluation" in cfg else ())


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def recipe_identity(cfg: dict) -> str:
    """Frozen technical recipe identity, shared by every arm and training seed.

    The optional clinical evaluation decision is included only when the
    explicit ``evaluation`` key exists, so legacy recipe hashes are unchanged.
    """
    train = {k: v for k, v in cfg["training"].items() if k != "seed"}
    identity = dict(recipe=cfg["recipe"], model=cfg["model"], input=cfg["input"], training=train)
    if "evaluation" in cfg:
        identity["evaluation"] = cfg["evaluation"]
    return canonical_hash(identity)


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
    if "evaluation" in cfg:
        from .evaluation import validate_evaluation_config
        validate_evaluation_config(cfg["evaluation"], model["num_classes"])
    backend = cfg["augmentation"].get("backend", LEGACY_BACKEND)
    if backend not in (LEGACY_BACKEND, CANDIDATE_BACKEND):
        raise ValueError("unknown augmentation backend")
    candidate = backend == CANDIDATE_BACKEND
    if candidate != (cfg["protocol_version"] == "candidate-v2"):
        raise ValueError("candidate-v2 protocol and candidate-opencv-v2 backend must agree")
    if candidate and model["weights"] == "IMAGENET1K_V2":
        weight_hash = model.get("weights_sha256", "")
        if not isinstance(weight_hash, str) or len(weight_hash) != 64 or any(char not in "0123456789abcdef" for char in weight_hash):
            raise ValueError("candidate pretrained weights require a lowercase 64-hex weights_sha256")
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
    if split.get("version") == "stratified-group-outer-v1":
        if not isinstance(dataset.get("split_file"), str) or not dataset["split_file"].strip():
            raise ValueError("stratified-group-outer-v1 requires an explicit persisted split_file")
    elif split.get("version") != "group-hash-v1":
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
    expected = ["basic"] if aug["arm"] == "basic" or aug["arm"].startswith("basic_") else []
    if "gamma" in aug["arm"]:
        expected.append("gamma")
    if "clahe" in aug["arm"]:
        expected.append("clahe")
    if not isinstance(ops, list) or [op.get("op") for op in ops] != expected:
        raise ValueError("augmentation operations/order must match the declared arm")
    evaluation_scope = training.get("evaluation_scope", "test")
    if evaluation_scope not in ("test", "selection-only"):
        raise ValueError("training.evaluation_scope must be test or selection-only")
    if evaluation_scope == "selection-only" and (aug["arm"] != "none" or ops or hpo["protocol"] != "within-outer-fold"):
        raise ValueError("selection-only requires no augmentation and within-outer-fold HPO protocol")
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
                if candidate:
                    expected_range = [0.8, 1.2] if aug["arm"].endswith("light") else [0.67, 1.5]
                    if op["range"] != expected_range or op["p"] != 0.3:
                        raise ValueError("candidate-v2 gamma must retain its versioned provisional range and p=0.3")
            else:
                if candidate:
                    expected_clip = 2.0 if aug["arm"].endswith("light") else 4.0
                    grid = op.get("tile_grid_size")
                    if op.get("clip_limit") != expected_clip or grid != [8, 8] or any(type(value) is not int for value in grid) or op["p"] != 0.3 or "window_fraction" in op:
                        raise ValueError("candidate CLAHE requires OpenCV clip 2/4, tile_grid_size=[8,8], p=0.3; no legacy window_fraction")
                else:
                    if "tile_grid_size" in op:
                        raise ValueError("legacy CLAHE does not accept OpenCV tile_grid_size")
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
