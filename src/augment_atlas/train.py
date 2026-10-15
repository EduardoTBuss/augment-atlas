"""ResNet-50 classification smoke runner. ML imports occur only in run()."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time

from .contracts import atomic_json, canonical_hash, file_hash, load_config
from .data import confined_path, data_root, load_records, prepare_splits


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClassificationDataset:
    """Map-style dataset without requiring torch merely to import this file."""
    def __init__(self, cfg, records, epoch=0, training=False):
        self.cfg, self.records, self.epoch, self.training = cfg, records, epoch, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image
        from .augment import apply_augmentation
        row = self.records[index]
        cfg = self.cfg
        if cfg["dataset"]["adapter"] == "synthetic":
            rng = np.random.default_rng(int(canonical_hash([row["id"], cfg["dataset"]["version"]])[:16], 16))
            side = cfg["input"]["size"]
            a = rng.integers(0, 256, size=(side, side), dtype=np.uint8)
            image = Image.fromarray(a)
            if cfg["input"]["color"] == "rgb":
                image = image.convert("RGB")
        else:
            with Image.open(confined_path(data_root(cfg), row["path"])) as source:
                image = source.convert("L" if cfg["input"]["color"] == "grayscale_to_rgb" else "RGB")
        if self.training:
            rng = random.Random(int(canonical_hash([cfg["training"]["seed"], self.epoch, row["id"]])[:16], 16))
            image = apply_augmentation(image, cfg["augmentation"]["ops"], rng)
        image = image.resize((cfg["input"]["size"],) * 2, Image.Resampling.BILINEAR).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std, row["label"]


def _atomic_torch(torch, path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _rng_state(torch, np):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(torch, np, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def run(cfg: dict, output: Path, resume: str | None = None, stop_after_epoch: int | None = None) -> int:
    # Must precede importing torch or initializing CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import torchvision
    from torchvision.models import ResNet50_Weights, resnet50

    output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        raise ValueError("attempt output is already complete; use a separate attempt directory")
    if (output / "checkpoint.pt").exists() and not resume:
        raise ValueError("existing checkpoint requires explicit --resume")
    records = load_records(cfg)
    splits = prepare_splits(cfg, records, output)
    config_hash = canonical_hash(cfg)
    if (output / "config.json").exists() and canonical_hash(json.loads((output / "config.json").read_text(encoding="utf-8"))) != config_hash:
        raise ValueError("attempt directory already belongs to a different config")
    atomic_json(output / "config.json", cfg)
    training = cfg["training"]
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if training["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA is unavailable")
    device = torch.device(training["device"])
    model = resnet50(weights=None)
    weights_file = None
    if cfg["model"]["weights"] != "none":
        # Deliberately never call the download-capable torchvision constructor.
        filename = Path(ResNet50_Weights.IMAGENET1K_V2.url).name
        weights_file = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not weights_file.is_file():
            raise RuntimeError("ImageNet weights missing from local torch cache; downloads are disabled")
        model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Linear(model.fc.in_features, cfg["model"]["num_classes"])
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"])
    criterion = torch.nn.CrossEntropyLoss()
    environment = {"schema_version": 1, "utc": utc_now(), "python": platform.python_version(),
                   "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
                   "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "device": str(device), "precision": "float32", "autocast": False, "tf32": False,
                   "deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                   "batch_size": training["batch_size"], "accumulation_steps": training["accumulation_steps"],
                   "nominal_effective_batch_size": training["batch_size"] * training["accumulation_steps"],
                   "partial_final_batch": "sample-weighted gradient mean within each accumulation window",
                   "freeze_batchnorm": training.get("freeze_batchnorm", True),
                   "weights": cfg["model"]["weights"], "weights_sha256": file_hash(weights_file) if weights_file else None,
                   "code_commit": os.environ.get("ATLAS_CODE_COMMIT", "unavailable"),
                   "image_id": os.environ.get("ATLAS_IMAGE_ID", "unavailable"),
                   "config_sha256": config_hash, "splits_sha256": splits["sha256"]}
    environment["dataset_integrity"] = {"manifest_records_sha256": splits["manifest_sha256"],
                                         "image_hashes_present": sum("sha256" in row for row in records),
                                         "images_total": len(records),
                                         "synthetic": cfg["dataset"]["adapter"] == "synthetic"}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment["gpu"] = {"name": properties.name, "compute_capability": [properties.major, properties.minor],
                              "total_memory_bytes": properties.total_memory,
                              "uuid": str(getattr(properties, "uuid", "unavailable"))}
    atomic_json(output / "environment.json", environment)
    completed_epoch, best_loss, best_epoch, samples_processed = 0, math.inf, 0, 0
    history = []
    best_model = None
    if resume:
        # Only private, trusted checkpoints from this runner may be loaded.
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("schema_version") != 1 or state.get("config_sha256") != config_hash or state.get("splits_sha256") != splits["sha256"]:
            raise ValueError("checkpoint config/split compatibility mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # torch optimizer moves loaded tensors to their parameter devices.
        scheduler.load_state_dict(state["scheduler"])
        completed_epoch, best_loss, best_epoch = state["completed_epoch"], state["best_loss"], state["best_epoch"]
        samples_processed = state["samples_processed"]
        history = state.get("metrics", [])
        best_model = state.get("best_model")
        if completed_epoch and best_model is None:
            raise ValueError("checkpoint is missing the selected model")
        if len(history) != completed_epoch:
            raise ValueError("checkpoint epoch journal is inconsistent")
        _restore_rng(torch, np, state["rng"])
    interrupted = {"signal": None}
    def on_signal(number, _frame):
        interrupted["signal"] = number
    previous = {number: signal.signal(number, on_signal) for number in (signal.SIGTERM, signal.SIGINT)}
    lookup = {row["id"]: row for row in records}
    partition = {name: [lookup[i] for i in ids] for name, ids in splits["lists"].items()}
    started = time.monotonic()

    def save_checkpoint():
        _atomic_torch(torch, output / "checkpoint.pt", {
            "schema_version": 1, "config_sha256": config_hash, "splits_sha256": splits["sha256"],
            "completed_epoch": completed_epoch, "best_loss": best_loss, "best_epoch": best_epoch,
            "best_model": best_model,
            "samples_processed": samples_processed, "metrics": history, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": _rng_state(torch, np)})

    def progress(status):
        atomic_json(output / "progress.json", {"schema_version": 1, "status": status, "utc": utc_now(),
                    "epoch": completed_epoch, "epochs": training["epochs"], "samples_processed": samples_processed,
                    "elapsed_seconds": time.monotonic() - started, "config_sha256": config_hash,
                    "splits_sha256": splits["sha256"], "checkpoint": "checkpoint.pt",
                    "interruption_signal": interrupted["signal"]})

    def evaluate(name):
        model.eval()
        loader = torch.utils.data.DataLoader(ClassificationDataset(cfg, partition[name]), batch_size=training["batch_size"],
                                            shuffle=False, num_workers=training["workers"])
        loss_sum, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = model(images)
                    loss = criterion(logits, labels)
                loss_sum += loss.item() * len(labels)
                correct += int((logits.argmax(1) == labels).sum().item())
                count += len(labels)
        return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}

    try:
        if not resume:
            save_checkpoint()
        progress("running")
        # The epoch journal is committed inside the checkpoint and mirrored in
        # metrics.json; this survives a crash between independent file replaces.
        for epoch in range(completed_epoch + 1, training["epochs"] + 1):
            model.train()
            if training.get("freeze_batchnorm", True):
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            order = list(range(len(partition["train"])))
            random.Random(int(canonical_hash([seed, epoch, "order"])[:16], 16)).shuffle(order)
            batches = [order[i:i + training["batch_size"]] for i in range(0, len(order), training["batch_size"])]
            loader = torch.utils.data.DataLoader(ClassificationDataset(cfg, partition["train"], epoch, True),
                                                batch_sampler=batches, num_workers=training["workers"])
            optimizer.zero_grad(set_to_none=True)
            loss_sum, count = 0.0, 0
            accumulation = training["accumulation_steps"]
            for index, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                window_start = (index // accumulation) * accumulation
                window_samples = sum(len(batch) for batch in batches[window_start:window_start + accumulation])
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite training loss")
                (loss * len(labels) / window_samples).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(batches):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                loss_sum += loss.item() * len(labels)
                count += len(labels)
            samples_processed += count
            selection = evaluate("selection")
            if not math.isfinite(selection["loss"]):
                raise RuntimeError("non-finite selection loss")
            if selection["loss"] < best_loss:
                best_loss, best_epoch = selection["loss"], epoch
                best_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            scheduler.step()
            completed_epoch = epoch
            row = {"epoch": epoch, "utc": utc_now(), "train_loss": loss_sum / count, "selection": selection,
                   "samples_processed": samples_processed, "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            # The checkpoint commits the epoch; the public progress follows it.
            save_checkpoint()
            atomic_json(output / "metrics.json", history)
            progress("running")
            if interrupted["signal"] is not None or (stop_after_epoch is not None and epoch >= stop_after_epoch):
                progress("interrupted")
                return 128 + interrupted["signal"] if interrupted["signal"] else 75
        # Only selection chooses the epoch. External test is touched once here.
        if best_model is None:
            raise RuntimeError("no selected model was produced")
        model.load_state_dict(best_model)
        _atomic_torch(torch, output / "best.pt", {"model": best_model, "epoch": best_epoch,
                                                "selection_loss": best_loss, "config_sha256": config_hash})
        results = {"schema_version": 1, "phase": "smoke", "scientific_status": "provisional",
                   "best_epoch": best_epoch, "selection_loss": best_loss, "test": evaluate("test"),
                   "config_sha256": config_hash, "splits_sha256": splits["sha256"],
                   "inbreast_previously_examined_in_pilot": cfg["dataset"]["adapter"] == "inbreast_csv"}
        if interrupted["signal"] is not None:
            progress("interrupted")
            return 128 + interrupted["signal"]
        atomic_json(output / "results.json", results)
        atomic_json(output / "metrics.json", history)
        progress("completed")
        required = ["checkpoint.pt", "best.pt", "splits.json", "config.json", "environment.json", "metrics.json", "results.json", "progress.json"]
        artifacts = {name: file_hash(output / name) for name in required}
        sizes = {name: (output / name).stat().st_size for name in required}
        if any(size <= 0 for size in sizes.values()):
            raise RuntimeError("empty required artifact")
        atomic_json(output / "completed.json", {"schema_version": 1, "status": "completed", "utc": utc_now(),
                    "epoch": completed_epoch, "config_sha256": config_hash, "splits_sha256": splits["sha256"],
                    "artifacts": artifacts, "artifact_sizes": sizes})
        return 0
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--resume-if-exists", action="store_true", help="resume this attempt's checkpoint if present")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--preflight", action="store_true", help="validate contracts, manifest and persisted splits without importing torch")
    args = parser.parse_args()
    output = Path(args.output)
    try:
        cfg = load_config(args.config)
        if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
            raise ValueError("--stop-after-epoch must be positive")
        if args.preflight:
            records = load_records(cfg)
            splits = prepare_splits(cfg, records, output)
            print(json.dumps({"valid": True, "config_sha256": canonical_hash(cfg), "splits_sha256": splits["sha256"],
                              "counts": {name: len(ids) for name, ids in splits["lists"].items()}}))
            return 0
        resume = args.resume
        if args.resume_if_exists and resume is None and (output / "checkpoint.pt").is_file():
            resume = str(output / "checkpoint.pt")
        return run(cfg, output, resume, args.stop_after_epoch)
    except Exception as exc:
        atomic_json(output / "failed.json", {"utc": utc_now(), "status": "failed", "error_type": type(exc).__name__, "message": str(exc)})
        print(f"training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
