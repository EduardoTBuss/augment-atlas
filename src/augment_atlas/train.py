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

from .contracts import (GLOBAL_PHASES, atomic_json, canonical_hash, file_hash, load_config, required_artifacts,
                        validate_config)
from .data import confined_path, data_root, load_records, prepare_splits
from . import train_telemetry


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
        from .augment import apply_augmentation, apply_candidate_augmentation
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
        candidate = cfg['augmentation'].get('backend') == 'candidate-opencv-v2'
        if candidate:
            # Candidate v2 explicitly fixes resize BEFORE train-only augmentation.
            # Keep the legacy path unchanged for old configs/checkpoints.
            image = image.resize((cfg['input']['size'],) * 2, Image.Resampling.BILINEAR)
        if self.training and candidate:
            image = apply_candidate_augmentation(image, cfg['augmentation'],
                                                 [cfg['training']['seed'], self.epoch, row['id']])
        elif self.training:
            rng = random.Random(int(canonical_hash([cfg["training"]["seed"], self.epoch, row["id"]])[:16], 16))
            image = apply_augmentation(image, cfg["augmentation"]["ops"], rng)
        if not candidate:
            image = image.resize((cfg['input']['size'],) * 2, Image.Resampling.BILINEAR)
        image = image.convert('RGB')
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


def run(cfg: dict, output: Path, resume: str | None = None, stop_after_epoch: int | None = None, *,
        trainer_override: dict | None = None) -> int:
    validate_config(cfg)
    if cfg.get("phase") in GLOBAL_PHASES:
        return _run_global(cfg, Path(output), resume, stop_after_epoch, trainer_override=trainer_override)
    if trainer_override is not None:
        raise ValueError("trainer_override exists only for global hpo/final toy tests")
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
    evaluation_scope = training.get("evaluation_scope", "test")
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
        expected_weights_hash = cfg['model'].get('weights_sha256')
        if expected_weights_hash and file_hash(weights_file) != expected_weights_hash:
            raise ValueError('ImageNet cache checksum differs from the frozen model configuration')
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
    environment["evaluation_scope"] = evaluation_scope
    if "evaluation" in cfg:
        from .evaluation import clinical_environment_block
        environment["clinical_evaluation"] = clinical_environment_block(cfg)
    environment['augmentation_backend'] = cfg['augmentation'].get('backend', 'legacy-pilot-v1')
    environment['augmentation_resize_order'] = 'resize-before-augmentation' if cfg['augmentation'].get('backend') == 'candidate-opencv-v2' else 'augmentation-before-resize'
    if cfg['augmentation'].get('backend') == 'candidate-opencv-v2':
        import cv2
        from importlib.metadata import version
        environment['opencv'] = cv2.__version__
        environment['opencv_python_headless'] = version('opencv-python-headless')
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

    def evaluate(name, *, collect_scores=False):
        model.eval()
        loader = torch.utils.data.DataLoader(ClassificationDataset(cfg, partition[name]), batch_size=training["batch_size"],
                                            shuffle=False, num_workers=training["workers"])
        loss_sum, correct, count = 0.0, 0, 0
        scores = [] if collect_scores else None
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if collect_scores:
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError("non-finite clinical evaluation logits")
                    scores.extend(logits.softmax(dim=1).detach().cpu().tolist())
                loss_sum += loss.item() * len(labels)
                correct += int((logits.argmax(1) == labels).sum().item())
                count += len(labels)
        result = {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}
        return (result, scores) if collect_scores else result

    # Additive per-job telemetry (telemetry/, not a validated artifact); it
    # never touches RNG, batch order, loss, checkpoints or metrics.json.
    telemetry = train_telemetry.open_for_attempt(output, torch=torch, device=device, grad_clip_max_norm=1.0, context={
        "path": "legacy", "phase": cfg.get("phase", "smoke"), "resume": bool(resume),
        "start_epoch": completed_epoch + 1, "epochs": training["epochs"], "device": str(device),
        "batch_size": training["batch_size"], "accumulation_steps": training["accumulation_steps"],
        "train_images": len(partition["train"]),
        "batches_per_epoch": math.ceil(len(partition["train"]) / training["batch_size"]),
        "config_sha256": config_hash, "splits_sha256": splits["sha256"],
        "code_commit": environment["code_commit"], "image_id": environment["image_id"]})
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
            telemetry.epoch_start(epoch, batches=len(batches), images=len(order))
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
                grad_norm = None
                if (index + 1) % accumulation == 0 or index + 1 == len(batches):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                batch_loss = loss.item()
                loss_sum += batch_loss * len(labels)
                count += len(labels)
                telemetry.batch(epoch, index, len(labels), batch_loss, optimizer.param_groups[0]["lr"], grad_norm)
            telemetry.train_done(epoch)
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
            telemetry.epoch_end(epoch, row, selection_per_epoch=True)
            if interrupted["signal"] is not None or (stop_after_epoch is not None and epoch >= stop_after_epoch):
                telemetry.outcome("interrupted")
                progress("interrupted")
                return 128 + interrupted["signal"] if interrupted["signal"] else 75
        # Selection chooses the epoch. Technical HPO trials explicitly stop at
        # selection; the default evaluation retains the external-test contract.
        if best_model is None:
            raise RuntimeError("no selected model was produced")
        model.load_state_dict(best_model)
        _atomic_torch(torch, output / "best.pt", {"model": best_model, "epoch": best_epoch,
                                                "selection_loss": best_loss, "config_sha256": config_hash})
        results = {"schema_version": 1, "phase": "smoke", "scientific_status": "provisional",
                   "best_epoch": best_epoch, "selection_loss": best_loss,
                   "config_sha256": config_hash, "splits_sha256": splits["sha256"],
                   "inbreast_previously_examined_in_pilot": cfg["dataset"]["adapter"] == "inbreast_csv"}
        clinical_artifact = None
        if "evaluation" in cfg:
            # Opt-in: exactly one final pass over the scope's partition with the
            # already selected model; clinical metrics never influence best_epoch.
            from .evaluation import build_prediction_artifact, evaluation_partition
            final_partition = evaluation_partition(cfg)
            results[final_partition], scores = evaluate(final_partition, collect_scores=True)
            clinical_artifact, results["clinical_evaluation"] = build_prediction_artifact(
                cfg, splits, records, scores, best_epoch=best_epoch, environment=environment)
            if evaluation_scope == "selection-only":
                results["evaluation_scope"] = evaluation_scope
        elif evaluation_scope == "selection-only":
            results["evaluation_scope"] = evaluation_scope
            results["selection"] = evaluate("selection")
        else:
            results["test"] = evaluate("test")
        if interrupted["signal"] is not None:
            telemetry.outcome("interrupted")
            progress("interrupted")
            return 128 + interrupted["signal"]
        if clinical_artifact is not None:
            atomic_json(output / "predictions.json", clinical_artifact)
        atomic_json(output / "results.json", results)
        atomic_json(output / "metrics.json", history)
        progress("completed")
        required = required_artifacts(cfg)
        artifacts = {name: file_hash(output / name) for name in required}
        sizes = {name: (output / name).stat().st_size for name in required}
        if any(size <= 0 for size in sizes.values()):
            raise RuntimeError("empty required artifact")
        atomic_json(output / "completed.json", {"schema_version": 1, "status": "completed", "utc": utc_now(),
                    "epoch": completed_epoch, "config_sha256": config_hash, "splits_sha256": splits["sha256"],
                    "artifacts": artifacts, "artifact_sizes": sizes})
        telemetry.outcome("completed")
        return 0
    finally:
        telemetry.close()
        for number, handler in previous.items():
            signal.signal(number, handler)


# ---------------------------------------------------------------------------
# Global D/E path: phase hpo | final (docs/global-next-integration.md 7.1/7.3)
# ---------------------------------------------------------------------------
# D is never a test set; HPO sees only the D-view (D_fit/D_select, no E
# identity); the final job trains a fresh ImageNet model on D u (E \ E_f) for
# fixed epochs and reads E_f exactly once, after the last epoch. No HPO
# checkpoint can be loaded. Caveat (Bradshaw 2023, DOI 10.1148/ryai.220232):
# reshuffling folds does not remove selection bias if patients used to choose
# hyperparameters fall into a test set; that is why D stays out of every test.

TOY_OVERRIDE_GATE = "ATLAS_CONTAINER_CPU_TESTS"
_RECORD_KEYS = ("id", "path", "group", "label")


def _global_fail(code: str, message: str):
    raise ValueError(f"{code}: {message}")


def _toy_override(value):
    """Explicit CPU/0-worker override for toy tests; never a silent fallback.

    ``TRAINER_CONSTANTS_V1`` (device=cuda, workers=2) and the config are never
    changed. The override is a ``run()`` argument the CLI cannot pass, it needs
    the ``ATLAS_CONTAINER_CPU_TESTS=1`` gate, and it is recorded in the
    checkpoint lineage, ``environment.json``, ``results.json`` and
    ``completed.json`` so a host validator can refuse such attempts.
    """
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"device", "workers", "reason"} or value["device"] != "cpu"
            or type(value["workers"]) is not int or value["workers"] != 0
            or type(value["reason"]) is not str or not value["reason"].strip()):
        _global_fail("global_config_invalid", "trainer_override must be exactly {device: cpu, workers: 0, reason}")
    if os.environ.get(TOY_OVERRIDE_GATE) != "1":
        _global_fail("global_config_invalid", f"trainer_override requires the {TOY_OVERRIDE_GATE}=1 test gate")
    return dict(value)


def _global_augmentation(cfg: dict) -> dict:
    """TRAINER_CONSTANTS_V1 fixes resize BEFORE augmentation (candidate backend)."""
    from .contracts import CANDIDATE_BACKEND
    augmentation = cfg["augmentation"]
    backend = augmentation.get("backend")
    if backend not in (None, CANDIDATE_BACKEND) or (augmentation["ops"] and backend != CANDIDATE_BACKEND):
        _global_fail("recipe_adapter_unsupported",
                     f"global trainer v1 applies only {CANDIDATE_BACKEND} operations, after resize")
    return {"arm": augmentation["arm"], "ops": json.loads(json.dumps(augmentation["ops"])),
            "backend": CANDIDATE_BACKEND}


def _read_private_json(root: Path, relative: str, code: str):
    path = confined_path(root, relative)
    if not path.is_file():
        _global_fail(code, "a private global document is not available under the data root")
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError, RecursionError) as error:
        _global_fail(code, f"private global document cannot be parsed ({type(error).__name__})")


def _global_inputs(cfg: dict) -> dict:
    """Trusted, phase-specific inputs, all checked BEFORE torch is imported.

    hpo: only the hpo-view inside the mounted D-view; the pools and the
    manifest are not read because both list E identities.
    final: manifest records (image hashes), anchored pools, freeze bound to
    ``cfg.training`` and the re-derived ``global-final-view-v1`` document.
    """
    from . import global_development as gd
    from . import global_evaluation as ge
    from . import global_jobs as gj
    spec, bound = cfg["evaluation"], cfg["global"]
    if spec["classes"] != list(range(cfg["model"]["num_outputs"])):
        _global_fail("recipe_adapter_unsupported", "global trainer v1 requires classes 0..num_outputs-1")
    root = data_root(cfg)
    if cfg["phase"] == "hpo":
        view = _read_private_json(root, bound["hpo_view_file"], "view_hash_mismatch")
        records, train_ids, eval_ids = ge.check_hpo_view_local(view, cfg)
        if any(not confined_path(root, row["path"]).is_file() for row in records):
            _global_fail("records_identity_mismatch", "an hpo-view image is missing from the mounted D-view")
        pools = None
        integrity = {"source": "hpo_view", "records": len(records), "view_sha256": view["sha256"],
                     "view_files_sha256": view["view_files_sha256"],
                     "image_hashes": "D-view listing verified by the host runtime"}
    else:
        loaded = load_records({"dataset": dict(cfg["dataset"]), "model": {"num_classes": cfg["model"]["num_outputs"]}})
        records = sorted(({key: row[key] for key in _RECORD_KEYS} for row in loaded), key=lambda row: row["id"])
        pools = _read_private_json(root, bound["pools_file"], "pools_anchor_mismatch")
        if type(pools) is not dict or pools.get("records_sha256") != canonical_hash(records):
            _global_fail("records_identity_mismatch", "manifest records differ from pools.records_sha256")
        gj._anchored(records, pools, bound["pools_sha256"], global_task=True)
        freeze = _read_private_json(root, bound["freeze_file"], "freeze_not_sealed")
        try:
            if type(freeze) is not dict:
                raise ValueError("freeze must be an object")
            gd._checksum(freeze)
        except ValueError as error:
            _global_fail("freeze_not_sealed", str(error))
        if freeze["sha256"] != bound["freeze_sha256"] or freeze.get("pools_sha256") != bound["pools_sha256"]:
            _global_fail("freeze_not_sealed", "freeze document is not the anchored freeze of this config")
        recipe = freeze.get("frozen_recipe" if bound["freeze_kind"] == "hpo_d" else "recipe")
        gj.validate_global_config(cfg, recipe=recipe)
        view = gj.derive_final_view_doc(records, pools, bound["fold"], expected_pools_sha256=bound["pools_sha256"],
                                        freeze_sha256=bound["freeze_sha256"])
        eval_ids = ge.final_partition_ids(cfg, view, records, pools)
        train_ids = list(view["train"])
        integrity = {"source": "manifest", "manifest_records_sha256": canonical_hash(records),
                     "image_hashes_present": sum("sha256" in row for row in loaded), "images_total": len(loaded),
                     "pools_sha256": bound["pools_sha256"], "freeze_sha256": bound["freeze_sha256"]}
    lookup = {row["id"]: row for row in records}
    return {"view": view, "records": records, "pools": pools, "integrity": integrity,
            "train_records": [lookup[item] for item in train_ids], "eval_ids": list(eval_ids),
            "eval_records": [lookup[item] for item in eval_ids]}


def _state_sha256(state) -> str:
    """Byte-level identity of a state dict (names, dtypes, shapes, values)."""
    import hashlib
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        data = tensor.numpy().tobytes()
        for chunk in (header, data):
            digest.update(len(chunk).to_bytes(8, "big"))
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run_global(cfg: dict, output: Path, resume: str | None = None, stop_after_epoch: int | None = None, *,
                trainer_override: dict | None = None) -> int:
    from . import global_evaluation as ge
    from . import global_jobs as gj
    gj.validate_global_config(cfg)
    override = _toy_override(trainer_override)
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or stop_after_epoch < 1):
        raise ValueError("--stop-after-epoch must be positive")
    phase, training = cfg["phase"], cfg["training"]
    final = phase == "final"
    epochs, seed = training["epochs"], training["seed"]
    data_cfg = {"dataset": {key: cfg["dataset"][key] for key in ("id", "version", "adapter")},
                "input": dict(cfg["input"]), "augmentation": _global_augmentation(cfg), "training": {"seed": seed}}
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    if (output / "completed.json").exists():
        raise ValueError("attempt output is already complete; use a separate attempt directory")
    if checkpoint_path.exists() and not resume:
        raise ValueError("existing checkpoint requires explicit --resume")
    if resume is not None and (Path(resume).resolve() != checkpoint_path.resolve() or not checkpoint_path.is_file()):
        _global_fail("checkpoint_lineage_mismatch", "global jobs resume only this attempt's own checkpoint.pt")
    config_bytes = gj.render_config_bytes(cfg)
    config_hash = ge.config_sha256(cfg)
    if (output / "config.json").exists() and (output / "config.json").read_bytes() != config_bytes:
        raise ValueError("attempt directory already belongs to a different config")
    inputs = _global_inputs(cfg)
    view = inputs["view"]
    splits_hash = view["sha256"]
    if (output / "splits.json").exists():
        if canonical_hash(json.loads((output / "splits.json").read_text(encoding="utf-8"))) != canonical_hash(view):
            raise ValueError("attempt directory already belongs to a different split")

    # Must precede importing torch or initializing CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import torchvision
    from torchvision.models import ResNet50_Weights, resnet50

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
    device_name = override["device"] if override else training["device"]
    workers = override["workers"] if override else training["workers"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA is unavailable")
    device = torch.device(device_name)
    # Fresh model every job (no init_checkpoint exists in the closed schema).
    model = resnet50(weights=None)
    # Deliberately never call the download-capable torchvision constructor.
    weights_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(ResNet50_Weights.IMAGENET1K_V2.url).name
    if not weights_file.is_file():
        raise RuntimeError("ImageNet weights missing from local torch cache; downloads are disabled")
    weights_hash = file_hash(weights_file)
    if weights_hash != cfg["model"]["weights_sha256"]:
        _global_fail("recipe_mismatch", "ImageNet cache checksum differs from the frozen model configuration")
    model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
    # Re-seed so the new head depends only on the seed: arms are paired.
    torch.manual_seed(seed)
    model.fc = torch.nn.Linear(model.fc.in_features, cfg["model"]["num_outputs"])
    initial_state_sha256 = _state_sha256(model.state_dict())
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["optimizer"]["lr"],
                                  weight_decay=training["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                           eta_min=training["scheduler"]["eta_min"])
    criterion = torch.nn.CrossEntropyLoss()
    lineage = {"trainer": gj.TRAINER_CONSTANTS_V1["version"], "phase": phase, "config_sha256": config_hash,
               "splits_sha256": splits_hash, "initial_state_sha256": initial_state_sha256,
               "trainer_override": override}
    completed_epoch, samples_processed, history, evaluations = 0, 0, [], 0
    if resume:
        # Only this attempt's private checkpoint is ever loaded (path checked above).
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if type(state) is not dict or state.get("schema_version") != 1 or state.get("lineage") != lineage:
            _global_fail("checkpoint_lineage_mismatch",
                         "checkpoint is not this global job's lineage (phase/config/split/initial state)")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        completed_epoch, samples_processed = state["completed_epoch"], state["samples_processed"]
        history, evaluations = state["metrics"], state["evaluations"]
        if len(history) != completed_epoch or not 0 <= completed_epoch <= epochs:
            raise ValueError("checkpoint epoch journal is inconsistent")
        _restore_rng(torch, np, state["rng"])

    batch_size, accumulation = training["batch_size"], training["accumulation_steps"]
    train_records = inputs["train_records"]
    batches_per_epoch = math.ceil(len(train_records) / batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
    evaluated_name = "test" if final else "selection"
    needs_opencv = any(op["op"] == "clahe" for op in data_cfg["augmentation"]["ops"])
    environment = {"schema_version": 1, "utc": utc_now(), "phase": phase, "protocol_version": cfg["protocol_version"],
                   "python": platform.python_version(), "torch": torch.__version__,
                   "torchvision": torchvision.__version__, "numpy": np.__version__,
                   "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "device": str(device), "workers": workers, "trainer_override": override,
                   "trainer_constants": json.loads(json.dumps(gj.TRAINER_CONSTANTS_V1)),
                   "precision": "float32", "autocast": False, "tf32": False, "deterministic_algorithms": True,
                   "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                   "batch_size": batch_size, "accumulation_steps": accumulation,
                   "train_images": len(train_records), "optimizer_updates_per_epoch": updates_per_epoch,
                   "evaluated_partition": ge.PARTITIONS[phase], "evaluated_images": len(inputs["eval_records"]),
                   "weights": cfg["model"]["weights"], "weights_sha256": weights_hash,
                   "initial_state_sha256": initial_state_sha256,
                   "augmentation_backend": data_cfg["augmentation"]["backend"],
                   "augmentation_resize_order": "resize-before-augmentation",
                   "code_commit": os.environ.get("ATLAS_CODE_COMMIT", "unavailable"),
                   "image_id": os.environ.get("ATLAS_IMAGE_ID", "unavailable"),
                   "config_sha256": config_hash, "splits_sha256": splits_hash,
                   "global": {key: cfg["global"][key] for key in ("pools_sha256", "hpo_view_sha256", "plan_sha256",
                                                                  "candidate_id", "freeze_kind", "freeze_sha256",
                                                                  "fold")},
                   "dataset_integrity": inputs["integrity"]}
    if needs_opencv:
        import cv2
        environment["opencv"] = cv2.__version__
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment["gpu"] = {"name": properties.name, "compute_capability": [properties.major, properties.minor],
                              "total_memory_bytes": properties.total_memory,
                              "uuid": str(getattr(properties, "uuid", "unavailable"))}
    _atomic_bytes(output / "config.json", config_bytes)
    if not (output / "splits.json").exists():
        atomic_json(output / "splits.json", view)
    atomic_json(output / "environment.json", environment)

    interrupted = {"signal": None}

    def on_signal(number, _frame):
        interrupted["signal"] = number
    previous = {number: signal.signal(number, on_signal) for number in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()

    def save_checkpoint():
        _atomic_torch(torch, checkpoint_path, {
            "schema_version": 1, "lineage": lineage, "config_sha256": config_hash, "splits_sha256": splits_hash,
            "completed_epoch": completed_epoch, "samples_processed": samples_processed, "metrics": history,
            "evaluations": evaluations, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": _rng_state(torch, np)})

    def progress(status):
        atomic_json(output / "progress.json", {"schema_version": 1, "status": status, "utc": utc_now(),
                    "phase": phase, "epoch": completed_epoch, "epochs": epochs,
                    "samples_processed": samples_processed, "elapsed_seconds": time.monotonic() - started,
                    "config_sha256": config_hash, "splits_sha256": splits_hash, "checkpoint": "checkpoint.pt",
                    "interruption_signal": interrupted["signal"]})

    def evaluate_last_epoch():
        # The evaluated partition (E_f or D_select) is opened only here, once.
        if completed_epoch != epochs:
            _global_fail("test_access_before_last_epoch", "the evaluated partition opens only after the last epoch")
        model.eval()
        loader = torch.utils.data.DataLoader(ClassificationDataset(data_cfg, inputs["eval_records"]),
                                             batch_size=batch_size, shuffle=False, num_workers=workers)
        loss_sum, correct, count, scores = 0.0, 0, 0, []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("non-finite evaluation logits")
                scores.extend(logits.softmax(dim=1).detach().cpu().double().tolist())
                loss_sum += loss.item() * len(labels)
                correct += int((logits.argmax(1) == labels).sum().item())
                count += len(labels)
        if count != len(inputs["eval_ids"]):
            raise RuntimeError("evaluated partition was not covered exactly once")
        return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}, scores

    # Additive per-job telemetry (telemetry/, not a validated artifact); it
    # never touches RNG, batch order, loss, checkpoints or metrics.json and
    # never receives evaluated-partition metrics (E_f/D_select open only once).
    telemetry = train_telemetry.open_for_attempt(
        output, torch=torch, device=device, grad_clip_max_norm=gj.TRAINER_CONSTANTS_V1["gradient_clip_max_norm"],
        context={"path": "global", "phase": phase, "resume": bool(resume), "start_epoch": completed_epoch + 1,
                 "epochs": epochs, "device": str(device), "workers": workers, "trainer_override": override is not None,
                 "batch_size": batch_size, "accumulation_steps": accumulation, "train_images": len(train_records),
                 "batches_per_epoch": batches_per_epoch, "optimizer_updates_per_epoch": updates_per_epoch,
                 "arm": data_cfg["augmentation"]["arm"], "fold": cfg["global"]["fold"],
                 "candidate_id": cfg["global"]["candidate_id"], "config_sha256": config_hash,
                 "splits_sha256": splits_hash, "code_commit": environment["code_commit"],
                 "image_id": environment["image_id"]})
    try:
        if not resume:
            save_checkpoint()
        progress("running")
        # Fixed epochs, no per-epoch evaluation, no early stopping (ADR-05).
        for epoch in range(completed_epoch + 1, epochs + 1):
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            order = list(range(len(train_records)))
            random.Random(int(canonical_hash([seed, epoch, "order"])[:16], 16)).shuffle(order)
            batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
            loader = torch.utils.data.DataLoader(ClassificationDataset(data_cfg, train_records, epoch, True),
                                                 batch_sampler=batches, num_workers=workers)
            telemetry.epoch_start(epoch, batches=len(batches), images=len(order))
            optimizer.zero_grad(set_to_none=True)
            loss_sum, count, updates = 0.0, 0, 0
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
                grad_norm = None
                if (index + 1) % accumulation == 0 or index + 1 == len(batches):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                               gj.TRAINER_CONSTANTS_V1["gradient_clip_max_norm"])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                batch_loss = loss.item()
                loss_sum += batch_loss * len(labels)
                count += len(labels)
                telemetry.batch(epoch, index, len(labels), batch_loss, optimizer.param_groups[0]["lr"], grad_norm)
            telemetry.train_done(epoch)
            if updates != updates_per_epoch or count != len(train_records):
                raise RuntimeError("optimizer updates per epoch differ from ceil(ceil(n/batch)/accumulation)")
            samples_processed += count
            scheduler.step()
            completed_epoch = epoch
            history.append({"epoch": epoch, "utc": utc_now(), "train_loss": loss_sum / count,
                            "samples_processed": samples_processed, "optimizer_updates": updates,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            # The checkpoint commits the epoch; the public journal follows it.
            save_checkpoint()
            atomic_json(output / "metrics.json", history)
            progress("running")
            telemetry.epoch_end(epoch, history[-1], selection_per_epoch=False)
            if interrupted["signal"] is not None or (stop_after_epoch is not None and epoch >= stop_after_epoch):
                telemetry.outcome("interrupted")
                progress("interrupted")
                return 128 + interrupted["signal"] if interrupted["signal"] else 75
        if completed_epoch != epochs:
            raise RuntimeError("fixed epochs were not completed")
        final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        _atomic_torch(torch, output / "best.pt", {"model": final_state, "epoch": epochs,
                                                "checkpoint_selection": "last_epoch", "config_sha256": config_hash})
        # Journal the access in the checkpoint before the partition is read.
        evaluations += 1
        save_checkpoint()
        evaluated, scores = evaluate_last_epoch()
        if final:
            artifact = ge.build_global_predictions(cfg, view, inputs["records"], scores, pools=inputs["pools"],
                                                   epoch=epochs)
        else:
            artifact = ge.build_hpo_predictions_from_view(cfg, view, scores, epoch=epochs)
        previous_predictions = output / "predictions.json"
        if previous_predictions.exists():
            earlier = json.loads(previous_predictions.read_text(encoding="utf-8")).get("content_sha256")
            if earlier != artifact["content_sha256"]:
                raise RuntimeError("re-evaluation after resume did not reproduce the same predictions")
        primary = artifact["metrics"]["primary"]
        results = {"schema_version": 1, "phase": phase, "scientific_status": "provisional",
                   "protocol_version": cfg["protocol_version"], "checkpoint_selection": "last_epoch",
                   "best_epoch": epochs, "epochs": epochs, "config_sha256": config_hash,
                   "splits_sha256": splits_hash, "initial_state_sha256": initial_state_sha256,
                   "trainer_override": override, evaluated_name: evaluated,
                   f"{evaluated_name}_evaluations": evaluations,
                   "clinical_evaluation": {"artifact": ge.ARTIFACT_NAME, "partition": artifact["partition"],
                                           "samples": len(artifact["rows"]), "primary_metric": primary["metric"],
                                           "primary_value": primary["value"], "primary_reason": primary["reason"],
                                           "content_sha256": artifact["content_sha256"]},
                   "execution_verified": False, "scientific_eligible": False}
        if interrupted["signal"] is not None:
            telemetry.outcome("interrupted")
            progress("interrupted")
            return 128 + interrupted["signal"]
        atomic_json(output / "predictions.json", artifact)
        atomic_json(output / "results.json", results)
        atomic_json(output / "metrics.json", history)
        progress("completed")
        required = required_artifacts(cfg)
        artifacts = {name: file_hash(output / name) for name in required}
        sizes = {name: (output / name).stat().st_size for name in required}
        if any(size <= 0 for size in sizes.values()):
            raise RuntimeError("empty required artifact")
        atomic_json(output / "completed.json", {"schema_version": 1, "status": "completed", "utc": utc_now(),
                    "phase": phase, "epoch": completed_epoch, "config_sha256": config_hash,
                    "splits_sha256": splits_hash, "trainer_override": override,
                    "artifacts": artifacts, "artifact_sizes": sizes})
        telemetry.outcome("completed")
        return 0
    finally:
        telemetry.close()
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
        if args.preflight and cfg["phase"] in GLOBAL_PHASES:
            from .global_evaluation import config_sha256
            inputs = _global_inputs(cfg)
            print(json.dumps({"valid": True, "phase": cfg["phase"], "config_sha256": config_sha256(cfg),
                              "splits_sha256": inputs["view"]["sha256"],
                              "counts": {"train": len(inputs["train_records"]),
                                         "evaluated": len(inputs["eval_records"])}}))
            return 0
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
