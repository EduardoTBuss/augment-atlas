"""Explicit classification-only PIL operations; no imports of the old project.

Intensity formulas adapted from mamografia-estudo (Guilherme de Lima, MIT).
See docs/provenance.md for retained license and deliberately changed geometry.
"""
from __future__ import annotations


def apply_augmentation(image, operations: list[dict], rng):
    from PIL import Image, ImageEnhance
    for op in operations:
        if op["op"] == "basic":
            if rng.random() < op["flip_p"]:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < op["photometric_p"]:
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(*op["brightness"]))
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(*op["contrast"]))
        elif op["op"] == "gamma":
            applied, gamma = rng.random() < op["p"], rng.uniform(*op["range"])
            if applied:
                import numpy as np
                a = np.asarray(image, dtype=np.float64)
                # Per-channel extrema for RGB; luminance extrema for grayscale.
                axes = (0, 1) if a.ndim == 3 else None
                lo, hi = a.min(axis=axes), a.max(axis=axes)
                span = hi - lo
                safe = np.where(span > 0, span, 1)
                out = lo + span * np.clip((a - lo) / safe, 0, 1) ** gamma
                image = Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8), mode=image.mode)
        elif op["op"] == "clahe":
            if rng.random() < op["p"]:
                import numpy as np
                from skimage import exposure
                if image.mode != "L":
                    raise ValueError("CLAHE v1 expects an explicitly grayscale input")
                kernel = max(8, round(op["window_fraction"] * image.size[0]))
                out = exposure.equalize_adapthist(np.asarray(image, dtype=np.float64) / 255,
                                                kernel_size=kernel, clip_limit=op["clip_limit"], nbins=256)
                image = Image.fromarray(np.clip(np.round(out * 255), 0, 255).astype(np.uint8))
        else:
            raise ValueError("unknown augmentation operation")
    return image


def candidate_operation_draws(seed_identity: list, operation: dict) -> dict:
    """Local named RNG streams; adding basic cannot change intensity draws."""
    import random
    from .contracts import canonical_hash
    name = operation["op"]
    rng = random.Random(int(canonical_hash([seed_identity, name, "candidate-v2"]), 16))
    if name == "basic":
        return {"flip": rng.random() < operation["flip_p"],
                "photometric": rng.random() < operation["photometric_p"],
                "brightness": rng.uniform(*operation["brightness"]),
                "contrast": rng.uniform(*operation["contrast"])}
    if name == "gamma":
        return {"applied": rng.random() < operation["p"], "gamma": rng.uniform(*operation["range"])}
    if name == "clahe":
        return {"applied": rng.random() < operation["p"]}
    raise ValueError("unknown candidate augmentation operation")


def apply_candidate_augmentation(image, augmentation: dict, seed_identity: list):
    """Candidate uint8 policy, applied AFTER the caller resizes its image.

    Gamma uses a fixed [0,255] domain. OpenCV CLAHE uses grayscale or RGB Lab
    luminance. Basic retains only the legacy flip/brightness/contrast policy;
    geometry additions and final scientific parameters remain outside v2.
    """
    from PIL import Image, ImageEnhance
    from .contracts import CANDIDATE_BACKEND
    if augmentation.get("backend") != CANDIDATE_BACKEND:
        raise ValueError("candidate augmentation requires its explicit backend")
    if image.mode not in ("L", "RGB"):
        raise ValueError("candidate augmentation requires uint8 PIL L or RGB")
    for operation in augmentation["ops"]:
        name = operation["op"]
        draws = candidate_operation_draws(seed_identity, operation)
        if name == "basic":
            if draws["flip"]:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if draws["photometric"]:
                image = ImageEnhance.Brightness(image).enhance(draws["brightness"])
                image = ImageEnhance.Contrast(image).enhance(draws["contrast"])
        elif name == "gamma":
            if draws["applied"]:
                import numpy as np
                pixels = np.asarray(image, dtype=np.float64)
                out = np.clip(np.rint(255.0 * (pixels / 255.0) ** draws["gamma"]), 0, 255).astype(np.uint8)
                image = Image.fromarray(out)
        elif name == "clahe":
            if draws["applied"]:
                import cv2
                import numpy as np
                pixels = np.asarray(image, dtype=np.uint8)
                clahe = cv2.createCLAHE(clipLimit=float(operation["clip_limit"]),
                                       tileGridSize=tuple(operation["tile_grid_size"]))
                if image.mode == "L":
                    result = clahe.apply(pixels)
                else:
                    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                image = Image.fromarray(result)
    return image
