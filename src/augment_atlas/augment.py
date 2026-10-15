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
