"""Derived image ``isic2020-derived-512-v1`` (plan 3.3). Pillow is imported lazily.

Pillow only exists in the pinned container; importing this module never needs it.
``derive_member`` is a top-level function so a process pool can pickle it.

Steps: decode (no exif_transpose; Orientation and ICC only recorded), ``convert("RGB")``,
resize keeping aspect to short side 512 with LANCZOS and ``reducing_gap=None``, never
upscale, encode JPEG q95 4:4:4, baseline (no progressive/optimize), without EXIF/ICC.
The long side is ``round_half_up(long * 512 / short)`` computed in exact integers.
"""

import hashlib
import os
from pathlib import Path

_SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


class PillowUnavailable(RuntimeError):
    pass


def _pil():
    try:
        from PIL import Image, features
    except ImportError as exc:
        raise PillowUnavailable("Pillow is not importable: derive runs only inside the pinned container") from exc
    return Image, features


def pillow_available():
    try:
        _pil()
    except PillowUnavailable:
        return False
    return True


def decoder_versions():
    Image, features = _pil()
    import PIL
    return {"pillow": PIL.__version__, "libjpeg_turbo": features.version("libjpeg_turbo"),
            "jpeg": features.version_codec("jpg")}


def target_size(width, height, short_side):
    """Exact integer round-half-up; never upscale (returns the original size when short <= target)."""
    short, long_ = min(width, height), max(width, height)
    if short <= short_side:
        return width, height
    scaled = (2 * long_ * short_side + short) // (2 * short)
    return (short_side, scaled) if width <= height else (scaled, short_side)


def pixel_sha256(image):
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    digest = hashlib.sha256(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_member(src, dst_tmp, params):
    """Derive ``src`` into ``dst_tmp`` (fsynced). Returns a JSON-able dict.

    ``{"status": "ok", "original": {...}, "derived": {...}}`` or
    ``{"status": "skipped", "reason": "..."}`` when the decoder refuses the input
    (corrupt data, decompression-bomb warning). Never raises for bad image bytes.
    """
    import warnings
    Image, _features = _pil()
    short_side = params["short_side"]
    dst_tmp = Path(dst_tmp)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(src) as image:
                image.load()
                original = {"format": image.format, "mode": image.mode, "width": image.width, "height": image.height,
                            "exif_orientation": int(image.getexif().get(0x0112, 1) or 1),
                            "icc": bool(image.info.get("icc_profile"))}
                rgb = image.convert("RGB")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        dst_tmp.unlink(missing_ok=True)
        return {"status": "skipped", "reason": f"decode-failed:{type(exc).__name__}"}
    size = target_size(rgb.width, rgb.height, short_side)
    if size != (rgb.width, rgb.height):
        rgb = rgb.resize(size, Image.Resampling.LANCZOS, reducing_gap=None)
    rgb.info = {}  # never carry EXIF/ICC/comments into the derived file
    try:
        with open(dst_tmp, "wb") as sink:
            rgb.save(sink, "JPEG", quality=params["quality"], subsampling=_SUBSAMPLING[params["subsampling"]],
                     progressive=False, optimize=False, exif=b"")
            sink.flush()
            os.fsync(sink.fileno())
        with Image.open(dst_tmp) as check:
            check.load()
            derived = {"width": check.width, "height": check.height, "pixel_sha256": pixel_sha256(check),
                       "icc": bool(check.info.get("icc_profile")), "exif": bool(check.info.get("exif"))}
    except BaseException:
        dst_tmp.unlink(missing_ok=True)
        raise
    derived.update(bytes=dst_tmp.stat().st_size, sha256=_sha256_file(dst_tmp))
    return {"status": "ok", "original": original, "derived": derived,
            "short_side_lt_target": min(original["width"], original["height"]) < short_side}


__all__ = ["PillowUnavailable", "decoder_versions", "derive_member", "pillow_available", "pixel_sha256", "target_size"]
