"""Minimal RSNA JPEG-baseline DICOM reader; no pydicom, windowing or PHI export.

Only explicit-VR little-endian, single-frame CR MONOCHROME2 8-bit 1024²
JPEG baseline is accepted. Undefined-length sequences are skipped structurally.
This is deliberately not a general-purpose DICOM implementation.
"""
import hashlib
import io
import struct

from .derive import _pil
from .errors import IntegrityError

JPEG_BASELINE = "1.2.840.10008.1.2.4.50"
LONG_VR = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN", b"OD", b"OL",
           b"UC", b"UR", b"OV", b"SV", b"UV"}
U16 = {(0x28, 0x10): "Rows", (0x28, 0x11): "Columns",
       (0x28, 0x100): "BitsAllocated", (0x28, 0x101): "BitsStored",
       (0x28, 0x102): "HighBit", (0x28, 0x103): "PixelRepresentation",
       (0x28, 2): "SamplesPerPixel"}
STR = {(2, 0x10): "TransferSyntaxUID", (8, 0x18): "SOPInstanceUID",
       (0x20, 0x0D): "StudyInstanceUID", (8, 0x60): "Modality",
       (0x28, 4): "PhotometricInterpretation", (0x18, 0x5101): "ViewPosition",
       (0x28, 8): "NumberOfFrames"}


def _need(data, pos, size):
    if pos < 0 or pos + size > len(data):
        raise IntegrityError("truncated DICOM element")


def _item(data, pos):
    _need(data, pos, 8)
    group, element, length = struct.unpack_from("<HHI", data, pos)
    return (group, element), length, pos + 8


def _element(data, pos):
    _need(data, pos, 8)
    tag = struct.unpack_from("<HH", data, pos)
    vr = data[pos + 4:pos + 6]
    if not (len(vr) == 2 and all(65 <= c <= 90 for c in vr)):
        raise IntegrityError("DICOM requires explicit VR")
    if vr in LONG_VR:
        _need(data, pos, 12)
        if data[pos + 6:pos + 8] != b"\0\0":
            raise IntegrityError("invalid DICOM reserved bytes")
        length, = struct.unpack_from("<I", data, pos + 8)
        start = pos + 12
    else:
        length, = struct.unpack_from("<H", data, pos + 6)
        start = pos + 8
    if length != 0xFFFFFFFF:
        _need(data, start, length)
    return tag, length, start


def _skip_sequence(data, pos, depth=0):
    if depth > 32:
        raise IntegrityError("DICOM sequence nesting exceeds limit")
    while True:
        tag, length, start = _item(data, pos)
        if tag == (0xFFFE, 0xE0DD) and length == 0:
            return start
        if tag != (0xFFFE, 0xE000):
            raise IntegrityError("invalid DICOM sequence item")
        if length != 0xFFFFFFFF:
            _need(data, start, length)
            pos = start + length
            continue
        pos = start
        while True:
            _need(data, pos, 8)
            if struct.unpack_from("<HH", data, pos) == (0xFFFE, 0xE00D):
                _, size, pos = _item(data, pos)
                if size:
                    raise IntegrityError("invalid DICOM item delimiter")
                break
            _, size, value = _element(data, pos)
            pos = _skip_sequence(data, value, depth + 1) if size == 0xFFFFFFFF else value + size


def _jpeg(data, pos):
    tag, size, start = _item(data, pos)
    if tag != (0xFFFE, 0xE000) or size not in (0, 4):
        raise IntegrityError("expected single-frame basic offset table")
    _need(data, start, size)
    if size and data[start:start + size] != b"\0" * 4:
        raise IntegrityError("invalid single-frame basic offset table")
    pos, fragments = start + size, []
    while True:
        tag, size, start = _item(data, pos)
        if tag == (0xFFFE, 0xE0DD) and size == 0:
            break
        if tag != (0xFFFE, 0xE000) or size in (0, 0xFFFFFFFF) or size % 2:
            raise IntegrityError("invalid encapsulated JPEG fragment")
        _need(data, start, size)
        fragments.append(data[start:start + size])
        pos = start + size
    body = b"".join(fragments)
    if not body.startswith(b"\xff\xd8") or not fragments:
        raise IntegrityError("missing encapsulated JPEG frame")
    return body


def decode(data):
    """Return (decoded L image, safe metadata, SHA256 of decoded 1024² bytes)."""
    if len(data) > 64 * 1024 * 1024 or data[128:132] != b"DICM":
        raise IntegrityError("unsupported DICOM size or missing Part 10 preamble")
    pos, meta, jpeg = 132, {}, None
    while pos < len(data):
        tag, size, start = _element(data, pos)
        if tag[0] != 2 and meta.get("TransferSyntaxUID") != JPEG_BASELINE:
            raise IntegrityError("unsupported DICOM transfer syntax")
        if tag == (0x7FE0, 0x10):
            if size != 0xFFFFFFFF:
                raise IntegrityError("expected encapsulated DICOM pixel data")
            jpeg = _jpeg(data, start)
            break
        if size == 0xFFFFFFFF:
            pos = _skip_sequence(data, start)
            continue
        raw = data[start:start + size]
        if tag in U16:
            if size != 2:
                raise IntegrityError("invalid DICOM unsigned short")
            meta[U16[tag]], = struct.unpack("<H", raw)
        elif tag in STR:
            meta[STR[tag]] = raw.rstrip(b"\0 ").decode("ascii")
        pos = start + size
    required = {"TransferSyntaxUID": JPEG_BASELINE, "Rows": 1024, "Columns": 1024,
                "BitsAllocated": 8, "BitsStored": 8, "HighBit": 7,
                "PixelRepresentation": 0, "SamplesPerPixel": 1,
                "Modality": "CR", "PhotometricInterpretation": "MONOCHROME2"}
    if jpeg is None or any(meta.get(k) != v for k, v in required.items()):
        raise IntegrityError("DICOM does not match approved RSNA pixel specification")
    if meta.get("NumberOfFrames", "1") != "1" or meta.get("ViewPosition") not in ("PA", "AP"):
        raise IntegrityError("unsupported DICOM frame count or ViewPosition")
    if not meta.get("SOPInstanceUID") or not meta.get("StudyInstanceUID"):
        raise IntegrityError("DICOM missing image/study identity")
    Image, _ = _pil()
    with Image.open(io.BytesIO(jpeg)) as image:
        if image.format != "JPEG" or image.mode != "L" or image.size != (1024, 1024):
            raise IntegrityError("encapsulated JPEG disagrees with DICOM header")
        image.load()
        # New image strips embedded JPEG metadata; never copy EXIF/ICC/comments.
        pixels = image.tobytes()
        clean = Image.frombytes("L", image.size, pixels)
    return clean, meta, hashlib.sha256(pixels).hexdigest()


def derive(data):
    """Worker with no writes: return metadata and deterministic 512² PNG bytes."""
    image, meta, pixel_hash = decode(data)
    Image, _ = _pil()
    resized = image.resize((512, 512), Image.Resampling.BILINEAR, reducing_gap=None)
    sink = io.BytesIO()
    resized.save(sink, "PNG")
    body = sink.getvalue()
    return {"original": meta, "pixel_sha256": pixel_hash,
            "derived": {"width": 512, "height": 512, "mode": "L", "format": "PNG",
                        "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}}, body
