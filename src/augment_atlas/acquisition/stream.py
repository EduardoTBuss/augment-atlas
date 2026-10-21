"""Sequential streaming extraction of ZIP members, verified against the central directory.

One ranged stream covers ``[offset of member k, cd_offset)``. Members are read in
local-header order, inflated incrementally, checked (CRC32 and size against the CD,
local header against the CD, exact contiguity) and written to the spool as
``member-<cd index>.bin`` (names never become paths). Resuming at member ``k``
opens a new range at that member's local header, so earlier members are not
re-read; a member interrupted mid-way is re-read from its header.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
import zlib

from .errors import DivergenceError, IntegrityError, RetriesExhausted, SpoolPaused, TransientError

DEFAULT_READ_BYTES = 1 << 20
_LOCAL = struct.Struct("<4s5H3L2H")
_LOCAL_SIG, _DESCRIPTOR_SIG = b"PK\x03\x04", b"PK\x07\x08"
_U32 = 0xFFFFFFFF
_FLAG_DESCRIPTOR, _ENCRYPTION_FLAGS = 0x0008, 0x0001 | 0x0040 | 0x2000


@dataclass(frozen=True)
class MemberResult:
    position: int          # index in the offset-ordered entry list
    entry: object          # CDEntry
    sha256: str
    size: int
    path: Path             # final spool file


def spool_path(spool_dir, entry):
    return Path(spool_dir) / f"member-{entry.index:06d}.bin"


def directory_usage(spool_dir):
    """Bytes currently held in the spool (all regular files, including stray temps)."""
    total = 0
    with os.scandir(spool_dir) as items:
        for item in items:
            if item.is_file(follow_symlinks=False):
                total += item.stat(follow_symlinks=False).st_size
    return total


class _Cursor:
    """Exact reads over a range stream, tracking the absolute offset."""

    def __init__(self, stream, offset):
        self.stream = stream
        self.offset = offset

    def read_exact(self, n):
        parts, missing = [], n
        while missing:
            data = self.stream.read(missing)
            if not data:
                raise IntegrityError(f"archive ends at offset {self.offset} while {missing} more bytes were expected")
            parts.append(data)
            missing -= len(data)
            self.offset += len(data)
        return b"".join(parts)

    def chunks(self, n, size):
        missing = n
        while missing:
            data = self.stream.read(min(size, missing))
            if not data:
                raise IntegrityError(f"archive ends at offset {self.offset} inside member data")
            missing -= len(data)
            self.offset += len(data)
            yield data


def _local_zip64(extra):
    pos, blocks = 0, {}
    while pos + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, pos)
        if pos + 4 + size > len(extra):
            raise IntegrityError("truncated local extra field")
        blocks[tag] = extra[pos + 4:pos + 4 + size]
        pos += 4 + size
    if pos != len(extra):
        raise IntegrityError("trailing bytes in local extra field")
    return blocks.get(0x0001)


def _read_local_header(cursor, entry):
    start = cursor.offset
    if start != entry.local_header_offset:
        raise IntegrityError(f"stream at {start}, expected local header of {entry.name!r} at {entry.local_header_offset}")
    (sig, _version, flags, method, _time, _date, crc, csize, usize, name_len,
     extra_len) = _LOCAL.unpack(cursor.read_exact(_LOCAL.size))
    if sig != _LOCAL_SIG:
        raise IntegrityError(f"bad local header signature for {entry.name!r} at {start}")
    if flags & _ENCRYPTION_FLAGS:
        raise IntegrityError(f"encrypted local header for {entry.name!r}")
    if method != entry.method or (flags & _FLAG_DESCRIPTOR) != (entry.flags & _FLAG_DESCRIPTOR):
        raise IntegrityError(f"local header of {entry.name!r} disagrees with the CD (method/descriptor flag)")
    raw_name = cursor.read_exact(name_len)
    if raw_name != entry.raw_name:
        raise IntegrityError(f"local name {raw_name!r} != CD name {entry.raw_name!r}")
    zip64 = _local_zip64(cursor.read_exact(extra_len))
    if not flags & _FLAG_DESCRIPTOR:
        if _U32 in (csize, usize):
            if zip64 is None or len(zip64) < 16:
                raise IntegrityError(f"local ZIP64 sentinel without extra for {entry.name!r}")
            usize, csize = struct.unpack_from("<QQ", zip64, 0)
        if (crc, csize, usize) != (entry.crc32, entry.compressed_size, entry.uncompressed_size):
            raise IntegrityError(f"local header CRC/sizes of {entry.name!r} disagree with the CD")
    return zip64 is not None


def _read_descriptor(cursor, entry, zip64):
    head = cursor.read_exact(4)
    if head == _DESCRIPTOR_SIG and entry.crc32 != struct.unpack("<L", _DESCRIPTOR_SIG)[0]:
        head = cursor.read_exact(4)
    crc = struct.unpack("<L", head)[0]
    csize, usize = struct.unpack("<QQ" if zip64 else "<LL", cursor.read_exact(16 if zip64 else 8))
    if (crc, csize, usize) != (entry.crc32, entry.compressed_size, entry.uncompressed_size):
        raise IntegrityError(f"data descriptor of {entry.name!r} disagrees with the CD")


def _fsync_dir(path):
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _extract_one(cursor, entry, sink, read_bytes):
    """Copy/inflate one member's data into ``sink``; return ``(sha256 hex, size)`` after CD checks."""
    zip64 = _read_local_header(cursor, entry)
    crc, size, digest = 0, 0, hashlib.sha256()
    limit = entry.uncompressed_size
    inflater = zlib.decompressobj(-15) if entry.method == 8 else None

    def emit(data):
        nonlocal crc, size
        size += len(data)
        if size > limit:
            raise IntegrityError(f"{entry.name!r} inflates beyond its CD size {limit}")
        crc = zlib.crc32(data, crc)
        digest.update(data)
        if sink is not None:
            sink.write(data)

    for chunk in cursor.chunks(entry.compressed_size, read_bytes):
        if inflater is None:
            emit(chunk)
            continue
        if inflater.eof:
            raise IntegrityError(f"deflate stream of {entry.name!r} ended before its compressed size")
        try:
            data = chunk
            while data:
                # Bounded output per step: a hostile ratio cannot allocate past limit + read_bytes.
                emit(inflater.decompress(data, read_bytes))
                data = inflater.unconsumed_tail
        except zlib.error as exc:
            raise IntegrityError(f"corrupt deflate data in {entry.name!r}: {exc}") from exc
    if inflater is not None:
        try:
            emit(inflater.flush())
        except zlib.error as exc:
            raise IntegrityError(f"corrupt deflate data in {entry.name!r}: {exc}") from exc
        if not inflater.eof or inflater.unused_data:
            raise IntegrityError(f"deflate stream of {entry.name!r} does not end exactly at its compressed size")
    if entry.has_data_descriptor:
        _read_descriptor(cursor, entry, zip64)
    if size != entry.uncompressed_size:
        raise IntegrityError(f"{entry.name!r} produced {size} bytes, CD says {entry.uncompressed_size}")
    if crc != entry.crc32:
        raise IntegrityError(f"CRC32 mismatch for {entry.name!r}: got {crc:08x}, CD says {entry.crc32:08x}")
    return digest.hexdigest(), size


def stream_members(entries, opener, *, cd_offset, spool_dir, spool_cap_bytes, start=0, on_member=None,
                   read_bytes=DEFAULT_READ_BYTES, spool_usage=None, before_member=None):
    """Extract ``entries[start:]`` (offset-ordered, as from ``validate_entries``) to the spool.

    ``opener(first, last)`` returns a context-managed stream with ``read(n)`` over the
    inclusive byte range (``RangeClient.open_range`` fits). ``on_member(MemberResult)``
    runs after the spool file is durable; ``before_member(position, entry)`` runs right before a
    non-directory member is written to the spool (disk guard hook; its exceptions propagate).
    Returns ``len(entries)`` on completion.
    Raises SpoolPaused(next_index) when the next member does not fit under the cap,
    and TransientError with ``next_index`` set when the transport fails.
    """
    offsets = [entry.local_header_offset for entry in entries]
    if offsets != sorted(offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("entries must be strictly ordered by local header offset")
    if not 0 <= start <= len(entries):
        raise ValueError("start out of range")
    if start == len(entries):
        return start
    spool_dir = Path(spool_dir)
    usage = spool_usage or (lambda: directory_usage(spool_dir))
    position = start
    try:
        with opener(entries[start].local_header_offset, cd_offset - 1) as stream:
            cursor = _Cursor(stream, entries[start].local_header_offset)
            for position in range(start, len(entries)):
                entry = entries[position]
                if entry.is_dir:
                    _extract_one(cursor, entry, None, read_bytes)
                    continue
                if entry.uncompressed_size > spool_cap_bytes:
                    raise DivergenceError(f"{entry.name!r} ({entry.uncompressed_size} B) exceeds the spool cap")
                if usage() + entry.uncompressed_size > spool_cap_bytes:
                    raise SpoolPaused(f"spool cap reached before {entry.name!r}", next_index=position)
                if before_member is not None:
                    before_member(position, entry)
                result = _spool_member(cursor, entry, position, spool_dir, read_bytes)
                if on_member is not None:
                    on_member(result)
            if cursor.offset != cd_offset:
                raise IntegrityError(f"{cd_offset - cursor.offset} unaccounted bytes before the central directory")
    except TransientError as exc:
        exc.next_index = position
        raise
    return len(entries)


def _spool_member(cursor, entry, position, spool_dir, read_bytes):
    final = spool_path(spool_dir, entry)
    temp = final.with_suffix(".part")
    try:
        with open(temp, "wb") as sink:
            sha256, size = _extract_one(cursor, entry, sink, read_bytes)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temp, final)
        _fsync_dir(spool_dir)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return MemberResult(position=position, entry=entry, sha256=sha256, size=size, path=final)


def stream_with_resume(entries, opener, *, retry, start=0, **kwargs):
    """``stream_members`` with bounded resume: each transport failure reopens at the
    interrupted member's header after ``retry.delay``; ``retry.max_attempts`` sessions max.
    Integrity/divergence errors and SpoolPaused propagate immediately.
    """
    position, failures = start, 0
    while True:
        try:
            return stream_members(entries, opener, start=position, **kwargs)
        except TransientError as exc:
            failures += 1
            position = getattr(exc, "next_index", position)
            if failures >= retry.max_attempts:
                exhausted = RetriesExhausted(f"streaming stopped at member {position} after {failures} failures: {exc}")
                exhausted.next_index = position
                raise exhausted from exc
            retry.sleep(retry.delay(failures))


__all__ = ["MemberResult", "directory_usage", "spool_path", "stream_members", "stream_with_resume"]
