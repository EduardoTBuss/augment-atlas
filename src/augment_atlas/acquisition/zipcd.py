"""ZIP end records and central directory parsing with structural validation (stdlib).

Everything here is pure: callers fetch bytes (tail suffix, then the CD range) and
hand them in. Any structural doubt is an IntegrityError (exit 2); nothing tries to
"repair" an archive.
"""

from dataclasses import dataclass
import struct

from .errors import IntegrityError

DEFAULT_SUFFIX_BYTES = 131072
SUPPORTED_METHODS = {0: "stored", 8: "deflate"}

_EOCD = struct.Struct("<4s4H2LH")              # 22 bytes
_LOCATOR = struct.Struct("<4sLQL")             # 20 bytes
_EOCD64 = struct.Struct("<4sQ2H2L4Q")          # 56 bytes
_CDH = struct.Struct("<4s6H3L5H2L")            # 46 bytes
_EOCD_SIG, _LOCATOR_SIG, _EOCD64_SIG, _CDH_SIG = b"PK\x05\x06", b"PK\x06\x07", b"PK\x06\x06", b"PK\x01\x02"
_FLAG_ENCRYPTED, _FLAG_DESCRIPTOR, _FLAG_STRONG_ENCRYPTION, _FLAG_UTF8 = 0x0001, 0x0008, 0x0040, 0x0800
_FLAG_MASKED_HEADERS = 0x2000
_U16, _U32 = 0xFFFF, 0xFFFFFFFF
_LOCAL_HEADER_FIXED = 30
_DESCRIPTOR_MAX = 24  # signature + crc + two 8-byte sizes


@dataclass(frozen=True)
class EndRecord:
    cd_offset: int
    cd_size: int
    entry_count: int
    zip64: bool
    end_offset: int  # where the CD must end: EOCD64 offset (ZIP64) or EOCD offset


@dataclass(frozen=True)
class CDEntry:
    index: int                    # position in the central directory
    name: str
    raw_name: bytes
    method: int
    flags: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    is_dir: bool

    @property
    def has_data_descriptor(self):
        return bool(self.flags & _FLAG_DESCRIPTOR)


def find_end_record(suffix, suffix_offset, total_size):
    """Locate EOCD (and, if present, ZIP64 locator + EOCD64) inside a tail ``suffix``.

    ``suffix`` holds bytes ``[suffix_offset, total_size)``. The EOCD comment must end
    exactly at EOF; the last candidate satisfying that wins.
    """
    if suffix_offset + len(suffix) != total_size:
        raise IntegrityError("suffix does not end at the object size")
    position = suffix.rfind(_EOCD_SIG)
    while position >= 0:
        if position + _EOCD.size <= len(suffix):
            fields = _EOCD.unpack_from(suffix, position)
            if position + _EOCD.size + fields[7] == len(suffix):
                return _end_from_eocd(suffix, suffix_offset, position, fields)
        position = suffix.rfind(_EOCD_SIG, 0, position)
    raise IntegrityError("end of central directory record not found in suffix")


def _end_from_eocd(suffix, suffix_offset, position, fields):
    _sig, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, _comment_len = fields
    eocd_offset = suffix_offset + position
    locator_pos = position - _LOCATOR.size
    has_locator = locator_pos >= 0 and suffix[locator_pos:locator_pos + 4] == _LOCATOR_SIG
    needs_zip64 = _U16 in (disk_entries, entries) or _U32 in (cd_size, cd_offset)
    if not has_locator:
        if needs_zip64:
            raise IntegrityError("EOCD has ZIP64 sentinels but no ZIP64 locator in suffix")
        if disk or cd_disk or disk_entries != entries:
            raise IntegrityError("multi-disk archives are not supported")
        return _checked_end(cd_offset, cd_size, entries, False, eocd_offset)
    _sig, locator_disk, eocd64_offset, disks = _LOCATOR.unpack_from(suffix, locator_pos)
    if locator_disk != 0 or disks != 1:
        raise IntegrityError("multi-disk ZIP64 archives are not supported")
    if eocd64_offset + _EOCD64.size > suffix_offset + locator_pos or eocd64_offset < suffix_offset:
        raise IntegrityError("ZIP64 EOCD is outside the fetched suffix or overlaps the locator")
    rel = eocd64_offset - suffix_offset
    (sig, record_size, _made, _needed, disk64, cd_disk64, disk_entries64, entries64, cd_size64,
     cd_offset64) = _EOCD64.unpack_from(suffix, rel)
    if sig != _EOCD64_SIG:
        raise IntegrityError("bad ZIP64 EOCD signature")
    if eocd64_offset + 12 + record_size != suffix_offset + locator_pos:
        raise IntegrityError("ZIP64 EOCD size does not reach the locator")
    if disk64 or cd_disk64 or disk_entries64 != entries64:
        raise IntegrityError("multi-disk ZIP64 archives are not supported")
    for legacy, sentinel, value in ((entries, _U16, entries64), (cd_size, _U32, cd_size64), (cd_offset, _U32, cd_offset64)):
        if legacy != sentinel and legacy != value:
            raise IntegrityError("EOCD and ZIP64 EOCD disagree")
    return _checked_end(cd_offset64, cd_size64, entries64, True, eocd64_offset)


def _checked_end(cd_offset, cd_size, entries, zip64, end_offset):
    if cd_offset + cd_size != end_offset:
        raise IntegrityError(f"central directory [{cd_offset}, +{cd_size}) does not end at {end_offset}")
    if entries * _CDH.size > cd_size:
        raise IntegrityError("entry count does not fit in the central directory")
    return EndRecord(cd_offset=cd_offset, cd_size=cd_size, entry_count=entries, zip64=zip64, end_offset=end_offset)


def _zip64_extra(extra, usize, csize, offset, disk):
    """Resolve 0xFFFFFFFF fields from the 0x0001 extra, in the order the spec fixes."""
    wanted = [name for name, value, sentinel in (("usize", usize, _U32), ("csize", csize, _U32),
                                                 ("offset", offset, _U32), ("disk", disk, _U16))
              if value == sentinel]
    values = {"usize": usize, "csize": csize, "offset": offset, "disk": disk}
    blocks = _extra_blocks(extra)
    if not wanted:
        return values
    if 0x0001 not in blocks:
        raise IntegrityError("ZIP64 sentinel without ZIP64 extra field")
    data = blocks[0x0001]
    needed = 8 * len([name for name in wanted if name != "disk"]) + (4 if "disk" in wanted else 0)
    if len(data) < needed:
        raise IntegrityError("truncated ZIP64 extra field")
    pos = 0
    for name in wanted:
        if name == "disk":
            values[name] = struct.unpack_from("<L", data, pos)[0]
            pos += 4
        else:
            values[name] = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
    return values


def _extra_blocks(extra):
    blocks = {}
    pos = 0
    while pos < len(extra):
        if pos + 4 > len(extra):
            raise IntegrityError("truncated extra field header")
        tag, size = struct.unpack_from("<HH", extra, pos)
        if pos + 4 + size > len(extra):
            raise IntegrityError("truncated extra field body")
        if tag in blocks:
            raise IntegrityError(f"duplicate extra field 0x{tag:04x}")
        blocks[tag] = extra[pos + 4:pos + 4 + size]
        pos += 4 + size
    return blocks


def _decode_name(raw, flags, extra):
    try:
        name = raw.decode("utf-8") if flags & _FLAG_UTF8 else raw.decode("cp437")
    except UnicodeDecodeError as exc:
        raise IntegrityError(f"undecodable entry name {raw!r}") from exc
    unicode_path = _extra_blocks(extra).get(0x7075)
    if unicode_path is not None:
        # Info-ZIP Unicode Path: version(1) + crc32 of raw name(4) + utf-8 name. A second,
        # different name is a classic smuggling vector; accept it only if it agrees.
        if len(unicode_path) < 5 or unicode_path[0] != 1:
            raise IntegrityError("malformed Info-ZIP unicode path field")
        try:
            alt = unicode_path[5:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IntegrityError("undecodable Info-ZIP unicode path") from exc
        if alt != name:
            raise IntegrityError(f"entry has two different names: {name!r} vs {alt!r}")
    return name


def parse_central_directory(cd_bytes, end):
    """Parse every CD header; the bytes must be consumed exactly."""
    if len(cd_bytes) != end.cd_size:
        raise IntegrityError(f"central directory is {len(cd_bytes)} bytes, EOCD says {end.cd_size}")
    entries = []
    pos = 0
    for index in range(end.entry_count):
        if pos + _CDH.size > len(cd_bytes):
            raise IntegrityError(f"central directory truncated at entry {index}")
        (sig, _made, _needed, flags, method, _time, _date, crc, csize, usize, name_len, extra_len, comment_len,
         disk, _iattr, _eattr, offset) = _CDH.unpack_from(cd_bytes, pos)
        if sig != _CDH_SIG:
            raise IntegrityError(f"bad central directory signature at entry {index}")
        pos += _CDH.size
        end_pos = pos + name_len + extra_len + comment_len
        if end_pos > len(cd_bytes):
            raise IntegrityError(f"central directory entry {index} overruns the directory")
        raw_name = cd_bytes[pos:pos + name_len]
        extra = cd_bytes[pos + name_len:pos + name_len + extra_len]
        pos = end_pos
        sizes = _zip64_extra(extra, usize, csize, offset, disk)
        if sizes["disk"] != 0:
            raise IntegrityError("multi-disk entry")
        name = _decode_name(raw_name, flags, extra)
        entries.append(CDEntry(index=index, name=name, raw_name=raw_name, method=method, flags=flags, crc32=crc,
                               compressed_size=sizes["csize"], uncompressed_size=sizes["usize"],
                               local_header_offset=sizes["offset"], is_dir=name.endswith("/")))
    if pos != len(cd_bytes):
        raise IntegrityError(f"{len(cd_bytes) - pos} trailing bytes after the last central directory entry")
    return entries


def unsafe_name_reason(name):
    """Return why ``name`` is unsafe as a relative POSIX path, or None if it is acceptable."""
    if not name:
        return "empty name"
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in name):
        return "control character"
    if "\\" in name:
        return "backslash"
    if ":" in name:
        return "drive letter or stream separator"
    if name.startswith("/"):
        return "absolute path"
    parts = name.rstrip("/").split("/") if name.endswith("/") else name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "empty, '.' or '..' path component"
    return None


def validate_entries(entries, end):
    """Structural validation from plan 3.4 step 4 (everything except dataset counts)."""
    seen = {}
    for entry in entries:
        reason = unsafe_name_reason(entry.name)
        if reason:
            raise IntegrityError(f"unsafe entry name {entry.name!r}: {reason}")
        key = entry.name.casefold()
        if key in seen:
            raise IntegrityError(f"duplicate entry name {entry.name!r} (also {seen[key]!r})")
        seen[key] = entry.name
        if entry.flags & (_FLAG_ENCRYPTED | _FLAG_STRONG_ENCRYPTION | _FLAG_MASKED_HEADERS):
            raise IntegrityError(f"encrypted entry {entry.name!r}")
        if entry.method not in SUPPORTED_METHODS:
            raise IntegrityError(f"unsupported compression method {entry.method} for {entry.name!r}")
        if entry.method == 0 and entry.compressed_size != entry.uncompressed_size:
            raise IntegrityError(f"stored entry {entry.name!r} has csize != usize")
        if entry.is_dir and (entry.uncompressed_size or entry.compressed_size > 2):
            raise IntegrityError(f"directory entry {entry.name!r} carries data")
    ordered = sorted(entries, key=lambda item: item.local_header_offset)
    for current, following in zip(ordered, ordered[1:] + [None]):
        lower_bound = current.local_header_offset + _LOCAL_HEADER_FIXED + len(current.raw_name) + current.compressed_size
        limit = end.cd_offset if following is None else following.local_header_offset
        if lower_bound > limit:
            raise IntegrityError(f"entry {current.name!r} overlaps the next entry or the central directory")
    return ordered


def load_central_directory(read_at, total_size, *, suffix=None, suffix_len=DEFAULT_SUFFIX_BYTES):
    """Fetch tail + CD through ``read_at(offset, length)``; return ``(end, ordered_entries, cd_bytes)``.

    ``suffix`` may be a pre-fetched ``(offset, bytes)`` (e.g. from ``Range: bytes=-N``).
    Entries come back sorted by local header offset, ready for sequential streaming.
    """
    if suffix is None:
        start = max(0, total_size - suffix_len)
        suffix = (start, read_at(start, total_size - start))
    suffix_offset, tail = suffix
    end = find_end_record(tail, suffix_offset, total_size)
    if end.cd_offset >= suffix_offset:
        rel = end.cd_offset - suffix_offset
        cd_bytes = tail[rel:rel + end.cd_size]
    else:
        cd_bytes = read_at(end.cd_offset, end.cd_size)
    entries = parse_central_directory(cd_bytes, end)
    return end, validate_entries(entries, end), cd_bytes


__all__ = ["CDEntry", "DEFAULT_SUFFIX_BYTES", "EndRecord", "SUPPORTED_METHODS", "find_end_record",
           "load_central_directory", "parse_central_directory", "unsafe_name_reason", "validate_entries"]
