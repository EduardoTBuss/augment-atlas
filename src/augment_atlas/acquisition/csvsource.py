"""Small release CSVs: whole-object GET, size + ETag(MD5) verification, exact header check.

No Range is needed (a few MB). The response must be 200 with the pinned
Content-Length and ETag; when the pinned ETag is a simple S3 ETag (32 hex digits,
no ``-N`` multipart suffix) it is the MD5 of the object and is checked against the
body. The header must equal the configured list exactly (ADR-A8): no automatic
column mapping, any difference is exit 3.
"""

import codecs
import csv
import hashlib
import http.client
import io
import os
from pathlib import Path
import re
import ssl
from urllib.parse import urlsplit

from .errors import DivergenceError, IntegrityError, TransientError
from .http_range import normalize_etag

_SIMPLE_ETAG = re.compile(r"^[0-9a-f]{32}$")
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_READ = 1 << 16
_BOM = codecs.BOM_UTF8.decode("utf-8")


def has_utf8_bom(body):
    return body.startswith(codecs.BOM_UTF8)


def is_simple_md5_etag(etag):
    return bool(_SIMPLE_ETAG.match(etag))


def _fsync_dir(path):
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fetch_object(url, *, expected_bytes, expected_etag, timeout_s, retry, bucket=None, ssl_context=None,
                 user_agent="augment-atlas-acquisition/1"):
    """GET the whole object into memory with bounded retries; return ``(body, headers)``."""
    expected_etag = normalize_etag(expected_etag)
    parts = urlsplit(url)
    path = parts.path or "/"

    def once():
        if parts.scheme == "https":
            connection = http.client.HTTPSConnection(parts.hostname, parts.port, timeout=timeout_s,
                                                     context=ssl_context or ssl.create_default_context())
        else:
            connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout_s)
        try:
            try:
                connection.request("GET", path, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
                response = connection.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                raise TransientError(f"GET {url}: {exc!r}") from exc
            if response.status in _TRANSIENT_STATUS:
                raise TransientError(f"GET {url}: HTTP {response.status}")
            if response.status != 200:
                raise DivergenceError(f"GET {url}: HTTP {response.status} (redirects are not followed)")
            if (response.getheader("Content-Encoding") or "identity").lower() != "identity":
                raise DivergenceError(f"GET {url}: unexpected Content-Encoding {response.getheader('Content-Encoding')!r}")
            length = response.getheader("Content-Length")
            if length is None or not length.strip().isdigit():
                raise DivergenceError(f"GET {url}: missing or invalid Content-Length {length!r}")
            if int(length) != expected_bytes:
                raise DivergenceError(f"GET {url}: size changed: expected {expected_bytes}, got {int(length)}")
            etag = normalize_etag(response.getheader("ETag"))
            if etag != expected_etag:
                raise DivergenceError(f"GET {url}: ETag changed: expected {expected_etag!r}, got {etag!r}")
            parts_read, total = [], 0
            while total <= expected_bytes:
                try:
                    data = response.read(min(_READ, expected_bytes + 1 - total))
                except (OSError, http.client.HTTPException) as exc:
                    raise TransientError(f"GET {url}: read failed at {total}: {exc!r}") from exc
                if not data:
                    break
                if bucket is not None:
                    bucket.acquire(len(data))
                parts_read.append(data)
                total += len(data)
            if total < expected_bytes:
                raise TransientError(f"GET {url}: connection closed after {total} of {expected_bytes} bytes")
            if total > expected_bytes:
                raise DivergenceError(f"GET {url}: body longer than Content-Length")
            headers = {key.lower(): value for key, value in response.getheaders()}
            return b"".join(parts_read), headers
        finally:
            connection.close()

    return retry.run(once, describe=f"GET {url}")


def verify_bytes(body, *, expected_bytes, expected_etag):
    """Return ``{sha256, md5, bytes}``; size and simple-ETag MD5 must match (IntegrityError otherwise)."""
    if len(body) != expected_bytes:
        raise IntegrityError(f"object has {len(body)} bytes, pinned {expected_bytes}")
    md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()
    etag = normalize_etag(expected_etag)
    if is_simple_md5_etag(etag) and md5 != etag:
        raise IntegrityError(f"MD5 {md5} != simple ETag {etag}")
    return {"sha256": hashlib.sha256(body).hexdigest(), "md5": md5, "bytes": len(body),
            "etag_is_md5": is_simple_md5_etag(etag)}


def store_release_file(body, dest):
    """Atomically write ``dest``; an existing file must be byte-identical (release objects never change)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.read_bytes() != body:
            raise IntegrityError(f"retained release file {dest.name} differs from the pinned object")
        return
    temp = dest.with_name(dest.name + ".tmp")
    with open(temp, "wb") as sink:
        sink.write(body)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temp, dest)
    _fsync_dir(dest.parent)


def parse_csv(body, expected_header, *, name):
    """Decode strict UTF-8, demand the exact header, return a list of row dicts (all fields present).

    A leading UTF-8 BOM is an encoding artifact, not a column: it is removed before the exact
    header comparison (and reported by ``has_utf8_bom`` in the probe). Nothing else is mapped.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DivergenceError(f"{name}: not valid UTF-8: {exc}") from exc
    if text.startswith(_BOM):
        text = text[len(_BOM):]
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise DivergenceError(f"{name}: empty file") from None
    except csv.Error as exc:
        raise DivergenceError(f"{name}: malformed header: {exc}") from exc
    if header != list(expected_header):
        raise DivergenceError(f"{name}: header {header!r} != expected {list(expected_header)!r} (ADR-A8: no mapping)")
    rows = []
    try:
        for line_number, fields in enumerate(reader, start=2):
            if not fields:
                continue  # blank line (e.g. trailing newline variants)
            if len(fields) != len(header):
                raise DivergenceError(f"{name}: line {line_number} has {len(fields)} fields, header has {len(header)}")
            rows.append(dict(zip(header, fields)))
    except csv.Error as exc:
        raise DivergenceError(f"{name}: malformed CSV: {exc}") from exc
    return rows


__all__ = ["fetch_object", "has_utf8_bom", "is_simple_md5_etag", "parse_csv", "store_release_file", "verify_bytes"]
