"""Offline fixtures for acquisition tests: a local Range server and a ZIP builder.

Nothing here touches the network beyond 127.0.0.1.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
import struct
import threading
import zlib

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class ServerState:
    """Mutable behaviour of the fake object store (guarded by ``lock``)."""

    def __init__(self, data, etag="abc123-2"):
        self.lock = threading.Lock()
        self.data = data
        self.etag = etag
        self.last_modified = "Fri, 19 Sep 2025 12:00:00 GMT"
        self.accept_ranges = "bytes"
        self.ignore_range = False          # always answer 200 with the full body
        self.content_range_shift = 0       # lie in Content-Range start
        self.drop_offsets = []             # absolute offsets at which to cut a body once each
        self.fail_statuses = []            # statuses to return (once each) before serving
        self.send_last_modified = True
        self.requests = []                 # (method, Range, If-Range)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state = None  # injected per server

    def log_message(self, *args):  # keep test output clean
        pass

    def _common(self, length):
        state = self.state
        self.send_header("Content-Length", str(length))
        self.send_header("ETag", f'"{state.etag}"')
        if state.send_last_modified:
            self.send_header("Last-Modified", state.last_modified)
        if state.accept_ranges:
            self.send_header("Accept-Ranges", state.accept_ranges)

    def _maybe_fail(self):
        with self.state.lock:
            status = self.state.fail_statuses.pop(0) if self.state.fail_statuses else None
        if status is None:
            return False
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        return True

    def do_HEAD(self):
        state = self.state
        with state.lock:
            state.requests.append(("HEAD", None, None))
        if self._maybe_fail():
            return
        self.send_response(200)
        self._common(len(state.data))
        self.end_headers()

    def do_GET(self):
        state = self.state
        range_header = self.headers.get("Range")
        if_range = self.headers.get("If-Range")
        with state.lock:
            state.requests.append(("GET", range_header, if_range))
        if self._maybe_fail():
            return
        data = state.data
        size = len(data)
        start, end, partial = 0, size - 1, False
        match = _RANGE.match(range_header or "")
        if match and not state.ignore_range and (if_range is None or if_range == f'"{state.etag}"'):
            first, last = match.groups()
            if first == "":
                start, end = max(0, size - int(last)), size - 1
            else:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            partial = True
        self.send_response(206 if partial else 200)
        self._common(end - start + 1)
        if partial:
            self.send_header("Content-Range", f"bytes {start + state.content_range_shift}-{end}/{size}")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        cut = None
        with state.lock:
            for offset in state.drop_offsets:
                if start <= offset <= end:
                    cut = offset
                    state.drop_offsets.remove(offset)
                    break
        body = data[start:end + 1] if cut is None else data[start:cut]
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except OSError:
            pass


class LocalRangeServer:
    """Threaded HTTP server on 127.0.0.1 serving one object at ``/object.zip``."""

    def __init__(self, data, etag="abc123-2"):
        self.state = ServerState(data, etag)
        handler = type("Handler", (_Handler,), {"state": self.state})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    @property
    def url(self):
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/object.zip"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class FakeClock:
    """Monotonic clock whose ``sleep`` advances time instantly and records delays."""

    def __init__(self, now=1000.0):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------------------
# Minimal ZIP writer with full control (ZIP64 records, descriptors, hostile names)
# ---------------------------------------------------------------------------

def build_zip(members, *, zip64=False, prefix_len=0, comment=b""):
    """Return ``(bytes_after_prefix, logical_size)`` of a ZIP placed after ``prefix_len`` zeros.

    ``members``: dicts with ``name`` (str or bytes), ``data``; optional ``method`` (0/8),
    ``descriptor`` (bool), ``descriptor_signature`` (bool, default True), ``flags`` (extra
    bits), ``crc`` (override written in CD and local header/descriptor), ``extra`` (raw CD
    extra appended), ``utf8`` (bool). Offsets are absolute, so with ``prefix_len`` > 4 GiB
    every 64-bit field is exercised without allocating the prefix.
    """
    out = bytearray()
    central = bytearray()
    for member in members:
        raw_name = member["name"] if isinstance(member["name"], bytes) else member["name"].encode("utf-8")
        data = member["data"]
        method = member.get("method", 8)
        payload = data if method == 0 else _raw_deflate(data)
        crc = member.get("crc", zlib.crc32(data))
        descriptor = member.get("descriptor", False)
        flags = member.get("flags", 0) | (0x08 if descriptor else 0) | (0x800 if member.get("utf8", True) else 0)
        offset = prefix_len + len(out)
        usize, csize = len(data), len(payload)
        if zip64:
            local_extra = struct.pack("<HHQQ", 1, 16, 0 if descriptor else usize, 0 if descriptor else csize)
            l_crc, l_c, l_u = (0, 0xFFFFFFFF, 0xFFFFFFFF) if descriptor else (crc, 0xFFFFFFFF, 0xFFFFFFFF)
        else:
            local_extra = b""
            l_crc, l_c, l_u = (0, 0, 0) if descriptor else (crc, csize, usize)
        out += struct.pack("<4s5H3L2H", b"PK\x03\x04", 45 if zip64 else 20, flags, method, 0, 0x21,
                           l_crc, l_c, l_u, len(raw_name), len(local_extra))
        out += raw_name + local_extra + payload
        if descriptor:
            if member.get("descriptor_signature", True):
                out += b"PK\x07\x08"
            out += struct.pack("<LQQ" if zip64 else "<LLL", crc, csize, usize)
        if zip64:
            cd_extra = struct.pack("<HHQQQ", 1, 24, usize, csize, offset)
            c_c, c_u, c_off = 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        else:
            cd_extra = b""
            c_c, c_u, c_off = csize, usize, offset
        cd_extra += member.get("extra", b"")
        central += struct.pack("<4s6H3L5H2L", b"PK\x01\x02", 0x031E, 45 if zip64 else 20, flags, method, 0, 0x21,
                               crc, c_c, c_u, len(raw_name), len(cd_extra), 0, 0, 0, 0, c_off)
        central += raw_name + cd_extra
    cd_offset = prefix_len + len(out)
    out += central
    count = len(members)
    if zip64:
        eocd64_offset = prefix_len + len(out)
        out += struct.pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, count, count, len(central), cd_offset)
        out += struct.pack("<4sLQL", b"PK\x06\x07", 0, eocd64_offset, 1)
        out += struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, len(comment))
    else:
        out += struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, count, count, len(central), cd_offset, len(comment))
    out += comment
    return bytes(out), prefix_len + len(out)


def _raw_deflate(data):
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    return compressor.compress(data) + compressor.flush()


class VirtualObject:
    """Byte space of ``prefix_len`` zeros followed by ``tail``; served lazily by offset."""

    def __init__(self, tail, prefix_len=0):
        self.tail = tail
        self.prefix_len = prefix_len
        self.size = prefix_len + len(tail)

    def read_at(self, offset, length):
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError("out of range")
        head = b""
        if offset < self.prefix_len:
            zeros = min(length, self.prefix_len - offset)
            head, offset, length = b"\x00" * zeros, self.prefix_len, length - zeros
        start = offset - self.prefix_len
        return head + self.tail[start:start + length]
