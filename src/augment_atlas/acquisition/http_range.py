"""HTTP range transport: HEAD probe, Range/If-Range GET, token bucket, bounded backoff.

Stdlib only (``http.client``). Redirects are never followed: the pinned URL must
answer directly, otherwise the object is treated as divergent (exit 3).
"""

from dataclasses import dataclass, field
import http.client
import re
import ssl
import time
from urllib.parse import urlsplit

from .errors import DivergenceError, RetriesExhausted, TransientError

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_READ_BYTES = 1 << 20
DEFAULT_RATE_BYTES_S = 25 * 1024 * 1024  # plan 3.4: max_rate_mib_s=25
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_TRANSIENT_EXC = (OSError, http.client.HTTPException)  # socket.timeout, resets, BadStatusLine, ...


def normalize_etag(value):
    """Return the opaque strong ETag without quotes; refuse weak or empty validators.

    If-Range requires a strong validator (RFC 9110 13.1.5), so a weak ETag cannot
    protect a resume and is reported as divergence.
    """
    if value is None:
        raise DivergenceError("missing ETag")
    value = value.strip()
    if value.startswith("W/"):
        raise DivergenceError(f"weak ETag cannot be used with If-Range: {value!r}")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    if not value or '"' in value:
        raise DivergenceError(f"malformed ETag: {value!r}")
    return value


@dataclass(frozen=True)
class ObjectSpec:
    """What the config pins for one remote object."""

    url: str
    expected_size: int
    expected_etag: str | None = None  # None only on a first probe; then it is recorded

    def __post_init__(self):
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"unsupported URL: {self.url!r}")
        if not isinstance(self.expected_size, int) or self.expected_size <= 0:
            raise ValueError("expected_size must be a positive int")
        if self.expected_etag is not None:
            object.__setattr__(self, "expected_etag", normalize_etag(self.expected_etag))


@dataclass(frozen=True)
class ProbeResult:
    size: int
    etag: str
    last_modified: str
    headers: dict = field(default_factory=dict)


class TokenBucket:
    """Byte-rate limiter with injectable clock and sleep.

    Debt-based: ``acquire(n)`` always deducts ``n`` and sleeps exactly long enough to
    repay a negative balance. No loop depends on the clock advancing, so a fake clock
    cannot hang it, and requests larger than ``capacity`` need no special case.
    """

    def __init__(self, rate_bytes_s, capacity_bytes=None, clock=time.monotonic, sleep=time.sleep):
        if rate_bytes_s <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate_bytes_s)
        self.capacity = float(capacity_bytes if capacity_bytes is not None else rate_bytes_s)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._last = clock()

    def _refill(self):
        now = self._clock()
        elapsed = max(0.0, now - self._last)  # a clock going backwards never mints tokens
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def acquire(self, n):
        """Account for ``n`` bytes; return the seconds slept (0.0 if within budget)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return 0.0
        self._refill()
        self._tokens -= n
        if self._tokens >= 0:
            return 0.0
        delay = -self._tokens / self.rate
        self._sleep(delay)
        self._refill()
        return delay


@dataclass
class RetryPolicy:
    """Bounded exponential backoff. ``max_attempts`` counts the first try."""

    max_attempts: int = 8
    base_delay_s: float = 1.0
    factor: float = 2.0
    max_delay_s: float = 60.0
    sleep: object = time.sleep

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def delay(self, failures):
        """Delay after the ``failures``-th consecutive failure (1-based)."""
        return min(self.max_delay_s, self.base_delay_s * self.factor ** (failures - 1))

    def run(self, operation, describe="operation"):
        """Call ``operation()``; retry only TransientError, never more than the budget."""
        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except TransientError as exc:
                last = exc
                if attempt < self.max_attempts:
                    self.sleep(self.delay(attempt))
        raise RetriesExhausted(f"{describe}: {self.max_attempts} attempts failed; last: {last}") from last


def parse_content_range(value):
    """Parse ``bytes A-B/T`` into ints; refuse ``*`` totals and malformed values."""
    match = _CONTENT_RANGE.match((value or "").strip())
    if not match:
        raise DivergenceError(f"incoherent Content-Range: {value!r}")
    first, last, total = (int(group) for group in match.groups())
    if last < first:
        raise DivergenceError(f"incoherent Content-Range: {value!r}")
    return first, last, total


class RangeStream:
    """Body of a validated 206 response; enforces the exact byte count and the rate limit."""

    def __init__(self, connection, response, start, end, bucket=None):
        self._connection = connection
        self._response = response
        self.start = start
        self.end = end
        self.position = start
        self._bucket = bucket

    @property
    def remaining(self):
        return self.end + 1 - self.position

    def read(self, n=DEFAULT_READ_BYTES):
        """Return up to ``n`` bytes; b'' only at the declared end of the range."""
        n = min(n, self.remaining)
        if n <= 0:
            return b""
        try:
            data = self._response.read(n)
        except _TRANSIENT_EXC as exc:
            raise TransientError(f"read failed at offset {self.position}: {exc!r}") from exc
        if not data:
            # http.client returns b'' (no IncompleteRead) when the peer closes early.
            raise TransientError(f"connection closed at offset {self.position}, expected through {self.end}")
        self.position += len(data)
        if self._bucket is not None:
            self._bucket.acquire(len(data))
        return data

    def close(self):
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RangeClient:
    """Range transport for one pinned object."""

    def __init__(self, spec, *, timeout_s=DEFAULT_TIMEOUT_S, bucket=None, retry=None,
                 ssl_context=None, user_agent="augment-atlas-acquisition/1"):
        self.spec = spec
        self.timeout_s = timeout_s
        self.bucket = bucket
        self.retry = retry or RetryPolicy()
        self._ssl_context = ssl_context
        self._user_agent = user_agent
        parts = urlsplit(spec.url)
        self._scheme = parts.scheme
        self._host = parts.hostname
        self._port = parts.port
        self._path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        self.etag = spec.expected_etag  # pinned or learned at probe time

    # -- connection plumbing -------------------------------------------------
    def _connect(self):
        if self._scheme == "https":
            context = self._ssl_context or ssl.create_default_context()
            return http.client.HTTPSConnection(self._host, self._port, timeout=self.timeout_s, context=context)
        return http.client.HTTPConnection(self._host, self._port, timeout=self.timeout_s)

    def _request(self, method, headers):
        connection = self._connect()
        try:
            connection.request(method, self._path, headers={"User-Agent": self._user_agent, **headers})
            response = connection.getresponse()
        except _TRANSIENT_EXC as exc:
            connection.close()
            raise TransientError(f"{method} {self.spec.url}: {exc!r}") from exc
        if response.status in _TRANSIENT_STATUS:
            connection.close()
            raise TransientError(f"{method} {self.spec.url}: HTTP {response.status}")
        return connection, response

    def _check_etag(self, response, required):
        raw = response.getheader("ETag")
        if raw is None and not required:
            return
        etag = normalize_etag(raw)
        if self.etag is not None and etag != self.etag:
            raise DivergenceError(f"ETag changed: expected {self.etag!r}, got {etag!r}")

    # -- probe -----------------------------------------------------------------
    def probe(self):
        """HEAD the object and require ranges, pinned size, pinned ETag and Last-Modified."""
        return self.retry.run(self._probe_once, describe=f"HEAD {self.spec.url}")

    def _probe_once(self):
        connection, response = self._request("HEAD", {})
        try:
            if response.status != 200:
                raise DivergenceError(f"HEAD {self.spec.url}: HTTP {response.status} (redirects are not followed)")
            accept = [token.strip().lower() for token in (response.getheader("Accept-Ranges") or "").split(",")]
            if "bytes" not in accept:
                raise DivergenceError("server does not advertise Accept-Ranges: bytes")
            length = response.getheader("Content-Length")
            if length is None or not length.strip().isdigit():
                raise DivergenceError(f"missing or invalid Content-Length: {length!r}")
            size = int(length)
            if size != self.spec.expected_size:
                raise DivergenceError(f"size changed: expected {self.spec.expected_size}, got {size}")
            etag = normalize_etag(response.getheader("ETag"))
            if self.etag is not None and etag != self.etag:
                raise DivergenceError(f"ETag changed: expected {self.etag!r}, got {etag!r}")
            last_modified = response.getheader("Last-Modified")
            if not last_modified:
                raise DivergenceError("missing Last-Modified")
            self.etag = etag
            return ProbeResult(size=size, etag=etag, last_modified=last_modified,
                               headers={key.lower(): value for key, value in response.getheaders()})
        finally:
            response.close()
            connection.close()

    # -- ranged GET --------------------------------------------------------------
    def open_range(self, start, end=None):
        """GET ``bytes=start-end`` with If-Range; demand a coherent 206 or refuse.

        A 200 means the server ignored Range or If-Range no longer matched (object
        changed); both are refused, never silently consumed from byte 0.
        """
        size = self.spec.expected_size
        if self.etag is None:
            raise DivergenceError("open_range requires a pinned or probed ETag")
        last = size - 1 if end is None else end
        if not (0 <= start <= last < size):
            raise ValueError(f"invalid range {start}-{last} for size {size}")
        connection, response = self._request("GET", {"Range": f"bytes={start}-{last}", "If-Range": f'"{self.etag}"'})
        return self._validated_stream(connection, response, start, last)

    def open_suffix(self, length):
        """GET ``bytes=-length`` (used for the EOCD tail); returns a validated stream."""
        if length <= 0:
            raise ValueError("suffix length must be positive")
        if self.etag is None:
            raise DivergenceError("open_suffix requires a pinned or probed ETag")
        size = self.spec.expected_size
        connection, response = self._request("GET", {"Range": f"bytes=-{length}", "If-Range": f'"{self.etag}"'})
        return self._validated_stream(connection, response, max(0, size - length), size - 1)

    def _validated_stream(self, connection, response, start, last):
        try:
            if response.status == 200:
                raise DivergenceError("HTTP 200 to a ranged request: Range ignored or If-Range ETag mismatch")
            if response.status != 206:
                raise DivergenceError(f"ranged GET: HTTP {response.status}")
            first, final, total = parse_content_range(response.getheader("Content-Range"))
            if (first, final, total) != (start, last, self.spec.expected_size):
                raise DivergenceError(f"Content-Range {first}-{final}/{total} != requested {start}-{last}/{self.spec.expected_size}")
            length = response.getheader("Content-Length")
            if length is not None and (not length.strip().isdigit() or int(length) != last - start + 1):
                raise DivergenceError(f"Content-Length {length!r} incoherent with Content-Range")
            self._check_etag(response, required=False)
        except BaseException:
            response.close()
            connection.close()
            raise
        return RangeStream(connection, response, start, last, bucket=self.bucket)

    def fetch(self, start, length):
        """Read ``length`` bytes at ``start`` into memory, with bounded retries."""
        def once():
            with self.open_range(start, start + length - 1) as stream:
                return _drain(stream)
        return self.retry.run(once, describe=f"GET {start}+{length}")

    def fetch_suffix(self, length):
        """Return ``(offset, bytes)`` of the last ``length`` bytes, with bounded retries."""
        def once():
            with self.open_suffix(length) as stream:
                return stream.start, _drain(stream)
        return self.retry.run(once, describe=f"GET suffix {length}")


def _drain(stream):
    parts = []
    while stream.remaining:
        parts.append(stream.read(DEFAULT_READ_BYTES))
    return b"".join(parts)


__all__ = ["DEFAULT_RATE_BYTES_S", "DEFAULT_TIMEOUT_S", "ObjectSpec", "ProbeResult", "RangeClient", "RangeStream", "RetryPolicy",
           "TokenBucket", "normalize_etag", "parse_content_range"]
