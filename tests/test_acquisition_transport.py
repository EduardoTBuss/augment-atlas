import os
import socket
import unittest

from acquisition_helpers import FakeClock, LocalRangeServer
from augment_atlas.acquisition.errors import DivergenceError, RetriesExhausted, TransientError
from augment_atlas.acquisition.http_range import (DEFAULT_RATE_BYTES_S, ObjectSpec, RangeClient, RetryPolicy, TokenBucket,
                                                  normalize_etag, parse_content_range)

DATA = os.urandom(4096) + bytes(range(256)) * 16


def no_sleep_policy(attempts=3, record=None):
    return RetryPolicy(max_attempts=attempts, base_delay_s=0.5, factor=2.0, max_delay_s=1.5,
                       sleep=(record.append if record is not None else (lambda _s: None)))


class TokenBucketTests(unittest.TestCase):
    def test_burst_then_debt_is_repaid_by_exact_sleep(self):
        clock = FakeClock()
        bucket = TokenBucket(100, capacity_bytes=100, clock=clock, sleep=clock.sleep)
        self.assertEqual(bucket.acquire(100), 0.0)
        self.assertAlmostEqual(bucket.acquire(50), 0.5)
        clock.now += 1.0  # idle refill caps at capacity
        self.assertEqual(bucket.acquire(100), 0.0)
        self.assertAlmostEqual(bucket.acquire(1000), 10.0)  # larger than capacity: no special case
        self.assertAlmostEqual(sum(clock.sleeps), 10.5)

    def test_long_run_rate_matches_limit(self):
        clock = FakeClock()
        rate = DEFAULT_RATE_BYTES_S
        self.assertEqual(rate, 25 * 1024 * 1024)
        bucket = TokenBucket(rate, capacity_bytes=rate, clock=clock, sleep=clock.sleep)
        total = 0
        start = clock()
        for _ in range(400):
            bucket.acquire(1 << 20)
            total += 1 << 20
        elapsed = clock() - start
        self.assertAlmostEqual(elapsed, (total - rate) / rate, places=6)

    def test_clock_going_backwards_never_mints_tokens_and_cannot_hang(self):
        clock = FakeClock()
        bucket = TokenBucket(10, capacity_bytes=10, clock=clock, sleep=lambda _s: None)  # sleep does not advance
        bucket.acquire(10)
        clock.now -= 50
        self.assertAlmostEqual(bucket.acquire(5), 0.5)

    def test_invalid_parameters(self):
        for kwargs in ({"rate_bytes_s": 0}, {"rate_bytes_s": 1, "capacity_bytes": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TokenBucket(**kwargs)
        with self.assertRaises(ValueError):
            TokenBucket(1).acquire(-1)


class RetryPolicyTests(unittest.TestCase):
    def test_bounded_exponential_backoff_only_for_transient(self):
        delays = []
        policy = no_sleep_policy(attempts=4, record=delays)
        calls = []

        def always_fails():
            calls.append(1)
            raise TransientError("boom")

        with self.assertRaises(RetriesExhausted) as caught:
            policy.run(always_fails)
        self.assertEqual(len(calls), 4)
        self.assertEqual(delays, [0.5, 1.0, 1.5])  # capped at max_delay
        self.assertEqual(caught.exception.exit_code, 75)

        calls.clear()

        def diverges():
            calls.append(1)
            raise DivergenceError("object changed")

        with self.assertRaises(DivergenceError):
            policy.run(diverges)
        self.assertEqual(len(calls), 1)

    def test_rejects_non_positive_budget(self):
        with self.assertRaises(ValueError):
            RetryPolicy(max_attempts=0)


class HeaderParsingTests(unittest.TestCase):
    def test_etag_normalization(self):
        self.assertEqual(normalize_etag('"ebb8-472"'), "ebb8-472")
        self.assertEqual(normalize_etag("ebb8-472"), "ebb8-472")
        for bad in (None, 'W/"x"', '""', '"a"b"'):
            with self.subTest(bad=bad), self.assertRaises(DivergenceError):
                normalize_etag(bad)

    def test_content_range(self):
        self.assertEqual(parse_content_range("bytes 0-99/24707698022"), (0, 99, 24707698022))
        for bad in ("bytes 0-99/*", "bytes 9-1/10", "items 0-1/2", None, "bytes 0-1/2, 4-5/6"):
            with self.subTest(bad=bad), self.assertRaises(DivergenceError):
                parse_content_range(bad)


class RangeClientTests(unittest.TestCase):
    def setUp(self):
        self.server = LocalRangeServer(DATA, etag="abc123-2")
        self.server.__enter__()
        self.addCleanup(self.server.__exit__, None, None, None)

    def client(self, **spec_changes):
        values = {"url": self.server.url, "expected_size": len(DATA), "expected_etag": '"abc123-2"'}
        values.update(spec_changes)
        return RangeClient(ObjectSpec(**values), timeout_s=5, retry=no_sleep_policy())

    def test_probe_accepts_pinned_object(self):
        result = self.client().probe()
        self.assertEqual((result.size, result.etag), (len(DATA), "abc123-2"))
        self.assertEqual(result.last_modified, "Fri, 19 Sep 2025 12:00:00 GMT")
        learned = self.client(expected_etag=None)
        learned.probe()
        self.assertEqual(learned.etag, "abc123-2")

    def test_probe_refusals_are_divergence_and_not_retried(self):
        cases = {
            "no ranges": lambda s: setattr(s, "accept_ranges", None),
            "ranges none": lambda s: setattr(s, "accept_ranges", "none"),
            "etag changed": lambda s: setattr(s, "etag", "other-2"),
            "malformed etag": lambda s: setattr(s, "etag", "W/\"abc"),
            "no last-modified": lambda s: setattr(s, "send_last_modified", False),
            "404": lambda s: s.fail_statuses.append(404),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                state = self.server.state
                state.accept_ranges, state.etag, state.send_last_modified = "bytes", "abc123-2", True
                state.fail_statuses.clear()
                state.requests.clear()
                mutate(state)
                with self.assertRaises(DivergenceError):
                    self.client().probe()
                self.assertEqual(len(state.requests), 1)
        self.server.state.accept_ranges, self.server.state.etag = "bytes", "abc123-2"
        self.server.state.send_last_modified = True
        with self.assertRaises(DivergenceError):
            self.client(expected_size=len(DATA) + 1).probe()

    def test_probe_retries_transient_status_then_succeeds(self):
        self.server.state.fail_statuses.extend([503, 500])
        self.assertEqual(self.client().probe().size, len(DATA))
        self.assertEqual(len(self.server.state.requests), 3)

    def test_probe_gives_up_after_budget(self):
        self.server.state.fail_statuses.extend([503] * 10)
        with self.assertRaises(RetriesExhausted):
            self.client().probe()
        self.assertEqual(len(self.server.state.requests), 3)

    def test_range_get_sends_if_range_and_returns_exact_bytes(self):
        client = self.client()
        self.assertEqual(client.fetch(100, 300), DATA[100:400])
        method, range_header, if_range = self.server.state.requests[-1]
        self.assertEqual((method, range_header, if_range), ("GET", "bytes=100-399", '"abc123-2"'))
        offset, tail = client.fetch_suffix(1000)
        self.assertEqual((offset, tail), (len(DATA) - 1000, DATA[-1000:]))
        self.assertEqual(self.server.state.requests[-1][1], "bytes=-1000")
        offset, whole = client.fetch_suffix(len(DATA) * 2)
        self.assertEqual((offset, whole), (0, DATA))

    def test_open_ended_stream_reads_to_end(self):
        with self.client().open_range(len(DATA) - 10) as stream:
            self.assertEqual(stream.read(4), DATA[-10:-6])
            self.assertEqual(stream.read(100), DATA[-6:])
            self.assertEqual(stream.read(1), b"")

    def test_changed_etag_on_resume_yields_200_and_is_refused(self):
        client = self.client()
        client.probe()
        self.server.state.etag = "replaced-object-3"
        with self.assertRaises(DivergenceError) as caught:
            client.fetch(10, 10)
        self.assertIn("200", str(caught.exception))
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(len(self.server.state.requests), 2)  # HEAD + one GET: no retry

    def test_server_ignoring_range_is_refused(self):
        self.server.state.ignore_range = True
        with self.assertRaises(DivergenceError):
            self.client().fetch(10, 10)

    def test_incoherent_content_range_is_refused(self):
        self.server.state.content_range_shift = 1
        with self.assertRaises(DivergenceError):
            self.client().fetch(10, 10)

    def test_416_and_403_are_divergence(self):
        for status in (416, 403):
            with self.subTest(status=status):
                self.server.state.fail_statuses.append(status)
                with self.assertRaises(DivergenceError):
                    self.client().fetch(0, 10)

    def test_connection_cut_mid_body_is_transient_and_fetch_recovers(self):
        client = self.client()
        self.server.state.drop_offsets.append(2000)
        with client.open_range(0) as stream:
            got = b""
            with self.assertRaises(TransientError):
                while True:
                    chunk = stream.read(512)
                    if not chunk:
                        break
                    got += chunk
        self.assertEqual(got, DATA[:2000])
        self.server.state.drop_offsets.append(150)
        self.assertEqual(client.fetch(100, 300), DATA[100:400])

    def test_stream_is_rate_limited_through_bucket(self):
        clock = FakeClock()
        bucket = TokenBucket(1024, capacity_bytes=1024, clock=clock, sleep=clock.sleep)
        client = RangeClient(ObjectSpec(self.server.url, len(DATA), "abc123-2"), timeout_s=5, bucket=bucket,
                             retry=no_sleep_policy())
        self.assertEqual(client.fetch(0, len(DATA)), DATA)
        self.assertAlmostEqual(sum(clock.sleeps), (len(DATA) - 1024) / 1024)

    def test_invalid_ranges_and_missing_etag(self):
        client = self.client()
        for start, end in ((-1, 5), (5, 4), (0, len(DATA))):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                client.open_range(start, end)
        with self.assertRaises(DivergenceError):
            self.client(expected_etag=None).open_range(0, 1)

    def test_spec_validation(self):
        for kwargs in ({"url": "ftp://x/y", "expected_size": 1}, {"url": "http://h/x", "expected_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ObjectSpec(**kwargs)


class TimeoutTests(unittest.TestCase):
    def test_silent_server_times_out_as_transient(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)  # accepts at TCP level, never answers HTTP
        port = listener.getsockname()[1]
        client = RangeClient(ObjectSpec(f"http://127.0.0.1:{port}/x", 10, "e"), timeout_s=0.2,
                             retry=no_sleep_policy(attempts=2))
        with self.assertRaises(RetriesExhausted):
            client.probe()


if __name__ == "__main__":
    unittest.main()
