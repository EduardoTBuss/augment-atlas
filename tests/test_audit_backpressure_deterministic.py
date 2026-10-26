"""Audit 2026-10-26, review finding V1: timing-independent backpressure check.

The legacy ``HappyPathTests.test_two_workers_with_tiny_spool_apply_backpressure`` makes
derivation "slower than the stream" with a fixed 20 ms sleep in the pool workers. The
stream side is not bounded: per member the main process fsyncs the spool file (and, on
POSIX, the spool and derived directories) and commits the ledger twice with
``synchronous=FULL``. When those durable writes cost more than the derive side (about
10 ms per member with two workers), the stream is the bottleneck, the spool never fills
after the worker warm-up, and the legacy test sees only 2-3 ranges. That is the correct
behavior of the tool, so the legacy assertion depends on the host's fsync latency.

Here derivation is made infinitely slower than the stream without any clock: a stalling
deriver finishes a job only when the runner blocks on it. The runner, the stream, the
spool accounting and the ledger are the real ones; only the process pool is replaced.
"""
from collections import deque
import shutil
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquisition_e2e_helpers import GIB  # noqa: E402
from test_acquisition_isic_e2e import AcquisitionE2E  # noqa: E402

from augment_atlas.acquisition import pipeline  # noqa: E402

SPOOL_CAP = 2400  # bytes: three 700-byte synthetic members fit, a fourth does not


class StallingDeriver(pipeline._Deriver):
    """Same capacity as a two-worker pool, but a job completes only inside ``ready(block=True)``."""

    def __init__(self, fn, workers):
        super().__init__(fn, 0)  # no process pool
        self.capacity = 2 * workers
        self.jobs = deque()

    def submit(self, job, params):
        job.params = params
        self.jobs.append(job)

    def ready(self, block=False):
        if not block or not self.jobs:
            return []
        job = self.jobs.popleft()
        try:
            job.result = self.fn(str(job.spool), str(job.tmp), job.params)
        except Exception as exc:  # same contract as the pool path
            job.result = {"status": "error", "exception": exc}
        return [job]


class DeterministicBackpressureTests(AcquisitionE2E):
    def run_full(self):
        self.write_config(transfer={"workers": 2}, disk={"spool_cap_gib": SPOOL_CAP / GIB})
        peaks = []
        on_member = pipeline.Runner._on_member

        def tracking(runner, result):
            peaks.append(runner.spool_bytes)  # after the member was added, before any job completes
            return on_member(runner, result)
        with mock.patch.object(pipeline, "_Deriver", StallingDeriver), \
                mock.patch.object(pipeline.Runner, "_on_member", tracking):
            self.assertEqual(self.cli("full"), 0, self.last_stderr)
        return peaks

    def test_slow_derivation_pauses_and_reopens_the_stream_regardless_of_timing(self):
        peaks = self.run_full()
        self.assert_ready()
        starts = self.stream_starts()
        self.assertEqual(starts, sorted(set(starts)), "each reopening resumes further into the archive")
        self.assertGreater(len(starts), 3, "the connection is closed and reopened after each pause")
        self.assertLessEqual(max(peaks), SPOOL_CAP, "the spool never holds more than its cap")
        self.assertEqual(self.ledger_counts()["DERIVED"], len(self.release.rows))

    def test_the_number_of_reopenings_is_reproducible(self):
        first = len(self.run_full()), len(self.stream_starts())
        self.reset_root_and_server()
        second = len(self.run_full()), len(self.stream_starts())
        self.assertEqual(first, second)

    def reset_root_and_server(self):
        shutil.rmtree(self.root)
        self.serve(self.release.objects())


if __name__ == "__main__":
    unittest.main()
