import hashlib
import io
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zipfile
import zlib

from acquisition_helpers import FakeClock, LocalRangeServer, VirtualObject, build_zip
from augment_atlas.acquisition.errors import (DivergenceError, IntegrityError, RetriesExhausted, SpoolPaused,
                                              TransientError)
from augment_atlas.acquisition.http_range import ObjectSpec, RangeClient, RetryPolicy
from augment_atlas.acquisition.stream import directory_usage, stream_members, stream_with_resume
from augment_atlas.acquisition.zipcd import (find_end_record, load_central_directory, parse_central_directory,
                                             unsafe_name_reason, validate_entries)

GIB = 1 << 30


def payload(seed, size):
    """Deterministic, partly incompressible bytes."""
    block = hashlib.sha256(str(seed).encode()).digest()
    return (block * (size // 64 + 1))[: size // 2] + os.urandom(size - size // 2)


def mixed_members():
    return [
        {"name": "ISIC_2020_Training_JPEG/train/SYNTH_0000001.jpg", "data": payload(1, 6000), "method": 0},
        {"name": "ISIC_2020_Training_JPEG/train/", "data": b"", "method": 0},
        {"name": "ISIC_2020_Training_JPEG/train/SYNTH_0000002.jpg", "data": payload(2, 9000), "method": 8,
         "descriptor": True},
        {"name": "ISIC_2020_Training_JPEG/train/SYNTH_0000003.jpg", "data": payload(3, 7000), "method": 8,
         "descriptor": True, "descriptor_signature": False},
        {"name": "ISIC_2020_Training_JPEG/train/SYNTH_0000004.jpg", "data": payload(4, 5000), "method": 0,
         "descriptor": True},
        {"name": "ISIC_2020_Training_JPEG/train/SYNTH_0000005.jpg", "data": b"", "method": 8},
    ]


class FileOpener:
    """Range opener over an in-memory/virtual object, recording requested ranges."""

    def __init__(self, obj):
        self.obj = obj
        self.ranges = []

    def __call__(self, first, last):
        self.ranges.append((first, last))
        return _VirtualStream(self.obj, first, last)


class _VirtualStream:
    def __init__(self, obj, first, last):
        self.obj, self.position, self.last = obj, first, last

    def read(self, n):
        n = min(n, self.last + 1 - self.position)
        if n <= 0:
            return b""
        data = self.obj.read_at(self.position, n)
        self.position += n
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class UnseekableSink(io.RawIOBase):
    """Forces zipfile to write data descriptors (flag bit 3), as streaming writers do."""

    def __init__(self):
        self.buffer = bytearray()

    def writable(self):
        return True

    def write(self, data):
        self.buffer += data
        return len(data)

    def seekable(self):
        return False


def parse_bytes(data, prefix_len=0, suffix_len=131072):
    obj = VirtualObject(data, prefix_len)
    return load_central_directory(obj.read_at, obj.size, suffix_len=suffix_len)


class CentralDirectoryTests(unittest.TestCase):
    def test_stdlib_zip_with_forced_zip64_records_matches_zipfile(self):
        buffer = io.BytesIO()
        with mock.patch.object(zipfile, "ZIP64_LIMIT", 64):  # every offset/size field goes ZIP64
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("train/SYNTH_0000001.jpg", payload(1, 3000), compress_type=zipfile.ZIP_STORED)
                archive.writestr("train/SYNTH_0000002.jpg", payload(2, 4000), compress_type=zipfile.ZIP_DEFLATED)
                archive.writestr("train/SYNTH_0000003.jpg", payload(3, 5000), compress_type=zipfile.ZIP_DEFLATED)
        data = buffer.getvalue()
        end, entries, _cd = parse_bytes(data)
        self.assertTrue(end.zip64)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            expected = [(info.filename, info.compress_type, info.CRC, info.compress_size, info.file_size,
                         info.header_offset) for info in archive.infolist()]
        got = [(entry.name, entry.method, entry.crc32, entry.compressed_size, entry.uncompressed_size,
                entry.local_header_offset) for entry in entries]
        self.assertEqual(got, expected)

    def test_stdlib_streaming_zip_with_descriptors_and_zip64_extras(self):
        sink = UnseekableSink()
        with zipfile.ZipFile(sink, "w") as archive:
            for index, (method, force) in enumerate(((zipfile.ZIP_STORED, True), (zipfile.ZIP_DEFLATED, True),
                                                     (zipfile.ZIP_DEFLATED, False))):
                info = zipfile.ZipInfo(f"train/ISIC_000000{index}.jpg")
                info.compress_type = method
                with archive.open(info, "w", force_zip64=force) as handle:
                    handle.write(payload(index, 4000))
        end, entries, _cd = parse_bytes(bytes(sink.buffer))
        self.assertEqual(len(entries), 3)
        self.assertTrue(all(entry.has_data_descriptor for entry in entries))

    def test_synthetic_zip64_beyond_4_gib_and_cd_outside_suffix(self):
        data, size = build_zip(mixed_members(), zip64=True, prefix_len=5 * GIB, comment=b"release")
        reads = []
        obj = VirtualObject(data, 5 * GIB)

        def read_at(offset, length):
            reads.append((offset, length))
            return obj.read_at(offset, length)

        end, entries, cd_bytes = load_central_directory(read_at, size, suffix_len=200)
        self.assertTrue(end.zip64)
        self.assertGreater(end.cd_offset, 1 << 32)
        self.assertEqual(len(reads), 2)  # tail suffix, then the CD range it points to
        self.assertEqual(reads[1], (end.cd_offset, end.cd_size))
        self.assertEqual(len(cd_bytes), end.cd_size)
        self.assertEqual([entry.local_header_offset for entry in entries],
                         sorted(entry.local_header_offset for entry in entries))
        self.assertEqual(entries[0].local_header_offset, 5 * GIB)
        self.assertEqual(sum(entry.is_dir for entry in entries), 1)
        self.assertEqual([entry.method for entry in entries], [0, 0, 8, 8, 0, 8])

    def test_prefetched_suffix_is_used(self):
        data, size = build_zip(mixed_members())
        suffix = (size - 1024, data[-1024:])
        end, entries, _cd = load_central_directory(lambda *_a: self.fail("no extra read expected"), size,
                                                   suffix=suffix)
        self.assertFalse(end.zip64)
        self.assertEqual(len(entries), 6)

    def test_missing_or_inconsistent_end_records(self):
        data, size = build_zip(mixed_members(), zip64=True)
        with self.assertRaises(IntegrityError):
            find_end_record(data[:-5], 0, size)  # suffix not ending at EOF
        with self.assertRaises(IntegrityError):
            find_end_record(b"\x00" * 100, size - 100, size)  # no EOCD in suffix
        with self.assertRaises(IntegrityError):
            parse_bytes(data, suffix_len=60)  # EOCD64 outside the fetched suffix
        tampered = bytearray(data)
        eocd64 = tampered.rfind(b"PK\x06\x06")
        struct.pack_into("<Q", tampered, eocd64 + 32, 7)  # entry count disagrees
        with self.assertRaises(IntegrityError):
            parse_bytes(bytes(tampered))
        plain, _ = build_zip(mixed_members())
        tampered = bytearray(plain)
        eocd = tampered.rfind(b"PK\x05\x06")
        struct.pack_into("<L", tampered, eocd + 16, struct.unpack_from("<L", tampered, eocd + 16)[0] + 1)
        with self.assertRaises(IntegrityError):
            parse_bytes(bytes(tampered))  # CD no longer ends at the EOCD

    def test_trailing_bytes_and_truncated_cd(self):
        data, _ = build_zip(mixed_members())
        end, _entries, cd_bytes = parse_bytes(data)
        with self.assertRaises(IntegrityError):
            parse_central_directory(cd_bytes[:-1], end)
        fake_end = type(end)(end.cd_offset, end.cd_size, end.entry_count - 1, False, end.end_offset)
        with self.assertRaises(IntegrityError):
            parse_central_directory(cd_bytes, fake_end)

    def test_unsafe_names_are_refused(self):
        hostile = ["/etc/passwd", "../evil.jpg", "train/../../evil.jpg", "train\\..\\evil.jpg", "C:/evil.jpg",
                   "C:evil.jpg", "train//evil.jpg", "./evil.jpg", "evil\x00.jpg", "evil\n.jpg",
                   "file.jpg:stream", "train/.."]
        for name in hostile:
            with self.subTest(name=name):
                self.assertIsNotNone(unsafe_name_reason(name))
                data, _ = build_zip([{"name": "ok.jpg", "data": b"x"}, {"name": name.encode("utf-8"), "data": b"y"}])
                with self.assertRaises(IntegrityError):
                    parse_bytes(data)
        for name in ("ISIC_2020_Training_JPEG/train/SYNTH_0015719.jpg", "SYNTH_0015719.jpg", "train/", "a.b/c"):
            with self.subTest(name=name):
                self.assertIsNone(unsafe_name_reason(name))

    def test_duplicate_names_including_case_folding_are_refused(self):
        for second in ("train/SYNTH_0000001.jpg", "TRAIN/synth_0000001.JPG"):
            with self.subTest(second=second):
                data, _ = build_zip([{"name": "train/SYNTH_0000001.jpg", "data": b"a"},
                                     {"name": second, "data": b"b"}])
                with self.assertRaisesRegex(IntegrityError, "duplicate"):
                    parse_bytes(data)

    def test_unsupported_method_and_encryption_are_refused(self):
        data, _ = build_zip([{"name": "a.jpg", "data": b"abc", "method": 0}])
        bzip2 = bytearray(data)
        cd = bzip2.rfind(b"PK\x01\x02")
        struct.pack_into("<H", bzip2, cd + 10, 12)
        with self.assertRaisesRegex(IntegrityError, "unsupported compression"):
            parse_bytes(bytes(bzip2))
        for flag in (0x0001, 0x0040, 0x2000):
            with self.subTest(flag=flag):
                data, _ = build_zip([{"name": "a.jpg", "data": b"abc", "flags": flag}])
                with self.assertRaisesRegex(IntegrityError, "encrypted"):
                    parse_bytes(data)

    def test_second_name_in_unicode_path_extra_must_agree(self):
        def unicode_extra(name):
            body = b"\x01" + struct.pack("<L", zlib.crc32(b"a.jpg")) + name.encode()
            return struct.pack("<HH", 0x7075, len(body)) + body

        data, _ = build_zip([{"name": "a.jpg", "data": b"abc", "extra": unicode_extra("a.jpg")}])
        self.assertEqual(parse_bytes(data)[1][0].name, "a.jpg")
        data, _ = build_zip([{"name": "a.jpg", "data": b"abc", "extra": unicode_extra("../a.jpg")}])
        with self.assertRaisesRegex(IntegrityError, "two different names"):
            parse_bytes(data)

    def test_overlapping_entries_are_refused(self):
        data, _ = build_zip([{"name": "a.jpg", "data": payload(1, 500), "method": 0},
                             {"name": "b.jpg", "data": payload(2, 500), "method": 0}])
        tampered = bytearray(data)
        second_cd = tampered.rfind(b"PK\x01\x02")
        struct.pack_into("<L", tampered, second_cd + 42, 100)  # b.jpg "starts" inside a.jpg
        with self.assertRaisesRegex(IntegrityError, "overlaps"):
            parse_bytes(bytes(tampered))


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spool = Path(self.tmp.name)
        self.members = mixed_members()

    def archive(self, members=None, zip64=True, prefix_len=0):
        data, size = build_zip(members or self.members, zip64=zip64, prefix_len=prefix_len)
        end, entries, _cd = parse_bytes(data, prefix_len)
        return data, size, end, entries

    def run_stream(self, data, end, entries, prefix_len=0, **kwargs):
        results = []
        kwargs.setdefault("spool_cap_bytes", 1 << 20)
        count = stream_members(entries, FileOpener(VirtualObject(data, prefix_len)), cd_offset=end.cd_offset,
                               spool_dir=self.spool, on_member=results.append, **kwargs)
        return count, results

    def assert_results_match(self, results, names):
        expected = {member["name"]: member["data"] for member in self.members}
        self.assertEqual([result.entry.name for result in results], names)
        for result in results:
            original = expected[result.entry.name]
            self.assertEqual(result.path.read_bytes(), original)
            self.assertEqual(result.sha256, hashlib.sha256(original).hexdigest())
            self.assertEqual(result.size, len(original))

    def file_names(self):
        return [member["name"] for member in self.members if not member["name"].endswith("/")]

    def test_stored_deflate_and_descriptors_beyond_4_gib(self):
        data, _size, end, entries = self.archive(prefix_len=5 * GIB)
        count, results = self.run_stream(data, end, entries, prefix_len=5 * GIB, read_bytes=777)
        self.assertEqual(count, 6)
        self.assert_results_match(results, self.file_names())
        self.assertEqual(sorted(os.listdir(self.spool)), sorted(result.path.name for result in results))
        self.assertTrue(all(name.startswith("member-") and name.endswith(".bin") for name in os.listdir(self.spool)))

    def test_stdlib_streaming_zip_round_trip(self):
        sink = UnseekableSink()
        originals = {}
        with zipfile.ZipFile(sink, "w") as archive:
            for index, (method, force) in enumerate(((zipfile.ZIP_STORED, True), (zipfile.ZIP_DEFLATED, True),
                                                     (zipfile.ZIP_DEFLATED, False))):
                info = zipfile.ZipInfo(f"train/ISIC_000000{index}.jpg")
                info.compress_type = method
                originals[info.filename] = payload(index + 10, 8000)
                with archive.open(info, "w", force_zip64=force) as handle:
                    handle.write(originals[info.filename])
        data = bytes(sink.buffer)
        end, entries, _cd = parse_bytes(data)
        _count, results = self.run_stream(data, end, entries)
        self.assertEqual({r.entry.name: r.path.read_bytes() for r in results}, originals)

    def test_resume_from_member_k_reads_nothing_before_it(self):
        data, _size, end, entries = self.archive()
        opener = FileOpener(VirtualObject(data))
        results = []
        stream_members(entries, opener, cd_offset=end.cd_offset, spool_dir=self.spool, spool_cap_bytes=1 << 20,
                       start=3, on_member=results.append)
        self.assertEqual(opener.ranges, [(entries[3].local_header_offset, end.cd_offset - 1)])
        self.assert_results_match(results, [entry.name for entry in entries[3:]])

    def test_tampered_crc_in_cd_is_refused_without_partial_files(self):
        members = mixed_members()
        members[2] = dict(members[2], crc=zlib.crc32(members[2]["data"]) ^ 1)
        data, _size, end, entries = self.archive(members)
        with self.assertRaisesRegex(IntegrityError, "descriptor|CRC32"):
            self.run_stream(data, end, entries)
        self.assertEqual(sorted(os.listdir(self.spool)), ["member-000000.bin"])  # earlier member stays durable
        for zip64 in (False, True):
            with self.subTest(zip64=zip64):
                members = [{"name": "a.jpg", "data": payload(9, 2000), "method": 8,
                            "crc": zlib.crc32(payload(9, 2000)) ^ 0x80}]
                # local header and CD agree on the wrong CRC, so the content check must catch it
                data, _size, end, entries = self.archive(members, zip64=zip64)
                with self.assertRaisesRegex(IntegrityError, "CRC32 mismatch"):
                    self.run_stream(data, end, entries)
        self.assertFalse(any(name.endswith(".part") for name in os.listdir(self.spool)))

    def test_corrupted_payload_bytes_are_refused(self):
        for index in (0, 2):  # stored, deflate
            with self.subTest(index=index):
                data, _size, end, entries = self.archive()
                target = next(entry for entry in entries if entry.name == self.members[index]["name"])
                tampered = bytearray(data)
                spot = target.local_header_offset + 30 + len(target.raw_name) + 20 + target.compressed_size // 2
                tampered[spot] ^= 0xFF
                with self.assertRaises(IntegrityError):
                    self.run_stream(bytes(tampered), end, entries)

    def test_cd_size_smaller_than_inflated_output_is_refused(self):
        data, _size, end, entries = self.archive()
        deflated = entries[2]
        lying = entries[:2] + [type(deflated)(**{**deflated.__dict__, "uncompressed_size": 10})] + entries[3:]
        with self.assertRaisesRegex(IntegrityError, "beyond its CD size"):
            self.run_stream(data, end, lying)

    def test_gap_between_members_is_refused(self):
        data, _size, end, entries = self.archive()
        with self.assertRaisesRegex(IntegrityError, "expected local header"):
            self.run_stream(data, end, entries[:2] + entries[3:])

    def test_spool_cap_pauses_and_resumes_after_drain(self):
        data, _size, end, entries = self.archive()
        with self.assertRaises(SpoolPaused) as caught:
            self.run_stream(data, end, entries, spool_cap_bytes=16000)
        self.assertEqual(caught.exception.next_index, 3)
        self.assertEqual(caught.exception.exit_code, 75)
        self.assertLessEqual(directory_usage(self.spool), 16000)
        for path in self.spool.iterdir():
            path.unlink()
        count, results = self.run_stream(data, end, entries, spool_cap_bytes=16000, start=3)
        self.assertEqual(count, 6)
        self.assert_results_match(results, [entry.name for entry in entries[3:]])
        for path in self.spool.iterdir():
            path.unlink()
        with self.assertRaisesRegex(DivergenceError, "exceeds the spool cap"):
            self.run_stream(data, end, entries, spool_cap_bytes=8000)  # 9000-byte member can never fit

    def test_entries_must_be_offset_ordered(self):
        data, _size, end, entries = self.archive()
        with self.assertRaises(ValueError):
            self.run_stream(data, end, list(reversed(entries)))
        self.assertEqual(self.run_stream(data, end, entries, start=len(entries)), (len(entries), []))


class StreamOverHttpTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.spool = Path(tmp.name)
        self.members = mixed_members()
        self.data, self.size = build_zip(self.members, zip64=True)
        self.server = LocalRangeServer(self.data, etag="ebb879d3-472")
        self.server.__enter__()
        self.addCleanup(self.server.__exit__, None, None, None)
        self.clock = FakeClock()
        self.retry = RetryPolicy(max_attempts=4, base_delay_s=1.0, max_delay_s=8.0, sleep=self.clock.sleep)
        self.client = RangeClient(ObjectSpec(self.server.url, self.size, '"ebb879d3-472"'), timeout_s=5,
                                  retry=self.retry)
        self.client.probe()
        offset, tail = self.client.fetch_suffix(131072)
        self.end, self.entries, _cd = load_central_directory(self.client.fetch, self.size, suffix=(offset, tail))

    def stream(self, **kwargs):
        results = []
        count = stream_with_resume(self.entries, self.client.open_range, retry=self.retry, cd_offset=self.end.cd_offset,
                                   spool_dir=self.spool, spool_cap_bytes=1 << 20, on_member=results.append,
                                   read_bytes=1000, **kwargs)
        return count, results

    def gets(self):
        return [request for request in self.server.state.requests if request[0] == "GET"]

    def test_cut_mid_member_resumes_at_that_member_header(self):
        target = self.entries[3]
        self.server.state.drop_offsets.append(target.local_header_offset + 200)
        before = len(self.gets())
        count, results = self.stream()
        self.assertEqual(count, len(self.entries))
        expected = {member["name"]: member["data"] for member in self.members}
        self.assertEqual({r.entry.name: r.path.read_bytes() for r in results},
                         {name: data for name, data in expected.items() if not name.endswith("/")})
        self.assertEqual(len(results), 5)  # each file member delivered exactly once
        streaming_gets = self.gets()[before:]
        self.assertEqual([request[1] for request in streaming_gets],
                         [f"bytes={self.entries[0].local_header_offset}-{self.end.cd_offset - 1}",
                          f"bytes={target.local_header_offset}-{self.end.cd_offset - 1}"])
        self.assertTrue(all(request[2] == '"ebb879d3-472"' for request in streaming_gets))
        self.assertEqual(self.clock.sleeps, [1.0])

    def test_three_cuts_then_success_within_budget(self):
        self.server.state.drop_offsets.extend(
            [self.entries[0].local_header_offset + 40, self.entries[2].local_header_offset + 10,
             self.entries[5].local_header_offset + 5])
        count, results = self.stream()
        self.assertEqual((count, len(results)), (len(self.entries), 5))
        self.assertEqual(self.clock.sleeps, [1.0, 2.0, 4.0])

    def test_persistent_failures_exhaust_budget_with_resume_point(self):
        self.server.state.drop_offsets.extend([self.entries[2].local_header_offset + 10] * 10)
        with self.assertRaises(RetriesExhausted) as caught:
            self.stream()
        self.assertEqual(caught.exception.next_index, 2)
        self.assertEqual(caught.exception.exit_code, 75)
        self.assertFalse(any(name.endswith(".part") for name in os.listdir(self.spool)))

    def test_object_replaced_between_sessions_is_refused_by_if_range(self):
        self.server.state.drop_offsets.append(self.entries[3].local_header_offset + 100)
        original = self.retry.sleep

        def swap_object(seconds):
            self.server.state.etag = "different-object-473"
            original(seconds)

        self.retry.sleep = swap_object
        with self.assertRaisesRegex(DivergenceError, "200"):
            self.stream()
        self.assertEqual(self.gets()[-1][2], '"ebb879d3-472"')

    def test_server_answering_200_instead_of_206_is_refused(self):
        self.server.state.ignore_range = True
        with self.assertRaises(DivergenceError) as caught:
            self.stream()
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(os.listdir(self.spool), [])

    def test_transient_status_on_open_is_retried(self):
        self.server.state.fail_statuses.extend([503, 502])
        count, results = self.stream()
        self.assertEqual((count, len(results)), (len(self.entries), 5))
        self.assertEqual(self.clock.sleeps, [1.0, 2.0])

    def test_transient_error_carries_next_index_without_wrapper(self):
        self.server.state.drop_offsets.append(self.entries[4].local_header_offset + 1)
        with self.assertRaises(TransientError) as caught:
            stream_members(self.entries, self.client.open_range, cd_offset=self.end.cd_offset,
                           spool_dir=self.spool, spool_cap_bytes=1 << 20)
        self.assertEqual(caught.exception.next_index, 4)


if __name__ == "__main__":
    unittest.main()
