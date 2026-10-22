"""A bare tool name missing from PATH is ``tool_absent`` even when exec reports EACCES.

execvp returns EACCES instead of ENOENT for a bare name that is not found when
some PATH entry cannot be searched (e.g. an unprivileged user inheriting
root's PATH with ``/root/.local/bin``). That is an absent tool, not a denied
one; a tool that exists but cannot be executed stays ``permission_denied``.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from augment_atlas import telemetry_v1 as tv
from test_sentinel import RecordingReaders, SentinelHarness, load

MISSING = "atlas-no-such-nvidia-smi-xyz"


def eacces(argv, *args, **kwargs):
    raise PermissionError(13, "Permission denied", str(argv[0]))


class AbsentOrDeniedTests(unittest.TestCase):
    def test_unresolvable_bare_name_is_absent(self):
        exc = PermissionError(13, "Permission denied", MISSING)
        result = tv.absent_or_denied([MISSING, "-q"], exc)
        self.assertIsInstance(result, FileNotFoundError)
        self.assertEqual("tool_absent", tv.command_reason(result))

    def test_resolvable_bare_name_stays_denied(self):
        name = Path(sys.executable).name
        with mock.patch.dict(os.environ, {"PATH": str(Path(sys.executable).parent)}):
            exc = PermissionError(13, "Permission denied", name)
            self.assertIs(exc, tv.absent_or_denied([name], exc))

    def test_existing_but_not_executable_bare_name_stays_denied(self):
        with tempfile.TemporaryDirectory(prefix="atlas-noexec-") as tmp:
            tool = Path(tmp) / MISSING
            tool.write_text("#!/bin/sh\n", encoding="utf-8")
            tool.chmod(0o644)
            with mock.patch.dict(os.environ, {"PATH": tmp}):
                exc = PermissionError(13, "Permission denied", MISSING)
                self.assertIs(exc, tv.absent_or_denied([MISSING], exc))

    def test_explicit_path_stays_denied(self):
        exc = PermissionError(13, "Permission denied", "/nonexistent/" + MISSING)
        self.assertIs(exc, tv.absent_or_denied(["/nonexistent/" + MISSING], exc))

    def test_run_command_maps_eacces_on_missing_bare_name(self):
        with mock.patch.object(tv.subprocess, "run", side_effect=eacces):
            with self.assertRaises(FileNotFoundError) as caught:
                tv.run_command([MISSING, "--query-gpu=index"], 1.0)
        self.assertIsInstance(caught.exception.__cause__, PermissionError)
        name = Path(sys.executable).name
        with mock.patch.dict(os.environ, {"PATH": str(Path(sys.executable).parent)}), \
                mock.patch.object(tv.subprocess, "run", side_effect=eacces):
            with self.assertRaises(PermissionError):
                tv.run_command([name], 1.0)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root can search any directory")
    def test_real_exec_with_an_unsearchable_path_entry(self):
        with tempfile.TemporaryDirectory(prefix="atlas-eacces-") as tmp:
            locked = Path(tmp) / "locked"
            locked.mkdir()
            locked.chmod(0)
            try:
                path = os.pathsep.join([str(locked), os.environ.get("PATH", os.defpath)])
                with mock.patch.dict(os.environ, {"PATH": path}):
                    with self.assertRaises(FileNotFoundError):
                        tv.run_command([MISSING], 5.0)
            finally:
                locked.chmod(0o700)


class SentinelEaccesTests(SentinelHarness):
    def test_eacces_for_a_missing_tool_keeps_collecting_as_tool_absent(self):
        with mock.patch.object(tv.subprocess, "run", side_effect=eacces):
            _, summary = self.run_sentinel(3.0, readers=RecordingReaders(), popen=eacces, nvidia_smi=[MISSING],
                                           slow_interval_s=1.0)
        records = load(self.out)
        later = records["fast"][2:]
        self.assertTrue(later and all(r["gpus"] is None and r["null_reasons"]["gpus"] == "tool_absent" for r in later))
        events = [e["reason"] for e in records["events"] if e["event"] == "gpu_stream_gap"]
        self.assertEqual(["tool_absent"], events)
        self.assertEqual({"tool_absent"}, {r["null_reasons"]["gpus"] for r in records["slow"]})
        self.assertGreaterEqual(summary["fast_samples"], 5)

    def test_eacces_for_an_existing_tool_is_permission_denied(self):
        _, _ = self.run_sentinel(2.0, readers=RecordingReaders(), popen=eacces, nvidia_smi=[sys.executable],
                                 slow_interval_s=1.0)
        events = [e["reason"] for e in load(self.out)["events"] if e["event"] == "gpu_stream_gap"]
        self.assertEqual(["permission_denied"], events)


if __name__ == "__main__":
    unittest.main()
