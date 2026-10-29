"""Synthetic queue-branch snapshots; no network, Git writes or training data."""
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import queue_board as board
from augment_atlas.queue import ANCHOR_CAVEAT, render_record
from augment_atlas.schema import digest, new_record


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
COMMIT = "c" * 40
CAMPAIGN = "a" * 64
FREEZE = "f" * 64
POOLS = "1" * 64


def anchor(campaign=CAMPAIGN, freeze=FREEZE, source="hpo_d", dataset="synthetic-alpha"):
    result = {
        "schema_version": 1, "protocol": "global-anchor-v1", "kind": "campaign",
        "dataset": {"id": dataset, "version": dataset + "-v1"},
        "pools_sha256": POOLS, "document_sha256": campaign,
        "freeze_sha256": freeze, "campaign_sha256": campaign,
        "recipe_source": source, "code_commit": COMMIT, "label": "exploratory",
        "document_utc": "2026-10-29T09:00:00Z", "caveat": ANCHOR_CAVEAT,
    }
    result["sha256"] = digest(result)
    return result


def job(state="pending", *, phase="final", seed=11, fold=0, arm="basic",
        dataset="synthetic-alpha", campaign=CAMPAIGN, freeze=FREEZE,
        last="2026-10-29T11:55:00Z", end="2026-10-29T11:30:00Z",
        gpu="0", uuid="GPU-synthetic-0", reason=None):
    spec = {
        "protocol_version": "synthetic-" + phase, "dataset": dataset,
        "dataset_version": dataset + "-v1", "split_hash": "2" * 64,
        "recipe_hash": freeze or "3" * 64, "phase": phase, "fold": fold,
        "train_seed": seed, "arm": arm, "code_commit": COMMIT,
    }
    if phase != "smoke":
        spec["global"] = {
            "campaign_sha256": campaign, "freeze_sha256": freeze,
            "pools_sha256": POOLS, "plan_sha256": "4" * 64 if phase == "hpo" else None,
            "freeze_kind": None if phase == "hpo" else "hpo_d",
        }
    record = new_record(spec)
    if state == "pending":
        return record
    record.update(state=state, attempt_id="synthetic-attempt", token="synthetic-public-guard",
                  owner={"server": "synthetic-node", "gpu": gpu, "gpu_uuid": uuid})
    events = [{"kind": "claim", "state": "claimed", "at": "2026-10-29T10:59:00Z"}]
    if state != "claimed":
        events.append({"kind": "update", "state": "running", "at": "2026-10-29T11:00:02Z",
                       "observed_at_utc": "2026-10-29T11:00:00Z", "epoch": 0, "epochs": 20})
        events.append({"kind": "update", "state": state, "at": "2026-10-29T11:59:59Z",
                       "observed_at_utc": end if state == "succeeded" else last,
                       "epoch": 20 if state == "succeeded" else 3, "epochs": 20})
        if reason:
            events[-1]["cause"] = reason
    for seq, event in enumerate(events, 1):
        event.update(seq=seq, attempt_id=record["attempt_id"])
    record.update(history=events, seq=len(events), version=len(events))
    return record


def synthetic_snapshot():
    anchors = [anchor(), anchor("b" * 64, "e" * 64, "external_fixed", "synthetic-beta")]
    records = [
        job("succeeded"),
        job("succeeded", phase="hpo", seed=7, campaign=None, freeze=None, end="2026-10-29T11:10:00Z"),
        job("running", arm="running-fresh"),
        job("running", arm="running-boundary", gpu="1", uuid="GPU-synthetic-1", last="2026-10-29T11:30:00Z"),
        job("running", arm="running-stale", gpu="2", uuid="GPU-synthetic-2", last="2026-10-29T11:29:59Z"),
        job("running", arm="running-same-gpu", gpu="GPU-synthetic-0"),
        *[job(arm="pending-" + str(i), seed=23 if i == 3 else 11) for i in range(4)],
        job("failed", arm="failure", reason="gpu_error | synthetic\nfailure"),
        job("interrupted", arm="interruption", reason={"code": "thermal_stop"}),
        job("cancelled", arm="cancelled"),
        job("claimed", arm="claimed"),
        job("succeeded", dataset="synthetic-beta", campaign="b" * 64, freeze="e" * 64,
            end="2026-10-29T11:01:00Z"),
        job(dataset="synthetic-beta", campaign="b" * 64, freeze="e" * 64, arm="pending-beta"),
        job(phase="smoke", dataset="synthetic-tiny"),
    ]
    return records, anchors


class SyntheticBranch:
    """Read-only Git command double for committed refs/heads/queue and blobs.

    Commands outside the explicitly allowed reads fail, so tests cannot mutate
    any real repository or accidentally fetch data. Dirty checkout files and
    unrelated tree entries are deliberately present and must not be read.
    """
    def __init__(self, records=None, anchors=None, remote=False):
        self.records, self.anchors = synthetic_snapshot() if records is None else (records, anchors or [])
        self.remote = remote
        self.calls = []
        self.blobs = {}
        self.entries = []
        for record in self.records:
            self.add("jobs/" + record["job_id"] + ".md", render_record(record))
        for item in self.anchors:
            self.add(f"anchors/{item['dataset']['id']}/campaign-{item['document_sha256']}.json", json.dumps(item))
        self.entries += ["100644 blob " + "9" * 40 + "\tjobs/README.md",
                         "100644 blob " + "8" * 40 + "\tanchors/unrelated/data.csv"]

    def add(self, path, content, mode="100644"):
        blob = f"{len(self.blobs) + 1:040x}"
        self.blobs[blob] = content
        self.entries.append(f"{mode} blob {blob}\t{path}")

    def run(self, args, **kwargs):
        self.calls.append(args)
        command = args[3:]
        result = subprocess.CompletedProcess(args, 0, "", "")
        if command[0] == "rev-parse":
            if command[-1] == "refs/remotes/origin/queue^{commit}" and not self.remote:
                result.returncode = 1
            else:
                assert command[-1] in ("refs/remotes/origin/queue^{commit}", "refs/heads/queue^{commit}")
                result.stdout = COMMIT + "\n"
        elif command[0] == "ls-tree":
            assert command == ["ls-tree", "-rz", COMMIT, "--", "jobs/", "anchors/"]
            result.stdout = "\0".join(reversed(self.entries)) + "\0"
        elif command[:2] == ["cat-file", "blob"]:
            result.stdout = self.blobs[command[2]]
        else:
            raise AssertionError("unexpected or mutating Git command: " + repr(command))
        return result


def publisher_module():
    script = (ROOT / "scripts/campaign/publish_board.sh").read_text(encoding="utf-8")
    source = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    namespace = {"__name__": "publisher_fixture"}
    exec(compile(source, "publish_board.sh:python", "exec"), namespace)
    return namespace


class BoardTests(unittest.TestCase):
    def test_counts_eta_stale_and_golden_from_synthetic_queue_branch(self):
        fixture = SyntheticBranch()
        with patch.object(board.subprocess, "run", side_effect=fixture.run):
            commit, records, anchors = board.read_snapshot(ROOT)
        content = board.render_board(commit, records, anchors, NOW)
        self.assertIn("Total jobs: 17", content)
        self.assertIn("| 6 | 1 | 4 | 3 | 1 | 1 | 1 |", content)
        self.assertIn("| synthetic-alpha | synthetic-alpha-v1 | campaign:" + CAMPAIGN +
                      " | 2 | 00:20:00 | 4 | 2 | 00:40:00 estimate |", content)
        self.assertIn("00:30:01 stale", content)
        self.assertIn("2026-10-29T11:30:00Z | 00:30:00 |", content)
        self.assertIn("gpu_error &#124; synthetic failure", content)
        self.assertIn("thermal_stop", content)
        self.assertIn("unavailable estimate", content)
        self.assertEqual(content, (ROOT / "tests/fixtures/queue_board.md").read_text(encoding="utf-8"))

    def test_cli_ignores_dirty_checkout_and_preserves_queue_files(self):
        fixture = SyntheticBranch(remote=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            checkout = base / "queue"
            (checkout / "jobs").mkdir(parents=True)
            dirty = checkout / "jobs/dirty.md"
            dirty.write_text("do not read this uncommitted working file", encoding="utf-8")
            output = base / "BOARD.md"
            with patch.object(board.subprocess, "run", side_effect=fixture.run):
                self.assertEqual(board.main(["--queue-checkout", str(checkout), "--out", str(output),
                                             "--now", "2026-10-29T12:00:00Z"]), 0)
            self.assertEqual(output.read_text(encoding="utf-8"),
                             (ROOT / "tests/fixtures/queue_board.md").read_text(encoding="utf-8"))
            self.assertEqual(dirty.read_text(), "do not read this uncommitted working file")
            self.assertEqual(len([a for a in fixture.calls if "rev-parse" in a]), 1)

    def test_ref_required_no_head_fallback(self):
        with patch.object(board, "_git", return_value=subprocess.CompletedProcess([], 1, "", "")):
            with self.assertRaisesRegex(ValueError, "queue branch"):
                board.read_snapshot(ROOT)

    def test_malformed_record_fails_without_printing_payload(self):
        fixture = SyntheticBranch()
        fixture.blobs[next(iter(fixture.blobs))] = "unrelated non-record text"
        with patch.object(board.subprocess, "run", side_effect=fixture.run):
            with self.assertRaisesRegex(ValueError, "invalid queue document") as error:
                board.read_snapshot(ROOT)
        self.assertNotIn("unrelated", str(error.exception))

    def test_rejects_symlink_job_without_following_it(self):
        fixture = SyntheticBranch([], [])
        fixture.add("jobs/" + "f" * 64 + ".md", "../private", mode="120000")
        with patch.object(board.subprocess, "run", side_effect=fixture.run):
            with self.assertRaisesRegex(ValueError, "non-regular"):
                board.read_snapshot(ROOT)
        self.assertFalse(any("cat-file" in call for call in fixture.calls))

    def test_no_output_inside_queue_and_requires_explicit_utc(self):
        for now in ("invalid", "2026-10-29T12:00:00", "2026-10-29T12:00:00+01:00"):
            with self.subTest(now=now), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                board.main(["--queue-checkout", str(ROOT), "--out", str(ROOT.parent / "BOARD.md"), "--now", now])
            self.assertEqual(error.exception.code, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            board.main(["--queue-checkout", str(ROOT), "--out", str(ROOT / "BOARD.md")])
        self.assertEqual(error.exception.code, 2)

    def test_empty_and_order_independent(self):
        content = board.render_board(COMMIT, [], [], NOW)
        self.assertIn("Total jobs: 0", content)
        self.assertIn("No running jobs.", content)
        records, anchors = synthetic_snapshot()
        self.assertEqual(board.render_board(COMMIT, records, anchors, NOW),
                         board.render_board(COMMIT, records[::-1], anchors[::-1], NOW))

    def test_campaign_freeze_anchor_ambiguity_and_legacy_fallback(self):
        records, anchors = synthetic_snapshot()
        spec = copy.deepcopy(records[0]["spec"])
        spec["global"]["campaign_sha256"] = None
        self.assertEqual(board._campaign(spec, anchors), "campaign:" + CAMPAIGN)
        other = anchor("d" * 64)
        self.assertEqual(board._campaign(spec, anchors + [other]), "freeze:" + FREEZE)
        hpo = copy.deepcopy(records[1]["spec"])
        self.assertEqual(board._campaign(hpo, anchors), "campaign:" + CAMPAIGN)
        self.assertEqual(board._campaign(hpo, anchors + [other]), "plan:" + "4" * 64)
        self.assertEqual(board._campaign(records[-1]["spec"], anchors), "recipe:" + FREEZE)

    def test_retry_history_preserved_but_not_used_for_current_duration_or_progress(self):
        record = job("succeeded")
        old = [{"kind": "update", "state": "running", "at": "2026-10-28T01:00:00Z",
                "attempt_id": "old", "epoch": 99},
               {"kind": "update", "state": "failed", "at": "2026-10-28T02:00:00Z",
                "attempt_id": "old", "cause": "old_failure"}]
        record["history"] = old + record["history"]
        for seq, event in enumerate(record["history"], 1):
            event["seq"] = seq
        record["version"] = record["seq"] = len(record["history"])
        self.assertEqual(board._duration(record), 1800)
        self.assertIn("old_failure", board.render_board(COMMIT, [record], [], NOW))

    def test_observation_time_not_publication_and_bad_times_remove_capacity(self):
        for stamp, label in (("invalid", "unknown"), ("2026-10-29T12:01:00Z", "future"),
                             ("2026-10-29T11:29:00Z", "00:31:00 stale")):
            records = [job("succeeded"), job("running", arm="active", last=stamp), job(arm="pending")]
            content = board.render_board(COMMIT, records, [], NOW)
            self.assertIn(label, content)
            self.assertIn("| 1 | 00:30:00 | 1 | 0 | unavailable estimate |", content)
        legacy = job("running")
        del legacy["history"][-1]["observed_at_utc"]
        self.assertIn("2026-10-29T11:59:59Z | 00:00:01 |", board.render_board(COMMIT, [legacy], [], NOW))

    def test_legacy_epoch_total_and_missing_timing(self):
        record = job("running")
        record["history"][-1].pop("epochs")
        record["history"][-1]["total_epochs"] = 12
        self.assertIn("| 3/12 |", board.render_board(COMMIT, [record], [], NOW))
        record["history"][-1].pop("total_epochs")
        record["history"][1]["epochs"] = 12
        self.assertIn("| 3/12 |", board.render_board(COMMIT, [record], [], NOW))
        success = job("succeeded")
        success["history"][1]["observed_at_utc"] = "invalid"
        self.assertIsNone(board._duration(success))

    def test_recent_failure_limit_and_reason_escaping(self):
        records = [job("failed", arm=f"arm-{i}", reason="<script>|" + "x" * 200) for i in range(25)]
        content = board.render_board(COMMIT, records, [], NOW)
        recent = content.split("## Recent failures / interruptions")[1].split("## Campaign ETA")[0]
        self.assertEqual(recent.count("&lt;script&gt;&#124;"), 20)
        self.assertNotIn("x" * 200, content)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.module = publisher_module()
        self.calls = []

    def fake_git(self, clone, *args, **kwargs):
        self.calls.append((args, kwargs))
        stdout, status = "d" * 40 + "\n", 0
        if args[0] == "ls-remote":
            status = getattr(self, "remote_status", 0)
        if args[:2] == ("rev-parse", "--is-bare-repository"):
            stdout = "true\n"
        elif args[:3] == ("remote", "get-url", "origin"):
            stdout = "git@synthetic:queue.git\n"
        elif args[0] == "rev-parse" and args[-1].endswith("^{tree}"):
            stdout = "e" * 40 + "\n"
        return subprocess.CompletedProcess(args, status, stdout, "")

    def publish(self, directory):
        records, anchors = synthetic_snapshot()
        with patch.dict(self.module, git=self.fake_git,
                        read_snapshot=lambda _: (COMMIT, records, anchors)), contextlib.redirect_stdout(io.StringIO()):
            self.module["publish"](Path(directory), Path(directory) / "BOARD.md", {"GIT_SSH": "synthetic-ssh"})

    def test_publish_only_board_direct_child_and_board_only_tree(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            self.publish(temporary)
            self.assertIn("Total jobs: 17", (Path(temporary) / "BOARD.md").read_text(encoding="utf-8"))
        pushes = [args for args, _ in self.calls if args[0] == "push"]
        self.assertEqual(pushes, [("push", "origin", "d" * 40 + ":refs/heads/board")])
        commits = [args for args, _ in self.calls if args[0] == "commit-tree"]
        self.assertEqual(commits, [("commit-tree", "d" * 40, "-p", "d" * 40)])
        tree_input = next(kw["input"] for args, kw in self.calls if args[0] == "mktree")
        self.assertEqual(tree_input, "100644 blob " + "d" * 40 + "\tBOARD.md\n")
        self.assertFalse(any("force" in arg or arg.startswith("+") for args, _ in self.calls for arg in args))
        self.assertTrue(all(kw["env"]["GIT_SSH"] == "synthetic-ssh" for _, kw in self.calls))

    def test_missing_board_bootstraps_independent_root(self):
        self.remote_status = 2
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            self.publish(temporary)
        commit = next(args for args, _ in self.calls if args[0] == "commit-tree")
        self.assertNotIn("-p", commit)

    def test_remote_error_never_bootstraps_or_pushes(self):
        self.remote_status = 128
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary, self.assertRaises(RuntimeError):
            self.publish(temporary)
        self.assertFalse(any(args[0] in ("commit-tree", "push") for args, _ in self.calls))

    def test_push_rejection_is_not_retried_or_forced(self):
        original = self.fake_git
        def reject(clone, *args, **kwargs):
            result = original(clone, *args, **kwargs)
            if args[0] == "push":
                raise subprocess.CalledProcessError(1, args)
            return result
        records, anchors = synthetic_snapshot()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary, patch.dict(
                self.module, git=reject, read_snapshot=lambda _: (COMMIT, records, anchors)), self.assertRaises(subprocess.CalledProcessError):
            self.module["publish"](Path(temporary), Path(temporary) / "BOARD.md", {})
        self.assertEqual(sum(args[0] == "push" for args, _ in self.calls), 1)

    def test_bash_wrapper_syntax(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable; wrapper syntax requires server validation")
        result = subprocess.run([bash, "-n", str(ROOT / "scripts/campaign/publish_board.sh")],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_single_writer_stop_and_restart(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            state = Path(temporary)
            stop, lock = state / "board.stop", state / "publisher.lock"
            args = ["--repo", "git@synthetic:queue.git", "--state-dir", temporary, "--once"]
            lock.write_text("synthetic-owner", encoding="ascii")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.module["main"](args), 2)
            lock.unlink()
            stop.touch()
            self.assertEqual(self.module["main"](args), 0)
            self.assertFalse(lock.exists())
            stop.unlink()
            publications = []
            with patch.dict(self.module, git=self.fake_git, publish=lambda *a: publications.append(a)):
                self.assertEqual(self.module["main"](args), 0)
                self.assertEqual(self.module["main"](args), 0)
            self.assertEqual(len(publications), 2)
            self.assertFalse(lock.exists())

    def test_interval_and_stop_during_wait(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            stop = Path(temporary) / "board.stop"
            args = ["--repo", "git@synthetic:queue.git", "--state-dir", temporary, "--interval", "0.1"]
            with patch.dict(self.module, git=self.fake_git, publish=lambda *a: None), patch.object(
                    self.module["time"], "sleep", side_effect=lambda _: stop.touch()):
                self.assertEqual(self.module["main"](args), 0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.module["main"]([*args, "--interval", "nan"])


if __name__ == "__main__":
    unittest.main()
