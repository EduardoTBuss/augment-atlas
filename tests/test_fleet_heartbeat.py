"""F5 part B: heartbeat ref, derived liveness and fleet documents (docs/fleet-protocol.md 4.2-4.4, 6.3).

Cases 8.3-10 (heartbeat), 8.3-11 (``liveness()``) and 8.3-17 (``put_document``). Real Git:
one bare remote per test and independent clones, as in ``tests/test_queue.py``.
"""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from augment_atlas import fleet
from augment_atlas import heartbeat as hbm
from augment_atlas import node_profile as npf
from augment_atlas.heartbeat import HeartbeatPublisher, fetch_heartbeats, read_heartbeat
from augment_atlas.queue import GitQueue, QueueConflict
from augment_atlas.schema import canonical_json

try:  # ``discover -s tests`` puts tests/ on sys.path; ``-m unittest tests.x`` uses the package path.
    from test_fleet import BOOT_ID, fleet_spec, legacy_spec, x299_profile
    from test_fleet_queue import GitFixture, fleet_block, owner_for
except ImportError:  # pragma: no cover - alternative runner layout
    from tests.test_fleet import BOOT_ID, fleet_spec, legacy_spec, x299_profile
    from tests.test_fleet_queue import GitFixture, fleet_block, owner_for

SYNTHETIC = Path(__file__).parent / "fixtures" / "telemetry_v1" / "synthetic"
REF = "refs/heads/fleet-hb/x299/gpu0"
OTHER_BOOT = "bbbbbbbb-0000-4000-8000-00000000000b"
T0 = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
A = "a" * 32


def utc_z(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def hb_doc(node_id="x299", worker="gpu0", seq=1, at=T0, boot_id=BOOT_ID, ntp=True, status="running"):
    doc = json.loads((SYNTHETIC / "heartbeat-alive.json").read_text(encoding="utf-8"))
    doc.update(node_id=node_id, worker=worker, seq=seq, observed_at_utc=utc_z(at), boot_id=boot_id,
               ntp_synchronized=ntp, status=status)
    return doc


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


def remote_refs(remote):
    return git("--git-dir", str(remote), "for-each-ref", "--format=%(refname) %(objectname)").decode().splitlines()


class HeartbeatFixture(GitFixture):
    def setUp(self):
        super().setUp()
        self.bare = self.remote()
        self.journal = self.root / "state" / "liveness" / "x299-gpu0.jsonl"

    def publisher(self, ref=REF, name="hb", remote=None, **kwargs):
        return HeartbeatPublisher(self.root / name, remote or self.bare, ref, journal=self.journal, **kwargs)

    def tip(self, ref=REF):
        result = subprocess.run(["git", "--git-dir", str(self.bare), "rev-parse", "--verify", "-q", ref],
                                capture_output=True, text=True)
        return result.stdout.strip() or None

    def journal_lines(self):
        if not self.journal.exists():
            return []
        return [json.loads(line) for line in self.journal.read_text(encoding="utf-8").splitlines()]


class HeartbeatPublisherTests(HeartbeatFixture):
    """Case 8.3-10: single writer, fast-forward, increasing seq, best effort."""

    def test_publish_writes_only_heartbeat_json_in_own_ref(self):
        doc = hb_doc(seq=1)
        result = self.publisher().publish(doc)
        self.assertTrue(result.ok, result)
        self.assertEqual("published", result.reason)
        self.assertEqual(result.commit, self.tip())
        self.assertEqual(hbm.encode(doc), git("--git-dir", str(self.bare), "show", f"{REF}:heartbeat.json"))
        self.assertEqual([b"heartbeat.json"], git("--git-dir", str(self.bare), "ls-tree", "--name-only", REF).split())
        self.assertEqual([f"{REF} {result.commit}"], remote_refs(self.bare))
        self.assertEqual([], self.journal_lines())

    def test_republishing_the_same_document_is_idempotent(self):
        publisher = self.publisher()
        first = publisher.publish(hb_doc(seq=5))
        again = publisher.publish(hb_doc(seq=5))
        fresh_checkout = self.publisher(name="hb-restarted").publish(hb_doc(seq=5))
        self.assertEqual(("already_published", first.commit), (again.reason, again.commit))
        self.assertEqual(("already_published", first.commit), (fresh_checkout.reason, fresh_checkout.commit))
        self.assertTrue(again.ok and fresh_checkout.ok)
        self.assertEqual(first.commit, self.tip())
        self.assertEqual(b"1", git("--git-dir", str(self.bare), "rev-list", "--count", REF).strip())

    def test_each_heartbeat_is_a_fast_forward_child_with_increasing_seq(self):
        publisher = self.publisher()
        commits = [publisher.publish(hb_doc(seq=seq, at=T0 + timedelta(seconds=900 * seq))).commit
                   for seq in (1, 2, 7)]
        self.assertEqual(commits[-1], self.tip())
        for parent, child in zip(commits, commits[1:]):
            self.assertEqual(parent.encode(), git("--git-dir", str(self.bare), "rev-parse", f"{child}^").strip())
        before = self.tip()
        for stale_seq in (7, 6, 0):
            result = publisher.publish(hb_doc(seq=stale_seq, status="idle"))
            self.assertEqual((False, "seq_not_increasing"), (result.ok, result.reason))
        self.assertEqual(before, self.tip())
        lines = self.journal_lines()
        self.assertEqual(3, len(lines))
        self.assertTrue(all(line["git_ok_last"] is False and line["reason"] == "seq_not_increasing" for line in lines))

    def test_heartbeat_never_moves_the_queue_branch_nor_any_job(self):
        queue = GitQueue(self.root / "queue", self.bare)
        job_id = queue.seed([fleet_spec()])[0]["job_id"]
        claimed = queue.claim(job_id, owner_for("x299"), fleet_block(fleet_spec()))
        queue_head, queue_tree = self.remote_head(self.bare), self.remote_tree(self.bare)
        job_bytes = self.remote_blob(self.bare, job_id)
        publisher = HeartbeatPublisher(Path(str(queue.checkout) + "-hb"), self.bare, REF)
        for seq in range(1, 4):
            self.assertTrue(publisher.publish(hb_doc(seq=seq)).ok)
        self.assertEqual(queue_head, self.remote_head(self.bare))
        self.assertEqual(queue_tree, self.remote_tree(self.bare))
        self.assertEqual(job_bytes, self.remote_blob(self.bare, job_id))
        self.assertEqual(claimed, queue.list_jobs()[0])
        self.assertNotIn(b"jobs", git("--git-dir", str(self.bare), "ls-tree", "-r", "--name-only", REF))
        # The queue checkout itself is untouched (separate <checkout>-hb, 4.2).
        self.assertFalse(subprocess.run(["git", "-C", str(queue.checkout), "rev-parse", "--verify", "-q", REF],
                                        capture_output=True).stdout)

    def test_refs_outside_the_worker_pattern_are_refused_before_any_io(self):
        for ref in ("refs/heads/queue", "refs/heads/main", "refs/heads/fleet-hb/x299",
                    "refs/heads/fleet-hb/x299/gpu0/extra", "refs/heads/fleet-hb/X299/gpu0",
                    "refs/heads/fleet-hb/x299/../queue", "refs/heads/fleet-hb//gpu0", "refs/heads/fleet-hb/x299/-gpu",
                    "fleet-hb/x299/gpu0", "refs/tags/fleet-hb/x299/gpu0", None, 7):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                HeartbeatPublisher(self.root / "never", self.bare, ref)
        self.assertFalse((self.root / "never").exists())
        self.assertEqual([], remote_refs(self.bare))

    def test_document_of_another_worker_or_node_is_refused_without_push(self):
        publisher = self.publisher()
        for doc in (hb_doc(worker="gpu1"), hb_doc(node_id="sim-b")):
            with self.subTest(doc=(doc["node_id"], doc["worker"])), self.assertRaises(ValueError):
                publisher.publish(doc)
        bad = hb_doc()
        bad["status"] = "crashed"
        with self.assertRaises(ValueError):
            publisher.publish(bad)
        self.assertEqual([], remote_refs(self.bare))
        self.assertFalse((self.root / "hb").exists())

    def test_git_unreachable_returns_failure_without_raising(self):
        missing = self.root / "does-not-exist.git"
        result = self.publisher(remote=missing).publish(hb_doc(seq=3))
        self.assertEqual((False, "git_unavailable", True), (result.ok, result.reason, result.journal_ok))
        line, = self.journal_lines()
        self.assertEqual({"git_ok_last": False, "event": "heartbeat_publish_failed", "reason": "git_unavailable",
                          "seq": 3, "boot_id": BOOT_ID, "status": "running"},
                         {key: line[key] for key in ("git_ok_last", "event", "reason", "seq", "boot_id", "status")})
        self.assertTrue(line["utc"].endswith("Z"))

    def test_git_binary_missing_or_timeout_never_raises(self):
        for error in (FileNotFoundError("git"), subprocess.TimeoutExpired(["git"], 60), PermissionError("git")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(hbm.subprocess, "run", side_effect=error):
                    result = self.publisher().publish(hb_doc(seq=1))
                self.assertEqual((False, "git_unavailable"), (result.ok, result.reason))
        self.assertEqual(3, len(self.journal_lines()))
        self.assertEqual([], remote_refs(self.bare))

    def test_journal_failure_is_reported_not_raised(self):
        self.journal.mkdir(parents=True)  # a directory where the journal file should be
        result = self.publisher(remote=self.root / "missing.git").publish(hb_doc())
        self.assertEqual((False, "git_unavailable", False), (result.ok, result.reason, result.journal_ok))

    def test_journal_line_is_fsynced(self):
        with mock.patch.object(hbm.os, "fsync") as fsync:
            self.publisher(remote=self.root / "missing.git").publish(hb_doc())
        self.assertEqual(1, fsync.call_count)

    def test_rejected_push_is_reported_and_tip_unchanged(self):
        publisher = self.publisher()
        first = publisher.publish(hb_doc(seq=1))
        hook = self.bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8", newline="\n")
        hook.chmod(0o755)
        result = publisher.publish(hb_doc(seq=2))
        self.assertEqual((False, "push_not_confirmed"), (result.ok, result.reason))
        self.assertEqual(first.commit, self.tip())
        self.assertEqual("push_not_confirmed", self.journal_lines()[-1]["reason"])

    def test_lost_push_response_is_decided_by_refetch(self):
        publisher = self.publisher()
        original = hbm._Git.run

        def push_then_lose_response(git_self, *args, **kwargs):
            result = original(git_self, *args, **kwargs)
            if args and args[0] == "push":
                raise hbm.HeartbeatGitError("response lost")
            return result

        with mock.patch.object(hbm._Git, "run", push_then_lose_response):
            result = publisher.publish(hb_doc(seq=1))
        self.assertEqual((True, "published"), (result.ok, result.reason))
        self.assertEqual(result.commit, self.tip())

    def test_tip_written_by_someone_else_is_never_overwritten(self):
        foreign = self.root / "foreign"
        git("init", "-q", str(foreign))
        (foreign / "heartbeat.json").write_bytes(hbm.encode(hb_doc(worker="gpu1", seq=99)))
        git("-C", str(foreign), "add", "heartbeat.json")
        git("-C", str(foreign), "-c", "user.name=x", "-c", "user.email=x@example.invalid", "commit", "-qm", "rogue")
        git("-C", str(foreign), "push", "-q", str(self.bare), f"HEAD:{REF}")
        before = self.tip()
        result = self.publisher().publish(hb_doc(seq=100))
        self.assertEqual((False, "foreign_tip"), (result.ok, result.reason))
        self.assertEqual(before, self.tip())


class HeartbeatReaderTests(HeartbeatFixture):
    def test_fetch_heartbeats_returns_every_valid_worker_ref(self):
        for node_id, worker in (("x299", "gpu0"), ("x299", "gpu1"), ("sim-b", "gpu0")):
            ref = hbm.heartbeat_ref(node_id, worker)
            self.assertTrue(self.publisher(ref, name=f"hb-{node_id}-{worker}").publish(
                hb_doc(node_id, worker, seq=4)).ok)
        snapshot = fetch_heartbeats(self.root / "observer", self.bare)
        self.assertTrue(snapshot.fetch_ok)
        self.assertEqual(["refs/heads/fleet-hb/sim-b/gpu0", "refs/heads/fleet-hb/x299/gpu0",
                          "refs/heads/fleet-hb/x299/gpu1"], sorted(snapshot.heartbeats))
        self.assertEqual(hb_doc("sim-b", "gpu0", seq=4), snapshot.heartbeats["refs/heads/fleet-hb/sim-b/gpu0"])
        self.assertEqual({}, snapshot.invalid)
        self.assertEqual((True, hb_doc(seq=4)), read_heartbeat(self.root / "observer", self.bare, REF))

    def test_misplaced_or_invalid_documents_are_listed_not_trusted(self):
        rogue = self.root / "rogue"
        git("init", "-q", str(rogue))
        (rogue / "heartbeat.json").write_bytes(hbm.encode(hb_doc(worker="gpu1")))
        git("-C", str(rogue), "add", "heartbeat.json")
        git("-C", str(rogue), "-c", "user.name=x", "-c", "user.email=x@example.invalid", "commit", "-qm", "rogue")
        git("-C", str(rogue), "push", "-q", str(self.bare), f"HEAD:{REF}")
        git("-C", str(rogue), "push", "-q", str(self.bare), "HEAD:refs/heads/fleet-hb/BAD/gpu0")
        (rogue / "heartbeat.json").write_text("{not json", encoding="utf-8")
        git("-C", str(rogue), "-c", "user.name=x", "-c", "user.email=x@example.invalid", "commit", "-qam", "bad")
        git("-C", str(rogue), "push", "-q", str(self.bare), "HEAD:refs/heads/fleet-hb/sim-b/gpu0")
        snapshot = fetch_heartbeats(self.root / "observer", self.bare)
        self.assertTrue(snapshot.fetch_ok)
        self.assertEqual({}, snapshot.heartbeats)
        self.assertEqual({REF: "document_of_another_worker", "refs/heads/fleet-hb/BAD/gpu0": "ref_pattern",
                          "refs/heads/fleet-hb/sim-b/gpu0": "invalid_document"}, snapshot.invalid)
        self.assertEqual((True, None), read_heartbeat(self.root / "observer", self.bare, REF))

    def test_unreachable_observer_reports_fetch_failure_without_raising(self):
        snapshot = fetch_heartbeats(self.root / "observer", self.root / "missing.git")
        self.assertEqual((False, {}), (snapshot.fetch_ok, snapshot.heartbeats))
        self.assertEqual((False, None), read_heartbeat(self.root / "observer", self.root / "missing.git", REF))
        self.assertEqual((True, None), read_heartbeat(self.root / "observer", self.bare, REF))
        with self.assertRaises(ValueError):
            read_heartbeat(self.root / "observer", self.bare, "refs/heads/queue")


def claim_record(state="running", fleet_block_=True, lease=3600, boot_id=BOOT_ID, claim_at=T0):
    event = {"kind": "claim", "state": "claimed", "attempt_id": A, "token": "b" * 32, "seq": 2,
             "owner": owner_for("x299"), "at": claim_at.isoformat()}
    if fleet_block_:
        event["fleet"] = {"heartbeat_ref": REF, "lease_seconds": lease, "boot_id": boot_id}
    history = [{"kind": "seed", "state": "pending", "seq": 1, "at": (claim_at - timedelta(hours=1)).isoformat()}, event]
    if state == "running":
        history.append({"kind": "update", "state": "running", "attempt_id": A, "seq": 3, "event_id": "e1",
                        "observed_at_utc": utc_z(claim_at + timedelta(seconds=30)),
                        "at": (claim_at + timedelta(seconds=31)).isoformat()})
    return {"job_id": "c" * 64, "state": state, "attempt_id": A, "token": "b" * 32, "owner": owner_for("x299"),
            "version": len(history), "seq": len(history), "history": history}


class LivenessTests(unittest.TestCase):
    """Case 8.3-11: every derived state, exact boundaries, purity."""

    HB_AT = T0 + timedelta(seconds=60)

    def state(self, age_s, record=None, hb=None, others=None, **kwargs):
        record = record or claim_record()
        hbs = {REF: hb if hb is not None else hb_doc(at=self.HB_AT)}
        if others:
            hbs.update(others)
        return fleet.liveness(record, hbs, None, self.HB_AT + timedelta(seconds=age_s), **kwargs)

    def test_heartbeat_interval_boundaries_899_900_901_are_alive(self):
        for age in (0, 899, 900, 901, 1800):
            with self.subTest(age=age):
                result = self.state(age)
                self.assertEqual(("alive", age), (result["state"], result["age_s"]))
                self.assertNotIn("clock_untrusted", result["evidence"])

    def test_late_after_two_heartbeats_until_lease_inclusive(self):
        for age in (1801, 3599, 3600):
            with self.subTest(age=age):
                self.assertEqual("late", self.state(age)["state"])

    def test_suspect_stale_strictly_after_lease(self):
        result = self.state(3601)
        self.assertEqual(("suspect_stale", 3601), (result["state"], result["age_s"]))
        self.assertEqual({"heartbeat_ref": REF, "lease_seconds": 3600, "heartbeat_seconds": 900, "seq": 1,
                          "heartbeat_utc": utc_z(self.HB_AT), "heartbeat_predates_claim": False,
                          "other_node_workers": 0, "other_node_workers_expired": 0}, result["evidence"])

    def test_lease_comes_from_the_claim_block(self):
        record = claim_record(lease=1800)
        self.assertEqual("alive", self.state(1800, record)["state"])
        self.assertEqual("suspect_stale", self.state(1801, record)["state"])

    def test_future_skew_up_to_120s_is_tolerated(self):
        for ahead in (1, 119, 120):
            with self.subTest(ahead=ahead):
                result = self.state(-ahead)
                self.assertEqual(("alive", 0.0), (result["state"], result["age_s"]))
                self.assertEqual(ahead, result["evidence"]["future_s_tolerated"])
        result = self.state(-121)
        self.assertEqual(("late", 0.0, True, "future_timestamp", 121),
                         (result["state"], result["age_s"], result["evidence"]["clock_untrusted"],
                          result["evidence"]["reason"], result["evidence"]["future_s"]))

    def test_unsynchronized_clock_is_late_with_clock_untrusted_at_any_age(self):
        for ntp in (False, None):
            for age in (10, 3601, 50000):
                with self.subTest(ntp=ntp, age=age):
                    result = self.state(age, hb=hb_doc(at=self.HB_AT, ntp=ntp))
                    self.assertEqual("late", result["state"])
                    self.assertTrue(result["evidence"]["clock_untrusted"])
                    self.assertEqual("ntp_not_synchronized", result["evidence"]["reason"])

    def test_new_boot_id_is_owner_rebooted(self):
        result = self.state(5, hb=hb_doc(at=self.HB_AT, boot_id=OTHER_BOOT))
        self.assertEqual("owner_rebooted", result["state"])
        self.assertEqual((BOOT_ID, OTHER_BOOT), (result["evidence"]["boot_id_at_claim"],
                                                 result["evidence"]["boot_id_heartbeat"]))
        # Reboot wins over clock and lease: the attempt has lost its host.
        self.assertEqual("owner_rebooted", self.state(99999, hb=hb_doc(at=self.HB_AT, boot_id=OTHER_BOOT,
                                                                         ntp=False))["state"])

    def test_heartbeat_older_than_the_claim_does_not_decide_the_boot(self):
        old = hb_doc(at=T0 - timedelta(hours=5), boot_id=OTHER_BOOT)
        result = fleet.liveness(claim_record(), {REF: old}, None, T0 + timedelta(seconds=100))
        self.assertEqual(("alive", 100), (result["state"], result["age_s"]))
        self.assertTrue(result["evidence"]["heartbeat_predates_claim"])
        self.assertEqual("suspect_stale", fleet.liveness(claim_record(), {REF: old}, None,
                                                         T0 + timedelta(seconds=3601))["state"])

    def test_missing_heartbeat_ages_from_the_claim(self):
        record = claim_record()
        result = fleet.liveness(record, {}, None, T0 + timedelta(seconds=3600))
        self.assertEqual(("late", 3600, True), (result["state"], result["age_s"], result["evidence"]["heartbeat_missing"]))
        self.assertEqual("suspect_stale", fleet.liveness(record, {}, None, T0 + timedelta(seconds=3601))["state"])
        self.assertEqual("alive", fleet.liveness(record, None, None, T0 + timedelta(seconds=10))["state"])

    def test_document_of_another_worker_under_the_ref_is_ignored(self):
        result = self.state(10, hb=hb_doc(worker="gpu1", at=self.HB_AT + timedelta(hours=1)))
        self.assertEqual("document_of_another_worker", result["evidence"]["heartbeat_ignored"])
        self.assertTrue(result["evidence"]["heartbeat_missing"])

    def test_observer_fetch_failure_is_unknown_observer(self):
        result = self.state(10, observer_fetch_ok=False)
        self.assertEqual(("unknown_observer", None, False),
                         (result["state"], result["age_s"], result["evidence"]["observer_fetch_ok"]))

    def other(self, node_id, worker, age_s, now_age=3601):
        now = self.HB_AT + timedelta(seconds=now_age)
        return {hbm.heartbeat_ref(node_id, worker): hb_doc(node_id, worker, at=now - timedelta(seconds=age_s))}

    def test_half_of_other_nodes_expired_is_unknown_observer(self):
        others = {**self.other("sim-b", "gpu0", 3601), **self.other("sim-c", "gpu0", 10)}
        result = self.state(3601, others=others)
        self.assertEqual(("unknown_observer", "most_other_nodes_expired", 2, 1),
                         (result["state"], result["evidence"]["reason"], result["evidence"]["other_node_workers"],
                          result["evidence"]["other_node_workers_expired"]))
        exactly_lease = {**self.other("sim-b", "gpu0", 3600), **self.other("sim-c", "gpu0", 10)}
        self.assertEqual("suspect_stale", self.state(3601, others=exactly_lease)["state"])

    def test_minority_of_other_nodes_expired_is_suspect_stale(self):
        others = {**self.other("sim-b", "gpu0", 4000), **self.other("sim-c", "gpu0", 10),
                  **self.other("sim-d", "gpu0", 10)}
        self.assertEqual("suspect_stale", self.state(3601, others=others)["state"])

    def test_same_node_workers_do_not_count_as_observer_evidence(self):
        # x299/gpu1 dies with x299/gpu0 (same host): that is not the observer failing.
        others = {**self.other("x299", "gpu1", 3601), **self.other("sim-b", "gpu0", 10)}
        result = self.state(3601, others=others)
        self.assertEqual(("suspect_stale", 1, 0), (result["state"], result["evidence"]["other_node_workers"],
                                                   result["evidence"]["other_node_workers_expired"]))

    def test_all_worker_heartbeats_argument_is_used_for_the_observer_rule(self):
        own = {REF: hb_doc(at=self.HB_AT)}
        everyone = {**own, **self.other("sim-b", "gpu0", 5000)}
        now = self.HB_AT + timedelta(seconds=3601)
        self.assertEqual("unknown_observer", fleet.liveness(claim_record(), own, everyone, now)["state"])
        self.assertEqual("suspect_stale", fleet.liveness(claim_record(), own, {}, now)["state"])

    def test_claim_without_fleet_block_is_legacy_no_lease(self):
        record = claim_record(fleet_block_=False)
        result = fleet.liveness(record, {}, None, T0 + timedelta(seconds=331))
        # Age of the last event, by its observation time (T0 + 30 s) rather than its commit time.
        self.assertEqual(("legacy_no_lease", 301), (result["state"], result["age_s"]))
        claimed_only = claim_record(state="claimed", fleet_block_=False)
        self.assertEqual(331, fleet.liveness(claimed_only, {}, None, utc_z(T0 + timedelta(seconds=331)))["age_s"])

    def test_non_live_states_are_not_applicable(self):
        for state in ("pending", "succeeded", "failed", "interrupted", "cancelled"):
            with self.subTest(state=state):
                record = claim_record()
                record["state"] = state
                self.assertEqual({"state": "n/a", "age_s": None, "evidence": {"record_state": state}},
                                 fleet.liveness(record, {}, None, T0))

    def test_resumed_attempt_uses_its_own_claim(self):
        record = claim_record()
        record["history"].insert(1, {"kind": "claim", "state": "claimed", "attempt_id": "d" * 32,
                                     "at": (T0 - timedelta(days=3)).isoformat(),
                                     "fleet": {"heartbeat_ref": "refs/heads/fleet-hb/sim-b/gpu0",
                                               "lease_seconds": 3600, "boot_id": OTHER_BOOT}})
        result = self.state(10, record)
        self.assertEqual(("alive", REF), (result["state"], result["evidence"]["heartbeat_ref"]))

    def test_now_is_required_and_must_be_aware_utc(self):
        with self.assertRaises(ValueError):
            fleet.liveness(claim_record(), {}, None)
        with self.assertRaises(ValueError):
            fleet.liveness(claim_record(), {}, None, datetime(2026, 10, 19, 12, 0, 0))
        same = fleet.liveness(claim_record(), {}, None, "2026-10-19T12:10:00+00:00")
        self.assertEqual(same, fleet.liveness(claim_record(), {}, None, "2026-10-19T12:10:00Z"))
        self.assertEqual(same, fleet.liveness(claim_record(), {}, None,
                                              datetime(2026, 10, 19, 9, 10, tzinfo=timezone(timedelta(hours=-3)))))

    def test_liveness_is_pure_and_never_touches_git_or_the_queue(self):
        record, hbs = claim_record(), {REF: hb_doc(at=self.HB_AT), **self.other("sim-b", "gpu0", 9000)}
        snapshot = copy.deepcopy((record, hbs))
        forbidden = mock.Mock(side_effect=AssertionError("liveness must not do I/O or queue writes"))
        with mock.patch("subprocess.run", forbidden), mock.patch("subprocess.Popen", forbidden), \
             mock.patch.object(GitQueue, "claim", forbidden), mock.patch.object(GitQueue, "retry", forbidden), \
             mock.patch.object(GitQueue, "update", forbidden), mock.patch("builtins.open", forbidden):
            results = [fleet.liveness(record, hbs, None, self.HB_AT + timedelta(seconds=age))
                       for age in (10, 2000, 3601, 99999)]
        self.assertEqual(["alive", "late", "unknown_observer", "unknown_observer"], [r["state"] for r in results])
        self.assertEqual(snapshot, (record, hbs))
        self.assertEqual(results[2], fleet.liveness(record, hbs, None, self.HB_AT + timedelta(seconds=3601)))
        self.assertTrue(set(r["state"] for r in results) <= set(fleet.LIVENESS_STATES))


class LivenessEndToEndTests(HeartbeatFixture):
    """Real claim + real heartbeat ref + observer fetch; ``suspect_stale`` changes nothing."""

    def test_stale_lease_is_only_derived_and_never_written(self):
        queue = GitQueue(self.root / "queue", self.bare)
        job_id = queue.seed([fleet_spec()])[0]["job_id"]
        record = queue.claim(job_id, owner_for("x299"), fleet_block(fleet_spec()))
        hb_at = datetime.now(timezone.utc).replace(microsecond=0)
        self.assertTrue(HeartbeatPublisher(self.root / "queue-hb", self.bare, REF).publish(hb_doc(at=hb_at)).ok)
        snapshot = fetch_heartbeats(self.root / "observer", self.bare)
        before = (self.remote_head(self.bare), remote_refs(self.bare))
        alive = fleet.liveness(record, snapshot.heartbeats, None, hb_at + timedelta(seconds=900), snapshot.fetch_ok)
        stale = fleet.liveness(record, snapshot.heartbeats, None, hb_at + timedelta(seconds=7201), snapshot.fetch_ok)
        self.assertEqual(("alive", "suspect_stale"), (alive["state"], stale["state"]))
        self.assertEqual(before, (self.remote_head(self.bare), remote_refs(self.bare)))
        self.assertEqual("claimed", queue.list_jobs()[0]["state"])
        self.assertEqual(record, queue.list_jobs()[0])

    def test_legacy_claim_is_legacy_no_lease(self):
        queue = GitQueue(self.root / "queue", self.bare)
        job_id = queue.seed([legacy_spec()])[0]["job_id"]
        record = queue.claim(job_id, {"server": "synthetic-host"})
        result = fleet.liveness(record, {}, None, datetime.now(timezone.utc))
        self.assertEqual("legacy_no_lease", result["state"])
        self.assertGreaterEqual(result["age_s"], 0)


class FleetDocumentTests(GitFixture):
    """Case 8.3-17: ``put_document`` under ``fleet/`` keeps ``jobs/`` and vice versa."""

    PATH = "fleet/nodes/x299.json"

    def setUp(self):
        super().setUp()
        self.bare = self.remote()
        self.queue = GitQueue(self.root / "q1", self.bare)
        self.other = GitQueue(self.root / "q2", self.bare)
        self.job_ids = [record["job_id"] for record in self.queue.seed([fleet_spec(), legacy_spec()])]
        self.card = (canonical_json(npf.public_card(x299_profile(), "2026-10-19T12:00:00Z")) + "\n").encode()

    def jobs_listing(self):
        return git("--git-dir", str(self.bare), "ls-tree", "-r", "queue", "jobs/")

    def test_put_document_preserves_jobs_byte_for_byte(self):
        jobs_before, records_before = self.jobs_listing(), self.queue.list_jobs()
        self.assertEqual((None, None), self.queue.get_document(self.PATH))
        blob = self.queue.put_document(self.PATH, self.card, None)
        self.assertEqual(jobs_before, self.jobs_listing())
        self.assertEqual(records_before, self.other.list_jobs())
        self.assertEqual((self.card, blob), self.other.get_document(self.PATH))
        self.assertEqual(self.card, git("--git-dir", str(self.bare), "show", f"queue:{self.PATH}"))

    def test_claim_and_update_preserve_fleet_documents(self):
        blob = self.queue.put_document(self.PATH, self.card, None)
        record = self.other.claim(self.job_ids[0], owner_for("x299"), fleet_block(fleet_spec()))
        self.other.update(self.job_ids[0], record["attempt_id"], record["token"], record["version"],
                          {"event_id": "run-1", "state": "running", "epoch": 0})
        self.assertEqual((self.card, blob), self.queue.get_document(self.PATH))

    def test_cas_conflict_writes_nothing_and_rewrite_is_idempotent(self):
        blob = self.queue.put_document(self.PATH, self.card, None)
        self.assertEqual(blob, self.other.put_document(self.PATH, self.card, None))  # same bytes: idempotent
        head = self.remote_head(self.bare)
        changed = self.card.replace(b"2026-10-19T12:00:00Z", b"2026-10-19T13:00:00Z")
        for expected in (None, "0" * 40):
            with self.subTest(expected=expected), self.assertRaises(QueueConflict):
                self.other.put_document(self.PATH, changed, expected)
        self.assertEqual(head, self.remote_head(self.bare))
        new_blob = self.other.put_document(self.PATH, changed, blob)
        self.assertEqual((changed, new_blob), self.queue.get_document(self.PATH))

    def test_invalid_paths_and_cards_are_refused_before_any_write(self):
        head = self.remote_head(self.bare)
        bad_card = json.loads(self.card)
        bad_card["profile_sha256"] = "not-a-hash"
        cases = {
            "job path": (f"jobs/{self.job_ids[0]}.md", self.card),
            "traversal": ("fleet/../jobs/x.json", self.card),
            "absolute": ("/fleet/nodes/x299.json", self.card),
            "windows separator": ("fleet\\nodes\\x299.json", self.card),
            "outside fleet": ("README.json", self.card),
            "card of another node": ("fleet/nodes/sim-b.json", self.card),
            "invalid card": (self.PATH, canonical_json(bad_card).encode()),
            "not json": (self.PATH, b"{"),
            "empty": ("fleet/notes.json", b""),
        }
        for name, (path, content) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.queue.put_document(path, content, None)
        with self.assertRaises(ValueError):
            self.queue.put_document("fleet/notes.json", b"{}", "zz")
        with self.assertRaises(ValueError):
            self.queue.get_document("jobs/x.md")
        self.assertEqual(head, self.remote_head(self.bare))

    def test_put_document_preserves_local_untracked_and_index(self):
        sentinel = self.queue.checkout / "local-notes.txt"
        sentinel.write_text("keep local edits", encoding="utf-8")
        self.queue._git("add", "local-notes.txt")
        before = self.queue._git("ls-files", "--stage").stdout
        self.queue.put_document(self.PATH, self.card, None)
        self.assertEqual("keep local edits", sentinel.read_text(encoding="utf-8"))
        self.assertEqual(before, self.queue._git("ls-files", "--stage").stdout)


if __name__ == "__main__":
    unittest.main()
