"""F5 part A: ``GitQueue.claim(job_id, owner, fleet=None)`` (docs/fleet-protocol.md 3.2d, 3.3, 8.3-7/8).

Real Git: one bare remote per scenario and independent clones, as in ``tests/test_queue.py``.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
import uuid

from augment_atlas import fleet
from augment_atlas import node_profile as npf
from augment_atlas import queue as queue_module
from augment_atlas.queue import GitQueue, parse_record, render_record
from augment_atlas.schema import canonical_json, digest, prepare_job

try:  # ``discover -s tests`` puts tests/ on sys.path; ``-m unittest tests.x`` uses the package path.
    from test_fleet import BOOT_ID, IMAGE, MONO, RECORDS, fleet_spec, legacy_spec, x299_latest, x299_profile, x299_worker
except ImportError:  # pragma: no cover - alternative runner layout
    from tests.test_fleet import (BOOT_ID, IMAGE, MONO, RECORDS, fleet_spec, legacy_spec, x299_latest, x299_profile,
                                  x299_worker)


def old_schema():
    """``validate_record`` exactly as in b7a47f5 (fixture copy shared with tests/test_fleet.py)."""
    spec = importlib.util.spec_from_file_location("schema_b7a47f5_f5", RECORDS / "schema_b7a47f5.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyClaimQueue(GitQueue):
    """``claim`` copied verbatim from ad911a8 (pre-F5): the byte-identity oracle."""

    def claim(self, job_id, owner):
        if not isinstance(owner, dict) or not owner:
            raise ValueError("owner must be a nonempty public alias object")
        canonical_json(owner)
        attempt_id, token = uuid.uuid4().hex, uuid.uuid4().hex
        with self._lock:
            for _ in range(self.max_retries):
                head = self._fetch_head()
                records = self._records(head)
                current = self._get(records, job_id)
                if current["attempt_id"] == attempt_id and current["token"] == token:
                    return current
                if current["state"] != "pending" or any(records[dependency]["state"] != "succeeded" for dependency in current["spec"]["dependencies"]):
                    return None
                record = copy.deepcopy(current)
                record.update(state="claimed", attempt_id=attempt_id, token=token, owner=copy.deepcopy(owner))
                self._append(record, {"kind": "claim", "attempt_id": attempt_id, "token": token, "owner": copy.deepcopy(owner), "state": "claimed"})
                confirmed = self._publish(head, [record], f"queue: claim {job_id}")
                current = self._get(confirmed, job_id)
                if current["attempt_id"] == attempt_id and current["token"] == token:
                    return current
                if current["state"] != "pending":
                    return None
            raise queue_module.QueueConflict("claim exhausted concurrent retries")


def owner_for(node_id, worker="gpu0"):
    return {"server": node_id, "node_id": node_id, "worker": worker, "gpu": "0",
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000"}


def fleet_block(spec, node_id="x299", worker="gpu0", profile_sha256="6" * 64, latest=None):
    """What a node writes at claim time (3.2d), computed from the job it is about to claim."""
    gate = {"observed_at_utc": "2026-10-19T12:00:00Z", "gpu_temp_c": 41, "cpu_pkg_temp_c": 38,
            "free_disk_bytes": 43892981760, "gpu_idle": True}
    if latest is not None:
        gate.update(gpu_temp_c=latest["gpus"][0]["temp_c"], cpu_pkg_temp_c=latest["cpu_pkg_max_temp_c"],
                    free_disk_bytes=latest["disks"][0]["free_bytes"], gpu_idle=latest["gpus"][0]["compute_apps"] == 0)
    return {"protocol": "atlas.fleet.v1", "node_profile_sha256": profile_sha256, "coordinator_commit": "7" * 40,
            "boot_id": BOOT_ID, "image_id": IMAGE, "launch_plan_sha256": "8" * 64,
            "requirements_sha256": digest(fleet.job_requirements(spec)), "lease_seconds": 3600,
            "heartbeat_ref": f"refs/heads/fleet-hb/{node_id}/{worker}", "gate": gate}


class GitFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="augment-atlas-f5-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def remote(self, name="remote"):
        path = self.root / f"{name}.git"
        subprocess.run(["git", "init", "--bare", str(path)], check=True, capture_output=True)
        return path

    @staticmethod
    def remote_head(remote, branch="queue"):
        result = subprocess.run(["git", "--git-dir", str(remote), "rev-parse", "--verify", "-q", f"refs/heads/{branch}"],
                                capture_output=True, text=True)
        return result.stdout.strip() or None

    @staticmethod
    def remote_blob(remote, job_id, ref="queue"):
        return subprocess.run(["git", "--git-dir", str(remote), "show", f"{ref}:jobs/{job_id}.md"],
                              check=True, capture_output=True).stdout

    @staticmethod
    def remote_tree(remote, ref="queue"):
        return subprocess.run(["git", "--git-dir", str(remote), "rev-parse", f"{ref}^{{tree}}"],
                              check=True, capture_output=True, text=True).stdout.strip()


class LegacyClaimIdentityTests(GitFixture):
    """``fleet=None`` must publish exactly the bytes of the pre-F5 code path."""

    @staticmethod
    def frozen():
        ids = iter(uuid.UUID(int=n) for n in range(1, 1000))
        return (mock.patch.object(queue_module.uuid, "uuid4", side_effect=lambda: next(ids)),
                mock.patch.object(queue_module, "utc_now", return_value="2026-10-19T12:00:00.000000+00:00"))

    def claim_with(self, queue_class, call, specs):
        remote = self.remote(f"remote-{len(list(self.root.iterdir()))}")
        queue = queue_class(self.root / f"clone-{remote.stem}", remote)
        job_ids = [record["job_id"] for record in queue.seed(specs)]
        ids_patch, time_patch = self.frozen()
        with ids_patch, time_patch:
            results = [call(queue, job_id) for job_id in job_ids]
        return remote, job_ids, results

    def test_fleet_none_is_byte_identical_to_pre_f5_claim(self):
        cases = {"legacy spec": [legacy_spec(train_seed=31, min_memory_mb=1000)],
                 "fleet-v1 spec claimed without fleet": [fleet_spec(train_seed=32)]}
        owner = {"server": "synthetic-host", "gpu": "GPU-fixture"}
        for name, specs in cases.items():
            with self.subTest(name):
                old = self.claim_with(LegacyClaimQueue, lambda q, j: q.claim(j, owner), specs)
                positional = self.claim_with(GitQueue, lambda q, j: q.claim(j, owner), specs)
                explicit = self.claim_with(GitQueue, lambda q, j: q.claim(j, owner, fleet=None), specs)
                for new in (positional, explicit):
                    self.assertEqual(old[1], new[1])
                    self.assertEqual(old[2], new[2])
                    for job_id in old[1]:
                        self.assertEqual(self.remote_blob(old[0], job_id), self.remote_blob(new[0], job_id))
                    self.assertEqual(self.remote_tree(old[0]), self.remote_tree(new[0]))
                claim_event = explicit[2][0]["history"][-1]
                self.assertEqual({"kind", "attempt_id", "token", "owner", "state", "seq", "at"}, set(claim_event))


class FleetClaimPersistenceTests(GitFixture):
    """8.3-8: the claim persists ``fleet``; a reserved key inside it is a ``ValueError``."""

    def setUp(self):
        super().setUp()
        self.remote_path = self.remote()
        self.q1 = GitQueue(self.root / "node-a", self.remote_path)
        self.q2 = GitQueue(self.root / "node-b", self.remote_path)
        self.spec = fleet_spec(train_seed=41)
        self.job_id = self.q1.seed([self.spec])[0]["job_id"]

    def test_claim_persists_fleet_block_in_same_cas_event(self):
        owner, block = owner_for("x299"), fleet_block(self.spec)
        sent = copy.deepcopy(block)
        claimed = self.q1.claim(self.job_id, owner, fleet=block)
        self.assertEqual(sent, block, "claim must not mutate the caller's block")
        block["gate"]["gpu_temp_c"] = 99  # later caller mutation cannot leak into the record
        event = claimed["history"][-1]
        self.assertEqual(("claim", "claimed", 1), (event["kind"], event["state"], event["seq"]))
        self.assertEqual(sent, event["fleet"])
        self.assertEqual((claimed["attempt_id"], claimed["token"]), (event["attempt_id"], event["token"]))
        self.assertEqual(owner, claimed["owner"])
        # Confirmed by an independent clone (refetch is the only proof of ownership).
        [remote_record] = self.q2.list_jobs()
        self.assertEqual(claimed, remote_record)
        # Pre-existing: on Windows ``_git(input=...)`` runs in text mode, so the stored blob has CRLF.
        markdown = self.remote_blob(self.remote_path, self.job_id).decode("utf-8").replace("\r\n", "\n")
        # The human summary line prints ``str(owner)`` in the caller's key order (pre-existing),
        # so only the authoritative JSON payload is compared.
        self.assertIn(render_record(remote_record).split("<!-- augment-atlas-record-v1 -->")[1], markdown)
        self.assertEqual(remote_record, parse_record(markdown))
        # Additive: the unchanged b7a47f5 validator accepts the fleet claim (8.3-2 for a live claim).
        legacy = old_schema()
        self.assertEqual(remote_record, legacy.validate_record(copy.deepcopy(remote_record)))
        # The attempt continues through the existing update path.
        running = self.q1.update(self.job_id, claimed["attempt_id"], claimed["token"], claimed["version"],
                                 {"event_id": "start", "state": "running", "epoch": 0,
                                  "launch_plan_sha256": sent["launch_plan_sha256"]})
        self.assertEqual("running", running["state"])
        self.assertEqual(sent, running["history"][0]["fleet"])
        # A second claim on the owned job publishes nothing.
        head = self.remote_head(self.remote_path)
        self.assertIsNone(self.q2.claim(self.job_id, owner_for("x299", "gpu1"), fleet=fleet_block(self.spec, worker="gpu1")))
        self.assertEqual(head, self.remote_head(self.remote_path))

    def test_legacy_job_can_be_claimed_with_fleet_block(self):
        spec = legacy_spec(train_seed=42, min_memory_mb=1000)
        job_id = self.q1.seed([spec])[0]["job_id"]
        claimed = self.q2.claim(job_id, owner_for("x299"), fleet=fleet_block(spec))
        self.assertEqual(digest(fleet.derive_legacy(spec)), claimed["history"][-1]["fleet"]["requirements_sha256"])
        self.assertEqual(claimed, old_schema().validate_record(copy.deepcopy(claimed)))

    def assert_refused_without_write(self, owner, block, git_allowed=False):
        before = (self.remote_head(self.remote_path), self.remote_blob(self.remote_path, self.job_id))
        if git_allowed:
            with self.assertRaises(ValueError):
                self.q1.claim(self.job_id, owner, fleet=block)
        else:
            # Refused before any Git I/O: not even a fetch.
            with mock.patch.object(self.q1, "_git", side_effect=AssertionError("Git touched")), \
                    self.assertRaises(ValueError):
                self.q1.claim(self.job_id, owner, fleet=block)
        self.assertEqual(before, (self.remote_head(self.remote_path), self.remote_blob(self.remote_path, self.job_id)))
        self.assertEqual("pending", self.q2.list_jobs()[0]["state"])

    def test_reserved_key_inside_fleet_is_value_error_and_writes_nothing(self):
        for key in ("seq", "at", "attempt_id", "token", "payload_hash", "kind"):
            with self.subTest(key=key):
                block = fleet_block(self.spec)
                block[key] = "x"
                self.assert_refused_without_write(owner_for("x299"), block)

    def test_invalid_fleet_block_is_refused_without_writing(self):
        good = fleet_block(self.spec)
        def changed(**updates):
            block = copy.deepcopy(good)
            for key, value in updates.items():
                if value is KeyError:
                    del block[key]
                else:
                    block[key] = value
            return block
        bad_gate = copy.deepcopy(good)
        bad_gate["gate"]["note"] = "hot"
        private_gate = copy.deepcopy(good)
        private_gate["gate"]["observed_at_utc"] = "2026-10-19 12:00"
        cases = {
            "not an object": [good],
            "missing key": changed(launch_plan_sha256=KeyError),
            "unknown key": changed(extra=1),
            "unknown protocol": changed(protocol="atlas.fleet.v2"),
            "short hash": changed(launch_plan_sha256="8" * 63),
            "bad image": changed(image_id="ghcr.io/x/y:latest"),
            "bad boot id": changed(boot_id="not-a-boot-id"),
            "zero lease": changed(lease_seconds=0),
            "bool lease": changed(lease_seconds=True),
            "local path in ref": changed(heartbeat_ref="/home/x/hb"),
            "non-numeric gate value": bad_gate,
            "bad gate timestamp": private_gate,
        }
        for name, block in cases.items():
            with self.subTest(name):
                self.assert_refused_without_write(owner_for("x299"), block)

    def test_owner_and_block_must_describe_the_same_node_worker(self):
        cases = {
            "legacy owner (no node_id/worker)": {"server": "x299", "gpu": "0"},
            "server differs from node_id": {**owner_for("x299"), "server": "other"},
            "heartbeat ref of another worker": owner_for("x299", "gpu1"),
            "heartbeat ref of another node": owner_for("sim-b"),
        }
        for name, owner in cases.items():
            with self.subTest(name):
                self.assert_refused_without_write(owner, fleet_block(self.spec))

    def test_requirements_hash_must_match_the_claimed_spec(self):
        stale = fleet_block(self.spec)
        stale["requirements_sha256"] = digest(fleet.derive_legacy(self.spec))
        self.assert_refused_without_write(owner_for("x299"), stale, git_allowed=True)


class ConcurrentFleetClaimTests(GitFixture):
    """CAS: two concurrent fleet claims on the same job, exactly one wins (repeated rounds)."""

    def test_two_fleet_claims_same_job_exactly_one_wins(self):
        remote = self.remote()
        nodes = {"x299": GitQueue(self.root / "x299", remote), "sim-b": GitQueue(self.root / "sim-b", remote)}
        specs = [fleet_spec(train_seed=50 + n) for n in range(3)]
        job_ids = [record["job_id"] for record in nodes["x299"].seed(specs)]
        for spec, job_id in zip(specs, job_ids):
            with self.subTest(job_id=job_id):
                barrier = threading.Barrier(len(nodes))
                blocks = {node: fleet_block(spec, node_id=node, profile_sha256=digest(node)) for node in nodes}

                def attempt(node):
                    barrier.wait()
                    return node, nodes[node].claim(job_id, owner_for(node), fleet=blocks[node])

                with ThreadPoolExecutor(len(nodes)) as pool:
                    results = dict(pool.map(attempt, nodes))
                winners = [node for node, record in results.items() if record is not None]
                self.assertEqual(1, len(winners), results)
                [winner] = winners
                confirmed = next(record for record in nodes["sim-b"].list_jobs() if record["job_id"] == job_id)
                self.assertEqual(results[winner], confirmed)
                self.assertEqual(owner_for(winner), confirmed["owner"])
                self.assertEqual(blocks[winner], confirmed["history"][-1]["fleet"])
                self.assertEqual(1, sum(event["kind"] == "claim" for event in confirmed["history"]))


class TwoNodeSelectionTests(GitFixture):
    """8.3-7: two nodes, real Git, distinct profiles: exactly one confirmed claim on the
    contested job; the loser takes its next compatible job, or none."""

    def setUp(self):
        super().setUp()
        profile_a = x299_profile()
        profile_b = copy.deepcopy(profile_a)
        profile_b["node_id"] = "sim-b"
        profile_b["limits"]["allowed_gpus"] = [0]          # distinct profile, same hardware capture
        profile_b = npf.validate(profile_b, complete=True)
        latest_b = copy.deepcopy(x299_latest())
        latest_b["node_id"] = "sim-b"
        self.nodes = {}
        remote = self.remote()
        for node_id, profile, latest in (("x299", profile_a, x299_latest()), ("sim-b", profile_b, latest_b)):
            self.assertNotEqual(None, profile)
            self.nodes[node_id] = {"queue": GitQueue(self.root / node_id, remote), "profile": profile,
                                   "latest": latest, "worker": x299_worker(node_id=node_id),
                                   "card": npf.public_card(profile, "2026-10-19T13:00:00Z")}
        self.assertNotEqual(npf.profile_sha256(profile_a), npf.profile_sha256(profile_b))
        # J1 is first for everyone; J2 is restricted to sim-b only.
        self.j1 = fleet_spec(train_seed=61, scheduling={"priority": 5, "order": None})
        self.j2 = fleet_spec(train_seed=62, allowed_nodes=["sim-b"])
        self.ids = [record["job_id"] for record in self.nodes["x299"]["queue"].seed([self.j1, self.j2])]
        self.specs = dict(zip(self.ids, (self.j1, self.j2)))

    def run_node(self, node_id, barrier):
        """Section 7.1: plan (read-only), then claim in order; the CAS decides the race."""
        node = self.nodes[node_id]
        records = node["queue"].list_jobs()
        report = fleet.plan(records, node["profile"], node["worker"], latest=node["latest"], now=MONO, card=node["card"])
        barrier.wait()
        for job_id in report["order"]:
            block = fleet_block(self.specs[job_id], node_id=node_id,
                                profile_sha256=npf.profile_sha256(node["profile"]), latest=node["latest"])
            record = node["queue"].claim(job_id, owner_for(node_id), fleet=block)
            if record is not None:
                return report["order"], record
        return report["order"], None

    def test_contested_job_has_exactly_one_owner_and_loser_falls_through(self):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(2) as pool:
            futures = {node_id: pool.submit(self.run_node, node_id, barrier) for node_id in self.nodes}
            results = {node_id: future.result() for node_id, future in futures.items()}
        self.assertEqual([self.ids[0]], results["x299"][0], "x299 is not allowed on J2")
        self.assertEqual(self.ids, results["sim-b"][0])
        final = {record["job_id"]: record for record in self.nodes["x299"]["queue"].list_jobs()}
        j1_owner = final[self.ids[0]]["owner"]["node_id"]
        self.assertEqual("claimed", final[self.ids[0]]["state"])
        self.assertEqual(1, sum(event["kind"] == "claim" for event in final[self.ids[0]]["history"]))
        if j1_owner == "x299":
            self.assertEqual(self.ids[0], results["x299"][1]["job_id"])
            self.assertEqual(self.ids[1], results["sim-b"][1]["job_id"], "loser takes its next compatible job")
            self.assertEqual("sim-b", final[self.ids[1]]["owner"]["node_id"])
        else:
            self.assertEqual(self.ids[0], results["sim-b"][1]["job_id"])
            self.assertIsNone(results["x299"][1], "loser has no other compatible job")
            self.assertEqual("pending", final[self.ids[1]]["state"])
        for record in final.values():
            if record["state"] == "claimed":
                owner = record["owner"]["node_id"]
                event = record["history"][-1]
                self.assertEqual(npf.profile_sha256(self.nodes[owner]["profile"]), event["fleet"]["node_profile_sha256"])
                self.assertEqual(f"refs/heads/fleet-hb/{owner}/gpu0", event["fleet"]["heartbeat_ref"])

    def test_each_winner_is_reachable(self):
        """Both branches of 8.3-7 are exercised deterministically by letting one node go first."""
        for first, second in (("x299", "sim-b"), ("sim-b", "x299")):
            with self.subTest(first=first):
                self.setUp()
                solo = threading.Barrier(1)
                first_result = self.run_node(first, solo)
                second_result = self.run_node(second, solo)
                self.assertEqual(self.ids[0], first_result[1]["job_id"])
                if first == "x299":
                    self.assertEqual(self.ids[1], second_result[1]["job_id"])
                else:
                    self.assertIsNone(second_result[1])


if __name__ == "__main__":
    unittest.main()
