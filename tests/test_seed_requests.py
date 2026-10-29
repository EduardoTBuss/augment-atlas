"""Synthetic seed control plane tests. No network, Git writes, GPU or private inputs."""
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from augment_atlas import seed_requests as sr, queue_board as board, fleet
from augment_atlas.fleet_schema import validate_public_card
from augment_atlas.node_profile import public_card
from augment_atlas.queue import GitQueue, parse_record, render_record
from augment_atlas.runtime import DockerRuntime
from augment_atlas.schema import digest, job_start_refusal, new_record
from test_queue_board import anchor, job, CAMPAIGN, COMMIT

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
IMAGE = "sha256:" + "9" * 64
PROBE = {"torch": "2.5.1", "cuda": "12.4", "forward_backward": True}
FENCE = chr(96) * 3


def request(**overrides):
    return {"request_id": "r-0001", "dataset": "synthetic-alpha", "campaign_sha256": CAMPAIGN,
            "seeds": [23], "node_id": "node-a", "requested_on": "2026-10-30",
            "requested_by": "Eduardo", "decided_after_results": False, "note": "time budget", **overrides}


def card():
    profile = {
        "schema": "atlas.node-profile.v1", "node_id": "node-a", "status": "active",
        "hardware_class": "synthetic-gpu",
        "hardware": {"cpu": {"model": "synthetic", "sockets": 1, "cores": 4, "threads": 8, "tdp_w": 65},
                     "ram_bytes": 16 * 1024**3,
                     "gpus": [{"index": 0, "name": "synthetic", "vram_mib": 12000,
                               "compute_capability": [8, 6], "uuid": "GPU-" + "0"*8 + "-0000-0000-0000-" + "0"*12}],
                     "software": {key: "synthetic" for key in
                                  ("os", "kernel", "nvidia_driver", "docker", "nvidia_container_toolkit")}},
        "storage": [{"key": "runs", "reserve_bytes": 1}],
        "limits": {"max_concurrent_trainings": 1, "allowed_gpus": [0],
                   "container": {"cpus": 2, "memory_bytes": 1024**3, "shm_bytes": 1024**2},
                   "thermal": [], "min_idle_seconds_between_jobs": 0,
                   "training_retry": "never", "forbidden_host_changes": []},
        "guardian": {"kind": "synthetic", "required_for_claim": True},
        "liveness": {"heartbeat_seconds": 60, "lease_seconds": 120, "ntp_required": True},
        "datasets": [{"id": "synthetic-alpha", "version": "synthetic-alpha-v1", "manifest_sha256": "8"*64}],
        "images": [{"image_id": IMAGE, "probe": PROBE}],
        "weights": [], "sensors": {},
        "grid": {"g_co2_per_kwh": 1, "source": "synthetic", "region": "synthetic",
                 "reference_period": "2026", "retrieved_on": "2026-10-29"},
        "updated_at_utc": "2026-10-29T00:00:00Z",
    }
    return public_card(profile, "2026-10-29T00:00:00Z")


def markdown(requests):
    return "# Seed requests\n\n" + "\n".join(FENCE + "json\n" + json.dumps(r) + "\n" + FENCE for r in requests)


class FakeAuthorizer:
    """Only a synthetic seam: no claim to implement authorize-stage."""
    def __init__(self):
        self.calls = []
        self.transform = lambda specs: specs

    def authorize(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        req = kwargs["request"]
        specs = []
        for seed in req["seeds"]:
            for arm in sr.ARMS:
                for fold in range(5):
                    spec = job(seed=seed, arm=arm, fold=fold)["spec"]
                    spec.pop("job_id")
                    spec["config"] = f"configs/global/synthetic-alpha/final/{arm}-f{fold}-s{seed}.json"
                    spec["config_sha256"] = digest({"synthetic": spec["config"]})
                    spec["requirements"] = {
                        "schema": "atlas.requirements.v1", "gpu_required": True,
                        "min_vram_mib": 1000, "image_id": IMAGE,
                        "dataset": {"id": req["dataset"], "version": "synthetic-alpha-v1", "manifest_sha256": "8"*64}}
                    specs.append(spec)
        return sr.AuthorizedStage("7"*64, "6"*40, IMAGE, self.transform(specs))


class SeedRequestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.control, self.queue, self.cards, self.state = [self.base / p for p in ("control", "queue", "nodes", "state")]
        (self.control / "control").mkdir(parents=True)
        (self.queue / "anchors/synthetic-alpha").mkdir(parents=True)
        self.cards.mkdir()
        self.card = card()
        (self.cards / "node-a.json").write_text(json.dumps(self.card), encoding="utf-8")
        (self.queue / "anchors/synthetic-alpha/campaign.json").write_text(json.dumps(anchor()), encoding="utf-8")
        self.write_requests(request())
        self.fake = FakeAuthorizer()

    def write_requests(self, *requests):
        (self.control / sr.REQUEST_FILE).write_text(markdown(requests), encoding="utf-8")

    def process(self, **kwargs):
        return sr.process_requests(self.control, self.queue, self.cards, self.state, now=NOW, **kwargs)

    def apply(self, **kwargs):
        return self.process(apply=True, authorizer=self.fake, **kwargs)

    def records(self):
        return sr.read_queue(self.queue)[0]

    def test_valid_plan_is_read_only_and_lists_matrix_node_environment(self):
        row, = self.process()
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["actions"], ["authorize_stage", "materialize_jobs"])
        self.assertEqual(row["progress"]["expected"], 50)
        self.assertEqual(row["environment"]["environment_sha256"], digest({"image_id": IMAGE, "probe": PROBE}))
        self.assertFalse(self.state.exists())
        self.assertFalse((self.queue / "jobs").exists())
        self.assertEqual(self.fake.calls, [])

    def test_strict_schema_duplicate_ids_keys_fences_and_types(self):
        bad = [
            (request(extra=1), "unknown"), (request(seeds=[23, 23]), "duplicate"),
            (request(seeds=[11]), "11"), (request(seeds=[True]), "integers"),
            (request(decided_after_results="false"), "boolean"), (request(decided_after_results=True), "SAP"),
            (request(requested_on="2026-02-30"), "range"),
            ({k: v for k, v in request().items() if k != "decided_after_results"}, "missing"),
        ]
        for value, message in bad:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                sr.parse_requests(markdown([value]))
        for text, message in [(markdown([request(), request()]), "duplicate request_id"),
                              (FENCE+"json\n{}", "unclosed"),
                              (FENCE+'json\n{"request_id":"x","request_id":"y"}\n'+FENCE, "duplicate JSON key"),
                              (json.dumps(request()), "closed JSON fence"),
                              (FENCE+"json\nNaN\n"+FENCE, "nonfinite")]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                sr.parse_requests(text)

    def test_contextual_invalid_node_campaign_and_dataset(self):
        for changes, message in [({"node_id": "missing"}, "does not exist"),
                                 ({"campaign_sha256": "0"*64}, "anchored"),
                                 ({"dataset": "another"}, "anchored")]:
            with self.subTest(changes=changes):
                self.write_requests(request(**changes))
                row, = self.apply()
                self.assertEqual(row["status"], "invalid")
                self.assertIn(message, row["error"])
        self.assertEqual(self.fake.calls, [])

    def test_duplicate_campaign_seed_across_requests_rejects_both_nodes(self):
        self.write_requests(request(), request(request_id="r-0002", node_id="node-b"))
        rows = self.apply()
        self.assertEqual([r["status"] for r in rows], ["invalid", "invalid"])
        self.assertTrue(all("duplicate campaign seed" in r["error"] for r in rows))
        self.assertEqual(self.fake.calls, [])

    def test_seed_already_present_in_campaign_is_invalid(self):
        (self.queue / "jobs").mkdir()
        record = job(seed=23)
        (self.queue / "jobs" / (record["job_id"] + ".md")).write_text(render_record(record), encoding="utf-8")
        row, = self.apply()
        self.assertEqual(row["status"], "invalid")
        self.assertIn("already exists", row["error"])

    def test_apply_idempotence_affinity_environment_and_full_matrix(self):
        row, = self.apply()
        self.assertEqual(row["status"], "authorized")
        self.assertEqual(row["progress"]["total"], 50)
        before = {p.name: p.read_bytes() for p in (self.queue / "jobs").glob("*.md")}
        again, = self.apply()
        self.assertEqual(again["status"], "authorized")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.queue / "jobs").glob("*.md")})
        records = self.records()
        self.assertEqual(len(records), 50)
        for record in records:
            spec = record["spec"]
            self.assertEqual(spec["requirements"]["allowed_nodes"], ["node-a"])
            self.assertEqual(spec["requirements"]["hardware_class"], "synthetic-gpu")
            self.assertEqual(spec["compatibility"]["image_id"], IMAGE)
            self.assertEqual(spec["environment_sha256"], digest({"image_id": IMAGE, "probe": PROBE}))
            self.assertIsNone(job_start_refusal(spec, NOW, "node-a"))
            self.assertEqual(job_start_refusal(spec, NOW, "node-b"), "node_not_allowed")

    def test_authorizer_is_called_before_any_job_and_state_is_reserved(self):
        original = self.fake.authorize
        def authorize(**kwargs):
            self.assertFalse((self.queue / "jobs").exists())
            entry = sr.load_state(self.state)["requests"]["r-0001"]
            self.assertEqual(entry["status"], "authorizing")
            self.assertEqual(kwargs["request_sha256"], digest(request()))
            self.assertEqual(kwargs["node_card"], self.card)
            return original(**kwargs)
        self.fake.authorize = authorize
        self.apply()

    def test_authorized_request_edit_and_removal_are_visible_invalid(self):
        self.apply()
        original = (self.state / sr.STATE_FILE).read_bytes()
        self.write_requests(request(note="edited"))
        row, = self.apply()
        self.assertEqual(row["status"], "invalid")
        self.assertIn("immutable", row["error"])
        self.assertEqual((self.state / sr.STATE_FILE).read_bytes(), original)
        self.write_requests()
        row, = self.process()
        self.assertEqual(row["status"], "invalid")
        self.assertIn("removed", row["error"])

    def test_missing_real_adapter_fails_without_creating_jobs(self):
        row, = self.process(apply=True)
        self.assertEqual(row["status"], "invalid")
        self.assertIn("StageAuthorizer", row["error"])
        self.assertFalse((self.queue / "jobs").exists())
        self.assertFalse((self.state / sr.STATE_FILE).exists())

    def test_invalid_authorizer_matrix_or_image_does_not_create_jobs(self):
        for transform in [lambda specs: specs[:-1],
                          lambda specs: [{**specs[0], "requirements": {**specs[0]["requirements"], "image_id": "sha256:"+"0"*64}}, *specs[1:]]]:
            self.fake.transform = transform
            row, = self.apply()
            self.assertEqual(row["status"], "invalid")
            self.assertFalse((self.queue / "jobs").exists())

    def test_retry_after_lost_authorization_response_uses_same_idempotency_key(self):
        original = self.fake.authorize
        def lost(**kwargs):
            original(**kwargs)
            raise RuntimeError("synthetic lost response")
        self.fake.authorize = lost
        self.assertEqual(self.apply()[0]["status"], "invalid")
        self.fake.authorize = original
        self.assertEqual(self.apply()[0]["status"], "authorized")
        self.assertEqual([call["request_sha256"] for call in self.fake.calls], [digest(request())]*2)

    def test_resume_after_materialization_failure_does_not_authorize_twice(self):
        original = sr._materialize
        def partial(checkout, specs):
            original(checkout, specs[:3])
            raise OSError("synthetic interruption")
        with patch.object(sr, "_materialize", side_effect=partial):
            self.assertEqual(self.apply()[0]["status"], "invalid")
        self.assertEqual(len(self.records()), 3)
        self.assertEqual(self.apply()[0]["status"], "authorized")
        self.assertEqual(len(self.records()), 50)
        self.assertEqual(len(self.fake.calls), 1)

    def test_expiry_boundary_and_requested_date_at_cutoff(self):
        expired = sr.utc(sr.DEFAULT_STOP_AT)
        row, = sr.process_requests(self.control, self.queue, self.cards, self.state,
                                  now=expired, apply=True, authorizer=self.fake)
        self.assertEqual(row["status"], "expired")
        self.assertFalse((self.queue / "jobs").exists())
        self.write_requests(request(requested_on="2026-11-12"))
        self.assertEqual(self.apply()[0]["status"], "expired")
        self.assertEqual(self.fake.calls, [])

    def test_future_requested_date_waits_and_custom_cutoff_expires(self):
        self.write_requests(request(requested_on="2026-10-31"))
        self.assertEqual(self.apply()[0]["actions"], ["wait_until_requested_on"])
        self.write_requests(request())
        self.assertEqual(self.apply(stop_at="2026-10-30T12:00:00Z")[0]["status"], "expired")

    def test_deadline_crossed_during_authorization_prevents_job_creation(self):
        real = sr.datetime
        with patch.object(sr, "datetime") as clock:
            clock.fromisoformat = real.fromisoformat
            clock.now.side_effect = [NOW, NOW, sr.utc(sr.DEFAULT_STOP_AT)]
            rows = sr.process_requests(self.control, self.queue, self.cards, self.state,
                                       apply=True, authorizer=self.fake)
        self.assertEqual(rows[0]["status"], "expired")
        self.assertEqual(sr.load_state(self.state)["requests"]["r-0001"]["status"], "authorized")
        self.assertFalse((self.queue / "jobs").exists())

    def test_expiry_gate_in_queue_claim_and_runtime(self):
        self.apply()
        record = self.records()[0]
        spec = record["spec"]
        self.assertEqual(job_start_refusal(spec, sr.utc(sr.DEFAULT_STOP_AT), "node-a"), "expired")
        queue = object.__new__(GitQueue)
        queue._lock, queue.max_retries = threading.Lock(), 1
        queue._fetch_head = lambda: COMMIT
        queue._records = lambda _: {record["job_id"]: record}
        with patch("augment_atlas.queue.datetime") as clock:
            clock.now.return_value = NOW
            self.assertIsNone(queue.claim(record["job_id"], {"server": "node-b"}))
            clock.now.return_value = sr.utc(sr.DEFAULT_STOP_AT)
            self.assertIsNone(queue.claim(record["job_id"], {"server": "node-a"}))
        runtime = object.__new__(DockerRuntime)
        runtime.s = {"server": "node-a"}
        with patch("augment_atlas.runtime.dt") as clock:
            clock.datetime.now.return_value = sr.utc(sr.DEFAULT_STOP_AT)
            with self.assertRaisesRegex(ValueError, "expired"):
                runtime.check_start(spec)

    def test_fleet_rejects_other_node_and_expired_seed_request(self):
        self.apply()
        record = self.records()[0]
        report = fleet.plan([record], None, {"node_id": "node-b"}, now_utc=NOW)
        self.assertEqual(report["jobs"][0]["reason"], "node_not_allowed")
        report = fleet.plan([record], None, {"node_id": "node-a"}, now_utc=sr.utc(sr.DEFAULT_STOP_AT))
        self.assertEqual(report["jobs"][0]["reason"], "expired")

    def test_state_checksum_and_lock_fail_closed(self):
        self.apply()
        state = self.state / sr.STATE_FILE
        document = json.loads(state.read_text(encoding="utf-8"))
        document["requests"]["r-0001"]["request"]["node_id"] = "node-b"
        state.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.process()
        (self.state / "seed-requests.lock").write_text("synthetic", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "locked"):
            self.apply()

    def test_board_pending_authorized_running_done_expired_invalid(self):
        self.assertIn("## Seed requests", board.render_board(COMMIT, [], [anchor()], NOW, self.process()))
        self.apply()
        records = self.records()
        rows = sr.board_requests(self.control, records, [anchor()], self.cards, self.state, now=NOW)
        rendered = board.render_board(COMMIT, records, [anchor()], NOW, rows)
        self.assertIn("| r-0001 | synthetic-alpha | 23 | node-a | authorized | 0/50", rendered)
        self.assertIn("23: 0/50 pending=50", rendered)
        records[0]["state"] = "running"
        rows = sr.board_requests(self.control, records, [anchor()], self.cards, self.state, now=NOW)
        self.assertEqual(rows[0]["status"], "running")
        for record in records:
            record["state"] = "succeeded"
        self.assertEqual(sr.board_requests(self.control, records, [anchor()], self.cards, self.state, now=NOW)[0]["status"], "done")
        records = self.records()
        rows = sr.board_requests(self.control, records, [anchor()], self.cards, self.state, now=sr.utc(sr.DEFAULT_STOP_AT))
        self.assertEqual(rows[0]["status"], "expired")
        self.assertEqual(rows[0]["progress"]["by_seed"]["23"]["states"]["expired"], 50)
        self.write_requests(request(node_id="node-b"))
        rows = sr.board_requests(self.control, records, [anchor()], self.cards, self.state, now=NOW)
        self.assertEqual(rows[0]["status"], "invalid")
        self.assertIn("immutable", board.render_board(COMMIT, records, [anchor()], NOW, rows))

    def test_board_reports_malformed_file_instead_of_hiding_requests(self):
        (self.control / sr.REQUEST_FILE).write_text(FENCE+"json\n{}", encoding="utf-8")
        rows = sr.board_requests(self.control, [], [anchor()], self.cards, self.state, now=NOW)
        self.assertEqual(rows[0]["status"], "invalid")
        self.assertIn("unclosed", board.render_board(COMMIT, [], [anchor()], NOW, rows))

    def test_cli_plan_exit_status_and_no_fake_clock_on_apply(self):
        args = ["plan", "--control-checkout", str(self.control), "--queue-checkout", str(self.queue),
                "--node-cards", str(self.cards), "--state-dir", str(self.state), "--now", "2026-10-30T12:00:00Z"]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(sr.main(args), 0)
        self.assertEqual(json.loads(output.getvalue())[0]["progress"]["expected"], 50)
        args[0] = "apply"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            sr.main(args, authorizer=self.fake)
        self.assertEqual(error.exception.code, 2)
        self.write_requests(request(seeds=[11]))
        args[0] = "plan"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sr.main(args), 1)

    def test_board_cli_uses_committed_snapshot_and_waits_for_job_publication(self):
        self.apply()
        output = self.base / "BOARD.md"
        args = ["--queue-checkout", str(self.queue), "--control-checkout", str(self.control),
                "--node-cards", str(self.cards), "--state-dir", str(self.state),
                "--out", str(output), "--now", "2026-10-30T12:00:00Z"]
        # Local apply has 50 jobs; renderer deliberately sees a prior committed snapshot.
        with patch.object(board, "read_snapshot", return_value=(COMMIT, [], [anchor()])):
            self.assertEqual(board.main(args), 0)
        content = output.read_text(encoding="utf-8")
        self.assertIn("awaiting_queue_publication", content)
        self.assertIn("| node-a | authorized | 0/50", content)
        self.assertEqual(len(self.records()), 50)

    def test_partial_apply_with_running_jobs_finishes_matrix_without_resetting_progress(self):
        original = sr._materialize
        def partial(checkout, specs):
            original(checkout, specs[:3])
            raise OSError("synthetic interruption")
        with patch.object(sr, "_materialize", side_effect=partial):
            self.apply()
        record = self.records()[0]
        running = job("running", seed=23, arm=record["spec"]["arm"], fold=record["spec"]["fold"])
        record.update({k: running[k] for k in ("state", "version", "seq", "history", "owner", "attempt_id", "token")})
        path = self.queue / "jobs" / (record["job_id"] + ".md")
        path.write_text(render_record(record), encoding="utf-8")
        prior = path.read_bytes()
        self.assertEqual(self.apply()[0]["progress"]["states"]["running"], 1)
        self.assertEqual(path.read_bytes(), prior)
        self.assertEqual(len(self.records()), 50)
        self.assertEqual(len(self.fake.calls), 1)

    def test_expired_runtime_create_never_calls_docker_and_environment_is_pinned(self):
        self.apply()
        spec = self.records()[0]["spec"]
        runtime = object.__new__(DockerRuntime)
        runtime.s = {"server": "node-a"}
        runtime.image, runtime.environment = IMAGE, PROBE
        runtime.check_environment(spec)
        runtime.environment = {**PROBE, "cuda": "different"}
        with self.assertRaisesRegex(ValueError, "environment_mismatch"):
            runtime.check_environment(spec)
        with patch("augment_atlas.runtime.dt") as clock, patch("augment_atlas.runtime.command") as docker:
            clock.datetime.now.return_value = sr.utc(sr.DEFAULT_STOP_AT)
            with self.assertRaisesRegex(ValueError, "expired"):
                runtime.create("synthetic", spec, self.base)
            docker.assert_not_called()

    def test_node_probe_change_and_missing_applied_job_are_visible(self):
        self.apply()
        changed = copy.deepcopy(self.card)
        changed["images"][0]["probe"]["cuda"] = "different"
        (self.cards / "node-a.json").write_text(json.dumps(changed), encoding="utf-8")
        self.assertIn("environment/profile changed", self.process()[0]["error"])
        (self.cards / "node-a.json").write_text(json.dumps(self.card), encoding="utf-8")
        next((self.queue / "jobs").glob("*.md")).unlink()
        self.assertIn("disappeared", self.process()[0]["error"])


if __name__ == "__main__":
    unittest.main()
