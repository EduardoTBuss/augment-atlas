"""A1 private declarations and primary reporting without a Git transport or trainer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import global_campaign as gc, stage_gates as gates, stage_overlap as overlap
from augment_atlas.contracts import canonical_hash
from augment_atlas.queue import GitQueue, QueueConflict
import test_global_report as fixtures


class OverlapUnitTests(unittest.TestCase):
    def document(self):
        return {"sha256": "a" * 64, "campaign_sha256": "c" * 64,
                "dataset": {"id": "synthetic"}, "sealed_at_utc": "2026-10-30T18:00:00Z"}

    def test_nonempty_amendment_is_required_and_record_is_private_sealed_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            document = self.document()
            out = overlap.publish_record(tmp, document, "SAP amendment A1: parallel sensitivity")
            record = overlap.load_record(tmp, document)
            self.assertEqual(out["record_sha256"], canonical_hash({k: v for k, v in record.items() if k != "sha256"}))
            self.assertTrue(Path(out["record_path"]).is_relative_to(Path(tmp).resolve()))
            self.assertEqual(overlap.publish_record(tmp, document, record["amendment_and_reason"]), out)
            for text in (None, "", " ", "\t\n", False):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    overlap.publish_record(tmp, document, text)
            with self.assertRaisesRegex(ValueError, "no-clobber"):
                overlap.publish_record(tmp, document, "Different amendment reason")

    def test_resigned_record_cannot_rebind_authorization_campaign_or_release_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self.document()
            overlap.publish_record(tmp, doc, "SAP amendment A1: parallel sensitivity")
            record = overlap.load_record(tmp, doc)
            for key, value in (("campaign_sha256", "0" * 64), ("stage_authorization_sha256", "0" * 64),
                               ("allow_incomplete_stage1", False), ("amendment_and_reason", None)):
                bad = {**record, key: value}
                bad.pop("sha256")
                bad["sha256"] = canonical_hash(bad)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    overlap.validate_record(bad, doc["campaign_sha256"], doc["sha256"])

    def test_publication_gate_truth_table_requires_record_and_upgrade_for_own_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self.document()
            overlap.publish_record(tmp, doc, "SAP amendment A1: parallel sensitivity")
            record = overlap.load_record(tmp, doc)
            records = {"own": {"spec": {"global": {"campaign_sha256": doc["campaign_sha256"]}}},
                       "other": {"spec": {"global": {"campaign_sha256": "d" * 64}}}}
            anchor = {"kind": "stage_authorization", "campaign_sha256": doc["campaign_sha256"],
                      "document_sha256": doc["sha256"]}
            for upgrade, timing in ((None, None), (None, record), ("all restarted", None), ("all restarted", record)):
                queue = gates.StageGatedQueue.__new__(gates.StageGatedQueue)
                queue.all_workers_upgraded, queue.incomplete_stage1_record = upgrade, timing
                with self.subTest(upgrade=upgrade, timing=timing is not None), \
                     patch.object(GitQueue, "_check_anchor_preconditions"), \
                     patch.object(gates, "active_primary_jobs", return_value=["own", "other"]):
                    if upgrade is None or timing is None:
                        with self.assertRaises(QueueConflict):
                            queue._check_anchor_preconditions("head", records, anchor)
                    else:
                        queue._check_anchor_preconditions("head", records, anchor)
                        self.assertEqual(queue.released_primary_jobs, ["own", "other"])
                        self.assertEqual(queue.release_head, "head")

    def test_current_primary_requires_final_state_for_each_cell(self):
        for state in (None, "pending", "claimed", "running", "interrupted"):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "stage1_incomplete:"):
                overlap.require_final_primary({"job": None}, [] if state is None else [{"job_id": "job", "state": state}])
        for state in ("succeeded", "failed", "cancelled"):
            overlap.require_final_primary({"job": None}, [{"job_id": "job", "state": state}])


class CurrentReportUnitTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.fixture()
        self.chronology = {"protocol": gc.CHRONOLOGY_PROTOCOL, "campaign_sha256": self.fx.campaign["sha256"],
                           "structural_valid": True, "execution_verified": False, "scientific_eligible": False,
                           "head": "f" * 40, "C_freeze": "a" * 40, "C_campaign": "b" * 40,
                           "final_jobs": 150, "other_campaign_final_jobs": 0, "warnings": []}
        self.snapshot = {**self.chronology, "head": "e" * 40, "final_jobs": 50}
        self.chronology["stage1_chronology"] = self.snapshot

    def test_current_report_uses_full_head_while_legacy_scope_preserves_snapshot_bytes(self):
        baseline = self.fx.report(chronology=self.snapshot)
        historic = self.fx.report(chronology=self.chronology, scope="stage1")
        self.assertEqual(json.dumps(historic, sort_keys=True), json.dumps(baseline, sort_keys=True))
        current = self.fx.report(chronology=self.chronology, scope="stage1_current")
        self.assertEqual(current["chronology"]["head"], "f" * 40)
        self.assertEqual(current["chronology"]["final_jobs"], 150)
        self.assertEqual(current["seeds"]["authorized"], [11])
        self.assertEqual(current["matrix"]["authorized_cells"], 50)

    def test_current_report_refuses_incomplete_or_interrupted_primary_and_unaudited_chronology(self):
        with self.assertRaisesRegex(ValueError, "stage1_incomplete:"):
            self.fx.report(self.fx.entries(skip=[("none", 0)]), scope="stage1_current", chronology=self.chronology)
        entries = self.fx.entries(skip=[("none", 0)])
        entries.append(self.fx.entry("none", 0, status="interrupted", reason="job_interrupted", predictions=None))
        with self.assertRaisesRegex(ValueError, "stage1_incomplete:"):
            self.fx.report(entries, scope="stage1_current", chronology=self.chronology)
        with self.assertRaisesRegex(ValueError, "chronology_invalid:"):
            self.fx.report(scope="stage1_current")

    def test_current_report_retains_terminal_failure_as_an_explicit_result(self):
        entries = self.fx.entries(skip=[("none", 0)])
        entries.append(self.fx.entry("none", 0, status="failed", reason="job_failed", predictions=None))
        current = self.fx.report(entries, scope="stage1_current", chronology=self.chronology)
        cell = next(c for c in current["matrix"]["cells"] if (c["arm"], c["fold"], c["seed"]) == ("none", 0, 11))
        self.assertEqual((cell["status"], cell["reason"]), ("failed", "job_failed"))


if __name__ == "__main__":
    unittest.main()
