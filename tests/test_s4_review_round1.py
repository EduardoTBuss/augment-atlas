"""Slice S4 review hardening: collector reasons, comparison flags, re-signed score mutation, lazy telemetry context."""
import copy
import tempfile
import unittest
from pathlib import Path

from augment_atlas import global_report as gr
from augment_atlas import train_telemetry as tt
from augment_atlas.contracts import canonical_hash as H
from test_global_report import SEED, TELEMETRY_SUMMARY, fixture, resign


class ReasonCodeTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixture()

    def failed(self, reason):
        return self.fx.entry("none", 0, status="failed", reason=reason, initial_state=None)

    def test_free_text_paths_and_hostnames_are_not_reason_codes(self):
        for reason in ("timeout reading /data/x/img_001.png", "host-01.example.org down", "A" * 5, "x" * 65,
                       " leading", "trailing ", "", "with space", "../escape"):
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, "^report_input_invalid:"):
                    self.fx.report(self.fx.entries(skip={("none", 0)}) + [self.failed(reason)])

    def test_code_embedding_a_record_identifier_is_a_leak(self):
        row = self.fx.rows[0]
        needle = "".join(ch for ch in str(row["id"]).lower() if ch.isalnum())
        reason = f"failed_on_{needle}"
        with self.assertRaisesRegex(ValueError, "^identifier_leak:"):
            self.fx.report(self.fx.entries(skip={("none", 0)}) + [self.failed(reason)])

    def test_short_machine_codes_are_accepted(self):
        report = self.fx.report(self.fx.entries(skip={("none", 0)}) + [self.failed("docker_exit_1")])
        self.assertEqual(report["matrix"]["failed_cells"], 1)

    def test_unhashable_telemetry_status_is_a_typed_error(self):
        entry = self.fx.entry("none", 0, telemetry=dict(TELEMETRY_SUMMARY, status=[]))
        with self.assertRaisesRegex(ValueError, "^telemetry_summary_invalid:"):
            self.fx.report(self.fx.entries(skip={("none", 0)}) + [entry])


class ScoreMutationTests(unittest.TestCase):
    def test_resigned_score_mutation_is_caught_by_recomputation(self):
        fx = fixture()
        artifact = fx.artifact("clahe_heavy", 3)
        artifact["rows"][0]["scores"] = [0.5, 0.5]
        entries = [e for e in fx.entries() if (e["arm"], e["fold"]) != ("clahe_heavy", 3)]
        entries.append(fx.entry("clahe_heavy", 3, predictions=resign(artifact)))
        with self.assertRaisesRegex(ValueError, "^(metric|decision)_recompute_mismatch:|^predictions_invalid:"):
            fx.report(entries)


class ComparisonFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = gr.compare_campaigns(fixture().report(), fixture("hpo_d").report())

    def resigned(self, **changes):
        value = {k: v for k, v in self.comparison.items() if k != "sha256"}
        value.update(changes)
        value["sha256"] = H(value)
        return value

    def test_resigned_comparison_with_forbidden_flags_is_refused(self):
        for changes in ({"scientific_eligible": True}, {"execution_verified": True}, {"superiority_claim": True},
                        {"scientific_status": "final"}, {"inference": "inferential"}, {"hypothesis_test": "t"}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "^campaign_comparison_invalid:"):
                    gr.check_comparison(self.resigned(**changes))
                with tempfile.TemporaryDirectory() as out:
                    with self.assertRaisesRegex(ValueError, "^campaign_comparison_invalid:"):
                        gr.export_comparison(self.resigned(**changes), out)
                    self.assertEqual(list(Path(out).iterdir()), [])

    def test_untouched_comparison_passes(self):
        self.assertIs(gr.check_comparison(self.comparison), self.comparison)


class ArticleExportRootTests(unittest.TestCase):
    def test_root_is_explicit(self):
        campaign = fixture().campaign
        for root in (None, ""):
            with self.assertRaisesRegex(ValueError, "^export_conflict:"):
                gr.default_article_dir(campaign, root)

    def test_json_exports_carry_the_provisional_flags(self):
        fx = fixture()
        with tempfile.TemporaryDirectory() as out:
            gr.export_article_variables(fx.report(), fx.campaign, out)
            import json
            for name in ("hyperparameters.json", "seeds.json", "hashes.json"):
                doc = json.loads((Path(out) / name).read_text(encoding="utf-8"))
                self.assertEqual((doc["scientific_status"], doc["execution_verified"], doc["scientific_eligible"]),
                                 ("provisional", False, False))


class LazyTelemetryContextTests(unittest.TestCase):
    def test_failing_context_builder_degrades_to_null_telemetry(self):
        def broken():
            raise ZeroDivisionError("bad batch size")
        with tempfile.TemporaryDirectory() as out:
            telemetry = tt.open_for_attempt(Path(out), torch=None, device="cpu", context=broken)
            self.assertIsInstance(telemetry, tt.NullTelemetry)
            telemetry.close()
            self.assertFalse((Path(out) / tt.DIRECTORY).exists())

    def test_callable_context_is_evaluated(self):
        with tempfile.TemporaryDirectory() as out:
            telemetry = tt.open_for_attempt(Path(out), torch=None, device="cpu", context=lambda: {"path": "x"})
            self.assertIsInstance(telemetry, tt.TrainTelemetry)
            telemetry.outcome("completed")
            telemetry.close()


if __name__ == "__main__":
    unittest.main()
