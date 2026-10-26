"""Audit 2026-10-26: residual of review finding F1 (guard B-1'' across attempts).

Synthetic fixtures only; no Docker, GPU, training or real data.

``docs/global-cli.md`` (B-1''): "A non-succeeded HPO attempt that already wrote
``predictions.json`` or ``results.json`` is ``collection_invalid``". The collector
applies that to the CURRENT attempt (``global_hpo.collect_hpo_evidence``), but the
check over EARLIER attempts (``global_hpo._check_retries``) only looks at
``predictions.json``. An earlier failed attempt that left only ``results.json`` with a
score therefore stops blocking the collection once the job is retried.

The test below states the B-1'' contract and is marked ``expectedFailure`` because
the fix (extend ``_check_retries`` to ``results.json``) contradicts the legacy test
``test_global_hpo.CollectorTests.test_unfinished_retry_cannot_hide_an_earlier_attempt_with_predictions``,
which deletes only ``predictions.json`` and expects ``missing``. Changing that legacy
test needs Eduardo's authorization (docs/cloud-context.md 4.3). When the fix lands,
remove the decorator: an unexpected success makes this file fail loudly.
"""
import json
import unittest

from augment_atlas import global_evaluation as ge
from test_global_hpo import DEFAULT_PROFILE, RETRIED, Base


class EarlierAttemptResultsOnlyTests(Base):
    KEY = ("c01", 101)

    def collect_with_earlier_attempt(self, keep):
        """Attempt RETRIED failed after writing scores, was retried, attempt ATTEMPT succeeded.

        ``keep`` names the score artifacts of the earlier attempt that survive.
        """
        fx = self.fixture()
        records = []
        for key in sorted(fx.jobs):
            fx.write_attempt(key, DEFAULT_PROFILE[key[0]])
            records.append(fx.retried_record(key) if key == self.KEY else fx.record(key, "succeeded"))
        earlier = fx.write_attempt(self.KEY, "good", attempt=RETRIED)
        for name in (ge.ARTIFACT_NAME, "results.json"):
            if name not in keep:
                (earlier / name).unlink()
        return fx.collect(records)

    def test_earlier_attempt_with_predictions_is_refused(self):
        # Control: the part of the rule that is implemented today.
        with self.assertRaisesRegex(ValueError, "^collection_invalid: a retried job already had"):
            self.collect_with_earlier_attempt(keep={ge.ARTIFACT_NAME})

    def test_earlier_attempt_results_carry_a_score(self):
        # Control: the surviving results.json really exposes a primary score.
        fx = self.fixture()
        earlier = fx.write_attempt(self.KEY, "good", attempt=RETRIED)
        results = json.loads((earlier / "results.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(results["clinical_evaluation"]["primary_value"])

    @unittest.expectedFailure
    def test_earlier_attempt_with_only_results_json_is_refused(self):
        # B-1'' residual (P2): today this collects as if the earlier score never existed.
        with self.assertRaisesRegex(ValueError, "^collection_invalid:"):
            self.collect_with_earlier_attempt(keep={"results.json"})


if __name__ == "__main__":
    unittest.main()
