"""Review finding F1 (guard B-1'' of the HPO collector), synthetic fixtures only.

A non-succeeded attempt that already wrote ``predictions.json`` or ``results.json``
must make the whole collection ``collection_invalid``. The check has to run BEFORE
the per-job timeout and the GPU-minutes cap shortcuts: otherwise such an attempt
becomes a plain ``timeout`` row and a candidate is dropped after its own score was
visible. Attempts without scores and legitimately timed-out ``succeeded`` attempts
keep their behavior. No Docker, GPU, training or real data.
"""
import os
import unittest

from augment_atlas import global_evaluation as ge
from test_global_cli import DATASET
from augment_atlas.queue import GitQueue
from test_global_cli_hpo import GOOD, TIE, HpoCase
from test_global_hpo import DEFAULT_PROFILE, Base

NON_SUCCEEDED = ("failed", "cancelled", "interrupted")
KEYS = [("c00", 101), ("c01", 101), ("c02", 101), ("c03", 101)]


def budget(**changes):
    return {"max_candidates": 6, "max_seeds": 2, "max_epochs": 20, "job_timeout_s": 3600, "gpu_minutes_cap": 120,
            **changes}


class ScoresBeforeShortcutsTests(Base):
    def setUp(self):
        self.fx = None

    def collect(self, fx, plan, *, scores_for=(), scores="both"):
        """``plan`` maps key -> (state, minutes); keys not listed are missing from the snapshot.

        A ``succeeded`` job always has its attempt written. A non-succeeded job only gets
        scores when its key is in ``scores_for``; ``scores`` picks which artifact exists.
        """
        records = []
        for key in sorted(plan):
            state, minutes = plan[key]
            if state == "succeeded" or key in scores_for:
                out = fx.write_attempt(key, DEFAULT_PROFILE[key[0]])
                if state != "succeeded":
                    if scores == "predictions":
                        (out / "results.json").unlink()
                    elif scores == "results":
                        (out / ge.ARTIFACT_NAME).unlink()
                    elif scores == "dangling_symlink":
                        (out / "results.json").unlink()
                        (out / ge.ARTIFACT_NAME).unlink()
                        os.symlink(out / "nowhere", out / "results.json")
            records.append(fx.record(key, state, minutes=minutes))
        return fx.collect(records)

    def assertInvalid(self, fx, plan, **kwargs):
        with self.assertRaisesRegex(ValueError, r"^collection_invalid: non-succeeded HPO job [0-9a-f]{12} already wrote scores$"):
            self.collect(fx, plan, **kwargs)

    # --- plain failure (no shortcut involved): the existing guard still holds ---------------------------------------

    def test_failed_cancelled_interrupted_with_scores_are_collection_invalid(self):
        for state in NON_SUCCEEDED:
            for artifacts in ("both", "predictions", "results", "dangling_symlink"):
                with self.subTest(state=state, artifacts=artifacts):
                    fx = self.fixture()
                    plan = {key: ("succeeded", 1.0) for key in KEYS}
                    plan[("c01", 101)] = (state, 1.0)
                    self.assertInvalid(fx, plan, scores_for=[("c01", 101)], scores=artifacts)

    def test_non_succeeded_without_scores_stays_a_failed_row(self):
        for state in NON_SUCCEEDED:
            with self.subTest(state=state):
                fx = self.fixture()
                plan = {key: ("succeeded", 1.0) for key in KEYS}
                plan[("c01", 101)] = (state, 1.0)
                evidence, _ = self.collect(fx, plan)
                row = next(r for r in evidence["rows"] if r["candidate_id"] == "c01")
                self.assertEqual((row["status"], row["reason"], row["metric"]["value"]),
                                 ("failed", f"job_{state}", None))

    # --- per-job timeout ---------------------------------------------------------------------------------------------

    def test_job_timeout_does_not_hide_scores_of_a_non_succeeded_attempt(self):
        for state in NON_SUCCEEDED:
            with self.subTest(state=state):
                fx = self.fixture(budget=budget(job_timeout_s=90))
                plan = {key: ("succeeded", 1.0) for key in KEYS}
                plan[("c01", 101)] = (state, 2.0)  # 120 s > job_timeout_s
                self.assertInvalid(fx, plan, scores_for=[("c01", 101)])

    def test_job_timeout_without_scores_and_of_succeeded_attempts_is_still_a_timeout_row(self):
        for state in ("succeeded",) + NON_SUCCEEDED:
            with self.subTest(state=state):
                fx = self.fixture(budget=budget(job_timeout_s=90))
                plan = {key: ("succeeded", 1.0) for key in KEYS}
                plan[("c01", 101)] = (state, 2.0)
                evidence, manifest = self.collect(fx, plan)  # a succeeded attempt keeps its scores out of the ledger
                row = next(r for r in evidence["rows"] if r["candidate_id"] == "c01")
                self.assertEqual((row["status"], row["reason"], row["metric"]["value"], row["artifact_sha256"]),
                                 ("timeout", "job_timeout_s_exceeded", None, None))
                item = next(r for r in manifest["rows"] if r["candidate_id"] == "c01")
                self.assertEqual(item["queue_state"], state)

    # --- GPU-minutes cap ---------------------------------------------------------------------------------------------

    def test_first_job_over_the_cap_does_not_hide_scores_of_a_non_succeeded_attempt(self):
        for state in NON_SUCCEEDED:
            with self.subTest(state=state):
                fx = self.fixture(budget=budget(gpu_minutes_cap=1))
                plan = {key: ("succeeded", 1.0) for key in KEYS}
                plan[("c00", 101)] = (state, 2.0)  # the very first job already exceeds the 1-minute cap
                self.assertInvalid(fx, plan, scores_for=[("c00", 101)])

    def test_first_job_over_the_cap_without_scores_is_still_a_cap_timeout(self):
        for state in NON_SUCCEEDED:
            with self.subTest(state=state):
                fx = self.fixture(budget=budget(gpu_minutes_cap=1))
                plan = {key: ("succeeded", 1.0) for key in KEYS}
                plan[("c00", 101)] = (state, 2.0)
                evidence, _ = self.collect(fx, plan)
                self.assertEqual({(r["status"], r["reason"]) for r in evidence["rows"]},
                                 {("timeout", "gpu_minutes_cap_exceeded")})

    def test_jobs_after_the_cap_do_not_hide_scores_of_a_non_succeeded_attempt(self):
        for state in NON_SUCCEEDED:
            for victim in (("c02", 101), ("c03", 101)):  # the job that crosses the cap, and one after it
                with self.subTest(state=state, victim=victim[0]):
                    fx = self.fixture(budget=budget(gpu_minutes_cap=5))
                    plan = {key: ("succeeded", 2.0) for key in KEYS}  # c00, c01 fit (4 min); c02 crosses the cap
                    plan[victim] = (state, 2.0)
                    self.assertInvalid(fx, plan, scores_for=[victim])

    def test_jobs_after_the_cap_keep_their_rows_when_there_are_no_scores(self):
        fx = self.fixture(budget=budget(gpu_minutes_cap=5))
        plan = {key: ("succeeded", 2.0) for key in KEYS}
        plan[("c03", 101)] = ("failed", 2.0)
        evidence, manifest = self.collect(fx, plan)
        self.assertEqual([(r["candidate_id"], r["status"], r["reason"]) for r in evidence["rows"]],
                         [("c00", "completed", None), ("c01", "completed", None),
                          ("c02", "timeout", "gpu_minutes_cap_exceeded"), ("c03", "timeout", "gpu_minutes_cap_exceeded")])
        self.assertAlmostEqual(manifest["gpu_minutes_counted"], 4.0)

    def test_succeeded_attempts_over_the_cap_remain_timeout_rows(self):
        fx = self.fixture(budget=budget(gpu_minutes_cap=5))
        evidence, _ = self.collect(fx, {key: ("succeeded", 2.0) for key in KEYS})
        self.assertEqual([r["status"] for r in evidence["rows"]], ["completed", "completed", "timeout", "timeout"])

    def test_guard_is_independent_of_the_order_in_which_the_queue_lists_jobs(self):
        fx = self.fixture(budget=budget(gpu_minutes_cap=5))
        records = []
        for key in sorted(KEYS, reverse=True):
            state = "cancelled" if key == ("c03", 101) else "succeeded"
            out = fx.write_attempt(key, DEFAULT_PROFILE[key[0]])
            self.assertTrue(out.is_dir())
            records.append(fx.record(key, state, minutes=2.0))
        with self.assertRaisesRegex(ValueError, "^collection_invalid:"):
            fx.collect(records)


class NoPinNoFreezeTests(HpoCase):

    def hpo_jobs(self):
        queue = GitQueue(self.env.checkout, self.env.remote)
        return queue, {j["spec"]["global"]["candidate_id"]: j for j in queue.list_jobs() if j["spec"]["phase"] == "hpo"}
    """End to end through the CLI: a refused collection leaves no evidence, pin or freeze behind."""

    def written(self):
        directory = self.env.data / "global" / DATASET
        return sorted(path.name for path in directory.glob("*")) if directory.is_dir() else []

    def test_cancelled_or_interrupted_attempt_with_scores_leaves_no_pin_evidence_or_freeze(self):
        self.to_seeded_hpo()
        queue, jobs = self.hpo_jobs()
        for candidate in ("c00", "c02", "c03"):
            self.finish_hpo(queue, jobs[candidate], TIE)
        before = self.written()
        self.finish_hpo(queue, jobs["c01"], GOOD, end_state="interrupted")
        error = self.run_cli(self.collect_args(), code=2)
        self.assertTrue(error.startswith("collection_invalid:"), error)
        self.assertIn("already wrote scores", error)
        self.assertEqual(self.written(), before)
        self.assertFalse([name for name in self.written() if "pin" in name or "freeze" in name or "evidence" in name])
        error = self.seal_freeze("0" * 64, code=2)
        self.assertTrue(error.startswith("collection_invalid:") or error.startswith("cli_input_invalid:"), error)
        self.assertFalse([name for name in self.written() if "freeze" in name])


if __name__ == "__main__":
    unittest.main()
