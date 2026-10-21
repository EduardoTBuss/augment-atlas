"""Slice S4: article variable export (stable, versioned, ID-free, atomic, idempotent)."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from augment_atlas import global_report as gr
from augment_atlas.contracts import ARMS, canonical_hash as H
from test_global_report import SEED, TELEMETRY_SUMMARY, assert_no_identifiers, fixture

TELEMETRY = dict(TELEMETRY_SUMMARY, duration_s=301.25, gpu_power_max_w=None)


def tree(root):
    return {p.name: p.read_bytes() for p in sorted(Path(root).iterdir())}


def rows_of(data):
    return list(csv.reader(io.StringIO(data.decode("utf-8"))))


class ArticleExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = fixture()
        entries = cls.fx.entries(skip={("basic", 2), ("none", 0)})
        entries.append(cls.fx.entry("none", 0, telemetry=TELEMETRY))
        cls.report = cls.fx.report(entries)
        cls.hpo_fx = fixture("hpo_d")
        cls.hpo_report = cls.hpo_fx.report()

    def temp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def export(self, out, report=None, campaign=None, **kwargs):
        return gr.export_article_variables(report or self.report, campaign or self.fx.campaign, out, **kwargs)

    def test_writes_exactly_the_stable_files_and_is_byte_idempotent(self):
        out = self.temp() / "article" / "synthetic_fixture" / "campaign"
        first = self.export(out, records=self.fx.rows)
        self.assertEqual(sorted(first["written"]), sorted(gr.ARTICLE_FILES))
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(gr.ARTICLE_FILES))  # no temp leftovers
        before = tree(out)
        second = self.export(out, records=self.fx.rows)
        self.assertEqual((second["written"], sorted(second["unchanged"])), ([], sorted(gr.ARTICLE_FILES)))
        self.assertEqual(before, tree(out))
        other = self.temp()
        self.export(other)
        self.assertEqual(before, tree(other))
        self.assertEqual(first["files"], {name: hashlib.sha256(data).hexdigest() for name, data in before.items()})
        for name, data in before.items():
            self.assertNotIn(b"\r\n", data, name)

    def test_exported_files_carry_no_identifier(self):
        out = self.temp()
        self.export(out, records=self.fx.rows)
        self.export(self.temp(), report=self.hpo_report, campaign=self.hpo_fx.campaign, records=self.hpo_fx.rows)
        for name, data in tree(out).items():
            with self.subTest(name):
                assert_no_identifiers(self, data.decode("utf-8"), self.fx.rows)
        self.assertNotIn("train_ids", tree(out)["hyperparameters.json"].decode("utf-8"))

    def test_hyperparameters_seeds_and_hashes_content(self):
        out = self.temp()
        self.export(out)
        files = tree(out)
        hyper = json.loads(files["hyperparameters.json"])
        self.assertEqual(hyper["schema_version"], 1)
        self.assertEqual(hyper["origin"]["kind"], "external_fixed")
        self.assertEqual(hyper["origin"]["source"]["derived_from_this_dataset_scores"], False)
        self.assertEqual(hyper["frozen_recipe"]["optimizer"], {"name": "adamw",
                                                               "parameters": {"lr": 0.0001, "weight_decay": 0.0001}})
        self.assertIsNone(hyper["hpo"])
        self.assertEqual(hyper["decision"]["comparison"], "ge")
        seeds = json.loads(files["seeds.json"])
        self.assertEqual((seeds["final_policy"], seeds["authorized"], seeds["not_authorized"], seeds["hpo"]),
                         ([11, 13, 19], [11], [13, 19], None))
        self.assertEqual(seeds["partition"], {"development": 17, "selection": 29, "evaluation": 31})
        hashes = json.loads(files["hashes.json"])
        self.assertEqual(hashes["schema_version"], 1)
        for key in ("pools_sha256", "records_sha256", "freeze_sha256", "campaign_sha256", "configs_manifest_sha256",
                    "code_commit", "code_tree_sha256", "image_id", "report_sha256"):
            self.assertIn(key, hashes["hashes"])
        self.assertEqual(hashes["hashes"]["report_sha256"], self.report["sha256"])
        self.assertEqual(hashes["hashes"]["campaign_sha256"], self.fx.campaign["sha256"])
        self.assertEqual(hashes["files"], {name: hashlib.sha256(files[name]).hexdigest()
                                           for name in gr.ARTICLE_FILES if name != "hashes.json"})
        readme = files["README.md"].decode("utf-8")
        for name in gr.ARTICLE_FILES:
            self.assertIn(f"`{name}`", readme)
        self.assertIn("10.1148/ryai.220232", readme)
        self.assertIn("execution_verified=false", readme)

    def test_hpo_d_hyperparameters_list_candidates_and_selected(self):
        out = self.temp()
        self.export(out, report=self.hpo_report, campaign=self.hpo_fx.campaign)
        hyper = json.loads((out / "hyperparameters.json").read_bytes())
        self.assertEqual(hyper["origin"]["kind"], "hpo_d")
        self.assertEqual(hyper["origin"]["selection_partition"], "D_select")
        self.assertEqual(hyper["hpo"]["selected_candidate_id"], "a")
        self.assertEqual(hyper["hpo"]["selected_mean"], 0.8)
        self.assertEqual([(c["id"], c["parameters"], [t["value"] for t in c["trials"]]) for c in hyper["hpo"]["candidates"]],
                         [("a", {"lr": 0.01}, [0.8]), ("b", {"lr": 0.02}, [0.7])])
        self.assertEqual(json.loads((out / "seeds.json").read_bytes())["hpo"], [101])

    def test_metrics_csvs_have_fixed_columns_and_explicit_missing_cells(self):
        out = self.temp()
        self.export(out)
        by_fold = rows_of((out / "metrics_by_fold.csv").read_bytes())
        self.assertEqual(by_fold[0][:9], ["schema_version", "arm", "seed", "fold", "status", "reason", "primary_metric",
                                          "primary_value", "primary_reason"])
        self.assertEqual(len(by_fold), 1 + 150)
        index = {(r[1], int(r[2]), int(r[3])): dict(zip(by_fold[0], r)) for r in by_fold[1:]}
        self.assertEqual((index[("basic", SEED, 2)]["status"], index[("basic", SEED, 2)]["primary_value"],
                          index[("basic", SEED, 2)]["reason"]), ("missing", "null", "artifact_not_collected"))
        self.assertEqual(index[("none", 13, 0)]["status"], "not_authorized")
        cell = next(c for c in self.report["matrix"]["cells"] if (c["arm"], c["fold"], c["seed"]) == ("none", 1, SEED))
        self.assertEqual(float(index[("none", SEED, 1)]["primary_value"]), cell["primary"]["value"])
        summary = rows_of((out / "metrics_summary.csv").read_bytes())
        header = summary[0]
        self.assertEqual(header, ["schema_version", "arm", "seed", "aggregation", "metric", "value", "reason", "unit",
                                  "n_defined", "n_expected", "sd", "ci_low", "ci_high", "ci_reason"])
        records = [dict(zip(header, r)) for r in summary[1:]]
        basic = next(r for r in records if r["arm"] == "basic" and r["seed"] == "all"
                     and r["aggregation"] == "mean_over_folds_then_seeds" and r["metric"] == "binary_auroc")
        self.assertEqual((basic["value"], basic["reason"]), ("null", "metric_undefined:undefined_in_seeds"))
        none = next(r for r in records if r["arm"] == "none" and r["seed"] == "all"
                    and r["aggregation"] == "mean_over_folds_then_seeds" and r["metric"] == "binary_auroc")
        self.assertEqual(float(none["value"]),
                         next(a for a in self.report["per_arm"] if a["arm"] == "none")["primary"]["value"])
        self.assertEqual((none["ci_low"], none["ci_reason"]), ("null", gr.NO_INFERENCE_REASON))
        self.assertEqual({r["metric"] for r in records},
                         {"binary_auroc", "macro_f1", "balanced_accuracy", "average_precision"})
        aggregations = {r["aggregation"] for r in records}
        self.assertEqual(aggregations, {"mean_over_folds", "pooled_E", "mean_over_folds_then_seeds",
                                        "pooled_E_mean_over_seeds",
                                        "paired_difference_vs_none_mean_over_folds_then_seeds"})

    def test_telemetry_csv_has_fixed_columns_and_null_when_absent(self):
        out = self.temp()
        self.export(out)
        rows = rows_of((out / "telemetry_summary.csv").read_bytes())
        self.assertEqual(rows[0], ["schema_version", "arm", "seed", "fold", "cell_status", *gr.TELEMETRY_FIELDS])
        self.assertEqual(len(rows), 1 + 50)
        index = {(r[1], int(r[3])): dict(zip(rows[0], r)) for r in rows[1:]}
        self.assertEqual((index[("none", 0)]["duration_s"], index[("none", 0)]["train_images_processed"],
                          index[("none", 0)]["gpu_power_max_w"], index[("none", 0)]["status"]),
                         ("301.25", "6500", "null", "completed"))
        self.assertEqual({index[("none", 1)][name] for name in gr.TELEMETRY_FIELDS}, {"null"})
        self.assertEqual(index[("basic", 2)]["cell_status"], "missing")

    def test_conflicting_existing_file_is_refused_unless_overwrite(self):
        out = self.temp()
        self.export(out)
        before = tree(out)
        (out / "seeds.json").write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "^export_conflict:"):
            self.export(out)
        self.assertEqual((out / "seeds.json").read_bytes(), b"{}\n")
        self.assertEqual({k: v for k, v in tree(out).items() if k != "seeds.json"},
                         {k: v for k, v in before.items() if k != "seeds.json"})
        result = self.export(out, overwrite=True)
        self.assertEqual(result["written"], ["seeds.json"])
        self.assertEqual(tree(out), before)

    def test_export_refuses_tampered_report_or_foreign_campaign(self):
        with self.assertRaisesRegex(ValueError, "^report_invalid:"):
            self.export(self.temp(), campaign=self.hpo_fx.campaign)
        stale = copy.deepcopy(self.report)
        stale["label"] = "confirmatory_candidate"
        with self.assertRaisesRegex(ValueError, "^report_invalid:"):
            self.export(self.temp(), report=stale)
        leaked = copy.deepcopy(self.report)
        leaked["label_reasons"] = leaked["label_reasons"] + [self.fx.rows[0]["id"]]
        leaked.pop("sha256")
        leaked["sha256"] = H(leaked)
        out = self.temp()
        with self.assertRaisesRegex(ValueError, "^identifier_leak:"):
            self.export(out, report=leaked, records=self.fx.rows)
        self.assertEqual(list(out.iterdir()), [])

    def test_default_article_dir_and_comparison_export(self):
        path = gr.default_article_dir(self.fx.campaign, root="/tmp/article-root")
        self.assertEqual(path.parts[-2:], ("synthetic_fixture", f"external_fixed-{self.fx.campaign['sha256'][:12]}"))
        comparison = gr.compare_campaigns(self.report, self.hpo_report)
        out = self.temp()
        first = gr.export_comparison(comparison, out, records=self.fx.rows)
        self.assertEqual(sorted(first["written"]), sorted(gr.COMPARISON_FILES))
        before = tree(out)
        self.assertEqual(gr.export_comparison(comparison, out)["written"], [])
        self.assertEqual(before, tree(out))
        rows = rows_of(before["campaign_comparison.csv"])
        self.assertEqual(len(rows), 1 + len(ARMS) * 6)
        self.assertEqual({r[-1] for r in rows[1:]}, {"descriptive_only"})
        for data in before.values():
            assert_no_identifiers(self, data.decode("utf-8"), self.fx.rows)
        tampered = dict(comparison, superiority_claim=True)
        with self.assertRaisesRegex(ValueError, "^campaign_comparison_invalid:"):
            gr.export_comparison(tampered, self.temp())


if __name__ == "__main__":
    unittest.main()
