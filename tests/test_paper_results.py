"""Paper report checks using generated native synthetic predictions only."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import analysis_sap_v1 as sap
from augment_atlas import paper_results as paper
from augment_atlas.contracts import ARMS, canonical_hash
from test_analysis_sap_v1 import native_artifact, synthetic


def write_campaign(path, dataset, *, incomplete=False, seeds=(1,), rng_seed=7):
    """Native closed-schema artifacts, positive class first; no image files."""
    path = Path(path)
    rows = synthetic(rng_seed, groups_per_class=3, seeds=seeds)
    for arm in ARMS:
        for fold in sap.FOLDS:
            if incomplete and arm == "basic_gamma_heavy" and fold == 4:
                continue
            for seed in seeds:
                artifact = native_artifact(arm, fold)
                artifact.pop("content_sha256")
                artifact["dataset"]["id"] = dataset
                artifact["train_seed"] = seed
                # Different synthetic bindings per dataset, shared within its campaign.
                artifact["pools_sha256"] = hashlib.sha256((dataset + " pools").encode()).hexdigest()
                artifact["freeze_sha256"] = hashlib.sha256((dataset + " freeze").encode()).hexdigest()
                artifact["rows"] = []
                for row in rows:
                    if (row["arm"], row["fold"], row["seed"]) != (arm, fold, seed):
                        continue
                    score = row["score"]
                    artifact["rows"].append({"id": row["image"], "group": row["group"],
                                             "target": 4 if row["label"] else 9,
                                             "scores": [score, 1. - score],
                                             "prediction": 4 if score >= .5 else 9})
                artifact["content_sha256"] = canonical_hash(artifact)
                target = path / arm / str(fold) / str(seed) / "predictions.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
    return path


def write_studies(path):
    """Three studies: complete three-seed, missing H1 constituent, descriptive INbreast."""
    path = Path(path)
    return {"synthetic_alpha": write_campaign(path / "alpha", "synthetic_alpha", seeds=(1, 2, 3)),
            "synthetic_beta": write_campaign(path / "beta", "synthetic_beta", incomplete=True, rng_seed=8),
            "INbreast_synthetic": write_campaign(path / "inbreast", "INbreast_synthetic", rng_seed=9)}


class PaperResultsTests(unittest.TestCase):
    def test_sealed_sap_sha256(self):
        self.assertEqual(hashlib.sha256(Path(sap.__file__).read_bytes()).hexdigest(),
                         "bb3bb94a76705f2499f3d32c3778a0c3d212abc314d92ec08678d6ea1d2c2458")

    def test_native_three_dataset_cli_artifacts_and_sap_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaigns = write_studies(Path(tmp) / "inputs")
            out = Path(tmp) / "out"
            args = ["--out", str(out), "--seed", "123", "--B", "39",
                    "--recipe-source", "INbreast_synthetic=hpo_d"]
            for dataset, path in campaigns.items():
                args.extend(["--campaign", f"{dataset}={path}"])
            real_figure = paper.forest_figure
            def check_figure(reports):
                figure = real_figure(reports)
                self.assertEqual(len(figure.axes), 3)
                self.assertEqual(figure.get_size_inches()[0], 3.35)
                self.assertEqual(figure.get_size_inches()[1], 7.5)
                self.assertIsNone(figure._suptitle)
                for ax in figure.axes:
                    self.assertEqual(ax.get_title(), "")
                    self.assertEqual(ax.get_title(loc="left"), "")
                    self.assertEqual(ax.get_title(loc="right"), "")
                    self.assertEqual(ax.get_xlabel(), "Absolute delta AUROC")
                    self.assertEqual(len(ax.get_yticklabels()), 15)
                beta = next(ax for ax in figure.axes if ax.get_ylabel() == "synthetic_beta")
                self.assertTrue(any("H1 [NE]" == text.get_text() for text in beta.get_yticklabels()))
                return figure
            with patch.object(paper.sap, "analyze", wraps=sap.analyze) as analyze, \
                    patch.object(paper, "forest_figure", side_effect=check_figure):
                self.assertEqual(paper.main(args), 0)
                self.assertEqual(analyze.call_count, 3)
            bundle = json.loads((out / "results.json").read_text())
            self.assertEqual(bundle["provenance"]["sap_source_sha256"], paper._sha(sap.__file__))
            self.assertFalse(bundle["execution_verified"])
            self.assertFalse(bundle["scientific_eligible"])
            self.assertEqual(len(bundle["datasets"]), 3)
            for report in bundle["datasets"]:
                dataset = report["dataset"]
                rows, metadata = sap.read_predictions(campaigns[dataset], dataset)
                expected = sap.analyze(rows, dataset=dataset, B=39, seed=123,
                                       recipe_source="hpo_d" if "INbreast" in dataset else None)
                actual = dict(report)
                provenance = actual.pop("provenance")
                completeness = actual.pop("completeness")
                self.assertEqual(completeness, paper.prediction_completeness(rows, expected))
                self.assertEqual(actual, expected)
                self.assertEqual(provenance["pools_sha256"], metadata["pools_sha256"])
                self.assertEqual(provenance["freeze_sha256"], metadata["freeze_sha256"])
                self.assertEqual(provenance["inputs"], metadata["inputs"])
                table = (out / f"table-{paper._slug(dataset)}.tex").read_text()
                self.assertIn("97.5\\%", table)
                self.assertIn("Holm", table)
                self.assertIn("Positive groups", table)
                self.assertIn("Seed 3 O/E", table)
                for arm in ARMS:
                    self.assertIn(paper._tex(arm), table)
                for arm in report["primary"]["arms"]:
                    self.assertNotIn("auroc", arm)
                    line = next(line for line in table.splitlines()
                                if line.startswith(paper._tex(arm["arm"]) + " & "))
                    cells = line.split(" & ")
                    self.assertEqual(cells[1], paper._effect({
                        "estimate": (arm["metrics"] or {}).get("auroc"), "status": arm["status"]}))
                    self.assertNotIn("[", cells[1])
                self.assertIn("AUROC (point)", table)
                self.assertNotIn("arm_comparisons", report["primary"])
                self.assertEqual(len(report["primary"]["contrasts"]), 15)
                s4 = [item for item in report["primary"]["contrasts"] if item["id"].startswith("S4_")]
                self.assertEqual(len(s4), 8)
                for value in report["primary"]["contrasts"]:
                    self.assertIn(paper._tex(value["id"]), table)
                    self.assertIn(paper._effect(value["auroc"]), table)
                for value in s4:
                    target = next(name for name, coef in value["coefficients"].items() if coef == 1.)
                    reference = "basic" if target.startswith("basic_") else "none"
                    self.assertEqual(value["coefficients"], {target: 1., reference: -1.})
                    if dataset == "synthetic_alpha":
                        self.assertIsNotNone(value["auroc"]["ci95"])
                for contrast in report["primary"]["contrasts"][:2]:
                    value = contrast["auroc"]
                    if value.get("p_holm") is not None:
                        self.assertIn(f'{value["p_holm"]:.4g}', table)
                    if value.get("ci97_5") is not None:
                        self.assertIn(f'[{value["ci97_5"][0]:.4f}, {value["ci97_5"][1]:.4f}]', table)
                for warning in report["warnings"]:
                    self.assertIn(paper._tex(warning), table)
            beta_report = next(r for r in bundle["datasets"] if r["dataset"] == "synthetic_beta")
            self.assertEqual(beta_report["primary"]["contrasts"][0]["status"], "not_evaluable")
            self.assertEqual(beta_report["primary"]["contrasts"][1]["status"], "ok")
            beta_table = (out / f'table-{paper._slug("synthetic_beta")}.tex').read_text()
            self.assertIn("(NE)", beta_table)
            self.assertIn("4/4", beta_table)
            descriptive = next(r for r in bundle["datasets"] if r["dataset"] == "INbreast_synthetic")
            self.assertEqual(descriptive["analysis_scope"], "descriptive_hpo_d")
            self.assertTrue(all(item["auroc"]["ci95"] is None
                                for item in descriptive["primary"]["contrasts"]))
            self.assertIn("(D)", (out / f'table-{paper._slug("INbreast_synthetic")}.tex').read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            for name, expected in manifest["outputs"].items():
                self.assertEqual(paper._sha(out / name), expected)
            png = (out / "forest.png").read_bytes()
            pdf = (out / "forest.pdf").read_bytes()
            self.assertTrue(png.startswith(b"\x89PNG"))
            self.assertEqual(int.from_bytes(png[16:20], "big"), 1005)  # 3.35 inches at 300 dpi
            self.assertTrue(pdf.startswith(b"%PDF"))
            media = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)", pdf)
            self.assertIsNotNone(media)
            self.assertAlmostEqual(float(media.group(1)), 3.35 * 72)
            snippet = (out / "results.tex").read_text()
            self.assertIn(r"\input{\PaperResultsPath/forest.tex}", snippet)
            self.assertEqual(snippet.count(r"\input{"), 4)
            self.assertNotIn("improved", snippet)
            for path in out.iterdir():
                if path.suffix not in (".json", ".tex"):
                    continue
                text = path.read_text()
                self.assertNotIn("image-0-", text)
                self.assertNotIn("group-0-", text)
                self.assertNotIn(str(campaigns["synthetic_alpha"]), text)

    def test_completeness_missing_partial_nonfinite_and_single_class_cells(self):
        rows = synthetic(groups_per_class=3, seeds=(1, 2, 3))
        rows = [row for row in rows if not (
            row["arm"] == "basic_gamma_heavy" and row["fold"] == 4 and row["seed"] == 1)]
        # Remove a known image independently of the fixture's naming scheme.
        partial = next(row for row in rows if (row["arm"], row["fold"], row["seed"]) ==
                       ("gamma_light", 2, 2))
        rows.remove(partial)
        nonfinite = next(row for row in rows if (row["arm"], row["fold"], row["seed"]) ==
                         ("clahe_heavy", 3, 3))
        nonfinite["score"] = float("nan")
        report = sap.analyze(rows, dataset="synthetic", B=9)
        counts = {(item["arm"], item["training_seed"]): item
                  for item in paper.prediction_completeness(rows, report)}
        self.assertEqual(len(counts), 30)
        self.assertEqual(counts["none", 1]["observed_folds"], 5)
        self.assertEqual(counts["none", 1]["evaluable_folds"], 5)
        missing = counts["basic_gamma_heavy", 1]
        self.assertEqual((missing["observed_folds"], missing["evaluable_folds"]), (4, 4))
        self.assertEqual(missing["folds"][4]["reason"], "missing_fold")
        partial = counts["gamma_light", 2]
        self.assertEqual((partial["observed_folds"], partial["evaluable_folds"]), (5, 4))
        self.assertEqual(partial["folds"][2]["reason"], "unpaired_images")
        self.assertEqual(partial["folds"][2]["observed_images"],
                         partial["folds"][2]["expected_images"] - 1)
        self.assertEqual(counts["clahe_heavy", 3]["evaluable_folds"], 4)
        self.assertEqual(counts["clahe_heavy", 3]["folds"][3]["reason"], "nonfinite_scores")
        rows = [dict(row, label=0) for row in synthetic(groups_per_class=3, seeds=(1, 2, 3))]
        report = sap.analyze(rows, dataset="synthetic", B=9)
        self.assertTrue(all(item["observed_folds"] == 5 and item["evaluable_folds"] == 0
                            for item in paper.prediction_completeness(rows, report)))

    def test_unsealed_sap_refused_before_reading_or_publication(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paper, "_sha", return_value="changed"), \
                patch.object(paper.sap, "read_predictions") as reader:
            out = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "sealed SAP SHA-256"):
                paper.generate({"synthetic": Path(tmp) / "missing"}, out, seed=1, B=9)
            reader.assert_not_called()
            self.assertFalse(out.exists())

    def test_undefined_bootstrap_retains_point_without_intervals(self):
        rows = []
        for fold in sap.FOLDS:
            for image, group, label in (("a", "mixed", 0), ("b", "mixed", 1), ("c", "positive", 1)):
                for arm in ARMS:
                    rows.append(dict(dataset="synthetic", fold=fold, arm=arm, seed=1,
                                     image=f"{fold}-{image}", group=f"{fold}-{group}", label=label, score=.4))
        report = sap.analyze(rows, dataset="synthetic", B=99, seed=2)
        arm = report["primary"]["arms"][0]
        self.assertEqual(arm["metrics"]["auroc"], .5)
        self.assertEqual(arm["status"], "ok")
        contrast = report["primary"]["contrasts"][0]
        self.assertEqual(contrast["status"], "not_evaluable")
        self.assertIsNone(contrast["auroc"]["ci95"])
        report["completeness"] = paper.prediction_completeness(rows, report)
        self.assertIn("0.0000 (NE)", paper.dataset_table(report))
        self.assertIn("0.5000 & 5/5", paper.dataset_table(report))
        self.assertIn("undefined\\_bootstrap\\_replicates", paper.dataset_table(report))

    def test_input_errors_do_not_publish_and_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_campaign(Path(tmp) / "inputs", "synthetic")
            out = Path(tmp) / "out"
            invalid_args = [
                ["--campaign", "missing-assignment"],
                ["--campaign", f"synthetic={path}", "--campaign", f"synthetic={path}"],
                ["--campaign", f"wrong={path}"],
                ["--campaign", f"synthetic={path}", "--recipe-source", "other=hpo_d"],
                ["--campaign", f"synthetic={path}", "--recipe-source", "synthetic=invalid"],
                ["--campaign", f"synthetic={path}", "--B", "0"],
                ["--campaign", f"synthetic={path}", "--seed", "-1"],
            ]
            for args in invalid_args:
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(paper.main(["--out", str(out), "--seed", "1", "--B", "9"] + args), 2)
                    self.assertFalse(out.exists())
            out.mkdir()
            (out / "manifest.json").write_text("previous completed output")
            # A later invalid dataset must not partially publish an earlier study.
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(paper.main(["--out", str(out), "--seed", "1", "--B", "9",
                                             "--campaign", f"synthetic={path}",
                                             "--campaign", f"zzz_wrong={path}"]), 2)
            self.assertEqual((out / "manifest.json").read_text(), "previous completed output")
            self.assertEqual(list(out.iterdir()), [out / "manifest.json"])

    def test_native_inbreast_requires_declaration_and_refuses_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_campaign(Path(tmp) / "inputs", "INbreast_synthetic")
            out = Path(tmp) / "out"
            args = ["--campaign", f"INbreast_synthetic={path}", "--out", str(out), "--seed", "1", "--B", "9"]
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(paper.main(args), 2)
            self.assertIn("explicit recipe_source", stderr.getvalue())
            self.assertFalse(out.exists())
            artifact_path = next(path.rglob("predictions.json"))
            artifact = json.loads(artifact_path.read_text())
            artifact["rows"][0]["scores"] = [.5, .5]
            artifact_path.write_text(json.dumps(artifact))
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(paper.main(args + ["--recipe-source", "INbreast_synthetic=external_fixed"]), 2)
            self.assertIn("content_sha256", stderr.getvalue())
            self.assertFalse(out.exists())

    def test_latex_escaping_and_collision_safe_names(self):
        self.assertEqual(paper._tex("<8 >7"), r"\textless{}8 \textgreater{}7")
        self.assertNotEqual(paper._slug("a_b"), paper._slug("a-b"))
        self.assertNotIn("/", paper._slug("../../unsafe"))
        text = "a_b&c%#{d}$~^\\"
        self.assertEqual(paper._tex(text),
                         r"a\_b\&c\%\#\{d\}\$\textasciitilde{}\textasciicircum{}\textbackslash{}")
        rows = [dict(row, dataset=text) for row in synthetic(groups_per_class=3)]
        report = sap.analyze(rows, dataset=text, B=9)
        report["completeness"] = paper.prediction_completeness(rows, report)
        self.assertIn(paper._tex(text), paper.dataset_table(report))

    def test_reproducible_rerun_replaces_manifest_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_campaign(Path(tmp) / "inputs", "synthetic")
            out = Path(tmp) / "out"
            first = paper.generate({"synthetic": path}, out, seed=7, B=19)
            hashes = json.loads((out / "manifest.json").read_text())["outputs"]
            (out / "results.tex").write_text("stale content")
            second = paper.generate({"synthetic": path}, out, seed=7, B=19)
            self.assertEqual(first, second)
            self.assertEqual(hashes, json.loads((out / "manifest.json").read_text())["outputs"])
            with patch.object(paper, "forest_figure", side_effect=OSError("synthetic rendering failure")):
                with self.assertRaisesRegex(OSError, "rendering failure"):
                    paper.generate({"synthetic": path}, out, seed=7, B=19)
            self.assertEqual(hashes, json.loads((out / "manifest.json").read_text())["outputs"])
            self.assertFalse(list(Path(tmp).glob("paper-results-*")))


if __name__ == "__main__":
    unittest.main()
