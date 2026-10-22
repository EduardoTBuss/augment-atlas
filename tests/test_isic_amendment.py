"""ISIC 2020 amendments of 22/10/2026 (decisions S1, S2 and the eight extra duplicate pairs).

* The committed split amendment pins the acquisition config identity (unchanged, so
  an existing download ledger stays valid) and declares the three seeds and S2 rules.
* The accepted divergence for extra byte-identical pairs is operational: verify then
  merges the pairs in the union-find and writes READY; split-spec + make-pools +
  import-pools turn the verified root into sealed v1 pools.

Synthetic release served from 127.0.0.1 (the acquisition e2e harness); no network
beyond loopback, no real dataset.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquisition_e2e_helpers import SyntheticRelease  # noqa: E402
from test_acquisition_isic_e2e import AcquisitionE2E  # noqa: E402

from augment_atlas import global_development as gd  # noqa: E402
from augment_atlas import global_splits as gs  # noqa: E402
from augment_atlas.acquisition import config as config_mod, groups, isic2020  # noqa: E402
from augment_atlas.contracts import file_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "acquisition" / "isic2020-v1.json"
AMENDMENT = ROOT / "configs" / "acquisition" / "isic2020-v1-splits-amendment-20261022.json"
# Identity pinned in the ledger of the existing download; operational edits must never change it.
DOWNLOAD_IDENTITY = "605233a82acfa4b85f28488590bfdfdd61b4d61a683daf46dbc48d1cef33283b"


def amendment_for(raw, **changes):
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    value["amends"]["config_identity_sha256"] = config_mod.identity_of(raw)
    value["fractions"] = {"D": raw["splits"]["dev_fraction"], "D_select_given_D": raw["splits"]["select_fraction"]}
    value["thresholds"] = {k: raw["splits"][k] for k in ("min_select_positive_groups", "min_eval_fold_positive_groups")}
    value.update(changes)
    return value


class RepositoryDocumentsTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    def spec(self, **changes):
        value = {"dataset": {"id": "isic2020", "version": "isic2020-derived-512-v1"},
                 "task": {"type": "binary", "classes": [0, 1], "positive_class": 1, "endpoint": "binary_auroc",
                          "decision_rule": "fixed_threshold", "missing_policy": "undefined_no_drop",
                          "requires_patient_identity": True},
                 "grouping": {}, "provenance": {}, "source_sha256": "0" * 64,
                 "seeds": copy.deepcopy(self.amendment["seeds"]),
                 "fractions": copy.deepcopy(self.amendment["fractions"]),
                 "thresholds": copy.deepcopy(self.amendment["thresholds"])}
        value.update(changes)
        return value

    def test_config_identity_is_the_one_of_the_existing_download(self):
        cfg = config_mod.load(CONFIG)
        self.assertEqual(cfg.identity_sha256, DOWNLOAD_IDENTITY)
        self.assertEqual(self.amendment["amends"]["config_identity_sha256"], DOWNLOAD_IDENTITY)

    def test_amendment_values_are_the_decided_ones(self):
        seeds = self.amendment["seeds"]
        self.assertEqual(seeds["development"], self.raw["splits"]["split_seed"])
        self.assertNotIn(seeds["evaluation"], (seeds["development"], seeds["selection"]))
        self.assertEqual(self.amendment["fractions"], {"D": 0.2, "D_select_given_D": 0.3})
        self.assertEqual(self.amendment["thresholds"], {"min_select_positive_groups": 15,
                                                        "min_eval_fold_positive_groups": 50})
        self.assertIs(self.amendment["before_any_score"], True)
        gs.check_amendment(self.spec(), self.amendment, self.raw)

    def test_spec_or_amendment_drift_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^split_spec_invalid: spec seeds"):
            gs.check_amendment(self.spec(seeds={"development": 20261019, "selection": 20261019, "evaluation": 7}),
                               self.amendment, self.raw)
        with self.assertRaisesRegex(ValueError, "^seed_reuse_forbidden:"):
            gs.check_amendment(self.spec(seeds={"development": 1, "selection": 2, "evaluation": 1}),
                               self.amendment, self.raw)
        other = copy.deepcopy(self.raw)
        other["splits"]["split_seed"] += 1  # an identity change: the amendment is bound to the old identity
        with self.assertRaisesRegex(ValueError, "^split_spec_invalid: split amendment is bound"):
            gs.check_amendment(self.spec(), self.amendment, other)
        with self.assertRaisesRegex(ValueError, "^split_spec_invalid:"):
            gs.check_amendment(self.spec(), {**self.amendment, "before_any_score": False}, self.raw)

    def test_accepted_divergence_names_the_field_verify_emits(self):
        accepted = [d for d in self.raw["accepted_divergences"] if d["observed"] == 8]
        self.assertEqual(len(accepted), 1)
        self.assertEqual((accepted[0]["field"], accepted[0]["decided_by"]),
                         ("duplicates.identical_originals_not_listed", "Eduardo"))
        sha = {"A": "1", "B": "1", "C": "2", "D": "2"}
        _, divergences = isic2020.exact_hash_check([("A", "B")], sha)
        self.assertEqual([(d["field"], d["observed"]) for d in divergences],
                         [("duplicates.identical_originals_not_listed", 1)])
        isic2020.apply_acceptance(divergences, [{**accepted[0], "observed": 1}])
        self.assertEqual(isic2020.pending(divergences), [])
        for text in (CONFIG.read_text(), AMENDMENT.read_text()):
            self.assertNotRegex(text, r"ISIC_\d{7}|IP_\d{7}|IL_\d{7}")

    def test_extra_identical_pair_across_patients_merges_their_groups(self):
        records = {"A": {"patient_id": "P1"}, "B": {"patient_id": "P2"}, "C": {"patient_id": "P3"}}
        patient_to_group, _, stats = groups.build_groups(records, [("A", "B")], "g-")
        self.assertEqual(patient_to_group["P1"], patient_to_group["P2"])
        self.assertNotEqual(patient_to_group["P1"], patient_to_group["P3"])
        self.assertEqual(stats["cross_patient_pairs"], 1)


class ExtraPairsRelease(SyntheticRelease):
    """Two byte-identical pairs that are NOT in the official list (one crosses patients)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        names = [row["image_name"] for row in self.rows]
        self.extra = [(names[10], names[11]), (names[20], names[30])]
        for a, b in self.extra:
            self.data[b] = self.data[a]
            for key in ("target", "diagnosis", "benign_malignant"):
                self.row(b)[key] = self.row(a)[key]


class VerifiedRootToPoolsTests(AcquisitionE2E):
    n_patients = 60

    def make_release(self):
        return ExtraPairsRelease(n_patients=self.n_patients)

    def global_cli(self, *argv, code=0):
        import contextlib
        import io
        from augment_atlas import global_cli
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = global_cli.main(list(argv))
        self.assertEqual(code, result, err.getvalue())
        return json.loads(out.getvalue()) if code == 0 else err.getvalue()

    def test_extra_pairs_block_until_accepted_then_feed_sealed_pools(self):
        self.assertEqual(self.cli("full"), 3, "verify stops on the pending divergence")
        divergences = json.loads((self.root / "manifest" / "divergences.json").read_text())
        extra = [d for d in divergences["divergences"] if d["field"] == "duplicates.identical_originals_not_listed"]
        self.assertEqual([(d["observed"], d["accepted"]) for d in extra], [(2, False)])
        self.assertFalse((self.root / "manifest" / "READY.json").exists())
        accepted = {"field": "duplicates.identical_originals_not_listed", "observed": 2,
                    "resolution": "fixture: accepted, merged in the union-find", "decided_by": "Eduardo"}
        identity = config_mod.identity_of(self.raw)
        self.write_config(accepted_divergences=[accepted])
        self.assertEqual(config_mod.identity_of(self.raw), identity, "acceptance is operational")
        self.assertEqual(self.cli("verify"), 0, self.last_stderr)
        result = json.loads(self.last_stdout)["result"]
        records = {row["image_name"]: {"patient_id": row["patient_id"]} for row in self.release.rows}
        _, _, expected = groups.build_groups(records, self.release.pairs + self.release.extra, "g-")
        self.assertLess(expected["groups"], self.n_patients - 1, "the extra pairs merged more patients")
        self.assertEqual(result["counts"]["groups"], expected["groups"])
        self.assertEqual(result["duplicates"]["exact_hash_pairs"], 4)

        ready = json.loads((self.root / "manifest" / "READY.json").read_text())
        source = self.dir / "source.json"
        source.write_text(json.dumps({"adapter": "manifest_csv", "manifest": "manifest/manifest.csv",
                                      "manifest_sha256": ready["files"]["manifest.csv"]}))
        amendment = self.dir / "amendment.json"
        amendment.write_text(json.dumps(amendment_for(self.raw)))
        base = ["--data-root", str(self.root), "--source", str(source), "--dataset", "isic2020"]
        spec_path = self.dir / "split-spec.json"
        spec = self.global_cli("split-spec", *base, "--acquisition-root", str(self.root),
                               "--split-amendment", str(amendment), "--out", str(spec_path))
        self.assertEqual(spec["dataset"]["version"], "isic2020-derived-512-v1")
        text = spec_path.read_text()
        self.assertNotRegex(text, r"SYNTH_\d{7}|PSYNTH_\d{4}|LSYNTH_\d{7}|isic20-g-")
        made = self.global_cli("make-pools", *base, "--spec", str(spec_path), "--acquisition-config",
                               str(self.config_path), "--split-amendment", str(amendment))
        imported = self.global_cli("import-pools", *base, "--file", made["path"],
                                   "--expected-manifest", made["global_manifest_sha256"])
        pools = gd._read(Path(made["path"]).parent / f"pools-{imported['pools_sha256']}.json")
        self.assertEqual(pools["partition_provenance"]["seeds"], amendment_for(self.raw)["seeds"])
        manifest = (self.root / "manifest" / "manifest.csv").read_text().splitlines()[1:]
        group_of = {line.split(",")[0]: line.split(",")[1] for line in manifest}
        folds = [set(f["test_ids"]) for f in pools["evaluation_folds"]]
        lists = {name: set(pools["lists"][name]) for name in ("D_fit", "D_select")}
        placement = {}
        for name, ids in [*lists.items(), *((f"E{i}", ids) for i, ids in enumerate(folds))]:
            for item in ids:
                placement.setdefault(group_of[item], set()).add(name)
        self.assertTrue(all(len(where) == 1 for where in placement.values()), "a group never spans two pools")
        audit = self.global_cli("audit-pools", *base, "--pools-sha256", imported["pools_sha256"],
                                "--expected-manifest", made["global_manifest_sha256"])
        self.assertIs(audit["structural_valid"], True)
        # Seeds/fractions that differ from the amendment are refused before anything is written.
        loose = json.loads(spec_path.read_text())
        loose["seeds"]["evaluation"] = 99
        loose_path = self.dir / "loose.json"
        loose_path.write_text(json.dumps(loose))
        error = self.global_cli("make-pools", *base, "--spec", str(loose_path), "--acquisition-config",
                                str(self.config_path), "--split-amendment", str(amendment), code=2)
        self.assertIn("split_spec_invalid", error)
        self.assertEqual(file_hash(self.root / "manifest" / "manifest.csv"), ready["files"]["manifest.csv"])


if __name__ == "__main__":
    unittest.main()
