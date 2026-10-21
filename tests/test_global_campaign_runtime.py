"""Slice S3: the phase=final campaign gate of the host runtime, end to end without Docker.

Real local Git everywhere it matters: the pinned code checkout has a freeze
commit (src/, scripts/, docker/) and a final commit that only adds
``configs/global/**`` (Q6 policy ``same_code_identity_v1``, PENDING Eduardo);
the queue is a bare local repository with the freeze and campaign anchors and
one ``seed_final`` job. Docker/GPU are never called: ``commit``/``image`` are
set the way ``preflight`` would set them. Synthetic images, no torch.
"""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import global_campaign as gc
from augment_atlas import global_jobs as gj
from augment_atlas import runtime
from augment_atlas.contracts import canonical_hash as H
from augment_atlas.queue import GitQueue
from augment_atlas.storage import digest
from test_global_trainer import DATASET, GlobalFixture, final_policy, presets, sha256_bytes

IMAGE = "sha256:" + "a" * 64
STAGES = [{"stage": 1, "seeds": [11]}, {"stage": 2, "seeds": [13, 19]}]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


class CampaignGateWorld:
    def __init__(self, root):
        root = Path(root)
        self.fx = GlobalFixture(root)
        self.data = self.fx.root
        self.code = root / "code"
        self.code.mkdir()
        git(self.code, "init", "-q")
        for path, text in {"src/augment_atlas/placeholder.py": "x = 1\n", "scripts/worker.sh": "echo\n",
                           "docker/Dockerfile": "FROM scratch\n"}.items():
            (self.code / path).parent.mkdir(parents=True, exist_ok=True)
            (self.code / path).write_text(text, encoding="utf-8")
        git(self.code, "add", "-A")
        git(self.code, "commit", "-q", "-m", "code at the freeze")
        self.freeze_commit = git(self.code, "rev-parse", "HEAD")
        pools = self.fx.pools
        self.freeze = gj.build_external_freeze(
            self.fx.rows, pools, expected_pools_sha256=pools["sha256"], recipe=self.fx.recipe,
            source=gj.Q2_EXTERNAL_SOURCE, runtime=dict(self.fx.runtime, commit=self.freeze_commit),
            final_policy=final_policy(), frozen_at_utc="2026-10-19T12:00:00Z")
        self.fx.write_doc(self.data, "external-freeze", self.freeze)
        self.source = {"adapter": "manifest_csv", "manifest": "manifest.csv",
                       "manifest_sha256": sha256_bytes((self.data / "manifest.csv").read_bytes())}
        configs = gj.build_final_configs(
            self.fx.rows, pools, expected_pools_sha256=pools["sha256"], dataset_source=self.source,
            freeze=self.freeze, freeze_kind="external_fixed", expected_freeze_sha256=self.freeze["sha256"],
            augmentation_presets=presets(), seeds=[11])
        self.manifest = gj.materialize_configs("final", self.code, configs, pools_sha256=pools["sha256"],
                                               freeze_sha256=self.freeze["sha256"])
        git(self.code, "add", "-A")
        git(self.code, "commit", "-q", "-m", "final configs")
        self.code_commit = git(self.code, "rev-parse", "HEAD")
        identity = gc.derive_code_identity(self.code, freeze_commit=self.freeze_commit, code_commit=self.code_commit,
                                           image_id=IMAGE)
        self.campaign = gc.seal_campaign(
            self.fx.rows, pools, expected_pools_sha256=pools["sha256"], freeze=self.freeze,
            freeze_kind="external_fixed", expected_freeze_sha256=self.freeze["sha256"], dataset_source=self.source,
            augmentation_presets=presets(), execution_stages=copy.deepcopy(STAGES), code_commit=self.code_commit,
            image_id=IMAGE, code_identity=identity, label="exploratory", caveats=list(gc.CAMPAIGN_CAVEATS),
            sealed_at_utc="2026-10-19T13:00:00Z", queue_records=[], expected_manifest_sha256=self.manifest["sha256"])
        self.campaign_file = self.fx.write_doc(self.data, "campaign", self.campaign)
        self.remote = root / "queue.git"
        subprocess.run(["git", "init", "--bare", "-q", str(self.remote)], check=True, capture_output=True)
        self.queue = GitQueue(root / "operator-queue", self.remote)
        self.c_freeze = self.queue.publish_anchor(gc.build_anchor("recipe_freeze", self.freeze))["commit"]
        self.c_campaign = self.queue.publish_anchor(gc.build_anchor("campaign", self.campaign))["commit"]
        self.queue.seed_final([self.spec(0)], self.campaign["sha256"])
        cache = root / "torch-home" / "hub" / "checkpoints" / runtime.IMAGENET_V2_CHECKPOINT
        cache.parent.mkdir(parents=True)
        cache.write_bytes(self.fx.weights)
        self.settings = {"code": str(self.code), "data": str(self.data), "phase": "final", "gpu": "0",
                         "torch_cache": str(root / "torch-home"), "campaign_sha256": self.campaign["sha256"],
                         "campaign_commit": self.c_campaign, "queue_checkout": str(root / "worker-queue"),
                         "remote": str(self.remote)}

    def spec(self, fold, *, campaign_sha256=None, code_commit=None):
        name = f"{gj.config_dir(DATASET, 'final')}/none-f{fold}-s11.json"
        cfg = json.loads((self.code / name).read_text(encoding="utf-8"))
        view = gj.derive_final_view_doc(self.fx.rows, self.fx.pools, fold, expected_pools_sha256=self.fx.pools["sha256"],
                                        freeze_sha256=self.freeze["sha256"])
        return gj.final_job_spec(cfg, view, config_path=name, config_sha256=digest(self.code / name),
                                 code_commit=code_commit or self.code_commit,
                                 campaign_sha256=campaign_sha256 or self.campaign["sha256"])

    def host(self, **changes):
        host = runtime.DockerRuntime({**self.settings, **changes})
        host.commit, host.image, host.environment = self.code_commit, IMAGE, {"opencv": None}
        return host


class CampaignGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="atlas-campaign-gate-")
        cls.w = CampaignGateWorld(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_gate_passes_only_with_audited_campaign_and_chronology(self):
        host = self.w.host()
        with patch("augment_atlas.runtime.command") as docker:
            name, cfg, _, mount = host.global_config_for(self.w.spec(0))
            docker.assert_not_called()
        self.assertEqual((name, cfg["phase"], mount["kind"]), (self.w.spec(0)["config"], "final", "data_root"))
        (report,) = host._chronology.values()
        self.assertEqual((report["C_freeze"], report["C_campaign"]), (self.w.c_freeze, self.w.c_campaign))
        self.assertEqual((report["final_jobs"], report["recipe_source"]), (1, "external_fixed"))
        self.assertEqual((report["execution_verified"], report["scientific_eligible"]), (False, False))
        (audit, _), = host._campaign_audits.values()
        self.assertEqual((audit["q6_policy"], audit["q6_policy_status"]),
                         ("same_code_identity_v1", "pending_eduardo_confirmation"))
        self.assertEqual(host.config_for(self.w.spec(0))[0], self.w.spec(0)["config"])  # cached per HEAD

    def test_preflight_requires_pin_commit_and_queue_before_external_commands(self):
        for missing in ("campaign_sha256", "campaign_commit", "queue_checkout", "remote"):
            settings = {k: v for k, v in self.w.settings.items() if k != missing}
            with self.subTest(missing=missing), patch("augment_atlas.runtime.command") as external:
                with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
                    runtime.DockerRuntime(settings).preflight()
                external.assert_not_called()

    def test_gate_refusals(self):
        cases = {
            "job not seeded in the queue HEAD": (dict(), lambda: self.w.spec(1), "campaign_anchor_missing"),
            "job bound to another campaign": (dict(), lambda: self.w.spec(0, campaign_sha256=H("other")),
                                              "campaign_anchor_missing"),
            "worker pinned to another campaign": (dict(campaign_sha256=H("other")), lambda: self.w.spec(0),
                                                  "campaign_anchor_missing"),
            "rewritten history (wrong C_campaign)": (dict(campaign_commit=self.w.c_freeze), lambda: self.w.spec(0),
                                                     "chronology_invalid"),
        }
        for label, (changes, spec, code) in cases.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, f"^{code}:"):
                self.w.host(**changes).global_config_for(spec())
        host = self.w.host()
        host.image = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
            host.global_config_for(self.w.spec(0))
        host = self.w.host()
        host.commit = self.w.freeze_commit
        with self.assertRaisesRegex(ValueError, "^code_identity_mismatch:"):
            host.global_config_for(self.w.spec(0))

    def test_host_campaign_document_must_be_the_anchored_one(self):
        original = self.w.campaign_file.read_bytes()
        try:
            tampered = json.loads(original)
            tampered["label_reasons"] = []
            self.w.campaign_file.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "^campaign_invalid:"):  # stale checksum
                self.w.host().global_config_for(self.w.spec(0))
            tampered.pop("sha256")
            tampered["sha256"] = H(tampered)
            self.w.campaign_file.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):  # re-signed: not the pinned hash
                self.w.host().global_config_for(self.w.spec(0))
            self.w.campaign_file.unlink()
            with self.assertRaisesRegex(ValueError, "^campaign_anchor_missing:"):
                self.w.host().global_config_for(self.w.spec(0))
        finally:
            self.w.campaign_file.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
