"""Host-side contract tests. Docker/GPU/network interactions here are MOCKS.

Only the Linux flock test uses a real second process. These tests do not provide
evidence of a Docker container, CUDA kernel, real training or GitHub execution.
"""
import contextlib
import copy
import io
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from augment_atlas import runtime
from augment_atlas.queue import QueueError, StaleAttempt, parse_record, render_record
from augment_atlas.schema import new_record
from augment_atlas.storage import atomic_json, digest
from augment_atlas.contracts import canonical_hash

REQUIRED_RESULTS = ("checkpoint.pt", "best.pt", "splits.json", "config.json", "environment.json",
                    "metrics.json", "results.json", "progress.json")


def owned_record(version=1, state="claimed"):
    return {"job_id": "a" * 64, "attempt_id": "attempt-fixture", "token": "public-owner-fixture",
            "state": state, "version": version,
            "spec": {"code_commit": "b" * 40, "config_sha256": "c" * 64}}


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="augment-atlas-runtime-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)


class ProbeTests(unittest.TestCase):
    def test_probe_optional_opencv_reporting_uses_only_mock_ml_modules(self):
        """Execute the container probe text with MOCK modules, not CUDA/ML code."""
        for version,distribution in ((None,None),('4.12.0',None),('4.12.0','4.12.0.88')):
            with self.subTest(opencv=version,distribution=distribution):
                torch=Mock()
                torch.__version__='torch-fixture'
                torch.version.cuda='cuda-fixture'
                torch.cuda.is_available.return_value=True
                torch.cuda.get_device_capability.return_value=(5,2)
                torch.isfinite.return_value.all.return_value=True
                torchvision=Mock()
                torchvision.__version__='torchvision-fixture'
                cv2=Mock()
                cv2.__version__=version
                modules={'torch':torch,'torchvision':torchvision,'numpy':Mock(),'PIL':Mock(),
                         'skimage':Mock(),'cv2':cv2}
                output=io.StringIO()
                with patch.dict(sys.modules,modules), \
                     patch('importlib.util.find_spec',return_value=None if version is None else object()), \
                     patch('importlib.metadata.version',return_value=distribution,
                           side_effect=importlib.metadata.PackageNotFoundError if distribution is None else None), \
                     contextlib.redirect_stdout(output):
                    exec(runtime.PROBE,{})
                observed=json.loads(output.getvalue())
                self.assertEqual(version,observed['opencv'])
                self.assertEqual(distribution,observed['opencv_distribution'])


class SpoolTests(TemporaryTest):
    def test_mock_network_outage_keeps_durable_ids_order_and_terminal(self):
        path = self.root / "events.json"
        initial = owned_record()
        spool = runtime.Spool(path, initial)
        spool.add("running", epoch=0)
        spool.add("running", epoch=1)
        spool.add("succeeded", completed_sha256="d" * 64)
        events = copy.deepcopy(spool.events)
        unavailable = Mock()
        unavailable.update.side_effect = QueueError("synthetic network unavailable")
        with self.assertRaises(QueueError):
            spool.flush(unavailable)
        self.assertEqual(events, spool.events)
        recovered = runtime.Spool(path, initial)
        self.assertEqual(events, recovered.events)
        available = Mock()
        available.update.side_effect = [owned_record(2, "running"), owned_record(3, "running"),
                                        owned_record(4, "succeeded")]
        final = recovered.flush(available)
        self.assertEqual("succeeded", final["state"])
        self.assertEqual([], runtime.Spool(path, initial).events)
        calls = available.update.call_args_list
        self.assertEqual([1, 2, 3], [call.args[3] for call in calls])
        self.assertEqual(events, [call.args[4] for call in calls])
        self.assertEqual(3, len({event["event_id"] for event in events}))

    def test_mock_stale_attempt_is_not_silenced_or_removed(self):
        spool = runtime.Spool(self.root / "events.json", owned_record())
        spool.add("running", epoch=1)
        queue = Mock()
        queue.update.side_effect = StaleAttempt("synthetic ownership changed")
        with self.assertRaises(StaleAttempt):
            spool.flush(queue)
        self.assertEqual(1, len(spool.events))
        self.assertEqual(spool.events, runtime.Spool(spool.path, owned_record()).events)

    def test_restart_after_partial_flush_retains_acknowledged_version(self):
        """Crash window: the second event failed after the first was removed."""
        path = self.root / "events.json"
        initial = owned_record()
        spool = runtime.Spool(path, initial)
        spool.add("running", epoch=0)
        spool.add("succeeded", completed_sha256="d" * 64)
        terminal = copy.deepcopy(spool.events[1])
        queue = Mock()
        queue.update.side_effect = [owned_record(2, "running"), QueueError("synthetic outage")]
        with self.assertRaises(QueueError):
            spool.flush(queue)
        recovered = runtime.Spool(path, initial)  # active.json has not been saved yet
        self.assertEqual([terminal], recovered.events)
        self.assertEqual(2, recovered.record["version"], "spool must persist acknowledgement with queue version")
        online = Mock()
        online.update.return_value = owned_record(3, "succeeded")
        self.assertEqual("succeeded", recovered.flush(online)["state"])
        self.assertEqual(2, online.update.call_args.args[3])


class ArtifactTests(TemporaryTest):
    def write_results(self):
        from augment_atlas.data import load_records, prepare_splits
        fixture = Path(__file__).resolve().parents[1] / "configs" / "smoke" / "synthetic.json"
        config = json.loads(fixture.read_text(encoding="utf-8"))
        config_hash = canonical_hash(config)
        atomic_json(self.root / "config.json", config)
        (self.root / "splits.json").unlink(missing_ok=True)  # Reset corruption from a previous subtest.
        splits = prepare_splits(config, load_records(config), self.root)
        for name in ("checkpoint.pt", "best.pt"):
            # Opaque nonempty placeholders: this host test does not deserialize torch.
            (self.root / name).write_bytes(b"synthetic checkpoint fixture; not a torch checkpoint\n")
        atomic_json(self.root / "environment.json", {"schema_version": 1, "source": "mock-fixture",
                    "config_sha256": config_hash, "splits_sha256": splits["sha256"]})
        atomic_json(self.root / "metrics.json", [{"epoch": epoch, "selection_loss": 0.8 / epoch}
                                                 for epoch in range(1, config["training"]["epochs"] + 1)])
        atomic_json(self.root / "progress.json", {"schema_version": 1, "status": "completed",
                    "epoch": config["training"]["epochs"], "epochs": config["training"]["epochs"],
                    "samples_processed": 12,
                    "config_sha256": config_hash, "splits_sha256": splits["sha256"]})
        atomic_json(self.root / "results.json", {"schema_version": 1, "phase": "smoke",
                    "scientific_status": "provisional", "best_epoch": 1, "selection_loss": 0.4,
                    "test": {"samples": len(splits["lists"]["test"]), "loss": 0.5, "accuracy": 0.5},
                    "config_sha256": config_hash, "splits_sha256": splits["sha256"]})
        files = {name: digest(self.root / name) for name in REQUIRED_RESULTS}
        atomic_json(self.root / "completed.json", {"schema_version": 1, "status": "completed",
                    "epoch": config["training"]["epochs"], "config_sha256": config_hash,
                    "splits_sha256": splits["sha256"], "artifacts": files,
                    "artifact_sizes": {name: (self.root / name).stat().st_size for name in REQUIRED_RESULTS}})
        return files

    def test_missing_completion_manifest_rejected(self):
        with self.assertRaises(ValueError):
            runtime.validate_artifacts(self.root)

    def test_fixture_artifact_hashes_validated(self):
        files = self.write_results()
        summary = runtime.validate_artifacts(self.root)
        self.assertEqual(len(files), summary["artifact_count"])
        self.assertEqual(digest(self.root / "completed.json"), summary["completed_sha256"])

    def test_terminal_summary_finishes_markdown_without_periodic_progress_event(self):
        files=self.write_results()
        summary=runtime.validate_artifacts(self.root)
        self.assertEqual((2,2,12),(summary['epoch'],summary['epochs'],summary['samples_processed']))
        self.assertEqual(len(files),summary['artifact_count'])
        self.assertEqual(digest(self.root/'completed.json'),summary['completed_sha256'])
        record=new_record({'protocol_version':'smoke-v1','dataset':'synthetic','dataset_version':'v1',
                           'split_hash':'fixture-split','recipe_hash':'fixture-recipe','phase':'smoke',
                           'fold':0,'train_seed':3,'arm':'none','code_commit':'b'*40})
        record.update(state='succeeded',version=3,seq=3,attempt_id='attempt-fixture',token='token-fixture',
                      owner={'server':'host-fixture'})
        metadata={'at':'2026-10-16T03:00:00+00:00','attempt_id':'attempt-fixture'}
        record['history']=[{**metadata,'seq':1,'state':'claimed'},
                           {**metadata,'seq':2,'state':'running','epoch':0,'epochs':2},
                           {**metadata,'seq':3,'state':'succeeded',**summary}]
        markdown=render_record(record)
        self.assertIn('Epoch: 2/2 |',markdown)
        self.assertEqual(record,parse_record(markdown))

    def test_terminal_summary_excludes_unvalidated_optional_sample_count(self):
        for samples in (True,-1,2.5,'12',None):
            with self.subTest(samples=samples):
                self.write_results()
                progress=json.loads((self.root/'progress.json').read_text(encoding='utf-8'))
                atomic_json(self.root/'progress.json',{**progress,'samples_processed':samples})
                self.reseal('progress.json')
                self.assertNotIn('samples_processed',runtime.validate_artifacts(self.root))

    def test_corrupt_or_missing_required_artifact_rejected(self):
        for name in ("checkpoint.pt", "progress.json", "splits.json"):
            for operation in ("corrupt", "remove"):
                with self.subTest(name=name, operation=operation):
                    self.write_results()
                    target = self.root / name
                    if operation == "corrupt":
                        target.write_bytes(b"corrupt fixture")
                    else:
                        target.unlink()
                    with self.assertRaises(ValueError):
                        runtime.validate_artifacts(self.root)

    def test_path_escape_rejected_even_with_correct_external_hash(self):
        self.write_results()
        outside = self.root / "outside.bin"
        outside.write_bytes(b"outside")
        output = self.root / "attempt"
        output.mkdir()
        atomic_json(output / "completed.json", {"artifacts": {"../outside.bin": digest(outside)}})
        with self.assertRaises(ValueError):
            runtime.validate_artifacts(output)

    def test_required_artifacts_need_checksum_coverage(self):
        self.write_results()
        extra = self.root / "unrelated.txt"
        extra.write_bytes(b"unrelated fixture")
        atomic_json(self.root / "completed.json", {"artifacts": {"unrelated.txt": digest(extra)}})
        with self.assertRaises(ValueError):
            runtime.validate_artifacts(self.root)

    def reseal(self, name):
        """Update transport checksums so a semantic validation failure is isolated."""
        path = self.root / "completed.json"
        completion = json.loads(path.read_text(encoding="utf-8"))
        completion["artifacts"][name] = digest(self.root / name)
        completion["artifact_sizes"][name] = (self.root / name).stat().st_size
        atomic_json(path, completion)

    def test_rehashed_incomplete_progress_and_split_corruption_rejected(self):
        for name, mutation in (("progress.json", {"status": "running"}),
                               ("progress.json", {"epoch": 1}),
                               ("progress.json", {"epoch": 2.0}),
                               ("splits.json", {"sha256": "0" * 64}),
                               ("environment.json", {"config_sha256": "0" * 64})):
            with self.subTest(name=name, mutation=mutation):
                self.write_results()
                value = json.loads((self.root / name).read_text(encoding="utf-8"))
                atomic_json(self.root / name, {**value, **mutation})
                self.reseal(name)
                with self.assertRaises(ValueError):
                    runtime.validate_artifacts(self.root)

    def test_rehashed_invalid_evaluation_and_empty_checkpoint_rejected(self):
        for mutation in ({"samples": 0}, {"samples": True}, {"loss": float("nan")}, {"accuracy": 1.5}):
            with self.subTest(mutation=mutation):
                self.write_results()
                path = self.root / "results.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["test"].update(mutation)
                # Deliberately encode NaN to test rejection of untrusted JSON values.
                path.write_text(json.dumps(value), encoding="utf-8")
                self.reseal("results.json")
                with self.assertRaises(ValueError):
                    runtime.validate_artifacts(self.root)
        self.write_results()
        (self.root / "checkpoint.pt").write_bytes(b"")
        self.reseal("checkpoint.pt")
        with self.assertRaises(ValueError):
            runtime.validate_artifacts(self.root)

    def test_valid_artifacts_must_belong_to_claimed_job(self):
        self.write_results()
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        splits = json.loads((self.root / "splits.json").read_text(encoding="utf-8"))
        spec = {"split_hash": splits["sha256"], "train_seed": config["training"]["seed"],
                "arm": config["augmentation"]["arm"]}
        runtime.validate_artifacts(self.root, spec)
        for change in ({"split_hash": "0" * 64}, {"train_seed": spec["train_seed"] + 1}, {"arm": "basic"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runtime.validate_artifacts(self.root, {**spec, **change})

    def pretrained_results(self):
        """Reconcile the fixture hashes after changing the initialization contract."""
        self.write_results()
        config=json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        config['model'].update(weights='IMAGENET1K_V2',weights_sha256='e'*64)
        atomic_json(self.root/'config.json',config)
        config_hash=canonical_hash(config)
        for name in ('completed.json','environment.json','progress.json','results.json'):
            value=json.loads((self.root/name).read_text(encoding='utf-8'))
            value['config_sha256']=config_hash
            if name=='environment.json':value.update(weights='IMAGENET1K_V2',weights_sha256='e'*64)
            atomic_json(self.root/name,value)
        for name in ('config.json','environment.json','progress.json','results.json'):self.reseal(name)

    def test_pretrained_environment_must_match_config_and_explicit_spec(self):
        self.pretrained_results()
        config=json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        splits=json.loads((self.root/'splits.json').read_text(encoding='utf-8'))
        spec={'split_hash':splits['sha256'],'train_seed':config['training']['seed'],'arm':config['augmentation']['arm']}
        runtime.validate_artifacts(self.root,{**spec,'weights_sha256':'e'*64})
        for mismatch in ({'weights_sha256':'f'*64},{'model':{'weights_sha256':'f'*64}}):
            with self.subTest(mismatch=mismatch),self.assertRaises(ValueError):
                runtime.validate_artifacts(self.root,{**spec,**mismatch})
        for mutation in ({'weights':'none'},{'weights_sha256':'f'*64},{'weights_sha256':None}):
            with self.subTest(mutation=mutation):
                self.pretrained_results()
                env=json.loads((self.root/'environment.json').read_text(encoding='utf-8'))
                atomic_json(self.root/'environment.json',{**env,**mutation})
                self.reseal('environment.json')
                with self.assertRaises(ValueError):runtime.validate_artifacts(self.root)


class ConfigTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        fixture = Path(__file__).resolve().parents[1] / "configs" / "smoke" / "synthetic.json"
        self.cfg = json.loads(fixture.read_text(encoding="utf-8"))
        self.code = self.root / "code"
        self.data = self.root / "data"
        self.path = self.code / "configs" / "smoke" / "fixture.json"
        self.data.mkdir()
        atomic_json(self.path, self.cfg)
        self.settings = {"code": str(self.code), "data": str(self.data), "phase": "smoke"}
        self.runtime = runtime.DockerRuntime(self.settings)

    def job(self):
        from augment_atlas.cli import recipe_identity
        from augment_atlas.data import load_records, prepare_splits
        with tempfile.TemporaryDirectory(prefix="augment-atlas-split-fixture-") as directory:
            splits = prepare_splits(self.cfg, load_records(self.cfg), directory)
        return {"phase": "smoke", "config": "configs/smoke/fixture.json", "config_sha256": digest(self.path),
                "train_seed": self.cfg["training"]["seed"], "arm": self.cfg["augmentation"]["arm"],
                "split_hash": splits["sha256"], "recipe_hash": recipe_identity(self.cfg),
                "dataset": self.cfg["dataset"]["id"], "dataset_version": self.cfg["dataset"]["version"],
                "protocol_version": self.cfg["protocol_version"], "fold": self.cfg["splits"]["fold"]}

    def test_actual_synthetic_config_validation_uses_no_docker(self):
        torch_before = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
        name, cfg = self.runtime.config_for(self.job())
        self.assertEqual("configs/smoke/fixture.json", name)
        self.assertEqual("smoke", cfg["phase"])
        self.assertEqual(torch_before, {name for name in sys.modules if name == "torch" or name.startswith("torch.")})

    def test_invalid_config_phase_and_checksum_rejected(self):
        for field, value in (("phase", "final"), ("scientific_status", "approved")):
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.cfg)
                invalid[field] = value
                atomic_json(self.path, invalid)
                with self.assertRaises(ValueError):
                    self.runtime.config_for(self.job())
        atomic_json(self.path, self.cfg)
        wrong = self.job()
        wrong["config_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.runtime.config_for(wrong)

    def test_bad_path_and_seed_rejected(self):
        for change in ({"config": "configs/final/no.json"}, {"config": "configs/smoke/../../../outside.json"},
                       {"train_seed": self.cfg["training"]["seed"] + 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.runtime.config_for({**self.job(), **change})

    def test_stable_identity_must_match_loaded_configuration(self):
        for field in ("split_hash", "recipe_hash", "dataset", "dataset_version", "protocol_version", "fold"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.runtime.config_for({**self.job(), field: 999 if field == "fold" else "mismatch-fixture"})

    def test_final_phase_candidate_is_incompatible_before_config_validation(self):
        self.runtime.commit = "b" * 40
        self.runtime.image = "sha256:fixture"
        self.runtime.memory_mb = 12000
        self.runtime.gpu_uuid = "GPU-fixture"
        self.settings["datasets"] = [self.cfg["dataset"]["id"]]
        self.assertFalse(self.runtime.compatible({**self.job(), "phase": "final", "code_commit": self.runtime.commit}))

    def test_data_root_environment_restored_after_validation_error(self):
        with patch.dict(os.environ, {"ATLAS_DATA_ROOT": "original-fixture-root"}):
            with patch("augment_atlas.data.load_records", side_effect=ValueError("synthetic missing dataset")):
                with self.assertRaises(ValueError):
                    self.runtime.config_for(self.job())
            self.assertEqual("original-fixture-root", os.environ["ATLAS_DATA_ROOT"])

    def test_preflight_final_phase_fails_without_external_commands(self):
        self.settings["phase"] = "final"
        with patch("augment_atlas.runtime.command") as external:
            with self.assertRaises(ValueError):
                self.runtime.preflight()
            external.assert_not_called()

    def cached_weights_fixture(self):
        self.settings['torch_cache']=str(self.root/'torch-home')
        cached=Path(self.settings['torch_cache'])/'hub'/'checkpoints'/'resnet50-11ad3fa6.pth'
        cached.parent.mkdir(parents=True,exist_ok=True)
        cached.write_bytes(b'synthetic host cache bytes; not torch weights\n'*31)
        self.cfg['model'].update(weights='IMAGENET1K_V2',weights_sha256=digest(cached))
        atomic_json(self.path,self.cfg)
        return cached

    def test_matching_pretrained_host_cache_full_hash_is_validated_without_torch(self):
        cached=self.cached_weights_fixture()
        original=cached.read_bytes()
        with patch('augment_atlas.runtime.digest',wraps=digest) as streamed:
            _,cfg=self.runtime.config_for(self.job())
        self.assertEqual('IMAGENET1K_V2',cfg['model']['weights'])
        self.assertIn(cached.resolve(),[Path(call.args[0]) for call in streamed.call_args_list])
        self.assertEqual(original,cached.read_bytes())

    def test_missing_or_mismatched_pretrained_cache_never_falls_back(self):
        for failure in ('setting','missing','checksum','config-checksum'):
            with self.subTest(failure=failure):
                cached=self.cached_weights_fixture()
                if failure=='setting':self.settings.pop('torch_cache')
                elif failure=='missing':cached.unlink()
                elif failure=='checksum':cached.write_bytes(b'different synthetic bytes')
                else:
                    self.cfg['model'].pop('weights_sha256')
                    atomic_json(self.path,self.cfg)
                with self.assertRaises(ValueError):self.runtime.config_for(self.job())

    def test_explicit_spec_weight_hash_mismatch_is_rejected(self):
        self.cached_weights_fixture()
        with self.assertRaises(ValueError):
            self.runtime.config_for({**self.job(),'weights_sha256':'0'*64})

    def test_missing_pretrained_cache_fails_real_worker_selection_before_mock_claim(self):
        self.cached_weights_fixture().unlink()
        self.assert_worker_selection_rejected_before_mock_claim()

    def assert_worker_selection_rejected_before_mock_claim(self):
        settings={**self.settings,'runs':str(self.root/'runs'),'queue_checkout':str(self.root/'queue'),
                  'remote':'mock-remote','server':'host-fixture','gpu':'0'}
        self.runtime.gpu_uuid='GPU-fixture'
        queue=Mock()
        queue.list_jobs.return_value=[{'job_id':'a'*64,'state':'pending','spec':self.job()}]
        with patch('augment_atlas.runtime.DockerRuntime',return_value=self.runtime), \
             patch.object(self.runtime,'preflight'),patch.object(self.runtime,'probe'), \
             patch.object(self.runtime,'compatible',return_value=True), \
             patch('augment_atlas.runtime.GitQueue',return_value=queue), \
             patch('augment_atlas.runtime.gpu_lock',return_value=contextlib.nullcontext()), \
             patch('augment_atlas.runtime.command') as docker:
            with self.assertRaises(ValueError):runtime.worker(settings,once=True)
        queue.claim.assert_not_called()
        docker.assert_not_called()

    def candidate_backend_fixture(self):
        self.cfg['protocol_version']='candidate-v2'
        self.cfg['augmentation']['backend']='candidate-opencv-v2'
        atomic_json(self.path,self.cfg)

    def test_candidate_backend_requires_opencv_image_probe_before_mock_claim(self):
        self.candidate_backend_fixture()
        for environment in ({'opencv':None}, {'opencv':'4.12.0'},
                            {'opencv':'4.12.0','opencv_distribution':'4.11.0.86'},
                            {'opencv':'4.11.0','opencv_distribution':'4.12.0.88'}):
            with self.subTest(environment=environment):
                self.runtime.environment=environment
                self.assert_worker_selection_rejected_before_mock_claim()

    def test_candidate_backend_accepts_reported_opencv_and_legacy_works_without_it(self):
        self.runtime.environment={'opencv':None}
        self.runtime.config_for(self.job())
        self.candidate_backend_fixture()
        self.runtime.environment={'opencv':'4.12.0','opencv_distribution':'4.12.0.88'}
        self.runtime.config_for(self.job())

    def test_mock_docker_mounts_verified_pretrained_cache_readonly_and_sets_torch_home(self):
        self.cached_weights_fixture()
        self.runtime.commit='b'*40
        self.runtime.image='sha256:fixture'
        self.settings['gpu']='0'
        with patch('augment_atlas.runtime.command') as docker, \
             patch('augment_atlas.runtime.os.getuid',return_value=1000,create=True), \
             patch('augment_atlas.runtime.os.getgid',return_value=1000,create=True):
            self.runtime.create('container-fixture',self.job(),self.root/'output')
        args=docker.call_args.args[0]
        mount=f"type=bind,src={Path(self.settings['torch_cache']).resolve()},dst=/torch-cache,readonly"
        self.assertIn(mount,args)
        self.assertIn('TORCH_HOME=/torch-cache',args)
        self.assertIn('--network',args)
        self.assertEqual('none',args[args.index('--network')+1])

    def test_mock_random_initialization_preserves_cache_optional_legacy_command(self):
        self.runtime.commit='b'*40
        self.runtime.image='sha256:fixture'
        self.settings['gpu']='0'
        self.settings['torch_cache']=str(self.root/'does-not-exist')
        with patch('augment_atlas.runtime.command') as docker, \
             patch('augment_atlas.runtime.os.getuid',return_value=1000,create=True), \
             patch('augment_atlas.runtime.os.getgid',return_value=1000,create=True):
            self.runtime.create('container-fixture',self.job(),self.root/'output')
        self.assertFalse(any('/torch-cache' in str(arg) or str(arg).startswith('TORCH_HOME=')
                             for arg in docker.call_args.args[0]))


class WorkerBoundaryTests(TemporaryTest):
    def run_with_mock_runtime(self, config_error=None, ownership_error=None):
        """Mock Docker and queue boundaries, preserving real worker control flow."""
        settings = {"code": str(self.root), "data": str(self.root), "runs": str(self.root / "runs"),
                    "queue_checkout": str(self.root / "queue"), "remote": "synthetic-remote",
                    "phase": "smoke", "server": "host-fixture", "gpu": "0"}
        host = Mock()
        host.commit = "b" * 40
        host.image = "sha256:fixture"
        host.gpu_uuid = "GPU-fixture"
        host.environment = {"synthetic": True}
        host.compatible.return_value = True
        if config_error:
            host.config_for.side_effect = config_error
        host.inspect.return_value = {"State": {"Status": "created", "Running": False, "ExitCode": 0}}
        queue = Mock()
        candidate = owned_record(0, "pending")
        queue.list_jobs.return_value = [candidate]
        queue.claim.return_value = owned_record()
        spool = Mock()
        spool.events = []
        spool.record = owned_record()
        if ownership_error:
            spool.flush.side_effect = ownership_error
        patches = [patch("augment_atlas.runtime.DockerRuntime", return_value=host),
                   patch("augment_atlas.runtime.GitQueue", return_value=queue),
                   patch("augment_atlas.runtime.gpu_lock", return_value=contextlib.nullcontext()),
                   patch("augment_atlas.runtime.Spool", return_value=spool),
                   patch("augment_atlas.runtime.command", side_effect=AssertionError("mock detected forbidden Docker start"))]
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            try:
                runtime.worker(settings, once=True)
            finally:
                if config_error:
                    queue.claim.assert_not_called()
                if ownership_error:
                    host.create.assert_not_called()  # pre-existing created container fixture

    def test_mock_invalid_configuration_fails_before_claim(self):
        with self.assertRaisesRegex(ValueError, "invalid config fixture"):
            self.run_with_mock_runtime(config_error=ValueError("invalid config fixture"))

    def test_mock_start_conflict_is_propagated_before_docker_start(self):
        with self.assertRaises(StaleAttempt):
            self.run_with_mock_runtime(ownership_error=StaleAttempt("synthetic owner lost"))


class ResumeCrashTests(TemporaryTest):
    def run_recovery(self, *, started_after_crash=False, still_running=False, pending_resume=False):
        """Real durable journals, mock Docker states and queue acknowledgements."""
        settings = {"code": str(self.root), "data": str(self.root), "runs": str(self.root / "runs"),
                    "queue_checkout": str(self.root / "queue"), "remote": "synthetic-remote",
                    "phase": "smoke", "server": "host-fixture", "gpu": "0", "resume": True}
        host = Mock()
        host.commit = "b" * 40
        host.image = "sha256:fixture"
        host.gpu_uuid = "GPU-fixture"
        host.environment = {"source": "mock-fixture"}
        interrupted = owned_record(3, "interrupted")
        resumed = owned_record(4, "running")
        out = self.root / "runs" / interrupted["job_id"] / interrupted["attempt_id"]
        out.mkdir(parents=True)
        (out / "checkpoint.pt").write_bytes(b"mock existing checkpoint")
        slot = self.root / "runs" / "worker-GPU-fixture"
        slot.mkdir()
        name = "atlas-attemptfixture"
        active = slot / "active.json"
        atomic_json(active, {"record": interrupted, "output": str(out), "container": name,
                            "image": host.image, "stage": "resuming", "prior_started_at": "old-start"})
        spool = runtime.Spool(out / "events-pending.json", interrupted if pending_resume else resumed)
        pending_id = None
        if pending_resume:
            spool.add("running", manual_resume=True)
            pending_id = spool.events[0]["event_id"]
        else:
            spool.save()  # Remote ack persisted, old active.json still interrupted.
        first_started = "new-start" if started_after_crash else "old-start"
        initial = {"State": {"Status": "running" if still_running else "exited", "Running": still_running,
                             "ExitCode": 75, "StartedAt": first_started}}
        finished = {"State": {"Status": "exited", "Running": False, "ExitCode": 0, "StartedAt": "new-start"}}
        host.inspect.side_effect = [initial, finished, finished]
        queue = Mock()
        acknowledgements = []
        def acknowledge(job_id, attempt_id, token, version, event):
            self.assertEqual(interrupted["job_id"], job_id)
            self.assertEqual(interrupted["attempt_id"], attempt_id)
            self.assertEqual(interrupted["token"], token)
            acknowledgements.append((version, copy.deepcopy(event)))
            return owned_record(version + 1, event["state"])
        queue.update.side_effect = acknowledge
        with patch("augment_atlas.runtime.DockerRuntime", return_value=host), \
             patch("augment_atlas.runtime.GitQueue", return_value=queue), \
             patch("augment_atlas.runtime.gpu_lock", return_value=contextlib.nullcontext()), \
             patch("augment_atlas.runtime.command", return_value="mock Docker log") as docker, \
             patch("augment_atlas.runtime.append_sample", return_value={}), \
             patch("augment_atlas.runtime.validate_artifacts", return_value={"source": "mock validation"}):
            runtime.worker(settings, once=True)
        queue.claim.assert_not_called()
        host.create.assert_not_called()
        starts = [call for call in docker.call_args_list if call.args[0][:2] == ["docker", "start"]]
        self.assertEqual(0 if started_after_crash else 1, len(starts))
        self.assertFalse(active.exists(), "completed attempt must clear active journal")
        completed = runtime.Spool(out / "events-pending.json", interrupted)
        self.assertEqual("succeeded", completed.record["state"])
        self.assertEqual(interrupted["attempt_id"], completed.record["attempt_id"])
        resumes = [(version, event) for version, event in acknowledgements if event.get("manual_resume")]
        if pending_resume:
            self.assertEqual(1, len(resumes))
            self.assertEqual(3, resumes[0][0])
            self.assertEqual(pending_id, resumes[0][1]["event_id"])
        else:
            self.assertEqual([], resumes, "acknowledged manual resume must not be emitted twice")

    def test_mock_resume_acknowledged_before_active_journal_saved(self):
        self.run_recovery()

    def test_mock_pending_resume_event_reuses_its_durable_id(self):
        self.run_recovery(pending_resume=True)

    def test_mock_child_started_before_crash_and_is_still_alive(self):
        self.run_recovery(started_after_crash=True, still_running=True)

    def test_mock_child_started_and_finished_before_coordinator_restart(self):
        self.run_recovery(started_after_crash=True)


@unittest.skipIf(os.name == "nt", "flock is a Linux runtime boundary; Windows coordinator skips it")
class LinuxLockTests(TemporaryTest):
    def test_real_second_process_cannot_lock_same_gpu(self):
        script = (
            "import sys\nfrom augment_atlas.runtime import gpu_lock\n"
            "try:\n with gpu_lock(sys.argv[1], 'GPU-fixture'): pass\n"
            "except RuntimeError: sys.exit(23)\n"
        )
        environment = os.environ.copy()
        source = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        with runtime.gpu_lock(self.root, "GPU-fixture"):
            blocked = subprocess.run([sys.executable, "-c", script, str(self.root)],
                                     capture_output=True, text=True, env=environment, timeout=15)
        self.assertEqual(23, blocked.returncode, blocked.stderr)
        released = subprocess.run([sys.executable, "-c", script, str(self.root)],
                                  capture_output=True, text=True, env=environment, timeout=15)
        self.assertEqual(0, released.returncode, released.stderr)


if __name__ == "__main__":
    unittest.main()
