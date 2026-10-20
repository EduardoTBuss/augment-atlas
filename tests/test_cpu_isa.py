"""CPU ISA cap for training containers. Docker/GPU/queue interactions are MOCKS.

The x299 host resets under AVX-512 CPU kernels, so a missing settings key must
cap the container at AVX2; "native" restores the exact pre-cap command.
"""
import contextlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from augment_atlas import runtime
from augment_atlas.storage import atomic_json, digest

AVX2_ENV = ['-e', 'ATEN_CPU_CAPABILITY=avx2', '-e', 'MKL_ENABLE_INSTRUCTIONS=AVX2',
            '-e', 'ONEDNN_MAX_CPU_ISA=AVX2', '-e', 'DNNL_MAX_CPU_ISA=AVX2',
            '-e', 'OPENBLAS_CORETYPE=Haswell',
            '-e', 'NPY_DISABLE_CPU_FEATURES=AVX512F AVX512CD AVX512_SKX',
            '-e', 'OPENCV_CPU_DISABLE=AVX512_SKX']
INVALID = ('AVX2', 'avx512', 'avx2 ', '', 'none', None, 2, ['avx2'])


class CpuIsaEnvTests(unittest.TestCase):
    def test_missing_key_defaults_to_avx2_cap(self):
        self.assertEqual(AVX2_ENV, runtime.cpu_isa_env({}))
        self.assertEqual('avx2', runtime.cpu_isa_max({}))

    def test_explicit_avx2_equals_default_and_native_adds_nothing(self):
        self.assertEqual(AVX2_ENV, runtime.cpu_isa_env({'cpu_isa_max': 'avx2'}))
        self.assertEqual([], runtime.cpu_isa_env({'cpu_isa_max': 'native'}))

    def test_returned_list_is_a_fresh_copy(self):
        first = runtime.cpu_isa_env({})
        first.append('mutated')
        self.assertEqual(AVX2_ENV, runtime.cpu_isa_env({}))

    def test_invalid_values_fail_closed(self):
        for value in INVALID:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'cpu_isa_max'):
                runtime.cpu_isa_env({'cpu_isa_max': value})


class CreateArgumentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='augment-atlas-cpu-isa-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        fixture = Path(__file__).resolve().parents[1] / 'configs' / 'smoke' / 'synthetic.json'
        self.cfg = json.loads(fixture.read_text(encoding='utf-8'))
        self.code = self.root / 'code'
        self.data = self.root / 'data'
        self.path = self.code / 'configs' / 'smoke' / 'fixture.json'
        self.data.mkdir()
        atomic_json(self.path, self.cfg)
        self.out = self.root / 'output'

    def job(self):
        from augment_atlas.cli import recipe_identity
        from augment_atlas.data import load_records, prepare_splits
        with tempfile.TemporaryDirectory(prefix='augment-atlas-split-fixture-') as directory:
            splits = prepare_splits(self.cfg, load_records(self.cfg), directory)
        return {'phase': 'smoke', 'config': 'configs/smoke/fixture.json', 'config_sha256': digest(self.path),
                'train_seed': self.cfg['training']['seed'], 'arm': self.cfg['augmentation']['arm'],
                'split_hash': splits['sha256'], 'recipe_hash': recipe_identity(self.cfg),
                'dataset': self.cfg['dataset']['id'], 'dataset_version': self.cfg['dataset']['version'],
                'protocol_version': self.cfg['protocol_version'], 'fold': self.cfg['splits']['fold']}

    def pretrained(self, settings):
        settings['torch_cache'] = str(self.root / 'torch-home')
        cached = Path(settings['torch_cache']) / 'hub' / 'checkpoints' / runtime.IMAGENET_V2_CHECKPOINT
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b'synthetic host cache bytes; not torch weights\n' * 31)
        self.cfg['model'].update(weights='IMAGENET1K_V2', weights_sha256=digest(cached))
        atomic_json(self.path, self.cfg)

    def settings(self, **extra):
        return {'code': str(self.code), 'data': str(self.data), 'phase': 'smoke', 'gpu': '0', **extra}

    def create_args(self, settings, resume=False):
        host = runtime.DockerRuntime(settings)
        host.commit = 'b' * 40
        host.image = 'sha256:fixture'
        with patch('augment_atlas.runtime.command') as docker, \
             patch('augment_atlas.runtime.os.getuid', return_value=1000, create=True), \
             patch('augment_atlas.runtime.os.getgid', return_value=1000, create=True):
            host.create('container-fixture', self.job(), self.out, resume=resume)
        docker.assert_called_once()
        return docker.call_args.args[0]

    def legacy_args(self, settings, pretrained=False, resume=False):
        """Literal docker create command of b7a47f5, before the ISA cap existed."""
        code, data = Path(settings['code']).resolve(), Path(settings['data']).resolve()
        args = ['docker', 'create', '--name', 'container-fixture',
                '--label', 'org.augment-atlas.attempt=container-fixture',
                '--network', 'none', '--gpus', 'device=0', '--user', '1000:1000', '--cpus', '2',
                '--memory', '4g', '--shm-size', '256m',
                '--security-opt', 'no-new-privileges', '--cap-drop', 'ALL',
                '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUNBUFFERED=1', '-e', 'PYTHONPATH=/code/src',
                '-e', 'ATLAS_DATA_ROOT=/data', '-e', 'HOME=/tmp', '-e', 'OMP_NUM_THREADS=2',
                '-e', 'ATLAS_CODE_COMMIT=' + 'b' * 40, '-e', 'ATLAS_IMAGE_ID=sha256:fixture',
                '-e', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
                '--mount', f'type=bind,src={code},dst=/code,readonly',
                '--mount', f'type=bind,src={data},dst=/data,readonly',
                '--mount', f'type=bind,src={self.out},dst=/output']
        if pretrained:
            cache = Path(settings['torch_cache']).expanduser().resolve()
            args += ['--mount', f'type=bind,src={cache},dst=/torch-cache,readonly', '-e', 'TORCH_HOME=/torch-cache']
        args += ['--workdir', '/code', '--entrypoint', 'python', 'sha256:fixture',
                 '-m', 'augment_atlas.train', '--config', '/code/configs/smoke/fixture.json',
                 '--output', '/output', '--resume-if-exists']
        if resume:
            args += ['--resume', '/output/checkpoint.pt']
        return args

    def assert_cap_is_only_addition(self, capped, native):
        start = capped.index(AVX2_ENV[1]) - 1
        self.assertEqual(AVX2_ENV, capped[start:start + len(AVX2_ENV)])
        self.assertEqual(native, capped[:start] + capped[start + len(AVX2_ENV):])
        # Container options must precede the image positional argument.
        self.assertLess(start + len(AVX2_ENV), capped.index('sha256:fixture'))

    def test_default_without_key_adds_avx2_env_to_docker_create(self):
        settings = self.settings()
        args = self.create_args(settings)
        for pair in AVX2_ENV[1::2]:
            self.assertEqual('-e', args[args.index(pair) - 1])
        self.assertEqual(['docker', 'create'], args[:2])

    def test_native_is_byte_identical_to_legacy_command(self):
        for pretrained in (False, True):
            for resume in (False, True):
                with self.subTest(pretrained=pretrained, resume=resume):
                    settings = self.settings(cpu_isa_max='native')
                    if pretrained:
                        self.pretrained(settings)
                    args = self.create_args(settings, resume=resume)
                    self.assertEqual(self.legacy_args(settings, pretrained, resume), args)
                    self.assertFalse(any(str(arg).startswith(('ATEN_CPU_CAPABILITY=', 'MKL_ENABLE_INSTRUCTIONS=',
                                                              'ONEDNN_MAX_CPU_ISA=', 'DNNL_MAX_CPU_ISA=',
                                                              'OPENBLAS_CORETYPE=', 'NPY_DISABLE_CPU_FEATURES=',
                                                              'OPENCV_CPU_DISABLE=')) for arg in args))

    def test_default_and_avx2_only_append_cap_to_legacy_command(self):
        for pretrained in (False, True):
            for resume in (False, True):
                for extra in ({}, {'cpu_isa_max': 'avx2'}):
                    with self.subTest(pretrained=pretrained, resume=resume, extra=extra):
                        settings = self.settings(**extra)
                        if pretrained:
                            self.pretrained(settings)
                        capped = self.create_args(settings, resume=resume)
                        native = self.create_args({**settings, 'cpu_isa_max': 'native'}, resume=resume)
                        self.assert_cap_is_only_addition(capped, native)
                        self.assertEqual(self.legacy_args(settings, pretrained, resume), native)
                        self.assertEqual(len(native) + len(AVX2_ENV), len(capped))

    def test_invalid_value_fails_create_before_validation_or_docker(self):
        for value in INVALID:
            with self.subTest(value=value):
                host = runtime.DockerRuntime(self.settings(cpu_isa_max=value))
                with patch('augment_atlas.runtime.command') as docker, \
                     patch.object(host, 'config_for') as config, \
                     self.assertRaisesRegex(ValueError, 'cpu_isa_max'):
                    host.create('container-fixture', self.job(), self.out)
                docker.assert_not_called()
                config.assert_not_called()

    def test_invalid_value_fails_preflight_before_external_commands(self):
        for value in INVALID:
            with self.subTest(value=value):
                host = runtime.DockerRuntime(self.settings(cpu_isa_max=value))
                with patch('augment_atlas.runtime.command') as external, \
                     self.assertRaisesRegex(ValueError, 'cpu_isa_max'):
                    host.preflight()
                external.assert_not_called()

    def test_valid_values_pass_preflight_isa_check(self):
        for extra in ({}, {'cpu_isa_max': 'avx2'}, {'cpu_isa_max': 'native'}):
            with self.subTest(extra=extra):
                host = runtime.DockerRuntime(self.settings(image='image-fixture', **extra))
                with patch('augment_atlas.runtime.command', side_effect=['', 'b' * 40, 'sha256:fixture',
                                                                         'GPU-fixture, 12000, 40']):
                    self.assertEqual({'gpu_uuid': 'GPU-fixture', 'image_id': 'sha256:fixture'}, host.preflight())


class _StopAtStart(Exception):
    pass


class RuntimeProvenanceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='augment-atlas-cpu-isa-worker-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def runtime_json(self, **extra):
        """Real worker control flow up to the Docker start boundary, all mocked."""
        base = Path(tempfile.mkdtemp(dir=self.root))  # fresh runs/ per call
        settings = {'code': str(base), 'data': str(base), 'runs': str(base / 'runs'),
                    'queue_checkout': str(base / 'queue'), 'remote': 'synthetic-remote',
                    'phase': 'smoke', 'server': 'host-fixture', 'gpu': '0', **extra}
        record = {'job_id': 'a' * 64, 'attempt_id': 'attempt-fixture', 'token': 'public-owner-fixture',
                  'state': 'claimed', 'version': 1, 'spec': {'code_commit': 'b' * 40, 'config_sha256': 'c' * 64}}
        host = Mock()
        host.commit = 'b' * 40
        host.image = 'sha256:fixture'
        host.gpu_uuid = 'GPU-fixture'
        host.environment = {'synthetic': True}
        host.compatible.return_value = True
        host.inspect.side_effect = [None, {'State': {'Status': 'created', 'Running': False, 'ExitCode': 0}}]
        queue = Mock()
        queue.list_jobs.return_value = [{**record, 'state': 'pending', 'version': 0}]
        queue.claim.return_value = record
        spool = Mock()
        spool.events = []
        spool.record = record

        def docker(args, timeout=60):
            if args[:2] == ['docker', 'start']:
                raise _StopAtStart()
            raise AssertionError('unexpected external command')

        with patch('augment_atlas.runtime.DockerRuntime', return_value=host), \
             patch('augment_atlas.runtime.GitQueue', return_value=queue), \
             patch('augment_atlas.runtime.gpu_lock', return_value=contextlib.nullcontext()), \
             patch('augment_atlas.runtime.Spool', return_value=spool), \
             patch('augment_atlas.runtime.command', side_effect=docker), \
             self.assertRaises(_StopAtStart):
            runtime.worker(settings, once=True)
        host.create.assert_called_once()
        out = base / 'runs' / record['job_id'] / record['attempt_id']
        return json.loads((out / 'runtime.json').read_text(encoding='utf-8'))

    def test_runtime_json_records_effective_cap_additively(self):
        for extra, wanted in (({}, 'avx2'), ({'cpu_isa_max': 'avx2'}, 'avx2'),
                              ({'cpu_isa_max': 'native'}, 'native')):
            with self.subTest(extra=extra):
                provenance = self.runtime_json(**extra)
                self.assertEqual(wanted, provenance['cpu_isa_max'])
                self.assertEqual({'job_id', 'attempt_id', 'code_commit', 'image_id', 'owner', 'environment',
                                  'cpu_isa_max'}, set(provenance))


if __name__ == '__main__':
    unittest.main()


class ProbeIsaTests(unittest.TestCase):
    """The CUDA probe also initialises torch on the CPU, so it carries the same ISA cap."""

    def _probe_args(self, settings):
        from unittest import mock
        rt = runtime.DockerRuntime.__new__(runtime.DockerRuntime)
        rt.s, rt.image = settings, "sha256:" + "0" * 64
        seen = []
        with mock.patch.object(runtime, "command", side_effect=lambda args, **kw: seen.append(args) or "{}"):
            rt.probe()
        return seen[0]

    def test_probe_default_and_native(self):
        base = {"gpu": 0}
        default = self._probe_args(dict(base))
        native = self._probe_args(dict(base, cpu_isa_max="native"))
        isa = runtime.cpu_isa_env({})
        i = default.index("--entrypoint")
        self.assertEqual(isa, default[i - len(isa):i])
        self.assertEqual(native, default[:i - len(isa)] + default[i:])
