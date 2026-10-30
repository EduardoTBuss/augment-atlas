"""Real C9b/X5 adapter against disposable local Git queues; no Docker/network."""
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import global_campaign as gc, global_collect as col, global_jobs as gj
from augment_atlas import global_report as gr, runtime, seed_requests as sr
from augment_atlas.global_hpo import environment_sha256
from augment_atlas.queue import GitQueue
from augment_atlas.seed_stage_authorizer import RealStageAuthorizer
import test_stage_authorization as stagefixture
import test_seed_requests as seeds
import test_global_cli as fixtures


class RealAuthorizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stagefixture.StageAuthorizationTests.setUpClass()
        cls.addClassCleanup(stagefixture.StageAuthorizationTests.tearDownClass)
        cls.env, cls.campaign, cls.freeze, cls.pools = (getattr(stagefixture.StageAuthorizationTests, key) for key in
                                                     ("env", "campaign", "freeze", "pools"))

    def setUp(self):
        self.fixture = stagefixture.StageAuthorizationTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = Path(self.fixture.tmp.name)
        self.control, self.cards, self.state = (self.base / name for name in ('control', 'nodes', 'state'))
        (self.control / 'control').mkdir(parents=True)
        self.cards.mkdir()
        self.card = seeds.card()
        self.card['datasets'] = [{**self.campaign['dataset'], 'manifest_sha256': self.env.source_doc['manifest_sha256']}]
        self.probe = json.loads(self.env.probe.read_bytes())
        self.card['images'] = [{'image_id': fixtures.IMAGE, 'probe': self.probe}]
        self.request = seeds.request(request_id=self._testMethodName, dataset=fixtures.DATASET,
            campaign_sha256=self.campaign['sha256'], seeds=[13, 19], requested_on='2026-10-29')
        self.adapter = self.make_adapter()
        self.write_inputs()

    def make_adapter(self):
        return RealStageAuthorizer(code_repo=self.env.code, data_root=self.env.data, source=self.env.source,
            remote=self.fixture.queue.remote, min_vram_mib=1000)

    def write_inputs(self):
        (self.control / sr.REQUEST_FILE).write_text(seeds.markdown([self.request]), encoding='utf-8')
        (self.cards / 'node-a.json').write_text(json.dumps(self.card), encoding='utf-8')

    def process(self, apply=True, adapter=None):
        self.write_inputs()
        return sr.process_requests(self.control, self.fixture.queue.checkout, self.cards, self.state,
            apply=apply, authorizer=adapter or self.adapter, now=seeds.NOW)[0]

    def document(self):
        return self.adapter._document(self.request)

    def test_real_authorize_anchor_seed_claim_and_idempotent_restart(self):
        row = self.process()
        self.assertEqual(row['status'], 'authorized', row)
        self.assertEqual(row['progress']['total'], 100)
        doc = self.document()
        jobs = self.fixture.queue.list_jobs()
        for job in jobs:
            spec = job['spec']
            self.assertEqual(spec['code_commit'], stagefixture.StageAuthorizationTests.execution)
            self.assertEqual(spec['requirements']['allowed_nodes'], ['node-a'])
            self.assertEqual(spec['environment_sha256'], self.campaign['runtime']['environment_sha256'])
            self.assertEqual(spec['seed_request']['stage_sha256'], doc['sha256'])
        chronology = gc.audit_chronology(self.fixture.queue, self.campaign, freeze=self.freeze,
            stage_authorization=doc, code_repo=self.env.code)
        self.assertEqual(chronology['final_jobs'], 100)
        self.assertEqual(jobs[0]['spec']['seed_request']['anchor_commit'], chronology['C_stage_authorization'])
        forged = copy.deepcopy(jobs[0]['spec'])
        forged['seed_request']['anchor_commit'] = '0'*40
        with self.assertRaisesRegex(ValueError, 'campaign_invalid'):
            self.fixture.queue.seed_final([forged], self.campaign['sha256'], stage_authorization=doc,
                                          campaign=self.campaign, code_repo=self.env.code)
        self.assertIsNone(self.fixture.queue.claim(jobs[0]['job_id'], {'server': 'node-b'}))
        claimed = self.fixture.queue.claim(jobs[0]['job_id'], {'server': 'node-a'},
            stage_context=(self.campaign, doc, self.env.code))
        self.assertEqual(claimed['state'], 'claimed')
        self.assertEqual(self.process(adapter=self.make_adapter())['status'], 'running')
        self.assertEqual(len(self.fixture.queue.list_jobs()), 100)

    def test_probe_mismatch_requires_explicit_boolean_and_keeps_image_pinned(self):
        self.card['images'][0]['probe'] = {'torch': 'other', 'cuda': None}
        for flag in (None, False):
            if flag is not None:
                self.request['cross_environment'] = flag
            row = self.process(apply=False)
            self.assertEqual(row['status'], 'invalid')
            self.assertIn('cross_environment', row['error'])
            self.assertEqual(self.process()['status'], 'invalid')
            self.assertFalse(self.state.joinpath(sr.STATE_FILE).exists())
        self.request['cross_environment'] = True
        self.card['images'][0]['image_id'] = 'sha256:' + '0'*64
        row = self.process()
        self.assertEqual(row['status'], 'invalid')
        self.assertIn('image', row['error'])
        for value in ('true', 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                sr.validate_request({**self.request, 'cross_environment': value})

    def test_cross_environment_runtime_collection_and_report_preserve_primary(self):
        primary_before = gr.build_report(self.campaign, [], self.pools, self.env.records,
            expected_campaign_sha256=self.campaign['sha256'], **stagefixture.StageAuthorizationTests.common)
        self.request['cross_environment'] = True
        self.probe = {'torch': 'declared-other', 'cuda': None}
        self.card['images'][0]['probe'] = self.probe
        row = self.process()
        self.assertEqual(row['status'], 'authorized', row)
        doc = self.document()
        self.assertTrue(doc['cross_environment'])
        self.assertEqual(doc['runtime']['environment_sha256'], environment_sha256(fixtures.IMAGE, self.probe))
        self.assertNotEqual(doc['runtime']['environment_sha256'], self.campaign['runtime']['environment_sha256'])
        job = self.fixture.queue.list_jobs()[0]
        spec = job['spec']
        host = runtime.DockerRuntime(dict(code=str(self.env.code), data=str(self.env.data), phase='final', gpu='0',
            campaign_sha256=self.campaign['sha256'], campaign_commit=self.fixture.c_campaign,
            stage_authorization_sha256=doc['sha256'], remote=str(self.fixture.queue.remote),
            queue_checkout=str(self.base / 'runtime-queue'), server='node-a'))
        host.commit, host.image, host.environment = stagefixture.StageAuthorizationTests.execution, fixtures.IMAGE, self.probe
        with patch.object(host, 'weights_cache_for', return_value=None):
            _, _, trusted, _ = host.global_config_for(spec)
        host.check_environment(spec)
        host.environment = json.loads(self.env.probe.read_bytes())
        with self.assertRaisesRegex(ValueError, 'code_identity_mismatch'):
            host.campaign_gate(spec, trusted)
        host.environment = self.probe
        host.check_environment(spec)
        forged = copy.deepcopy(spec)
        forged['requirements']['allowed_nodes'] = ['node-b']
        with self.assertRaisesRegex(ValueError, 'campaign_invalid'):
            from augment_atlas.stage_authorization import job_binding
            job_binding(forged, self.campaign, doc)
        raw_claim = self.fixture.queue.claim
        with patch.object(self.fixture.queue, 'claim', side_effect=lambda job_id, owner:
                raw_claim(job_id, {'server': 'node-a'}, stage_context=(self.campaign, doc, self.env.code))):
            stagefixture.StageAuthorizationTests.tc.finish(self.fixture.queue, job, 'succeeded',
                                                         initial=fixtures.H(['initial', spec['fold'], spec['train_seed']]))
        kwargs = dict(**stagefixture.StageAuthorizationTests.common, stage_authorization=doc, code_repo=self.env.code)
        snapshot = self.fixture.queue.list_jobs()
        with self.assertRaisesRegex(ValueError, 'code_identity_mismatch'):
            col.collect_final_entries(self.campaign, snapshot, self.env.runs, records=self.env.records,
                                      pools=self.pools, scope='all_authorized', **kwargs)
        completed_job = next(j for j in snapshot if j['job_id'] == job['job_id'])
        provenance_path = self.env.runs / job['job_id'] / completed_job['attempt_id'] / 'runtime.json'
        provenance = json.loads(provenance_path.read_bytes())
        provenance['environment'] = self.probe
        provenance_path.write_text(json.dumps(provenance), encoding='utf-8')
        entries, manifest = col.collect_final_entries(self.campaign, self.fixture.queue.list_jobs(), self.env.runs,
            records=self.env.records, pools=self.pools, scope='all_authorized', **kwargs)
        self.assertEqual((len(entries), manifest['authorized_cells']), (1, 150))
        report = gr.build_report(self.campaign, entries, self.pools, self.env.records,
            expected_campaign_sha256=self.campaign['sha256'], scope='all_authorized', **kwargs)
        self.assertEqual(report['stage_authorization']['environment_label'], 'different environment (declared)')
        self.assertTrue(all(r.get('environment') == 'different environment (declared)'
                            for r in report['per_seed'] if r['seed'] in [13, 19]))
        self.assertNotIn(11, doc['seeds'])
        self.assertEqual(gr.public_summary(report)['stage_authorization']['environment_label'],
                         'different environment (declared)')
        primary_after = gr.build_report(self.campaign, [], self.pools, self.env.records,
            expected_campaign_sha256=self.campaign['sha256'], **kwargs)
        self.assertEqual(gj.render_config_bytes(primary_before), gj.render_config_bytes(primary_after))

    def test_lost_anchor_response_reuses_durable_authorization(self):
        publish = GitQueue.publish_anchor
        def lost(queue, anchor):
            publish(queue, anchor)
            raise RuntimeError('synthetic lost anchor response')
        with patch.object(GitQueue, 'publish_anchor', lost):
            self.assertEqual(self.process()['status'], 'invalid')
        doc = self.document()
        self.assertEqual(self.fixture.queue.list_jobs(), [])
        self.assertEqual(self.process(adapter=self.make_adapter())['status'], 'authorized')
        self.assertEqual(self.document(), doc)
        self.assertEqual(len(self.fixture.queue.list_jobs()), 100)

    def test_lost_seed_response_resumes_without_reauthorizing(self):
        publish = self.adapter.seed_final
        def lost(**kwargs):
            publish(**kwargs)
            raise RuntimeError('synthetic lost seed response')
        with patch.object(self.adapter, 'seed_final', side_effect=lost):
            self.assertEqual(self.process()['status'], 'invalid')
        self.assertEqual(sr.load_state(self.state)['requests'][self.request['request_id']]['status'], 'authorized')
        fresh = self.make_adapter()
        with patch.object(fresh, 'authorize', side_effect=AssertionError('must not reauthorize')):
            self.assertEqual(self.process(adapter=fresh)['status'], 'authorized')
        self.assertEqual(len(self.fixture.queue.list_jobs()), 100)

    def test_cli_cross_environment_seal_and_anchor_use_supplied_probe(self):
        probe = self.base / 'other-probe.json'
        probe.write_text(json.dumps({'torch': 'declared-other', 'cuda': None}), encoding='utf-8')
        run_cli = stagefixture.StageAuthorizationTests.tc.run_cli
        authorized = run_cli(self.env.base('authorize-stage') + ['--code', str(self.env.code),
            '--campaign-sha256', self.campaign['sha256'], '--stage', '2', '--image-id', fixtures.IMAGE,
            '--environment-probe', str(probe), '--cross-environment', '--decided-on', '2026-10-29',
            '--decided-after-primary-results', 'false', '--rationale', 'time and budget only',
            '--sealed-at', '2026-10-29T18:00:00Z'])
        doc = json.loads(Path(authorized['path']).read_bytes())
        self.assertEqual(doc['runtime']['environment_sha256'],
                         environment_sha256(fixtures.IMAGE, json.loads(probe.read_bytes())))
        anchored = run_cli(self.env.base('anchor') + ['--code', str(self.env.code),
            '--remote', str(self.fixture.queue.remote), '--checkout', str(self.fixture.queue.checkout),
            '--kind', 'stage_authorization', '--document-sha256', doc['sha256'],
            '--image-id', fixtures.IMAGE, '--environment-probe', str(probe)])
        self.assertEqual(anchored['document_sha256'], doc['sha256'])

    def test_bundled_cli_config_plan_and_undeclared_seed_rejection(self):
        config = self.base / 'authorizer.json'
        config.write_text(json.dumps(dict(code_repo=str(self.env.code), data_root=str(self.env.data),
            source=str(self.env.source), remote=str(self.fixture.queue.remote), min_vram_mib=1000)), encoding='utf-8')
        args = ['plan', '--control-checkout', str(self.control), '--queue-checkout', str(self.fixture.queue.checkout),
                '--node-cards', str(self.cards), '--state-dir', str(self.state), '--authorizer-config', str(config),
                '--now', '2026-10-30T12:00:00Z']
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(sr.main(args), 0)
        self.assertEqual(json.loads(out.getvalue())[0]['status'], 'pending')
        self.assertFalse(self.state.exists())
        applied = subprocess.run([sys.executable, '-m', 'augment_atlas.seed_requests', 'apply', *args[1:-2]],
            cwd=fixtures.ROOT, env={**os.environ, 'PYTHONPATH': str(fixtures.ROOT / 'src')},
            capture_output=True, text=True)
        self.assertEqual(applied.returncode, 0, applied.stdout + applied.stderr)
        self.assertEqual(json.loads(applied.stdout)[0]['progress']['total'], 100)
        self.assertIn('withheld', self.adapter.public_error(ValueError('invalid: ' + self.env.records[0]['id'])))
        self.request.update(request_id=self.request['request_id'] + '-undeclared', seeds=[23])
        self.assertIn('declared later', self.process()['error'])


if __name__ == '__main__':
    unittest.main()
