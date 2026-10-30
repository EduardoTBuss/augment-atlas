"""C9b: sealed sensitivity authorization with real, disposable Git repositories (no training)."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from augment_atlas import global_campaign as gc, global_cli as cli, global_collect as col
from augment_atlas import global_jobs as gj, global_report as gr, runtime, stage_authorization as sa
from augment_atlas.queue import GitQueue, QueueConflict, anchor_path, render_anchor
from augment_atlas.schema import new_record
import test_global_cli as fixtures
import test_global_cli_hpo as hpo_fixtures
from test_global_campaign import resign


class StageAuthorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tc = fixtures.CliCase('runTest')
        cls.tc.setUp()
        cls.env = env = cls.tc.env
        cls.freeze_out, cls.campaign_out = cls.tc.to_campaign()
        fixtures.git(env.code, 'config', 'core.autocrlf', 'false')
        cls.campaign = json.loads(Path(cls.campaign_out['path']).read_bytes())
        cls.freeze = json.loads((env.data / 'global' / fixtures.DATASET /
                                 ('external-freeze-' + cls.freeze_out['freeze_sha256'] + '.json')).read_bytes())
        cls.pools = json.loads((env.data / 'global' / fixtures.DATASET / ('pools-' + env.pools_sha + '.json')).read_bytes())
        cls.original = fixtures.git(env.code, 'rev-parse', 'HEAD')
        cls.primary_bytes = {p.relative_to(env.code).as_posix(): p.read_bytes()
                             for p in (env.code / cls.campaign['configs']['root']).glob('*.json')}
        # Config preparation grants no execution; it is additive and idempotent.
        cls.stage_manifest = sa.materialize(cls.campaign, 2, env.code)
        fixtures.git(env.code, 'add', '-A')
        fixtures.git(env.code, 'commit', '-qm', 'synthetic stage configs')
        cls.execution = fixtures.git(env.code, 'rev-parse', 'HEAD')
        cls.auth_args = dict(stage=2, execution_code_commit=cls.execution, code_repo=env.code,
            image_id=fixtures.IMAGE, environment_sha256=cls.campaign['runtime']['environment_sha256'],
            decided_on='2026-10-29', decided_after_primary_results=True,
            rationale='Eduardo decided after viewing the INbreast primary result; declared seeds are sensitivity only.',
            sealed_at_utc='2026-10-29T18:00:00Z')
        cls.auth = sa.authorize_stage(cls.campaign, **cls.auth_args)
        cls.common = dict(freeze=cls.freeze, dataset_source=env.source_doc, augmentation_presets=cli._presets(env.code))
        cls.stage_jobs = col.expected_final_jobs(cls.campaign, env.records, cls.pools, **cls.common,
            stage_authorization=cls.auth, code_repo=env.code, scope='stage')
        cls.primary_jobs = col.expected_final_jobs(cls.campaign, env.records, cls.pools, **cls.common)
        cls.auth_path = env.data / 'global' / fixtures.DATASET / ('stage-authorization-' + cls.auth['sha256'] + '.json')
        cls.auth_path.write_bytes(gj.render_config_bytes(cls.auth))

    @classmethod
    def tearDownClass(cls):
        cls.tc.doCleanups()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atlas-stage-queue-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        remote = root / 'queue.git'
        subprocess.run(['git', 'init', '--bare', '-q', str(remote)], check=True, capture_output=True)
        self.queue = GitQueue(root / 'queue', remote)
        self.queue.publish_anchor(gc.build_anchor('recipe_freeze', self.freeze))
        self.c_campaign = self.queue.publish_anchor(gc.build_anchor('campaign', self.campaign))['commit']
        self.seed_kw = dict(stage_authorization=self.auth, campaign=self.campaign, code_repo=self.env.code)
        self.audit_kw = dict(stage_authorization=self.auth, code_repo=self.env.code)

    def anchor(self):
        return self.queue.publish_anchor(gc.build_anchor('stage_authorization', self.auth))

    def stage_spec(self):
        return next(v[5] for v in self.stage_jobs.values() if v[:3] == ('none', 0, 13))

    def audit(self):
        return gc.audit_chronology(self.queue, self.campaign, freeze=self.freeze,
                                   recorded_campaign_commit=self.c_campaign, **self.audit_kw)

    def test_no_authorization_keeps_original_refusals_and_matrix(self):
        self.assertEqual(len(self.primary_jobs), 50)
        self.assertEqual(self.campaign['matrix']['authorized_seeds'], [11])
        with self.assertRaisesRegex(QueueConflict, '^code_identity_mismatch:'):
            self.queue.seed_final([self.stage_spec()], self.campaign['sha256'])
        with self.assertRaisesRegex(ValueError, '^campaign_anchor_missing:'):
            self.queue.seed_final([self.stage_spec()], self.campaign['sha256'], **self.seed_kw)
        head = self.queue._fetch_head()
        from augment_atlas.schema import prepare_job
        early_spec = dict(self.stage_spec(), code_commit=self.original)
        early_spec.pop('job_id')
        early = new_record(prepare_job(early_spec))
        self.assertTrue(self.queue._push(self.queue._commit(head, [early], 'synthetic unauthorized job')))
        with self.assertRaisesRegex(ValueError, '^campaign_invalid:'):
            gc.audit_chronology(self.queue, self.campaign)

    def test_valid_authorization_seed_claim_and_audits(self):
        anchored = self.anchor()
        specs = [v[5] for v in self.stage_jobs.values()]
        jobs = self.queue.seed_final(specs, self.campaign['sha256'], **self.seed_kw)
        self.assertEqual(len(jobs), 100)
        self.assertEqual({j['spec']['train_seed'] for j in jobs}, {13, 19})
        self.assertEqual({j['spec']['code_commit'] for j in jobs}, {self.execution})
        self.assertEqual(len(self.queue.seed_final(specs, self.campaign['sha256'], **self.seed_kw)), 100)
        self.assertEqual(self.audit()['final_jobs'], 100)
        self.assertEqual(self.audit()['C_stage_authorization'], anchored['commit'])
        job = next(j for j in jobs if j['spec']['job_id'] == self.stage_spec()['job_id'])
        with self.assertRaisesRegex(QueueConflict, '^campaign_invalid:'):
            self.queue.claim(job['job_id'], {'server':'synthetic'})
        claimed = self.queue.claim(job['job_id'], {'server':'synthetic'},
            stage_context=(self.campaign, self.auth, self.env.code))
        self.assertEqual(claimed['state'], 'claimed')
        gc.audit_campaign(self.env.records, self.pools, self.campaign,
            expected_campaign_sha256=self.campaign['sha256'], **self.common, **self.audit_kw)
        sa.materialize(self.campaign, 2, self.env.code, authorization=self.auth)
        for path, data in self.primary_bytes.items():
            self.assertEqual((self.env.code / path).read_bytes(), data)
        self.assertEqual(sa.manifest(self.campaign,2,sa.configs(self.campaign,2,self.env.code)), self.stage_manifest)

    def test_closed_schema_seed_date_runtime_and_decision_checks(self):
        mutations = [lambda a:a.update(seeds=[19,13]), lambda a:a.update(seeds=[13]),
                     lambda a:a.pop('decided_after_primary_results'),
                     lambda a:a.update(decided_after_primary_results='true'),
                     lambda a:a.update(decided_on='2026-10-22'),
                     lambda a:a.update(sealed_at_utc='2026-10-22T00:00:00Z'),
                     lambda a:a.update(rationale=' '), lambda a:a.update(extra=True),
                     lambda a:a['runtime'].update(image_id='sha256:'+'0'*64),
                     lambda a:a['runtime'].update(environment_sha256='0'*64),
                     lambda a:a['freeze'].update(sha256='0'*64),
                     lambda a:a['stage1_configs'].update(manifest_sha256='0'*64)]
        for mutate in mutations:
            bad = copy.deepcopy(self.auth)
            mutate(bad)
            with self.subTest(bad=bad.keys()), self.assertRaises(ValueError):
                sa.audit(resign(bad), self.campaign, code_repo=self.env.code)
        self.assertTrue(self.auth['decided_after_primary_results'])
        self.assertEqual(self.auth['decided_on'], '2026-10-29')
        self.assertNotIn('decided_at', self.auth)
        foreign = copy.deepcopy(self.stage_spec())
        foreign['global']['campaign_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, '^campaign_anchor_missing:'):
            sa.job_binding(foreign, self.campaign, self.auth)
        self.assertIn('(11)', self.auth['seed_interpretation'])
        with self.assertRaisesRegex(ValueError, '^code_identity_mismatch:'):
            sa.authorize_stage(self.campaign, **dict(self.auth_args, image_id='sha256:'+'0'*64))

    def test_git_rejects_code_docs_primary_config_and_unrelated_additions(self):
        # All mutations live only in a disposable fixture repository.
        cases = [('src/code.py', b'changed trainer\n'), ('docs/added.md', b'added doc\n'),
                 (self.campaign['configs']['root']+'/none-f0-s11.json', b'changed primary config\n'),
                 ('configs/global/other/final/extra.json', b'{}'),
                 (sa.directory(self.campaign,2)+'/extra.json', b'{}')]
        for path, data in cases:
            target = self.env.code / path
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                fixtures.git(self.env.code,'add','-A')
                fixtures.git(self.env.code,'commit','-qm','synthetic invalid mutation')
                bad_commit = fixtures.git(self.env.code,'rev-parse','HEAD')
                with self.subTest(path=path), self.assertRaisesRegex(ValueError,'^code_identity_mismatch:'):
                    sa.authorize_stage(self.campaign, **dict(self.auth_args, execution_code_commit=bad_commit))
                bad = resign(dict(self.auth, execution_code_commit=bad_commit))
                with self.assertRaises(ValueError):
                    self.queue.seed_final([self.stage_spec()], self.campaign['sha256'],
                        **dict(self.seed_kw, stage_authorization=bad))
            finally:
                fixtures.git(self.env.code,'reset','--hard',self.execution)

    def test_job_before_or_in_authorization_anchor_is_campaign_invalid(self):
        for same_commit in (False, True):
            if same_commit:
                # Fresh remote for the same-commit boundary.
                self.setUp()
            head = self.queue._fetch_head()
            with_job = self.queue._commit(head, [new_record(self.stage_spec())], 'synthetic early stage job')
            if same_commit:
                path = anchor_path(fixtures.DATASET,'stage_authorization',self.auth['sha256'])
                both = self.queue._commit_files(with_job,{path:render_anchor(gc.build_anchor('stage_authorization',self.auth))},'anchor')
                commit = self.queue._git('commit-tree', both+'^{tree}', '-p',head,input='synthetic same commit\n').stdout.strip()
                self.assertTrue(self.queue._push(commit))
            else:
                self.assertTrue(self.queue._push(with_job))
                self.anchor()
            with self.subTest(same_commit=same_commit), self.assertRaisesRegex(ValueError,'^campaign_invalid:'):
                self.audit()
            with self.assertRaisesRegex(ValueError,'^campaign_invalid:'):
                self.queue.claim(self.stage_spec()['job_id'],{'server':'synthetic'},
                    stage_context=(self.campaign,self.auth,self.env.code))

    def test_runtime_enforces_pin_commit_image_probe_and_chronology(self):
        self.anchor()
        spec = self.stage_spec()
        self.queue.seed_final([spec],self.campaign['sha256'],**self.seed_kw)
        settings = dict(code=str(self.env.code),data=str(self.env.data),phase='final',gpu='0',
            campaign_sha256=self.campaign['sha256'],campaign_commit=self.c_campaign,
            stage_authorization_sha256=self.auth['sha256'],remote=str(self.queue.remote),
            queue_checkout=str(Path(self.tmp.name)/'worker'))
        def host():
            h = runtime.DockerRuntime(settings)
            h.commit,h.image,h.environment = self.execution,fixtures.IMAGE,json.loads(self.env.probe.read_bytes())
            return h
        h = host()
        with patch.object(h,'weights_cache_for',return_value=None):
            name, cfg, trusted, mount = h.global_config_for(spec)
        self.assertEqual(cfg['training']['seed'],13)
        self.assertIn('/stage-2/',name)
        self.assertEqual(mount['kind'],'data_root')
        for attr, value in [('commit',self.original),('image','sha256:'+'0'*64),('environment',{'different':'probe'})]:
            bad = host()
            setattr(bad,attr,value)
            with self.subTest(attr=attr), self.assertRaisesRegex(ValueError,'^code_identity_mismatch:'):
                bad.campaign_gate(spec,trusted)
        primary = next(v[5] for v in self.primary_jobs.values())
        bad = host()
        with self.assertRaisesRegex(ValueError,'^code_identity_mismatch:'):
            bad.campaign_gate(primary,trusted)
        bad = host()
        del bad.s['stage_authorization_sha256']
        with self.assertRaises(ValueError):
            bad.campaign_gate(spec,trusted)


    def test_worker_restart_rechecks_stage_before_inspecting_container(self):
        from unittest.mock import Mock
        runs = Path(self.tmp.name)/'runs'
        slot = runs/'worker-synthetic-gpu'
        slot.mkdir(parents=True)
        spec = self.stage_spec()
        (slot/'active.json').write_text(json.dumps({
            'record': {'spec':spec}, 'output':str(runs/'attempt'), 'container':'synthetic-container',
            'image':fixtures.IMAGE, 'stage':'resuming'}), encoding='utf-8')
        host = Mock()
        host.commit, host.image, host.gpu_uuid = self.execution, fixtures.IMAGE, 'synthetic-gpu'
        host.config_for.side_effect = ValueError('campaign_invalid: stage authorization missing after restart')
        settings = dict(runs=str(runs), queue_checkout=str(Path(self.tmp.name)/'worker'),
                        remote=str(self.queue.remote), server='synthetic', gpu='0')
        with patch('augment_atlas.runtime.DockerRuntime',return_value=host), \
             patch('augment_atlas.runtime.gpu_lock',return_value=contextlib.nullcontext()), \
             patch('augment_atlas.runtime.signal.signal'):
            with self.assertRaisesRegex(ValueError,'^campaign_invalid:'):
                runtime.worker(settings,once=True)
        host.config_for.assert_called_once_with(spec)
        host.inspect.assert_not_called()

    def test_stage1_report_and_collection_bit_identical_with_stage2_jobs(self):
        jobs = self.queue.seed_final([v[5] for v in self.primary_jobs.values()],self.campaign['sha256'])
        # A failed primary entry proves the comparison includes real cells, not just an empty report.
        claimed = self.queue.claim(jobs[0]['job_id'],{'server':'synthetic'})
        running = self.queue.update(claimed['job_id'],claimed['attempt_id'],claimed['token'],claimed['version'],
                                    {'event_id':'run','state':'running'})
        self.queue.update(running['job_id'],running['attempt_id'],running['token'],running['version'],
                          {'event_id':'failed','state':'failed'})
        before_chronology = gc.audit_chronology(self.queue,self.campaign,freeze=self.freeze)
        entries, collection = col.collect_final_entries(self.campaign,self.queue.list_jobs(),self.env.runs,
            records=self.env.records,pools=self.pools,**self.common)
        before = gr.build_report(self.campaign,entries,self.pools,self.env.records,
            expected_campaign_sha256=self.campaign['sha256'],chronology=before_chronology,**self.common)
        self.anchor()
        self.queue.seed_final([v[5] for v in self.stage_jobs.values()],self.campaign['sha256'],**self.seed_kw)
        chronology = self.audit()
        after_entries, after_collection = col.collect_final_entries(self.campaign,self.queue.list_jobs(),self.env.runs,
            records=self.env.records,pools=self.pools,**self.common,**self.audit_kw)
        after = gr.build_report(self.campaign,after_entries,self.pools,self.env.records,
            expected_campaign_sha256=self.campaign['sha256'],chronology=chronology,**self.common,**self.audit_kw)
        self.assertEqual(gj.render_config_bytes(collection),gj.render_config_bytes(after_collection))
        self.assertEqual(gj.render_config_bytes(before),gj.render_config_bytes(after))
        all_entries, all_collection = col.collect_final_entries(self.campaign,self.queue.list_jobs(),self.env.runs,
            records=self.env.records,pools=self.pools,scope='all_authorized',**self.common,**self.audit_kw)
        sensitivity = gr.build_report(self.campaign,all_entries,self.pools,self.env.records,
            expected_campaign_sha256=self.campaign['sha256'],chronology=chronology,
            scope='all_authorized',**self.common,**self.audit_kw)
        self.assertEqual(all_collection['authorized_cells'],150)
        self.assertEqual(sensitivity['seeds']['authorized'],[11,13,19])
        self.assertEqual(sensitivity['scope'],'all_authorized')
        self.assertEqual(sensitivity['stage_authorization']['sha256'],self.auth['sha256'])
        self.assertNotEqual(before['sha256'],sensitivity['sha256'])


class StageCliTests(fixtures.CliCase):
    def test_operator_sequence_and_primary_report_bytes(self):
        env = self.env
        freeze, campaign = self.to_campaign()
        self.run_cli(env.base('anchor')+['--code',str(env.code),*env.queue_args(),'--kind','campaign',
                                       '--document-sha256',campaign['campaign_sha256']])
        self.run_cli(env.base('seed-final')+self.campaign_args(campaign))
        report_args = env.base('report')+self.campaign_args(campaign)+['--runs',str(env.runs),'--host-validation','minimal']
        before = self.run_cli(report_args)
        materialize = env.base('materialize-final')+['--code',str(env.code),*self.freeze_args(freeze),
            '--campaign-sha256',campaign['campaign_sha256'],'--stage','2']
        self.assertEqual(self.run_cli(materialize)['configs'],100)
        fixtures.git(env.code,'add','-A'); fixtures.git(env.code,'commit','-qm','synthetic stage additions')
        authorization = self.run_cli(env.base('authorize-stage')+['--code',str(env.code),
            '--campaign-sha256',campaign['campaign_sha256'],'--stage','2','--image-id',fixtures.IMAGE,
            '--environment-probe',str(env.probe),'--decided-on','2026-10-29','--decided-after-primary-results','true',
            '--rationale','Eduardo decided after viewing primary INbreast results; declared sensitivity only.',
            '--sealed-at','2026-10-29T18:00:00Z'])
        pin = ['--stage-authorization-sha256',authorization['stage_authorization_sha256']]
        self.assertEqual(self.run_cli(materialize+pin)['configs'],100)
        anchor = env.base('anchor')+['--code',str(env.code),*env.queue_args(),'--kind','stage_authorization',
            '--document-sha256',authorization['stage_authorization_sha256'],'--image-id',fixtures.IMAGE,
            '--environment-probe',str(env.probe)]
        self.run_cli(anchor)
        seeded = self.run_cli(env.base('seed-final')+self.campaign_args(campaign)+pin)
        self.assertEqual((seeded['jobs'],seeded['authorized_seeds']),(100,[13,19]))
        self.run_cli(anchor)  # idempotent even after sensitivity jobs exist
        # Collect a real synthetic successful sensitivity artifact through the existing collector.
        queue = GitQueue(env.checkout,env.remote)
        stage_job = next(j for j in queue.list_jobs() if (j['spec']['arm'],j['spec']['fold'],j['spec']['train_seed']) == ('none',0,13))
        auth_doc = json.loads(Path(authorization['path']).read_bytes())
        campaign_doc = json.loads(Path(campaign['path']).read_bytes())
        raw_claim = queue.claim
        with patch.object(queue,'claim',side_effect=lambda job_id,owner: raw_claim(job_id,owner,
                stage_context=(campaign_doc,auth_doc,env.code))):
            self.finish(queue,stage_job,'succeeded',initial=fixtures.H(['initial',0,13]))
        self.run_cli(env.base('audit-chronology')+self.campaign_args(campaign)+pin)
        after = self.run_cli(report_args+pin)
        self.assertEqual(before['report_sha256'],after['report_sha256'])
        self.assertEqual(Path(before['path']).read_bytes(),Path(after['path']).read_bytes())
        collection = self.run_cli(env.base('collect')+self.campaign_args(campaign)+pin+
            ['--runs',str(env.runs),'--host-validation','minimal','--scope','all_authorized'])
        self.assertEqual(collection['authorized_cells'],150)
        sensitivity = self.run_cli(report_args+pin+['--scope','all_authorized'])
        self.assertNotEqual(before['report_sha256'],sensitivity['report_sha256'])
        sensitivity_doc = json.loads(Path(sensitivity['path']).read_bytes())
        self.assertEqual(sensitivity_doc['matrix']['succeeded_cells'],1)
        self.assertEqual(gr.public_summary(sensitivity_doc)['scope'],'all_authorized')



class StageHpoCliTests(hpo_fixtures.HpoCase):
    def test_hpo_campaign_stage_keeps_plan_freeze_and_primary_manifest(self):
        env = self.env
        # The isolated workspace is long; the D-view embeds two SHA paths.
        if os.name == 'nt':
            env.views = Path('\\\\?\\' + str(env.views.resolve()))
            def clean_long_view():
                fixture_root = Path('\\\\?\\' + str(env.code.parent.resolve()))
                if not env.views.resolve().is_relative_to(fixture_root):
                    raise RuntimeError('fixture cleanup escaped its own root')
                tempfile.TemporaryDirectory._rmtree(str(env.views))
            self.addCleanup(clean_long_view)
        self.to_seeded_hpo()
        self.run_hpo()
        evidence = self.run_cli(self.collect_args())
        freeze = self.seal_freeze(evidence['evidence_sha256'])
        self.run_cli(env.base('anchor') + ['--kind','recipe_freeze','--document-sha256',freeze['freeze_sha256'],
                                         '--freeze-kind','hpo_d', *self.collection_tail()])
        freeze_args = ['--freeze-sha256',freeze['freeze_sha256'],'--freeze-kind','hpo_d','--stages','11;13,19']
        self.run_cli(env.base('materialize-final') + ['--code',str(env.code),*freeze_args])
        fixtures.git(env.code,'add','-A'); fixtures.git(env.code,'commit','-qm','synthetic HPO final configs')
        campaign = self.run_cli(env.base('seal-campaign') + ['--code',str(env.code),*freeze_args,*env.queue_args(),
            '--image-id',fixtures.IMAGE,'--environment-probe',str(env.probe),
            '--sealed-at','2026-10-23T15:00:00Z','--search-space-sha256',self.space['search_space_sha256']])
        self.run_cli(env.base('anchor') + ['--code',str(env.code),*env.queue_args(),'--kind','campaign',
                                         '--document-sha256',campaign['campaign_sha256']])
        primary = (env.code / gj.config_dir(fixtures.DATASET,'final') / 'MANIFEST.json').read_bytes()
        materialize = env.base('materialize-final') + ['--code',str(env.code),*freeze_args,
            '--campaign-sha256',campaign['campaign_sha256'],'--stage','2']
        self.run_cli(materialize)
        fixtures.git(env.code,'add','-A'); fixtures.git(env.code,'commit','-qm','synthetic HPO sensitivity configs')
        authorized = self.run_cli(env.base('authorize-stage') + ['--code',str(env.code),
            '--campaign-sha256',campaign['campaign_sha256'],'--stage','2','--image-id',fixtures.IMAGE,
            '--environment-probe',str(env.probe),'--decided-on','2026-10-29','--decided-after-primary-results','true',
            '--rationale','Post-primary declared sensitivity; original HPO plan and freeze remain fixed.',
            '--sealed-at','2026-10-29T18:00:00Z'])
        document = json.loads(Path(authorized['path']).read_bytes())
        self.assertEqual(document['freeze']['plan_sha256'], self.plan['plan_sha256'])
        self.assertEqual(document['freeze']['sha256'], freeze['freeze_sha256'])
        self.assertEqual(document['stage1_configs']['manifest_sha256'],json.loads(primary)['sha256'])
        pin = ['--stage-authorization-sha256',authorized['stage_authorization_sha256']]
        self.run_cli(materialize + pin)
        self.run_cli(env.base('anchor') + ['--code',str(env.code),*env.queue_args(),'--kind','stage_authorization',
            '--document-sha256',authorized['stage_authorization_sha256'],
            '--image-id',fixtures.IMAGE,'--environment-probe',str(env.probe)])
        seeded = self.run_cli(env.base('seed-final') + self.campaign_args(campaign) + pin)
        self.assertEqual((seeded['jobs'],seeded['authorized_seeds']),(100,[13,19]))
        self.run_cli(env.base('audit-chronology') + self.campaign_args(campaign) + pin)
        self.assertEqual(primary,(env.code/gj.config_dir(fixtures.DATASET,'final')/'MANIFEST.json').read_bytes())

if __name__ == '__main__':
    unittest.main()
