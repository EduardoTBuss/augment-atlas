"""Explicit phase/config CLI. Importing this module does not import PyTorch."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from . import fleet
from .contracts import load_config,canonical_hash,recipe_identity
from .data import load_records,prepare_splits
from .queue import GitQueue,QueueError,parse_record
from .runtime import worker,command,expected_weights_sha256
from .storage import digest


# recipe_identity is defined in contracts (stdlib-only) and re-exported here.


def legacy_seed_spec(cfg,phase,split_hash,code_commit,config_rel,config_sha256):
    """The pre-fleet seed spec, byte for byte (b7a47f5). Its identity fields are the job_id."""
    return dict(protocol_version=cfg['protocol_version'],dataset=cfg['dataset']['id'],dataset_version=cfg['dataset']['version'],
                split_hash=split_hash,recipe_hash=recipe_identity(cfg),phase=phase,fold=cfg['splits']['fold'],
                train_seed=cfg['training']['seed'],arm=cfg['augmentation']['arm'],
                code_commit=code_commit,config=config_rel,
                config_sha256=config_sha256,dependencies=cfg['recipe']['dependencies'],compatibility={'min_memory_mb':1000})


def fleet_seed_spec(cfg,phase,split_hash,code_commit,config_rel,config_sha256,a):
    """Legacy spec + additive spec.requirements/compatibility/scheduling (fleet v1, section 3.2)."""
    container=None
    if (a.container_cpus is None)!=(a.container_memory_bytes is None):
        raise ValueError('--container-cpus and --container-memory-bytes go together')
    if a.container_cpus is not None:
        cpus=int(a.container_cpus) if float(a.container_cpus).is_integer() else a.container_cpus
        container={'cpus':cpus,'memory_bytes':a.container_memory_bytes}
    cc=None
    if a.min_compute_capability:
        try:cc=[int(part) for part in a.min_compute_capability.split('.')]
        except ValueError:raise ValueError('--min-compute-capability must look like 5.2') from None
    req=fleet.seed_requirements(cfg,phase=phase,image_id=a.image_id,weights_sha256=expected_weights_sha256(cfg),
                                image_ref=a.image_ref,min_vram_mib=a.min_vram_mib,min_compute_capability=cc,
                                hardware_class=a.hardware_class,allowed_nodes=a.allowed_node,
                                min_free_disk_bytes=a.min_free_disk_bytes,container=container,
                                expected_epoch_seconds_max=a.expected_epoch_seconds_max)
    scheduling=None
    if a.priority is not None or a.order is not None:
        scheduling={'priority':a.priority or 0,'order':a.order}
    return fleet.with_requirements(legacy_seed_spec(cfg,phase,split_hash,code_commit,config_rel,config_sha256),req,scheduling)


def _fleet_flags_used(a):
    """True if any fleet-only seed flag differs from its default (legacy seeding must stay byte-identical)."""
    return any(getattr(a,name) not in (None,default) for name,default in
               [('image_ref',None),('min_vram_mib',1000),('min_compute_capability',None),('hardware_class',None),
                ('allowed_node',None),('min_free_disk_bytes',0),('container_cpus',None),('container_memory_bytes',None),
                ('expected_epoch_seconds_max',None),('priority',None),('order',None)])


def _json_file(path):
    """None if absent, 'unreadable' if not JSON (the gates refuse both with distinct reasons)."""
    if path is None:return None
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return None
    except ValueError:return 'unreadable'


def fleet_explain(a):
    """Read-only steps 0-5 of section 7.1 over local files (or a read-only queue fetch)."""
    if a.jobs_dir:
        records=[parse_record(path.read_text(encoding='utf-8')) for path in sorted(Path(a.jobs_dir).glob('*.md'))]
    else:
        if not (a.remote and a.checkout):raise ValueError('use --jobs-dir or --remote with --checkout')
        records=GitQueue(a.checkout,a.remote,a.branch).list_jobs()
    worker_facts=_json_file(a.worker)
    if not isinstance(worker_facts,dict):raise ValueError('--worker must be a JSON object of worker facts')
    profile=_json_file(a.profile)
    if a.profile and not isinstance(profile,dict):raise ValueError('--profile must be a readable JSON object')
    now=time.monotonic() if a.now_mono is None else a.now_mono
    report=fleet.explain(records,profile,worker_facts,latest=_json_file(a.latest),now=now,card=_json_file(a.card),
                         idle_since_mono=a.idle_since_mono,job_id=a.job)
    if a.json:print(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    else:sys.stdout.write(fleet.format_explain(report))
    return 0 if report['order'] else 1


def main(argv=None):
    p=argparse.ArgumentParser(description='AugmentAtlas provisional smoke prototype')
    sub=p.add_subparsers(dest='action',required=True)
    w=sub.add_parser('worker'); w.add_argument('--settings',required=True); w.add_argument('--phase',required=True,choices=['smoke','hpo','final'])
    w.add_argument('--once',action='store_true'); w.add_argument('--resume',action='store_true',help='Explicitly resume the stopped checkpoint in the current attempt')
    s=sub.add_parser('seed',help='Add exactly one explicit technical job')
    s.add_argument('--config',required=True); s.add_argument('--phase',required=True,choices=['smoke'])
    s.add_argument('--remote',required=True); s.add_argument('--checkout',required=True); s.add_argument('--branch',default='queue')
    s.add_argument('--code',default='.'); s.add_argument('--data',required=True)
    s.add_argument('--image-id',help='sha256:<64hex> from docker image inspect --format {{.Id}}; omit (with no other fleet flag) for the legacy pre-fleet spec')
    s.add_argument('--image-ref'); s.add_argument('--min-vram-mib',type=int,default=1000)
    s.add_argument('--min-compute-capability',help='e.g. 5.2'); s.add_argument('--hardware-class')
    s.add_argument('--allowed-node',action='append',help='repeatable; default: every node')
    s.add_argument('--min-free-disk-bytes',type=int,default=0)
    s.add_argument('--container-cpus',type=float); s.add_argument('--container-memory-bytes',type=int)
    s.add_argument('--expected-epoch-seconds-max',type=int)
    s.add_argument('--priority',type=int); s.add_argument('--order',type=int)
    l=sub.add_parser('status'); l.add_argument('--remote',required=True); l.add_argument('--checkout',required=True); l.add_argument('--branch',default='queue')
    f=sub.add_parser('fleet',help='Fleet protocol v1 tools (read-only)')
    fsub=f.add_subparsers(dest='fleet_action',required=True)
    e=fsub.add_parser('explain',help='Why each job would (not) be claimed by this worker; reads only')
    e.add_argument('--worker',required=True,help='JSON worker facts (see augment_atlas.fleet)')
    e.add_argument('--profile',help='node_profile.json; omit for the explicit legacy mode')
    e.add_argument('--card',help='published card fleet/nodes/<node_id>.json'); e.add_argument('--latest',help='sentinel latest.json')
    e.add_argument('--jobs-dir',help='directory of rendered jobs/<job_id>.md')
    e.add_argument('--remote'); e.add_argument('--checkout'); e.add_argument('--branch',default='queue')
    e.add_argument('--now-mono',type=float,help='consumer monotonic clock (default: now)')
    e.add_argument('--idle-since-mono',type=float); e.add_argument('--job'); e.add_argument('--json',action='store_true')
    a=p.parse_args(argv)
    if a.action=='seed' and not a.image_id and _fleet_flags_used(a):
        p.error('fleet seed flags require --image-id (without any of them the legacy spec is seeded)')
    if a.action=='fleet':
        try:return fleet_explain(a)
        except (ValueError,OSError,QueueError) as exc:
            print(json.dumps({'error':str(exc)}),file=sys.stderr); return 2
    if a.action=='worker':
        settings=json.loads(Path(a.settings).read_text())
        if settings.get('phase')!=a.phase:raise ValueError('Settings phase must match explicit CLI phase')
        settings['resume']=a.resume
        worker(settings,a.once)
    elif a.action=='status':
        for record in GitQueue(a.checkout,a.remote,a.branch).list_jobs():
            print(record['job_id'],record['state'],record['attempt_id'])
    else:
        root=Path(a.code).resolve(); config=Path(a.config).resolve(); cfg=load_config(config)
        if not config.is_relative_to(root): raise ValueError('Config must be in pinned code checkout')
        os.environ['ATLAS_DATA_ROOT']=str(Path(a.data).resolve())
        records=load_records(cfg)
        with tempfile.TemporaryDirectory(prefix='atlas-splits-') as d:
            splits=prepare_splits(cfg,records,d)
        seed_args=(cfg,a.phase,splits['sha256'],command(['git','-C',root,'rev-parse','HEAD']),config.relative_to(root).as_posix(),digest(config))
        spec=fleet_seed_spec(*seed_args,a) if a.image_id else legacy_seed_spec(*seed_args)
        result=GitQueue(a.checkout,a.remote,a.branch).seed([spec])
        for record in result:print(record['job_id'])


if __name__=='__main__':sys.exit(main())
