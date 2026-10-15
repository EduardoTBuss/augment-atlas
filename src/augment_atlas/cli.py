"""Explicit phase/config CLI. Importing this module does not import PyTorch."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from .contracts import load_config,canonical_hash
from .data import load_records,prepare_splits
from .queue import GitQueue
from .runtime import worker,command
from .storage import digest


def recipe_identity(cfg):
    train={k:v for k,v in cfg['training'].items() if k!='seed'}
    return canonical_hash(dict(recipe=cfg['recipe'],model=cfg['model'],input=cfg['input'],training=train))


def main():
    p=argparse.ArgumentParser(description='AugmentAtlas provisional smoke prototype')
    sub=p.add_subparsers(dest='action',required=True)
    w=sub.add_parser('worker'); w.add_argument('--settings',required=True); w.add_argument('--phase',required=True,choices=['smoke'])
    w.add_argument('--once',action='store_true'); w.add_argument('--resume',action='store_true',help='Explicitly resume the stopped checkpoint in the current attempt')
    s=sub.add_parser('seed',help='Add exactly one explicit technical job')
    s.add_argument('--config',required=True); s.add_argument('--phase',required=True,choices=['smoke'])
    s.add_argument('--remote',required=True); s.add_argument('--checkout',required=True); s.add_argument('--branch',default='queue')
    s.add_argument('--code',default='.'); s.add_argument('--data',required=True)
    l=sub.add_parser('status'); l.add_argument('--remote',required=True); l.add_argument('--checkout',required=True); l.add_argument('--branch',default='queue')
    a=p.parse_args()
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
        spec=dict(protocol_version=cfg['protocol_version'],dataset=cfg['dataset']['id'],dataset_version=cfg['dataset']['version'],
                  split_hash=splits['sha256'],recipe_hash=recipe_identity(cfg),phase=a.phase,fold=cfg['splits']['fold'],
                  train_seed=cfg['training']['seed'],arm=cfg['augmentation']['arm'],
                  code_commit=command(['git','-C',root,'rev-parse','HEAD']),config=config.relative_to(root).as_posix(),
                  config_sha256=digest(config),dependencies=cfg['recipe']['dependencies'],compatibility={'min_memory_mb':1000})
        result=GitQueue(a.checkout,a.remote,a.branch).seed([spec])
        for record in result:print(record['job_id'])


if __name__=='__main__':main()
