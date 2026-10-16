"""Host worker. Git credentials never enter the training container.

Optional settings.torch_cache is the host TORCH_HOME root, containing
hub/checkpoints/resnet50-11ad3fa6.pth. Pretrained jobs require its exact full
SHA256 from model.weights_sha256; the cache is mounted read-only at /torch-cache.
Random-initialization jobs need no cache. No weight downloads are performed.
"""
from __future__ import annotations
import contextlib
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

from .storage import atomic_json, digest
from .queue import GitQueue, QueueError, QueueConflict
from .telemetry import append_sample


def command(args, timeout=60):
    r = subprocess.run([str(x) for x in args],capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'{args[0]} {args[1] if len(args)>1 else ""} failed ({r.returncode}); inspect private operator logs')
    return r.stdout.strip()


@contextlib.contextmanager
def gpu_lock(root, gpu_uuid):
    """All workers on a host must share this directory, indexed by GPU UUID."""
    import fcntl
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    key=re.sub('[^a-zA-Z0-9_-]','_',gpu_uuid)
    with (root/(key+'.lock')).open('a+') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('GPU already has an AugmentAtlas worker') from None
        yield


PROBE = '''import json,torch,torchvision,numpy,PIL,skimage,importlib.util,importlib.metadata
opencv=None
opencv_distribution=None
if importlib.util.find_spec('cv2') is not None:
    import cv2
    opencv=cv2.__version__
try:
    opencv_distribution=importlib.metadata.version('opencv-python-headless')
except importlib.metadata.PackageNotFoundError:
    pass
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available()
x=torch.randn(2,3,16,16,device='cuda',requires_grad=True)
torch.nn.Conv2d(3,4,3).cuda()(x).square().mean().backward()
torch.cuda.synchronize()
assert x.grad is not None and torch.isfinite(x.grad).all()
print(json.dumps(dict(torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,capability=torch.cuda.get_device_capability(),forward_backward=True,opencv=opencv,opencv_distribution=opencv_distribution)))
'''


IMAGENET_V2_CHECKPOINT = 'resnet50-11ad3fa6.pth'


def expected_weights_sha256(cfg,spec=None):
    """Reconcile immutable config/spec weight identity without importing torch."""
    expected=cfg['model'].get('weights_sha256') if cfg['model']['weights']!='none' else None
    if cfg['model']['weights']!='none':
        if cfg['model']['weights']!='IMAGENET1K_V2' or not isinstance(expected,str) or not re.fullmatch('[a-f0-9]{64}',expected):
            raise ValueError('IMAGENET1K_V2 requires an explicit full lowercase model.weights_sha256')
    if spec:
        declared=[]
        if 'weights_sha256' in spec:declared.append(spec['weights_sha256'])
        if isinstance(spec.get('model'),dict) and 'weights_sha256' in spec['model']:
            declared.append(spec['model']['weights_sha256'])
        if any(value!=expected for value in declared):
            raise ValueError('Job/config pretrained weights checksum mismatch')
    return expected


class DockerRuntime:
    def __init__(self, settings):
        self.s=settings
        self.code=Path(settings['code']).resolve()
        self.data=Path(settings['data']).resolve()
        self.image=None

    def inspect(self,name):
        r=subprocess.run(['docker','inspect',name],capture_output=True,text=True,encoding='utf-8',timeout=20)
        if r.returncode:
            # Distinguish a missing container from an unavailable Docker daemon.
            command(['docker','info','--format','{{.ServerVersion}}'])
            return None
        return json.loads(r.stdout)[0]

    def preflight(self):
        if self.s['phase'] != 'smoke':
            raise ValueError('This prototype only executes explicitly selected smoke phase')
        if not self.code.is_dir() or not self.data.is_dir():
            raise ValueError('code and local data directories must exist')
        if command(['git','-C',self.code,'status','--porcelain']):
            raise ValueError('Training checkout must be clean and fixed to a commit')
        self.commit=command(['git','-C',self.code,'rev-parse','HEAD'])
        self.image=command(['docker','image','inspect',self.s['image'],'--format','{{.Id}}'])
        gpu=command(['nvidia-smi','-i',str(self.s['gpu']),'--query-gpu=uuid,memory.total,temperature.gpu','--format=csv,noheader,nounits']).split(',')
        self.gpu_uuid=gpu[0].strip(); self.memory_mb=float(gpu[1]); self.temperature=float(gpu[2])
        if self.temperature >= self.s.get('max_temperature_c',85):
            raise RuntimeError('GPU too hot for smoke preflight')
        return dict(gpu_uuid=self.gpu_uuid,image_id=self.image)

    def probe(self):
        self.environment=json.loads(command(['docker','run','--rm','--network','none','--gpus','device='+str(self.s['gpu']),
                                            '--entrypoint','python',self.image,'-c',PROBE],timeout=120))
        return self.environment

    def weights_cache_for(self,cfg,spec):
        """Verify host bytes before ownership is claimed; missing cache fails closed."""
        expected=expected_weights_sha256(cfg,spec)
        if expected is None:return None
        setting=self.s.get('torch_cache')
        if not isinstance(setting,str) or not setting.strip():
            raise ValueError('Pretrained jobs require settings.torch_cache pointing to a host TORCH_HOME')
        root=Path(setting).expanduser().resolve()
        checkpoint=(root/'hub'/'checkpoints'/IMAGENET_V2_CHECKPOINT).resolve()
        if not root.is_dir() or not checkpoint.is_relative_to(root) or not checkpoint.is_file():
            raise ValueError('Required cached IMAGENET1K_V2 checkpoint is unavailable')
        if digest(checkpoint)!=expected:
            raise ValueError('Cached IMAGENET1K_V2 checkpoint checksum mismatch')
        return root

    def config_for(self,spec):
        name=spec.get('config')
        if not isinstance(name,str) or not name.startswith('configs/smoke/'):
            raise ValueError('Job requires a config under configs/smoke/')
        path=(self.code/name).resolve()
        if not path.is_relative_to(self.code) or not path.is_file():
            raise ValueError('Invalid training config path')
        from .contracts import load_config,CANDIDATE_BACKEND
        from .data import load_records,prepare_splits
        from .cli import recipe_identity
        import tempfile
        cfg=load_config(path)
        if cfg['augmentation'].get('backend')==CANDIDATE_BACKEND:
            environment=getattr(self,'environment',{})
            if environment.get('opencv')!='4.12.0' or environment.get('opencv_distribution')!='4.12.0.88':
                raise ValueError('candidate-opencv-v2 requires cv2=4.12.0 and opencv-python-headless=4.12.0.88 before claim')
        if spec.get('config_sha256') != digest(path):
            raise ValueError('Training config checksum mismatch')
        self.weights_cache_for(cfg,spec) # streaming hash before a worker may claim
        # Local roots are runtime inputs, never public absolute paths.
        old=os.environ.get('ATLAS_DATA_ROOT'); os.environ['ATLAS_DATA_ROOT']=str(self.data)
        try:
            records=load_records(cfg,check_files=True)
            with tempfile.TemporaryDirectory(prefix='atlas-preflight-') as d:
                splits=prepare_splits(cfg,records,d)
        finally:
            if old is None: os.environ.pop('ATLAS_DATA_ROOT',None)
            else: os.environ['ATLAS_DATA_ROOT']=old
        if cfg['phase']!='smoke' or cfg['scientific_status']!='provisional':
            raise ValueError('Only provisional smoke configurations are executable')
        if spec['train_seed']!=cfg['training']['seed'] or spec['arm']!=cfg['augmentation']['arm']:
            raise ValueError('Job/config seed or arm mismatch')
        if spec['split_hash']!=splits['sha256'] or spec['recipe_hash']!=recipe_identity(cfg):
            raise ValueError('Job/config split or frozen recipe mismatch')
        if spec['dataset']!=cfg['dataset']['id'] or spec['dataset_version']!=cfg['dataset']['version'] or spec['protocol_version']!=cfg['protocol_version'] or spec['fold']!=cfg['splits']['fold']:
            raise ValueError('Job/config version or fold mismatch')
        return name,cfg

    def compatible(self,spec):
        c=spec.get('compatibility',{})
        return (spec['phase']==self.s['phase'] and spec['code_commit']==self.commit
                and spec.get('dataset') in self.s.get('datasets',[])
                and float(c.get('min_memory_mb',0))<=self.memory_mb
                and c.get('image_id',self.image)==self.image
                and c.get('gpu_uuid',self.gpu_uuid)==self.gpu_uuid)

    def create(self,name,spec,out,resume=False):
        config,cfg=self.config_for(spec) # revalidate cache immediately before create
        args=['docker','create','--name',name,'--label','org.augment-atlas.attempt='+name,
              '--network','none','--gpus','device='+str(self.s['gpu']),
              '--user',f'{os.getuid()}:{os.getgid()}','--cpus',str(self.s.get('cpus',2)),
              '--memory',self.s.get('memory','4g'),'--shm-size','256m',
              '--security-opt','no-new-privileges','--cap-drop','ALL',
              '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1','-e','PYTHONPATH=/code/src',
              '-e','ATLAS_DATA_ROOT=/data','-e','HOME=/tmp','-e','OMP_NUM_THREADS=2',
              '-e','ATLAS_CODE_COMMIT='+self.commit,'-e','ATLAS_IMAGE_ID='+self.image,
              '-e','CUBLAS_WORKSPACE_CONFIG=:4096:8',
              '--mount',f'type=bind,src={self.code},dst=/code,readonly',
              '--mount',f'type=bind,src={self.data},dst=/data,readonly',
              '--mount',f'type=bind,src={out},dst=/output']
        if cfg['model']['weights']!='none':
            cache=Path(self.s['torch_cache']).expanduser().resolve()
            args+=['--mount',f'type=bind,src={cache},dst=/torch-cache,readonly','-e','TORCH_HOME=/torch-cache']
        args+=['--workdir','/code','--entrypoint','python',self.image,
              '-m','augment_atlas.train','--config','/code/'+config,'--output','/output','--resume-if-exists']
        if resume: args+=['--resume','/output/checkpoint.pt']
        command(args)


class Spool:
    def __init__(self,path,record):
        self.path=Path(path); self.record=record
        self.events=[]
        if self.path.exists():
            saved=json.loads(self.path.read_text())
            if isinstance(saved,list): self.events=saved
            else:
                if any(saved['record'][k]!=record[k] for k in ('job_id','attempt_id','token')):
                    raise ValueError('Spool belongs to another attempt')
                self.record=saved['record']; self.events=saved['events']

    def save(self):
        atomic_json(self.path,dict(record=self.record,events=self.events))

    def add(self,state,**data):
        self.events.append(dict(event_id=uuid.uuid4().hex,state=state,observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**data))
        self.save()

    def flush(self,queue):
        while self.events:
            event=self.events[0]
            record=queue.update(self.record['job_id'],self.record['attempt_id'],self.record['token'],
                                self.record['version'],event)
            self.record=record
            self.events.pop(0); self.save()
        return self.record


def validate_artifacts(out,spec=None):
    try:
        return _validate_artifacts(out,spec)
    except (TypeError,AttributeError,KeyError) as exc:
        raise ValueError('Malformed completion artifacts') from exc


def _validate_artifacts(out,spec=None):
    out=Path(out)
    p=out/'completed.json'
    if not p.is_file(): raise ValueError('Missing completed.json')
    value=json.loads(p.read_text())
    files=value.get('artifacts',value.get('checksums'))
    if not isinstance(files,dict) or not files: raise ValueError('Missing artifact checksums')
    required={'checkpoint.pt','best.pt','splits.json','config.json','environment.json','metrics.json','results.json','progress.json'}
    if not required.issubset(files): raise ValueError('Completion checksums must cover every required artifact')
    if value.get('status')!='completed': raise ValueError('Completion status is invalid')
    for name,wanted in files.items():
        path=(out/name).resolve()
        if not path.is_relative_to(out.resolve()) or not path.is_file() or path.stat().st_size==0 or digest(path)!=wanted:
            raise ValueError('Invalid completed artifact')
    for name in required:
        if not (out/name).is_file(): raise ValueError('Missing required result')
    from .contracts import load_config,canonical_hash
    cfg=load_config(out/'config.json')
    config_hash=canonical_hash(cfg)
    splits=json.loads((out/'splits.json').read_text())
    if splits.get('sha256')!=canonical_hash({k:v for k,v in splits.items() if k!='sha256'}):
        raise ValueError('Invalid split manifest checksum')
    progress=json.loads((out/'progress.json').read_text())
    results=json.loads((out/'results.json').read_text())
    metrics=json.loads((out/'metrics.json').read_text())
    environment=json.loads((out/'environment.json').read_text())
    weights_hash=expected_weights_sha256(cfg,spec)
    if weights_hash is not None and (environment.get('weights')!=cfg['model']['weights'] or environment.get('weights_sha256')!=weights_hash):
        raise ValueError('Environment pretrained weights identity mismatch')
    if not isinstance(metrics,list) or len(metrics)!=cfg['training']['epochs']:
        raise ValueError('Incomplete epoch history')
    for record in (value,progress,results,environment):
        if record.get('config_sha256')!=config_hash or record.get('splits_sha256')!=splits['sha256']:
            raise ValueError('Completion config/split identity mismatch')
    if (progress.get('status')!='completed' or type(progress.get('epoch')) is not int
            or progress['epoch']!=cfg['training']['epochs'] or type(value.get('epoch')) is not int
            or value['epoch']!=progress['epoch']):
        raise ValueError('Training did not complete every configured epoch')
    scope=cfg['training'].get('evaluation_scope','test')
    if scope=='selection-only':
        if results.get('evaluation_scope')!=scope or environment.get('evaluation_scope')!=scope or 'test' in results:
            raise ValueError('Selection-only evaluation scope mismatch')
        evaluation=results.get('selection',{})
        if spec and 'recipe_hash' in spec:
            from .cli import recipe_identity
            if spec['recipe_hash']!=recipe_identity(cfg):
                raise ValueError('Selection-only frozen recipe identity mismatch')
    else:
        if results.get('evaluation_scope','test')!='test' or environment.get('evaluation_scope','test')!='test':
            raise ValueError('Test evaluation scope mismatch')
        evaluation=results.get('test',{})
    loss,accuracy=evaluation.get('loss'),evaluation.get('accuracy')
    if type(evaluation.get('samples')) is not int or evaluation['samples']<=0 or type(loss) not in (float,int) or not math.isfinite(loss) or type(accuracy) not in (float,int) or not math.isfinite(accuracy) or not 0<=accuracy<=1:
        raise ValueError('Invalid evaluation artifact')
    if spec and (spec['split_hash']!=splits['sha256'] or cfg['training']['seed']!=spec['train_seed'] or cfg['augmentation']['arm']!=spec['arm']):
        raise ValueError('Result belongs to another job')
    summary=dict(completed_sha256=digest(p),artifact_count=len(files),epoch=progress['epoch'],epochs=cfg['training']['epochs'])
    samples=progress.get('samples_processed')
    if type(samples) is int and samples>=0:summary['samples_processed']=samples
    return summary


def worker(settings,once=False):
    runtime=DockerRuntime(settings)
    runtime.preflight()
    root=Path(settings['runs']).resolve(); root.mkdir(parents=True,exist_ok=True)
    queue=GitQueue(settings['queue_checkout'],settings['remote'],settings.get('branch','queue'))
    with gpu_lock(settings.get('locks','/tmp/augment-atlas-gpu-locks'),runtime.gpu_uuid):
        runtime.probe()
        owner=dict(server=settings['server'],gpu=str(settings['gpu']),gpu_uuid=runtime.gpu_uuid)
        slot=root/('worker-'+runtime.gpu_uuid); slot.mkdir(exist_ok=True)
        active=slot/'active.json'
        stopping=[False]
        previous={}
        for sig in (signal.SIGTERM,signal.SIGINT):
            previous[sig]=signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
        try:
            while not stopping[0]:
                if active.exists():
                    local=json.loads(active.read_text())
                    record=local['record']; out=Path(local['output']); name=local['container']
                    if record['spec']['code_commit']!=runtime.commit or local['image']!=runtime.image:
                        raise RuntimeError('Active attempt code/image differs; restore original environment')
                else:
                    jobs=queue.list_jobs()
                    if any(j['state'] in ('claimed','running') and j.get('owner')==owner for j in jobs):
                        raise RuntimeError('Remote active attempt lacks local journal; manual recovery required')
                    record=None
                    for candidate in jobs:
                        if candidate['state']!='pending' or not runtime.compatible(candidate['spec']): continue
                        runtime.config_for(candidate['spec']) # invalid inputs fail BEFORE claim
                        record=queue.claim(candidate['job_id'],owner)
                        if record: break
                    if not record:
                        if once:return
                        time.sleep(settings.get('poll_seconds',30)); continue
                    out=root/record['job_id']/record['attempt_id']; out.mkdir(parents=True,exist_ok=False)
                    name='atlas-'+record['attempt_id'].replace('-','')
                    local=dict(record=record,output=str(out),container=name,image=runtime.image,stage='prepared')
                    atomic_json(active,local)
                    atomic_json(out/'runtime.json',dict(job_id=record['job_id'],attempt_id=record['attempt_id'],
                                code_commit=runtime.commit,image_id=runtime.image,owner=owner,environment=runtime.environment))
                spool=Spool(out/'events-pending.json',record)
                record=spool.record
                container=runtime.inspect(name)
                if record['state']=='interrupted':
                    if not settings.get('resume'):
                        raise RuntimeError('Attempt interrupted; use explicit --resume after inspecting the stopped container')
                    if container is None or container['State']['Running'] or not (out/'checkpoint.pt').is_file():
                        raise RuntimeError('Resume requires confirmed stopped container and valid local checkpoint')
                    if local['stage']!='resuming':
                        local['stage']='resuming'; local['prior_started_at']=container['State'].get('StartedAt')
                        atomic_json(active,local)
                    if not any(e.get('manual_resume') for e in spool.events):spool.add('running',manual_resume=True)
                    spool.flush(queue) # confirmed remote resume before starting again
                    record=spool.record; local['record']=record; atomic_json(active,local)
                if local['stage']=='resuming':
                    if record['state']!='running':raise RuntimeError('Unconfirmed resume remains stopped')
                    if container is None:raise RuntimeError('Resume container disappeared')
                    if not container['State']['Running'] and container['State'].get('StartedAt')==local['prior_started_at']:
                        command(['docker','start',name])
                    local['stage']='launched'; atomic_json(active,local)
                    container=runtime.inspect(name)
                if container is None:
                    if local['stage']!='prepared':
                        raise RuntimeError('Container disappeared; inspect previous execution before recovery')
                    runtime.create(name,record['spec'],out)
                    container=runtime.inspect(name)
                if container['State']['Status']=='created':
                    local['stage']='launched'; atomic_json(active,local)
                    if not any(e['state']=='running' for e in spool.events) and record['state']=='claimed':
                        spool.add('running',epoch=0)
                    try: spool.flush(queue)
                    except QueueConflict: raise
                    except QueueError: pass
                    local['record']=spool.record; atomic_json(active,local)
                    command(['docker','start',name])
                last_progress=last_sample=0
                while True:
                    state=runtime.inspect(name)
                    if state is None: raise RuntimeError('Active container missing')
                    if stopping[0] and state['State']['Running']:
                        command(['docker','kill','--signal','TERM',name])
                    now=time.monotonic()
                    progress={}
                    if (out/'progress.json').exists():
                        try: progress=json.loads((out/'progress.json').read_text())
                        except (OSError,ValueError): pass
                    # Publish only a fixed safe numeric subset.
                    numbers={k:progress[k] for k in ('epoch','epochs','samples_processed','throughput') if isinstance(progress.get(k),(int,float))}
                    if now-last_sample>=settings.get('telemetry_seconds',30):
                        row=append_sample(out/'telemetry.jsonl',dict(job_id=record['job_id'],attempt_id=record['attempt_id'],
                             server=settings['server'],code_commit=runtime.commit,image_id=runtime.image,config_sha256=record['spec']['config_sha256'],**numbers),
                             settings['gpu'],out,settings.get('optional_collectors',True))
                        last_sample=now
                        if row.get('gpu'):
                            try:
                                if float(row['gpu']['temperature.gpu'])>=settings.get('max_temperature_c',85):
                                    stopping[0]=True; command(['docker','kill','--signal','TERM',name])
                            except (ValueError,TypeError): pass
                    if now-last_progress>=settings.get('progress_seconds',3600) and state['State']['Running']:
                        spool.add('running',**numbers); last_progress=now
                    try:
                        spool.flush(queue); local['record']=spool.record; atomic_json(active,local)
                    except QueueConflict:
                        if state['State']['Running']: command(['docker','kill','--signal','TERM',name])
                        raise
                    except QueueError: pass # running child and checkpoints survive network failure
                    if not state['State']['Running']:break
                    time.sleep(settings.get('monitor_seconds',2))
                (out/'docker.log').write_text(command(['docker','logs',name]),encoding='utf-8')
                # Recovery after an already-spooled terminal event must not append another.
                if not any(e['state'] in ('succeeded','failed','interrupted') for e in spool.events) and spool.record['state'] not in ('succeeded','failed','interrupted'):
                    try:
                        if state['State']['ExitCode']!=0:raise ValueError('Nonzero training exit')
                        summary=validate_artifacts(out,record['spec'])
                        spool.add('succeeded',**summary)
                    except (ValueError,OSError,KeyError):
                        spool.add('interrupted' if (out/'checkpoint.pt').exists() and (stopping[0] or progress.get('status')=='interrupted') else 'failed',exit_code=state['State']['ExitCode'])
                while spool.events:
                    try: spool.flush(queue)
                    except QueueConflict:raise
                    except QueueError:
                        if stopping[0]: return # durable spool is reconciled on restart
                        time.sleep(settings.get('poll_seconds',30))
                if spool.record['state']=='interrupted':
                    local['record']=spool.record; local['stage']='interrupted'; atomic_json(active,local)
                    return
                active.unlink()
                if once:return
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
