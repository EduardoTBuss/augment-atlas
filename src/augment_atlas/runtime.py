"""Host worker. Git credentials never enter the training container.

Optional settings.torch_cache is the host TORCH_HOME root, containing
hub/checkpoints/resnet50-11ad3fa6.pth. Pretrained jobs require its exact full
SHA256 from model.weights_sha256; the cache is mounted read-only at /torch-cache.
Random-initialization jobs need no cache. No weight downloads are performed.
"""
from __future__ import annotations
import contextlib
import copy
from dataclasses import dataclass
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import uuid
import zipfile

from .storage import atomic_json, digest
from .queue import GitQueue, QueueError, QueueConflict, StaleAttempt
from .telemetry import append_sample


def command(args, timeout=60):
    r = subprocess.run([str(x) for x in args],capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'{args[0]} {args[1] if len(args)>1 else ""} failed ({r.returncode}); inspect private operator logs')
    return r.stdout.strip()


def request_training_stop(runtime, observed):
    """Signal the inspected container ID once; tolerate only confirmed exit races.

    A TERM handler may finish training between inspect and the Docker request.
    A failing client is harmless only when that exact container is now exited.
    Missing containers, daemon failures and live/changed identities still fail.
    """
    container_id=observed.get('Id')
    if not isinstance(container_id,str) or not container_id:
        raise RuntimeError('Cannot signal training without its inspected container ID')
    try:
        command(['docker','kill','--signal','TERM',container_id])
    except (RuntimeError,subprocess.TimeoutExpired):
        latest=runtime.inspect(container_id)
        if (latest is None or latest.get('Id')!=container_id or
                latest.get('State',{}).get('Running') is not False or
                latest.get('State',{}).get('Status') not in ('exited','dead')):
            raise
        return latest
    return observed


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


# --------------------------------------------------------------------------
# Global hpo/final branch (slice S1, part C). Errors are ValueError("<code>: ...")
# with the codes of section 6.11 of docs/global-next-integration.md.
# --------------------------------------------------------------------------

GLOBAL_RUNTIME_PHASES = ('hpo','final')
D_VIEW_LISTING = 'LISTING.json'


def _global_fail(code,message):
    raise ValueError(f'{code}: {message}')


def is_global_spec(spec):
    """A queue spec belongs to the global branch by phase OR by config location."""
    return isinstance(spec,dict) and (spec.get('phase') in GLOBAL_RUNTIME_PHASES or
                                      (isinstance(spec.get('config'),str) and spec['config'].startswith('configs/global/')))


def _views_root(settings):
    setting=settings.get('views')
    if not isinstance(setting,str) or not setting.strip():
        _global_fail('hpo_isolation_unproven','HPO workers require settings.views (host root of views/<pools>/D); '
                     'the full data root is never mounted for HPO')
    root=Path(setting).expanduser().resolve()
    if not root.is_dir():
        _global_fail('hpo_isolation_unproven','settings.views is not an existing host directory')
    return root


def _read_host_json(root,relative,code):
    from .data import confined_path
    try:
        path=confined_path(Path(root),relative)
        if not path.is_file():
            _global_fail(code,'a private global document is missing from the host data root')
        with path.open(encoding='utf-8') as stream:
            return json.load(stream)
    except (OSError,ValueError,RecursionError) as error:
        if str(error).startswith(code+':'):raise
        _global_fail(code,f'private global document cannot be read ({type(error).__name__})')


@dataclass(frozen=True)
class GlobalTrustedInputs:
    """Host-derived inputs of ONE global job; nothing comes from the attempt output.

    ``config_bytes`` are the materialized bytes pinned by the job (sha ==
    spec.config_sha256 == MANIFEST entry); ``records`` are the manifest records
    projected to {id, path, group, label}; ``image_sha256`` maps path -> sha256
    from the host manifest; ``view`` is the hpo-view (hpo) or the re-derived
    ``global-final-view-v1`` document (final), whose sha256 is the split_hash.
    """
    cfg: dict
    config_name: str
    config_bytes: bytes
    config_sha256: str
    records: list
    image_sha256: dict
    pools: dict
    view: dict
    recipe: dict


def _global_config_bytes(code,spec):
    """Only the expected materialized path; bytes, MANIFEST and canonical rendering agree."""
    from . import global_jobs as gj
    from .contracts import canonical_hash
    name=spec.get('config'); phase=spec.get('phase'); dataset=spec.get('dataset')
    if phase not in GLOBAL_RUNTIME_PHASES or not isinstance(dataset,str):
        _global_fail('global_config_invalid','global jobs require phase hpo|final and a dataset id')
    directory=gj.config_dir(dataset,phase)
    if (not isinstance(name,str) or not name.startswith(directory+'/') or '/' in name[len(directory)+1:]
            or not name.endswith('.json') or name.endswith('/MANIFEST.json')):
        _global_fail('global_config_invalid',f'global job config must be a materialized file under {directory}/')
    code=Path(code).resolve()
    path=(code/name).resolve()
    if path!=code.joinpath(*name.split('/')) or not path.is_relative_to(code) or not path.is_file():
        _global_fail('global_config_invalid','invalid or indirect global config path')
    data=path.read_bytes()
    sha=gj._bytes_sha256(data)
    if sha!=spec.get('config_sha256'):
        _global_fail('config_identity_mismatch','config bytes differ from the job config_sha256')
    try:
        cfg=json.loads(data.decode('ascii'))
    except (UnicodeDecodeError,ValueError,RecursionError) as error:
        _global_fail('config_identity_mismatch',f'materialized config cannot be parsed ({type(error).__name__})')
    gj.validate_global_config(cfg)
    if gj.render_config_bytes(cfg)!=data:
        _global_fail('config_identity_mismatch','config bytes are not the deterministic materialized rendering')
    manifest_path=path.parent/'MANIFEST.json'
    try:
        manifest_bytes=manifest_path.read_bytes()
        manifest=json.loads(manifest_bytes.decode('ascii'))
    except (OSError,UnicodeDecodeError,ValueError,RecursionError) as error:
        _global_fail('config_identity_mismatch',f'MANIFEST.json of the phase directory is unreadable ({type(error).__name__})')
    keys={'schema_version','protocol','phase','dataset','pools_sha256','plan_sha256','freeze_sha256','files','sha256'}
    if type(manifest) is not dict or set(manifest)!=keys or gj.render_config_bytes(manifest)!=manifest_bytes:
        _global_fail('config_identity_mismatch','MANIFEST.json is not a materialized global manifest')
    bound=('plan_sha256','freeze_sha256') if phase=='hpo' else ('freeze_sha256','plan_sha256')
    if (manifest['sha256']!=canonical_hash({k:v for k,v in manifest.items() if k!='sha256'}) or
            manifest['protocol']!=gj.MANIFEST_PROTOCOL or manifest['phase']!=phase or
            manifest['dataset']!={'id':cfg['dataset']['id'],'version':cfg['dataset']['version']} or
            manifest['pools_sha256']!=cfg['global']['pools_sha256'] or
            manifest[bound[0]]!=cfg['global'][bound[0]] or manifest[bound[1]] is not None):
        _global_fail('config_identity_mismatch','MANIFEST.json is not bound to this config phase/pools/plan|freeze')
    if type(manifest['files']) is not list or [f for f in manifest['files'] if type(f) is dict and f.get('path')==name]!=[{'path':name,'sha256':sha}]:
        _global_fail('config_identity_mismatch','MANIFEST.json does not pin these config bytes')
    return name,data,sha,cfg


def load_global_trusted_inputs(code,data,spec):
    """Trusted inputs of a global job from the pinned checkout and host data root only."""
    from . import global_development as gd
    from . import global_evaluation as ge
    from . import global_jobs as gj
    from .contracts import canonical_hash
    from .data import load_records
    from .schema import IDENTITY_FIELDS
    name,config_bytes,sha,cfg=_global_config_bytes(code,spec)
    if not isinstance(data,(str,os.PathLike)) or not str(data).strip():
        _global_fail('records_identity_mismatch','global jobs require settings.data; no default data root is assumed')
    root=Path(data).expanduser().resolve()
    if not root.is_dir():
        _global_fail('records_identity_mismatch','global data root must be an existing host directory')
    bound=cfg['global']
    try:
        loaded=load_records({'dataset':dict(cfg['dataset']),'model':{'num_classes':cfg['model']['num_outputs']}},
                            check_files=True,root=root)
    except (OSError,ValueError) as error:
        _global_fail('records_identity_mismatch',f'host manifest: {error}')
    records=sorted(({k:row[k] for k in ('id','path','group','label')} for row in loaded),key=lambda row:row['id'])
    images={row['path']:row.get('sha256') for row in loaded}
    pools=_read_host_json(root,bound['pools_file'],'pools_anchor_mismatch')
    if type(pools) is not dict or pools.get('records_sha256')!=canonical_hash(records):
        _global_fail('records_identity_mismatch','host manifest records differ from pools.records_sha256')
    gj._anchored(records,pools,bound['pools_sha256'],global_task=True)
    if cfg['phase']=='hpo':
        plan=_read_host_json(root,gj._private_name('plan',cfg['dataset']['id'],bound['plan_sha256']),'plan_invalid')
        try:
            candidates=gd._plan(plan,pools,bound['plan_sha256'])
        except ValueError as error:
            _global_fail('plan_invalid',str(error))
        candidate=candidates.get(bound['candidate_id'])
        if candidate is None or canonical_hash(candidate)!=bound['candidate_config_sha256']:
            _global_fail('recipe_mismatch','config candidate is not the anchored plan candidate')
        if cfg['training']['seed'] not in plan['seeds']:
            _global_fail('global_config_invalid','HPO seed is not a selection seed of the anchored plan')
        recipe=candidate['recipe']
        gj.validate_global_config(cfg,recipe=recipe)
        view=_read_host_json(root,bound['hpo_view_file'],'view_hash_mismatch')
        ge.check_hpo_view(view,cfg,records,pools)
        expected=gj.hpo_job_spec(cfg,config_path=name,config_sha256=sha,code_commit=spec.get('code_commit'))
    else:
        freeze=_read_host_json(root,bound['freeze_file'],'freeze_not_sealed')
        if type(freeze) is not dict or freeze.get('sha256')!=bound['freeze_sha256']:
            _global_fail('freeze_not_sealed','freeze document is not the anchored freeze of this config')
        if bound['freeze_kind']=='external_fixed':
            gj.audit_external_freeze(records,pools,freeze,expected_pools_sha256=bound['pools_sha256'])
            recipe=freeze['recipe']
        else:
            # Full hpo_d audit needs the anchored evidence/plan (S3 campaign seal);
            # the campaign gate refuses final claims until then.
            try:
                gd._checksum(freeze)
            except ValueError as error:
                _global_fail('freeze_not_sealed',str(error))
            if freeze.get('pools_sha256')!=bound['pools_sha256']:
                _global_fail('freeze_not_sealed','freeze bound to other pools')
            recipe=freeze.get('frozen_recipe')
        gj.validate_global_config(cfg,recipe=recipe)
        view=gj.derive_final_view_doc(records,pools,bound['fold'],expected_pools_sha256=bound['pools_sha256'],
                                      freeze_sha256=bound['freeze_sha256'])
        ge.check_final_view(view,cfg,records,pools)
        campaign=(spec.get('global') or {}).get('campaign_sha256') if isinstance(spec.get('global'),dict) else None
        expected=gj.final_job_spec(cfg,view,config_path=name,config_sha256=sha,code_commit=spec.get('code_commit'),
                                   campaign_sha256=campaign)
    for key in tuple(IDENTITY_FIELDS)+('job_id','config','config_sha256','global','dependencies'):
        if key not in spec or canonical_hash(spec[key])!=canonical_hash(expected[key]):
            _global_fail('global_config_invalid',f'job spec {key} differs from the materialized config identity')
    return GlobalTrustedInputs(cfg=copy.deepcopy(cfg),config_name=name,config_bytes=config_bytes,config_sha256=sha,
                               records=copy.deepcopy(records),image_sha256=dict(images),pools=copy.deepcopy(pools),
                               view=copy.deepcopy(view),recipe=copy.deepcopy(recipe))


def d_view_listing(directory,view_file):
    """Recompute the D-view listing [[path, sha256], ...] sorted by path (spec 6.3).

    Excluded: the hpo-view document itself (its hash depends on this listing)
    and an optional LISTING.json, which must then equal the listing. Symlinks
    and non-regular files are refused: they could reach E outside the view.
    """
    from .contracts import canonical_hash
    directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        _global_fail('hpo_isolation_unproven','D-view is not a real directory')
    listing=[]; extra=None
    for base,folders,files in os.walk(directory,followlinks=False):
        for folder in folders:
            if (Path(base)/folder).is_symlink():
                _global_fail('hpo_isolation_unproven','D-view contains a symlinked directory')
        for file in files:
            path=Path(base)/file
            relative=path.relative_to(directory).as_posix()
            if path.is_symlink() or not path.is_file():
                _global_fail('hpo_isolation_unproven','D-view contains a symlink or non-regular file')
            if relative==view_file:continue
            if relative==D_VIEW_LISTING:
                extra=path;continue
            listing.append([relative,digest(path)])
    listing.sort()
    if extra is not None:
        try:
            same=canonical_hash(json.loads(extra.read_text(encoding='utf-8')))==canonical_hash(listing)
        except (OSError,TypeError,ValueError,RecursionError):
            same=False
        if not same:
            _global_fail('hpo_isolation_unproven','D-view LISTING.json differs from the recomputed listing')
    return listing


def expected_d_view(settings,pools_sha256):
    return _views_root(settings)/pools_sha256/'D'


def verify_d_view(settings,trusted):
    """Recompute the D-view of an HPO job; return the ONLY data mount allowed.

    Files present == hpo-view record paths, bytes == host manifest hashes,
    listing hash == view.view_files_sha256, the hpo-view inside == host view.
    """
    from .contracts import canonical_hash
    cfg,view=trusted.cfg,trusted.view
    directory=expected_d_view(settings,cfg['global']['pools_sha256'])
    if not directory.resolve().is_relative_to(_views_root(settings)) or not directory.is_dir():
        _global_fail('hpo_isolation_unproven','materialized D-view views/<pools_sha256>/D is missing')
    view_file=cfg['global']['hpo_view_file']
    try:
        inside=json.loads((directory/view_file).read_text(encoding='utf-8'))
    except (OSError,ValueError,RecursionError):
        _global_fail('hpo_isolation_unproven','D-view lacks the hpo-view document read by the container')
    if canonical_hash(inside)!=canonical_hash(view):
        _global_fail('view_hash_mismatch','hpo-view inside the D-view differs from the host hpo-view')
    listing=d_view_listing(directory,view_file)
    paths=[path for path,_ in listing]
    if paths!=sorted(row['path'] for row in view['records']):
        _global_fail('hpo_isolation_unproven','D-view files are not exactly the hpo-view (D) images')
    if any(trusted.image_sha256.get(path)!=sha for path,sha in listing):
        _global_fail('records_identity_mismatch','D-view image bytes differ from the host manifest hashes')
    if canonical_hash(listing)!=view['view_files_sha256']:
        _global_fail('hpo_isolation_unproven','recomputed D-view listing differs from view_files_sha256')
    return dict(kind='d_view',source=str(directory.resolve()),destination='/data',readonly=True,
                pools_sha256=cfg['global']['pools_sha256'],hpo_view_sha256=view['sha256'],
                view_files_sha256=view['view_files_sha256'])


def record_data_mount(out,mount):
    """Add the verified /data mount to the attempt's runtime.json before docker create."""
    path=Path(out)/'runtime.json'
    provenance=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    provenance['data_mount']=dict(mount)
    atomic_json(path,provenance)


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
        if self.s['phase']=='final':
            # Fails before any external command, exactly like the legacy refusal.
            _global_fail('campaign_anchor_missing','phase=final workers need the S3 campaign anchor/chronology '
                         'audit, which is not implemented in this slice; final jobs are never claimed')
        if self.s['phase']=='hpo':
            _views_root(self.s)
        elif self.s['phase'] != 'smoke':
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

    def global_config_for(self,spec):
        """Global hpo/final claim check; returns (config name, cfg, trusted inputs, data mount).

        Everything comes from the pinned checkout (materialized bytes + MANIFEST)
        and the operator-configured host data/views roots, before any claim.
        HPO: the D-view listing is recomputed and becomes the ONLY /data mount.
        Final: refused at the campaign gate (S3) after all S1 checks.
        """
        if spec.get('phase')!=self.s.get('phase'):
            _global_fail('global_config_invalid','job phase differs from this worker phase')
        trusted=load_global_trusted_inputs(self.code,self.s.get('data'),spec)
        cfg=trusted.cfg
        if cfg['augmentation']['ops']:
            environment=getattr(self,'environment',{})
            if environment.get('opencv')!='4.12.0' or environment.get('opencv_distribution')!='4.12.0.88':
                raise ValueError('candidate-opencv-v2 requires cv2=4.12.0 and opencv-python-headless=4.12.0.88 before claim')
        self.weights_cache_for(cfg,spec) # streaming hash before a worker may claim
        if cfg['phase']=='hpo':
            mount=verify_d_view(self.s,trusted)
        else:
            self.campaign_gate(spec,trusted)
            mount=dict(kind='data_root',source=str(self.data),destination='/data',readonly=True)
        return spec['config'],cfg,trusted,mount

    def campaign_gate(self,spec,trusted):
        """Spec 6.5: a final job needs anchors/<ds>/campaign-<sha>.json audited in the queue HEAD (S3)."""
        _global_fail('campaign_anchor_missing','final jobs require the S3 campaign anchor audit, '
                     'which is not implemented in this slice')

    def config_for(self,spec):
        if is_global_spec(spec):
            name,cfg,_,_=self.global_config_for(spec)
            return name,cfg
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
        if cfg['phase'] in GLOBAL_RUNTIME_PHASES:
            _global_fail('global_config_invalid','global configs are executable only from configs/global/<dataset>/<phase>/')
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
        mount=None
        if is_global_spec(spec):
            # Revalidate config, inputs, cache and D-view listing immediately before create.
            config,cfg,_,mount=self.global_config_for(spec)
            record_data_mount(out,mount)
            data_source=mount['source']
        else:
            config,cfg=self.config_for(spec) # revalidate cache immediately before create
            data_source=self.data
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
              '--mount',f'type=bind,src={data_source},dst=/data,readonly',
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


def _stopped_checkpoint(out,spec,code):
    """Verify host provenance and archive bytes, without loading pickle or torch.

    This deliberately does not assert tensor/model/optimizer resumability. A CPU
    checkpoint load in the pinned training environment is a separate acceptance
    check before an operator explicitly resumes the attempt.
    """
    if is_global_spec(spec):
        return _stopped_global_checkpoint(out,spec,code)
    from .contracts import load_config,canonical_hash
    out=Path(out)
    source=(Path(code)/spec['config']).resolve()
    if not source.is_relative_to(Path(code).resolve()) or digest(source)!=spec['config_sha256']:
        raise ValueError('checkpoint_config_identity')
    cfg=load_config(out/'config.json')
    if canonical_hash(cfg)!=canonical_hash(load_config(source)):
        raise ValueError('checkpoint_config_identity')
    splits=json.loads((out/'splits.json').read_text(encoding='utf-8'))
    if (splits.get('sha256')!=spec['split_hash'] or
            splits['sha256']!=canonical_hash({k:v for k,v in splits.items() if k!='sha256'})):
        raise ValueError('checkpoint_split_identity')
    progress=json.loads((out/'progress.json').read_text(encoding='utf-8'))
    if (progress.get('config_sha256')!=canonical_hash(cfg) or
            progress.get('splits_sha256')!=splits['sha256'] or
            progress.get('checkpoint')!='checkpoint.pt'):
        raise ValueError('checkpoint_progress_identity')
    epoch,epochs,samples=(progress.get(k) for k in ('epoch','epochs','samples_processed'))
    if (type(epoch) is not int or type(epochs) is not int or type(samples) is not int or
            epochs!=cfg['training']['epochs'] or not 0<=epoch<=epochs or samples<0):
        raise ValueError('checkpoint_progress_invalid')
    checkpoint=out/'checkpoint.pt'
    if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size==0:
        raise ValueError('checkpoint_missing')
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            names=archive.namelist()
            if (len(names)!=len(set(names)) or
                    sum(name.endswith('/data.pkl') for name in names)!=1 or
                    not any('/data/' in name for name in names) or archive.testzip() is not None):
                raise ValueError('checkpoint_archive_invalid')
    except (zipfile.BadZipFile,RuntimeError) as exc:
        raise ValueError('checkpoint_archive_invalid') from exc
    return (dict(epoch=epoch,epochs=epochs,samples_processed=samples),
            dict(sha256=digest(checkpoint),size_bytes=checkpoint.stat().st_size,
                 validation='zip-crc-and-job-artifact-identity',
                 config_sha256=canonical_hash(cfg),splits_sha256=splits['sha256']))


def _stopped_global_checkpoint(out,spec,code):
    """Global analogue of _stopped_checkpoint: identity by materialized bytes, no pickle/torch.

    config_sha256 is the hash of the pinned bytes (not canonical_hash(cfg)); an
    attempt that recorded a non-null trainer_override is never resumable.
    """
    from .contracts import canonical_hash
    out=Path(out)
    try:
        _,config_bytes,config_sha,cfg=_global_config_bytes(code,spec)
    except ValueError as exc:
        raise ValueError('checkpoint_config_identity') from exc
    if (out/'config.json').read_bytes()!=config_bytes:
        raise ValueError('checkpoint_config_identity')
    splits=json.loads((out/'splits.json').read_text(encoding='utf-8'))
    if (splits.get('sha256')!=spec['split_hash'] or
            splits['sha256']!=canonical_hash({k:v for k,v in splits.items() if k!='sha256'})):
        raise ValueError('checkpoint_split_identity')
    progress=json.loads((out/'progress.json').read_text(encoding='utf-8'))
    environment=json.loads((out/'environment.json').read_text(encoding='utf-8'))
    if 'trainer_override' not in environment or environment['trainer_override'] is not None:
        raise ValueError('checkpoint_trainer_override_forbidden')
    for record in (progress,environment):
        if record.get('config_sha256')!=config_sha or record.get('splits_sha256')!=splits['sha256']:
            raise ValueError('checkpoint_progress_identity')
    if progress.get('checkpoint')!='checkpoint.pt' or progress.get('phase')!=cfg['phase']:
        raise ValueError('checkpoint_progress_identity')
    epoch,epochs,samples=(progress.get(k) for k in ('epoch','epochs','samples_processed'))
    if (type(epoch) is not int or type(epochs) is not int or type(samples) is not int or
            epochs!=cfg['training']['epochs'] or not 0<=epoch<=epochs or samples<0):
        raise ValueError('checkpoint_progress_invalid')
    checkpoint=out/'checkpoint.pt'
    if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size==0:
        raise ValueError('checkpoint_missing')
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            names=archive.namelist()
            if (len(names)!=len(set(names)) or
                    sum(name.endswith('/data.pkl') for name in names)!=1 or
                    not any('/data/' in name for name in names) or archive.testzip() is not None):
                raise ValueError('checkpoint_archive_invalid')
    except (zipfile.BadZipFile,RuntimeError) as exc:
        raise ValueError('checkpoint_archive_invalid') from exc
    return (dict(epoch=epoch,epochs=epochs,samples_processed=samples),
            dict(sha256=digest(checkpoint),size_bytes=checkpoint.stat().st_size,
                 validation='zip-crc-and-global-job-artifact-identity',
                 config_sha256=config_sha,splits_sha256=splits['sha256']))


def reconcile_stopped_attempt(settings,active_path,*,reason,runtime=None,queue=None):
    """Reconcile an owned, stopped attempt without training or a CUDA probe.

    The existing journal, token, versions and durable event IDs remain the only
    authority. Queue conflicts propagate. A Git outage returns pending=True;
    calling this function again replays the same event, including lost replies.
    The caller must ensure the worker is stopped before invoking reconciliation.
    """
    if not isinstance(reason,str) or not reason.strip() or len(reason)>500 or '\n' in reason or '\r' in reason:
        raise ValueError('Reconciliation requires a short public audit reason')
    active=Path(active_path).resolve()
    root=Path(settings['runs']).resolve()
    if not active.is_relative_to(root) or active.name!='active.json':
        raise ValueError('Active journal is outside configured runs')
    local=json.loads(active.read_text(encoding='utf-8'))
    initial=local['record']; out=Path(local['output']).resolve()
    if out!=root/initial['job_id']/initial['attempt_id']:
        raise ValueError('Active output does not belong to journal attempt')
    owner=initial.get('owner') or {}
    if (owner.get('server')!=settings['server'] or str(owner.get('gpu'))!=str(settings['gpu']) or
            not owner.get('gpu_uuid') or active.parent.name!='worker-'+owner['gpu_uuid']):
        raise StaleAttempt('Active journal does not belong to configured worker')
    if local.get('stage') not in ('launched','interrupted','reconciliation-pending','failed','succeeded','cancelled'):
        raise ValueError('Only previously launched attempts may be reconciled')
    provenance=json.loads((out/'runtime.json').read_text(encoding='utf-8'))
    for key,wanted in dict(job_id=initial['job_id'],attempt_id=initial['attempt_id'],owner=owner,
                           code_commit=initial['spec']['code_commit'],image_id=local['image']).items():
        if provenance.get(key)!=wanted:
            raise ValueError('Runtime provenance differs from active attempt')
    spool=Spool(out/'events-pending.json',initial)
    record=spool.record
    if record.get('owner')!=owner or record.get('spec')!=initial['spec']:
        raise StaleAttempt('Spool ownership or immutable spec differs from journal')
    runtime=runtime if runtime is not None else DockerRuntime(settings)
    container=runtime.inspect(local['container'])
    if container is None:
        raise ValueError('Reconciliation requires the original stopped container')
    state=container.get('State',{})
    if state.get('Running') is not False or state.get('Status') not in ('exited','dead'):
        raise ValueError('Reconciliation requires a confirmed stopped, previously executed container')
    config=container.get('Config',{})
    env=config.get('Env') or []
    mounts=container.get('Mounts') or []
    if (container.get('Name','').lstrip('/')!=local['container'] or container.get('Image')!=local['image'] or
            (config.get('Labels') or {}).get('org.augment-atlas.attempt')!=local['container'] or
            'ATLAS_CODE_COMMIT='+record['spec']['code_commit'] not in env or
            'ATLAS_IMAGE_ID='+local['image'] not in env or
            not any(m.get('Destination')=='/output' and m.get('Type')=='bind' and
                    Path(m.get('Source','')).resolve()==out and m.get('RW') is True for m in mounts)):
        raise ValueError('Stopped container identity or output mount differs from journal')
    if type(state.get('ExitCode')) is not int:
        raise ValueError('Stopped container lacks an exit status')
    terminals={'succeeded','failed','interrupted','cancelled'}
    terminal=next((event for event in spool.events if event['state'] in terminals),None)
    checkpoint=None
    if terminal is None and record['state'] not in terminals:
        numbers={}
        error=None
        completed=False
        try:
            if state['ExitCode']==0 and (out/'completed.json').exists():
                numbers=validate_completion(settings,out,record['spec'])
                completed=True
            else:
                numbers,checkpoint=_stopped_checkpoint(out,record['spec'],settings['code'])
        except (ValueError,OSError,KeyError,TypeError,AttributeError) as exc:
            error=str(exc) if str(exc).startswith('checkpoint_') else 'checkpoint_artifacts_invalid'
        interrupted_exit=state['ExitCode'] in (75,130,137,143)
        training_failed=not completed and ((out/'failed.json').exists() or not interrupted_exit)
        target=('succeeded' if completed else 'interrupted') if error is None and state.get('OOMKilled') is not True and not training_failed else 'failed'
        data=dict(reason=reason,exit_code=state['ExitCode'],oom_killed=state.get('OOMKilled') is True,
                  reconciliation=True,**numbers)
        if checkpoint is not None:data['checkpoint']=checkpoint
        if error is not None:data['validation_error']=error
        if training_failed:data['training_failed']=True
        spool.add(target,**data)
        terminal=spool.events[-1]
    target=terminal['state'] if terminal is not None else record['state']
    if terminal is not None:checkpoint=terminal.get('checkpoint')
    elif record['state'] in terminals:
        checkpoint=next((event.get('checkpoint') for event in reversed(record.get('history',[]))
                         if event.get('attempt_id')==record['attempt_id'] and event.get('state')==target),None)
    local.update(record=spool.record,stage='reconciliation-pending',reconciliation_state=target)
    atomic_json(active,local)
    confirmed=False
    # Persist the decision BEFORE any Git work (including GitQueue construction).
    # An outer operator timeout can therefore never lose the terminal event.
    try:
        queue=queue if queue is not None else GitQueue(settings['queue_checkout'],settings['remote'],settings.get('branch','queue'))
        current=next((item for item in queue.list_jobs() if item['job_id']==record['job_id']),None)
        if current is None:
            raise QueueConflict('Journal job is absent from queue')
        if any(current.get(k)!=record.get(k) for k in ('attempt_id','token','owner','spec')):
            raise StaleAttempt('Stopped attempt is no longer the queue owner')
        if not spool.events and current['version']!=record['version']:
            raise QueueConflict('Queue version differs from journal; inspect before reconciliation')
        spool.flush(queue)
        confirmed=True
    except QueueConflict:
        raise
    except QueueError:
        pass
    local.update(record=spool.record,stage=target if confirmed else 'reconciliation-pending')
    atomic_json(active,local)
    return dict(record=spool.record,state=target,pending=not confirmed,pending_events=len(spool.events),
                remote_confirmed=confirmed,active=str(active),checkpoint=checkpoint)


@dataclass(frozen=True)
class TrustedEvaluationInputs:
    """Host-derived expectations for ONE clinical opt-in job.

    Built only from the pinned code checkout config (checked against the job's
    config_sha256) and the operator-configured host data directory. Nothing in
    it comes from the container-writable attempt output: the complete manifest
    records provide the ID->group->truth mapping that splits.json lacks.
    """
    cfg: dict
    config_file_sha256: str
    splits: dict
    records: list


def load_trusted_evaluation_inputs(config_path,data,*,expected_config_sha256):
    """Load trusted inputs from explicit paths; never consults ATLAS_DATA_ROOT or /data."""
    from .contracts import load_config
    from .data import load_records,prepare_splits
    path=Path(config_path).resolve()
    if not path.is_file() or digest(path)!=expected_config_sha256:
        raise ValueError('Trusted clinical config checksum mismatch')
    cfg=load_config(path)
    if cfg['phase'] in GLOBAL_RUNTIME_PHASES:
        _global_fail('global_config_invalid','global configs use load_global_trusted_inputs, never the clinical opt-in')
    if 'evaluation' not in cfg:
        raise ValueError('Trusted inputs are only defined for clinical evaluation opt-in')
    if not isinstance(data,(str,os.PathLike)) or not str(data).strip():
        raise ValueError('Clinical evaluation requires settings.data; no default data root is assumed')
    root=Path(data).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Clinical evaluation data root must be an existing host directory')
    records=load_records(cfg,check_files=True,root=root)
    with tempfile.TemporaryDirectory(prefix='atlas-clinical-trust-') as directory:
        splits=prepare_splits(cfg,records,directory,root=root)
    return TrustedEvaluationInputs(cfg=copy.deepcopy(cfg),config_file_sha256=expected_config_sha256,
                                   splits=copy.deepcopy(splits),records=copy.deepcopy(records))


def trusted_evaluation_inputs(settings,spec):
    """Resolve the job config inside settings.code; None means a legacy config.

    The opt-in decision comes from the pinned checkout, never from the attempt's
    own config.json. Specs without a config path cannot have been claimed by
    config_for; they keep the legacy call and validate_artifacts still rejects an
    output config that declares evaluation without trusted inputs.
    """
    if not isinstance(spec,dict) or 'config' not in spec:
        return None
    code=Path(settings['code']).resolve()
    name=spec['config']
    if not isinstance(name,str) or not name.startswith('configs/smoke/'):
        raise ValueError('Job requires a config under configs/smoke/')
    path=(code/name).resolve()
    if not path.is_relative_to(code) or not path.is_file():
        raise ValueError('Invalid training config path')
    # The opt-in decision must be made on the exact bytes the job pinned: a
    # checkout moved after claim fails closed for legacy and clinical jobs alike.
    if digest(path)!=spec.get('config_sha256'):
        raise ValueError('Pinned job config checksum mismatch; settings.code differs from the job')
    if 'evaluation' not in json.loads(path.read_text(encoding='utf-8')):
        return None
    return load_trusted_evaluation_inputs(path,settings.get('data'),expected_config_sha256=spec.get('config_sha256'))


def validate_completion(settings,out,spec):
    """Worker/reconciliation entry: legacy jobs keep the exact legacy call.

    Parser failures (deeply nested JSON, oversized CSV fields) are validation
    failures of this attempt, never exceptions that escape the worker loop.
    """
    import csv
    try:
        if is_global_spec(spec):
            try:
                return validate_global_completion(settings,out,spec)
            except (TypeError,AttributeError,KeyError,IndexError) as exc:
                raise ValueError('global_artifacts_invalid: malformed global completion artifacts') from exc
        try:
            trusted=trusted_evaluation_inputs(settings,spec)
        except (TypeError,AttributeError,KeyError) as exc:
            raise ValueError('Malformed trusted clinical evaluation inputs') from exc
        if trusted is None:
            return validate_artifacts(out,spec)
        return validate_artifacts(out,spec,trusted=trusted)
    except (RecursionError,csv.Error) as exc:
        raise ValueError('Completion artifacts or host manifest cannot be parsed') from exc


def validate_artifacts(out,spec=None,*,trusted=None):
    """Validate a completed attempt. ``trusted`` is required iff config opts in.

    Without ``evaluation`` in the attempt config and with trusted=None this is
    the unchanged legacy eight-artifact validation.
    """
    try:
        return _validate_artifacts(out,spec,trusted)
    except (TypeError,AttributeError,KeyError) as exc:
        raise ValueError('Malformed completion artifacts') from exc


def _validate_clinical(out,cfg,splits,results,metrics,environment,files,spec,trusted):
    """Revalidate predictions.json against host-trusted config/split/manifest."""
    from .contracts import canonical_hash,recipe_identity,required_artifacts
    from .evaluation import (PREDICTION_ARTIFACT,evaluation_partition,partition_accuracy_bounds,
                             selected_epoch,validate_prediction_artifact)
    if not isinstance(trusted,TrustedEvaluationInputs) or 'evaluation' not in trusted.cfg:
        raise ValueError('Clinical evaluation opt-in requires trusted host config/manifest inputs')
    if 'evaluation' not in cfg or canonical_hash(cfg)!=canonical_hash(trusted.cfg):
        raise ValueError('Attempt config differs from the trusted clinical job config')
    if canonical_hash(splits)!=canonical_hash(trusted.splits):
        raise ValueError('Attempt splits differ from host-derived clinical splits')
    if not set(required_artifacts(cfg)).issubset(files):
        raise ValueError('Completion checksums must cover the clinical prediction artifact')
    if (not isinstance(spec,dict) or spec.get('config_sha256')!=trusted.config_file_sha256 or
            spec.get('recipe_hash')!=recipe_identity(trusted.cfg) or
            spec.get('split_hash')!=trusted.splits['sha256'] or
            not isinstance(spec.get('code_commit'),str) or not spec['code_commit']):
        raise ValueError('Clinical job spec config/recipe/split/code identity mismatch')
    if environment.get('code_commit')!=spec['code_commit']:
        raise ValueError('Clinical environment code commit differs from job')
    # Top-level results carry exactly the trainer's keys: no extra claims
    # (e.g. scientifically_eligible) can ride along outside clinical_evaluation.
    allowed={'schema_version','phase','scientific_status','best_epoch','selection_loss','config_sha256',
             'splits_sha256','inbreast_previously_examined_in_pilot','clinical_evaluation',
             'evaluation_scope',evaluation_partition(cfg)}
    pilot=results.get('inbreast_previously_examined_in_pilot',cfg['dataset']['adapter']=='inbreast_csv')
    if (not set(results)<=allowed or results.get('phase')!='smoke' or
            results.get('scientific_status')!='provisional' or
            pilot is not (cfg['dataset']['adapter']=='inbreast_csv')):
        raise ValueError('Clinical results.json declares keys or status outside the trainer contract')
    best_epoch,best_loss=selected_epoch(metrics)
    if (type(results.get('best_epoch')) is not int or results['best_epoch']!=best_epoch or
            results.get('selection_loss')!=best_loss):
        raise ValueError('Selected epoch differs from the selection-loss journal')
    artifact=json.loads((out/PREDICTION_ARTIFACT).read_text(encoding='utf-8'))
    validate_prediction_artifact(trusted.cfg,trusted.splits,trusted.records,artifact,
                                 results.get('clinical_evaluation'),expected_best_epoch=best_epoch,
                                 environment=environment)
    summary=results[evaluation_partition(cfg)]
    lower,upper=partition_accuracy_bounds(artifact)
    correct=summary['accuracy']*summary['samples']
    if (summary['samples']!=len(artifact['rows']) or abs(correct-round(correct))>1e-6 or
            not lower<=round(correct)<=upper):
        raise ValueError('Partition accuracy is inconsistent with stored probabilities')


def _validate_artifacts(out,spec=None,trusted=None):
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
    if cfg['phase'] in GLOBAL_RUNTIME_PHASES:
        _global_fail('global_config_invalid','global attempts are validated only by validate_completion '
                     'with host-trusted inputs (validate_global_completion)')
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
    if 'evaluation' in cfg or trusted is not None:
        _validate_clinical(out,cfg,splits,results,metrics,environment,files,spec,trusted)
    summary=dict(completed_sha256=digest(p),artifact_count=len(files),epoch=progress['epoch'],epochs=cfg['training']['epochs'])
    samples=progress.get('samples_processed')
    if type(samples) is int and samples>=0:summary['samples_processed']=samples
    return summary


_GLOBAL_METRIC_KEYS={'epoch','utc','train_loss','samples_processed','optimizer_updates','learning_rate'}


def _accuracy_bounds(rows):
    """Correct-count bounds from stored probabilities (ties may go either way)."""
    lower=upper=0
    for row in rows:
        scores=row['scores']; best=max(scores)
        tied=[index for index,value in enumerate(scores) if value==best]
        if row['target'] in tied:
            upper+=1
            lower+=len(tied)==1
    return lower,upper


def validate_global_completion(settings,out,spec):
    """Completed global attempt vs host-trusted inputs (9 artifacts, spec 6.6/7.3).

    Trust comes only from the pinned checkout and settings.data/settings.views.
    The prediction artifact is recomputed by global_evaluation with the host
    records and anchored pools; a non-null trainer_override is always refused.
    """
    trusted=load_global_trusted_inputs(settings['code'],settings.get('data'),spec)
    return _validate_global_artifacts(settings,Path(out),spec,trusted)


def _validate_global_artifacts(settings,out,spec,trusted):
    from . import global_evaluation as ge
    from . import global_jobs as gj
    from .contracts import canonical_hash,required_artifacts

    def bad(message):
        _global_fail('global_artifacts_invalid',message)

    def read(name):
        return json.loads((out/name).read_text(encoding='utf-8'))

    cfg,view=trusted.cfg,trusted.view
    phase,epochs=cfg['phase'],cfg['training']['epochs']
    config_sha,split=trusted.config_sha256,view['sha256']
    p=out/'completed.json'
    if not p.is_file():bad('missing completed.json')
    completed=read('completed.json')
    files=completed.get('artifacts')
    required=set(required_artifacts(cfg))
    if type(files) is not dict or set(files)!=required:
        bad('completion checksums must cover exactly the nine global artifacts')
    for name,wanted in files.items():
        path=out/name
        if (path.is_symlink() or not path.resolve().is_relative_to(out.resolve()) or not path.is_file() or
                path.stat().st_size==0 or digest(path)!=wanted):
            bad('invalid completed artifact')
    environment,results,progress,metrics=(read(n) for n in ('environment.json','results.json','progress.json','metrics.json'))
    for label,document in (('completed',completed),('environment',environment),('results',results)):
        if type(document) is not dict or 'trainer_override' not in document or document['trainer_override'] is not None:
            _global_fail('trainer_override_forbidden',f'{label} records a trainer_override; toy/CPU attempts are never accepted')
    if (out/'config.json').read_bytes()!=trusted.config_bytes:
        _global_fail('config_identity_mismatch','attempt config.json differs from the pinned materialized bytes')
    if canonical_hash(read('splits.json'))!=canonical_hash(view):
        _global_fail('view_hash_mismatch','attempt splits.json differs from the host-derived view')
    if spec.get('split_hash')!=split:
        _global_fail('view_hash_mismatch','job split_hash differs from the host-derived view')
    for document in (completed,progress,results,environment):
        if (type(document) is not dict or document.get('config_sha256')!=config_sha or
                document.get('splits_sha256')!=split or document.get('phase')!=phase):
            bad('completion config/split/phase identity mismatch')
    if (completed.get('status')!='completed' or progress.get('status')!='completed' or
            type(progress.get('epoch')) is not int or progress['epoch']!=epochs or
            type(completed.get('epoch')) is not int or completed['epoch']!=epochs or progress.get('epochs')!=epochs):
        bad('training did not complete every fixed epoch')
    if (type(metrics) is not list or len(metrics)!=epochs or
            any(type(row) is not dict or set(row)!=_GLOBAL_METRIC_KEYS or row['epoch']!=index+1
                for index,row in enumerate(metrics))):
        bad('metrics.json must journal exactly the fixed epochs, without per-epoch evaluation')
    evaluated='test' if phase=='final' else 'selection'
    keys={'schema_version','phase','scientific_status','protocol_version','checkpoint_selection','best_epoch','epochs',
          'config_sha256','splits_sha256','initial_state_sha256','trainer_override',evaluated,f'{evaluated}_evaluations',
          'clinical_evaluation','execution_verified','scientific_eligible'}
    count=results.get(f'{evaluated}_evaluations')
    if (set(results)!=keys or results['scientific_status']!='provisional' or
            results['protocol_version']!=cfg['protocol_version'] or results['checkpoint_selection']!='last_epoch' or
            type(results['best_epoch']) is not int or results['best_epoch']!=epochs or results['epochs']!=epochs or
            results['execution_verified'] is not False or results['scientific_eligible'] is not False or
            type(count) is not int or count<1):
        _global_fail('epoch_selection_forbidden' if results.get('best_epoch')!=epochs else 'global_artifacts_invalid',
                     f'results.json outside the global {phase} trainer contract')
    bound=cfg['global']
    if (environment.get('protocol_version')!=cfg['protocol_version'] or
            environment.get('code_commit')!=spec.get('code_commit') or
            environment.get('weights')!=cfg['model']['weights'] or
            environment.get('weights_sha256')!=cfg['model']['weights_sha256'] or
            environment.get('device')!=cfg['training']['device'] or environment.get('workers')!=cfg['training']['workers'] or
            canonical_hash(environment.get('trainer_constants'))!=canonical_hash(gj.TRAINER_CONSTANTS_V1) or
            environment.get('initial_state_sha256')!=results['initial_state_sha256'] or
            environment.get('global')!={k:bound[k] for k in ('pools_sha256','hpo_view_sha256','plan_sha256','candidate_id',
                                                            'freeze_kind','freeze_sha256','fold')} or
            environment.get('evaluated_partition')!=ge.PARTITIONS[phase]):
        bad('environment.json differs from the job/config identity')
    provenance=read('runtime.json') if (out/'runtime.json').is_file() else {}
    mount=provenance.get('data_mount') if type(provenance) is dict else None
    if phase=='hpo':
        expected=dict(kind='d_view',source=str(expected_d_view(settings,bound['pools_sha256']).resolve()),
                      destination='/data',readonly=True,pools_sha256=bound['pools_sha256'],
                      hpo_view_sha256=view['sha256'],view_files_sha256=view['view_files_sha256'])
        integrity=environment.get('dataset_integrity')
        if (mount!=expected or type(integrity) is not dict or integrity.get('source')!='hpo_view' or
                integrity.get('view_files_sha256')!=view['view_files_sha256']):
            _global_fail('hpo_isolation_unproven','runtime.json does not record the verified D-view as the only /data mount')
        train_images=len(view['lists']['train'])
    else:
        data=Path(settings['data']).expanduser().resolve()
        if mount!=dict(kind='data_root',source=str(data),destination='/data',readonly=True):
            bad('runtime.json does not record the read-only data root mount of the final job')
        train_images=len(view['train'])
    artifact=read(ge.ARTIFACT_NAME)
    summary=ge.validate_global_predictions(cfg,view,trusted.records,artifact,pools=trusted.pools)
    declared={k:summary[k] for k in ('partition','samples','primary_metric','primary_value','primary_reason','content_sha256')}
    if canonical_hash(results['clinical_evaluation'])!=canonical_hash({'artifact':ge.ARTIFACT_NAME,**declared}):
        _global_fail('metric_recompute_mismatch','results.clinical_evaluation differs from the recomputed artifact')
    rows=artifact['rows']
    measured=results[evaluated]
    if (type(measured) is not dict or set(measured)!={'loss','accuracy','samples'} or
            type(measured['samples']) is not int or measured['samples']!=len(rows) or
            type(measured['loss']) not in (int,float) or not math.isfinite(measured['loss']) or measured['loss']<0 or
            type(measured['accuracy']) not in (int,float) or not 0<=measured['accuracy']<=1):
        bad(f'results.{evaluated} is not a finite summary of the evaluated partition')
    lower,upper=_accuracy_bounds(rows)
    correct=measured['accuracy']*measured['samples']
    if abs(correct-round(correct))>1e-6 or not lower<=round(correct)<=upper:
        _global_fail('metric_recompute_mismatch','evaluated accuracy is inconsistent with stored probabilities')
    if environment.get('evaluated_images')!=len(rows) or environment.get('train_images')!=train_images:
        bad('environment image counts differ from the host-derived view')
    result=dict(completed_sha256=digest(p),artifact_count=len(files),epoch=progress['epoch'],epochs=epochs)
    samples=progress.get('samples_processed')
    if type(samples) is int and samples>=0:result['samples_processed']=samples
    return result


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
                stop_requested=False
                while True:
                    state=runtime.inspect(name)
                    if state is None: raise RuntimeError('Active container missing')
                    if stopping[0] and state['State']['Running'] and not stop_requested:
                        state=request_training_stop(runtime,state)
                        stop_requested=True
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
                            try:temperature=float(row['gpu']['temperature.gpu'])
                            except (ValueError,TypeError):temperature=None
                            if temperature is not None and temperature>=settings.get('max_temperature_c',85):
                                stopping[0]=True
                                if not stop_requested:
                                    state=request_training_stop(runtime,state)
                                    stop_requested=True
                    if now-last_progress>=settings.get('progress_seconds',3600) and state['State']['Running']:
                        spool.add('running',**numbers); last_progress=now
                    try:
                        spool.flush(queue); local['record']=spool.record; atomic_json(active,local)
                    except QueueConflict as conflict:
                        if state['State']['Running'] and not stop_requested:
                            try:request_training_stop(runtime,state)
                            except Exception as stop_error:raise conflict from stop_error
                        raise
                    except QueueError: pass # running child and checkpoints survive network failure
                    if not state['State']['Running']:break
                    time.sleep(settings.get('monitor_seconds',2))
                (out/'docker.log').write_text(command(['docker','logs',name]),encoding='utf-8')
                # Recovery after an already-spooled terminal event must not append another.
                if not any(e['state'] in ('succeeded','failed','interrupted') for e in spool.events) and spool.record['state'] not in ('succeeded','failed','interrupted'):
                    try:
                        if state['State']['ExitCode']!=0:raise ValueError('Nonzero training exit')
                        summary=validate_completion(settings,out,record['spec'])
                        spool.add('succeeded',**summary)
                    except (ValueError,OSError,KeyError):
                        interrupted=((out/'checkpoint.pt').exists() and not (out/'failed.json').exists()
                                     and state['State'].get('OOMKilled') is not True
                                     and state['State']['ExitCode'] in (75,130,137,143)
                                     and (stopping[0] or progress.get('status')=='interrupted'))
                        spool.add('interrupted' if interrupted else 'failed',exit_code=state['State']['ExitCode'],**numbers)
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
