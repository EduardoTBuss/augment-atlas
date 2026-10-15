"""Best-effort unprivileged local telemetry. No patient data is read."""
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

_CPU_PREVIOUS = None


def _text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def sample(context, gpu, root, optional=True):
    global _CPU_PREVIOUS
    start = time.monotonic()
    row = dict(context, utc=dt.datetime.now(dt.timezone.utc).isoformat(), monotonic_s=start,
               gpu=None, cpu=None, memory=None, disk=None, network=None, uptime_s=None,
               disk_io=None, accessible_errors=None)
    mem = _text('/proc/meminfo')
    if mem:
        row['memory'] = {k: int(v.split()[0]) * 1024 for k,v in
                         (line.split(':',1) for line in mem.splitlines()) if k in
                         ('MemTotal','MemAvailable','SwapTotal','SwapFree')}
    up = _text('/proc/uptime')
    row['uptime_s'] = float(up.split()[0]) if up else None
    disk = shutil.disk_usage(root)
    row['disk'] = dict(total_bytes=disk.total, free_bytes=disk.free)
    row['cpu'] = dict(load_average=list(os.getloadavg()) if hasattr(os,'getloadavg') else None,
                      counters=_text('/proc/stat'), frequencies_khz=None, temperature_c=None,
                      energy_uj=None,utilization_percent=None)
    if row['cpu']['counters']:
        counts=[int(v) for v in row['cpu']['counters'].splitlines()[0].split()[1:9]]
        current=(sum(counts),counts[3]+counts[4])
        if _CPU_PREVIOUS and current[0]>_CPU_PREVIOUS[0]:
            row['cpu']['utilization_percent']=100*(1-(current[1]-_CPU_PREVIOUS[1])/(current[0]-_CPU_PREVIOUS[0]))
        _CPU_PREVIOUS=current
    row['disk_io'] = _text('/proc/diskstats')
    row['network'] = _text('/proc/net/dev')
    if optional:
        fields = 'uuid,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,power.limit'
        try:
            r = subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu='+fields,
                                '--format=csv,noheader,nounits'], capture_output=True,text=True,timeout=4)
            if r.returncode == 0:
                vals = next(csv.reader([r.stdout.strip()]))
                values=[vals[0].strip()]
                for v in vals[1:]:
                    try: values.append(float(v.strip()))
                    except ValueError: values.append(None)
                row['gpu'] = dict(zip(fields.split(','), values))
            throttle = subprocess.run(['nvidia-smi','-i',str(gpu),'-q','-d','PERFORMANCE'],capture_output=True,text=True,timeout=4)
            row['gpu_performance'] = throttle.stdout if throttle.returncode == 0 else None
        except (OSError,subprocess.TimeoutExpired,StopIteration):
            pass
        cpus = list(Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'))
        row['cpu']['frequencies_khz'] = [int(v) for p in cpus if (v:=_text(p)) and v.isdigit()] or None
        temps = []
        for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
            v = _text(p)
            if v and v.lstrip('-').isdigit():
                temps.append(dict(domain=_text(p.parent/'type'),celsius=int(v)/1000))
        row['cpu']['temperature_c'] = temps or None
        energies=[]
        for p in Path('/sys/class/powercap').glob('intel-rapl*/energy_uj'):
            v=_text(p)
            if v and v.isdigit():
                energies.append(dict(domain=_text(p.parent/'name'),energy_uj=int(v),max_range_uj=_text(p.parent/'max_energy_range_uj')))
        row['cpu']['energy_uj']=energies or None
    row['collection_s'] = time.monotonic()-start
    return row


def append_sample(path, context, gpu, root, optional=True):
    row = sample(context,gpu,root,optional)
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n')
        f.flush()
    return row
