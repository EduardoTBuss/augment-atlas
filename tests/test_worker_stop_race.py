"""Regression for Docker exit between inspection and a requested TERM.

Docker, telemetry and transport are mocked; durable spool/worker flow are real.
"""
import contextlib
import copy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch

from augment_atlas import runtime
from augment_atlas.queue import QueueConflict
from augment_atlas.storage import atomic_json

CID='1'*64


def container(running=True,**changes):
    value=dict(Id=CID,State=dict(Running=running,Status='running' if running else 'exited',ExitCode=0 if running else 143,OOMKilled=False))
    value.update(changes)
    return value


class StopRaceTests(unittest.TestCase):
    def test_failed_or_timed_out_signal_accepts_only_the_same_exited_container(self):
        for error in (RuntimeError('already stopped'),subprocess.TimeoutExpired(['docker'],1)):
            for status in ('exited','dead'):
                with self.subTest(error=type(error).__name__,status=status):
                    stopped=container(False);stopped['State']['Status']=status
                    host=Mock();host.inspect.return_value=stopped
                    with patch.object(runtime,'command',side_effect=error) as command:
                        self.assertEqual(stopped,runtime.request_training_stop(host,container()))
                    command.assert_called_once_with(['docker','kill','--signal','TERM',CID])
                    host.inspect.assert_called_once_with(CID)

    def test_unconfirmed_stop_identity_or_daemon_failure_is_not_suppressed(self):
        cases=(None,container(),container(False,Id='2'*64),dict(Id=CID,State=dict(Running=False,Status='created')))
        for observed in cases:
            with self.subTest(observed=observed):
                host=Mock();host.inspect.return_value=observed
                with patch.object(runtime,'command',side_effect=RuntimeError('signal failed')):
                    with self.assertRaisesRegex(RuntimeError,'signal failed'):runtime.request_training_stop(host,container())
        host=Mock();host.inspect.side_effect=RuntimeError('daemon unavailable')
        with patch.object(runtime,'command',side_effect=RuntimeError('signal failed')):
            with self.assertRaisesRegex(RuntimeError,'daemon unavailable'):runtime.request_training_stop(host,container())

    def test_missing_inspected_id_does_not_signal_a_reused_name(self):
        with patch.object(runtime,'command') as command:
            with self.assertRaisesRegex(RuntimeError,'container ID'):runtime.request_training_stop(Mock(),{'Name':'reused-name'})
            command.assert_not_called()

    def exercise_worker(self,*,signal_error=True,temperature_stop=False,oom=False,trainer_failure=False,conflict=False,operator_signal=True,stop_confirmed=True,reinspect_error=None):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);runs=root/'runs';out=runs/('a'*64)/'attempt-fixture';out.mkdir(parents=True)
            active=runs/'worker-GPU-fixture'/'active.json'
            record=dict(job_id='a'*64,attempt_id='attempt-fixture',token='test-capability',state='running',version=3,spec=dict(code_commit='b'*40,config_sha256='c'*64))
            atomic_json(active,dict(record=record,output=str(out),container='atlas-test',image='sha256:test',stage='launched'))
            (out/'checkpoint.pt').write_bytes(b'mock checkpoint; this test does not assert tensor resumability')
            atomic_json(out/'progress.json',dict(status='interrupted',epoch=4,epochs=10000,samples_processed=2464))
            if trainer_failure:atomic_json(out/'failed.json',dict(status='failed',message='private fixture'))
            host=Mock();host.commit='b'*40;host.image='sha256:test';host.gpu_uuid='GPU-fixture'
            handlers={};observations=0;stopped=container(False);stopped['State']['OOMKilled']=oom
            def install(number,handler):
                previous=handlers.get(number);handlers[number]=handler;return previous
            def inspect(_name):
                nonlocal observations
                observations+=1
                if observations==1 and operator_signal and not temperature_stop:handlers[signal.SIGTERM](signal.SIGTERM,None)
                if observations>=3 and reinspect_error is not None:raise reinspect_error
                if signal_error:return container() if observations<=2 or not stop_confirmed else copy.deepcopy(stopped)
                return container() if observations<=4 else copy.deepcopy(stopped)
            host.inspect.side_effect=inspect
            queue=Mock();updates=[]
            def acknowledge(_job,_attempt,_token,version,event):
                if conflict:raise QueueConflict('owner version changed')
                updates.append(copy.deepcopy(event));return {**record,'version':version+1,'state':event['state']}
            queue.update.side_effect=acknowledge
            kills=[]
            def command(args):
                if args[:2]==['docker','kill']:
                    kills.append(args)
                    if signal_error:raise RuntimeError('container already stopped')
                return ''
            settings=dict(code=str(root),data=str(root),runs=str(runs),queue_checkout=str(root/'queue'),remote='mock',phase='smoke',server='test',gpu='0',telemetry_seconds=0,monitor_seconds=0)
            sample={'gpu':{'temperature.gpu':'90'}} if temperature_stop else {}
            with patch.object(runtime,'DockerRuntime',return_value=host),patch.object(runtime,'GitQueue',return_value=queue),patch.object(runtime,'gpu_lock',return_value=contextlib.nullcontext()),patch.object(runtime.signal,'signal',side_effect=install),patch.object(runtime,'append_sample',return_value=sample),patch.object(runtime,'command',side_effect=command):
                if conflict:
                    # Force a real flush path during the running iteration.
                    spool=runtime.Spool(out/'events-pending.json',record);spool.add('running',epoch=3)
                    with self.assertRaises(QueueConflict) as failure:runtime.worker(settings,once=True)
                    if not stop_confirmed:self.assertIsInstance(failure.exception.__cause__,RuntimeError)
                elif reinspect_error is not None:
                    with self.assertRaises(type(reinspect_error)):runtime.worker(settings,once=True)
                else:runtime.worker(settings,once=True)
            self.assertEqual([['docker','kill','--signal','TERM',CID]],kills)
            if not conflict and reinspect_error is None:
                saved=runtime.Spool(out/'events-pending.json',record)
                self.assertEqual([],saved.events)
                self.assertEqual('failed' if oom or trainer_failure else 'interrupted',saved.record['state'])
                terminal=updates[-1];self.assertEqual((4,10000,2464),(terminal['epoch'],terminal['epochs'],terminal['samples_processed']))
                if saved.record['state']=='interrupted':self.assertEqual('interrupted',json.loads(active.read_text())['stage'])

    def test_worker_flushes_terminal_when_container_exits_between_inspect_and_term(self):self.exercise_worker()
    def test_worker_requests_term_once_while_waiting_and_telemetry_also_requests_stop(self):self.exercise_worker(signal_error=False,temperature_stop=True)
    def test_oom_or_trainer_failure_never_becomes_interrupted(self):
        self.exercise_worker(oom=True);self.exercise_worker(trainer_failure=True)
    def test_queue_conflict_remains_the_failure_when_stop_is_already_requested(self):self.exercise_worker(conflict=True)
    def test_queue_conflict_requests_stop_without_an_operator_signal_and_keeps_cas_error(self):
        self.exercise_worker(conflict=True,operator_signal=False)
        self.exercise_worker(conflict=True,operator_signal=False,stop_confirmed=False)
    def test_docker_reinspection_value_error_is_not_swallowed_by_temperature_parsing(self):
        self.exercise_worker(temperature_stop=True,operator_signal=False,reinspect_error=ValueError('invalid Docker response'))


if __name__=='__main__':unittest.main()
