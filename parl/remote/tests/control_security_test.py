#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os
import pickle
import secrets
import unittest
from unittest.mock import patch

import zmq

from parl.remote import control_serialization as codec, remote_constants as tags
from parl.remote.master import Master
from parl.remote.message import AllocatedCpu, AllocatedGpu, InitializedJob, InitializedWorker
from parl.remote.monitor import app as monitor_app
from parl.remote.log_server import app as log_app
from parl.remote.security import SecureContext, _curve_keys, get_control_bind_host
from parl.remote.zmq_utils import create_client_socket

MARKER = 'XPARL_SECURITY_REGRESSION_MARKER'


class MaliciousPickle(object):
    def __reduce__(self):
        return eval, ("__import__('os').environ.__setitem__('" + MARKER + "', 'executed')", )


def make_job():
    return InitializedJob('127.0.0.1:9001', '127.0.0.1:9002', '127.0.0.1:9003', '127.0.0.1:9004', 123, 'job-1',
                          '127.0.0.1:9005')


class ControlSerializationTest(unittest.TestCase):
    def test_worker_and_allocated_job_round_trip(self):
        job = make_job()
        job.worker_address = None
        self.assertIsNone(codec.loads(codec.dumps(job), InitializedJob).worker_address)
        job.worker_address = '127.0.0.1:9004'
        job.allocated_cpu = AllocatedCpu(job.worker_address, 1)
        job.allocated_gpu = AllocatedGpu(job.worker_address, '')
        worker = InitializedWorker(job.worker_address, [job], job.allocated_cpu, job.allocated_gpu, 'worker')
        restored = codec.loads(codec.dumps(worker), InitializedWorker)
        self.assertEqual(vars(restored.allocated_cpu), vars(worker.allocated_cpu))
        self.assertEqual(vars(restored.allocated_gpu), vars(worker.allocated_gpu))
        self.assertEqual(restored.initialized_jobs[0].job_id, job.job_id)
        self.assertEqual(restored.initialized_jobs[0].allocated_cpu.n_cpu, 1)

    def test_pickle_payload_never_executes(self):
        with patch.dict(os.environ):
            os.environ.pop(MARKER, None)
            with self.assertRaises(ValueError):
                codec.loads(pickle.dumps(MaliciousPickle()))
            self.assertNotIn(MARKER, os.environ)

    def test_invalid_json_and_record_types(self):
        invalid = [
            b'{}junk', b'{"a":1,"a":2}', b'NaN', b'Infinity', b'[]', b'{"__xparl_type__":"os.system","fields":{}}',
            b'{"__xparl_type__":"AllocatedCpu","fields":{"worker_address":"x","n_cpu":true}}',
            b'{"__xparl_type__":"AllocatedCpu","fields":{"worker_address":"x","n_cpu":-1}}'
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                codec.loads(data)

    def test_limits_and_unknown_python_types(self):
        with self.assertRaises(ValueError):
            codec.loads(b' ' * (codec.MAX_CONTROL_BYTES + 1))
        with self.assertRaises(ValueError):
            codec.loads(b'[' * 40 + b'0' + b']' * 40)
        with self.assertRaises(ValueError):
            codec.dumps(object())
        with self.assertRaises(ValueError):
            codec.dumps({'value': float('inf')})

    def test_status_schemas(self):
        client = {'file_path': 'train.py', 'actor_num': 1, 'time': '0:00:01', 'log_monitor_url': 'http://localhost/'}
        self.assertEqual(codec.loads_status(codec.dumps(client)), client)
        worker = {
            'vacant_memory': 1.0,
            'used_memory': 2.0,
            'vacant_gpu_memory': 0,
            'used_gpu_memory': 0,
            'load_time': '12:00',
            'load_value': 0.1
        }
        self.assertEqual(codec.loads_status(codec.dumps(worker), worker=True), worker)
        for data in ({}, dict(client, actor_num=True), dict(worker, load_time=[])):
            with self.assertRaises(ValueError):
                codec.loads_status(codec.dumps(data), worker='load_value' in data)


class ControlTransportTest(unittest.TestCase):
    def setUp(self):
        self.token = secrets.token_hex(32)
        self.environment = patch.dict(os.environ, {'XPARL_AUTH_TOKEN': self.token, 'XPARL_BIND_HOST': '127.0.0.1'})
        self.environment.start()
        with patch('parl.remote.master.logger.set_dir'), patch(
                'parl.remote.master.get_ip_address', return_value='127.0.0.1'):
            self.master = Master(0)
        self.master.client_socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.endpoint = self.master.client_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        self.address = self.endpoint[len('tcp://'):]
        self.contexts = [self.master.ctx]
        self.client_ctx = SecureContext()
        self.contexts.append(self.client_ctx)
        self.client = create_client_socket(self.client_ctx, self.address)
        self.client.setsockopt(zmq.RCVTIMEO, 2000)

    def tearDown(self):
        self.master.exit()
        for ctx in reversed(self.contexts):
            ctx.destroy(linger=0)
        self.environment.stop()

    def exchange(self, frames):
        self.client.send_multipart(frames)
        self.master._receive_message()
        return self.client.recv_multipart()

    def assert_healthy(self):
        self.assertEqual(self.exchange([tags.CHECK_VERSION_TAG])[0], tags.NORMAL_TAG)

    def test_plaintext_client_cannot_reach_master(self):
        ctx = zmq.Context()
        self.contexts.append(ctx)
        client = ctx.socket(zmq.REQ)
        client.linger = 0
        self.addCleanup(client.close, 0)
        client.setsockopt(zmq.RCVTIMEO, 200)
        client.connect(self.endpoint)
        client.send_multipart([tags.WORKER_INITIALIZED_TAG, pickle.dumps(MaliciousPickle())])
        with self.assertRaises(zmq.Again):
            client.recv_multipart()
        self.assertEqual(self.master.client_socket.poll(100), 0)
        self.assert_healthy()

    def test_wrong_token_cannot_reach_master(self):
        with patch.dict(os.environ, {'XPARL_AUTH_TOKEN': secrets.token_hex(32)}):
            ctx = SecureContext()
        self.contexts.append(ctx)
        client = create_client_socket(ctx, self.address)
        self.addCleanup(client.close, 0)
        client.setsockopt(zmq.RCVTIMEO, 200)
        client.send_multipart([tags.NORMAL_TAG])
        with self.assertRaises(zmq.Again):
            client.recv_multipart()
        self.assertEqual(self.master.client_socket.poll(100), 0)
        self.assert_healthy()

    def test_unknown_curve_client_key_is_rejected(self):
        ctx = zmq.Context()
        self.contexts.append(ctx)
        client = ctx.socket(zmq.REQ)
        client.linger = 0
        self.addCleanup(client.close, 0)
        client.curve_publickey, client.curve_secretkey = zmq.curve_keypair()
        client.curve_serverkey = _curve_keys(self.token, 'server')[0]
        client.setsockopt(zmq.RCVTIMEO, 200)
        client.connect(self.endpoint)
        client.send_multipart([tags.NORMAL_TAG])
        with self.assertRaises(zmq.Again):
            client.recv_multipart()
        self.assertEqual(self.master.client_socket.poll(100), 0)
        self.assert_healthy()

    def test_all_deserialization_branches_reject_pickle_and_recover(self):
        payload = pickle.dumps(MaliciousPickle())
        messages = [[tags.WORKER_INITIALIZED_TAG, payload], [tags.NEW_JOB_TAG, payload, b'old-job'],
                    [tags.CLIENT_STATUS_UPDATE_TAG, b'client', payload],
                    [tags.WORKER_STATUS_UPDATE_TAG, b'worker', payload]]
        with patch.dict(os.environ):
            os.environ.pop(MARKER, None)
            for frames in messages:
                with self.subTest(tag=frames[0]):
                    self.assertEqual(self.exchange(frames), [tags.INVALID_MESSAGE_TAG])
                    self.assertNotIn(MARKER, os.environ)
                    self.assert_healthy()

    def test_bad_frames_and_unknown_workers_do_not_stop_master(self):
        messages = [[b'unknown'], [tags.NEW_JOB_TAG], [tags.CLIENT_CONNECT_TAG, b'\xff', b'host', b'id'],
                    [tags.CLIENT_SUBMIT_TAG, b'client', b'id', b'not-a-number', b'0'],
                    [tags.CLIENT_SUBMIT_TAG, b'client', b'id', b'-1', b'0'],
                    [tags.WORKER_STATUS_UPDATE_TAG, b'unknown', b'{}']]
        for frames in messages:
            with self.subTest(frames=frames):
                self.assertEqual(self.exchange(frames), [tags.INVALID_MESSAGE_TAG])
                self.assert_healthy()

    def test_client_status_and_monitor_json(self):
        status = {'file_path': 'train.py', 'actor_num': 2, 'time': '0:00:01', 'log_monitor_url': 'http://localhost/'}
        self.master.client_hostname['client'] = 'test-client'
        self.assertEqual(
            self.exchange([tags.CLIENT_STATUS_UPDATE_TAG, b'client',
                           codec.dumps(status)]), [tags.NORMAL_TAG])
        response = self.exchange([tags.MONITOR_TAG])
        self.assertEqual(codec.loads(response[1])['clients']['client']['client_hostname'], 'test-client')
        self.assert_healthy()


class ConfigurationAndHttpTest(unittest.TestCase):
    def test_missing_or_short_token_fails_closed(self):
        for token in ('', 'short'):
            with patch.dict(os.environ, {'XPARL_AUTH_TOKEN': token}), self.assertRaises(ValueError):
                SecureContext()

    def test_bind_is_loopback_by_default(self):
        with patch.dict(os.environ):
            os.environ.pop('XPARL_BIND_HOST', None)
            self.assertEqual(get_control_bind_host(), '127.0.0.1')

    def test_monitor_and_logs_require_credentials(self):
        token = secrets.token_hex(32)
        with patch.dict(os.environ, {'XPARL_AUTH_TOKEN': token}):
            credentials = base64.b64encode(('xparl:' + token).encode()).decode()
            for app, route in ((monitor_app, '/cluster'), (log_app, '/get-log')):
                client = app.test_client()
                self.assertEqual(client.get(route).status_code, 401)
                self.assertEqual(
                    client.get(route, headers={
                        'Authorization': 'Basic eHBhcmw6d3Jvbmc='
                    }).status_code, 401)
                # A protected static asset avoids dependence on a running cluster.
                with client.get('/static/favicon.ico', headers={'Authorization': 'Basic ' + credentials}) as response:
                    self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
