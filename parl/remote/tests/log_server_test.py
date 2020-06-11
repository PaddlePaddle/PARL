#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import json
import multiprocessing
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import requests

import parl
from parl.remote.client import disconnect, get_global_client
from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.utils import _IS_WINDOWS


@parl.remote_class
class Actor(object):
    def __init__(self, number=None, arg1=None, arg2=None):
        self.number = number
        self.arg1 = arg1
        self.arg2 = arg2
        print("Init actor...")
        self.init_output = "Init actor...\n"

    def sim_output(self, start, end):
        output = ""
        print(self.number)
        output += str(self.number)
        output += "\n"
        for i in range(start, end):
            print(i)
            output += str(i)
            output += "\n"
        return self.init_output + output


class TestLogServer(unittest.TestCase):
    def tearDown(self):
        disconnect()

    #In windows, multiprocessing.Process cannot run the method of class, but static method is ok.
    @staticmethod
    def _connect_and_create_actor(cluster_addr):
        parl.connect(cluster_addr)
        outputs = []
        for i in range(2):
            actor = Actor(number=i)
            ret = actor.sim_output(1, 4)
            assert ret != ""
            outputs.append(ret)
        return outputs

    def test_log_server(self):
        master_port = 8401
        # start the master
        master = Master(port=master_port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)

        cluster_addr = 'localhost:{}'.format(master_port)
        log_server_port = 8402
        worker = Worker(cluster_addr, 4, log_server_port=log_server_port)
        outputs = self._connect_and_create_actor(cluster_addr)

        # Get status
        status = master._get_status()
        client_jobs = pickle.loads(status).get('client_jobs')
        self.assertIsNotNone(client_jobs)

        # Get job id
        client = get_global_client()
        jobs = client_jobs.get(client.client_id)
        self.assertIsNotNone(jobs)

        for job_id, log_server_addr in jobs.items():
            log_url = "http://{}/get-log".format(log_server_addr)
            # Test response without job_id
            r = requests.get(log_url)
            self.assertEqual(r.status_code, 400)
            # Test normal response
            r = requests.get(log_url, params={'job_id': job_id})
            self.assertEqual(r.status_code, 200)
            log_content = json.loads(r.text).get('log')
            self.assertIsNotNone(log_content)
            log_content = log_content.replace('\r\n', '\n')
            self.assertIn(log_content, outputs)

            # Test download
            download_url = "http://{}/download-log".format(log_server_addr)
            r = requests.get(download_url, params={'job_id': job_id})
            self.assertEqual(r.status_code, 200)
            log_content = r.text.replace('\r\n', '\n')
            self.assertIn(log_content, outputs)

        disconnect()
        worker.exit()
        master.exit()

    def test_monitor_query_log_server(self):
        master_port = 8403
        monitor_port = 8404
        # start the master
        master = Master(port=master_port, monitor_port=monitor_port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        # start the cluster monitor
        monitor_file = __file__.replace(
            os.path.join('tests', 'log_server_test.pyc'), 'monitor.py')
        monitor_file = monitor_file.replace(
            os.path.join('tests', 'log_server_test.py'), 'monitor.py')
        command = [
            sys.executable, monitor_file, "--monitor_port",
            str(monitor_port), "--address", "localhost:" + str(master_port)
        ]
        if _IS_WINDOWS:
            FNULL = tempfile.TemporaryFile()
        else:
            FNULL = open(os.devnull, 'w')
        monitor_proc = subprocess.Popen(
            command,
            stdout=FNULL,
            stderr=subprocess.STDOUT,
        )

        # Start worker
        cluster_addr = 'localhost:{}'.format(master_port)
        log_server_port = 8405
        worker = Worker(cluster_addr, 4, log_server_port=log_server_port)

        # Test monitor API
        outputs = self._connect_and_create_actor(cluster_addr)
        time.sleep(5)  # Wait for the status update
        client = get_global_client()
        jobs_url = "{}/get-jobs?client_id={}".format(master.monitor_url,
                                                     client.client_id)
        r = requests.get(jobs_url)
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.text)
        for job in data:
            log_url = job.get('log_url')
            self.assertIsNotNone(log_url)
            r = requests.get(log_url)
            self.assertEqual(r.status_code, 200)
            log_content = json.loads(r.text).get('log')
            self.assertIsNotNone(log_content)
            log_content = log_content.replace('\r\n', '\n')
            self.assertIn(log_content, outputs)

            # Test download
            download_url = job.get('download_url')
            r = requests.get(download_url)
            self.assertEqual(r.status_code, 200)
            log_content = r.text.replace('\r\n', '\n')
            self.assertIn(log_content, outputs)

        # Clean context
        monitor_proc.kill()
        monitor_proc.wait()
        disconnect()
        worker.exit()
        master.exit()


if __name__ == '__main__':
    unittest.main()
