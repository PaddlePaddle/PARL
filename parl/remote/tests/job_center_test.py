#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import socket
from parl.remote.job_center import JobCenter
from parl.remote.message import InitializedWorker, InitializedJob


class InitializedWorker(object):
    def __init__(self,
                 worker_address,
                 master_heartbeat_address='localhost:8010',
                 initialized_jobs=[],
                 cpu_num=4,
                 hostname=None):
        self.worker_address = worker_address
        self.master_heartbeat_address = master_heartbeat_address
        self.initialized_jobs = initialized_jobs
        self.cpu_num = cpu_num
        self.hostname = hostname


class ImportTest(unittest.TestCase):
    def setUp(self):
        jobs = []
        for i in range(5):
            job = InitializedJob(
                job_address='172.18.182.39:{}'.format(1234 + i),
                worker_heartbeat_address='172.18.182.39:48724',
                client_heartbeat_address='172.18.182.39:48725',
                ping_heartbeat_address='172.18.182.39:48726',
                worker_address='172.18.182.39:8001',
                pid=1234)
            jobs.append(job)

        self.worker1 = InitializedWorker(
            worker_address='172.18.182.39:8001',
            initialized_jobs=jobs,
            hostname=socket.gethostname())

        jobs = []
        for i in range(5):
            job = InitializedJob(
                job_address='172.18.182.39:{}'.format(2234 + i),
                worker_heartbeat_address='172.18.182.39:48724',
                client_heartbeat_address='172.18.182.39:48725',
                ping_heartbeat_address='172.18.182.39:48726',
                worker_address='172.18.182.39:8002',
                pid=1234)
            jobs.append(job)

        self.worker2 = InitializedWorker(
            worker_address='172.18.182.39:8002',
            initialized_jobs=jobs,
            hostname=socket.gethostname())

    def test_add_worker(self):

        job_center = JobCenter('localhost')
        job_center.add_worker(self.worker1)
        job_center.add_worker(self.worker2)

        self.assertEqual(len(job_center.job_pool), 10)
        self.assertEqual(job_center.worker_dict[self.worker1.worker_address],
                         self.worker1)

    def test_drop_worker(self):
        job_center = JobCenter('localhost')
        job_center.add_worker(self.worker1)
        job_center.add_worker(self.worker2)
        job_center.drop_worker(self.worker2.worker_address)

        self.assertEqual(
            set(job_center.job_pool.values()),
            set(self.worker1.initialized_jobs))
        self.assertEqual(len(job_center.worker_dict), 1)

    def test_request_job(self):
        job_center = JobCenter('localhost')
        job_address1 = job_center.request_job()
        self.assertTrue(job_address1 is None)

        job_center.add_worker(self.worker1)
        job_address2 = job_center.request_job()
        self.assertTrue(job_address2 in self.worker1.initialized_jobs)
        self.assertEqual(len(job_center.job_pool), 4)

    def test_reset_job(self):
        job_center = JobCenter('localhost')
        job_center.add_worker(self.worker1)

        job_address = job_center.request_job()
        self.assertTrue(job_address in self.worker1.initialized_jobs)
        self.assertEqual(len(job_center.job_pool), 4)

        job_center.reset_job(job_address)
        self.assertEqual(len(job_center.job_pool), 5)

    def test_update_job(self):

        job_center = JobCenter('localhost')
        job_center.add_worker(self.worker1)
        job_center.add_worker(self.worker2)

        job = InitializedJob(
            job_address='172.18.182.39:{}'.format(9245),
            worker_heartbeat_address='172.18.182.39:48724',
            client_heartbeat_address='172.18.182.39:48725',
            ping_heartbeat_address='172.18.182.39:48726',
            worker_address='172.18.182.39:478727',
            pid=1234)
        job_center.update_job('172.18.182.39:2234', job, '172.18.182.39:8002')

        current_job_address = set([
            job.job_address for job in job_center.
            worker_dict['172.18.182.39:8002'].initialized_jobs
        ])
        self.assertEqual(
            current_job_address,
            set([
                '172.18.182.39:9245', '172.18.182.39:2235',
                '172.18.182.39:2236', '172.18.182.39:2237',
                '172.18.182.39:2238'
            ]))
        job_pool_address = set(job_center.job_pool.keys())
        self.assertEqual(
            job_pool_address,
            set([
                '172.18.182.39:9245', '172.18.182.39:2235',
                '172.18.182.39:2236', '172.18.182.39:2237',
                '172.18.182.39:2238', '172.18.182.39:1234',
                '172.18.182.39:1235', '172.18.182.39:1236',
                '172.18.182.39:1237', '172.18.182.39:1238'
            ]))

        job_center.drop_worker(self.worker2.worker_address)
        self.assertEqual(5, len(self.worker1.initialized_jobs))

    def test_cpu_num(self):
        job_center = JobCenter('localhost')
        job_center.add_worker(self.worker1)
        self.assertEqual(job_center.cpu_num, 5)
        job_center.add_worker(self.worker2)
        self.assertEqual(job_center.cpu_num, 10)
        job_center.request_job()
        self.assertEqual(job_center.cpu_num, 9)

    def test_worker_num(self):
        job_center = JobCenter('localhost')
        job_center.add_worker(self.worker1)
        self.assertEqual(job_center.worker_num, 1)
        job_center.add_worker(self.worker2)
        self.assertEqual(job_center.worker_num, 2)
        job_center.drop_worker(self.worker1.worker_address)
        self.assertEqual(job_center.worker_num, 1)


if __name__ == '__main__':
    unittest.main()
