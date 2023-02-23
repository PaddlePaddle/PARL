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
from parl.remote.worker_manager import WorkerManager
from parl.remote.message import InitializedWorker, InitializedJob, AllocatedCpu, AllocatedGpu


class ImportTest(unittest.TestCase):
    def setUp(self):
        jobs = []
        n_cpu = 5
        for i in range(n_cpu):
            job = InitializedJob(
                job_address='172.18.182.39:{}'.format(1234 + i),
                worker_heartbeat_address='172.18.182.39:48724',
                ping_heartbeat_address='172.18.182.39:48726',
                worker_address='172.18.182.39:8001',
                pid=1234)
            jobs.append(job)

        allocated_cpu = AllocatedCpu('172.18.182.39:8001', n_cpu)
        allocated_gpu = AllocatedGpu('172.18.182.39:8001', "")
        self.worker1 = InitializedWorker(
            worker_address='172.18.182.39:8001',
            initialized_jobs=jobs,
            allocated_cpu=allocated_cpu,
            allocated_gpu=allocated_gpu,
            hostname=socket.gethostname())

        n_cpu = 5
        jobs = []
        for i in range(n_cpu):
            job = InitializedJob(
                job_address='172.18.182.39:{}'.format(2234 + i),
                worker_heartbeat_address='172.18.182.39:48724',
                ping_heartbeat_address='172.18.182.39:48726',
                worker_address='172.18.182.39:8002',
                pid=1234)
            jobs.append(job)

        allocated_cpu = AllocatedCpu('172.18.182.39:8002', n_cpu)
        allocated_gpu = AllocatedGpu('172.18.182.39:8002', "")
        self.worker2 = InitializedWorker(
            worker_address='172.18.182.39:8002',
            initialized_jobs=jobs,
            allocated_cpu=allocated_cpu,
            allocated_gpu=allocated_gpu,
            hostname=socket.gethostname())

    def test_add_worker(self):

        worker_manager = WorkerManager('localhost')
        worker_manager.add_worker(self.worker1)
        worker_manager.add_worker(self.worker2)

        self.assertEqual(worker_manager.job_num, 10)
        self.assertEqual(worker_manager.cpu_num, 10)
        self.assertEqual(
            set(worker_manager.worker_vacant_jobs[self.worker1.worker_address].values()),
            set(self.worker1.initialized_jobs))

    def test_remove_worker(self):
        worker_manager = WorkerManager('localhost')
        worker_manager.add_worker(self.worker1)
        worker_manager.add_worker(self.worker2)
        worker_manager.remove_worker(self.worker2.worker_address)

        self.assertEqual(
            set(worker_manager.worker_vacant_jobs[self.worker1.worker_address].values()),
            set(self.worker1.initialized_jobs))
        self.assertEqual(worker_manager.worker_num, 1)

    def test_request_job(self):
        worker_manager = WorkerManager('localhost')
        job_address1 = worker_manager.request_job(n_cpu=1)
        self.assertTrue(job_address1 is None)

        worker_manager.add_worker(self.worker1)
        job_address2 = worker_manager.request_job(n_cpu=1)
        self.assertTrue(job_address2 in self.worker1.initialized_jobs)
        self.assertEqual(worker_manager.job_num, 4)
        self.assertEqual(worker_manager.cpu_num, 4)

    def test_update_job(self):

        worker_manager = WorkerManager('localhost')
        worker_manager.add_worker(self.worker1)
        worker_manager.add_worker(self.worker2)

        job = InitializedJob(
            job_address='172.18.182.39:{}'.format(9245),
            worker_heartbeat_address='172.18.182.39:48724',
            ping_heartbeat_address='172.18.182.39:48726',
            worker_address='172.18.182.39:478727',
            pid=1234)
        worker_manager.update_job('172.18.182.39:2234', job, '172.18.182.39:8002')

        current_job_address = set(worker_manager.worker_vacant_jobs['172.18.182.39:8002'].keys())
        self.assertEqual(
            current_job_address,
            set([
                '172.18.182.39:9245', '172.18.182.39:2235', '172.18.182.39:2236', '172.18.182.39:2237',
                '172.18.182.39:2238'
            ]))
        job_pool_address = set(
            list(worker_manager.worker_vacant_jobs[self.worker1.worker_address].keys()) +
            list(worker_manager.worker_vacant_jobs[self.worker2.worker_address].keys()))
        self.assertEqual(
            job_pool_address,
            set([
                '172.18.182.39:9245', '172.18.182.39:2235', '172.18.182.39:2236', '172.18.182.39:2237',
                '172.18.182.39:2238', '172.18.182.39:1234', '172.18.182.39:1235', '172.18.182.39:1236',
                '172.18.182.39:1237', '172.18.182.39:1238'
            ]))

        worker_manager.remove_worker(self.worker2.worker_address)
        self.assertEqual(5, len(self.worker1.initialized_jobs))

    def test_cpu_num(self):
        worker_manager = WorkerManager('localhost')
        worker_manager.add_worker(self.worker1)
        self.assertEqual(worker_manager.cpu_num, 5)
        worker_manager.add_worker(self.worker2)
        self.assertEqual(worker_manager.cpu_num, 10)
        job = worker_manager.request_job(n_cpu=1)
        self.assertEqual(worker_manager.cpu_num, 9)

    def test_worker_num(self):
        worker_manager = WorkerManager('localhost')
        worker_manager.add_worker(self.worker1)
        self.assertEqual(worker_manager.worker_num, 1)
        worker_manager.add_worker(self.worker2)
        self.assertEqual(worker_manager.worker_num, 2)
        worker_manager.remove_worker(self.worker1.worker_address)
        self.assertEqual(worker_manager.worker_num, 1)


if __name__ == '__main__':
    unittest.main()
