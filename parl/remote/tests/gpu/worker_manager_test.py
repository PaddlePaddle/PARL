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
from parl.utils.test_utils import XparlTestCase


class ImportTest(unittest.TestCase):
    def setUp(self):
        jobs = []
        gpu_num = 4
        gpu = "0,1,2,3"
        for i in range(gpu_num):
            job = InitializedJob(
                job_address='192.168.0.1:{}'.format(1000 + i),
                worker_heartbeat_address='192.168.0.1:{}'.format(1000 + i + gpu_num),
                ping_heartbeat_address='192.168.0.1:{}'.format(1000 + i + gpu_num * 3),
                worker_address='192.168.0.1:8000',
                pid=1000 + i)
            jobs.append(job)

        allocated_cpu = AllocatedCpu('192.168.0.1:8000', 0)
        allocated_gpu = AllocatedGpu('192.168.0.1:8000', gpu)
        self.worker1 = InitializedWorker('192.168.0.1:8000', jobs, allocated_cpu, allocated_gpu, 'worker1')

        jobs = []
        gpu_num = 4
        gpu = "0,1,2,3"
        for i in range(gpu_num):
            job = InitializedJob(
                job_address='192.168.0.2:{}'.format(1000 + i),
                worker_heartbeat_address='192.168.0.2:{}'.format(1000 + i + gpu_num),
                ping_heartbeat_address='192.168.0.2:{}'.format(1000 + i + gpu_num * 3),
                worker_address='192.168.0.2:8000',
                pid=1000 + i)
            jobs.append(job)

        allocated_cpu = AllocatedCpu('192.168.0.2:8000', 0)
        allocated_gpu = AllocatedGpu('192.168.0.2:8000', gpu)
        self.worker2 = InitializedWorker('192.168.0.2:8000', jobs, allocated_cpu, allocated_gpu, 'worker2')

        jobs = []
        cpu_num = 4
        for i in range(cpu_num):
            job = InitializedJob(
                job_address='192.168.0.3:{}'.format(1000 + i),
                worker_heartbeat_address='192.168.0.3:{}'.format(1000 + i + cpu_num),
                ping_heartbeat_address='192.168.0.3:{}'.format(1000 + i + cpu_num * 3),
                worker_address='192.168.0.3:8000',
                pid=1000 + i)
            jobs.append(job)

        allocated_cpu = AllocatedCpu('192.168.0.3:8000', cpu_num)
        allocated_gpu = AllocatedGpu('192.168.0.3:8000', '')
        self.worker3 = InitializedWorker('192.168.0.3:8000', jobs, allocated_cpu, allocated_gpu, 'worker3')

        jobs = []
        gpu_num = 8
        gpu = '0,1,2,3,4,5,6,7'
        for i in range(gpu_num):
            job = InitializedJob(
                job_address='192.168.0.5:{}'.format(1000 + i),
                worker_heartbeat_address='192.168.0.5:{}'.format(1000 + i + gpu_num),
                ping_heartbeat_address='192.168.0.5:{}'.format(1000 + i + gpu_num * 3),
                worker_address='192.168.0.5:8000',
                pid=1000 + i)
            jobs.append(job)

        allocated_cpu = AllocatedCpu('192.168.0.5:8000', 0)
        allocated_gpu = AllocatedGpu('192.168.0.5:8000', gpu)
        self.worker5 = InitializedWorker('192.168.0.5:8000', jobs, allocated_cpu, allocated_gpu, 'worker5')

    def test_add_worker(self):
        worker_manager = WorkerManager('localhost', ['gpu'])
        worker_manager.add_worker(self.worker1)
        self.assertEqual(worker_manager.job_num, 4)
        self.assertEqual(self.worker1.worker_address in worker_manager.worker_vacant_jobs, True)

        worker_manager.add_worker(self.worker2)
        self.assertEqual(worker_manager.job_num, 8)
        self.assertEqual(self.worker2.worker_address in worker_manager.worker_vacant_jobs, True)

        flag = worker_manager.add_worker(self.worker3)
        self.assertEqual(flag, False)
        self.assertEqual(worker_manager.job_num, 8)

    def test_remove_worker(self):
        worker_manager = WorkerManager('localhost', ['gpu'])
        worker_manager.add_worker(self.worker1)
        worker_manager.add_worker(self.worker2)

        self.assertEqual(worker_manager.job_num, 8)
        self.assertEqual(self.worker1.worker_address in worker_manager.worker_vacant_jobs, True)
        self.assertEqual(self.worker2.worker_address in worker_manager.worker_vacant_jobs, True)

        worker_manager.remove_worker(self.worker1.worker_address)
        self.assertEqual(worker_manager.job_num, 4)
        self.assertEqual(self.worker1.worker_address in worker_manager.worker_vacant_jobs, False)

        worker_manager.remove_worker(self.worker2.worker_address)
        self.assertEqual(worker_manager.job_num, 0)
        self.assertEqual(self.worker2.worker_address in worker_manager.worker_vacant_jobs, False)

    def test_request_job(self):
        worker_manager = WorkerManager('localhost', ['gpu'])
        job1 = worker_manager.request_job(n_cpu=0, n_gpu=2)
        self.assertTrue(job1 is None)

        worker_manager.add_worker(self.worker1)
        job2 = worker_manager.request_job(n_cpu=0, n_gpu=2)
        self.assertTrue(job2 in self.worker1.initialized_jobs)
        self.assertEqual(worker_manager.job_num, 3)
        self.assertEqual(worker_manager.gpu_num, 2)

        job3 = worker_manager.request_job(n_cpu=0, n_gpu=4)
        self.assertTrue(job3 is None)

        worker_manager.add_worker(self.worker2)
        job4 = worker_manager.request_job(n_cpu=0, n_gpu=8)
        self.assertTrue(job4 is None)

        worker_manager.add_worker(self.worker5)
        job5 = worker_manager.request_job(n_cpu=0, n_gpu=9)
        self.assertTrue(job5 is None)

    def test_update_job(self):

        worker_manager = WorkerManager('localhost', ['gpu'])
        worker_manager.add_worker(self.worker1)
        worker_manager.add_worker(self.worker2)

        job = InitializedJob(
            job_address='192.168.0.1:{}'.format(2000),
            worker_heartbeat_address='192.168.0.1:{}'.format(2000 + 1),
            ping_heartbeat_address='192.168.0.1:{}'.format(2000 + 3),
            worker_address='192.168.0.1:8000',
            pid=2000)
        worker_manager.update_job('192.168.0.1:1000', job, '192.168.0.1:8000')

        current_job_address = set(list(worker_manager.worker_vacant_jobs['192.168.0.1:8000'].keys()))
        self.assertEqual(current_job_address,
                         set(['192.168.0.1:1001', '192.168.0.1:1002', '192.168.0.1:1003', '192.168.0.1:2000']))
        all_job_address = set(
            list(worker_manager.worker_vacant_jobs[self.worker1.worker_address].keys()) +
            list(worker_manager.worker_vacant_jobs[self.worker2.worker_address].keys()))
        self.assertEqual(
            all_job_address,
            set([
                '192.168.0.1:1001', '192.168.0.1:1002', '192.168.0.1:1003', '192.168.0.1:2000', '192.168.0.2:1000',
                '192.168.0.2:1001', '192.168.0.2:1002', '192.168.0.2:1003'
            ]))

        worker_manager.remove_worker(self.worker2.worker_address)
        self.assertEqual(4, len(self.worker1.initialized_jobs))

    def test_gpu_num(self):
        worker_manager = WorkerManager('localhost', ['gpu'])
        worker_manager.add_worker(self.worker1)
        self.assertEqual(worker_manager.gpu_num, 4)
        worker_manager.add_worker(self.worker2)
        self.assertEqual(worker_manager.gpu_num, 8)
        job = worker_manager.request_job(n_cpu=0, n_gpu=2)
        self.assertEqual(worker_manager.gpu_num, 6)

    def test_worker_num(self):
        worker_manager = WorkerManager('localhost', ['gpu'])
        worker_manager.add_worker(self.worker1)
        self.assertEqual(worker_manager.worker_num, 1)
        worker_manager.add_worker(self.worker2)
        self.assertEqual(worker_manager.worker_num, 2)
        worker_manager.remove_worker(self.worker1.worker_address)
        self.assertEqual(worker_manager.worker_num, 1)


if __name__ == '__main__':
    unittest.main()
