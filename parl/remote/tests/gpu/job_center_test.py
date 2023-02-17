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


class ImportTest(unittest.TestCase):
    def setUp(self):
        jobs = []
        gpu_num = 4
        gpu_ids = "0,1,2,3"
        for i in range(gpu_num):
            job = InitializedJob(
                job_address='192.168.0.1:{}'.format(1000 + i),
                worker_heartbeat_address='192.168.0.1:{}'.format(1000 + i + gpu_num),
                client_heartbeat_address='192.168.0.1:{}'.format(1000 + i + gpu_num * 2),
                ping_heartbeat_address='192.168.0.1:{}'.format(1000 + i + gpu_num * 3),
                worker_address='192.168.0.1:8000',
                pid=1000 + i)
            jobs.append(job)

        self.worker1 = InitializedWorker('192.168.0.1:8000', jobs, 0, gpu_ids, 'worker1')

        jobs = []
        gpu_num = 4
        gpu_ids = "0,1,2,3"
        for i in range(gpu_num):
            job = InitializedJob(
                job_address='192.168.0.2:{}'.format(1000 + i),
                worker_heartbeat_address='192.168.0.2:{}'.format(1000 + i + gpu_num),
                client_heartbeat_address='192.168.0.2:{}'.format(1000 + i + gpu_num * 2),
                ping_heartbeat_address='192.168.0.2:{}'.format(1000 + i + gpu_num * 3),
                worker_address='192.168.0.2:8000',
                pid=1000 + i)
            jobs.append(job)

        self.worker2 = InitializedWorker('192.168.0.2:8000', jobs, 0, gpu_ids, 'worker2')

        jobs = []
        cpu_num = 4
        for i in range(cpu_num):
            job = InitializedJob(
                job_address='192.168.0.3:{}'.format(1000 + i),
                worker_heartbeat_address='192.168.0.3:{}'.format(1000 + i + cpu_num),
                client_heartbeat_address='192.168.0.3:{}'.format(1000 + i + cpu_num * 2),
                ping_heartbeat_address='192.168.0.3:{}'.format(1000 + i + cpu_num * 3),
                worker_address='192.168.0.3:8000',
                pid=1000 + i)
            jobs.append(job)

        gpu_ids = ''
        self.worker3 = InitializedWorker('192.168.0.3:8000', jobs, cpu_num, gpu_ids, 'worker3')

        jobs = []
        gpu_num = 8
        gpu_ids = '0,1,2,3,4,5,6,7'
        for i in range(gpu_num):
            job = InitializedJob(
                job_address='192.168.0.5:{}'.format(1000 + i),
                worker_heartbeat_address='192.168.0.5:{}'.format(1000 + i + gpu_num),
                client_heartbeat_address='192.168.0.5:{}'.format(1000 + i + gpu_num * 2),
                ping_heartbeat_address='192.168.0.5:{}'.format(1000 + i + gpu_num * 3),
                worker_address='192.168.0.5:8000',
                pid=1000 + i)
            jobs.append(job)

        self.worker5 = InitializedWorker('192.168.0.5:8000', jobs, 0, gpu_ids, 'worker5')

    def test_add_worker(self):
        job_center = JobCenter('localhost', 'gpu')
        job_center.add_worker(self.worker1)
        self.assertEqual(len(job_center.job_pool), 4)
        self.assertEqual(job_center.worker_dict[self.worker1.worker_address], self.worker1)

        job_center.add_worker(self.worker2)
        self.assertEqual(len(job_center.job_pool), 8)
        self.assertEqual(job_center.worker_dict[self.worker2.worker_address], self.worker2)

        flag = job_center.add_worker(self.worker3)
        self.assertEqual(flag, False)
        self.assertEqual(len(job_center.job_pool), 8)

    def test_drop_worker(self):
        job_center = JobCenter('localhost', 'gpu')
        job_center.add_worker(self.worker1)
        job_center.add_worker(self.worker2)

        self.assertEqual(len(job_center.job_pool), 8)
        self.assertEqual(job_center.worker_dict[self.worker1.worker_address], self.worker1)
        self.assertEqual(job_center.worker_dict[self.worker2.worker_address], self.worker2)

        job_center.drop_worker(self.worker1.worker_address)
        self.assertEqual(len(job_center.job_pool), 4)
        self.assertEqual(self.worker1.worker_address in job_center.worker_dict, False)

        job_center.drop_worker(self.worker2.worker_address)
        self.assertEqual(len(job_center.job_pool), 0)
        self.assertEqual(self.worker2.worker_address in job_center.worker_dict, False)

    def test_request_job(self):
        job_center = JobCenter('localhost', 'gpu')
        job1 = job_center.request_job(n_gpus=2)
        self.assertTrue(job1 is None)

        job_center.add_worker(self.worker1)
        job2 = job_center.request_job(n_gpus=2)
        self.assertTrue(job2 in self.worker1.initialized_jobs)
        self.assertEqual(len(job_center.job_pool), 3)
        self.assertEqual(job_center.gpu_num, 2)

        job3 = job_center.request_job(n_gpus=4)
        self.assertTrue(job3 is None)

        job_center.add_worker(self.worker5)
        job4 = job_center.request_job(n_gpus=9)
        self.assertTrue(job4 is None)

    def test_update_job(self):

        job_center = JobCenter('localhost', 'gpu')
        job_center.add_worker(self.worker1)
        job_center.add_worker(self.worker2)

        job = InitializedJob(
            job_address='192.168.0.1:{}'.format(2000),
            worker_heartbeat_address='192.168.0.1:{}'.format(2000 + 1),
            client_heartbeat_address='192.168.0.1:{}'.format(2000 + 2),
            ping_heartbeat_address='192.168.0.1:{}'.format(2000 + 3),
            worker_address='192.168.0.1:8000',
            pid=2000)
        job_center.update_job('192.168.0.1:1000', job, '192.168.0.1:8000')

        current_job_address = set(
            [job.job_address for job in job_center.worker_dict['192.168.0.1:8000'].initialized_jobs])
        self.assertEqual(current_job_address,
                         set(['192.168.0.1:1001', '192.168.0.1:1002', '192.168.0.1:1003', '192.168.0.1:2000']))
        job_pool_address = set(job_center.job_pool.keys())
        self.assertEqual(
            job_pool_address,
            set([
                '192.168.0.1:1001', '192.168.0.1:1002', '192.168.0.1:1003', '192.168.0.1:2000', '192.168.0.2:1000',
                '192.168.0.2:1001', '192.168.0.2:1002', '192.168.0.2:1003'
            ]))

        job_center.drop_worker(self.worker2.worker_address)
        self.assertEqual(4, len(self.worker1.initialized_jobs))

    def test_gpu_num(self):
        job_center = JobCenter('localhost', 'gpu')
        job_center.add_worker(self.worker1)
        self.assertEqual(job_center.gpu_num, 4)
        job_center.add_worker(self.worker2)
        self.assertEqual(job_center.gpu_num, 8)
        job = job_center.request_job(n_gpus=2)
        self.assertEqual(job_center.gpu_num, 6)

    def test_worker_num(self):
        job_center = JobCenter('localhost', 'gpu')
        job_center.add_worker(self.worker1)
        self.assertEqual(job_center.worker_num, 1)
        job_center.add_worker(self.worker2)
        self.assertEqual(job_center.worker_num, 2)
        job_center.drop_worker(self.worker1.worker_address)
        self.assertEqual(job_center.worker_num, 1)


if __name__ == '__main__':
    unittest.main()
