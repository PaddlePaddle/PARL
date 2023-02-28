#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from parl.remote.job_pool import JobPool
from parl.remote.message import InitializedWorker, InitializedJob


class JobPoolTest(unittest.TestCase):
    def setUp(self):
        self.jobs_0 = []
        for i in range(5):
            job = InitializedJob(
                job_address='172.18.182.39:{}'.format(1234 + i),
                worker_heartbeat_address='172.18.182.39:48724',
                ping_heartbeat_address='172.18.182.39:48726',
                worker_address='172.18.182.39:8001',
                pid=1234)
            self.jobs_0.append(job)

        self.jobs_1 = []
        for i in range(5):
            job = InitializedJob(
                job_address='172.18.182.39:{}'.format(2234 + i),
                worker_heartbeat_address='172.18.182.39:48724',
                ping_heartbeat_address='172.18.182.39:48726',
                worker_address='172.18.182.39:8002',
                pid=1234)
            self.jobs_1.append(job)

    def test_remove_job(self):
        job_pool = JobPool(worker_address='172.18.182.39:8005',
                initialized_jobs=self.jobs_0, cpu_num=5)
        job_pool.remove_job('172.18.182.39:1234')
        self.assertEqual(len(job_pool.jobs), 4)

    def test_add_job(self):
        job_pool = JobPool(worker_address='172.18.182.39:8005',
                initialized_jobs=self.jobs_0[:3], cpu_num=5)
        self.assertEqual(len(job_pool.jobs), 3)
        job_pool.add_job(self.jobs_0[3])
        job_pool.add_job(self.jobs_0[4])
        self.assertEqual(len(job_pool.jobs), 5)
        with self.assertRaises(AssertionError):
            job_pool.add_job(self.jobs_1[0])


if __name__ == '__main__':
    unittest.main()
