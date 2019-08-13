#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import threading
from collections import defaultdict


class JobCenter(object):
    """The job center deals with everythin related to jobs.

    Attributes:
        job_pool (set): A set to store the job address of vacant cpu.
        worker_dict (dict): A dict to store connected workers.
        worker_hostname (dict): A dict to record worker hostname.
        worker_vacant_jobs (dict): Record how many vacant jobs does each
                                   worker has.
        master_ip (str): IP address of the master node.
    """

    def __init__(self, master_ip):
        self.job_pool = dict()
        self.worker_dict = {}
        self.worker_hostname = defaultdict(int)
        self.worker_vacant_jobs = {}
        self.lock = threading.Lock()
        self.master_ip = master_ip

    @property
    def cpu_num(self):
        """"Return vacant cpu number."""
        return len(self.job_pool)

    @property
    def worker_num(self):
        """Return connected worker number."""
        return len(self.worker_dict)

    def add_worker(self, worker):
        """When a new worker connects, add its hostname to worker_hostname.

        Args:
            worker (InitializedWorker): New worker with initialized jobs.
        """
        self.lock.acquire()
        self.worker_dict[worker.worker_address] = worker
        for job in worker.initialized_jobs:
            self.job_pool[job.job_address] = job

        self.worker_vacant_jobs[worker.worker_address] = len(
            worker.initialized_jobs)

        if self.master_ip and worker.worker_address.split(
                ':')[0] == self.master_ip:
            self.worker_hostname[worker.worker_address] = "Master"
            self.master_ip = None
        else:
            self.worker_hostname[worker.hostname] += 1
            self.worker_hostname[worker.worker_address] = "{}:{}".format(
                worker.hostname, self.worker_hostname[worker.hostname])
        self.lock.release()

    def drop_worker(self, worker_address):
        """Remove jobs from job_pool when a worker dies.

        Args:
            worker_address (str): the worker_address of a worker to be
                                  removed from the job center.
        """
        self.lock.acquire()
        worker = self.worker_dict[worker_address]
        for job in worker.initialized_jobs:
            if job.job_address in self.job_pool:
                self.job_pool.pop(job.job_address)
        self.worker_dict.pop(worker_address)
        self.worker_vacant_jobs.pop(worker_address)
        self.lock.release()

    def request_job(self):
        """Return a job_address when the client submits a job.

        If there is no vacant CPU in the cluster, this will return None.

        Return:
            An ``InitializedJob`` that has information about available job address.
        """
        self.lock.acquire()
        job = None
        if len(self.job_pool):
            job_address, job = self.job_pool.popitem()
            self.worker_vacant_jobs[job.worker_address] -= 1
            assert self.worker_vacant_jobs[job.worker_address] >= 0
        self.lock.release()
        return job

    def reset_job(self, job):
        """Reset a job and add the job_address to the job_pool.

        Args:
            job(``InitializedJob``): The job information of the restarted job.
        """
        self.lock.acquire()
        self.job_pool[job.job_address] = job
        self.lock.release()

    def update_job(self, killed_job_address, new_job, worker_address):
        """When worker kill an old job, it will start a new job.

        Args:
            killed_job_address (str): The job address of the killed job.
            new_job(``InitializedJob``): Information of the new job.
            worker_address (str): The worker which kills an old job.
        """
        self.lock.acquire()
        self.job_pool[new_job.job_address] = new_job

        if killed_job_address in self.job_pool:
            self.job_pool.pop(killed_job_address)

        to_del_idx = None
        for i, job in enumerate(
                self.worker_dict[worker_address].initialized_jobs):
            if job.job_address == killed_job_address:
                to_del_idx = i
                break

        del self.worker_dict[worker_address].initialized_jobs[to_del_idx]
        self.worker_dict[worker_address].initialized_jobs.append(new_job)

        if killed_job_address not in self.job_pool:
            self.worker_vacant_jobs[worker_address] += 1

        self.lock.release()

    def get_vacant_cpu(self, worker_address):
        """Return vacant cpu number of a worker."""
        return self.worker_vacant_jobs[worker_address]

    def get_total_cpu(self, worker_address):
        """Return total cpu number of a worker."""
        return len(self.worker_dict[worker_address].initialized_jobs)

    def get_hostname(self, worker_address):
        """Return the hostname of a worker."""
        return self.worker_hostname[worker_address]
