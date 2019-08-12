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


class JobCenter(object):
    """The job center deals with everythin related to jobs.

    Attributes:
        job_pool (set): A set to store the job address of vacant cpu.
        worker_dict (dict): A dict to store connected workers.
    """

    def __init__(self):
        self.job_pool = dict()
        self.worker_dict = {}
        self.lock = threading.Lock()

    @property
    def cpu_num(self):
        """"Return vacant cpu number."""
        return len(self.job_pool)

    @property
    def worker_num(self):
        """Return connected worker number."""
        return len(self.worker_dict)

    def add_worker(self, worker):
        """A new worker connects.

        Args:
            worker (InitializedWorker): New worker with initialized jobs.
        """
        self.lock.acquire()
        self.worker_dict[worker.worker_address] = worker
        for job in worker.initialized_jobs:
            self.job_pool[job.job_address] = job
        self.lock.release()

    def drop_worker(self, worker_address):
        """Remove jobs from job_pool when a worker dies.

        Args:
            worker (start): Old worker to be removed from the cluster.        
        """
        self.lock.acquire()
        worker = self.worker_dict[worker_address]
        for job in worker.initialized_jobs:
            if job.job_address in self.job_pool:
                self.job_pool.pop(job.job_address)
        self.worker_dict.pop(worker_address)
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
        self.lock.release()
