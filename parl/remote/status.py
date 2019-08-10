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
from parl.utils import logger
import signal
import os


class WorkerStatus(object):
    """Maintain worker's information in a worker node. 

    Attributes:
        cpu_num(int): The number of CPUs to be used in this worker.
        jobs(set): A set that records job addresses provided to the master.
        worker_address(str): Address of the worker.
    """

    def __init__(self, worker_address, initialized_jobs, cpu_num):
        self.worker_address = worker_address
        self.jobs = dict()
        for job in initialized_jobs:
            self.jobs[job.job_address] = job
        self._lock = threading.Lock()
        self.cpu_num = cpu_num

    def remove_job(self, killed_job):
        """Rmove a job from internal job pool.

        Args:
            killed_job(str): Job address to be removed.

        Returns: True if removing the job succeeds.
        """
        ret = False
        self._lock.acquire()
        if killed_job in self.jobs:
            pid = self.jobs[killed_job].pid
            self.jobs.pop(killed_job)
            ret = True
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                logger.warning("job:{} has been killed before".format(pid))
            logger.info("[Worker] kills a job:{}".format(killed_job))
        self._lock.release()
        return ret

    def clear(self):
        """Remove all the jobs"""
        self._lock.acquire()
        for job in self.jobs.values():
            try:
                os.kill(job.pid, signal.SIGTERM)
            except OSError:
                logger.warning("job:{} has been killed before".format(job.pid))
            logger.info("[Worker] kills a job:{}".format(job.pid))
        self.jobs = dict()
        self._lock.release()

    def add_job(self, new_job):
        """Add a new job to interal job pool.
        
        Args:
            new_job(InitializedJob): Initialized job to be added to the self.jobs.
        """
        self._lock.acquire()
        self.jobs[new_job.job_address] = new_job
        assert len(self.jobs) <= self.cpu_num
        self._lock.release()
