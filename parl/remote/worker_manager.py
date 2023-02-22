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
import random
from collections import defaultdict
from collections import namedtuple
from parl.utils import logger
from parl.remote import remote_constants
from parl.remote.cpu_resource import CpuResource
from parl.remote.gpu_resource import GpuResource
from parl.remote.message import InitializedCpu
from parl.remote.message import InitializedGpu


class WorkerManager(object):
    """The WorkerManager deals with everything related to workers.

    Attributes:
        worker_hostname (dict): A dict to record worker hostname.
        worker_vacant_jobs (dict): Record which vacant jobs does each
                                   worker has.
        worker_used_jobs (dict): Record which used jobs does each
                                   worker has.
        master_ip (str): IP address of the master node.
    """

    def __init__(self, master_ip, devices=[remote_constants.CPU]):
        self.worker_hostname = {}
        self.worker_vacant_jobs = defaultdict(dict)
        self.worker_used_jobs = defaultdict(dict)
        self.lock = threading.Lock()
        self.master_ip = master_ip
        self.devices = devices
        self.cpu_resource = CpuResource()
        self.gpu_resource = GpuResource()

    @property
    def job_num(self):
        """"Return vacant job number."""
        with self.lock:
            return sum([len(vacant_jobs) for vacant_jobs in self.worker_vacant_jobs.values()])

    @property
    def cpu_num(self):
        """"Return vacant cpu number."""
        if remote_constants.CPU not in self.devices:
            return 0
        with self.lock:
            return self.cpu_resource.cpu_num()

    @property
    def gpu_num(self):
        """"Return vacant gpu number."""
        if remote_constants.GPU not in self.devices:
            return 0
        with self.lock:
            return self.gpu_resource.gpu_num()

    @property
    def worker_num(self):
        """Return connected worker number."""
        with self.lock:
            return len(self.worker_vacant_jobs)

    def add_worker(self, worker):
        """Add jobs, cpus, gpus from JobCenter when a worker connects.

        Args:
            worker (InitializedWorker): New worker with initialized job.
        Return: True if succeeds.
        """
        if remote_constants.GPU not in self.devices and worker.initialized_gpu.gpu != "":
            error_message = "[WorkerCenter] without GPU device rejects a worker with GPUs"
            logger.error(error_message)
            return False
        elif remote_constants.CPU not in self.devices and worker.initialized_cpu.n_cpu > 0:
            error_message = "[WorkerCenter] without CPU device rejects a worker with CPUs"
            logger.error(error_message)
            return False
        with self.lock:
            for job in worker.initialized_jobs:
                self.worker_vacant_jobs[worker.worker_address][job.job_address] = job
            self.cpu_resource.add_cpu(worker.worker_address, worker.initialized_cpu)
            self.gpu_resource.add_gpu(worker.worker_address, worker.initialized_gpu)

            if self.master_ip and worker.worker_address.split(':')[0] == self.master_ip:
                self.worker_hostname[worker.worker_address] = "Master"
                self.master_ip = None
            else:
                index = 1 + len(
                    [hostname for hostname in self.worker_hostname.values() if hostname.startswith(worker.hostname)])
                self.worker_hostname[worker.worker_address] = "{}:{}".format(worker.hostname, index)
        return True

    def drop_worker(self, worker_address):
        """Remove jobs, cpus, gpus from JobCenter when a worker dies.

        Args:
            worker_address (str): the worker_address of a worker to be
                                  removed from the worker manager.
        """
        with self.lock:
            self.worker_vacant_jobs.pop(worker_address, None)
            self.worker_used_jobs.pop(worker_address, None)
            self.worker_hostname.pop(worker_address, None)
            self.cpu_resource.drop_cpu(worker_address)
            self.gpu_resource.drop_gpu(worker_address)

    def request_job(self, n_cpu=1, n_gpu=0):
        """Return a job_address when the client submits a job.

        If there is no vacant CPU and GPU in the cluster, this will return None.

        Args:
            n_cpu (int): request a Job with n_cpu CPUs
            n_gpu (int): request a Job with n_gpu GPUs

        Return:
            An ``InitializedJob`` that has information about available job address.
        """
        with self.lock:
            candidates = self.filter(n_cpu, n_gpu)
            if candidates:
                candidates = list(candidates)
                random.shuffle(candidates)
                worker_address = candidates[0]
                initialized_resource = self.acquire(worker_address, n_cpu, n_gpu)
                job_address, job = self.worker_vacant_jobs[worker_address].popitem()
                job.initialized_cpu = initialized_resource.initialized_cpu
                job.initialized_gpu = initialized_resource.initialized_gpu
                self.worker_used_jobs[worker_address][job_address] = job
                return job
            return None

    def filter(self, n_cpu, n_gpu):
        candidates = set(self.worker_vacant_jobs.keys())
        if n_gpu > 0:
            candidates = candidates.intersection(set(self.gpu_resource.filter(n_gpu)))
        if n_cpu > 0:
            candidates = candidates.intersection(set(self.cpu_resource.filter(n_cpu)))
        return candidates

    def acquire(self, worker_address, n_cpu, n_gpu):
        initialized_gpu = InitializedGpu(worker_address, "")
        initialized_cpu = InitializedCpu(worker_address, 0)
        if n_gpu > 0:
            initialized_gpu = self.gpu_resource.acquire(worker_address, n_gpu)
        if n_cpu > 0:
            initialized_cpu = self.cpu_resource.acquire(worker_address, n_cpu)
        InitializedResource = namedtuple("InitializedResource", ["initialized_cpu", "initialized_gpu"])
        return InitializedResource(initialized_cpu=initialized_cpu, initialized_gpu=initialized_gpu)

    def release(self, killed_job):
        if killed_job.initialized_cpu:
            self.cpu_resource.release(killed_job.worker_address, killed_job.initialized_cpu)
        if killed_job.initialized_gpu:
            self.gpu_resource.release(killed_job.worker_address, killed_job.initialized_gpu)

    def reset_job(self, job):
        """Reset a job and add the job_address to the worker_vacant_jobs.

        Args:
            job(``InitializedJob``): The job information of the restarted job.
        """
        with self.lock:
            self.worker_used_jobs[job.worker_address].pop(job.job_address, None)
            self.worker_vacant_jobs[job.worker_address][job.job_address] = job
            self.release(job)

    def update_job(self, killed_job_address, new_job, worker_address):
        """When worker kill an old job, it will start a new job.

        Args:
            killed_job_address (str): The job address of the killed job.
            new_job(``InitializedJob``): Information of the new job.
            worker_address (str): The worker which kills an old job.
        """
        with self.lock:
            self.worker_vacant_jobs[worker_address][new_job.job_address] = new_job

            killed_job = None
            for vacant_jobs in self.worker_vacant_jobs.values():
                if killed_job_address in vacant_jobs:
                    killed_job = vacant_jobs.pop(killed_job_address)
                    break
            for used_jobs in self.worker_used_jobs.values():
                if killed_job_address in used_jobs:
                    killed_job = used_jobs.pop(killed_job_address)
                    break

            self.release(killed_job)

    def get_vacant_cpu(self, worker_address):
        """Return vacant cpu number of a worker."""
        if remote_constants.CPU not in self.devices:
            return 0
        with self.lock:
            return self.cpu_resource.get_vacant_cpu(worker_address)

    def get_vacant_gpu(self, worker_address):
        """Return vacant gpu number of a worker."""
        if remote_constants.GPU not in self.devices:
            return 0
        with self.lock:
            return self.gpu_resource.get_vacant_gpu(worker_address)

    def get_total_cpu(self, worker_address):
        """Return total cpu number of a worker."""
        if remote_constants.CPU not in self.devices:
            return 0
        with self.lock:
            return self.cpu_resource.get_total_cpu(worker_address)

    def get_total_gpu(self, worker_address):
        """Return total gpu number of a worker."""
        if remote_constants.GPU not in self.devices:
            return 0
        with self.lock:
            return self.gpu_resource.get_total_gpu(worker_address)

    def get_hostname(self, worker_address):
        """Return the hostname of a worker."""
        with self.lock:
            return self.worker_hostname[worker_address]
