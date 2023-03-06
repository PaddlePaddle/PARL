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
from parl.remote.message import AllocatedCpu
from parl.remote.message import AllocatedGpu


class WorkerManager(object):
    """A thread-safe data structure used to respond to resource request,
       maintaining the available computation resources in each worker.

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
        error_message = "invalid devices: {}".format(",".join(devices))
        assert len(set(devices) - set([remote_constants.CPU, remote_constants.GPU])) == 0, error_message
        self.devices = devices
        self.cpu_resource = CpuResource()
        self.gpu_resource = GpuResource()

    @property
    def job_num(self):
        """"Return the number of vacant jobs."""
        with self.lock:
            return sum([len(vacant_jobs) for vacant_jobs in self.worker_vacant_jobs.values()])

    @property
    def cpu_num(self):
        """"Return the number of vacant CPUs."""
        if remote_constants.CPU not in self.devices:
            return 0
        with self.lock:
            return self.cpu_resource.cpu_num()

    @property
    def gpu_num(self):
        """"Return the number of vacant GPUs."""
        if remote_constants.GPU not in self.devices:
            return 0
        with self.lock:
            return self.gpu_resource.gpu_num()

    @property
    def worker_num(self):
        """Return the number of workers connected to the master node."""
        with self.lock:
            return len(self.worker_vacant_jobs)

    def add_worker(self, worker):
        """Record initialized jobs and CPU/GPU information when a worker connects.

        Args:
            worker (InitializedWorker): New worker with initialized jobs and CPU/GPU information.
        Return: True if succeeds.
        """
        if remote_constants.GPU not in self.devices and worker.allocated_gpu.gpu != "":
            error_message = "The CPU cluster is not allowed to accept a worker with GPU resources."
            logger.error(error_message)
            return False
        elif remote_constants.CPU not in self.devices and worker.allocated_cpu.n_cpu > 0:
            error_message = "The GPU cluster is not allowed to accept a worker with CPU resources."
            logger.error(error_message)
            return False
        with self.lock:
            for job in worker.initialized_jobs:
                self.worker_vacant_jobs[worker.worker_address][job.job_address] = job
            self.cpu_resource.add_cpu(worker.worker_address, worker.allocated_cpu)
            self.gpu_resource.add_gpu(worker.worker_address, worker.allocated_gpu)

            if self.master_ip and worker.worker_address.split(':')[0] == self.master_ip:
                self.worker_hostname[worker.worker_address] = "Master"
                self.master_ip = None
            else:
                index = 1 + len(
                    [hostname for hostname in self.worker_hostname.values() if hostname.startswith(worker.hostname)])
                self.worker_hostname[worker.worker_address] = "{}:{}".format(worker.hostname, index)
        return True

    def remove_worker(self, worker_address):
        """Remove jobs, cpus, gpus from JobCenter when a worker dies.

        Args:
            worker_address (str): the worker_address of a worker to be
                                  removed from the worker manager.
        """
        with self.lock:
            self.worker_vacant_jobs.pop(worker_address, None)
            self.worker_used_jobs.pop(worker_address, None)
            self.worker_hostname.pop(worker_address, None)
            self.cpu_resource.remove_cpu(worker_address)
            self.gpu_resource.remove_gpu(worker_address)

    def request_job(self, n_cpu=1, n_gpu=0):
        """Return a job when the client submits a job.

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
                random.shuffle(candidates)
                worker_address = candidates[0]
                job_address, job = self.worker_vacant_jobs[worker_address].popitem()
                job.allocated_cpu, job.allocated_gpu = self.allocate(worker_address, n_cpu, n_gpu)
                self.worker_used_jobs[worker_address][job_address] = job
                return job
            logger.warning("No enough cpu/gpu resources at the moment")
            return None

    def filter(self, n_cpu, n_gpu):
        """Return a list of worker_address, has n_cpu vacant CPUs and n_gpu vacant GPUs at least.

        If there is no vacant CPU and GPU in the cluster, this will return empty list.

        Args:
            n_cpu (int): request a Job with n_cpu CPUs
            n_gpu (int): request a Job with n_gpu GPUs

        Return:
            An ``InitializedJob`` that has information about available computation resources.
        """
        candidates = set(self.worker_vacant_jobs.keys())
        if n_gpu > 0:
            candidates = candidates.intersection(set(self.gpu_resource.filter(n_gpu)))
        if n_cpu > 0:
            candidates = candidates.intersection(set(self.cpu_resource.filter(n_cpu)))
        return list(candidates)

    def allocate(self, worker_address, n_cpu, n_gpu):
        """Allocate n_cpu CPUs and n_gpu GPUS from the worker.
        Args:
            worker_address (str): The worker which allocates CPU and GPU resources.
            n_cpu (int): allocate n_cpu CPUs
            n_gpu (int): allocate n_gpu GPUs
        Returns:
            allocated_cpu (``AllocateCpu``): The allocation information of CPU
            allocated_gpu (``AllocateGpu``): The allocation information of GPU
        """
        allocated_cpu = AllocatedCpu(worker_address, 0)
        allocated_gpu = AllocatedGpu(worker_address, "")
        if n_cpu > 0:
            allocated_cpu = self.cpu_resource.allocate(worker_address, n_cpu)
        if n_gpu > 0:
            allocated_gpu = self.gpu_resource.allocate(worker_address, n_gpu)
        return allocated_cpu, allocated_gpu

    def remove_job(self, killed_job):
        """Remove a job information when worker kill an old job
        Args:
            killed_job (``InitializedJob``): Information of the old job
        """
        if killed_job.allocated_cpu:
            self.cpu_resource.recycle(killed_job.worker_address, killed_job.allocated_cpu)
        if killed_job.allocated_gpu:
            self.gpu_resource.recycle(killed_job.worker_address, killed_job.allocated_gpu)

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

            self.remove_job(killed_job)

    def get_vacant_cpu(self, worker_address):
        """Return the number of vacant CPUs in the worker given its address."""
        if remote_constants.CPU not in self.devices:
            return 0
        with self.lock:
            return self.cpu_resource.get_vacant_cpu(worker_address)

    def get_vacant_gpu(self, worker_address):
        """Return the number of vacant GPUs in the worker given its address."""
        if remote_constants.GPU not in self.devices:
            return 0
        with self.lock:
            return self.gpu_resource.get_vacant_gpu(worker_address)

    def get_total_cpu(self, worker_address):
        """Return the number of CPUs in the worker given its address."""
        if remote_constants.CPU not in self.devices:
            return 0
        with self.lock:
            return self.cpu_resource.get_total_cpu(worker_address)

    def get_total_gpu(self, worker_address):
        """Return the number of GPUs in the worker given its address."""
        if remote_constants.GPU not in self.devices:
            return 0
        with self.lock:
            return self.gpu_resource.get_total_gpu(worker_address)

    def get_hostname(self, worker_address):
        """Return the hostname of a worker."""
        with self.lock:
            return self.worker_hostname[worker_address]
