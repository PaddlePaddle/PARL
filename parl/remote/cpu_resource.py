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

from collections import defaultdict
from parl.remote.message import InitializedCpu


class CpuResource(object):
    """The cpu center deals with everything related to allocation for CPUs.

    Attributes:
        worker_vacant_cpus (dict): Record how many vacant cpus does each
                                   worker has.
        worker_used_cpus (dict): Record how many used cpus does each
                                   worker has.
    """

    def __init__(self):
        self.worker_vacant_cpus = defaultdict(int)
        self.worker_used_cpus = defaultdict(int)

    def add_cpu(self, worker_address, initialized_cpu):
        """add CPU resource to worker_vacant_cpus
        Args:
            initialized_cpu (InitializedCpu): Record the CPU resource used in a job.
        """
        self.worker_vacant_cpus[worker_address] += initialized_cpu.n_cpu

    def drop_cpu(self, worker_address):
        """Remove cpus from CpuResource when a worker dies.
        """
        self.worker_vacant_cpus.pop(worker_address, None)
        self.worker_used_cpus.pop(worker_address, None)

    def filter(self, n_cpu):
        """filter workers that contain n_cpu CPUs at least
        """
        result = []
        for worker_address, vacant_cpus in self.worker_vacant_cpus.items():
            if vacant_cpus >= n_cpu:
                result.append(worker_address)
        return result

    def acquire(self, worker_address, n_cpu):
        """acquire n_cpu CPUs on worker_address
        """
        self.worker_vacant_cpus[worker_address] -= n_cpu
        self.worker_used_cpus[worker_address] += n_cpu
        assert self.worker_vacant_cpus[worker_address] >= 0
        return InitializedCpu(worker_address, n_cpu)

    def release(self, worker_address, initialized_cpu):
        """release n_cpu CPUs on worker_address
        """
        self.worker_vacant_cpus[worker_address] += initialized_cpu.n_cpu
        self.worker_used_cpus[worker_address] -= initialized_cpu.n_cpu
        assert self.worker_used_cpus[worker_address] >= 0

    def get_vacant_cpu(self, worker_address):
        """Return vacant cpu number of a worker."""
        return self.worker_vacant_cpus[worker_address]

    def cpu_num(self):
        """"Return vacant cpu number."""
        return sum(self.worker_vacant_cpus.values())

    def get_total_cpu(self, worker_address):
        """Return total cpu number of a worker."""
        return self.worker_vacant_cpus[worker_address] + self.worker_used_cpus[worker_address]
