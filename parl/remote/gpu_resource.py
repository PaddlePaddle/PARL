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
from parl.remote.message import AllocatedGpu


class GpuResource(object):
    """Record GPU information in each worker and respond to resource allocation of GPUs.

    Attributes:
        worker_vacant_gpus (dict): Record how many vacant gpus does each
                                   worker has.
        worker_used_gpus (dict): Record how many used gpus does each
                                   worker has.
    """

    def __init__(self):
        self.worker_vacant_gpus = defaultdict(list)
        self.worker_used_gpus = defaultdict(list)

    def add_gpu(self, worker_address, allocated_gpu):
        """add GPU resource to worker_vacant_gpus
        Args:
            allocated_gpu (AllocatedGpu): Record the GPU resource used in a job.
        """
        if allocated_gpu.gpu:
            self.worker_vacant_gpus[worker_address].extend(allocated_gpu.gpu.split(","))

    def remove_gpu(self, worker_address):
        """Remove gpus from GpuResource when a worker dies.
        """
        self.worker_vacant_gpus.pop(worker_address, None)
        self.worker_used_gpus.pop(worker_address, None)

    def filter(self, n_gpu):
        """filter workers that contain n_gpu GPUs at least
        """
        result = []
        for worker_address, vacant_gpus in self.worker_vacant_gpus.items():
            if len(vacant_gpus) >= n_gpu:
                result.append(worker_address)
        return result

    def allocate(self, worker_address, n_gpu):
        """allocate n_gpu GPUs from the worker
        """
        gpu = ",".join(self.worker_vacant_gpus[worker_address][0:n_gpu])
        self.worker_used_gpus[worker_address].extend(self.worker_vacant_gpus[worker_address][0:n_gpu])
        self.worker_vacant_gpus[worker_address] = self.worker_vacant_gpus[worker_address][n_gpu:]
        assert len(self.worker_vacant_gpus[worker_address]) >= 0
        return AllocatedGpu(worker_address, gpu)

    def recycle(self, worker_address, allocated_gpu):
        """recycle n_gpu GPUs to the worker
        """
        for gpu_id in allocated_gpu.gpu.split(","):
            self.worker_vacant_gpus[worker_address].append(gpu_id)
            if gpu_id in self.worker_used_gpus[worker_address]:
                self.worker_used_gpus[worker_address].remove(gpu_id)

    def get_vacant_gpu(self, worker_address):
        """Return the number of vacant GPUs in the worker."""
        return sum([len(gpu) for gpu in self.worker_vacant_gpus[worker_address]])

    def get_total_gpu(self, worker_address):
        """Return the number of GPUs of all the workers."""
        vacant_gpu = sum([len(gpu) for gpu in self.worker_vacant_gpus[worker_address]])
        used_gpu = sum([len(gpu) for gpu in self.worker_used_gpus[worker_address]])
        return vacant_gpu + used_gpu

    def gpu_num(self):
        """"Return the number vacant of GPUs of all the workers."""
        return sum([len(gpu) for gpu in self.worker_vacant_gpus.values()])
