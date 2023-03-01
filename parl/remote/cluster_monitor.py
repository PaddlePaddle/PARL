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

import cloudpickle
import threading
from collections import defaultdict, deque
from parl.utils import to_str


class ClusterMonitor(object):
    """The client monitor watches the cluster status.

    Attributes:
        status (dict): A dict to store workers status and clients status.
    """

    def __init__(self):
        self.status = {
            'workers': defaultdict(dict),
            'clients': defaultdict(dict),
            'client_jobs': defaultdict(dict),
        }
        self.lock = threading.Lock()

    def add_worker_status(self, worker_address, hostname, total_cpus, total_gpus):
        """Record worker status when it is connected to the cluster.
        
        Args:
            worker_address (str): worker ip address
            hostname (str): worker hostname 
            total_cpus(int): the number of CPU in the worker
            total_gpus(int): the number of GPU in the worker
        """
        self.lock.acquire()
        worker_status = self.status['workers'][worker_address]
        worker_status['load_value'] = deque(maxlen=10)
        worker_status['load_time'] = deque(maxlen=10)
        worker_status['hostname'] = hostname
        worker_status['vacant_cpus'] = total_cpus
        worker_status['used_cpus'] = 0
        worker_status['vacant_gpus'] = total_gpus
        worker_status['used_gpus'] = 0
        self.lock.release()

    def add_client_job(self, client_id, job_info):
        self.lock.acquire()
        self.status['client_jobs'][client_id].update(job_info)
        self.lock.release()

    def update_client_status(self, client_address, client_status):
        """Update client status with message send from client heartbeat.
        
        Args:
            client_address (str): client ip address.
            client_status (dict): client status information
                                  (hostname, file_path, actor_num, elapsed_time).
        """
        self.lock.acquire()
        self.status['clients'][client_address] = client_status
        self.lock.release()

    def update_worker_status(self, update_status, worker_address, vacant_cpus, total_cpus, vacant_gpus, total_gpus):
        """Update a worker status.

        Args:
            update_status (dict): worker updated status information 
                                (vacant_memory, used_memory, load_time, load_value).
            worker_address (str): worker ip address.
            vacant_cpus (int): the number of available CPUs.
            total_cpus (int): total cpu number.
            vacant_gpus (int): the number of available GPUs.
            total_gpus (int): total gpu number.
        """
        self.lock.acquire()
        worker_status = self.status['workers'][worker_address]
        worker_status['vacant_memory'] = update_status['vacant_memory']
        worker_status['used_memory'] = update_status['used_memory']
        worker_status['vacant_gpu_memory'] = update_status['vacant_gpu_memory']
        worker_status['used_gpu_memory'] = update_status['used_gpu_memory']
        worker_status['load_time'].append(update_status['load_time'])
        worker_status['load_value'].append(update_status['load_value'])

        worker_status['vacant_cpus'] = vacant_cpus
        worker_status['used_cpus'] = total_cpus - vacant_cpus

        worker_status['vacant_gpus'] = vacant_gpus
        worker_status['used_gpus'] = total_gpus - vacant_gpus
        self.lock.release()

    def drop_worker_status(self, worker_address):
        """Drop worker status when it exits.

        Args:
            worker_address (str): IP address of the exited worker.
        """
        self.lock.acquire()
        self.status['workers'].pop(worker_address)
        self.lock.release()

    def drop_client_status(self, client_address):
        """Drop client status when it exits.

        Args:
            client_address (str): IP address of the exited client.
        """
        self.lock.acquire()
        if client_address in self.status['clients']:
            self.status['clients'].pop(client_address)
        self.lock.release()

    def get_status_info(self):
        """Return a message of current cluster status."""
        self.lock.acquire()
        worker_num = len(self.status['workers'])
        clients_num = len(self.status['clients'])
        used_cpus = 0
        vacant_cpus = 0
        used_gpus = 0
        vacant_gpus = 0
        for worker in self.status['workers'].values():
            used_cpus += worker.get('used_cpus', 0)
            vacant_cpus += worker.get('vacant_cpus', 0)
            used_gpus += worker.get('used_gpus', 0)
            vacant_gpus += worker.get('vacant_gpus', 0)
        self.lock.release()
        status_info = "has {} used cpus, {} vacant cpus, {} used_gpus, {} vacant_gpus.".format(
                used_cpus, vacant_cpus, used_gpus, vacant_gpus)
        return status_info

    def get_status(self):
        """Return a cloudpickled status."""
        self.lock.acquire()
        status = cloudpickle.dumps(self.status)
        self.lock.release()
        return status
