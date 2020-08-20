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


class InitializedJob(object):
    def __init__(self,
                 job_address,
                 worker_heartbeat_address,
                 client_heartbeat_address,
                 ping_heartbeat_address,
                 worker_address,
                 pid,
                 job_id=None,
                 log_server_address=None):
        """
    Args:
      job_address(str): Job address to which the new task connect.
      worker_heartbeat_address(str): Optional. The address to which the worker sends heartbeat signals.
      client_heartbeat_address(str): Address to which the client sends heartbeat signals.
      ping_heartbeat_address(str): the server address to which the client sends ping signals.
                                    The signal is used to check if the job is alive.
      worker_address(str): Worker's server address that receive command from the master.
      pid(int): Optional. Process id of the job.
      is_alive(True): Optional. This flag is used in worker to make sure that only alive jobs can be added into the worker_status.
    """
        self.job_address = job_address
        self.worker_heartbeat_address = worker_heartbeat_address
        self.client_heartbeat_address = client_heartbeat_address
        self.ping_heartbeat_address = ping_heartbeat_address
        self.worker_address = worker_address
        self.pid = pid
        self.is_alive = True
        self.job_id = job_id
        self.log_server_address = log_server_address


class InitializedWorker(object):
    def __init__(self, master_heartbeat_address, initialized_jobs, cpu_num,
                 hostname):
        """
    Args:
      worker_address(str): Worker server address that receives commands from the master.
      master_heartbeat_address(str): Address to which the worker send heartbeat signals to.
      initialized_jobs(list): A list of ``InitializedJob`` containing the information for initialized jobs.
      cpu_num(int): The number of CPUs used in this worker.
    """
        self.worker_address = master_heartbeat_address
        self.initialized_jobs = initialized_jobs
        self.cpu_num = cpu_num
        self.hostname = hostname
