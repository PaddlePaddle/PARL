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

import os
import pickle
import threading
import time
import zmq
from collections import deque, defaultdict
import parl
import sys
from parl.utils import to_str, to_byte, logger, get_ip_address
from parl.remote import remote_constants
from parl.remote.worker_manager import WorkerManager
from parl.remote.cluster_monitor import ClusterMonitor
from parl.remote.grpc_heartbeat import HeartbeatClientThread
import cloudpickle
import time
from parl.remote.utils import get_version


class Master(object):
    """Base class for a master node, the control center for our cluster, which provides connections to workers and clients.

    There is only one master node in each cluster, and it is responsible for
    receiving jobs from the clients and allocating computation resources to
    run the jobs.

    To start a master node, we use the following xparl command line api:

    .. code-block:: python

        xparl start --port localhost:1234

    At the same time, a local worker will be started and connect to the
    master node.

    Attributes:
        worker_manager (WorkerManager): A thread-safe data structure used to 
                                        respond to resource request, maintaining
                                        the available computation resources in
                                        each worker.
        client_socket (zmq.Context.socket): A socket that receives submitted
                                           job from the client, and later sends
                                           job_address back to the client.
        master_ip(str): The ip address of the master node.
        cpu_num(int): The number of available CPUs in the cluster.
        worker_num(int): The number of workers connected to this cluster.
        cluster_monitor(dict): A dict to record worker status and client status.
        client_hostname(dict): A dict to store hostname for each client address.

    Args:
        port: The ip port that the master node binds to.
    """

    def __init__(self, port, monitor_port=None, device=remote_constants.CPU):
        self.ctx = zmq.Context()
        self.master_ip = get_ip_address()
        self.all_client_heartbeat_threads = []
        self.all_worker_heartbeat_threads = []
        self.monitor_url = "http://{}:{}".format(self.master_ip, monitor_port)
        logger.set_dir(os.path.expanduser('~/.parl_data/master/{}_{}'.format(self.master_ip, port)))
        self.client_socket = self.ctx.socket(zmq.REP)
        self.client_socket.bind("tcp://*:{}".format(port))
        self.client_socket.linger = 0
        self.port = port
        self.device = device

        self.worker_manager = WorkerManager(self.master_ip, [self.device])
        self.cluster_monitor = ClusterMonitor()
        self.master_is_alive = True
        self.client_hostname = defaultdict(int)

    def _get_status(self):
        return self.cluster_monitor.get_status()

    def _print_workers(self):
        """Display `worker_pool` infomation."""
        if self.device == remote_constants.CPU:
            logger.info("Master connects to {} workers and have {} vacant CPUs.\n".format(
                self.worker_num, self.cpu_num))
        else:
            logger.info("Master connects to {} workers and have {} vacant GPUs.\n".format(
                self.worker_num, self.gpu_num))

    @property
    def cpu_num(self):
        return self.worker_manager.cpu_num

    @property
    def gpu_num(self):
        return self.worker_manager.gpu_num

    @property
    def worker_num(self):
        return self.worker_manager.worker_num

    def _receive_message(self):
        """Master node will receive various types of message: (1) worker
        connection; (2) worker update; (3) client connection; (4) job
        submittion; (5) reset job.
        """
        message = self.client_socket.recv_multipart()
        tag = message[0]

        # a new worker connects to the master
        if tag == remote_constants.WORKER_CONNECT_TAG:
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        elif tag == remote_constants.MONITOR_TAG:
            status = self._get_status()
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG, status])

        # `xparl status` command line API
        elif tag == remote_constants.STATUS_TAG:
            status_info = self.cluster_monitor.get_status_info()
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG, to_byte(status_info)])

        elif tag == remote_constants.WORKER_INITIALIZED_TAG:
            initialized_worker = cloudpickle.loads(message[1])
            worker_address = initialized_worker.worker_address
            success = self.worker_manager.add_worker(initialized_worker)
            if not success:
                if self.device == remote_constants.GPU:
                    logger.error("GPU cluster is not allowed a CPU worker to join in")
                    self.client_socket.send_multipart([remote_constants.REJECT_CPU_WORKER_TAG])
                else:
                    logger.error("CPU cluster is not allowed a GPU worker to join in")
                    self.client_socket.send_multipart([remote_constants.REJECT_GPU_WORKER_TAG])
            else:
                hostname = self.worker_manager.get_hostname(worker_address)
                total_cpus = self.worker_manager.get_total_cpu(worker_address)
                total_gpus = self.worker_manager.get_total_gpu(worker_address)
                self.cluster_monitor.add_worker_status(worker_address, hostname, total_cpus, total_gpus)
                logger.info("A new worker {} is added, ".format(worker_address) +
                            "the cluster has {} CPUs, has {} GPUs.\n".format(self.cpu_num, self.gpu_num))

                def heartbeat_exit_callback_func(worker_address):
                    self.worker_manager.remove_worker(worker_address)
                    self.cluster_monitor.drop_worker_status(worker_address)
                    logger.warning("\n[Master] Cannot connect to the worker " + "{}. ".format(worker_address) +
                                   "Worker_pool will drop this worker.")
                    self._print_workers()
                    logger.warning("Exit worker monitor from master.")

                # a thread for sending heartbeat signals to the client
                thread = HeartbeatClientThread(
                    worker_address,
                    heartbeat_exit_callback_func=heartbeat_exit_callback_func,
                    exit_func_args=(worker_address, ))
                self.all_worker_heartbeat_threads.append(thread)
                thread.setDaemon(True)
                thread.start()

                self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        # a client connects to the master
        elif tag == remote_constants.CLIENT_CONNECT_TAG:
            # `client_heartbeat_address` is the
            #      `reply_master_heartbeat_address` of the client

            client_heartbeat_address = to_str(message[1])
            client_hostname = to_str(message[2])
            client_id = to_str(message[3])
            self.client_hostname[client_heartbeat_address] = client_hostname
            logger.info("Client {} is connected.".format(client_heartbeat_address))

            def heartbeat_exit_callback_func(client_heartbeat_address):
                self.cluster_monitor.drop_client_status(client_heartbeat_address)
                logger.warning("[Master] cannot connect to the client " + "{}. ".format(client_heartbeat_address) +
                               "Please check if it is still alive.")
                logger.info("Master connects to {} workers and have {} vacant CPUs, {} vacant GPUs.\n".format(
                    self.worker_num, self.cpu_num, self.gpu_num))

            # a thread for sending heartbeat signals to the client
            thread = HeartbeatClientThread(
                client_heartbeat_address,
                heartbeat_exit_callback_func=heartbeat_exit_callback_func,
                exit_func_args=(client_heartbeat_address, ))
            self.all_client_heartbeat_threads.append(thread)
            thread.setDaemon(True)
            thread.start()

            log_monitor_address = "{}/logs?client_id={}".format(self.monitor_url, client_id)
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG, to_byte(log_monitor_address)])

        elif tag == remote_constants.CHECK_VERSION_TAG:
            self.client_socket.send_multipart([
                remote_constants.NORMAL_TAG,
                to_byte(parl.__version__),
                to_byte(str(sys.version_info.major)),
                to_byte(str(sys.version_info.minor)),
                to_byte(str(get_version('pyarrow')))
            ])

        # a client submits a job to the master
        elif tag == remote_constants.CLIENT_SUBMIT_TAG:
            # check available CPU/GPU resources
            n_cpu = int(message[3])
            n_gpu = int(message[4])
            if self.device == remote_constants.CPU and n_gpu > 0:
                self.client_socket.send_multipart([remote_constants.REJECT_GPU_JOB_TAG])
            elif self.device == remote_constants.GPU and n_cpu > 0:
                self.client_socket.send_multipart([remote_constants.REJECT_CPU_JOB_TAG])
            elif n_gpu > remote_constants.MAX_N_GPU:
                self.client_socket.send_multipart([remote_constants.REJECT_INVALID_GPU_JOB_TAG])
            else:
                logger.info("Submitting job...")
                job_info = self.worker_manager.request_job(n_cpu=n_cpu, n_gpu=n_gpu)
                if job_info:
                    self.client_socket.send_multipart([remote_constants.NORMAL_TAG, cloudpickle.dumps(job_info)])
                    client_id = to_str(message[2])
                    self.cluster_monitor.add_client_job(client_id, {job_info.job_id: job_info.log_server_address})
                    self._print_workers()
                else:
                    if n_gpu > 0:
                        self.client_socket.send_multipart([remote_constants.GPU_TAG])
                    else:
                        self.client_socket.send_multipart([remote_constants.CPU_TAG])

        # a worker updates
        elif tag == remote_constants.NEW_JOB_TAG:
            initialized_job = cloudpickle.loads(message[1])
            last_job_address = to_str(message[2])

            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])
            self.worker_manager.update_job(last_job_address, initialized_job, initialized_job.worker_address)
            if self.device == remote_constants.CPU:
                logger.info("A worker updated. cpu_num:{}".format(self.cpu_num))
            else:
                logger.info("A worker updated. gpu_num:{}".format(self.gpu_num))

            self._print_workers()

        # client update status periodically
        elif tag == remote_constants.CLIENT_STATUS_UPDATE_TAG:
            client_heartbeat_address = to_str(message[1])
            client_status = cloudpickle.loads(message[2])

            client_status['client_hostname'] = self.client_hostname[client_heartbeat_address]
            self.cluster_monitor.update_client_status(client_heartbeat_address, client_status)
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        # worker update status periodically
        elif tag == remote_constants.WORKER_STATUS_UPDATE_TAG:
            worker_address = to_str(message[1])
            worker_status = cloudpickle.loads(message[2])

            vacant_cpus = self.worker_manager.get_vacant_cpu(worker_address)
            total_cpus = self.worker_manager.get_total_cpu(worker_address)
            vacant_gpus = self.worker_manager.get_vacant_gpu(worker_address)
            total_gpus = self.worker_manager.get_total_gpu(worker_address)
            self.cluster_monitor.update_worker_status(worker_status, worker_address, vacant_cpus, total_cpus,
                                                      vacant_gpus, total_gpus)

            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        # check before start a worker
        elif tag == remote_constants.NORMAL_TAG:
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        else:
            raise NotImplementedError()

    def exit(self):
        """ Close the master.
        """
        self.master_is_alive = False

        for thread in self.all_client_heartbeat_threads:
            if thread.is_alive():
                thread.exit()

        for thread in self.all_worker_heartbeat_threads:
            if thread.is_alive():
                thread.exit()

    def run(self):
        """An infinite loop waiting for messages from the workers and
        clients.

        Master node will receive four types of messages:

        1. A new worker connects to the master node.
        2. A connected worker sending new job address after it kills an old
           job.
        3. A new client connects to the master node.
        4. A connected client submits a job after a remote object is created.
        """
        self.client_socket.linger = 0
        self.client_socket.setsockopt(zmq.RCVTIMEO, remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)

        while self.master_is_alive:
            try:
                self._receive_message()
                pass
            except zmq.error.Again as e:
                #detect whether `self.master_is_alive` is True periodically
                pass

        logger.warning("[Master] Exit master.")
