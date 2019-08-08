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

from collections import defaultdict
from parl.utils import to_str, to_byte, logger
from parl.remote import remote_constants
from parl.remote.job_center import JobCenter
import cloudpickle
import queue
import time


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
        job_center (JobCenter): A thread-safe data structure that stores the job address of vacant cpus.
        client_socket (zmq.Context.socket): A socket that receives submitted
                                           job from the client, and later sends
                                           job_address back to the client.
        cpu_num(int): The number of available CPUs in the cluster.
        worker_num(int): The number of workers connected to this cluster.

    Args:
        port: The ip port that the master node binds to.
    """

    def __init__(self, port):
        logger.set_dir(os.path.expanduser('~/.parl_data/master/'))
        self.ctx = zmq.Context()

        self.client_socket = self.ctx.socket(zmq.REP)
        self.client_socket.bind("tcp://*:{}".format(port))
        self.client_socket.linger = 0
        self.port = port

        self.job_center = JobCenter()

        self.master_is_alive = True

    def _create_worker_monitor(self, worker_heartbeat_address, worker_address):
        """When a new worker connects to the master, a socket is created to
        send heartbeat signals to the worker.
        """
        worker_heartbeat_socket = self.ctx.socket(zmq.REQ)
        worker_heartbeat_socket.linger = 0
        worker_heartbeat_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)
        worker_heartbeat_socket.connect("tcp://" + worker_heartbeat_address)

        connected = True
        while connected and self.master_is_alive:
            try:
                worker_heartbeat_socket.send_multipart(
                    [remote_constants.HEARTBEAT_TAG])
                _ = worker_heartbeat_socket.recv_multipart()
                time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)
            except zmq.error.Again as e:
                self.job_center.drop_worker(worker_address)
                logger.warning("\n[Master] Cannot connect to the worker " +
                               "{}. ".format(worker_address) +
                               "Worker_pool will drop this worker.")
                self._print_workers()
                connected = False
            except zmq.error.ZMQError as e:
                break

        worker_heartbeat_socket.close(0)
        logger.warning("Exit worker monitor from master.")

    def _create_client_monitor(self, client_heartbeat_address):
        """When a new client connects to the master, a socket is created to
        send heartbeat signals to the client.
        """

        client_heartbeat_socket = self.ctx.socket(zmq.REQ)
        client_heartbeat_socket.linger = 0
        client_heartbeat_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)
        client_heartbeat_socket.connect("tcp://" + client_heartbeat_address)

        client_is_alive = True
        while client_is_alive and self.master_is_alive:
            try:
                client_heartbeat_socket.send_multipart(
                    [remote_constants.HEARTBEAT_TAG])
                _ = client_heartbeat_socket.recv_multipart()
            except zmq.error.Again as e:
                client_is_alive = False
                logger.warning("[Master] cannot connect to the client " +
                               "{}. ".format(client_heartbeat_address) +
                               "Please check if it is still alive.")
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)
        logger.warning("Master exits client monitor for {}.\n".format(
            client_heartbeat_address))
        logger.info(
            "Master connects to {} workers and have {} vacant CPUs.\n".format(
                self.worker_num, self.cpu_num))
        client_heartbeat_socket.close(0)

    def _print_workers(self):
        """Display `worker_pool` infomation."""
        logger.info(
            "Master connects to {} workers and have {} vacant CPUs.\n".format(
                self.worker_num, self.cpu_num))

    @property
    def cpu_num(self):
        return self.job_center.cpu_num

    @property
    def worker_num(self):
        return self.job_center.worker_num

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

        elif tag == remote_constants.WORKER_INITIALIZED_TAG:

            initialized_worker = cloudpickle.loads(message[1])
            self.job_center.add_worker(initialized_worker)
            logger.info("A new worker {} is added, ".format(initialized_worker.
                                                            worker_address) +
                        "the cluster has {} CPUs.\n".format(self.cpu_num))

            # a thread for sending heartbeat signals to `worker.address`
            thread = threading.Thread(
                target=self._create_worker_monitor,
                args=(initialized_worker.master_heartbeat_address,
                      initialized_worker.worker_address))
            thread.start()

            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        # a client connects to the master
        elif tag == remote_constants.CLIENT_CONNECT_TAG:
            client_heartbeat_address = to_str(message[1])
            logger.info(
                "Client {} is connected.".format(client_heartbeat_address))

            thread = threading.Thread(
                target=self._create_client_monitor,
                args=(client_heartbeat_address, ))
            thread.start()
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        # a client submits a job to the master
        elif tag == remote_constants.CLIENT_SUBMIT_TAG:
            client_address = to_str(message[1])

            # check available CPU resources
            if self.cpu_num:
                logger.info("Submitting job...")
                job = self.job_center.request_job()
                self.client_socket.send_multipart([
                    remote_constants.NORMAL_TAG,
                    to_byte(job.job_address),
                    to_byte(job.client_heartbeat_address),
                    to_byte(job.ping_heartbeat_address)
                ])
                self._print_workers()
            else:
                self.client_socket.send_multipart([remote_constants.CPU_TAG])

        # a worker updates
        elif tag == remote_constants.NEW_JOB_TAG:
            initialized_job = cloudpickle.loads(message[1])
            last_job_address = to_str(message[2])

            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])
            self.job_center.update_job(last_job_address, initialized_job,
                                       initialized_job.worker_address)
            logger.info("A worker updated. cpu_num:{}".format(self.cpu_num))

            self._print_workers()

        # check before start a worker
        elif tag == remote_constants.NORMAL_TAG:
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        # a job reset itself
        elif tag == remote_constants.RESET_JOB_TAG:
            initialized_job = cloudpickle.loads(message[1])
            self.job_center.reset_job(initialized_job)

            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        else:
            raise NotImplementedError()

    def exit(self):
        """ Close the master.
        """
        self.master_is_alive = False

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
        self.client_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)

        while self.master_is_alive:
            try:
                self._receive_message()
                pass
            except zmq.error.Again as e:
                #detect whether `self.master_is_alive` is True periodically
                pass

        logger.warning("[Master] Exit master.")
