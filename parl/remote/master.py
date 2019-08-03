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
        worker_pool (dict): A dict to store connected workers.
        job_pool (list): A list to store the job address of vacant cpu, when
                         this number is 0, the master will refuse to create
                         new remote object.
        client_job_dict (dict): A dict of list to record the job submitted by
                                each client.
        job_worker_dict (dict): A dict to record the job and related worker.
        client_socket (zmq.Context.socket): A socket which receives submitted
                                           job from the client, and later sends
                                           job_address back to the client.
        worker_socket (zmq.Context.socket): A socket which receives job
                                            addresses from the worker node.

    Args:
        port: the ip port that the master node binds to.
    """

    def __init__(self, port):
        logger.set_dir(os.path.expanduser('~/.parl_data/master/'))
        self.lock = threading.Lock()
        self.ctx = zmq.Context()

        self.client_socket = self.ctx.socket(zmq.REP)
        self.client_socket.bind("tcp://*:{}".format(port))
        self.client_socket.linger = 0
        self.port = port

        self.worker_pool = {}
        self.worker_locks = {}
        self.job_pool = []

        self.client_job_dict = defaultdict(list)
        self.worker_job_dict = defaultdict(list)
        self.job_worker_dict = {}

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
                for job in self.worker_job_dict[worker_address]:
                    if job in self.job_pool:
                        self.job_pool.remove(job)
                    self.job_worker_dict.pop(job)
                self.worker_job_dict.pop(worker_address)
                self.worker_pool.pop(worker_address)
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
        """when a new client connects to the master, a socket is created to
        send heartbeat signals to the client.
        """

        client_heartbeat_socket = self.ctx.socket(zmq.REQ)
        client_heartbeat_socket.linger = 0
        client_heartbeat_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)
        client_heartbeat_socket.connect("tcp://" + client_heartbeat_address)

        self.client_is_alive = True
        while self.client_is_alive and self.master_is_alive:
            try:
                client_heartbeat_socket.send_multipart(
                    [remote_constants.HEARTBEAT_TAG])
                _ = client_heartbeat_socket.recv_multipart()
            except zmq.error.Again as e:
                self.client_is_alive = False
                logger.warning("[Master] cannot connect to the client " +
                               "{}. ".format(client_heartbeat_address) +
                               "Please check if it is still alive.")
                self._kill_client_jobs(client_heartbeat_address)
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)
        logger.warning("Master exits client monitor for {}.\n".format(
            client_heartbeat_address))
        logger.info(
            "Master connects to {} workers and have {} vacant CPUs.\n".format(
                len(self.worker_pool), len(self.job_pool)))
        client_heartbeat_socket.close(0)

    def _kill_client_jobs(self, client_address):
        """set timeout in case the worker and client quit at the same time.
        """
        jobs = self.client_job_dict[client_address]

        for job_address in jobs:
            if job_address in self.job_worker_dict:
                worker_address = self.job_worker_dict[job_address]
                worker_socket = self.worker_pool[worker_address].worker_socket
                self.worker_locks[worker_address].acquire()
                worker_socket.send_multipart(
                    [remote_constants.KILLJOB_TAG,
                     to_byte(job_address)])
                try:
                    _ = worker_socket.recv_multipart()
                except zmq.error.Again as e:
                    logger.warning("Error in recv kill_client_job")
                self.worker_locks[worker_address].release()
                self.job_worker_dict.pop(job_address)
        self.client_job_dict.pop(client_address)

    def _print_workers(self):
        """Display `worker_pool` infomation."""
        logger.info(
            "Master connects to {} workers and have {} vacant CPUs.\n".format(
                len(self.worker_pool), len(self.job_pool)))

    def _receive_message(self):
        """master node will receive four types of message: (1) worker
        connection; (2) worker update; (3) client connection; (4) job
        submittion.
        """
        message = self.client_socket.recv_multipart()
        tag = message[0]

        # a new worker connects to the master
        if tag == remote_constants.WORKER_CONNECT_TAG:
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        elif tag == remote_constants.WORKER_INITIALIZED_TAG:
            worker = pickle.loads(message[1])
            worker_heartbeat_address = to_str(message[2])

            # maintain job & worker relations
            for job_address in worker.job_pool:
                self.job_worker_dict[job_address] = worker.address
            self.worker_job_dict[worker.address] = worker.job_pool
            self.job_pool.extend(worker.job_pool)

            # a new socket for submitting job to the worker
            worker_socket = self.ctx.socket(zmq.REQ)
            worker_socket.linger = 0
            worker_socket.setsockopt(zmq.RCVTIMEO, 10000)
            worker_socket.connect("tcp://{}".format(worker.address))
            worker.worker_socket = worker_socket
            self.worker_pool[worker.address] = worker
            self.worker_locks[worker.address] = threading.Lock()

            logger.info(
                "A new worker {} is added, ".format(worker.address) +
                "the cluster has {} CPUs.\n".format(len(self.job_pool)))

            # a thread for sending heartbeat signals to `worker.address`
            thread = threading.Thread(
                target=self._create_worker_monitor,
                args=(
                    worker_heartbeat_address,
                    worker.address,
                ))
            thread.setDaemon(True)
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
            thread.setDaemon(True)
            thread.start()
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        # a client submits a job to the master
        elif tag == remote_constants.CLIENT_SUBMIT_TAG:
            client_address = to_str(message[1])
            done_flag = False

            # check available CPU resources
            if len(self.job_pool):
                logger.info("Submitting job...")
                job_address = self.job_pool.pop(0)
                worker_address = self.job_worker_dict[job_address]
                self.worker_job_dict[worker_address].remove(job_address)
                self.client_socket.send_multipart(
                    [remote_constants.NORMAL_TAG,
                     to_byte(job_address)])
                self.client_job_dict[client_address].append(job_address)
                self._print_workers()
            else:
                self.client_socket.send_multipart([remote_constants.CPU_TAG])

        # a worker updates
        elif tag == remote_constants.NEW_JOB_TAG:
            worker_address = to_str(message[1])
            new_job_address = to_str(message[2])
            killed_job_address = to_str(message[3])

            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])
            logger.info("A worker updated.")

            if killed_job_address in self.job_worker_dict:
                self.job_worker_dict.pop(killed_job_address)
            if killed_job_address in self.worker_job_dict[worker_address]:
                self.worker_job_dict[worker_address].remove(killed_job_address)
            if killed_job_address in self.job_pool:
                self.job_pool.remove(killed_job_address)

            # add new job_address to job_pool
            self.job_pool.append(new_job_address)
            self.job_worker_dict[new_job_address] = worker_address
            self.worker_job_dict[worker_address].append(new_job_address)

            self._print_workers()

        # check before start a worker
        elif tag == remote_constants.NORMAL_TAG:
            self.client_socket.send_multipart([remote_constants.NORMAL_TAG])

        else:
            raise NotImplementedError()

    def exit(self):
        self.master_is_alive = False
        self.ctx.destroy()

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
        while self.master_is_alive:
            try:
                self._receive_message()
            except zmq.error.ContextTerminated as e:
                pass

        logger.warning("[Master] Exit master.")
