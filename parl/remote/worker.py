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
import multiprocessing
import os
import subprocess
import sys
import time
import threading
import zmq

from parl.utils import get_ip_address, to_byte, to_str, logger
from parl.remote import remote_constants


class WorkerInfo(object):
    """A WorkerInfo object records the computation resources of a worker.
    """
    def __init__(self, address, cpu_num, job_pool):
        self.address = address
        self.cpu_num = cpu_num
        self.job_pool = job_pool
        self.worker_socket = None


class Worker(object):
    """Worker provides the cpu computation resources for the cluster.

    A worker node is connected to the master node and will send its
    computation resources information to the master node. When a worker
    node is created, it will start `cpu_num` empty jobs and these jobs'
    ip addresses will be send to the master node. Further, when an old
    job is killed, worker will start a new job and send the new job ip
    address to the master node.

    To start a worker, we use the following xparl command line api:

    .. code-block:: python

        xparl connect --address localhost:1234 --cpu_num 8

    Attributes:
        job_pid (dict): A dict of subprocess id and its address.
        master_address (str): Master's ip address.
        request_master_socket (zmq.Context.socket): A socket which sends job
                                                    address to the master node.
        reply_master_socket (zmq.Context.socket): A socket which accepts
                                                  submitted job from master
                                                  node.
        reply_job_socket (zmq.Context.socket): A socket which receives
                                               job_address from the job.
    Args:
        master_address (str): IP address of the master node.
        cpu_num (int): Number of cpu to be used on the worker.
    """

    def __init__(self, master_address, cpu_num=None):
        os.makedirs(os.path.expanduser('~/.parl_data/worker/'), exist_ok=True)
        self.lock = threading.Lock()
        self.heartbeat_socket_initialized = threading.Event()
        self.ctx = zmq.Context.instance()
        self.job_pid = {}
        self.master_address = master_address
        self.master_is_alive = True
        self.worker_is_alive = True
        self._set_cpu_num(cpu_num)
        self._create_sockets()
        self._create_worker()

    def _set_cpu_num(self, cpu_num=None):
        """set useable cpu number for worker"""
        if cpu_num is not None:
            assert isinstance(cpu_num, int), "cpu_num is expected to be an" +\
                "INT，but you have passed a(n) {}".format(type(cpu_num))
            self.cpu_num = cpu_num
        else:
            self.cpu_num = multiprocessing.cpu_count()

    def _create_sockets(self):
        """ Each worker has three sockets at start:

        (1) request_master_socket: sends job address to master node.
        (2) reply_master_socket: accepts submitted job from master node.
        (3) reply_job_socket: receives job_address from subprocess.

        When a job is start, a new heartbeat socket is created to receive
        heartbeat signal from the job.

        """

        # request_master_socket: sends job address to master
        self.request_master_socket = self.ctx.socket(zmq.REQ)
        self.request_master_socket.linger = 0

        # wait for 0.5 second to check whether master is started
        self.request_master_socket.setsockopt(zmq.RCVTIMEO, 500)
        self.request_master_socket.connect("tcp://" + self.master_address)

        # reply_master_socket: receives submitted job from master
        self.reply_master_socket = self.ctx.socket(zmq.REP)
        self.reply_master_socket.linger = 0
        self.worker_ip = get_ip_address()
        reply_master_port = self.reply_master_socket.bind_to_random_port(
            "tcp://*")
        self.reply_master_address = "{}:{}".format(self.worker_ip,
                                                   reply_master_port)

        # reply_job_socket: receives job_address from subprocess
        self.reply_job_socket = self.ctx.socket(zmq.REP)
        self.reply_job_socket.linger = 0
        reply_job_port = self.reply_job_socket.bind_to_random_port("tcp://*")
        self.reply_job_address = "{}:{}".format(self.worker_ip, reply_job_port)

    def _create_worker(self):
        """create a WorkerInfo instance and send it to the master."""
        try:
            self.request_master_socket.send_multipart([
                remote_constants.WORKER_CONNECT_TAG])
            _ = self.request_master_socket.recv_multipart()
        except zmq.error.Again as e:
            logger.error("Can not connect to the master, "
                         "please check if master is started.")
            self.master_is_alive = False
            return

        self._init_jobs()
        self.request_master_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S*1000)

        self.worker = WorkerInfo(self.reply_master_address, self.cpu_num,
                                 list(self.job_pid.keys()))
        reply_thread = threading.Thread(
            target=self._reply_heartbeat,
            args=("master {}".format(self.master_address),),
            daemon=True)
        reply_thread.start()
        self.heartbeat_socket_initialized.wait()

        self.request_master_socket.send_multipart([
            remote_constants.WORKER_INITIALIZED_TAG,
            cloudpickle.dumps(self.worker),
            to_byte(self.heartbeat_master_address)])
        _ = self.request_master_socket.recv_multipart()

    def _init_job(self):
        """Create one job."""
        command = ["python", "{}/job.py".format(__file__[:-10]),
                   "--worker_address", self.reply_job_address]

        with open(os.devnull, "w") as null:
            pid = subprocess.Popen(command, stdout=null, stderr=null)

        self.lock.acquire()
        job_message = self.reply_job_socket.recv_multipart()
        self.reply_job_socket.send_multipart([remote_constants.NORMAL_TAG])
        job_address = to_str(job_message[1])
        heartbeat_job_address = to_str(job_message[2])
        self.job_pid[job_address] = pid
        self.lock.release()

        # a thread for sending heartbeat signals to job
        thread = threading.Thread(
            target=self._create_job_monitor,
            args=(job_address, heartbeat_job_address,),
            daemon=True)
        thread.start()
        return job_address

    def _init_jobs(self):
        """Create cpu_num jobs when the worker is created."""
        job_threads = []
        for _ in range(self.cpu_num):
            t = threading.Thread(target=self._init_job, daemon=True)
            t.start()
            job_threads.append(t)
        for th in job_threads:
            th.join()

    def _kill_job(self, job_address):
        """kill problematic job process and update worker information"""
        if job_address in self.job_pid:
            self.job_pid[job_address].kill()
            self.job_pid.pop(job_address)
            logger.warning("Worker kills job process {},".format(job_address))

            # When a old job is killed, the worker will create a new job.
            if self.master_is_alive:
                new_job_address = self._init_job()

                self.lock.acquire()
                self.request_master_socket.send_multipart([
                    remote_constants.NEW_JOB_TAG,
                    to_byte(self.reply_master_address),
                    to_byte(new_job_address),
                    to_byte(job_address)])
                _ = self.request_master_socket.recv_multipart()
                self.lock.release()

    def _create_job_monitor(self, job_address, heartbeat_job_address):
        """Sending heartbeat signals to check target's status"""

        # job_heartbeat_socket: sends heartbeat signal to job
        job_heartbeat_socket = self.ctx.socket(zmq.REQ)
        job_heartbeat_socket.linger = 0
        job_heartbeat_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S*1000)
        job_heartbeat_socket.connect("tcp://" + heartbeat_job_address)

        job_is_alive = True
        while job_is_alive and self.master_is_alive:
            try:
                job_heartbeat_socket.send_multipart([
                    remote_constants.HEARTBEAT_TAG])
                _ = job_heartbeat_socket.recv_multipart()
                time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

            except zmq.error.Again as e:
                job_is_alive = False
                if job_address in self.job_pid:
                    logger.warning("[Worker] No heartbeat reply from the job, "
                                   "will kill {}.".format(job_address))
                    self._kill_job(job_address)

            except zmq.error.ZMQError as e:
                break

        job_heartbeat_socket.close(0)

    def _reply_heartbeat(self, target):
        """Worker will kill its jobs when it lost connection with the master.
        """

        socket = self.ctx.socket(zmq.REP)
        socket.linger = 0
        socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_RCVTIMEO_S*1000)
        heartbeat_master_port =\
            socket.bind_to_random_port("tcp://*")
        self.heartbeat_master_address = "{}:{}".format(self.worker_ip,
                                                       heartbeat_master_port)
        self.heartbeat_socket_initialized.set()
        logger.set_dir(os.path.expanduser('~/.parl_data/worker/{}'.format(
            self.heartbeat_master_address)))
        logger.info("[Worker] Connect to the master node successfully. "
                    "({} CPUs)".format(self.cpu_num))
        while self.master_is_alive:
            try:
                message = socket.recv_multipart()
                socket.send_multipart([remote_constants.HEARTBEAT_TAG])
            except zmq.error.Again as e:
                self.master_is_alive = False
                for job_address in list(self.job_pid.keys()):
                    self._kill_job(job_address)
            except zmq.error.ContextTerminated as e:
                break
        socket.close(0)
        logger.warning("Worker exit replying heartbeat for master.")
        if self.worker_is_alive:
            self.exit()

    def exit(self):
        """Exit all zmq sockets related to the worker."""
        self.worker_is_alive = False
        self.ctx.destroy()

    def run(self):
        """An infinite loop waiting for killing job commands from
        the mater node.

        After creating `cpu_num` jobs and sending job addresses to the master
        node, a worker will keep waiting for killing job commands from master
        node to release computation resources occupied by a dead client. Then
        the worker will kill the jobs related to the dead client and create
        new jobs and update job addresses to the master node.
        """

        while self.master_is_alive and self.worker_is_alive:
            try:
                message = self.reply_master_socket.recv_multipart()
                tag = message[0]

                if tag == remote_constants.KILLJOB_TAG:
                    job_address = to_str(message[1])
                    self.reply_master_socket.send_multipart([
                        remote_constants.NORMAL_TAG])
                    self._kill_job(job_address)

                else:
                    raise NotImplementedError
            except zmq.error.ZMQError as e:
                self.worker_is_alive = False

        logger.warning("[Worker] Exit Worker {}.".format(
            self.reply_master_address))
