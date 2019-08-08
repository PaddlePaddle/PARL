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
import signal
import subprocess
import sys
import time
import threading
import warnings
import zmq

from parl.utils import get_ip_address, to_byte, to_str, logger
from parl.remote import remote_constants
from parl.remote.message import InitializedWorker

if sys.version_info.major == 3:
    warnings.simplefilter("ignore", ResourceWarning)


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
        self.lock = threading.Lock()
        self.heartbeat_socket_initialized = threading.Event()
        self.ctx = zmq.Context.instance()
        self.job_pid = {}
        self.master_address = master_address
        self.master_is_alive = True
        self.worker_is_alive = True
        self.lock = threading.Lock()
        self._set_cpu_num(cpu_num)
        self._create_sockets()
        self._create_worker()

    def _set_cpu_num(self, cpu_num=None):
        """set useable cpu number for worker"""
        if cpu_num is not None:
            assert isinstance(
                cpu_num, int
            ), "cpu_num should be INT type, please check the input type."
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
        logger.set_dir(
            os.path.expanduser('~/.parl_data/worker/{}'.format(
                self.reply_master_address)))
        # reply_job_socket: receives job_address from subprocess
        self.reply_job_socket = self.ctx.socket(zmq.REP)
        self.reply_job_socket.linger = 0
        reply_job_port = self.reply_job_socket.bind_to_random_port("tcp://*")
        self.reply_job_address = "{}:{}".format(self.worker_ip, reply_job_port)

    def _create_worker(self):
        """Create jobs and send a instance of ``InitializedWorker`` that contains the worker information to the master."""
        try:
            self.request_master_socket.send_multipart(
                [remote_constants.WORKER_CONNECT_TAG])
            _ = self.request_master_socket.recv_multipart()
        except zmq.error.Again as e:
            logger.error("Can not connect to the master, "
                         "please check if master is started.")
            self.master_is_alive = False
            return

        initialized_jobs = self._init_jobs(job_num=self.cpu_num)
        self.request_master_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)

        self.reply_thread = threading.Thread(
            target=self._reply_heartbeat,
            args=("master {}".format(self.master_address), ))
        self.reply_thread.start()
        self.heartbeat_socket_initialized.wait()
        initialized_worker = InitializedWorker(self.reply_master_address,
                                               self.master_heartbeat_address,
                                               initialized_jobs, self.cpu_num)

        self.request_master_socket.send_multipart([
            remote_constants.WORKER_INITIALIZED_TAG,
            cloudpickle.dumps(initialized_worker)
        ])
        _ = self.request_master_socket.recv_multipart()

    def _init_jobs(self, job_num):
        """Create jobs.

        Args:
            job_num(int): the number of jobs to create.
        """
        job_file = __file__.replace('worker.pyc', 'job.py')
        job_file = job_file.replace('worker.py', 'job.py')
        command = [
            sys.executable, job_file, "--worker_address",
            self.reply_job_address, "--master_address", self.master_address
        ]

        # avoid that many jobs are killed and restarted at the same time.
        self.lock.acquire()

        # Redirect the output to DEVNULL
        FNULL = open(os.devnull, 'w')
        for _ in range(job_num):
            subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT)
        FNULL.close()

        new_jobs = []
        for _ in range(job_num):
            job_message = self.reply_job_socket.recv_multipart()
            self.reply_job_socket.send_multipart([remote_constants.NORMAL_TAG])
            initialized_job = cloudpickle.loads(job_message[1])
            self.job_pid[initialized_job.job_address] = int(
                initialized_job.pid)
            new_jobs.append(initialized_job)

            # a thread for sending heartbeat signals to job
            thread = threading.Thread(
                target=self._create_job_monitor,
                args=(
                    initialized_job.job_address,
                    initialized_job.worker_heartbeat_address,
                ))
            thread.start()
        self.lock.release()
        assert len(new_jobs) > 0, "init jobs failed"
        return new_jobs

    def _kill_job(self, job_address):
        """kill problematic job process and update worker information"""
        if job_address in self.job_pid:
            self.lock.acquire()
            pid = self.job_pid[job_address]
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                logger.warn("job:{} has been killed before".format(pid))
            self.job_pid.pop(job_address)
            logger.warning("Worker kills job process {},".format(job_address))
            self.lock.release()

            # When a job ends exceptionally, the worker will create a new job.
            if self.master_is_alive:
                initialized_job = self._init_jobs(job_num=1)[0]
                initialized_job.worker_address = self.reply_master_address

                self.lock.acquire()
                self.request_master_socket.send_multipart([
                    remote_constants.NEW_JOB_TAG,
                    cloudpickle.dumps(initialized_job),
                    to_byte(job_address)
                ])
                _ = self.request_master_socket.recv_multipart()
                self.lock.release()

    def _create_job_monitor(self, job_address, heartbeat_job_address):
        """Send heartbeat signals to check target's status"""

        # job_heartbeat_socket: sends heartbeat signal to job
        job_heartbeat_socket = self.ctx.socket(zmq.REQ)
        job_heartbeat_socket.linger = 0
        job_heartbeat_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)
        job_heartbeat_socket.connect("tcp://" + heartbeat_job_address)

        job_is_alive = True
        while job_is_alive and self.master_is_alive and self.worker_is_alive:
            try:
                job_heartbeat_socket.send_multipart(
                    [remote_constants.HEARTBEAT_TAG])
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
        socket.setsockopt(zmq.RCVTIMEO,
                          remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)
        heartbeat_master_port =\
            socket.bind_to_random_port("tcp://*")
        self.master_heartbeat_address = "{}:{}".format(self.worker_ip,
                                                       heartbeat_master_port)
        self.heartbeat_socket_initialized.set()
        logger.info("[Worker] Connect to the master node successfully. "
                    "({} CPUs)".format(self.cpu_num))
        while self.master_is_alive and self.worker_is_alive:
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
        logger.warning(
            "[Worker] lost connection with the master, will exit replying heartbeat for master."
        )
        # exit the worker
        self.worker_is_alive = False

    def exit(self):
        """close the worker"""
        self.worker_is_alive = False

    def run(self):
        """Keep running until it lost connection with the master.
        """
        self.reply_thread.join()
