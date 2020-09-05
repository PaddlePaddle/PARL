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
import psutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import threading
import warnings
import zmq
from datetime import datetime
import parl
from parl.utils import get_ip_address, to_byte, to_str, logger, _IS_WINDOWS, kill_process
from parl.remote import remote_constants
from parl.remote.message import InitializedWorker
from parl.remote.status import WorkerStatus
from six.moves import queue


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
        master_address (str): Master's ip address.
        request_master_socket (zmq.Context.socket): A socket which sends job
                                                    address to the master node.
        reply_job_socket (zmq.Context.socket): A socket which receives
                                               job_address from the job.
        kill_job_socket (zmq.Context.socket): A socket that receives commands to kill the job from jobs.
        job_buffer (str): A buffer that stores initialized jobs for providing new jobs in a short time.

    Args:
        master_address (str): IP address of the master node.
        cpu_num (int): Number of cpu to be used on the worker.
    """

    def __init__(self, master_address, cpu_num=None, log_server_port=None):
        self.lock = threading.Lock()
        self.heartbeat_socket_initialized = threading.Event()
        self.ctx = zmq.Context.instance()
        self.master_address = master_address
        self.master_is_alive = True
        self.worker_is_alive = True
        self.worker_status = None  # initialized at `self._create_jobs`
        self._set_cpu_num(cpu_num)
        self.job_buffer = queue.Queue(maxsize=self.cpu_num)
        self._create_sockets()
        self.check_version()
        # create log server
        self.log_server_proc, self.log_server_address = self._create_log_server(
            port=log_server_port)

        # create a thread that waits commands from the job to kill the job.
        self.kill_job_thread = threading.Thread(target=self._reply_kill_job)
        self.kill_job_thread.setDaemon(True)
        self.kill_job_thread.start()

        self._create_jobs()

        # create a thread that initializes jobs and adds them into the job_buffer
        job_thread = threading.Thread(target=self._fill_job_buffer)
        job_thread.setDaemon(True)
        job_thread.start()

    def _set_cpu_num(self, cpu_num=None):
        """set useable cpu number for worker"""
        if cpu_num is not None:
            assert isinstance(
                cpu_num, int
            ), "cpu_num should be INT type, please check the input type."
            self.cpu_num = cpu_num
        else:
            self.cpu_num = multiprocessing.cpu_count()

    def check_version(self):
        '''Verify that the parl & python version in 'worker' process matches that of the 'master' process'''
        self.request_master_socket.send_multipart(
            [remote_constants.CHECK_VERSION_TAG])
        message = self.request_master_socket.recv_multipart()
        tag = message[0]
        if tag == remote_constants.NORMAL_TAG:
            worker_parl_version = parl.__version__
            worker_python_version = str(sys.version_info.major)
            assert worker_parl_version == to_str(message[1]) and worker_python_version == to_str(message[2]),\
                '''Version mismatch: the "master" is of version "parl={}, python={}". However, 
                "parl={}, python={}"is provided in your environment.'''.format(
                        to_str(message[1]), to_str(message[2]),
                        worker_parl_version, worker_python_version
                    )
        else:
            raise NotImplementedError

    def _create_sockets(self):
        """ Each worker has three sockets at start:

        (1) request_master_socket: sends job address to master node.
        (2) reply_job_socket: receives job_address from subprocess.
        (3) kill_job_socket : receives commands to kill the job from jobs.

        When a job starts, a new heartbeat socket is created to receive
        heartbeat signals from the job.

        """
        self.worker_ip = get_ip_address()

        # request_master_socket: sends job address to master
        self.request_master_socket = self.ctx.socket(zmq.REQ)
        self.request_master_socket.linger = 0

        # wait for 0.5 second to check whether master is started
        self.request_master_socket.setsockopt(zmq.RCVTIMEO, 500)
        self.request_master_socket.connect("tcp://" + self.master_address)

        # reply_job_socket: receives job_address from subprocess
        self.reply_job_socket = self.ctx.socket(zmq.REP)
        self.reply_job_socket.linger = 0
        reply_job_port = self.reply_job_socket.bind_to_random_port("tcp://*")
        self.reply_job_address = "{}:{}".format(self.worker_ip, reply_job_port)

        # kill_job_socket
        self.kill_job_socket = self.ctx.socket(zmq.REP)
        self.kill_job_socket.linger = 0
        kill_job_port = self.kill_job_socket.bind_to_random_port("tcp://*")
        self.kill_job_address = "{}:{}".format(self.worker_ip, kill_job_port)

    def _create_jobs(self):
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

        self.reply_master_hearbeat_thread = threading.Thread(
            target=self._reply_heartbeat,
            args=("master {}".format(self.master_address), ))
        self.reply_master_hearbeat_thread.start()
        self.heartbeat_socket_initialized.wait()

        for job in initialized_jobs:
            job.worker_address = self.master_heartbeat_address

        initialized_worker = InitializedWorker(self.master_heartbeat_address,
                                               initialized_jobs, self.cpu_num,
                                               socket.gethostname())
        self.request_master_socket.send_multipart([
            remote_constants.WORKER_INITIALIZED_TAG,
            cloudpickle.dumps(initialized_worker)
        ])

        _ = self.request_master_socket.recv_multipart()
        self.worker_status = WorkerStatus(self.master_heartbeat_address,
                                          initialized_jobs, self.cpu_num)

    def _fill_job_buffer(self):
        """An endless loop that adds initialized job into the job buffer"""
        initialized_jobs = []
        while self.worker_is_alive:
            if self.job_buffer.full() is False:
                job_num = self.cpu_num - self.job_buffer.qsize()
                if job_num > 0:
                    initialized_jobs = self._init_jobs(job_num=job_num)
                    for job in initialized_jobs:
                        self.job_buffer.put(job)

            time.sleep(0.02)
        self.exit()

    def _init_jobs(self, job_num):
        """Create jobs.

        Args:
            job_num(int): the number of jobs to create.
        """
        job_file = __file__.replace('worker.pyc', 'job.py')
        job_file = job_file.replace('worker.py', 'job.py')
        command = [
            sys.executable, job_file, "--worker_address",
            self.reply_job_address, "--log_server_address",
            self.log_server_address
        ]

        if sys.version_info.major == 3:
            warnings.simplefilter("ignore", ResourceWarning)

        # avoid that many jobs are killed and restarted at the same time.
        self.lock.acquire()

        # Redirect the output to DEVNULL
        FNULL = open(os.devnull, 'w')
        for _ in range(job_num):
            subprocess.Popen(
                command,
                stdout=FNULL,
                stderr=subprocess.STDOUT,
                close_fds=True)
        FNULL.close()

        new_jobs = []
        for _ in range(job_num):
            job_message = self.reply_job_socket.recv_multipart()
            self.reply_job_socket.send_multipart(
                [remote_constants.NORMAL_TAG,
                 to_byte(self.kill_job_address)])
            initialized_job = cloudpickle.loads(job_message[1])
            new_jobs.append(initialized_job)

            # a thread for sending heartbeat signals to job
            thread = threading.Thread(
                target=self._create_job_monitor, args=(initialized_job, ))
            thread.setDaemon(True)
            thread.start()
        self.lock.release()
        assert len(new_jobs) > 0, "init jobs failed"
        return new_jobs

    def _kill_job(self, job_address):
        """Kill a job process and update worker information"""
        success = self.worker_status.remove_job(job_address)
        if success:
            while True:
                initialized_job = self.job_buffer.get()
                initialized_job.worker_address = self.master_heartbeat_address
                if initialized_job.is_alive:
                    self.worker_status.add_job(initialized_job)
                    if not initialized_job.is_alive:  # make sure that the job is still alive.
                        self.worker_status.remove_job(
                            initialized_job.job_address)
                        continue
                else:
                    logger.warning(
                        "[Worker] a dead job found. The job buffer will not accept this one."
                    )
                if initialized_job.is_alive:
                    break

            self.lock.acquire()
            self.request_master_socket.send_multipart([
                remote_constants.NEW_JOB_TAG,
                cloudpickle.dumps(initialized_job),
                to_byte(job_address)
            ])
            _ = self.request_master_socket.recv_multipart()
            self.lock.release()

    def _create_job_monitor(self, job):
        """Send heartbeat signals to check target's status"""

        # job_heartbeat_socket: sends heartbeat signal to job
        job_heartbeat_socket = self.ctx.socket(zmq.REQ)
        job_heartbeat_socket.linger = 0
        job_heartbeat_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)
        job_heartbeat_socket.connect("tcp://" + job.worker_heartbeat_address)

        job.is_alive = True
        while job.is_alive and self.master_is_alive and self.worker_is_alive:
            try:
                job_heartbeat_socket.send_multipart(
                    [remote_constants.HEARTBEAT_TAG])
                _ = job_heartbeat_socket.recv_multipart()
                time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)
            except zmq.error.Again as e:
                job.is_alive = False
                logger.warning(
                    "[Worker] lost connection with the job:{}".format(
                        job.job_address))
                if self.master_is_alive and self.worker_is_alive:
                    self._kill_job(job.job_address)

            except zmq.error.ZMQError as e:
                break

        job_heartbeat_socket.close(0)

    def _reply_kill_job(self):
        """Worker starts a thread to wait jobs' commands to kill the job"""
        self.kill_job_socket.linger = 0
        self.kill_job_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)
        while self.worker_is_alive and self.master_is_alive:
            try:
                message = self.kill_job_socket.recv_multipart()
                tag = message[0]
                assert tag == remote_constants.KILLJOB_TAG
                to_kill_job_address = to_str(message[1])
                self._kill_job(to_kill_job_address)
                self.kill_job_socket.send_multipart(
                    [remote_constants.NORMAL_TAG])
            except zmq.error.Again as e:
                #detect whether `self.worker_is_alive` is True periodically
                pass

    def _get_worker_status(self):
        now = datetime.strftime(datetime.now(), '%H:%M:%S')
        virtual_memory = psutil.virtual_memory()
        total_memory = round(virtual_memory[0] / (1024**3), 2)
        used_memory = round(virtual_memory[3] / (1024**3), 2)
        vacant_memory = round(total_memory - used_memory, 2)
        if _IS_WINDOWS:
            load_average = round(psutil.getloadavg()[0], 2)
        else:
            load_average = round(os.getloadavg()[0], 2)
        return (vacant_memory, used_memory, now, load_average)

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

        logger.set_dir(
            os.path.expanduser('~/.parl_data/worker/{}'.format(
                self.master_heartbeat_address.replace(':', '_'))))

        self.heartbeat_socket_initialized.set()
        logger.info("[Worker] Connect to the master node successfully. "
                    "({} CPUs)".format(self.cpu_num))
        while self.master_is_alive and self.worker_is_alive:
            try:
                message = socket.recv_multipart()
                worker_status = self._get_worker_status()
                socket.send_multipart([
                    remote_constants.HEARTBEAT_TAG,
                    to_byte(str(worker_status[0])),
                    to_byte(str(worker_status[1])),
                    to_byte(worker_status[2]),
                    to_byte(str(worker_status[3]))
                ])
            except zmq.error.Again as e:
                self.master_is_alive = False
            except zmq.error.ContextTerminated as e:
                break
        socket.close(0)
        logger.warning(
            "[Worker] lost connection with the master, will exit reply heartbeat for master."
        )
        self.worker_status.clear()
        self.log_server_proc.kill()
        self.log_server_proc.wait()
        # exit the worker
        self.worker_is_alive = False
        self.exit()

    def _create_log_server(self, port):
        log_server_file = __file__.replace('worker.pyc', 'log_server.py')
        log_server_file = log_server_file.replace('worker.py', 'log_server.py')

        if port is None:
            port = "0"  # `0` means using a random port in flask
        command = [
            sys.executable, log_server_file, "--port",
            str(port), "--log_dir", "~/.parl_data/job/", "--line_num", "500"
        ]

        if sys.version_info.major == 3:
            warnings.simplefilter("ignore", ResourceWarning)

        if _IS_WINDOWS:
            FNULL = tempfile.TemporaryFile()
        else:
            FNULL = open(os.devnull, 'w')
        log_server_proc = subprocess.Popen(
            command, stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True)
        FNULL.close()

        log_server_address = "{}:{}".format(self.worker_ip, port)
        return log_server_proc, log_server_address

    def exit(self):
        """close the worker"""
        self.worker_is_alive = False
        kill_process('remote/job.py.*{}'.format(self.reply_job_address))

    def run(self):
        """Keep running until it lost connection with the master.
        """
        if self.worker_is_alive:
            self.reply_master_hearbeat_thread.join()
