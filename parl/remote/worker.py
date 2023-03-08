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
import multiprocessing as mp
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
import pynvml
import parl
from parl.utils import get_ip_address, to_byte, to_str, logger, _IS_WINDOWS
from parl.remote import remote_constants
from parl.remote.message import InitializedWorker, AllocatedCpu, AllocatedGpu
from parl.remote.status import WorkerStatus
from parl.remote.zmq_utils import create_server_socket, create_client_socket
from parl.remote.grpc_heartbeat import HeartbeatServerThread, HeartbeatClientThread
from six.moves import queue
from parl.remote.utils import get_version, XPARL_PYTHON


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
        remove_job_socket (zmq.Context.socket): A socket that receives commands to remove the job from jobs immediately.
                                                Used to remove the dead job immediately and allocate a new job
                                                instead of waiting for the heartbeat failure.
        job_buffer (str): A buffer that stores initialized jobs for providing new jobs in a short time.

    Args:
        master_address (str): IP address of the master node.
        cpu_num (int): Number of cpu to be used on the worker.
        gpu (str): Comma separated list of GPU(s) to use.
    """

    def __init__(self, master_address, cpu_num=None, log_server_port=None, gpu=''):
        # At first, fork a subprocess with blank context to launch job.py
        self.cmd_queue = mp.Queue()
        process = mp.Process(target=self._launch_cmd, args=(self.cmd_queue, ))
        process.daemon = True
        process.start()

        # initialzation
        self.pid = str(os.getpid())
        self.lock = threading.Lock()
        self.ctx = zmq.Context.instance()
        self.master_address = master_address
        self.master_is_alive = True
        self.worker_is_alive = True
        self.worker_status = None  # initialized at `self._create_jobs`
        self._set_cpu_num(cpu_num)
        self._set_gpu_num(gpu)
        self.gpu = gpu
        self.device_count = self.cpu_num + self.gpu_num
        self.job_buffer = queue.Queue(maxsize=self.device_count)
        self._create_sockets()
        self.check_env_consistency()
        # create log server
        self.log_server_proc, self.log_server_address = self._create_log_server(port=log_server_port)

        # create a thread that waits commands from the job to kill the job.
        self.remove_job_thread = threading.Thread(target=self._reply_remove_job)
        self.remove_job_thread.setDaemon(True)
        self.remove_job_thread.start()

        self._create_jobs()

        # create a thread that initializes jobs and adds them into the job_buffer
        job_thread = threading.Thread(target=self._fill_job_buffer)
        job_thread.setDaemon(True)
        job_thread.start()

        thread = threading.Thread(target=self._update_worker_status_to_master)
        thread.setDaemon(True)
        thread.start()

    def _set_cpu_num(self, cpu_num=None):
        """set useable cpu number for worker"""
        if cpu_num is not None:
            assert isinstance(cpu_num, int), "cpu_num should be INT type, please check the input type."
            self.cpu_num = cpu_num
        else:
            self.cpu_num = mp.cpu_count()

    def _set_gpu_num(self, gpu=''):
        """set useable gpu number for worker"""
        self.gpu_num = 0
        if gpu:
            self.gpu_num = len(gpu.split(','))
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            if self.gpu_num > device_count:
                error_message = "gpu:{} exceeds device_count:{}".format(gpu, device_count)
                logger.error(error_message)
                assert self.gpu_num <= device_count, error_message

    def check_env_consistency(self):
        '''Verify that the parl & python version as well as some other packages in 'worker' process
            matches that of the 'master' process'''
        self.request_master_socket.send_multipart([remote_constants.CHECK_VERSION_TAG])
        message = self.request_master_socket.recv_multipart()
        tag = message[0]
        if tag == remote_constants.NORMAL_TAG:
            worker_parl_version = parl.__version__
            worker_python_version_major = str(sys.version_info.major)
            worker_python_version_minor = str(sys.version_info.minor)
            assert worker_parl_version == to_str(message[1]) and worker_python_version_major == to_str(message[2])\
                and worker_python_version_minor == to_str(message[3]),\
                '''Version mismatch: the "master" is of version "parl={}, python={}.{}". However, \
                "parl={}, python={}.{}"is provided in your environment.'''.format(
                        to_str(message[1]), to_str(message[2]), to_str(message[3]),
                        worker_parl_version, worker_python_version_major, worker_python_version_minor
                    )
            worker_pyarrow_version = str(get_version('pyarrow'))
            master_pyarrow_version = to_str(message[4])
            if worker_pyarrow_version != master_pyarrow_version:
                if master_pyarrow_version == 'None':
                    error_message = """"pyarrow" is provided in your current enviroment, however, it is not \
found in "master"'s environment. To use "pyarrow" for serialization, please install \
"pyarrow={}" in "master"'s environment!""".format(worker_pyarrow_version)
                elif worker_pyarrow_version == 'None':
                    error_message = """"pyarrow" is provided in "master"'s enviroment, however, it is not \
found in your current environment. To use "pyarrow" for serialization, please install \
"pyarrow={}" in your current environment!""".format(master_pyarrow_version)
                else:
                    error_message = '''Version mismatch: the 'master' is of version 'pyarrow={}'. However, \
'pyarrow={}'is provided in your current environment.'''.format(master_pyarrow_version, worker_pyarrow_version)
                raise Exception(error_message)
        else:
            raise NotImplementedError

    def _create_sockets(self):
        """Each worker maintains four sockets:

        (1) request_master_socket: sends job address to master node.
        (2) reply_job_socket: receives job_address from subprocess.
        (3) remove_job_socket : receives commands to remove the job from jobs immediately.
                                Used to remove the dead job immediately and allocate a new job
                                instead of waiting for the heartbeat failure.
        (4) reply_log_server_socket: receives log_server_heartbeat_address from subprocess.

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

        # remove_job_socket
        self.remove_job_socket = self.ctx.socket(zmq.REP)
        self.remove_job_socket.linger = 0
        remove_job_port = self.remove_job_socket.bind_to_random_port("tcp://*")
        self.remove_job_address = "{}:{}".format(self.worker_ip, remove_job_port)

        # reply_log_server_socket: receives log_server_heartbeat_address from subprocess
        self.reply_log_server_socket, reply_log_server_port = create_server_socket(self.ctx)
        self.reply_log_server_address = "{}:{}".format(self.worker_ip, reply_log_server_port)

    def _create_jobs(self):
        """Create jobs and send a instance of ``InitializedWorker`` that contains the worker information to the master."""
        try:
            self.request_master_socket.send_multipart([remote_constants.WORKER_CONNECT_TAG])
            _ = self.request_master_socket.recv_multipart()
        except zmq.error.Again as e:
            logger.error("Can not connect to the master, " "please check if master is started.")
            self.master_is_alive = False
            return

        initialized_jobs = self._init_jobs(job_num=self.device_count)
        self.request_master_socket.setsockopt(zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)

        def master_heartbeat_exit_callback_func():
            logger.warning("[Worker] lost connection with the master, will exit reply heartbeat for master.")
            if self.worker_status is not None:
                self.worker_status.clear()
            self.log_server_proc.kill()
            self.log_server_proc.wait()
            # exit the worker
            self.exit()

        self.master_heartbeat_thread = HeartbeatServerThread(
            heartbeat_exit_callback_func=master_heartbeat_exit_callback_func)
        self.master_heartbeat_thread.setDaemon(True)
        self.master_heartbeat_thread.start()
        self.master_heartbeat_address = self.master_heartbeat_thread.get_address()

        logger.set_dir(
            os.path.expanduser('~/.parl_data/worker/{}'.format(self.master_heartbeat_address.replace(':', '_'))))
        if self.cpu_num:
            logger.info("[Worker] Connect to the master node successfully. " "({} CPUs)".format(self.cpu_num))
        elif self.gpu_num:
            logger.info("[Worker] Connect to the master node successfully. " "({} GPUs)".format(self.gpu_num))

        for job in initialized_jobs:
            job.worker_address = self.master_heartbeat_address

        allocated_cpu = AllocatedCpu(self.master_heartbeat_address, self.cpu_num)
        allocated_gpu = AllocatedGpu(self.master_heartbeat_address, self.gpu)
        initialized_worker = InitializedWorker(self.master_heartbeat_address, initialized_jobs, allocated_cpu,
                                               allocated_gpu, socket.gethostname())
        self.request_master_socket.send_multipart(
            [remote_constants.WORKER_INITIALIZED_TAG,
             cloudpickle.dumps(initialized_worker)])

        message = self.request_master_socket.recv_multipart()
        if message[0] == remote_constants.REJECT_CPU_WORKER_TAG:
            logger.error("GPU cluster rejects a CPU worker to join in")
            self.worker_is_alive = False
        elif message[0] == remote_constants.REJECT_GPU_WORKER_TAG:
            logger.error("CPU cluster rejects a GPU worker to join in")
            self.worker_is_alive = False
        else:
            self.worker_status = WorkerStatus(self.master_heartbeat_address, initialized_jobs, self.cpu_num,
                                              self.gpu_num)

    def _fill_job_buffer(self):
        """An endless loop that adds initialized job into the job buffer"""
        initialized_jobs = []
        while self.worker_is_alive:
            if self.job_buffer.full() is False:
                job_num = self.device_count - self.job_buffer.qsize()
                if job_num > 0:
                    initialized_jobs = self._init_jobs(job_num=job_num)
                    for job in initialized_jobs:
                        self.job_buffer.put(job)

            time.sleep(0.02)
        self.exit()

    def _launch_cmd(self, cmd_queue):
        FNULL = open(os.devnull, 'w')
        while True:
            command = cmd_queue.get()
            subprocess.Popen(command, stdout=FNULL, close_fds=True)
        FNULL.close()

    def _init_jobs(self, job_num):
        """Create jobs.

        Args:
            job_num(int): the number of jobs to create.
        """
        job_file = __file__.replace('worker.pyc', 'job.py')
        job_file = job_file.replace('worker.py', 'job.py')
        command = XPARL_PYTHON + [
            job_file, "--worker_address", self.reply_job_address, "--log_server_address", self.log_server_address
        ]

        if sys.version_info.major == 3:
            warnings.simplefilter("ignore", ResourceWarning)

        # avoid that many jobs are killed and restarted at the same time.
        self.lock.acquire()

        for _ in range(job_num):
            self.cmd_queue.put(command)

        new_jobs = []
        for _ in range(job_num):
            job_init_message = self.reply_job_socket.recv_multipart()
            self.reply_job_socket.send_multipart([remote_constants.NORMAL_TAG, to_byte(self.remove_job_address), to_byte(self.pid)])
            initialized_job = cloudpickle.loads(job_init_message[1])
            new_jobs.append(initialized_job)

            def heartbeat_exit_callback_func(job):
                job.is_alive = False
                logger.warning("[Worker] lost connection with the job:{}".format(job.job_address))
                if self.master_is_alive and self.worker_is_alive:
                    self._remove_job(job.job_address)

            # a thread for sending heartbeat signals to job
            thread = HeartbeatClientThread(
                initialized_job.worker_heartbeat_address,
                heartbeat_exit_callback_func=heartbeat_exit_callback_func,
                exit_func_args=(initialized_job, ))
            thread.setDaemon(True)
            thread.start()

        self.lock.release()
        assert len(new_jobs) > 0, "init jobs failed"
        return new_jobs

    def _remove_job(self, job_address):
        """Kill a job process and update worker information"""
        success = self.worker_status.remove_job(job_address)
        if success:
            while True:
                initialized_job = self.job_buffer.get()
                initialized_job.worker_address = self.master_heartbeat_address
                if initialized_job.is_alive:
                    self.worker_status.add_job(initialized_job)
                    if not initialized_job.is_alive:  # make sure that the job is still alive.
                        self.worker_status.remove_job(initialized_job.job_address)
                        continue
                else:
                    logger.warning("[Worker] a dead job found. The job buffer will not accept this one.")
                if initialized_job.is_alive:
                    break

            self.lock.acquire()
            self.request_master_socket.send_multipart(
                [remote_constants.NEW_JOB_TAG,
                 cloudpickle.dumps(initialized_job),
                 to_byte(job_address)])
            _ = self.request_master_socket.recv_multipart()
            self.lock.release()

    def _reply_remove_job(self):
        """Worker starts a thread to wait jobs' commands to remove the job immediately"""
        self.remove_job_socket.linger = 0
        self.remove_job_socket.setsockopt(zmq.RCVTIMEO, remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)
        while self.worker_is_alive and self.master_is_alive:
            try:
                message = self.remove_job_socket.recv_multipart()
                tag = message[0]
                assert tag == remote_constants.KILLJOB_TAG
                to_remove_job_address = to_str(message[1])
                logger.info("[Worker] A job requests the worker to stop this job.")
                self._remove_job(to_remove_job_address)
                self.remove_job_socket.send_multipart([remote_constants.NORMAL_TAG])
            except zmq.error.Again as e:
                #detect whether `self.worker_is_alive` is True periodically
                pass

    def _get_worker_status(self):
        now = datetime.strftime(datetime.now(), '%H:%M:%S')
        virtual_memory = psutil.virtual_memory()
        total_memory = round(virtual_memory[0] / (1024**3), 2)
        used_memory = round(virtual_memory[3] / (1024**3), 2)
        vacant_memory = round(total_memory - used_memory, 2)
        used_gpu_memory = 0
        vacant_gpu_memory = 0
        if self.gpu:
            pynvml.nvmlInit()
            rate = 0.0
            for gpu_id in self.gpu.split(','):
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
                memery_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_gpu_memory += int(memery_info.used / (1024 * 1024))
                vacant_gpu_memory += int(memery_info.free / (1024 * 1024))
                rate += pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            load_average = round(rate / len(self.gpu.split('.')), 2)
            pynvml.nvmlShutdown()
        else:
            if _IS_WINDOWS:
                load_average = round(psutil.getloadavg()[0], 2)
            else:
                load_average = round(os.getloadavg()[0], 2)

        update_status = {
            "vacant_memory": vacant_memory,
            "used_memory": used_memory,
            "vacant_gpu_memory": vacant_gpu_memory,
            "used_gpu_memory": used_gpu_memory,
            "load_time": now,
            "load_value": load_average
        }
        return update_status

    def _update_worker_status_to_master(self):
        while self.master_is_alive and self.worker_is_alive:
            worker_status = self._get_worker_status()

            self.lock.acquire()
            try:
                self.request_master_socket.send_multipart([
                    remote_constants.WORKER_STATUS_UPDATE_TAG,
                    to_byte(self.master_heartbeat_address),
                    cloudpickle.dumps(worker_status)
                ])
                message = self.request_master_socket.recv_multipart()
            except zmq.error.Again as e:
                self.master_is_alive = False
            finally:
                self.lock.release()

            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

    def _create_log_server(self, port):
        log_server_file = __file__.replace('worker.pyc', 'log_server.py')
        log_server_file = log_server_file.replace('worker.py', 'log_server.py')

        if port is None:
            port = "0"  # `0` means using a random port in flask
        command = XPARL_PYTHON + [
            log_server_file,
            "--port",
            str(port),
            "--log_dir",
            "~/.parl_data/job/",
            "--line_num",
            "500",
            "--worker_address",
            self.reply_log_server_address,
        ]

        if sys.version_info.major == 3:
            warnings.simplefilter("ignore", ResourceWarning)

        if _IS_WINDOWS:
            FNULL = tempfile.TemporaryFile()
        else:
            FNULL = open(os.devnull, 'w')
        log_server_proc = subprocess.Popen(command, stdout=FNULL, close_fds=True)
        FNULL.close()

        log_server_address = "{}:{}".format(self.worker_ip, port)

        message = self.reply_log_server_socket.recv_multipart()
        log_server_heartbeat_addr = to_str(message[1])
        self.reply_log_server_socket.send_multipart([remote_constants.NORMAL_TAG])

        def heartbeat_exit_callback_func():
            # only output warning
            logger.warning("[Worker] lost connection with the log_server.")

        # a thread for sending heartbeat signals to log_server
        thread = HeartbeatClientThread(
            log_server_heartbeat_addr, heartbeat_exit_callback_func=heartbeat_exit_callback_func)
        thread.setDaemon(True)
        thread.start()

        return log_server_proc, log_server_address

    def exit(self):
        """close the worker"""
        self.worker_is_alive = False
        if self.master_heartbeat_thread.is_alive():
            self.master_heartbeat_thread.exit()

    def run(self):
        """Keep running until it lost connection with the master.
        """
        if self.worker_is_alive:
            self.master_heartbeat_thread.join()
