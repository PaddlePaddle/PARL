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
# set the environment variables before importing any DL framework.
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['XPARL'] = 'True'
os.environ['XPARL_igonre_core'] = 'true'

# Fix cloudpickle compatible problem we known.
import compatible_trick

import argparse
import cloudpickle
import pickle
import psutil
import re
import sys
import tempfile
import shutil
import threading
import time
import traceback
import zmq
import importlib
import parl
from multiprocessing import Process, Pipe
from parl.utils import to_str, to_byte, get_ip_address, logger
from parl.remote.communication import loads_argument, loads_return,\
    dumps_argument, dumps_return
from parl.remote import remote_constants
from parl.utils.exceptions import SerializeError, DeserializeError
from parl.remote.message import InitializedJob
from parl.remote.utils import redirect_output_to_file
from parl.remote.remote_class_serialization import load_remote_class
from parl.remote.zmq_utils import create_server_socket, create_client_socket
from parl.remote.grpc_heartbeat import HeartbeatServerThread, HeartbeatClientThread

DL_FRAMEWORKS = ["paddle", "torch"]
for dl_framework in DL_FRAMEWORKS:
    assert dl_framework not in sys.modules, "{} imported".format(dl_framework)


class Job(object):
    """Base class for the job.

    After establishing connection with the remote object, the job will
    create a remote class instance locally and enter an infinite loop
    in a separate process, waiting for commands from the remote object.

    """

    def __init__(self, worker_address, log_server_address):
        """
        Args:
            worker_address(str): worker_address for sending job information(e.g, pid)

        Attributes:
            pid (int): Job process ID.
            max_memory (float): Maximum memory (MB) can be used by each remote instance.
            gpu (str): id list of GPUs can be used by each remote instance.
            job_id (str): Unique ID for the job. 
            instance_id (str): Unique instance ID to which the job connects.
        """
        self.max_memory = None
        self.gpu = ""

        self.job_address_receiver, job_address_sender = Pipe()
        self.job_id_receiver, job_id_sender = Pipe()

        self.worker_address = worker_address
        self.log_server_address = log_server_address
        self.job_ip = get_ip_address()
        self.pid = os.getpid()
        self.worker_pid = None
        """
        NOTE:
            In Windows, it will raise errors when creating threading.Lock before starting multiprocess.Process.
        """
        self.lock = threading.Lock()
        th = threading.Thread(target=self._create_sockets)
        th.setDaemon(True)
        th.start()

        process = psutil.Process(self.pid)
        self.init_memory = float(process.memory_info()[0]) / (1024**2)

        self.run(job_address_sender, job_id_sender)

        with self.lock:
            self.remove_job_socket.send_multipart([remote_constants.KILLJOB_TAG, to_byte(self.job_address)])
            try:
                _ = self.remove_job_socket.recv_multipart()
            except zmq.error.Again as e:
                pass
            os._exit(0)

    def _create_sockets(self):
        """Create three sockets for each job in the main process.

        (1) job_socket(functional socket): sends job_address and heartbeat_address to worker.
        (2) reply_client_ping_socket: replies ping message of client.
        (3) remove_job_socket: sends a command to the corresponding worker to remove the job.
                               Used to ask the worker removing the dead job immediately 
                               instead of waiting for the heartbeat failure.

        Create two heartbeat server threads for each job:
        (1) worker_heartbeat_server_thread: reply heartbeat signal from the worker.
        (2) client_heartbeat_client_thread: send heartbeat signal to the client.
        """
        # wait for another process to create reply socket
        self.job_address = self.job_address_receiver.recv()
        self.job_id = self.job_id_receiver.recv()

        self.ctx = zmq.Context()
        # create the job_socket
        self.job_socket = create_client_socket(self.ctx, self.worker_address, heartbeat_timeout=True)

        # a thread that reply ping signals from the client
        reply_client_ping_socket, port = create_server_socket(self.ctx)
        reply_client_ping_address = "{}:{}".format(self.job_ip, port)
        ping_thread = threading.Thread(target=self._reply_ping, args=(reply_client_ping_socket, ))
        ping_thread.setDaemon(True)
        ping_thread.start()

        # a thread that reply heartbeat signals from the worker
        def worker_heartbeat_exit_callback_func():
            logger.warning("[Job]lost connection with the worker, will exit")
            os._exit(1)

        worker_heartbeat_server_thread = HeartbeatServerThread(
            heartbeat_exit_callback_func=worker_heartbeat_exit_callback_func)
        worker_heartbeat_server_thread.setDaemon(True)
        worker_heartbeat_server_thread.start()

        # sends job information to the worker
        initialized_job = InitializedJob(self.job_address, worker_heartbeat_server_thread.get_address(),
                                         reply_client_ping_address, None, self.pid, self.job_id,
                                         self.log_server_address)

        try:
            self.job_socket.send_multipart([remote_constants.NORMAL_TAG, cloudpickle.dumps(initialized_job)])
            message = self.job_socket.recv_multipart()
        except zmq.error.Again as e:
            logger.warning("[Job] Cannot connect to the worker {}. ".format(self.worker_address) + "Job will quit.")
            self.job_socket.close(0)
            os._exit(0)

        tag = message[0]
        assert tag == remote_constants.NORMAL_TAG
        # create the remove_job_socket
        remove_job_address = to_str(message[1])
        self.worker_pid = int(to_str(message[2]))
        worker_heartbeat_server_thread.set_host_pid(self.worker_pid)
        self.remove_job_socket = self.ctx.socket(zmq.REQ)
        self.remove_job_socket.setsockopt(zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)
        self.remove_job_socket.connect("tcp://{}".format(remove_job_address))

    def _check_used_memory(self):
        """Check if the memory used by this job exceeds self.max_memory."""
        while True:
            if self.max_memory is not None:
                process = psutil.Process(self.pid)
                used_memory = float(process.memory_info()[0]) / (1024**2)
                if used_memory > self.max_memory + self.init_memory:
                    break
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

        # out of memory
        logger.error("Memory used by this job exceeds {}. This job will exist.".format(self.max_memory))

        stop_message = "Job {} exceeds max memory usage, will stop this job.".format(self.job_address)
        self.client_heartbeat_client_thread.stop(remote_constants.HEARTBEAT_OUT_OF_MEMORY_TAG, stop_message)

        with self.lock:
            self.remove_job_socket.send_multipart([remote_constants.KILLJOB_TAG, to_byte(self.job_address)])
            try:
                _ = self.remove_job_socket.recv_multipart()
            except zmq.error.Again as e:
                pass
        os._exit(1)

    def _reply_ping(self, socket):
        """Create a socket server that reply the ping signal from client.
        This signal is used to make sure that the job is still alive.
        """
        message = socket.recv_multipart()
        client_heartbeat_server_addr = to_str(message[1])
        max_memory = to_str(message[2])
        if max_memory != 'None':
            self.max_memory = float(max_memory)
        self.gpu = to_str(message[3])
        self.instance_id = to_str(message[4])
        socket.send_multipart([remote_constants.HEARTBEAT_TAG])

        def client_heartbeat_exit_callback_func():
            with self.lock:
                self.remove_job_socket.send_multipart([remote_constants.KILLJOB_TAG, to_byte(self.job_address)])
                try:
                    _ = self.remove_job_socket.recv_multipart()
                except zmq.error.Again as e:
                    pass
            logger.warning("[Job] lost connection with the client. This job will exit.")
            os._exit(1)

        # a thread that sends heartbeat signals from the client
        self.client_heartbeat_client_thread = HeartbeatClientThread(
            client_id=self.instance_id,
            heartbeat_server_addr=client_heartbeat_server_addr,
            heartbeat_exit_callback_func=client_heartbeat_exit_callback_func)
        self.client_heartbeat_client_thread.setDaemon(True)
        self.client_heartbeat_client_thread.start()

        memory_monitor_thread = threading.Thread(target=self._check_used_memory)
        memory_monitor_thread.setDaemon(True)
        memory_monitor_thread.start()

        socket.close(0)

    def wait_for_files(self, reply_socket, job_address):
        """Wait for python files from remote object.

        When a remote object receives the allocated job address, it will send
        the python files to the job. Later, the job will save these files to a
        temporary directory and add the temporary diretory to Python's working
        directory.

        Args:
            reply_socket (sockert): main socket to accept commands of remote object.
            job_address (String): address of reply_socket.

        Returns:
            A temporary directory containing the python files.
        """

        message = reply_socket.recv_multipart()
        tag = message[0]
        if tag == remote_constants.SEND_FILE_TAG:
            pyfiles = pickle.loads(message[1])
            envdir = tempfile.mkdtemp()

            for empty_subfolder in pyfiles['empty_subfolders']:
                empty_subfolder_path = os.path.join(envdir, empty_subfolder)
                if not os.path.exists(empty_subfolder_path):
                    os.makedirs(empty_subfolder_path)

            # save python files to temporary directory
            for file, code in pyfiles['python_files'].items():
                file = os.path.join(envdir, file)
                with open(file, 'wb') as code_file:
                    code_file.write(code)

            # save other files to current directory
            for file, content in pyfiles['other_files'].items():
                # create directory (i.e. ./rom_files/)
                if os.sep in file:
                    try:
                        sep = os.sep
                        recursive_dirs = os.path.join(*(file.split(sep)[:-1]))
                        recursive_dirs = os.path.join(envdir, recursive_dirs)
                        os.makedirs(recursive_dirs)
                    except OSError as e:
                        pass
                file = os.path.join(envdir, file)
                with open(file, 'wb') as f:
                    f.write(content)
            reply_socket.send_multipart([remote_constants.NORMAL_TAG])
            return envdir
        else:
            logger.error("NotImplementedError:{}, received tag:{}".format(job_address, tag))
            raise NotImplementedError

    def wait_for_connection(self, reply_socket):
        """Wait for connection from the remote object.

        The remote object will send its class information and initialization
        arguments to the job, these parameters are then used to create a
        local instance in the job process.

        Args:
            reply_socket (sockert): main socket to accept commands of remote object.

        Returns:
            A local instance of the remote class object.
        """

        message = reply_socket.recv_multipart()
        tag = message[0]
        obj = None

        if tag == remote_constants.INIT_OBJECT_TAG:
            try:
                if self.gpu:
                    os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
                del os.environ['XPARL_igonre_core']
                importlib.reload(parl)
                cls = load_remote_class(message[1])
                args, kwargs = cloudpickle.loads(message[2])
                with redirect_output_to_file(self.logfile_path, os.devnull):
                    obj = cls(*args, **kwargs)
            except Exception as e:
                traceback_str = str(traceback.format_exc())
                error_str = str(e)
                logger.error("traceback:\n{}".format(traceback_str))
                reply_socket.send_multipart(
                    [remote_constants.EXCEPTION_TAG,
                     to_byte(error_str + "\ntraceback:\n" + traceback_str)])
                return None
            reply_socket.send_multipart([remote_constants.NORMAL_TAG, dumps_return(set(obj.__dict__.keys()))])
        else:
            logger.error("Message from job {}".format(message))
            reply_socket.send_multipart(
                [remote_constants.EXCEPTION_TAG, b"[job]Unkonwn tag when tried to receive the class definition"])
            raise NotImplementedError

        return obj

    def run(self, job_address_sender, job_id_sender):
        """An infinite loop waiting for a new task.

        Args:
            job_address_sender(sending end of multiprocessing.Pipe): send job address of reply_socket to main process.
        """
        ctx = zmq.Context()

        # create the reply_socket
        reply_socket = ctx.socket(zmq.REP)
        job_port = reply_socket.bind_to_random_port(addr="tcp://*")
        reply_socket.linger = 0
        job_ip = get_ip_address()
        job_address = "{}:{}".format(job_ip, job_port)

        job_id = job_address.replace(':', '_') + '_' + str(int(time.time()))
        self.log_dir = os.path.expanduser('~/.parl_data/job/{}'.format(job_id))
        logger.set_dir(self.log_dir)
        self.logfile_path = os.path.join(self.log_dir, 'stdout.log')

        logger.info("[Job] Job {} initialized. Reply heartbeat socket Address: {}.".format(job_id, job_address))

        job_address_sender.send(job_address)
        job_id_sender.send(job_id)

        # receive source code from the actor and append them to the environment variables.
        envdir = self.wait_for_files(reply_socket, job_address)
        sys.path.insert(0, envdir)
        os.chdir(envdir)

        try:
            obj = self.wait_for_connection(reply_socket)
            assert obj is not None

            self.single_task(obj, reply_socket, job_address)
        except Exception as e:
            logger.error("Error occurs when running a single task. We will reset this job. \nReason:{}".format(e))
            traceback_str = str(traceback.format_exc())
            logger.error("traceback:\n{}".format(traceback_str))
        shutil.rmtree(envdir)

    def single_task(self, obj, reply_socket, job_address):
        """An infinite loop waiting for commands from the remote object.

        Each job will receive two kinds of message from the remote object:

        1. When the remote object calls a function, job will run the
           function on the local instance and return the results to the
           remote object.
        2. When the remote object is deleted, the job will quit and release
           related computation resources.

        Args:
            reply_socket (sockert): main socket to accept commands of remote object.
            job_address (String): address of reply_socket.
        """

        while True:
            message = reply_socket.recv_multipart()
            tag = message[0]
            if tag in [
                    remote_constants.CALL_TAG,
                    remote_constants.GET_ATTRIBUTE_TAG,
                    remote_constants.SET_ATTRIBUTE_TAG,
            ]:
                try:
                    if tag == remote_constants.CALL_TAG:
                        function_name = to_str(message[1])
                        data = message[2]
                        args, kwargs = loads_argument(data)

                        # Redirect stdout to stdout.log temporarily
                        with redirect_output_to_file(self.logfile_path, os.devnull):
                            ret = getattr(obj, function_name)(*args, **kwargs)

                        ret = dumps_return(ret)
                        reply_socket.send_multipart(
                            [remote_constants.NORMAL_TAG, ret,
                             dumps_return(set(obj.__dict__.keys()))])

                    elif tag == remote_constants.GET_ATTRIBUTE_TAG:
                        attribute_name = to_str(message[1])
                        with redirect_output_to_file(self.logfile_path, os.devnull):
                            ret = getattr(obj, attribute_name)
                        ret = dumps_return(ret)
                        reply_socket.send_multipart([remote_constants.NORMAL_TAG, ret])
                    elif tag == remote_constants.SET_ATTRIBUTE_TAG:
                        attribute_name = to_str(message[1])
                        attribute_value = loads_return(message[2])
                        with redirect_output_to_file(self.logfile_path, os.devnull):
                            setattr(obj, attribute_name, attribute_value)
                        reply_socket.send_multipart(
                            [remote_constants.NORMAL_TAG,
                             dumps_return(set(obj.__dict__.keys()))])
                    else:
                        pass

                except Exception as e:
                    # reset the job

                    error_str = str(e)
                    logger.error(error_str)

                    if type(e) == AttributeError:
                        reply_socket.send_multipart([remote_constants.ATTRIBUTE_EXCEPTION_TAG, to_byte(error_str)])
                        raise AttributeError

                    elif type(e) == SerializeError:
                        reply_socket.send_multipart([remote_constants.SERIALIZE_EXCEPTION_TAG, to_byte(error_str)])
                        raise SerializeError

                    elif type(e) == DeserializeError:
                        reply_socket.send_multipart([remote_constants.DESERIALIZE_EXCEPTION_TAG, to_byte(error_str)])
                        raise DeserializeError

                    else:
                        traceback_str = str(traceback.format_exc())
                        logger.error("traceback:\n{}".format(traceback_str))
                        reply_socket.send_multipart(
                            [remote_constants.EXCEPTION_TAG,
                             to_byte(error_str + "\ntraceback:\n" + traceback_str)])
                        break

            # receive DELETE_TAG from actor, and stop replying worker heartbeat
            elif tag == remote_constants.KILLJOB_TAG:
                reply_socket.send_multipart([remote_constants.NORMAL_TAG])
                logger.warning("An actor exits and this job {} will exit.".format(job_address))
                break
            else:
                logger.error("The job receives an unknown message: {}".format(message))
                raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_address", required=True, type=str, help="worker_address")
    parser.add_argument(
        "--log_server_address",
        required=True,
        type=str,
        help="log_server_address, address of the log web server on worker")
    args = parser.parse_args()
    job = Job(args.worker_address, args.log_server_address)
