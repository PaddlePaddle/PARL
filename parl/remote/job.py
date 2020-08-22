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

# Fix cloudpickle compatible problem we known.
import compatible_trick

import os
os.environ['XPARL'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import cloudpickle
import pickle
import psutil
import re
import sys
import tempfile
import threading
import time
import traceback
import zmq
from multiprocessing import Process, Pipe
from parl.utils import to_str, to_byte, get_ip_address, logger
from parl.utils.communication import loads_argument, loads_return,\
    dumps_argument, dumps_return
from parl.remote import remote_constants
from parl.utils.exceptions import SerializeError, DeserializeError
from parl.remote.message import InitializedJob
from parl.remote.utils import load_remote_class, redirect_stdout_to_file


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
        """
        self.max_memory = None

        self.job_address_receiver, job_address_sender = Pipe()
        self.job_id_receiver, job_id_sender = Pipe()

        self.worker_address = worker_address
        self.log_server_address = log_server_address
        self.job_ip = get_ip_address()
        self.pid = os.getpid()
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
            self.kill_job_socket.send_multipart(
                [remote_constants.KILLJOB_TAG,
                 to_byte(self.job_address)])
            try:
                _ = self.kill_job_socket.recv_multipart()
            except zmq.error.Again as e:
                pass
            os._exit(0)

    def _create_sockets(self):
        """Create five sockets for each job in main process.

        (1) job_socket(functional socket): sends job_address and heartbeat_address to worker.
        (2) ping_heartbeat_socket: replies ping message of client.
        (3) worker_heartbeat_socket: replies heartbeat message of worker.
        (4) client_heartbeat_socket: replies heartbeat message of client.
        (5) kill_job_socket: sends a command to the corresponding worker to kill the job.

        """
        # wait for another process to create reply socket
        self.job_address = self.job_address_receiver.recv()
        self.job_id = self.job_id_receiver.recv()

        self.ctx = zmq.Context()
        # create the job_socket
        self.job_socket = self.ctx.socket(zmq.REQ)
        self.job_socket.connect("tcp://{}".format(self.worker_address))

        # a thread that reply ping signals from the client
        ping_heartbeat_socket, ping_heartbeat_address = self._create_heartbeat_server(
            timeout=False)
        ping_thread = threading.Thread(
            target=self._reply_ping, args=(ping_heartbeat_socket, ))
        ping_thread.setDaemon(True)
        ping_thread.start()

        # a thread that reply heartbeat signals from the worker
        worker_heartbeat_socket, worker_heartbeat_address = self._create_heartbeat_server(
        )
        worker_thread = threading.Thread(
            target=self._reply_worker_heartbeat,
            args=(worker_heartbeat_socket, ))
        worker_thread.setDaemon(True)

        # a thread that reply heartbeat signals from the client
        client_heartbeat_socket, client_heartbeat_address = self._create_heartbeat_server(
        )
        self.client_thread = threading.Thread(
            target=self._reply_client_heartbeat,
            args=(client_heartbeat_socket, ))
        self.client_thread.setDaemon(True)

        # sends job information to the worker
        initialized_job = InitializedJob(
            self.job_address, worker_heartbeat_address,
            client_heartbeat_address, ping_heartbeat_address, None, self.pid,
            self.job_id, self.log_server_address)
        self.job_socket.send_multipart(
            [remote_constants.NORMAL_TAG,
             cloudpickle.dumps(initialized_job)])
        message = self.job_socket.recv_multipart()
        worker_thread.start()

        tag = message[0]
        assert tag == remote_constants.NORMAL_TAG
        # create the kill_job_socket
        kill_job_address = to_str(message[1])
        self.kill_job_socket = self.ctx.socket(zmq.REQ)
        self.kill_job_socket.setsockopt(
            zmq.RCVTIMEO, remote_constants.HEARTBEAT_TIMEOUT_S * 1000)
        self.kill_job_socket.connect("tcp://{}".format(kill_job_address))

    def _check_used_memory(self):
        """Check if the memory used by this job exceeds self.max_memory."""
        stop_job = False
        if self.max_memory is not None:
            process = psutil.Process(self.pid)
            used_memory = float(process.memory_info()[0]) / (1024**2)
            if used_memory > self.max_memory + self.init_memory:
                stop_job = True
        return stop_job

    def _reply_ping(self, socket):
        """Create a socket server that reply the ping signal from client.
        This signal is used to make sure that the job is still alive.
        """
        message = socket.recv_multipart()
        max_memory = to_str(message[1])
        if max_memory != 'None':
            self.max_memory = float(max_memory)
        socket.send_multipart([remote_constants.HEARTBEAT_TAG])
        self.client_thread.start()
        socket.close(0)

    def _create_heartbeat_server(self, timeout=True):
        """Create a socket server that will raises timeout exception.
        """
        heartbeat_socket = self.ctx.socket(zmq.REP)
        if timeout:
            heartbeat_socket.setsockopt(
                zmq.RCVTIMEO, remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)
        heartbeat_socket.linger = 0
        heartbeat_port = heartbeat_socket.bind_to_random_port(addr="tcp://*")
        heartbeat_address = "{}:{}".format(self.job_ip, heartbeat_port)
        return heartbeat_socket, heartbeat_address

    def _reply_client_heartbeat(self, socket):
        """Create a socket that replies heartbeat signals from the client.
        If the job losts connection with the client, it will exit too.
        """
        while True:
            try:
                message = socket.recv_multipart()
                stop_job = self._check_used_memory()
                socket.send_multipart([
                    remote_constants.HEARTBEAT_TAG,
                    to_byte(str(stop_job)),
                    to_byte(self.job_address)
                ])
                if stop_job == True:
                    logger.error(
                        "Memory used by this job exceeds {}. This job will exist."
                        .format(self.max_memory))
                    time.sleep(5)
                    socket.close(0)
                    os._exit(1)
            except zmq.error.Again as e:
                logger.warning(
                    "[Job] Cannot connect to the client. This job will exit and inform the worker."
                )
                break
        socket.close(0)
        with self.lock:
            self.kill_job_socket.send_multipart(
                [remote_constants.KILLJOB_TAG,
                 to_byte(self.job_address)])
            try:
                _ = self.kill_job_socket.recv_multipart()
            except zmq.error.Again as e:
                pass
        logger.warning("[Job]lost connection with the client, will exit")
        os._exit(1)

    def _reply_worker_heartbeat(self, socket):
        """create a socket that replies heartbeat signals from the worker.
        If the worker has exited, the job will exit automatically.
        """
        while True:
            try:
                message = socket.recv_multipart()
                socket.send_multipart([remote_constants.HEARTBEAT_TAG])
            except zmq.error.Again as e:
                logger.warning("[Job] Cannot connect to the worker{}. ".format(
                    self.worker_address) + "Job will quit.")
                break
        socket.close(0)
        os._exit(1)

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
                if '/' in file:
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
            logger.error("NotImplementedError:{}, received tag:{}".format(
                job_address, ))
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
                file_name, class_name, end_of_file = cloudpickle.loads(
                    message[1])
                cls = load_remote_class(file_name, class_name, end_of_file)
                args, kwargs = cloudpickle.loads(message[2])
                logfile_path = os.path.join(self.log_dir, 'stdout.log')
                with redirect_stdout_to_file(logfile_path):
                    obj = cls(*args, **kwargs)
            except Exception as e:
                traceback_str = str(traceback.format_exc())
                error_str = str(e)
                logger.error("traceback:\n{}".format(traceback_str))
                reply_socket.send_multipart([
                    remote_constants.EXCEPTION_TAG,
                    to_byte(error_str + "\ntraceback:\n" + traceback_str)
                ])
                return None
            reply_socket.send_multipart([
                remote_constants.NORMAL_TAG,
                dumps_return(set(obj.__dict__.keys()))
            ])
        else:
            logger.error("Message from job {}".format(message))
            reply_socket.send_multipart([
                remote_constants.EXCEPTION_TAG,
                b"[job]Unkonwn tag when tried to receive the class definition"
            ])
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
        logger.info(
            "[Job] Job {} initialized. Reply heartbeat socket Address: {}.".
            format(job_id, job_address))

        job_address_sender.send(job_address)
        job_id_sender.send(job_id)

        try:
            # receive source code from the actor and append them to the environment variables.
            envdir = self.wait_for_files(reply_socket, job_address)
            sys.path.insert(0, envdir)
            os.chdir(envdir)

            obj = self.wait_for_connection(reply_socket)
            assert obj is not None
            self.single_task(obj, reply_socket, job_address)
        except Exception as e:
            logger.error(
                "Error occurs when running a single task. We will reset this job. \nReason:{}"
                .format(e))
            traceback_str = str(traceback.format_exc())
            logger.error("traceback:\n{}".format(traceback_str))

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
                        logfile_path = os.path.join(self.log_dir, 'stdout.log')
                        with redirect_stdout_to_file(logfile_path):
                            ret = getattr(obj, function_name)(*args, **kwargs)

                        ret = dumps_return(ret)
                        reply_socket.send_multipart([
                            remote_constants.NORMAL_TAG, ret,
                            dumps_return(set(obj.__dict__.keys()))
                        ])

                    elif tag == remote_constants.GET_ATTRIBUTE_TAG:
                        attribute_name = to_str(message[1])
                        logfile_path = os.path.join(self.log_dir, 'stdout.log')
                        with redirect_stdout_to_file(logfile_path):
                            ret = getattr(obj, attribute_name)
                        ret = dumps_return(ret)
                        reply_socket.send_multipart(
                            [remote_constants.NORMAL_TAG, ret])
                    elif tag == remote_constants.SET_ATTRIBUTE_TAG:
                        attribute_name = to_str(message[1])
                        attribute_value = loads_return(message[2])
                        logfile_path = os.path.join(self.log_dir, 'stdout.log')
                        with redirect_stdout_to_file(logfile_path):
                            setattr(obj, attribute_name, attribute_value)
                        reply_socket.send_multipart([
                            remote_constants.NORMAL_TAG,
                            dumps_return(set(obj.__dict__.keys()))
                        ])
                    else:
                        pass

                except Exception as e:
                    # reset the job

                    error_str = str(e)
                    logger.error(error_str)

                    if type(e) == AttributeError:
                        reply_socket.send_multipart([
                            remote_constants.ATTRIBUTE_EXCEPTION_TAG,
                            to_byte(error_str)
                        ])
                        raise AttributeError

                    elif type(e) == SerializeError:
                        reply_socket.send_multipart([
                            remote_constants.SERIALIZE_EXCEPTION_TAG,
                            to_byte(error_str)
                        ])
                        raise SerializeError

                    elif type(e) == DeserializeError:
                        reply_socket.send_multipart([
                            remote_constants.DESERIALIZE_EXCEPTION_TAG,
                            to_byte(error_str)
                        ])
                        raise DeserializeError

                    else:
                        traceback_str = str(traceback.format_exc())
                        logger.error("traceback:\n{}".format(traceback_str))
                        reply_socket.send_multipart([
                            remote_constants.EXCEPTION_TAG,
                            to_byte(error_str + "\ntraceback:\n" +
                                    traceback_str)
                        ])
                        break

            # receive DELETE_TAG from actor, and stop replying worker heartbeat
            elif tag == remote_constants.KILLJOB_TAG:
                reply_socket.send_multipart([remote_constants.NORMAL_TAG])
                logger.warning("An actor exits and this job {} will exit.".
                               format(job_address))
                break
            else:
                logger.error(
                    "The job receives an unknown message: {}".format(message))
                raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_address", required=True, type=str, help="worker_address")
    parser.add_argument(
        "--log_server_address",
        required=True,
        type=str,
        help="log_server_address, address of the log web server on worker")
    args = parser.parse_args()
    job = Job(args.worker_address, args.log_server_address)
