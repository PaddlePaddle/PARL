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
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XPARL'] = 'True'
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
from parl.utils import to_str, to_byte, get_ip_address, logger
from parl.utils.communication import loads_argument, loads_return,\
    dumps_argument, dumps_return
from parl.remote import remote_constants
from parl.utils.exceptions import SerializeError, DeserializeError
from parl.remote.message import InitializedJob


class Job(object):
    """Base class for the job.

    After establishing connection with the remote object, the job will
    create a remote class instance locally and enter an infinite loop,
    waiting for commands from the remote object.

    """

    def __init__(self, worker_address):
        """
        Args:
            worker_address(str): worker_address for sending job information(e.g, pid)

        Attributes:
            pid (int): Job process ID.
            max_memory (float): Maximum memory (MB) can be used by each remote instance. 
        """
        self.job_is_alive = True
        self.worker_address = worker_address
        self.pid = os.getpid()
        self.max_memory = None
        self.lock = threading.Lock()
        self._create_sockets()

    def _create_sockets(self):
        """Create three sockets for each job.

        (1) reply_socket(main socket): receives the command(i.e, the function name and args) 
            from the actual class instance, completes the computation, and returns the result of
            the function.
        (2) job_socket(functional socket): sends job_address and heartbeat_address to worker.
        (3) kill_job_socket: sends a command to the corresponding worker to kill the job.

        """

        self.ctx = zmq.Context()

        # create the reply_socket
        self.reply_socket = self.ctx.socket(zmq.REP)
        job_port = self.reply_socket.bind_to_random_port(addr="tcp://*")
        self.reply_socket.linger = 0
        self.job_ip = get_ip_address()
        self.job_address = "{}:{}".format(self.job_ip, job_port)

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
        self.ping_heartbeat_address = ping_heartbeat_address

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
            client_heartbeat_address, self.ping_heartbeat_address, None,
            self.pid)
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
            if used_memory > self.max_memory:
                stop_job = True
        return stop_job

    def _reply_ping(self, socket):
        """Create a socket server that reply the ping signal from client.
        This signal is used to make sure that the job is still alive.
        """
        while self.job_is_alive:
            message = socket.recv_multipart()
            socket.send_multipart([remote_constants.HEARTBEAT_TAG])
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
        self.client_is_alive = True
        while self.client_is_alive and self.job_is_alive:
            try:
                message = socket.recv_multipart()
                stop_job = self._check_used_memory()
                socket.send_multipart([
                    remote_constants.HEARTBEAT_TAG,
                    to_byte(str(stop_job)),
                    to_byte(self.job_address)
                ])
                if stop_job == True:
                    socket.close(0)
                    os._exit(1)
            except zmq.error.Again as e:
                logger.warning(
                    "[Job] Cannot connect to the client. This job will exit and inform the worker."
                )
                self.client_is_alive = False
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

        self.worker_is_alive = True
        # a flag to decide when to exit heartbeat loop
        while self.worker_is_alive and self.job_is_alive:
            try:
                message = socket.recv_multipart()
                socket.send_multipart([remote_constants.HEARTBEAT_TAG])
            except zmq.error.Again as e:
                logger.warning("[Job] Cannot connect to the worker{}. ".format(
                    self.worker_address) + "Job will quit.")
                self.worker_is_alive = False
                self.job_is_alive = False
        socket.close(0)
        os._exit(1)

    def wait_for_files(self):
        """Wait for python files from remote object.

        When a remote object receives the allocated job address, it will send
        the python files to the job. Later, the job will save these files to a
        temporary directory and add the temporary diretory to Python's working
        directory.

        Returns:
            A temporary directory containing the python files.
        """

        while True:
            message = self.reply_socket.recv_multipart()
            tag = message[0]
            if tag == remote_constants.SEND_FILE_TAG:
                pyfiles = pickle.loads(message[1])
                envdir = tempfile.mkdtemp()
                for file in pyfiles:
                    code = pyfiles[file]
                    file = os.path.join(envdir, file)
                    with open(file, 'wb') as code_file:
                        code_file.write(code)
                self.reply_socket.send_multipart([remote_constants.NORMAL_TAG])
                return envdir
            else:
                logger.error("NotImplementedError:{}, received tag:{}".format(
                    self.job_address, ))
                raise NotImplementedError

    def wait_for_connection(self):
        """Wait for connection from the remote object.

        The remote object will send its class information and initialization
        arguments to the job, these parameters are then used to create a
        local instance in the job process.

        Returns:
            A local instance of the remote class object.
        """

        message = self.reply_socket.recv_multipart()
        tag = message[0]
        obj = None
        if tag == remote_constants.INIT_OBJECT_TAG:
            cls = cloudpickle.loads(message[1])
            args, kwargs = cloudpickle.loads(message[2])
            max_memory = to_str(message[3])
            if max_memory != 'None':
                self.max_memory = float(max_memory)

            try:
                obj = cls(*args, **kwargs)
            except Exception as e:
                traceback_str = str(traceback.format_exc())
                error_str = str(e)
                logger.error("traceback:\n{}".format(traceback_str))
                self.reply_socket.send_multipart([
                    remote_constants.EXCEPTION_TAG,
                    to_byte(error_str + "\ntraceback:\n" + traceback_str)
                ])
                self.client_is_alive = False
                return None

            self.reply_socket.send_multipart([remote_constants.NORMAL_TAG])
        else:
            logger.error("Message from job {}".format(message))
            self.reply_socket.send_multipart([
                remote_constants.EXCEPTION_TAG,
                b"[job]Unkonwn tag when tried to receive the class definition"
            ])
            raise NotImplementedError

        return obj

    def run(self):
        """An infinite loop waiting for a new task.
        """
        # receive source code from the actor and append them to the environment variables.
        envdir = self.wait_for_files()
        sys.path.append(envdir)
        self.client_is_alive = True
        self.client_thread.start()

        try:
            obj = self.wait_for_connection()
            assert obj is not None
            self.single_task(obj)
        except Exception as e:
            logger.error(
                "Error occurs when running a single task. We will reset this job. Reason:{}"
                .format(e))
            traceback_str = str(traceback.format_exc())
            logger.error("traceback:\n{}".format(traceback_str))
        with self.lock:
            self.kill_job_socket.send_multipart(
                [remote_constants.KILLJOB_TAG,
                 to_byte(self.job_address)])
            try:
                _ = self.kill_job_socket.recv_multipart()
            except zmq.error.Again as e:
                pass
            os._exit(1)

    def single_task(self, obj):
        """An infinite loop waiting for commands from the remote object.

        Each job will receive two kinds of message from the remote object:

        1. When the remote object calls a function, job will run the
           function on the local instance and return the results to the
           remote object.
        2. When the remote object is deleted, the job will quit and release
           related computation resources.
        """

        while self.job_is_alive and self.client_is_alive:
            message = self.reply_socket.recv_multipart()

            tag = message[0]

            if tag == remote_constants.CALL_TAG:
                try:
                    function_name = to_str(message[1])
                    data = message[2]
                    args, kwargs = loads_argument(data)
                    ret = getattr(obj, function_name)(*args, **kwargs)
                    ret = dumps_return(ret)

                    self.reply_socket.send_multipart(
                        [remote_constants.NORMAL_TAG, ret])

                except Exception as e:
                    # reset the job
                    self.client_is_alive = False

                    error_str = str(e)
                    logger.error(error_str)

                    if type(e) == AttributeError:
                        self.reply_socket.send_multipart([
                            remote_constants.ATTRIBUTE_EXCEPTION_TAG,
                            to_byte(error_str)
                        ])
                        raise AttributeError

                    elif type(e) == SerializeError:
                        self.reply_socket.send_multipart([
                            remote_constants.SERIALIZE_EXCEPTION_TAG,
                            to_byte(error_str)
                        ])
                        raise SerializeError

                    elif type(e) == DeserializeError:
                        self.reply_socket.send_multipart([
                            remote_constants.DESERIALIZE_EXCEPTION_TAG,
                            to_byte(error_str)
                        ])
                        raise DeserializeError

                    else:
                        traceback_str = str(traceback.format_exc())
                        logger.error("traceback:\n{}".format(traceback_str))
                        self.reply_socket.send_multipart([
                            remote_constants.EXCEPTION_TAG,
                            to_byte(error_str + "\ntraceback:\n" +
                                    traceback_str)
                        ])
                        break

            # receive DELETE_TAG from actor, and stop replying worker heartbeat
            elif tag == remote_constants.KILLJOB_TAG:
                self.reply_socket.send_multipart([remote_constants.NORMAL_TAG])
                self.client_is_alive = False
                logger.warning(
                    "An actor exits and this job {} will exit.".format(
                        self.job_address))
                break
            else:
                logger.error(
                    "The job receives an unknown message: {}".format(message))
                raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_address", required=True, type=str, help="worker_address")
    args = parser.parse_args()
    job = Job(args.worker_address)
    job.run()
