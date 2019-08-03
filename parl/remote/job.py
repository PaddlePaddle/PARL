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


class Job(object):
    """Base class for the job.

    After establishing connection with the remote object, the job will
    create a remote class instance locally and enter an infinite loop,
    waiting for commands from the remote object.
    """

    def __init__(self, worker_address):
        self.job_is_alive = True
        self.worker_address = worker_address
        self._create_sockets()

    def _create_sockets(self):
        """Create two sockets for each job.

        (1) reply_socket: receives the command(i.e, the function name and
            args) from the actual class instance, and returns the result of
            the function.
        (2) job_socket: sends job_address and heartbeat_address to worker.

        """

        self.ctx = zmq.Context()

        # reply_socket: receives class, parameters and call function from
        # @remote.class and send computed results to the @remote.class.
        self.reply_socket = self.ctx.socket(zmq.REP)
        self.reply_socket.linger = 0

        job_port = self.reply_socket.bind_to_random_port(addr="tcp://*")
        self.job_ip = get_ip_address()
        self.job_address = "{}:{}".format(self.job_ip, job_port)

        reply_thread = threading.Thread(
            target=self._reply_heartbeat,
            args=("worker {}".format(self.worker_address), ))
        reply_thread.setDaemon(True)
        reply_thread.start()

    def _reply_heartbeat(self, target):
        """reply heartbeat signals to the target"""

        socket = self.ctx.socket(zmq.REP)
        socket.setsockopt(zmq.RCVTIMEO,
                          remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)
        socket.linger = 0
        heartbeat_worker_port = socket.bind_to_random_port(addr="tcp://*")
        heartbeat_worker_address = "{}:{}".format(self.job_ip,
                                                  heartbeat_worker_port)

        # job_socket: sends job_address and heartbeat_address to worker
        job_socket = self.ctx.socket(zmq.REQ)
        job_socket.connect("tcp://{}".format(self.worker_address))
        job_socket.send_multipart([
            remote_constants.NORMAL_TAG,
            to_byte(self.job_address),
            to_byte(heartbeat_worker_address),
            to_byte(str(os.getpid()))
        ])
        _ = job_socket.recv_multipart()

        # a flag to decide when to exit heartbeat loop
        self.worker_is_alive = True
        while self.worker_is_alive and self.job_is_alive:
            try:
                message = socket.recv_multipart()
                socket.send_multipart([remote_constants.HEARTBEAT_TAG])

            except zmq.error.Again as e:
                logger.warning("[Job] Cannot connect to {}. ".format(target) +
                               "Job will quit.")
                self.worker_is_alive = False
                self.job_is_alive = False

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
                logger.warning(message)
                raise NotImplementedError

    def wait_for_connection(self):
        """Wait for connection from the remote object.

        The remote object will send its class information and initialization
        arguments to the job, these parameters are then used to create a
        local instance in the job process.

        Returns:
            A local instance of the remote class object.
        """

        while True:
            message = self.reply_socket.recv_multipart()
            tag = message[0]
            if tag == remote_constants.INIT_OBJECT_TAG:
                cls = cloudpickle.loads(message[1])
                args, kwargs = cloudpickle.loads(message[2])
                obj = cls(*args, **kwargs)
                self.reply_socket.send_multipart([remote_constants.NORMAL_TAG])
                return obj
            else:
                logger.error("Message from job {}".format(message))
                raise NotImplementedError

    def run(self):
        """An infinite loop waiting for commands from the remote object.

        Each job will receive two kinds of message from the remote object:

        1. When the remote object calls a function, job will run the
           function on the local instance and return the results to the
           remote object.
        2. When the remote object is deleted, the job will quit and release
           related computation resources.
        """

        # receive files
        envdir = self.wait_for_files()
        sys.path.append(envdir)

        obj = self.wait_for_connection()

        while self.job_is_alive:
            message = self.reply_socket.recv_multipart()
            tag = message[0]

            if tag == remote_constants.CALL_TAG:
                assert obj is not None
                try:
                    function_name = to_str(message[1])
                    data = message[2]
                    args, kwargs = loads_argument(data)
                    ret = getattr(obj, function_name)(*args, **kwargs)
                    ret = dumps_return(ret)

                    self.reply_socket.send_multipart(
                        [remote_constants.NORMAL_TAG, ret])

                except Exception as e:
                    error_str = str(e)
                    logger.error(error_str)
                    self.job_is_alive = False

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

                    else:
                        traceback_str = str(traceback.format_exc())
                        logger.error("traceback:\n{}".format(traceback_str))
                        self.reply_socket.send_multipart([
                            remote_constants.EXCEPTION_TAG,
                            to_byte(error_str + "\ntraceback:\n" +
                                    traceback_str)
                        ])

            # receive DELETE_TAG from actor, and stop replying worker heartbeat
            elif tag == remote_constants.KILLJOB_TAG:
                self.reply_socket.send_multipart([remote_constants.NORMAL_TAG])
                self.job_is_alive = False
                logger.warning("An actor exits and will quit job {}.".format(
                    self.job_address))
            else:
                logger.error("Job message: {}".format(message))
                raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_address", required=True, type=str, help="worker_address")
    args = parser.parse_args()
    job = Job(args.worker_address)
    job.run()
