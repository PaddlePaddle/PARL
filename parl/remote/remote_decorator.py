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
import os
import threading
import time
import zmq
import numpy as np
import inspect

from parl.utils import get_ip_address, logger, to_str, to_byte
from parl.utils.communication import loads_argument, loads_return,\
    dumps_argument, dumps_return
from parl.remote import remote_constants
from parl.remote.exceptions import RemoteError, RemoteAttributeError,\
    RemoteDeserializeError, RemoteSerializeError, ResourceError
from parl.remote.client import get_global_client


def remote_class(*args, **kwargs):
    """A Python decorator that enables a class to run all its functions
    remotely.

    Each instance of the remote class can be seemed as a task submitted
    to the cluster by the global client, which is created automatically
    when we call parl.connect(master_address). After global client
    submits the task, the master node will send an available job address
    to this remote instance. Then the remote object will send local python
    files, class definition and initialization arguments to the related job.

    In this way, we can run distributed applications easily and efficiently.

    .. code-block:: python

        @parl.remote_class
        class Actor(object):
            def __init__(self, x):
                self.x = x

            def step(self):
                self.x += 1
                return self.x

        actor = Actor()
        actor.step()

        # Set maximum memory usage to 300 MB for each object.
        @parl.remote_class(max_memory=300)
        class LimitedActor(object):
           ...

    Args:
        max_memory (float): Maximum memory (MB) can be used by each remote
                            instance, the unit is in MB and default value is
                            none(unlimited).

    Returns:
        A remote wrapper for the remote class.

    Raises:
        Exception: An exception is raised if the client is not created
                   by `parl.connect(master_address)` beforehand.
    """

    def decorator(cls):
        class RemoteWrapper(object):
            """
            Wrapper for remote class in client side.
            """

            def __init__(self, *args, **kwargs):
                """
                Args:
                    args, kwargs: arguments for the initialization of the unwrapped
                    class.
                """
                self.GLOBAL_CLIENT = get_global_client()

                self.ctx = self.GLOBAL_CLIENT.ctx

                # GLOBAL_CLIENT will set `master_is_alive` to False when hearbeat
                # finds the master is dead.
                if self.GLOBAL_CLIENT.master_is_alive:
                    job_address = self.request_cpu_resource(
                        self.GLOBAL_CLIENT, max_memory)
                else:
                    raise Exception("Can not submit job to the master. "
                                    "Please check if master is still alive.")

                if job_address is None:
                    raise ResourceError("Cannot submit the job to the master. "
                                        "Please add more CPU resources to the "
                                        "master or try again later.")

                self.internal_lock = threading.Lock()

                # Send actor commands like `init` and `call` to the job.
                self.job_socket = self.ctx.socket(zmq.REQ)
                self.job_socket.linger = 0
                self.job_socket.connect("tcp://{}".format(job_address))
                self.job_address = job_address
                self.job_shutdown = False

                self.send_file(self.job_socket)
                file_name = inspect.getfile(cls)[:-3]
                class_name = cls.__name__
                self.job_socket.send_multipart([
                    remote_constants.INIT_OBJECT_TAG,
                    cloudpickle.dumps([file_name, class_name]),
                    cloudpickle.dumps([args, kwargs]),
                ])
                message = self.job_socket.recv_multipart()
                tag = message[0]
                if tag == remote_constants.EXCEPTION_TAG:
                    traceback_str = to_str(message[1])
                    self.job_shutdown = True
                    raise RemoteError('__init__', traceback_str)

            def __del__(self):
                """Delete the remote class object and release remote resources."""
                self.job_socket.setsockopt(zmq.RCVTIMEO, 1 * 1000)
                if not self.job_shutdown:
                    try:
                        self.job_socket.send_multipart(
                            [remote_constants.KILLJOB_TAG])
                        _ = self.job_socket.recv_multipart()
                        self.job_socket.close(0)
                    except AttributeError:
                        pass
                    except zmq.error.ZMQError:
                        pass

            def send_file(self, socket):
                try:
                    socket.send_multipart([
                        remote_constants.SEND_FILE_TAG,
                        self.GLOBAL_CLIENT.pyfiles
                    ])
                    _ = socket.recv_multipart()
                except zmq.error.Again as e:
                    logger.error("Send python files failed.")

            def request_cpu_resource(self, global_client, max_memory):
                """Try to request cpu resource for 1 second/time for 300 times."""
                cnt = 300
                while cnt > 0:
                    job_address = global_client.submit_job(max_memory)
                    if job_address is not None:
                        return job_address
                    if cnt % 30 == 0:
                        logger.warning(
                            "No vacant cpu resources at the moment, "
                            "will try {} times later.".format(cnt))
                    cnt -= 1
                return None

            def __getattr__(self, attr):
                """Call the function of the unwrapped class."""

                def wrapper(*args, **kwargs):
                    if self.job_shutdown:
                        raise RemoteError(
                            attr, "This actor losts connection with the job.")
                    self.internal_lock.acquire()
                    data = dumps_argument(*args, **kwargs)

                    self.job_socket.send_multipart(
                        [remote_constants.CALL_TAG,
                         to_byte(attr), data])

                    message = self.job_socket.recv_multipart()
                    tag = message[0]

                    if tag == remote_constants.NORMAL_TAG:
                        ret = loads_return(message[1])

                    elif tag == remote_constants.EXCEPTION_TAG:
                        error_str = to_str(message[1])
                        self.job_shutdown = True
                        raise RemoteError(attr, error_str)

                    elif tag == remote_constants.ATTRIBUTE_EXCEPTION_TAG:
                        error_str = to_str(message[1])
                        self.job_shutdown = True
                        raise RemoteAttributeError(attr, error_str)

                    elif tag == remote_constants.SERIALIZE_EXCEPTION_TAG:
                        error_str = to_str(message[1])
                        self.job_shutdown = True
                        raise RemoteSerializeError(attr, error_str)

                    elif tag == remote_constants.DESERIALIZE_EXCEPTION_TAG:
                        error_str = to_str(message[1])
                        self.job_shutdown = True
                        raise RemoteDeserializeError(attr, error_str)

                    else:
                        self.job_shutdown = True
                        raise NotImplementedError()

                    self.internal_lock.release()
                    return ret

                return wrapper

        RemoteWrapper._original = cls
        return RemoteWrapper

    max_memory = kwargs.get('max_memory')
    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])
    return decorator
