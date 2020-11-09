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
import sys

from parl.utils import get_ip_address, logger, to_str, to_byte
from parl.utils.communication import loads_argument, loads_return,\
    dumps_argument, dumps_return
from parl.remote import remote_constants
from parl.remote.exceptions import RemoteError, RemoteAttributeError,\
    RemoteDeserializeError, RemoteSerializeError, ResourceError
from parl.remote.client import get_global_client
from parl.remote.utils import locate_remote_file


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
        # we are not going to create a remote actor in job.py
        if 'XPARL' in os.environ and os.environ['XPARL'] == 'True':
            logger.warning(
                "Note: this object will be runnning as a local object")
            return cls

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
                self.remote_attribute_keys_set = set()
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
                module_path = inspect.getfile(cls)
                if module_path.endswith('pyc'):
                    module_path = module_path[:-4]
                elif module_path.endswith('py'):
                    module_path = module_path[:-3]
                else:
                    raise FileNotFoundError(
                        "cannot not find the module:{}".format(module_path))
                res = inspect.getfile(cls)
                file_path = locate_remote_file(module_path)
                cls_source = inspect.getsourcelines(cls)
                end_of_file = cls_source[1] + len(cls_source[0])
                class_name = cls.__name__
                self.job_socket.send_multipart([
                    remote_constants.INIT_OBJECT_TAG,
                    cloudpickle.dumps([file_path, class_name, end_of_file]),
                    cloudpickle.dumps([args, kwargs]),
                ])
                message = self.job_socket.recv_multipart()
                tag = message[0]
                if tag == remote_constants.NORMAL_TAG:
                    self.remote_attribute_keys_set = loads_return(message[1])
                elif tag == remote_constants.EXCEPTION_TAG:
                    traceback_str = to_str(message[1])
                    self.job_shutdown = True
                    raise RemoteError('__init__', traceback_str)
                else:
                    pass

            def __del__(self):
                """Delete the remote class object and release remote resources."""
                try:
                    self.job_socket.setsockopt(zmq.RCVTIMEO, 1 * 1000)
                except AttributeError:
                    pass
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
                    except TypeError:
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

            def set_remote_attr(self, attr, value):
                self.internal_lock.acquire()
                self.job_socket.send_multipart([
                    remote_constants.SET_ATTRIBUTE_TAG,
                    to_byte(attr),
                    dumps_return(value)
                ])
                message = self.job_socket.recv_multipart()
                tag = message[0]
                if tag == remote_constants.NORMAL_TAG:
                    self.remote_attribute_keys_set = loads_return(message[1])
                    self.internal_lock.release()
                else:
                    self.job_shutdown = True
                    raise NotImplementedError()
                return

            def get_remote_attr(self, attr):
                """Call the function of the unwrapped class."""
                #check if attr is a attribute or a function
                is_attribute = attr in self.remote_attribute_keys_set

                def wrapper(*args, **kwargs):
                    if self.job_shutdown:
                        raise RemoteError(
                            attr, "This actor losts connection with the job.")
                    self.internal_lock.acquire()
                    if is_attribute:
                        self.job_socket.send_multipart([
                            remote_constants.GET_ATTRIBUTE_TAG,
                            to_byte(attr)
                        ])
                    else:
                        data = dumps_argument(*args, **kwargs)
                        self.job_socket.send_multipart(
                            [remote_constants.CALL_TAG,
                             to_byte(attr), data])

                    message = self.job_socket.recv_multipart()
                    tag = message[0]

                    if tag == remote_constants.NORMAL_TAG:
                        ret = loads_return(message[1])
                        if not is_attribute:
                            self.remote_attribute_keys_set = loads_return(
                                message[2])
                        self.internal_lock.release()
                        return ret

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

                return wrapper() if is_attribute else wrapper

        def proxy_wrapper_func(remote_wrapper):
            '''
            The 'proxy_wrapper_func' is defined on the top of class 'RemoteWrapper'
            in order to set and get attributes of 'remoted_wrapper' and the corresponding 
            remote models individually. 

            With 'proxy_wrapper_func', it is allowed to define a attribute (or method) of
            the same name in 'RemoteWrapper' and remote models.
            '''

            class ProxyWrapper(object):
                def __init__(self, *args, **kwargs):
                    self.xparl_remote_wrapper_obj = remote_wrapper(
                        *args, **kwargs)

                def __getattr__(self, attr):
                    return self.xparl_remote_wrapper_obj.get_remote_attr(attr)

                def __setattr__(self, attr, value):
                    if attr == 'xparl_remote_wrapper_obj':
                        super(ProxyWrapper, self).__setattr__(attr, value)
                    else:
                        self.xparl_remote_wrapper_obj.set_remote_attr(
                            attr, value)

            return ProxyWrapper

        RemoteWrapper._original = cls
        proxy_wrapper = proxy_wrapper_func(RemoteWrapper)
        return proxy_wrapper

    max_memory = kwargs.get('max_memory')
    """
        Users may pass some arguments to the decorator (e.g., parl.remote_class(10)).
        The following code tries to handle this issue.

        The `args` is different in the following two decorating way, and we should return different wrapper.
        @parl.remote_class     -> args: (<class '__main__.Actor'>,) -> we should return decorator(cls)
        @parl.remote_class(10) -> args: (10,)                       -> we should return decorator
    """
    if len(args) == 1 and callable(args[0]):  # args[0]: cls
        # The first element in the `args` is a class, we should return decorator(cls)
        return decorator(args[0])

    # The first element in the `args` is not a class, we should return decorator
    return decorator
