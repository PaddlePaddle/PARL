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
import sys
import threading
import zmq

from parl.utils import logger, to_str, to_byte
from parl.remote.communication import loads_argument, loads_return,\
    dumps_argument, dumps_return
from parl.remote.client import get_global_client
from parl.remote import remote_constants
from parl.remote.exceptions import RemoteError, RemoteAttributeError,\
    RemoteDeserializeError, RemoteSerializeError, ResourceError, FutureFunctionError
from parl.remote.future_mode.actor_ref_monitor import ActorRefMonitor
from parl.remote.remote_class_serialization import dump_remote_class

XPARL_RESERVED_PREFIX = "_xparl"


class RemoteWrapper(object):
    """
    Wrapper for remote class in client side.
    1. request cpu resource and submit job to the master
    2. return the result of the function called by the user, using the remote computation resource.
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

        # instance of actor class which is decorated by @remote_class(wait=False).
        # use the reference count of the object to detect whether
        # the object has been deleted or out of scope.
        proxy_wrapper_nowait_object = kwargs.get('_xparl_proxy_wrapper_nowait')
        if proxy_wrapper_nowait_object is None:
            actor_ref_monitor = None
        else:
            actor_ref_monitor = ActorRefMonitor(proxy_wrapper_nowait_object)

        # class which is decorated by @remote_class
        cls = kwargs.get('_xparl_remote_class')

        # max_memory argument in @remote_class decorator
        max_memory = kwargs.get('_xparl_remote_class_max_memory')

        if self.GLOBAL_CLIENT.master_is_alive:
            job_address = self.request_cpu_resource(
                self.GLOBAL_CLIENT, max_memory, actor_ref_monitor)
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

        for key in list(kwargs.keys()):
            if key.startswith(XPARL_RESERVED_PREFIX):
                del kwargs[key]

        self.job_socket.send_multipart([
            remote_constants.INIT_OBJECT_TAG,
            dump_remote_class(cls),
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
            if not self.job_shutdown:
                self.job_socket.setsockopt(zmq.RCVTIMEO, 1 * 1000)

                self.job_socket.send_multipart([remote_constants.KILLJOB_TAG])
                _ = self.job_socket.recv_multipart()
                self.job_socket.close(0)

        except AttributeError:
            pass
        except zmq.error.ZMQError:
            pass
        except TypeError:
            pass

    def has_attr(self, attr):
        has_attr = attr in self.remote_attribute_keys_set
        return has_attr

    def get_attrs(self):
        return self.remote_attribute_keys_set

    def send_file(self, socket):
        try:
            socket.send_multipart(
                [remote_constants.SEND_FILE_TAG, self.GLOBAL_CLIENT.pyfiles])
            _ = socket.recv_multipart()
        except zmq.error.Again as e:
            logger.error("Send python files failed.")

    def request_cpu_resource(self, global_client, max_memory,
                             actor_ref_monitor):
        """Try to request cpu resource for 1 second/time for 300 times."""
        cnt = 300
        while cnt > 0:
            job_address = global_client.submit_job(max_memory,
                                                   actor_ref_monitor)
            if job_address is not None:
                return job_address
            if cnt % 30 == 0:
                logger.warning("No vacant cpu resources at the moment, "
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
                raise RemoteError(attr,
                                  "This actor losts connection with the job.")
            self.internal_lock.acquire()
            if is_attribute:
                self.job_socket.send_multipart(
                    [remote_constants.GET_ATTRIBUTE_TAG,
                     to_byte(attr)])
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
                    self.remote_attribute_keys_set = loads_return(message[2])
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
