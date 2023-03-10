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
import multiprocessing as mp

from parl.utils import logger, to_str, to_byte
from parl.remote.communication import loads_argument, loads_return,\
    dumps_argument, dumps_return
from parl.remote.client import get_global_client
from parl.remote import remote_constants
from parl.remote.exceptions import RemoteError, RemoteAttributeError,\
    RemoteDeserializeError, RemoteSerializeError, ResourceError, FutureFunctionError
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

        # class which is decorated by @remote_class
        cls = kwargs.get('_xparl_remote_class')

        # max_memory argument in @remote_class decorator
        max_memory = kwargs.get('_xparl_remote_class_max_memory')
        n_gpu = kwargs.get('_xparl_remote_class_n_gpu', 0)
        self.job_is_alive = mp.Value('i', True)

        if self.GLOBAL_CLIENT.connected_to_master:
            job_info = self.request_resource(self.GLOBAL_CLIENT, max_memory, n_gpu)
        else:
            raise Exception("Can not submit job to the master. " "Please check if master is still alive.")

        if job_info is None:
            raise ResourceError("Cannot submit the job to the master. "
                                "Please add more computation resources to the "
                                "master or try again later.")
        job_address = job_info.job_address

        self.internal_lock = threading.Lock()

        # Send actor commands like `init` and `call` to the job.
        self.job_socket = self.ctx.socket(zmq.REQ)
        self.job_socket.linger = 0
        self.job_socket.connect("tcp://{}".format(job_address))
        # check the result every 20s to detect the job is still alive.
        self.job_socket.setsockopt(zmq.RCVTIMEO, 20 * 1000)
        self.job_address = job_address

        self.send_file()

        for key in list(kwargs.keys()):
            if key.startswith(XPARL_RESERVED_PREFIX):
                del kwargs[key]
        serlization_finished = True
        try:
            self.job_socket.send_multipart([
                remote_constants.INIT_OBJECT_TAG,
                dump_remote_class(cls),
                cloudpickle.dumps([args, kwargs]),
            ])
        except TypeError:
            serlization_finished = False
            logger.error("[xparl] fail to serialize the arguments for class initialization. \n\
                        For more information, please check the documentation in: \n\
                        https://parl.readthedocs.io/en/latest/questions/distributed_training.html#recommended-data-types-in-xparl"
                         )
        if not serlization_finished:
            raise RemoteSerializeError('__init__', "fail to finish serialization.")

        message = self._receive_from_remote_instance('__init__')
        tag = message[0]
        if tag == remote_constants.NORMAL_TAG:
            self.remote_attribute_keys_set = loads_return(message[1])
        elif tag == remote_constants.EXCEPTION_TAG:
            traceback_str = to_str(message[1])
            self.job_is_alive.value = False
            raise RemoteError('__init__', traceback_str)
        else:
            pass

    def __del__(self):
        """Delete the remote class object and release remote resources."""
        try:
            if self.job_is_alive.value == True:
                self.job_socket.setsockopt(zmq.RCVTIMEO, 1 * 1000)
                self.job_socket.send_multipart([remote_constants.KILLJOB_TAG])
                _ = self.job_socket.recv_multipart()
                self.job_socket.close(0)
        except Exception as e:
            pass

    def has_attr(self, attr):
        has_attr = attr in self.remote_attribute_keys_set
        return has_attr

    def get_attrs(self):
        return self.remote_attribute_keys_set

    def send_file(self):
        try:
            self.job_socket.send_multipart([remote_constants.SEND_FILE_TAG, self.GLOBAL_CLIENT.pyfiles])
            _ = self.job_socket.recv_multipart()
        except zmq.error.Again as e:
            logger.error("[Client] Fail to send python files to the remote instance.")

    def request_resource(self, global_client, max_memory, n_gpu):
        """Try to request cpu resource for 1 second/time for 300 times."""
        cnt = 300
        while cnt > 0:
            job_info = global_client.submit_job(max_memory, n_gpu, self.job_is_alive)
            if job_info is not None:
                return job_info
            if cnt % 30 == 0:
                logger.warning("No vacant cpu/gpu resources at the moment, " "will try {} times later.".format(cnt))
            cnt -= 1
        return None

    def set_remote_attr(self, attr, value):
        self.internal_lock.acquire()
        self.job_socket.send_multipart([remote_constants.SET_ATTRIBUTE_TAG, to_byte(attr), dumps_return(value)])
        message = self._receive_from_remote_instance(attr)
        tag = message[0]
        if tag == remote_constants.NORMAL_TAG:
            self.remote_attribute_keys_set = loads_return(message[1])
            self.internal_lock.release()
        else:
            self.job_is_alive.value = False
            raise NotImplementedError()
        return
    
    def _receive_from_remote_instance(self, attr):
        """Receive message from remote instance while checking the job status every  20 seconds.
        """
        message = None
        while True:
            try:
                message = self.job_socket.recv_multipart()
                return message
            except zmq.error.Again as e:
                pass
            if not self.job_is_alive.value:
                raise RemoteError(attr, "This instance is disconncted with the remote instance.")
        return message

    def get_remote_attr(self, attr):
        """Call the function of the unwrapped class."""
        #check if attr is a attribute or a function
        is_attribute = attr in self.remote_attribute_keys_set

        def wrapper(*args, **kwargs):
            if not self.job_is_alive.value:
                raise RemoteError(attr, "This instance is disconncted with the remote instance.")
            self.internal_lock.acquire()
            if is_attribute:
                self.job_socket.send_multipart([remote_constants.GET_ATTRIBUTE_TAG, to_byte(attr)])
            else:
                data = dumps_argument(*args, **kwargs)
                self.job_socket.send_multipart([remote_constants.CALL_TAG, to_byte(attr), data])

            message = self._receive_from_remote_instance(attr)
            tag = message[0]

            if tag == remote_constants.NORMAL_TAG:
                ret = loads_return(message[1])
                if not is_attribute:
                    self.remote_attribute_keys_set = loads_return(message[2])
                self.internal_lock.release()
                return ret

            elif tag == remote_constants.EXCEPTION_TAG:
                error_str = to_str(message[1])
                self.job_is_alive.value = False
                raise RemoteError(attr, error_str)

            elif tag == remote_constants.ATTRIBUTE_EXCEPTION_TAG:
                error_str = to_str(message[1])
                self.job_is_alive.value = False
                raise RemoteAttributeError(attr, error_str)

            elif tag == remote_constants.SERIALIZE_EXCEPTION_TAG:
                error_str = to_str(message[1])
                self.job_is_alive.value = False
                raise RemoteSerializeError(attr, error_str)

            elif tag == remote_constants.DESERIALIZE_EXCEPTION_TAG:
                error_str = to_str(message[1])
                self.job_is_alive.value = False
                raise RemoteDeserializeError(attr, error_str)

            else:
                self.job_is_alive.value = False
                raise NotImplementedError()

        return wrapper() if is_attribute else wrapper
