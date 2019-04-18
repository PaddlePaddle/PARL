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

import threading
import zmq
from parl.remote import remote_constants
from parl.remote.exceptions import *
from parl.utils import logger, to_str, to_byte
from parl.utils.communication import dumps_argument, loads_return


class RemoteObject(object):
    """
    Provides interface to call functions of object in remote client.
    """

    def __init__(self, remote_client_address, zmq_context=None):
        """
        Args:
            remote_client_address: address(ip:port) of remote client
            zmq_context: zmq.Context()
        """
        if zmq_context is None:
            self.zmq_context = zmq.Context()
        else:
            self.zmq_context = zmq_context

        # socket for sending function call to remote object and receiving result
        self.command_socket = None
        # lock for thread safety
        self.internal_lock = threading.Lock()
        self._connect_remote_client(remote_client_address)

    def _connect_remote_client(self, remote_client_address):
        """
        Build connection with the remote client to send function call.
        """
        socket = self.zmq_context.socket(zmq.REQ)
        logger.info("[connect_remote_client] client_address:{}".format(
            remote_client_address))
        socket.connect("tcp://{}".format(remote_client_address))
        self.command_socket = socket
        self.command_socket.linger = 0

    def __getattr__(self, attr):
        """
        Provides interface to call functions of object in remote client.
            1. send fucntion name and packed auguments to remote client;
            2. remote clinet execute the function of the object really;
            3. receive function return from remote client.

        Args:
            attr(str): a function name specify which function to run.
        """

        def wrapper(*args, **kwargs):
            self.internal_lock.acquire()

            data = dumps_argument(*args, **kwargs)

            self.command_socket.send_multipart(
                [remote_constants.NORMAL_TAG,
                 to_byte(attr), data])

            message = self.command_socket.recv_multipart()
            tag = message[0]
            if tag == remote_constants.NORMAL_TAG:
                ret = loads_return(message[1])
            elif tag == remote_constants.EXCEPTION_TAG:
                error_str = to_str(message[1])
                raise RemoteError(attr, error_str)
            elif tag == remote_constants.ATTRIBUTE_EXCEPTION_TAG:
                error_str = to_str(message[1])
                raise RemoteAttributeError(attr, error_str)
            elif tag == remote_constants.SERIALIZE_EXCEPTION_TAG:
                error_str = to_str(message[1])
                raise RemoteSerializeError(attr, error_str)
            elif tag == remote_constants.DESERIALIZE_EXCEPTION_TAG:
                error_str = to_str(message[1])
                raise RemoteDeserializeError(attr, error_str)
            else:
                raise NotImplementedError()

            self.internal_lock.release()
            return ret

        return wrapper
