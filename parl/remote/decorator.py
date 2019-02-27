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

import zmq
from parl.utils import logger
import numpy as np
from parl.utils.communication import dumps_argument, loads_argument
from parl.utils.communication import dumps_return, loads_return
from parl.utils.machine_info import get_ip_address
import pyarrow
import threading
"""
Three steps to create a remote class -- 
1. add a decroator(@virtual) before the definition of the class.
2. create an instance of remote class
3. call function `remote_run` with server address

@virtual 
Class Simulator(object):
    ...

sim = Simulator()
sim.remote_run(server_ip='172.18.202.45', port=8001)

"""


def virtual(cls, location='client'):
    """
    Class wrapper for wrapping a normal class as a remote class that can run in different machines.
    Two kinds of wrapper are provided for the client as well as the server.

    Args:
        location(str): specify which wrapper to use, available locations: client/server.
                       users are expected to use `client`.
    """
    assert location in ['client', 'server'], \
        'Remote Class has to be placed at client side or server side.'

    class ClientWrapper(object):
        """
        Wrapper for remote class at client side. After the decoration, 
        the initial class is able to be called to run any function at sever side.
        """

        def __init__(self, *args):
            """
            Args:
                args: arguments for the initialisation of the initial class.
            """
            self.unwrapped = cls(*args)
            self.conect_socket = None
            self.reply_socket = None

        def create_reply_socket(self):
            """
            In fact, we have also a socket server in client side. This server keeps running 
            and waits for requests (e.g. call a function) from server side.
            """
            client_ip = get_ip_address()
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            free_port = None
            for port in range(6000, 8000):
                try:
                    socket.bind("tcp://*:{}".format(port))
                    logger.info(
                        "[create_reply_socket] free_port:{}".format(port))
                    free_port = port
                    break
                except zmq.error.ZMQError:
                    logger.warn(
                        "[create_reply_socket]cannot bind port:{}, retry".
                        format(port))
            if free_port is not None:
                return socket, client_ip, free_port
            else:
                logger.error(
                    "cannot find any available port from 6000 to 8000")
                sys.exit(1)

        def connect_server(self, server_ip, server_port):
            """
            create the connection between client side and server side.

            Args:
                server_ip(str): the ip of the server.
                server_port(int): the connection port of the server.
            """
            self.reply_socket, local_ip, local_port = self.create_reply_socket(
            )

            logger.info("connecting {}:{}".format(server_ip, server_port))
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect("tcp://{}:{}".format(server_ip, server_port))
            client_id = np.random.randint(int(1e18))
            logger.info("client_id:{}".format(client_id))
            socket.send_string('{}:{} {}'.format(local_ip, local_port,
                                                 client_id))
            message = socket.recv_string()
            logger.info("[connect_server] done, message from server:{}".format(
                message))
            self.connect_socket = socket

        def __getattr__(self, attr):
            """
            Call the function of the initial class. The wrapped class do not have 
            same functions as the unwrapped one. 
            We have to call the function of the function in unwrapped class,
            This implementation utilise a function wrapper.
            """
            if hasattr(self.unwrapped, attr):

                def wrapper(*args, **kw):
                    return getattr(self.unwrapped, attr)(*args, **kw)

                return wrapper
            raise AttributeError(attr)

        def remote_run(self, server_ip, server_port):
            """
            connect server and wait for requires of running functions from server side.

            Args:
                server_ip(str): server's ip
                server_port(int): server's port
            """
            self.connect_server(server_ip, server_port)
            while True:
                function_name = self.reply_socket.recv_string()
                self.reply_socket.send_string("OK")
                data = self.reply_socket.recv()
                args, kw = loads_argument(data)
                ret = getattr(self.unwrapped, function_name)(*args, **kw)
                ret = dumps_return(ret)
                self.reply_socket.send(ret)

    class ServerWrapper(object):
        """
        Wrapper for remote class at server side. 
        """

        def __init__(self, *args):
            """
            Args:
                args: arguments used to initialize the initial class
            """
            self.unwrapped = (cls(*args)).unwrapped
            self.command_socket = None
            self.internal_lock = threading.Lock()

        def connect_client(self, client_info):
            """
            build another connection with the client to send command to the client.
            """
            client_address, client_id = client_info.split(' ')
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            logger.info(
                "[connect_client] client_address:{}".format(client_address))
            socket.connect("tcp://{}".format(client_address))
            self.command_socket = socket
            self.client_id = client_id

        def __getattr__(self, attr):
            """
            Run the function at client side. we also implement this through a wrapper.

            Args:
                attr(str): a function name specify which function to run.
            """
            if hasattr(self.unwrapped, attr):

                def wrapper(*args, **kw):
                    self.internal_lock.acquire()
                    self.command_socket.send_string(attr)
                    self.command_socket.recv_string()
                    data = dumps_argument(*args, **kw)
                    self.command_socket.send(data)
                    ret = self.command_socket.recv()
                    ret = loads_return(ret)
                    self.internal_lock.release()
                    return ret

                return wrapper
            return ret

    if location == 'client':
        return ClientWrapper
    else:
        return ServerWrapper
