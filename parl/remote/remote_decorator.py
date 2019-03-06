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

import numpy as np
import pyarrow
import threading
import time
import zmq
from parl.remote import remote_constants
from parl.utils import get_ip_address, logger, to_str, to_byte, SerializeError, DeserializeError
from parl.utils.communication import loads_argument, dumps_return
"""
Three steps to create a remote class:
1. add a decroator(@parl.remote) before the definition of the class;
2. create an instance of remote class;
3. call function `as_remote` with server address.

@parl.remote
Class Simulator(object):
    ...

sim = Simulator()
sim.as_remote(server_ip='172.18.202.45', port=8001)

"""


def remote(cls):
    class ClientWrapper(object):
        """
        Wrapper for remote class in client side.
        when as_remote function called, the object initialized in the client can 
        handle function call from server.
        """

        def __init__(self, *args, **kwargs):
            """
            Args:
                args, kwargs: arguments for the initialisation of the unwrapped class.
            """
            self.unwrapped = cls(*args, **kwargs)

            self.zmq_context = None
            self.poller = None

            # socket for connecting server and telling ip and port of client to server
            self.connect_socket = None
            # socket for handle function call from server side
            self.reply_socket = None

        def _create_reply_socket(self, remote_ip, remote_port):
            """
            In fact, we also have a socket server in client side. This server keeps running 
            and waits for requests (e.g. call a function) from server side.
            """
            if remote_ip is None:
                client_ip = get_ip_address()
            else:
                client_ip = remote_ip

            self.zmq_context = zmq.Context()
            socket = self.zmq_context.socket(zmq.REP)

            free_port = None
            if remote_port is None:
                for port in range(6000, 8000):
                    try:
                        socket.bind("tcp://*:{}".format(port))
                        logger.info(
                            "[_create_reply_socket] free_port:{}".format(port))
                        free_port = port
                        break
                    except zmq.error.ZMQError:
                        logger.warning(
                            "[_create_reply_socket]cannot bind port:{}, retry".
                            format(port))
            else:
                socket.bind("tcp://*:{}".format(remote_port))
                free_port = remote_port

            if free_port is not None:
                return socket, client_ip, free_port
            else:
                logger.error(
                    "cannot find any available port from 6000 to 8000")
                sys.exit(1)

        def _connect_server(self, server_ip, server_port, remote_ip,
                            remote_port):
            """
            Create the connection between client side and server side.

            Args:
                server_ip(str): the ip of the server.
                server_port(int): the connection port of the server.
                remote_ip: the ip of the client itself.
                remote_port: the port of the client itself, 
                           which used to create reply socket.
            """
            self.reply_socket, local_ip, local_port = self._create_reply_socket(
                remote_ip, remote_port)
            self.reply_socket.linger = 0

            socket = self.zmq_context.socket(zmq.REQ)
            socket.connect("tcp://{}:{}".format(server_ip, server_port))

            client_id = np.random.randint(int(1e18))
            logger.info("client_id:{}".format(client_id))
            logger.info("connecting {}:{}".format(server_ip, server_port))

            client_info = '{}:{} {}'.format(local_ip, local_port, client_id)
            socket.send_multipart(
                [remote_constants.CONNECT_TAG,
                 to_byte(client_info)])

            message = socket.recv_multipart()
            logger.info("[connect_server] done, message from server:{}".format(
                message))
            self.connect_socket = socket
            self.connect_socket.linger = 0

        def _exit_remote(self):
            # Following release order matters

            #self.reply_socket.close()

            #self.connect_socket.close()
            self.poller.unregister(self.connect_socket)

            #self.zmq_context.term()
            self.zmq_context.destroy()

        def _heartbeat_loop(self):
            """
            Periodically detect whether the server is alive or not
            """
            self.poller = zmq.Poller()
            self.poller.register(self.connect_socket, zmq.POLLIN)
            logger.info('[debug] poller register connect socket')

            while True:
                logger.info('[debug] connect socket send HEARTBEAT')
                self.connect_socket.send_multipart(
                    [remote_constants.HEARTBEAT_TAG])

                # wait for at most 10s to receive response
                logger.info('[debug] poller.poll')
                socks = dict(self.poller.poll(10000))

                if socks.get(self.connect_socket) == zmq.POLLIN:
                    logger.info('connect socket recv return of heartbeat')
                    _ = self.connect_socket.recv_multipart()
                else:
                    logger.warning(
                        '[HeartBeat] Server no response, will exit now!')
                    self._exit_remote()

                    break

                # HeartBeat interval 10s
                time.sleep(10)

        def __getattr__(self, attr):
            """
            Call the function of the unwrapped class.
            """

            def wrapper(*args, **kwargs):
                return getattr(self.unwrapped, attr)(*args, **kwargs)

            return wrapper

        def _reply_loop(self):
            while True:
                try:
                    message = self.reply_socket.recv_multipart()

                    try:
                        function_name = to_str(message[1])
                        data = message[2]
                        args, kwargs = loads_argument(data)
                        ret = getattr(self.unwrapped, function_name)(*args,
                                                                     **kwargs)
                        ret = dumps_return(ret)

                    except Exception as e:
                        error_str = str(e)
                        logger.error(e)

                        if type(e) == AttributeError:
                            self.reply_socket.send_multipart([
                                remote_constants.ATTRIBUTE_EXCEPTION_TAG,
                                to_byte(error_str)
                            ])
                        elif type(e) == SerializeError:
                            self.reply_socket.send_multipart([
                                remote_constants.SERIALIZE_EXCEPTION_TAG,
                                to_byte(error_str)
                            ])
                        elif type(e) == DeserializeError:
                            self.reply_socket.send_multipart([
                                remote_constants.DESERIALIZE_EXCEPTION_TAG,
                                to_byte(error_str)
                            ])
                        else:
                            self.reply_socket.send_multipart([
                                remote_constants.EXCEPTION_TAG,
                                to_byte(error_str)
                            ])

                        time.sleep(1)
                        self._exit_remote()
                        break

                    self.reply_socket.send_multipart(
                        [remote_constants.NORMAL_TAG, ret])

                except zmq.ContextTerminated:
                    logger.warning(
                        'Zmq context termnated, exiting reply loop thread.')
                    break

        def as_remote(self,
                      server_ip,
                      server_port,
                      remote_ip=None,
                      remote_port=None):
            """
            Client will connect server and wait for function calls from server side.

            Args:
                server_ip(str): server's ip
                server_port(int): server's port
                remote_ip: the ip of the client itself.
                remote_port: the port of the client itself, 
                           which used to create reply socket.
            """
            self._connect_server(server_ip, server_port, remote_ip,
                                 remote_port)

            reply_thread = threading.Thread(target=self._reply_loop)
            reply_thread.setDaemon(True)
            reply_thread.start()

            self._heartbeat_loop()

        def remote_closed(self):
            """
            Check whether as_remote mode is closed
            """
            assert self.reply_socket is not None, 'as_remote function should be called first!'
            assert self.connect_socket is not None, 'as_remote function should be called first!'
            return self.reply_socket.closed and self.connect_socket.closed

    return ClientWrapper
