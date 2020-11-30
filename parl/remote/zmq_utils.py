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
from parl.remote import remote_constants


def create_server_socket(ctx, heartbeat_timeout=False):
    """Create a server socket with a random port (support raising timeout exception).

    Args:
        ctx(zmq.Context()): context of zmq
        heartbeat_timeout(bool): whether to set the timeout(HEARTBEAT_RCVTIMEO_S) for
               receiving operation on the server socket. (The default value is False)

    Returns:
        socket(zmq.Context().socket): socket of the server.
        port(int): port of the server socket.
    """
    socket = ctx.socket(zmq.REP)
    if heartbeat_timeout:
        socket.setsockopt(zmq.RCVTIMEO,
                          remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)
    socket.linger = 0
    port = socket.bind_to_random_port(addr="tcp://*")
    return socket, port


def create_client_socket(ctx, server_socket_address, heartbeat_timeout=False):
    """Create a client socket to connect the `server_socket_address`
    (support raising timeout exception).

    Args:
        ctx(zmq.Context()): context of zmq
        server_socket_address(str): address of server socket
        heartbeat_timeout(bool): whether to set the timeout(HEARTBEAT_RCVTIMEO_S) for
               sending operation on the client socket. (The default value is False)

    Returns:
        socket(zmq.Context().socket): socket of the client.
    """
    socket = ctx.socket(zmq.REQ)
    if heartbeat_timeout:
        socket.setsockopt(zmq.RCVTIMEO,
                          remote_constants.HEARTBEAT_RCVTIMEO_S * 1000)
    socket.linger = 0
    socket.connect("tcp://{}".format(server_socket_address))

    return socket
