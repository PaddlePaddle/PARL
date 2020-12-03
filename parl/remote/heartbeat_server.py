#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import threading
import zmq
from parl.remote import remote_constants
from parl.remote.zmq_utils import create_server_socket, create_client_socket
from parl.utils import get_ip_address, to_byte, logger


class HeartbeatServerThread(threading.Thread):
    def __init__(self, heartbeat_client_recv_addr, server_type, client_type):
        """Create a thread to run the heartbeat server.
        Note that the process will exit directly if the heartbeat detection fails.

        Args:
            heartbeat_client_recv_addr(str): address of another server socket in the heartbeat client 
                    which used to receive the heartbeat server address.
            server_type(str): type of server (E.g. master/worker/job/log_server), which is used for logging.
            client_type(str): type of client (E.g. master/worker/job/log_server), which is used for logging.
        """
        assert server_type in ['master', 'worker', 'job', 'log_server']
        assert client_type in ['master', 'worker', 'job', 'log_server']

        self.heartbeat_client_recv_addr = heartbeat_client_recv_addr
        self.server_type = server_type
        self.client_type = client_type
        self.ctx = zmq.Context()

        self.send_addr_to_client_socket = create_client_socket(
            self.ctx, heartbeat_client_recv_addr, heartbeat_timeout=True)

        threading.Thread.__init__(self)

    def _send_addr_to_client(self, heartbeat_server_addr):
        try:
            self.send_addr_to_client_socket.send_multipart([
                remote_constants.HEARTBEAT_TAG,
                to_byte(heartbeat_server_addr),
            ])
            message = self.send_addr_to_client_socket.recv_multipart()
        except zmq.error.Again as e:
            err_str = "Can not connect to the {}, please " \
                      "check if {} is started and ensure the input " \
                      "address {} is correct.".format(self.client_type,
                              self.client_type, self.heartbeat_client_recv_addr)
            logger.warning(err_str)
            raise Exception(err_str)

    def run(self):
        socket, port = create_server_socket(self.ctx, heartbeat_timeout=True)
        heartbeat_server_addr = "{}:{}".format(get_ip_address(), port)

        self._send_addr_to_client(heartbeat_server_addr)

        while True:
            try:
                message = socket.recv_multipart()
                socket.send_multipart([remote_constants.HEARTBEAT_TAG])
            except zmq.error.Again as e:
                logger.warning("[{}] lost connnect with the {}."
                               "Please check if it is still alive.".format(
                                   self.server_type, self.client_type))
                socket.close(0)
                os._exit(0)
