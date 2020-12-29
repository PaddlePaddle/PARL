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
import grpc
import time
import threading
from parl.remote import remote_constants
from parl.remote.grpc_heartbeat import heartbeat_pb2
from parl.remote.grpc_heartbeat import heartbeat_pb2_grpc
from parl.utils import logger


class HeartbeatClientThread(threading.Thread):
    def __init__(self,
                 heartbeat_server_addr,
                 heartbeat_exit_callback_func,
                 exit_func_args=(),
                 exit_func_kwargs={}):
        """Create a thread to run the heartbeat client.

            Args:
                heartbeat_server_addr(str): the address of the heartbeat server.
                heartbeat_exit_callback_func(function): A callback function, which will be called after the 
                                                        heartbeat exit.
                exit_func_args(tuple): the argument tuple for calling the heartbeat_exit_callback_func. Defaults to ().
                exit_func_kwargs(dict): the argument dict for calling the heartbeat_exit_callback_func. Defaults to {}.
        """
        assert isinstance(heartbeat_server_addr, str)
        assert callable(
            heartbeat_exit_callback_func), "It should be a function."
        assert isinstance(exit_func_args, tuple)
        assert isinstance(exit_func_kwargs, dict)

        threading.Thread.__init__(self)
        self.heartbeat_server_addr = heartbeat_server_addr

        self.heartbeat_exit_callback_func = heartbeat_exit_callback_func
        self._exit_func_args = exit_func_args
        self._exit_func_kwargs = exit_func_kwargs

        self.exit_flag = False

    def exit(self):
        self.exit_flag = True

    def run(self):
        # unset http_proxy and https_proxy
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if 'https_proxy' in os.environ:
            del os.environ['https_proxy']

        with grpc.insecure_channel(
                self.heartbeat_server_addr,
                options=[('grpc.max_receive_message_length', -1),
                         ('grpc.max_send_message_length', -1)]) as channel:
            stub = heartbeat_pb2_grpc.GrpcHeartbeatStub(channel)

            while True:
                if self.exit_flag:
                    break

                try:
                    response = stub.Send(
                        heartbeat_pb2.Request(
                            tag=remote_constants.HEARTBEAT_TAG),
                        timeout=remote_constants.HEARTBEAT_RCVTIMEO_S)

                    if response.tag == remote_constants.HEARTBEAT_TAG:
                        pass
                    elif response.tag == remote_constants.HEARTBEAT_OUT_OF_MEMORY_TAG:
                        logger.error(response.extra_message)
                        break
                    else:
                        raise NotImplementedError

                except grpc._channel._InactiveRpcError as e:
                    break

                time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

        # heartbeat is exit, call the exit function.
        self.heartbeat_exit_callback_func(*self._exit_func_args,
                                          **self._exit_func_kwargs)
