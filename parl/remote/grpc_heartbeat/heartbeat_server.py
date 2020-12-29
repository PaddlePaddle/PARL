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

import grpc
import os
import time
import threading
from concurrent import futures
from parl.remote import remote_constants
from parl.remote.grpc_heartbeat import heartbeat_pb2
from parl.remote.grpc_heartbeat import heartbeat_pb2_grpc
from parl.utils import logger, get_ip_address


class GrpcHeartbeatServer(heartbeat_pb2_grpc.GrpcHeartbeatServicer):
    def __init__(self):
        self.last_heartbeat_time = time.time()
        self.stop_tag = None
        self.stop_message = None
        self.has_asked_client_to_stop = False
        self.exit_flag = False

    def Send(self, request, context):
        if self.stop_tag is not None:
            self.has_asked_client_to_stop = True
            return heartbeat_pb2.Reply(
                tag=self.stop_tag, extra_message=self.stop_message)

        self.last_heartbeat_time = time.time()
        return heartbeat_pb2.Reply(tag=remote_constants.HEARTBEAT_TAG)

    def stop(self, stop_tag, stop_message):
        """stop the heartbeat server and send the stop_message to the client.
        
        Args:
            stop_tag(byte): tag to inform why stop the heartbeat.
            stop_message(str): error message which will be sent to the client.
        """
        self.stop_tag = stop_tag
        self.stop_message = stop_message

    def exit(self):
        """exit the heartbeat server.
        """
        self.exit_flag = True

    def timeout_timer(self):
        while True:
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

            if self.exit_flag:
                break

            if self.stop_tag is not None and self.has_asked_client_to_stop == True:
                break

            if time.time(
            ) - self.last_heartbeat_time > remote_constants.HEARTBEAT_RCVTIMEO_S:
                # heartbeat exit
                break


class HeartbeatServerThread(threading.Thread):
    def __init__(self,
                 heartbeat_exit_callback_func,
                 exit_func_args=(),
                 exit_func_kwargs={}):
        """Create a thread to run the heartbeat server.

            Args:
                heartbeat_exit_callback_func(function): A callback function, which will be called after the 
                                                        heartbeat exit.
                exit_func_args(tuple): the argument tuple for calling the heartbeat_exit_callback_func. Defaults to ().
                exit_func_kwargs(dict): the argument dict for calling the heartbeat_exit_callback_func. Defaults to {}.
        """
        assert callable(
            heartbeat_exit_callback_func), "It should be a function."
        assert isinstance(exit_func_args, tuple)
        assert isinstance(exit_func_kwargs, dict)

        threading.Thread.__init__(self)
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=1),
            options=[('grpc.max_receive_message_length', -1),
                     ('grpc.max_send_message_length', -1)])
        self.heartbeat_server = GrpcHeartbeatServer()

        heartbeat_pb2_grpc.add_GrpcHeartbeatServicer_to_server(
            self.heartbeat_server, self.grpc_server)

        port = self.grpc_server.add_insecure_port('[::]:0')

        self.address = "{}:{}".format(get_ip_address(), port)

        self.heartbeat_exit_callback_func = heartbeat_exit_callback_func
        self._exit_func_args = exit_func_args
        self._exit_func_kwargs = exit_func_kwargs

    def get_address(self):
        return self.address

    def stop(self, stop_tag, stop_message):
        """stop the heartbeat server and send the stop_message to the client.
        
        Args:
            stop_tag(byte): tag to inform why stop the heartbeat.
            stop_message(str): error message which will be sent to the client.
        """
        assert stop_tag in [remote_constants.HEARTBEAT_OUT_OF_MEMORY_TAG], \
                "the stop tag `{}` is not supported".format(stop_tag)

        self.heartbeat_server.stop(stop_tag, stop_message)

    def run(self):
        # unset http_proxy and https_proxy
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if 'https_proxy' in os.environ:
            del os.environ['https_proxy']

        self.grpc_server.start()

        # a life-long while loop
        self.heartbeat_server.timeout_timer()

        # The heartbeat is exit, try to stop the grpc server.
        self.grpc_server.stop(0)

        # heartbeat is exit, call the exit function.
        self.heartbeat_exit_callback_func(*self._exit_func_args,
                                          **self._exit_func_kwargs)

    def exit(self):
        self.heartbeat_server.exit()
