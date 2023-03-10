#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import psutil
import multiprocessing as mp


class GrpcHeartbeatServer(heartbeat_pb2_grpc.GrpcHeartbeatServicer):
    def __init__(self, client_count=None, host_is_alive=True, dead_job_queue=None):
        self.last_heartbeat_time = time.time()
        self.last_heartbeat_table = dict()
        self.exit_flag = False
        self.client_count = client_count
        self.dead_job_queue = dead_job_queue
        self.host_is_alive = host_is_alive
        self.host_pid = None

    def Send(self, request, context):
        client_id = request.client_id
        self.last_heartbeat_time = time.time()
        self.last_heartbeat_table[client_id] = time.time()
        return heartbeat_pb2.Reply(tag=remote_constants.HEARTBEAT_TAG)

    def exit(self):
        """exit the heartbeat server.
        """
        self.exit_flag = True

    def timeout_timer(self):
        while True:
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

            if (self.host_pid is not None) and (not psutil.pid_exists(self.host_pid)):
                self.exit()

            if self.exit_flag:
                break

            if time.time() - self.last_heartbeat_time > remote_constants.HEARTBEAT_RCVTIMEO_S:
                # heartbeat exit
                break

    def _parent_process_is_running(self):
        if not self.host_is_alive.value:
            return False
        ppid = os.getppid()
        return ppid != 1

    def timeout_time_mp(self):

        while self._parent_process_is_running():
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

            cur_time = time.time()
            to_del_client = []
            for client_id, last_heartbeat_time in self.last_heartbeat_table.items():
                if cur_time - last_heartbeat_time > remote_constants.HEARTBEAT_RCVTIMEO_S:
                    to_del_client.append(client_id)
            for client_id in to_del_client:
                del self.last_heartbeat_table[client_id]
                self.dead_job_queue.put(client_id)
            self.client_count.value = len(self.last_heartbeat_table)

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

    def set_host_pid(self, host_pid):
        self.heartbeat_server.host_pid = host_pid

    def exit(self):
        self.heartbeat_server.exit()

class HeartbeatServerProcess(mp.Process):
    def __init__(self, port, client_count, host_is_alive, dead_job_queue):
        """Create a process to run the heartbeat server.
            Args:
                port(mp.Value): notify the main prcoess of the severt port.
        """

        mp.Process.__init__(self)
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=500),
            options=[('grpc.max_receive_message_length', -1),
                     ('grpc.max_send_message_length', -1)])
        self.heartbeat_server = GrpcHeartbeatServer(client_count, host_is_alive, dead_job_queue)

        heartbeat_pb2_grpc.add_GrpcHeartbeatServicer_to_server(
            self.heartbeat_server, self.grpc_server)

        with port.get_lock():
            port.value = self.grpc_server.add_insecure_port('[::]:0')

    def run(self):
        # unset http_proxy and https_proxy
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if 'https_proxy' in os.environ:
            del os.environ['https_proxy']

        self.grpc_server.start()

        # a life-long while loop
        self.heartbeat_server.timeout_time_mp()

        # The heartbeat is exit, try to stop the grpc server.
        self.grpc_server.stop(0)
