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

import time
import unittest
from parl.remote.grpc_heartbeat import HeartbeatServerThread
from parl.remote.grpc_heartbeat import HeartbeatClientThread
from parl.remote import remote_constants


class TestHeartbeatClientArguments(unittest.TestCase):
    def setUp(self):
        self.server_exited = False
        self.client_exited = False

    def test_heartbeat_client_exit_with_args(self):
        def server_exit_func():
            print("exit heartbeat server")
            self.server_exited = True

        heartbeat_server_thread = HeartbeatServerThread(server_exit_func)
        heartbeat_server_thread.start()

        server_address = heartbeat_server_thread.get_address()

        arg1_value = 10

        def client_exit_func(arg1):
            print("exit heartbeat client")
            assert arg1 == arg1_value
            self.client_exited = True

        heartbeat_client_thread = HeartbeatClientThread(
            server_address, client_exit_func, exit_func_args=(arg1_value, ))
        heartbeat_client_thread.start()

        time.sleep(remote_constants.HEARTBEAT_RCVTIMEO_S * 2)

        # check server and client are still alive after HEARTBEAT_RCVTIMEO_S * 2
        assert heartbeat_server_thread.is_alive()
        assert heartbeat_client_thread.is_alive()

        heartbeat_client_thread.exit()  # manually exit the client

        # wait for threads exiting
        for _ in range(10):
            if not heartbeat_server_thread.is_alive(
            ) and not heartbeat_client_thread.is_alive():
                break
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

        # check heartbeat server and client are exited
        assert not heartbeat_server_thread.is_alive()
        assert not heartbeat_client_thread.is_alive()

        assert self.server_exited == True
        assert self.client_exited == True

    def test_heartbeat_client_exit_with_wrong_args(self):
        def server_exit_func():
            print("exit heartbeat server")
            self.server_exited = True

        heartbeat_server_thread = HeartbeatServerThread(server_exit_func)
        heartbeat_server_thread.start()

        server_address = heartbeat_server_thread.get_address()

        arg1_value = 10

        def client_exit_func(arg1):
            print("exit heartbeat client")
            assert arg1 == arg1_value
            self.client_exited = True

        heartbeat_client_thread = HeartbeatClientThread(
            server_address,
            client_exit_func,
            exit_func_args=(arg1_value, "wrong_args"))
        heartbeat_client_thread.start()

        time.sleep(remote_constants.HEARTBEAT_RCVTIMEO_S * 2)

        # check server and client are still alive after HEARTBEAT_RCVTIMEO_S * 2
        assert heartbeat_server_thread.is_alive()
        assert heartbeat_client_thread.is_alive()

        heartbeat_client_thread.exit()  # manually exit the client
        # will raise an exception in the backend thread

        # wait for threads exiting
        for _ in range(10):
            if not heartbeat_server_thread.is_alive(
            ) and not heartbeat_client_thread.is_alive():
                break
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

        # check heartbeat server and client are exited
        assert not heartbeat_server_thread.is_alive()
        assert not heartbeat_client_thread.is_alive()

        assert self.server_exited == True
        assert self.client_exited == False  # the heartbeat client cannot exit normally

    def test_heartbeat_client_exit_with_kwargs(self):
        def server_exit_func():
            print("exit heartbeat server")
            self.server_exited = True

        heartbeat_server_thread = HeartbeatServerThread(server_exit_func)
        heartbeat_server_thread.start()

        server_address = heartbeat_server_thread.get_address()

        arg1_value = 10

        def client_exit_func(arg1):
            print("exit heartbeat client")
            assert arg1 == arg1_value
            self.client_exited = True

        heartbeat_client_thread = HeartbeatClientThread(
            server_address,
            client_exit_func,
            exit_func_kwargs={"arg1": arg1_value})
        heartbeat_client_thread.start()

        time.sleep(remote_constants.HEARTBEAT_RCVTIMEO_S * 2)

        # check server and client are still alive after HEARTBEAT_RCVTIMEO_S * 2
        assert heartbeat_server_thread.is_alive()
        assert heartbeat_client_thread.is_alive()

        heartbeat_client_thread.exit()  # manually exit the client

        # wait for threads exiting
        for _ in range(10):
            if not heartbeat_server_thread.is_alive(
            ) and not heartbeat_client_thread.is_alive():
                break
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

        # check heartbeat server and client are exited
        assert not heartbeat_server_thread.is_alive()
        assert not heartbeat_client_thread.is_alive()

        assert self.server_exited == True
        assert self.client_exited == True

    def test_heartbeat_client_exit_with_wrong_kwargs(self):
        def server_exit_func():
            print("exit heartbeat server")
            self.server_exited = True

        heartbeat_server_thread = HeartbeatServerThread(server_exit_func)
        heartbeat_server_thread.start()

        server_address = heartbeat_server_thread.get_address()

        arg1_value = 10

        def client_exit_func(arg1):
            print("exit heartbeat client")
            assert arg1 == arg1_value
            self.client_exited = True

        heartbeat_client_thread = HeartbeatClientThread(
            server_address,
            client_exit_func,
            exit_func_kwargs={"wrong_args": arg1_value})
        heartbeat_client_thread.start()

        time.sleep(remote_constants.HEARTBEAT_RCVTIMEO_S * 2)

        # check server and client are still alive after HEARTBEAT_RCVTIMEO_S * 2
        assert heartbeat_server_thread.is_alive()
        assert heartbeat_client_thread.is_alive()

        heartbeat_client_thread.exit()  # manually exit the client
        # will raise an exception in the backend thread

        # wait for threads exiting
        for _ in range(10):
            if not heartbeat_server_thread.is_alive(
            ) and not heartbeat_client_thread.is_alive():
                break
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

        # check heartbeat server and client are exited
        assert not heartbeat_server_thread.is_alive()
        assert not heartbeat_client_thread.is_alive()

        assert self.server_exited == True
        assert self.client_exited == False  # the heartbeat client cannot exit normally


if __name__ == '__main__':
    unittest.main()
