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


class TestHeartbeat(unittest.TestCase):
    def setUp(self):
        self.server_exited = False
        self.client_exited = False

    def test_heartbeat_connection(self):
        def server_exit_func():
            print("exit heartbeat server")
            self.server_exited = True

        heartbeat_server_thread = HeartbeatServerThread(server_exit_func)
        heartbeat_server_thread.start()
        time.sleep(1)

        server_address = heartbeat_server_thread.get_address()

        def client_exit_func():
            print("exit heartbeat client")
            self.client_exited = True

        heartbeat_client_thread = HeartbeatClientThread(
            server_address, client_exit_func)
        heartbeat_client_thread.start()

        time.sleep(remote_constants.HEARTBEAT_RCVTIMEO_S * 2)

        # check server and client are still alive after HEARTBEAT_RCVTIMEO_S * 2
        assert heartbeat_server_thread.is_alive()
        assert heartbeat_client_thread.is_alive()

        heartbeat_server_thread.exit()  # manually exit the server
        heartbeat_client_thread.exit()  # manually exit the client

        # wait for threads exiting
        for _ in range(6):
            if not heartbeat_server_thread.is_alive(
            ) and not heartbeat_client_thread.is_alive():
                break
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

        # check heartbeat server and client are exited
        assert not heartbeat_server_thread.is_alive()
        assert not heartbeat_client_thread.is_alive()

        assert self.server_exited == True
        assert self.client_exited == True

    def test_heartbeat_server_will_exit_if_client_is_not_started(self):
        def server_exit_func():
            print("exit heartbeat server")
            self.server_exited = True

        heartbeat_server_thread = HeartbeatServerThread(server_exit_func)
        heartbeat_server_thread.start()

        time.sleep(remote_constants.HEARTBEAT_RCVTIMEO_S * 2)

        assert not heartbeat_server_thread.is_alive()
        assert self.server_exited == True

    def test_heartbeat_client_will_exit_if_server_is_not_started(self):
        def client_exit_func():
            print("exit heartbeat client")
            self.client_exited = True

        heartbeat_client_thread = HeartbeatServerThread(client_exit_func)
        heartbeat_client_thread.start()

        time.sleep(remote_constants.HEARTBEAT_RCVTIMEO_S * 2)

        assert not heartbeat_client_thread.is_alive()
        assert self.client_exited == True

    def test_heartbeat_server_will_exit_after_client_is_exited(self):
        def server_exit_func():
            print("exit heartbeat server")
            self.server_exited = True

        heartbeat_server_thread = HeartbeatServerThread(server_exit_func)
        heartbeat_server_thread.start()

        server_address = heartbeat_server_thread.get_address()

        def client_exit_func():
            print("exit heartbeat client")
            self.client_exited = True

        heartbeat_client_thread = HeartbeatClientThread(
            server_address, client_exit_func)
        heartbeat_client_thread.start()

        time.sleep(remote_constants.HEARTBEAT_RCVTIMEO_S * 2)

        # check server and client are still alive after HEARTBEAT_RCVTIMEO_S * 2
        assert heartbeat_server_thread.is_alive()
        assert heartbeat_client_thread.is_alive()

        heartbeat_client_thread.exit()  # manually exit the client

        # wait for heartbeat server exiting
        for _ in range(6):
            if not heartbeat_server_thread.is_alive(
            ) and not heartbeat_client_thread.is_alive():
                break
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

        # check heartbeat server and client are exited
        assert not heartbeat_server_thread.is_alive()
        assert not heartbeat_client_thread.is_alive()

        assert self.server_exited == True
        assert self.client_exited == True

    def test_heartbeat_client_will_exit_after_server_is_exited(self):
        def server_exit_func():
            print("exit heartbeat server")
            self.server_exited = True

        heartbeat_server_thread = HeartbeatServerThread(server_exit_func)
        heartbeat_server_thread.start()

        server_address = heartbeat_server_thread.get_address()

        def client_exit_func():
            print("exit heartbeat client")
            self.client_exited = True

        heartbeat_client_thread = HeartbeatClientThread(
            server_address, client_exit_func)
        heartbeat_client_thread.start()

        time.sleep(remote_constants.HEARTBEAT_RCVTIMEO_S * 2)

        # check server and client are still alive after HEARTBEAT_RCVTIMEO_S * 2
        assert heartbeat_server_thread.is_alive()
        assert heartbeat_client_thread.is_alive()

        heartbeat_server_thread.exit()  # manually exit the server

        # wait for heartbeat server exiting
        for _ in range(6):
            if not heartbeat_server_thread.is_alive(
            ) and not heartbeat_client_thread.is_alive():
                break
            time.sleep(remote_constants.HEARTBEAT_INTERVAL_S)

        # check heartbeat server and client are exited
        assert not heartbeat_server_thread.is_alive()
        assert not heartbeat_client_thread.is_alive()

        assert self.server_exited == True
        assert self.client_exited == True

if __name__ == '__main__':
    unittest.main()
