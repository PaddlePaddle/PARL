#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import parl
import threading
import unittest
from parl.remote import *


@parl.remote
class Simulator:
    def __init__(self, arg1, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2

    def get_arg1(self):
        return self.arg1

    def get_arg2(self):
        return self.arg2

    def set_arg1(self, value):
        self.arg1 = value

    def set_arg2(self, value):
        self.arg2 = value

    def get_unable_serialize_object(self):
        return self


class TestRemote(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator(1, arg2=2)

        # run client in a new thread to fake a remote client
        self.client_thread = threading.Thread(
            target=self.sim.as_remote, args=(
                'localhost',
                '8008',
            ))
        self.client_thread.setDaemon(True)
        self.client_thread.start()

        self.remote_manager = RemoteManager(port='8008')

    def tearDown(self):
        self.remote_manager.close()

    def test_remote_object(self):
        print(inspect.stack()[0][3])
        remote_sim = self.remote_manager.get_remote()
        logger.info('got remote sim')

        logger.info('remote_sim.get_arg1()')
        self.assertEqual(remote_sim.get_arg1(), 1)
        self.assertEqual(remote_sim.get_arg2(), 2)

        logger.info('remote_sim.set_arg1()')
        ret = remote_sim.set_arg1(3)
        self.assertIsNone(ret)
        ret = remote_sim.set_arg2(4)
        self.assertIsNone(ret)

        self.assertEqual(remote_sim.get_arg1(), 3)
        self.assertEqual(remote_sim.get_arg2(), 4)

    def test_remote_object_with_wrong_getattr_get_variable(self):
        print(inspect.stack()[0][3])
        remote_sim = self.remote_manager.get_remote()

        try:
            remote_sim.get_arg3()
        except RemoteAttributeError:
            # expected
            return

        assert False

    def test_remote_object_with_wrong_getattr_set_variable(self):
        print(inspect.stack()[0][3])
        remote_sim = self.remote_manager.get_remote()

        try:
            remote_sim.set_arg3(3)
        except RemoteAttributeError:
            # expected
            return

        assert False

    def test_remote_object_with_wrong_argument(self):
        print(inspect.stack()[0][3])
        remote_sim = self.remote_manager.get_remote()

        try:
            remote_sim.set_arg1(wrong_arg=1)
        except RemoteError:
            # expected
            return

        assert False

    def test_remote_object_with_unable_serialize_argument(self):
        print(inspect.stack()[0][3])
        remote_sim = self.remote_manager.get_remote()

        try:
            remote_sim.set_arg1(wrong_arg=remote_sim)
        except SerializeError:
            # expected
            return

        assert False

    def test_remote_object_with_unable_serialize_return(self):
        print(inspect.stack()[0][3])
        remote_sim = self.remote_manager.get_remote()

        try:
            remote_sim.get_unable_serialize_object()
        except RemoteSerializeError:
            # expected
            return

        assert False

    def test_mutli_remote_object(self):
        print(inspect.stack()[0][3])
        time.sleep(1)
        # run second client
        sim2 = Simulator(11, arg2=22)
        client_thread2 = threading.Thread(
            target=sim2.as_remote, args=(
                'localhost',
                '8008',
            ))
        client_thread2.setDaemon(True)
        client_thread2.start()

        time.sleep(1)
        remote_sim1 = self.remote_manager.get_remote()
        remote_sim2 = self.remote_manager.get_remote()

        self.assertEqual(remote_sim1.get_arg1(), 1)
        self.assertEqual(remote_sim2.get_arg1(), 11)

    def test_mutli_remote_object_with_one_failed(self):
        print(inspect.stack()[0][3])
        time.sleep(1)
        # run second client
        sim2 = Simulator(11, arg2=22)
        client_thread2 = threading.Thread(
            target=sim2.as_remote, args=(
                'localhost',
                '8008',
            ))
        client_thread2.setDaemon(True)
        client_thread2.start()

        time.sleep(1)
        remote_sim1 = self.remote_manager.get_remote()
        remote_sim2 = self.remote_manager.get_remote()

        try:
            # make remote sim1 failed
            remote_sim1.get_arg3()
        except:
            pass

        self.assertEqual(remote_sim2.get_arg1(), 11)

    def test_heartbeat_after_server_closed(self):
        print(inspect.stack()[0][3])
        remote_sim = self.remote_manager.get_remote()
        logger.info('got remote sim')

        time.sleep(1)
        logger.info('closing server')
        self.remote_manager.close()

        # heartbeat interval (10s) + max waiting reply (10s)
        time.sleep(20)

        logger.info('check self.sim.remote_closed')
        self.assertTrue(self.sim.remote_closed())

    def test_set_client_ip_port_manually(self):
        print(inspect.stack()[0][3])
        time.sleep(1)
        # run second client
        sim2 = Simulator(11, arg2=22)
        client_thread2 = threading.Thread(
            target=sim2.as_remote,
            args=(
                'localhost',
                8008,
                'localhost',
                6666,
            ))
        client_thread2.setDaemon(True)
        client_thread2.start()

        time.sleep(1)
        remote_sim1 = self.remote_manager.get_remote()
        remote_sim2 = self.remote_manager.get_remote()

        self.assertEqual(remote_sim1.get_arg1(), 1)
        self.assertEqual(remote_sim2.get_arg1(), 11)


if __name__ == '__main__':
    unittest.main()
