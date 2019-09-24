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

import os
import parl
import unittest
import time
import threading

from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect


@parl.remote_class()
class Actor(object):
    def __init__(self, x=10):
        self.x = x

    def check_local_file(self):
        return os.path.exists('./rom_files/pong.bin')


class TestSendFile(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def actor(self):
        actor = Actor()
        time.sleep(2)

    def test_send_file(self):
        port = 1239
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(2)
        worker = Worker('localhost:{}'.format(port), 1)
        time.sleep(2)
        parl.connect(
            'localhost:{}'.format(port),
            distributed_files=['./rom_files/pong.bin'])
        time.sleep(3)
        actor = Actor()
        self.assertEqual(True, actor.check_local_file())
        time.sleep(3)
        del actor
        time.sleep(10)

        worker.exit()
        master.exit()


if __name__ == '__main__':
    unittest.main()
