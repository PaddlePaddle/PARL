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

import unittest
import parl
from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.utils import logger
import subprocess
import time
import threading
import subprocess
import sys


@parl.remote_class
class Actor(object):
    def __init__(self, arg1=None, arg2=None):
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
        return UnableSerializeObject()

    def add_one(self, value):
        value += 1
        return value

    def add(self, x, y):
        time.sleep(3)
        return x + y

    def will_raise_exception_func(self):
        x = 1 / 0


class TestJob(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_acor_exit_exceptionally(self):
        port = 1337
        master = Master(port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker1 = Worker('localhost:{}'.format(port), 1)

        file_path = __file__.replace('reset_job_test', 'simulate_client')
        command = [sys.executable, file_path]
        proc = subprocess.Popen(command)
        for _ in range(6):
            if master.cpu_num == 0:
                break
            else:
                time.sleep(10)
        self.assertEqual(master.cpu_num, 0)
        proc.kill()

        parl.connect('localhost:{}'.format(port))
        actor = Actor()
        master.exit()
        worker1.exit()
        disconnect()


if __name__ == '__main__':
    unittest.main()
