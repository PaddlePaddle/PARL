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
from parl.utils import logger, _IS_WINDOWS
import os
import threading
import time
import subprocess


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


class TestJobAlone(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_job_exit_exceptionally(self):
        master = Master(port=1334)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker1 = Worker('localhost:1334', 4)
        time.sleep(10)
        self.assertEqual(worker1.job_buffer.full(), True)
        time.sleep(1)
        self.assertEqual(master.cpu_num, 4)
        print("We are going to kill all the jobs.")
        if _IS_WINDOWS:
            command = r'''for /F "skip=2 tokens=2 delims=," %a in ('wmic process where "commandline like '%remote\\job.py%'" get processid^,status /format:csv') do taskkill /F /T /pid %a'''
            print(os.popen(command).read())
        else:
            command = (
                "ps aux | grep remote/job.py | awk '{print $2}' | xargs kill -9"
            )
            subprocess.call([command], shell=True)
        parl.connect('localhost:1334')
        actor = Actor()
        self.assertEqual(actor.add_one(1), 2)
        time.sleep(20)

        master.exit()
        worker1.exit()


if __name__ == '__main__':
    unittest.main()
