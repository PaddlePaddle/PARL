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
from parl.remote.monitor import ClusterMonitor
import time
import threading
from parl.remote import exceptions
from parl.utils.test_utils import XparlTestCase
import os
import signal

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

    def add_one(self, value):
        value += 1
        return value

    def add(self, x, y):
        time.sleep(3)
        return x + y

    def will_raise_exception_func(self):
        x = 1 / 0


class TestClusterMonitor(XparlTestCase):

    def remove_ten_workers(self):
        for i, proc in enumerate(self.worker_process):
            if i == 10: break
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

    def test_twenty_worker(self):
        self.add_master()
        cluster_monitor = ClusterMonitor('localhost:{}'.format(self.port))

        for _ in range(20):
            self.add_worker(n_cpu=1)


        check_flag = False
        for _ in range(10):
            if 20 == len(cluster_monitor.data['workers']):
                check_flag = True
                break
            time.sleep(10)
        self.assertTrue(check_flag)

        self.remove_ten_workers()

        check_flag = False
        for _ in range(10):
            if 10 == len(cluster_monitor.data['workers']):
                check_flag = True
                break
            time.sleep(10)
        self.assertTrue(check_flag)

        self.remove_all_workers()
        # check if the number of workers drops to 0
        check_flag = False
        for _ in range(10):
            if 0 == len(cluster_monitor.data['workers']):
                check_flag = True
                break
            time.sleep(10)
        self.assertTrue(check_flag)

if __name__ == '__main__':
    unittest.main(failfast=True)
