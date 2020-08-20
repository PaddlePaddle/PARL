#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import time
import threading
from parl.remote.client import disconnect
from parl.remote import exceptions
import subprocess
from parl.utils import logger
import paddle.fluid as fluid
import os


@parl.remote_class
class Actor(object):
    def __init__(self, cuda=False):
        if cuda:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())


class TestCluster(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_gpu(self):
        master = Master(port=8241)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)

        worker1 = Worker('localhost:8241', 4)

        parl.connect('localhost:8241')

        if parl.utils.is_gpu_available():
            actor = Actor(cuda=True)
        else:
            actor = Actor(cuda=False)

        del actor
        master.exit()
        worker1.exit()


if __name__ == '__main__':
    unittest.main()
