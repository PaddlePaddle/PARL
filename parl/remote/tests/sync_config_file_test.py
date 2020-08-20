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
import os
import time
import threading
import sys
import numpy as np
import json


@parl.remote_class
class Actor(object):
    def __init__(self, random_array, config_file):
        self.random_array = random_array
        self.config_file = config_file

    def random_sum(self):
        return np.load(self.random_array).sum()

    def read_config(self):
        with open(self.config_file, 'r') as f:
            config_file = json.load(f)
        return config_file['test']


class TestConfigfile(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_sync_config_file(self):
        master = Master(port=1335)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker = Worker('localhost:1335', 1)

        random_file = 'random.npy'
        random_array = np.random.randn(3, 5)
        np.save(random_file, random_array)
        random_sum = random_array.sum()

        with open('config.json', 'w') as f:
            config_file = {'test': 1000}
            json.dump(config_file, f)

        parl.connect('localhost:1335', ['random.npy', 'config.json'])
        actor = Actor('random.npy', 'config.json')
        time.sleep(5)
        os.remove('./random.npy')
        os.remove('./config.json')
        remote_sum = actor.random_sum()
        self.assertEqual(remote_sum, random_sum)
        time.sleep(10)

        remote_config = actor.read_config()
        self.assertEqual(config_file['test'], remote_config)

        del actor
        worker.exit()
        master.exit()


if __name__ == '__main__':
    unittest.main()
