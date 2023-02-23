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
import time
import threading

from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.utils import get_free_tcp_port
from parl.utils.test_utils import XparlTestCase


@parl.remote_class(n_gpu=2)
class Actor(object):
    def __init__(self):
        pass


class TestCluster(XparlTestCase):
    def test_class_decorated_by_remote_class(self):
        self.add_master(device="gpu")
        self.add_worker(n_cpu=0, gpu="0,1")
        port = self.port
        parl.connect('localhost:{}'.format(port))

        actor = Actor()

    def test_function_decorated_by_remote_class(self):
        self.add_master(device="gpu")
        self.add_worker(n_cpu=0, gpu="0,1")
        port = self.port
        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(AssertionError):
            @parl.remote_class(n_gpu=2)
            def func():
                pass

        actor = Actor()


    def test_passing_arguments_with_unsupported_argument_names(self):
        with self.assertRaises(AssertionError):
            @parl.remote_class(xxx=10)
            class Actor2(object):
                def __init__(self):
                    pass


if __name__ == '__main__':
    unittest.main()
