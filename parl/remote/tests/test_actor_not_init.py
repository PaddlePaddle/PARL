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
from parl.remote import Master, Worker, RemoteError
from parl.remote.client import disconnect


@parl.remote_class
class Actor(object):
    def __init__(self, arg1=None, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2
        will_raise_exception_func()

    def will_raise_exception_func(self):
        x = 1 / 0

class TestClient(unittest.TestCase):
  def test_not_init(self):
    """client is expected to raise an error and say that the master has not been started"""

    master = Master(port=1235)
    th = threading.Thread(target=master.run)
    th.start()
    time.sleep(1)

    worker1 = Worker('localhost:1235', 4)
    parl.connect('localhost:1235')

    def create_actor():
      actor = Actor()
    self.assertRaises(RemoteError, create_actor)

    worker1.exit()
    time.sleep(30)
    disconnect()
    time.sleep(30)
    master.exit()

if __name__ == '__main__':
    unittest.main()
