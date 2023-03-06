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
import unittest
from parl.utils import logger
import parl
from parl.remote.client import disconnect
from parl.remote.master import Master
from parl.utils import get_free_tcp_port
from parl.remote.worker import Worker
from parl.utils.test_utils import XparlTestCase
import time
import threading
import os
import signal
import subprocess

c = 10
port = get_free_tcp_port()
if __name__ == '__main__':
    master = Master(port=port)
    th = threading.Thread(target=master.run)
    th.setDaemon(True)
    th.start()
time.sleep(5)
cluster_addr = 'localhost:{}'.format(port)
parl.connect(cluster_addr)

@parl.remote_class
class Actor(object):
    def add(self, a, b):
        return a + b + c

def _create_worker(n_cpu, gpu):
    worker = Worker('localhost:{}'.format(port), n_cpu, None, gpu)
    worker.run()

def add_worker( n_cpu, gpu=""):
    command = [
        "xparl", "connect", "--address", "localhost:{}".format(port), "--cpu_num",
        str(n_cpu), "--gpu", gpu
    ]
    time.sleep(3)
    p_worker = subprocess.Popen(command, close_fds=True, preexec_fn=os.setsid)
    time.sleep(2)
    return p_worker

p_worker = add_worker(n_cpu=1)

actor = Actor()


class TestRecursive_actor(XparlTestCase):
    def test_global_running(self):
        self.add_worker(n_cpu=1)
        self.assertEqual(actor.add(1, 2), 13)
        master.exit()
        os.killpg(os.getpgid(p_worker.pid), signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
