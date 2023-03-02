#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from parl.utils import get_free_tcp_port
from parl.utils import is_port_available
from parl.utils import logger
import multiprocessing as mp
import threading
from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.remote import remote_constants
import time
import os
import signal
import psutil
import subprocess


class XparlTestCase(unittest.TestCase):
    def setUp(self):
        self.port = get_free_tcp_port()
        self.ctx = mp.get_context()
        self.worker_events= []
        self.master_exit_event = mp.Event()

    def tearDown(self):
        print('start tearDown')
        self.master_exit_event.set()
        self.remove_all_workers()
        disconnect()

    def _create_master(self, device):
        master = Master(port=self.port, device=device)
        th = threading.Thread(target=master.run)
        th.setDaemon(True)
        th.start()
        self.master_exit_event.wait()
        master.exit()

    def _create_worker(self, n_cpu, gpu, exit_event):
        worker = Worker('localhost:{}'.format(self.port), n_cpu, None, gpu)
        exit_event.wait()
        worker.exit()

    def add_master(self, device=remote_constants.CPU):
        p_master = self.ctx.Process(target=self._create_master, args=(device, ))
        p_master.start()
        while is_port_available(self.port):
            logger.info("Master[localhost:{}] starting".format(self.port))
            time.sleep(1)
        logger.info("Master[localhost:{}] started".format(self.port))
        time.sleep(10)

    def add_worker(self, n_cpu, gpu=""):
        exit_event = mp.Event()
        p_worker = self.ctx.Process(target=self._create_worker, args=(n_cpu, n_gpu, exit_event))
        p_worker.start()
        self.worker_events.append(exit_event)

    def remove_all_workers(self):
        for event in self.worker_events:
            event.set()
