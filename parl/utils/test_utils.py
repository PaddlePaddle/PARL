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
        self.sub_process = []
        self.sub_process_type = []
        self.worker_process = []

    def tearDown(self):
        print('start tearDown')
        assert len(self.sub_process) == len(self.sub_process_type)
        for i, proc in enumerate(self.sub_process):
            if self.sub_process_type[i] == 'master':
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
            elif self.sub_process_type[i] == 'worker':
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                raise NotImplementedError
        disconnect()

    def _create_master(self, device):
        master = Master(port=self.port, device=device)
        master.run()

    def _create_worker(self, n_cpu, gpu):
        worker = Worker('localhost:{}'.format(self.port), n_cpu, None, gpu)
        worker.run()

    def add_master(self, device=remote_constants.CPU):
        p_master = self.ctx.Process(target=self._create_master, args=(device, ))
        p_master.start()
        self.sub_process.append(p_master)
        self.sub_process_type.append('master')
        while is_port_available(self.port):
            logger.info("Master[localhost:{}] starting".format(self.port))
            time.sleep(1)
        logger.info("Master[localhost:{}] started".format(self.port))
        time.sleep(10)

    def add_worker(self, n_cpu, gpu=""):
        command = [
            "xparl", "connect", "--address", "localhost:{}".format(self.port), "--cpu_num",
            str(n_cpu), "--gpu", gpu
        ]
        time.sleep(3)
        p_worker = subprocess.Popen(command, close_fds=True, preexec_fn=os.setsid)
        time.sleep(2)
        self.sub_process.append(p_worker)
        self.sub_process_type.append('worker')
        self.worker_process.append(p_worker)

    def remove_all_workers(self):
        for proc in self.worker_process:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
