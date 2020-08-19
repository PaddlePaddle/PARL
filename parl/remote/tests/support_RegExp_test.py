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
import os
import shutil
import parl
from parl.remote.master import Master
from parl.remote.worker import Worker
import time
import threading
from parl.remote.client import disconnect
from parl.remote import exceptions
from parl.utils import logger


@parl.remote_class
class Actor(object):
    def file_exists(self, filename):
        return os.path.exists(filename)


class TestCluster(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_distributed_files_with_RegExp(self):
        if os.path.exists('distribute_test_dir'):
            shutil.rmtree('distribute_test_dir')
        os.mkdir('distribute_test_dir')
        f = open('distribute_test_dir/test1.txt', 'wb')
        f.close()
        f = open('distribute_test_dir/test2.txt', 'wb')
        f.close()
        f = open('distribute_test_dir/data1.npy', 'wb')
        f.close()
        f = open('distribute_test_dir/data2.npy', 'wb')
        f.close()
        logger.info("running:test_distributed_files_with_RegExp")
        master = Master(port=8435)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:8435', 1)
        parl.connect(
            'localhost:8435',
            distributed_files=[
                'distribute_test_dir/test*',
                'distribute_test_dir/*npy',
            ])
        actor = Actor()
        self.assertTrue(actor.file_exists('distribute_test_dir/test1.txt'))
        self.assertTrue(actor.file_exists('distribute_test_dir/test2.txt'))
        self.assertTrue(actor.file_exists('distribute_test_dir/data1.npy'))
        self.assertTrue(actor.file_exists('distribute_test_dir/data2.npy'))
        self.assertFalse(actor.file_exists('distribute_test_dir/data3.npy'))
        shutil.rmtree('distribute_test_dir')
        master.exit()
        worker1.exit()

    def test_miss_match_case(self):
        if os.path.exists('distribute_test_dir_2'):
            shutil.rmtree('distribute_test_dir_2')
        os.mkdir('distribute_test_dir_2')
        f = open('distribute_test_dir_2/test1.txt', 'wb')
        f.close()
        f = open('distribute_test_dir_2/data1.npy', 'wb')
        f.close()
        logger.info("running:test_distributed_files_with_RegExp_error_case")
        master = Master(port=8436)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:8436', 1)

        def connect_test():
            parl.connect(
                'localhost:8436',
                distributed_files=['distribute_test_dir_2/miss_match*'])

        self.assertRaises(ValueError, connect_test)
        shutil.rmtree('distribute_test_dir_2')
        master.exit()
        worker1.exit()

    def test_distribute_folder(self):
        if os.path.exists('distribute_test_dir_3'):
            shutil.rmtree('distribute_test_dir_3')
        os.mkdir('distribute_test_dir_3')
        os.mkdir('distribute_test_dir_3/subfolder_test')
        os.mkdir('distribute_test_dir_3/empty_folder')
        f = open('distribute_test_dir_3/subfolder_test/test1.txt', 'wb')
        f.close()
        f = open('distribute_test_dir_3/subfolder_test/data1.npy', 'wb')
        f.close()
        logger.info("running:test_distributed_folder")
        master = Master(port=8437)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:8437', 1)
        parl.connect(
            'localhost:8437', distributed_files=[
                'distribute_test_dir_3',
            ])
        actor = Actor()
        self.assertTrue(
            actor.file_exists(
                'distribute_test_dir_3/subfolder_test/test1.txt'))
        self.assertTrue(
            actor.file_exists(
                'distribute_test_dir_3/subfolder_test/data1.npy'))
        self.assertTrue(
            actor.file_exists('distribute_test_dir_3/empty_folder'))
        shutil.rmtree('distribute_test_dir_3')
        master.exit()
        worker1.exit()


if __name__ == '__main__':
    unittest.main()
