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
import time
import threading
from parl.remote import exceptions
from parl.utils import logger
from parl.utils.test_utils import XparlTestCase


@parl.remote_class
class Actor(object):
    def file_exists(self, filename):
        return os.path.exists(filename)


class TestCluster(XparlTestCase):
    def test_distributed_files_with_RegExp(self):
        self.add_master()
        self.add_worker(n_cpu=1)
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
        parl.connect(
            'localhost:{}'.format(self.port),
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

    def test_miss_match_case(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        if os.path.exists('distribute_test_dir_2'):
            shutil.rmtree('distribute_test_dir_2')
        os.mkdir('distribute_test_dir_2')
        f = open('distribute_test_dir_2/test1.txt', 'wb')
        f.close()
        f = open('distribute_test_dir_2/data1.npy', 'wb')
        f.close()
        logger.info("running:test_distributed_files_with_RegExp_error_case")

        def connect_test():
            parl.connect(
                'localhost:{}'.format(self.port),
                distributed_files=['distribute_test_dir_2/miss_match*'])

        self.assertRaises(ValueError, connect_test)
        shutil.rmtree('distribute_test_dir_2')

    def test_distribute_folder(self):
        self.add_master()
        self.add_worker(n_cpu=1)
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
        parl.connect(
            'localhost:{}'.format(self.port),
            distributed_files=[
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


if __name__ == '__main__':
    unittest.main(failfast=True)
