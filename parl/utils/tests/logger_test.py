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
from parl.utils import logger
import threading as th
import os.path


class TestLogger(unittest.TestCase):
    def test_set_level(self):
        logger.set_level(logger.INFO)
        logger.set_dir('./test_dir')

        logger.debug('debug')
        logger.info('info')
        logger.warning('warn')
        logger.error('error')

    def test_thread_info(self):
        def thread_func():
            logger.info('test thread')

        th_list = []
        for i in range(10):
            t = th.Thread(target=thread_func)
            t.start()
            th_list.append(t)

        for t in th_list:
            t.join()

    def test_auto_set_dir(self):
        logger.auto_set_dir(action='d')
        logger.auto_set_dir(action='n')
        logger.auto_set_dir(action='k')

    def test_set_dir(self):
        logger.set_dir('./logger_dir')
        temp_file = './logger_dir/tmp.file'
        with open(temp_file, 'w') as t_file:
            t_file.write("Are you OK? From Mr.Lei")
        self.assertTrue(os.path.isfile(temp_file))
        logger.set_dir('./logger_dir')
        self.assertFalse(os.path.isfile(temp_file))


if __name__ == '__main__':
    unittest.main()
