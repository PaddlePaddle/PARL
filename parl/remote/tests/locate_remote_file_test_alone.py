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

import shutil
import unittest
import parl
import os
import sys
import time
import threading
import queue
from parl.utils.test_utils import XparlTestCase
from parl.remote import exceptions
from parl.utils import logger, get_free_tcp_port


class TestCluster(XparlTestCase):
    def _write_remote_actor_to_file(self, file_path):
        with open(file_path, 'w') as f:
            f.write('''\
import parl

@parl.remote_class
class Actor(object):
    def __init__(self):
        pass
    
    def add_one(self, x):
        return x + 1
''')

    def _gen_remote_class_in_absolute_path(self, file_name):
        # E.g.: /A/B/C/test.py

        cur_dir = os.getcwd()  # /A/B/C
        parent_dir = os.path.split(cur_dir)[0]  # /A/B

        # /A/B/parl_unittest_abs_dir
        abs_dir = os.path.join(parent_dir,
                               "parl_unittest_abs_dir_{}".format(time.time()))

        if os.path.exists(abs_dir):
            logger.warning("removing directory: {}".format(abs_dir))
            shutil.rmtree(abs_dir)
        os.mkdir(abs_dir)

        file_path = os.path.join(abs_dir, file_name)

        self._write_remote_actor_to_file(file_path)

        logger.info("create file: {}".format(file_path))
        return abs_dir

    def _gen_remote_class_in_relative_path(self, file_name):
        relative_dir = "../parl_unittest_relative_dir_{}".format(time.time())

        if os.path.exists(relative_dir):
            logger.warning("removing directory: {}".format(relative_dir))
            shutil.rmtree(relative_dir)
        os.mkdir(relative_dir)

        file_path = os.path.join(relative_dir, file_name)

        self._write_remote_actor_to_file(file_path)

        logger.info("create file: {}".format(file_path))
        return relative_dir

    def test_locate_remote_file_with_absolute_env_path(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))

        abs_dir = self._gen_remote_class_in_absolute_path("abs_actor.py")

        sys.path.append(abs_dir)  # add absolute environment path
        import abs_actor
        actor = abs_actor.Actor()

        self.assertEqual(actor.add_one(1), 2)
        shutil.rmtree(abs_dir)
        sys.path.remove(abs_dir)

    def test_locate_remote_file_with_absolute_env_path_in_multi_threads(self):
        self.add_master()
        self.add_worker(n_cpu=10)
        parl.connect('localhost:{}'.format(self.port))
        abs_dir = self._gen_remote_class_in_absolute_path("abs_actor2.py")
        sys.path.append(abs_dir)  # add absolute environment path
        import abs_actor2

        def run(q):
            try:
                actor = abs_actor2.Actor()
                self.assertEqual(actor.add_one(1), 2)
            except Exception as e:
                q.put(False)
                raise e
            q.put(True)

        result = queue.Queue()
        threads = []
        for _ in range(10):
            th = threading.Thread(target=run, args=(result, ))
            th.start()
            threads.append(th)

        for th in threads:
            th.join()
            no_exception = result.get()
            assert no_exception

        shutil.rmtree(abs_dir)
        sys.path.remove(abs_dir)

    def test_locate_remote_file_with_relative_env_path_without_distributing_files(
            self):
        self.add_master()
        self.add_worker(n_cpu=1)
        relative_dir = self._gen_remote_class_in_relative_path(
            "relative_actor1.py")
        parl.connect('localhost:{}'.format(self.port))

        sys.path.append(relative_dir)  # add relative environment path
        import relative_actor1

        with self.assertRaises(exceptions.RemoteError):
            actor = relative_actor1.Actor()

        shutil.rmtree(relative_dir)
        sys.path.remove(relative_dir)

    def test_locate_remote_file_with_relative_env_path_with_distributing_files(
            self):
        self.add_master()
        self.add_worker(n_cpu=1)
        relative_dir = self._gen_remote_class_in_relative_path(
            "relative_actor2.py")
        parl.connect(
            'localhost:{}'.format(self.port),
            distributed_files=["{}/*".format(relative_dir)])

        sys.path.append(relative_dir)  # add relative environment path
        import relative_actor2
        actor = relative_actor2.Actor()
        self.assertEqual(actor.add_one(1), 2)
        shutil.rmtree(relative_dir)
        sys.path.remove(relative_dir)

if __name__ == '__main__':
    unittest.main(failfast=True)
