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

import threading
import unittest
from parl.core.fluid.model_helper import global_model_helper
from six.moves.queue import Queue


class GlobalModelHelperTest(unittest.TestCase):
    def test_generate_model_id(self):
        id1 = global_model_helper.generate_model_id()
        id2 = global_model_helper.generate_model_id()
        self.assertNotEqual(id1, id2)

    def _gen_model_id(self, q):
        model_id = global_model_helper.generate_model_id()
        q.put(model_id)

    def test_generate_model_id_with_multi_thread(self):
        q = Queue()
        t1 = threading.Thread(target=self._gen_model_id, args=(q, ))
        t2 = threading.Thread(target=self._gen_model_id, args=(q, ))
        t1.start()
        t2.start()

        t1.join()
        t2.join()

        id1 = q.get()
        id2 = q.get()

        self.assertNotEqual(id1, id2)

    def test_register_model_id(self):
        global_model_helper.register_model_id('my_model_0')
        global_model_helper.register_model_id('my_model_1')

        with self.assertRaises(AssertionError):
            global_model_helper.register_model_id('my_model_0')

    def _register_model_id(self, q):
        try:
            global_model_helper.register_model_id('my_model_2')
        except AssertionError:
            q.put(False)
        else:
            q.put(True)

    def test_register_model_id_with_multi_thread(self):
        q = Queue()
        t1 = threading.Thread(target=self._register_model_id, args=(q, ))
        t2 = threading.Thread(target=self._register_model_id, args=(q, ))

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        return1 = q.get()
        return2 = q.get()
        assert (return1 is True and return2 is False) or \
                (return1 is False and return2 is True)

    def test_registet_model_id_with_used_model_id(self):
        model_id = global_model_helper.generate_model_id()
        with self.assertRaises(AssertionError):
            global_model_helper.register_model_id(model_id)


if __name__ == '__main__':
    unittest.main()
