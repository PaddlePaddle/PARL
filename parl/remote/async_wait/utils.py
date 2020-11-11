#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import time
import threading
from collections import namedtuple
from parl.remote.exceptions import FutureGetRepeatedlyError, AsyncFunctionError


class FutureObject(object):
    def __init__(self, output_queue):
        """Define a new class to avoid user calling the `put` function of output_queue

        Args:
            output_queue(queue.Queue): queue to get the result of function calling
                                           or set_attr/get_attr
        """
        self._output_queue = output_queue
        self._already_get = False
        self.internal_lock = threading.Lock()

    def get(self):
        """
        """
        with self.internal_lock:
            if self._already_get:
                raise FutureGetRepeatedlyError()

            result = self._output_queue.get()

            if isinstance(result, AsyncFunctionError):
                time.sleep(
                    0.1
                )  # waiting for another thread printing the error message
                raise result
            self._already_get = True
            return result


CallingRequest = namedtuple(
    'CallingRequest',
    ['calling_type', 'attr', 'value', 'args', 'kwargs', 'future_return_queue'])
