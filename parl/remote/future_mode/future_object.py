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
from parl.remote.exceptions import FutureGetRepeatedlyError, FutureFunctionError


class FutureObject(object):
    def __init__(self, output_queue):
        """This class is used to encapsulate the output_queue(`queue.Queue`),
        and provides a `get` function. When calling a function of a class
        decorated by `pare.remote_class(wait=False)`, user will get a `FutureObject`
        immediately and can get the real return by calling the `get` function of 
        the `future` object.

        For example:
            ```python
            import parl

            @parl.remote_class(wait=False)
            class Actor(object):
                def __init__(self):
                    pass

                def func(self):
                    # do something
                    return 0

            parl.connect("localhost:8010")

            actors = Actor()
            future_object = actor.func() # return a `FutureObject`
            result = future_object.get()
            ```

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

            if isinstance(result, FutureFunctionError):
                time.sleep(
                    0.1
                )  # waiting for another thread printing the error message
                raise result
            self._already_get = True
            return result
