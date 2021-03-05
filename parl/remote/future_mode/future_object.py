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
from six.moves import queue
from parl.remote.exceptions import FutureGetRepeatedlyError, FutureFunctionError, FutureObjectEmpty


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

    def get(self, block=True, timeout=None):
        """
        Remove and return the data from the `FutureObject`. 
        If optional args block is true and timeout is None (the default), block if 
        necessary until an item is available. If timeout is a positive number, it 
        blocks at most timeout seconds and raises the `parl.remote.exceptions.FutureObjectEmpty`
        exception if no itemwas available within that time. Otherwise (block is false), return an item
        if one is immediately available, else raise the `parl.remote.exceptions.FutureObjectEmpty`
        exception (timeout is ignored in that case).
        """
        if timeout is not None:
            assert timeout > 0, "`timeout` must be a non-negative number"

        with self.internal_lock:
            if self._already_get:
                raise FutureGetRepeatedlyError()

            try:
                result = self._output_queue.get(block=block, timeout=timeout)
            except queue.Empty:
                raise FutureObjectEmpty

            if isinstance(result, FutureFunctionError):
                time.sleep(
                    0.1
                )  # waiting for another thread printing the error message
                raise result
            self._already_get = True
            return result

    def get_nowait(self):
        """Equivalent to get(block=False).
        """
        return self.get(block=False)

    def empty(self):
        """
        Return True if the `FutureObject` is empty, False otherwise.
        if empty() returns False it doesn’t guarantee that a subsequent call to get() will not block.
        """
        return self._output_queue.empty()
