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

from six.moves import queue
import time
import threading
import weakref

from collections import namedtuple
from parl.remote.exceptions import FutureFunctionError
from parl.remote.future_mode.future_object import FutureObject

XPARL_RESERVED_PREFIX = "_xparl"
RESERVED_NAME_ERROR_STR = "Name starting with `_xparl` is the reserved name in xparl, please don't use the name `{}`."

CallingRequest = namedtuple(
    'CallingRequest',
    ['calling_type', 'attr', 'value', 'args', 'kwargs', 'future_return_queue'])


def proxy_wrapper_nowait_func(remote_wrapper):
    """This function implements the remote decorator for asynchronous mode.
    With this decorator, the function called by user return a `future` object immediately
    and it will not be blocked. Users can get the real return of the function by calling 
    the `get` function of the `future` object.

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

        actors = [Actor() for _ in range(10)]
        future_objects = [actor.func() for actor in actors]
        results = [future_obj.get() for future_obj in future_objects]
        ```
    """
    original_class = remote_wrapper._original
    max_memory = remote_wrapper._max_memory
    n_gpu = remote_wrapper._n_gpu

    class ProxyWrapperNoWait(object):
        def __init__(self, *args, **kwargs):
            for key in kwargs:
                assert not key.startswith(
                    XPARL_RESERVED_PREFIX), RESERVED_NAME_ERROR_STR.format(key)

            kwargs['_xparl_proxy_wrapper_nowait'] = self
            kwargs['_xparl_remote_class'] = original_class
            kwargs['_xparl_remote_class_max_memory'] = max_memory
            kwargs['_xparl_remote_class_n_gpu'] = n_gpu

            self._xparl_remote_wrapper_calling_queue = queue.Queue()
            self._xparl_remote_wrapper_internal_lock = threading.Lock()
            self._xparl_calling_finished_event = threading.Event()
            self._xparl_remote_object_exception = None

            object_thread = threading.Thread(
                target=self._run_object_in_backend, args=(args, kwargs))
            object_thread.setDaemon(True)
            object_thread.start()

        def destroy(self):
            calling_request = CallingRequest(
                calling_type="destroy",
                attr=None,
                value=None,
                args=None,
                kwargs=None,
                future_return_queue=None)
            self._xparl_remote_wrapper_calling_queue.put(calling_request)

        def _run_object_in_backend(self, args, kwargs):
            try:
                self._xparl_remote_wrapper_obj = remote_wrapper(
                    *args, **kwargs)
                for key in self._xparl_remote_wrapper_obj.get_attrs():
                    assert not key.startswith(
                        XPARL_RESERVED_PREFIX), RESERVED_NAME_ERROR_STR.format(
                            key)
            except Exception as e:
                async_error = FutureFunctionError('__init__')
                self._xparl_remote_object_exception = async_error
                self._xparl_calling_finished_event.set()
                raise e
            """
            NOTE:
                We should set the event after the initialization of self._xparl_remote_wrapper_obj.
                Only after the initialization is complete can we call the function of actor.
            """
            self._xparl_calling_finished_event.set()

            while True:
                calling_request = self._xparl_remote_wrapper_calling_queue.get(
                )
                assert calling_request.calling_type in ["setattr", "getattr", "destroy"]

                try:
                    if calling_request.calling_type == "setattr":
                        self._xparl_remote_wrapper_obj.set_remote_attr(
                            calling_request.attr, calling_request.value)
                    elif calling_request.calling_type == "getattr":
                        is_attribute = self._xparl_remote_wrapper_obj.has_attr(
                            calling_request.attr)

                        if is_attribute:
                            return_result = self._xparl_remote_wrapper_obj.get_remote_attr(
                                calling_request.attr)
                        else:
                            function_wrapper = self._xparl_remote_wrapper_obj.get_remote_attr(
                                calling_request.attr)
                            return_result = function_wrapper(
                                *calling_request.args,
                                **calling_request.kwargs)
                        calling_request.future_return_queue.put(return_result)
                    else:
                        break

                except Exception as e:
                    async_error = FutureFunctionError(calling_request.attr)
                    self._xparl_remote_object_exception = async_error
                    if calling_request.calling_type == "getattr":
                        calling_request.future_return_queue.put(async_error)
                    raise e

                finally:
                    self._xparl_calling_finished_event.set()

        def __getattr__(self, attr):
            self._xparl_remote_wrapper_internal_lock.acquire()

            self._xparl_calling_finished_event.wait(
            )  # waiting for last function finishing before calling has_attr

            if self._xparl_remote_object_exception is not None:
                time.sleep(
                    0.1
                )  # waiting for another thread printing the error message
                raise self._xparl_remote_object_exception
            """
            Don't use the following way, which will call the __getattr__ function and acquire the lock again.
                is_attribute = self._xparl_remote_wrapper_obj.has_attr(attr)
            """
            is_attribute = self.__dict__['_xparl_remote_wrapper_obj'].has_attr(
                attr)

            self._xparl_remote_wrapper_internal_lock.release()

            def wrapper(*args, **kwargs):
                self._xparl_remote_wrapper_internal_lock.acquire()

                self._xparl_calling_finished_event.wait()
                self._xparl_calling_finished_event.clear()

                if self._xparl_remote_object_exception is not None:
                    time.sleep(
                        0.1
                    )  # waiting for another thread printing the error message
                    raise self._xparl_remote_object_exception

                future_return_queue = queue.Queue()
                calling_request = CallingRequest(
                    calling_type="getattr",
                    attr=attr,
                    value=None,
                    args=args,
                    kwargs=kwargs,
                    future_return_queue=future_return_queue)

                self.__dict__['_xparl_remote_wrapper_calling_queue'].put(
                    calling_request)

                future_object = FutureObject(future_return_queue)

                self._xparl_remote_wrapper_internal_lock.release()
                return future_object

            if is_attribute:
                future_object = wrapper()
                return future_object.get()
            else:
                return wrapper

        def __setattr__(self, attr, value):
            if attr.startswith(XPARL_RESERVED_PREFIX):
                super(ProxyWrapperNoWait, self).__setattr__(attr, value)
            else:
                self._xparl_remote_wrapper_internal_lock.acquire()

                self._xparl_calling_finished_event.wait()
                self._xparl_calling_finished_event.clear()

                if self._xparl_remote_object_exception is not None:
                    time.sleep(
                        0.1
                    )  # waiting for another thread to print the error message
                    raise self._xparl_remote_object_exception

                calling_request = CallingRequest(
                    calling_type="setattr",
                    attr=attr,
                    value=value,
                    args=None,
                    kwargs=None,
                    future_return_queue=None)
                self._xparl_remote_wrapper_calling_queue.put(calling_request)

                self._xparl_remote_wrapper_internal_lock.release()

    return ProxyWrapperNoWait
