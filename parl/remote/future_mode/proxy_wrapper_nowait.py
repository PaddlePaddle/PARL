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

import queue
import time
import threading

from parl.remote.exceptions import FutureFunctionError
from parl.remote.future_mode.utils import FutureObject, CallingRequest


def proxy_wrapper_nowait_func(remote_wrapper, max_memory):
    '''
    The 'proxy_wrapper_nowait_func' is defined on the top of class 'RemoteWrapper'
    when using the `@remote_class(wait=False)` decorator. The main purpose is as follows:

    1. set and get attributes of 'remoted_wrapper' and the corresponding 
    remote models individually. (With 'proxy_wrapper_nowait_func', it is allowed to 
    define a attribute (or method) of the same name in 'RemoteWrapper' and remote models.

    2. provide asynchronous function calling. When user calling a function, he will get
    the return(`FutureObject`) immediately and he need call the `get` function to get the
    real return. For example:
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
    '''

    class ProxyWrapperNoWait(object):
        def __init__(self, *args, **kwargs):
            assert '__xparl_proxy_wrapper_nowait__' not in kwargs, "`__xparl_proxy_wrapper_nowait__` is the reserved variable name in xparl, please use other names"
            kwargs['__xparl_proxy_wrapper_nowait__'] = self

            assert '__xparl_remote_class__' not in kwargs, "`__xparl_remote_class__` is the reserved variable name in xparl, please use other names"
            kwargs['__xparl_remote_class__'] = remote_wrapper._original

            assert '__xparl_remote_class_max_memory__' not in kwargs, "`__xparl_remote_class_max_memory__` is the reserved variable name in xparl, please use other names"
            kwargs['__xparl_remote_class_max_memory__'] = max_memory

            self.xparl_remote_wrapper_calling_queue = queue.Queue()
            self.xparl_remote_wrapper_internal_lock = threading.Lock()
            self.xparl_calling_finished_event = threading.Event()
            self.xparl_remote_object_exception = None

            object_thread = threading.Thread(
                target=self._run_object_in_backend, args=(args, kwargs))
            object_thread.setDaemon(True)
            object_thread.start()

        def _run_object_in_backend(self, args, kwargs):
            try:
                self.xparl_remote_wrapper_obj = remote_wrapper(*args, **kwargs)

                assert not self.xparl_remote_wrapper_obj.has_attr(
                    'xparl_remote_wrapper_obj'
                ), "`xparl_remote_wrapper_obj` is the reserved variable name in PARL, please use other names"
                assert not self.xparl_remote_wrapper_obj.has_attr(
                    'xparl_remote_wrapper_calling_queue'
                ), "`xparl_remote_wrapper_calling_queue` is the reserved variable name in PARL, please use other names"
                assert not self.xparl_remote_wrapper_obj.has_attr(
                    'xparl_remote_wrapper_internal_lock'
                ), "`xparl_remote_wrapper_internal_lock` is the reserved variable name in PARL, please use other names"
                assert not self.xparl_remote_wrapper_obj.has_attr(
                    'xparl_calling_finished_event'
                ), "`xparl_calling_finished_event` is the reserved variable name in PARL, please use other names"
                assert not self.xparl_remote_wrapper_obj.has_attr(
                    'xparl_remote_object_exception'
                ), "`xparl_remote_object_exception` is the reserved variable name in PARL, please use other names"
            except Exception as e:
                async_error = FutureFunctionError('__init__')
                self.xparl_remote_object_exception = async_error
                self.xparl_calling_finished_event.set()
                raise e
            """
            NOTE:
                We should set the event after the initialization of self.xparl_remote_wrapper_obj.
                Only after the initialization is complete can we call the function of actor.
            """
            self.xparl_calling_finished_event.set()

            while True:
                calling_request = self.xparl_remote_wrapper_calling_queue.get()

                if calling_request.calling_type == "setattr":
                    try:
                        self.xparl_remote_wrapper_obj.set_remote_attr(
                            calling_request.attr, calling_request.value)
                    except Exception as e:
                        async_error = FutureFunctionError(calling_request.attr)
                        self.xparl_remote_object_exception = async_error
                        self.xparl_calling_finished_event.set()
                        raise e

                    self.xparl_calling_finished_event.set()

                elif calling_request.calling_type == "getattr":
                    try:
                        is_attribute = self.xparl_remote_wrapper_obj.has_attr(
                            calling_request.attr)

                        if is_attribute:
                            return_result = self.xparl_remote_wrapper_obj.get_remote_attr(
                                calling_request.attr)
                        else:
                            function_wrapper = self.xparl_remote_wrapper_obj.get_remote_attr(
                                calling_request.attr)
                            return_result = function_wrapper(
                                *calling_request.args,
                                **calling_request.kwargs)
                    except Exception as e:
                        async_error = FutureFunctionError(calling_request.attr)
                        self.xparl_remote_object_exception = async_error
                        calling_request.future_return_queue.put(async_error)
                        self.xparl_calling_finished_event.set()
                        raise e

                    calling_request.future_return_queue.put(return_result)

                    self.xparl_calling_finished_event.set()
                else:
                    assert False, "undefined calling type"

        def __getattr__(self, attr):
            self.xparl_remote_wrapper_internal_lock.acquire()

            self.xparl_calling_finished_event.wait(
            )  # waiting for last function finishing before calling has_attr

            if self.xparl_remote_object_exception is not None:
                time.sleep(
                    0.1
                )  # waiting for another thread printing the error message
                raise self.xparl_remote_object_exception
            """
            Don't use the following way, which will call the __getattr__ function and acquire the lock again.
                is_attribute = self.xparl_remote_wrapper_obj.has_attr(attr)
            """
            is_attribute = self.__dict__['xparl_remote_wrapper_obj'].has_attr(
                attr)

            self.xparl_remote_wrapper_internal_lock.release()

            def wrapper(*args, **kwargs):
                self.xparl_remote_wrapper_internal_lock.acquire()

                self.xparl_calling_finished_event.wait()
                self.xparl_calling_finished_event.clear()

                if self.xparl_remote_object_exception is not None:
                    time.sleep(
                        0.1
                    )  # waiting for another thread printing the error message
                    raise self.xparl_remote_object_exception

                future_return_queue = queue.Queue()
                calling_request = CallingRequest(
                    calling_type="getattr",
                    attr=attr,
                    value=None,
                    args=args,
                    kwargs=kwargs,
                    future_return_queue=future_return_queue)

                self.__dict__['xparl_remote_wrapper_calling_queue'].put(
                    calling_request)

                future_object = FutureObject(future_return_queue)

                self.xparl_remote_wrapper_internal_lock.release()
                return future_object

            if is_attribute:
                future_object = wrapper()
                return future_object.get()
            else:
                return wrapper

        def __setattr__(self, attr, value):
            if attr in [
                    'xparl_remote_wrapper_obj',
                    'xparl_remote_wrapper_calling_queue',
                    'xparl_remote_wrapper_internal_lock',
                    'xparl_calling_finished_event',
                    'xparl_remote_object_exception'
            ]:
                super(ProxyWrapperNoWait, self).__setattr__(attr, value)
            else:
                self.xparl_remote_wrapper_internal_lock.acquire()

                self.xparl_calling_finished_event.wait()
                self.xparl_calling_finished_event.clear()

                if self.xparl_remote_object_exception is not None:
                    time.sleep(
                        0.1
                    )  # waiting for another thread printing the error message
                    raise self.xparl_remote_object_exception

                calling_request = CallingRequest(
                    calling_type="setattr",
                    attr=attr,
                    value=value,
                    args=None,
                    kwargs=None,
                    future_return_queue=None)
                self.xparl_remote_wrapper_calling_queue.put(calling_request)

                self.xparl_remote_wrapper_internal_lock.release()

    return ProxyWrapperNoWait
