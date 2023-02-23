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

XPARL_RESERVED_PREFIX = "_xparl"
RESERVED_NAME_ERROR_STR = "Name starting with `_xparl` is the reserved name in xparl, please don't use the name `{}`."


def proxy_wrapper_func(remote_wrapper):
    '''
    The 'proxy_wrapper_func' is defined on the top of class 'RemoteWrapper'
    in order to set and get attributes of 'remoted_wrapper' and the corresponding 
    remote models individually. 

    This decorator function allows the user access the attributes of remote objects 
    like local objects.

    For example:
        ```python
        import parl

        @parl.remote_class()
        class Actor(object):
            def __init__(self):
                self.arg1 = 0

            def func(self):
                # do something
                return 0

        parl.connect("localhost:8010")

        actor = Actor()
        print(actor.arg1)
        actor.arg2 = 10
        print(actor.func())
        ```
    '''

    original_class = remote_wrapper._original
    max_memory = remote_wrapper._max_memory
    n_gpu = remote_wrapper._n_gpu

    class ProxyWrapper(object):
        def __init__(self, *args, **kwargs):
            for key in kwargs:
                assert not key.startswith(XPARL_RESERVED_PREFIX), RESERVED_NAME_ERROR_STR.format(key)

            # The following variables will be used in the RemoteWrapper, so we put them
            # into the kwargs.
            kwargs['_xparl_remote_class'] = original_class
            kwargs['_xparl_remote_class_max_memory'] = max_memory
            kwargs['_xparl_remote_class_n_gpu'] = n_gpu

            self._xparl_remote_wrapper_obj = remote_wrapper(*args, **kwargs)
            for key in self._xparl_remote_wrapper_obj.get_attrs():
                assert not key.startswith(XPARL_RESERVED_PREFIX), RESERVED_NAME_ERROR_STR.format(key)

        def __getattr__(self, attr):
            return self._xparl_remote_wrapper_obj.get_remote_attr(attr)

        def __setattr__(self, attr, value):
            if attr.startswith(XPARL_RESERVED_PREFIX):
                super(ProxyWrapper, self).__setattr__(attr, value)
            else:
                self._xparl_remote_wrapper_obj.set_remote_attr(attr, value)

    return ProxyWrapper
