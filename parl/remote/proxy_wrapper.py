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


def proxy_wrapper_func(remote_wrapper, max_memory):
    '''
    The 'proxy_wrapper_func' is defined on the top of class 'RemoteWrapper'
    in order to set and get attributes of 'remoted_wrapper' and the corresponding 
    remote models individually. 

    With 'proxy_wrapper_func', it is allowed to define a attribute (or method) of
    the same name in 'RemoteWrapper' and remote models.
    '''

    class ProxyWrapper(object):
        def __init__(self, *args, **kwargs):
            assert '__xparl_proxy_wrapper_nowait__' not in kwargs, "`__xparl_proxy_wrapper_nowait__` is the reserved variable name in xparl, please use other names"

            assert '__xparl_remote_class__' not in kwargs, "`__xparl_remote_class__` is the reserved variable name in xparl, please use other names"
            kwargs['__xparl_remote_class__'] = remote_wrapper._original

            assert '__xparl_remote_class_max_memory__' not in kwargs, "`__xparl_remote_class_max_memory__` is the reserved variable name in xparl, please use other names"
            kwargs['__xparl_remote_class_max_memory__'] = max_memory

            self.xparl_remote_wrapper_obj = remote_wrapper(*args, **kwargs)
            assert not self.xparl_remote_wrapper_obj.has_attr(
                'xparl_remote_wrapper_obj'
            ), "`xparl_remote_wrapper_obj` is the reserved variable name in PARL, please use other names"

        def __getattr__(self, attr):
            return self.xparl_remote_wrapper_obj.get_remote_attr(attr)

        def __setattr__(self, attr, value):
            if attr == 'xparl_remote_wrapper_obj':
                super(ProxyWrapper, self).__setattr__(attr, value)
            else:
                self.xparl_remote_wrapper_obj.set_remote_attr(attr, value)

    return ProxyWrapper
