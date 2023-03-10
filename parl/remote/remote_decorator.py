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

import os
import inspect
import numpy as np

from parl.utils import logger
from parl.remote.remote_wrapper import RemoteWrapper
from parl.remote.proxy_wrapper import proxy_wrapper_func
from parl.remote.future_mode import proxy_wrapper_nowait_func


def remote_class(*args, **kwargs):
    """A Python decorator that enables a class to run all its functions
    remotely.

    Each instance of the remote class can be seemed as a task submitted
    to the cluster by the global client, which is created automatically
    when we call parl.connect(master_address). After global client
    submits the task, the master node will send an available job address
    to this remote instance. Then the remote object will send local python
    files, class definition and initialization arguments to the related job.

    In this way, we can run distributed applications easily and efficiently.

    .. code-block:: python

        @parl.remote_class
        class Actor(object):
            def __init__(self, x):
                self.x = x

            def step(self):
                self.x += 1
                return self.x

        actor = Actor()
        actor.step()

        # Set maximum memory usage to 300 MB for each object.
        @parl.remote_class(max_memory=300)
        class LimitedActor(object):
           ...

    Args:
        max_memory (float): Maximum memory (MB) can be used by each remote
                            instance, the unit is in MB and default value is
                            none(unlimited).
        n_gpu (int): The number of GPUs required to run the remote instance.

    Returns:
        A remote wrapper for the remote class.

    Raises:
        Exception: An exception is raised if the client is not created
                   by `parl.connect(master_address)` beforehand.
    """

    def decorator(cls):
        assert inspect.isclass(cls), "Only class can be decorated by `parl.remote_class`."

        # we are not going to create a remote actor in job.py
        if 'XPARL' in os.environ and os.environ['XPARL'] == 'True':
            logger.warning("Note: this object will be runnning as a local object")
            return cls

        RemoteWrapper._original = cls
        RemoteWrapper._max_memory = max_memory
        RemoteWrapper._n_gpu = n_gpu

        if wait:
            proxy_wrapper = proxy_wrapper_func(RemoteWrapper)
        else:
            # nowait
            proxy_wrapper = proxy_wrapper_nowait_func(RemoteWrapper)

        proxy_wrapper._original = cls
        return proxy_wrapper

    args_names = ['max_memory', 'wait', 'n_gpu']
    for key in kwargs:
        assert key in args_names, "Argument `{}` is not supported in the `@parl.remote_class`, supported arguments: {}".format(
            key, args_names)

    max_memory = kwargs.get('max_memory')
    wait = kwargs.get('wait', True)
    n_gpu = kwargs.get('n_gpu', 0)
    """
        Users may pass some arguments to the decorator (e.g., parl.remote_class(10)).
        The following code tries to handle this issue.

        The `args` is different in the following two decorating way, and we should return different wrapper.
        @parl.remote_class     -> args: (<class '__main__.Actor'>,) -> we should return decorator(cls)
        @parl.remote_class(10) -> args: (10,)                       -> we should return decorator
    """
    if len(args) == 1 and callable(args[0]):  # args[0]: cls
        # The first element in the `args` is a class, we should return decorator(cls)
        return decorator(args[0])

    # The first element in the `args` is not a class, we should return decorator
    return decorator
