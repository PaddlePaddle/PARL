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

from tensorboardX import SummaryWriter
from parl.utils import logger
from parl.utils.machine_info import get_ip_address

__all__ = []

_writer = None
_WRITTER_METHOD = ['add_scalar', 'add_histogram', 'close', 'flush']


def create_file_after_first_call(func_name):
    def call(*args, **kwargs):
        global _writer
        if _writer is None:
            logdir = logger.get_dir()
            if logdir is None:
                logdir = logger.auto_set_dir(action='d')
                logger.warning(
                    "[tensorboard] logdir is None, will save tensorboard files to {}\nView the data using: tensorboard --logdir=./{} --host={}"
                    .format(logdir, logdir, get_ip_address()))
            _writer = SummaryWriter(logdir=logger.get_dir())
        func = getattr(_writer, func_name)
        func(*args, **kwargs)
        _writer.flush()

    return call


# export writter functions
for func_name in _WRITTER_METHOD:
    locals()[func_name] = create_file_after_first_call(func_name)
    __all__.append(func_name)
