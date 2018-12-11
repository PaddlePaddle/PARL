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

import os
import subprocess
from parl.utils import logger

__all__ = ['get_gpu_count']


def get_gpu_count():
    """ get avaliable gpu count

    Returns:
        gpu_count: int    
    """

    gpu_count = 0

    env_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env_cuda_devices is not None:
        assert isinstance(env_cuda_devices, str)
        try:
            if not env_cuda_devices:
                return 0
            gpu_count = len(
                [x for x in env_cuda_devices.split(',') if int(x) >= 0])
            logger.info(
                'CUDA_VISIBLE_DEVICES found gpu count: {}'.format(gpu_count))
        except:
            logger.warn('Cannot find available GPU devices, using CPU now.')
            gpu_count = 0
    else:
        try:
            gpu_count = str(subprocess.check_output(["nvidia-smi",
                                                     "-L"])).count('UUID')
            logger.info('nvidia-smi -L found gpu count: {}'.format(gpu_count))
        except:
            logger.warn('Cannot find available GPU devices, using CPU now.')
            gpu_count = 0
    return gpu_count
