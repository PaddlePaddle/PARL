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
import platform
import subprocess
from parl.utils import logger
from parl.utils import utils

__all__ = ['get_gpu_count', 'get_ip_address', 'is_gpu_available']


def get_ip_address():
    """
    get the IP address of the host.
    """
    platform_sys = platform.system()

    # Only support Linux and MacOS
    if platform_sys != 'Linux' and platform_sys != 'Darwin':
        logger.warning(
            'get_ip_address only support Linux and MacOS, please set ip address manually.'
        )
        return None

    local_ip = None
    import socket
    try:
        # First way, tested in Ubuntu and MacOS
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        # Second way, tested in CentOS
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
        except:
            pass

    if local_ip == None or local_ip == '127.0.0.1' or local_ip == '127.0.1.1':
        logger.warning(
            'get_ip_address failed, please set ip address manually.')
        return None

    return local_ip


def get_gpu_count():
    """get avaliable gpu count

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


def is_gpu_available():
    """ check whether parl can access a GPU

    Returns:
      True if a gpu device can be found.
    """
    ret = get_gpu_count() > 0
    if utils._HAS_FLUID:
        from paddle import fluid
        if ret is True and not fluid.is_compiled_with_cuda():
            logger.warn("Found non-empty CUDA_VISIBLE_DEVICES. \
                But PARL found that Paddle was not complied with CUDA, which may cause issues."
                        )
    return ret
