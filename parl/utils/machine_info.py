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
import random
import socket
import subprocess
from parl.utils import logger, _HAS_FLUID, _IS_WINDOWS

__all__ = [
    'get_gpu_count', 'get_ip_address', 'is_gpu_available', 'get_free_tcp_port',
    'is_port_available', 'get_port_from_range'
]


def get_ip_address():
    """
    get the IP address of the host.
    """

    # Windows
    if _IS_WINDOWS:
        local_ip = socket.gethostbyname(socket.gethostname())
    else:
        # Linux and MacOS
        local_ip = None
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
            logger.info('Cannot find available GPU devices, using CPU now.')
            gpu_count = 0
    else:
        try:
            gpu_count = str(subprocess.check_output(["nvidia-smi",
                                                     "-L"])).count('UUID')
            logger.info('nvidia-smi -L found gpu count: {}'.format(gpu_count))
        except:
            logger.info('Cannot find available GPU devices, using CPU now.')
            gpu_count = 0
    return gpu_count


def is_gpu_available():
    """ check whether parl can access a GPU

    Returns:
      True if a gpu device can be found.
    """
    ret = get_gpu_count() > 0
    if _HAS_FLUID:
        from paddle import fluid
        if ret is True and not fluid.is_compiled_with_cuda():
            logger.warning("Found non-empty CUDA_VISIBLE_DEVICES. \
                But PARL found that Paddle was not complied with CUDA, which may cause issues."
                           )
    return ret


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return str(port)


def is_port_available(port):
    """ Check if a port is used.

    True if the port is available for connection.
    """
    port = int(port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    available = sock.connect_ex(('localhost', port))
    sock.close()
    return available


def get_port_from_range(start, end):
    while True:
        port = random.randint(start, end)
        if is_port_available(port):
            break

    return port
