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

import argparse
import os
import threading
from parl.remote import Master, Worker


def main(args):
    """Start a master or a worker through:

    1. xparl start --port 1234
    2. xparl connect --address localhost:1234 --cpu_num 8

    """

    if args.name == 'master':
        port = args.port
        monitor_port = args.monitor_port
        master = Master(port, monitor_port)
        master.run()

    elif args.name == 'worker':
        address = args.address
        log_server_port = args.log_server_port
        cpu_num = int(args.cpu_num) if args.cpu_num else None
        worker = Worker(address, cpu_num, log_server_port)
        worker.run()

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', default='master', type=str, help='master/worker')
    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--address', default='localhost:1234', type=str)
    parser.add_argument('--cpu_num', default='', type=str)
    parser.add_argument('--monitor_port', default='', type=str)
    parser.add_argument('--log_server_port', default='', type=str)
    args = parser.parse_args()
    main(args)
