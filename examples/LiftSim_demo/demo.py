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

from rlschool import LiftSim
from wrapper import Wrapper, ActionWrapper, ObservationWrapper
from rl_benchmark.dispatcher import RL_dispatcher
import sys
import argparse


# run main program with args
def run_main(args):

    parser = argparse.ArgumentParser(description='demo configuration')
    parser.add_argument(
        '--iterations',
        type=int,
        default=100000000,
        help='total number of iterations')
    args = parser.parse_args(args)
    print('iterations:', args.iterations)

    mansion_env = LiftSim()
    # mansion_env.seed(1988)

    mansion_env = Wrapper(mansion_env)
    mansion_env = ActionWrapper(mansion_env)
    mansion_env = ObservationWrapper(mansion_env)

    dispatcher = RL_dispatcher(mansion_env, args.iterations)
    dispatcher.run_episode()

    return 0


if __name__ == "__main__":
    run_main(sys.argv[1:])
