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

import argparse


def get_server_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, type=int, help='server port')
    parser.add_argument(
        '--logdir', type=str, help='directory to save model/tensorboard data')
    parser.add_argument(
        '--restore_model_path',
        type=str,
        help='restore model path for warm start')
    parser.add_argument(
        '--restore_from_one_head',
        action="store_true",
        help=
        'If set, will restore model from one head model. If ensemble_num > 1, will assign parameters of model0 to other models.'
    )
    parser.add_argument(
        '--restore_rpm_path', type=str, help='restore rpm path for warm start')
    parser.add_argument(
        '--ensemble_num',
        type=int,
        required=True,
        help='model number to ensemble')
    parser.add_argument(
        '--warm_start_batchs',
        type=int,
        default=100,
        help='collect how many batch data to warm start')
    args = parser.parse_args()
    return args


def get_client_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stage',
        default=0,
        type=int,
        help='''
          stage number, which decides change times of target velocity. 
          Eg: stage=0 will keep target_v 1.25m/s;
              stage=3 will change target velocity 3 times, just like Round2 env.'''
    )
    parser.add_argument('--ident', type=int, required=False, help='worker id')
    parser.add_argument('--ip', type=str, required=True, help='server ip')
    parser.add_argument('--port', type=int, required=True, help='server port')
    parser.add_argument(
        '--target_v', type=float, help='target velocity for training')
    parser.add_argument(
        '--act_penalty_lowerbound',
        type=float,
        help='lower bound of action l2 norm penalty')
    parser.add_argument(
        '--act_penalty_coeff',
        type=float,
        default=5.0,
        help='coefficient of action l2 norm penalty')
    parser.add_argument(
        '--vel_penalty_coeff',
        type=float,
        default=1.0,
        help='coefficient of velocity gap penalty')
    parser.add_argument(
        '--discrete_data',
        action="store_true",
        help=
        'if set, discrete target velocity in last stage (args.stage), make target velocity more uniform.'
    )
    parser.add_argument(
        '--discrete_bin',
        type=int,
        default=10,
        help='discrete target velocity in last stage to how many intervals')
    parser.add_argument(
        '--reward_type',
        type=str,
        help=
        "Choose reward type, 'RunFastest' or 'FixedTargetSpeed' or 'Round2'")
    parser.add_argument(
        '--debug',
        action="store_true",
        help='if set, will print debug information')
    args = parser.parse_args()

    assert args.reward_type in ['RunFastest', 'FixedTargetSpeed', 'Round2']

    return args
