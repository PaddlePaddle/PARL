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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cluster_address',
        default='localhost:8010',
        type=str,
        help='cluster address of xparl.')
    parser.add_argument(
        '--actor_num', type=int, required=True, help='number of actors.')
    parser.add_argument(
        '--logdir',
        type=str,
        default='logdir',
        help='directory to save model/tensorboard data')

    parser.add_argument(
        '--difficulty',
        type=int,
        required=True,
        help=
        'difficulty of L2M2019Env. difficulty=1 means Round 1 environment but target theta is always 0; difficulty=2 menas Round 1 environment; difficulty=3 means Round 2 environment.'
    )
    parser.add_argument(
        '--vel_penalty_coeff',
        type=float,
        default=1.0,
        help='coefficient of velocity penalty in reward shaping.')
    parser.add_argument(
        '--muscle_penalty_coeff',
        type=float,
        default=1.0,
        help='coefficient of muscle penalty in reward shaping.')
    parser.add_argument(
        '--penalty_coeff',
        type=float,
        default=1.0,
        help='coefficient of all penalty in reward shaping.')
    parser.add_argument(
        '--only_first_target',
        action="store_true",
        help=
        'if set, will terminate the environment run after the first target finished.'
    )

    parser.add_argument(
        '--saved_models_dir',
        type=str,
        required=True,
        help='directory of saved models.')
    parser.add_argument(
        '--offline_evaluate',
        action="store_true",
        help='if set, will evaluate models offline.')
    parser.add_argument(
        '--evaluate_times',
        default=300,
        type=int,
        help='evaluate episodes per model.')

    args = parser.parse_args()

    return args
