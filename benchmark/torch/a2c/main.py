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

import time
from learner import Learner
import torch
import os
import argparse
from parl.utils import logger


def main(config):
    cuda = torch.cuda.is_available()
    learner = Learner(config, cuda)
    assert config['log_metrics_interval_s'] > 0

    while not learner.should_stop():
        start = time.time()
        while time.time() - start < config['log_metrics_interval_s']:
            learner.step()
        learner.log_metrics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=3e-4, help='learning_rate')
    parser.add_argument(
        '--vf_loss_coeff',
        default=0.5,
        help='hyper-parameter for the value function loss')
    args = parser.parse_args()
    from a2c_config import config
    logger.set_dir(
        os.path.join('./train_log', 'lr_{}_vf_{}_norm'.format(
            args.lr, args.vf_loss_coeff)))
    config['start_lr'] = float(args.lr)
    config['vf_loss_coeff'] = float(args.vf_loss_coeff)
    main(config)
