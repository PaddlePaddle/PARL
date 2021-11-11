#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import random
import torch
from parl.utils import logger, tensorboard
import numpy as np
from data import MetaLearningDataLoader
from maml_model import MAMLModel
from maml_algorithm import MAML
from maml_agent import MAMLAgent
from config import Config


def main():
    """
    Runs a full training experiment with evaluations of the model on the val set at every epoch. Furthermore,
    will return the test set evaluation results on the best performing validation model.
    """
    config = Config()

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    data = MetaLearningDataLoader(
        config.num_training_tasks, config.training_batch_size,
        config.num_training_support, config.num_training_query,
        config.num_test_tasks, config.test_batch_size, config.num_test_support,
        config.num_test_query, config.amplitude, config.frequency,
        config.phase, config.x_range)

    model = MAMLModel(config.network_dims, device)
    algo = MAML(model, device, config.num_updates_per_iter, config.num_layers,
                config.task_learning_rate, config.learnable_learning_rates,
                config.meta_learning_rate, config.total_epochs,
                config.min_learning_rate, config.learning_rate_scheduler,
                config.second_order, config.use_multi_step_loss_optimization,
                config.multi_step_loss_num_epochs)
    agent = MAMLAgent(algo, data, config.total_iter_per_epoch)

    for i in range(config.total_epochs):
        logger.info('start epoch {}'.format(i + 1))
        agent.train_one_epoch(i)
        loss, h = agent.evaluate()
        tensorboard.add_scalar('test/loss', loss, i)
        logger.info('epoch {}: test loss: {:.3f}+-{:.3f}'.format(
            i + 1, loss, h))


if __name__ == '__main__':
    main()
