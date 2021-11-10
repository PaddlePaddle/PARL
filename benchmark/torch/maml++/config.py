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

from dataclasses import dataclass
import math


@dataclass
class Config:
    seed = 0

    network_dims = [1, 40, 40, 40, 1]
    num_layers = len(network_dims) - 1

    total_epochs = 80
    total_iter_per_epoch = 500

    second_order = True

    first_order_to_second_order_epoch = total_epochs // 2

    min_learning_rate = 0.00001
    meta_learning_rate = 0.001
    task_learning_rate = 0.001

    num_updates_per_iter = 3

    learnable_learning_rates = True
    learning_rate_scheduler = True

    use_multi_step_loss_optimization = True
    multi_step_loss_num_epochs = 10

    num_training_tasks = 1600
    training_batch_size = 16
    num_training_support = 5
    num_training_query = 10

    num_test_tasks = 10000
    test_batch_size = num_test_tasks // 100
    num_test_support = 5
    num_test_query = 100

    amplitude = (0.1, 5.0)
    frequency = (0.8, 1.2)
    phase = (0, math.pi)
    x_range = (-5, 5)
