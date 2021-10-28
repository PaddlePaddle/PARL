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

from typing import Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import math
import random
import numpy as np
from config import Config

class MetaLearningDataSet(Dataset):
    """ Dataset that generates samples from different sine curves by following the
    setting in Mate-SGD: https://arxiv.org/pdf/1707.09835.pdf
    config:
        num_classes: number of different sine curves.
        support: number of support samples for each sine curve.
        query: number of query samples for each sine curve.
        amplitude: range of amplitude of sine curves.
        frequency: range of frequency of sine curves.
        phase: range of phase of sine curves.
        x_range: range of x of sine curves.
        noise: if add noise to sine curves.
    """
    def __init__(self, num_classes: int, support: int, query: int, amplitude: Tuple[float] = (0.1, 5.0),
                frequency: Tuple[float] = (0.8, 1.2), phase: Tuple[float] = (0, math.pi), 
                x_range: Tuple[float] = (-5, 5), noise: bool = False):

        self.num_classes = num_classes
        self.support = support
        self.query = query
        self.noise = noise

        self.x = []
        self.y = []

        for _ in range(num_classes):
            task_amplitude = random.uniform(*amplitude)
            task_frequency = random.uniform(*frequency)
            task_phase = random.uniform(*phase)

            task_x = np.random.uniform(*x_range, support+query).astype(np.float32)
            task_y = task_amplitude * np.sin(task_frequency*task_x+task_phase, dtype=np.float32)

            self.x.append(task_x)
            self.y.append(task_y)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idxes = np.arange(self.support+self.query)
        np.random.shuffle(idxes)

        if self.noise:
            noisy_y = self.y[idx] + np.random.normal(0, 0.1, self.support+self.query).astype(np.float32)
        else:
            noisy_y = self.y[idx]

        return torch.from_numpy(self.x[idx][idxes[:self.support]]).unsqueeze(1), torch.from_numpy(noisy_y[idxes[:self.support]]).unsqueeze(1), \
            torch.from_numpy(self.x[idx][idxes[self.support:]]).unsqueeze(1), torch.from_numpy(noisy_y[idxes[self.support:]]).unsqueeze(1)

    def __len__(self) -> int:
        return self.num_classes



class MetaLearningDataLoader:
    def __init__(self, config: Config):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.

        Args:
            config

        """
        self.training_dataset = MetaLearningDataSet(config.num_training_sample, config.num_training_support, config.num_training_query)
        self.test_dataset = MetaLearningDataSet(config.num_test_sample, config.num_test_support, config.num_test_query)
        self.config = config

    def get_train_batches(self):
        """Returns a training batches data_loader"""

        data_loader = DataLoader(self.training_dataset, batch_size=self.config.training_batch_size, shuffle=True, drop_last=True)
        for sample_batched in data_loader:
            yield sample_batched


    def get_test_batches(self):
        """Returns a testing batches data_loader"""

        data_loader = DataLoader(self.test_dataset, batch_size=self.config.test_batch_size, shuffle=True, drop_last=True)
        for sample_batched in data_loader:
            yield sample_batched
