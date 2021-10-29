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

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np


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

    def __init__(self, num_classes, support, query, amplitude, frequency,
                 phase, x_range):

        self.num_classes = num_classes
        self.support = support
        self.query = query

        self.x = []
        self.y = []

        for _ in range(num_classes):
            task_amplitude = random.uniform(*amplitude)
            task_frequency = random.uniform(*frequency)
            task_phase = random.uniform(*phase)

            task_x = np.random.uniform(*x_range,
                                       support + query).astype(np.float32)
            task_y = task_amplitude * np.sin(
                task_frequency * task_x + task_phase, dtype=np.float32)

            self.x.append(task_x)
            self.y.append(task_y)

    def __getitem__(self, idx):

        # randomly split support and query
        idxes = np.arange(self.support + self.query)
        np.random.shuffle(idxes)

        return torch.from_numpy(self.x[idx][idxes[:self.support]]).unsqueeze(1), torch.from_numpy(self.y[idx][idxes[:self.support]]).unsqueeze(1), \
            torch.from_numpy(self.x[idx][idxes[self.support:]]).unsqueeze(1), torch.from_numpy(self.y[idx][idxes[self.support:]]).unsqueeze(1)

    def __len__(self):
        return self.num_classes


class MetaLearningDataLoader:
    def __init__(self, num_training_tasks, training_batch_size,
                 num_training_support, num_training_query, num_test_tasks,
                 test_batch_size, num_test_support, num_test_query, amplitude,
                 frequency, phase, x_range):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.

        Args:
            config

        """
        self.training_dataset = MetaLearningDataSet(
            num_training_tasks, num_training_support, num_training_query,
            amplitude, frequency, phase, x_range)
        self.test_dataset = MetaLearningDataSet(
            num_test_tasks, num_test_support, num_test_query, amplitude,
            frequency, phase, x_range)
        self.training_batch_size = training_batch_size
        self.test_batch_size = test_batch_size

    def get_train_batches(self):
        """Returns a training batches data_loader"""

        data_loader = DataLoader(
            self.training_dataset,
            batch_size=self.training_batch_size,
            shuffle=True,
            drop_last=True)
        for sample_batched in data_loader:
            yield sample_batched

    def get_test_batches(self):
        """Returns a testing batches data_loader"""

        data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=True,
            drop_last=True)
        for sample_batched in data_loader:
            yield sample_batched
