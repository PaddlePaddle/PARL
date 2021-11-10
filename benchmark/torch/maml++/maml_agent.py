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
import scipy
import parl
import tqdm


class MAMLAgent(parl.Agent):
    def __init__(self, algorithm, data, total_iter_per_epoch):
        super().__init__(algorithm)

        self.algorithm = algorithm
        self.data = data
        self.total_iter_per_epoch = total_iter_per_epoch

    def train_one_epoch(self, current_epoch):

        num_iters = 0
        with tqdm.tqdm(total=self.total_iter_per_epoch) as bar:
            while num_iters < self.total_iter_per_epoch:
                for data_batch in self.data.get_train_batches():
                    loss = self.algorithm.train_one_iter(
                        data_batch, current_epoch)
                    num_iters += 1
                    bar.update(1)

                    if num_iters >= self.total_iter_per_epoch:
                        break

    def evaluate(self):
        '''Evaluate current model on test set
        
        Returns:
            Confidence interval of test losses.
        '''
        total_loss = []
        for data_batch in self.data.get_test_batches():
            loss = self.algorithm.evaluate_one_iter(data_batch)
            total_loss.append(loss)

        return self._confidence_interval(total_loss)

    @staticmethod
    def _confidence_interval(data, confidence=0.95):
        '''
        Caculate confidence interval of losses.

        Args:
            data: A list of losses.
            confidence: Confidence level.

        Returns:
            Mean and margin of error.
        '''

        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

        return m, h
