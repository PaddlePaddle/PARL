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

from statistics import NormalDist
from typing import List, Tuple
import parl
import tqdm
from maml_algorithm import MAML
from data import MetaLearningDataLoader
from config import Config

class MAMLAgent(parl.Agent):
    def __init__(self, algorithm: MAML, data: MetaLearningDataLoader, config: Config):
        super().__init__(algorithm)

        self.algorithm = algorithm
        self.data = data
        self.config = config

    def train_one_epoch(self, current_epoch: int):

        num_iters = 0
        with tqdm.tqdm(total=self.config.total_iter_per_epoch) as bar:
            while num_iters < self.config.total_iter_per_epoch:
                for data_batch in self.data.get_train_batches():
                    loss = self.algorithm.train_one_iter(data_batch, current_epoch)
                    num_iters += 1
                    bar.update(1)
                    bar.set_description(f'training loss: {loss:.3f}')

                    if num_iters >= self.config.total_iter_per_epoch: break

    def evaluate(self) -> Tuple[float, float]:
        '''Evaluate current model on test set
        
        Returns:
            Confidence interval of test losses
        '''
        total_loss = []
        for data_batch in self.data.get_train_batches():
            loss = self.algorithm.evaluate_one_iter(data_batch)
            total_loss.append(loss)

        return self._confidence_interval(total_loss)

    @staticmethod
    def _confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]: 
        '''caaulate confidence interval of losses'''

        dist = NormalDist.from_samples(data)
        z = NormalDist().inv_cdf((1 + confidence) / 2.)
        h = dist.stdev * z / ((len(data) - 1) ** .5)
        return dist.mean, h



    
