#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import parl

__all__ = ['ES']


class ES(parl.Algorithm):
    def __init__(self, model):
        """ES algorithm.
        
        Since parameters of the model is updated in the numpy level, `learn` function is not needed
        in this algorithm.

        Args:
            model (`parl.Model`): policy model of ES algorithm.
        """
        self.model = model

    def predict(self, obs):
        return self.model.predict(obs)


class EnsembleES(parl.Algorithm):
    def __init__(self, model1, model2):
        """ES algorithm.
        
        Since parameters of the model is updated in the numpy level, `learn` function is not needed
        in this algorithm.

        Args:
            model1(`parl.Model`): policy model of ES algorithm.
            model2(`parl.Model`): policy model of ES algorithm.
        """
        self.model1 = model1
        self.model2 = model2

    def predict(self, obs):
        return (self.model1.predict(obs) + self.model2.predict(obs)) / 2.0
