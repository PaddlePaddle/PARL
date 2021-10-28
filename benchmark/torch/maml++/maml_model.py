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

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import parl
from config import Config


class MAMLModel(parl.Model):
    def __init__(self, config: Config, device: torch.device):
        """
        Builds a multi-layer perceptron. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param input_dim: Input shape.
        :param hidden_dim: Hidden dimension of each hidden layer. 
        :param output_dim: Output shape.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super().__init__()
        self.device = device

        self.network_dims = config.network_dims
        self.num_layers = config.num_layers

        self.params = list()

        for i in range(len(self.network_dims) - 1):
            weight = nn.Parameter(
                torch.zeros((self.network_dims[i + 1], self.network_dims[i])))
            torch.nn.init.kaiming_normal_(weight)
            self.params.append(weight)
            self.params.append(
                nn.Parameter(torch.zeros((self.network_dims[i + 1]))))

    def forward(self, x: torch.Tensor, params: List = None) -> torch.Tensor:
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.

        Args:
            x: Input
            params: If params are None then internal parameters are used. If params are a dictionary with keys the
            same as the layer names then they will be used instead.

        """

        if params is None:
            params = self.params

        assert len(params) // 2 == self.num_layers

        for i in range(self.num_layers):
            x = F.linear(x, params[i * 2], params[i * 2 + 1])
            if i != len(self.network_dims) - 2:
                x = F.relu(x)

        return x

    def zero_grad(self, params=None):
        if params is None:
            for param in self.params:
                if param.requires_grad:
                    if param.grad is not None:
                        param.grad.zero_()
        else:
            for param in params:
                if param.requires_grad:
                    if param.grad is not None:
                        param.grad.zero_()

    def get_weights(self) -> torch.Tensor:

        return self.params

    def set_weights(self, new_weights: List):
        self.params.clear()

        for weight in new_weights:
            self.params.append(weight.detach().clone())
