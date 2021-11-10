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

import torch
import torch.nn as nn
import torch.nn.functional as F
import parl


class MAMLModel(parl.Model):
    def __init__(self, network_dims, device):
        """
        Builds a multi-layer perceptron. It also provides functionality for passing external parameters to be
        used at inference time.
        
        Args:
            network_dims: List, define the dimension of each linear layer.
            device: Cpu or cuda.
        """
        super().__init__()

        self.network_dims = network_dims
        self.num_layers = len(network_dims) - 1

        self.params = list()

        for i in range(len(self.network_dims) - 1):
            weight = nn.Parameter(
                torch.zeros((self.network_dims[i + 1], self.network_dims[i]),
                            device=device))
            torch.nn.init.kaiming_normal_(weight)
            self.params.append(weight)
            self.params.append(
                nn.Parameter(
                    torch.zeros((self.network_dims[i + 1]), device=device)))

    def forward(self, x, params=None):
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

            # pass to an activation function if it's not the last layer
            if i != len(self.network_dims) - 2:
                x = F.relu(x)

        return x

    def zero_grad(self, params=None):
        if params is None:
            for param in self.params:
                if param.requires_grad and param.grad is not None:
                    param.grad.zero_()
        else:
            for param in params:
                if param.requires_grad and param.grad is not None:
                    param.grad.zero_()

    def get_weights(self):

        return self.params

    def set_weights(self, new_weights):
        self.params.clear()

        for weight in new_weights:
            self.params.append(weight.detach().clone())
