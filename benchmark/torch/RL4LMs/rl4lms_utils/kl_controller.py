#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class KLController:
    def __init__(self, kl_coeff, target_kl=None):
        self._kl_coeff = kl_coeff
        self._target_kl = target_kl

    def step(self, kl_div: torch.tensor):
        """
        Adapts the KL coeff
        """
        if self._target_kl is not None:
            diff_to_target = (kl_div - self._target_kl) / self._target_kl
            e_t = torch.clip(diff_to_target, -0.2, 0.2).item()
            self._kl_coeff = self._kl_coeff * (1 + 0.1 * e_t)

    @property
    def kl_coeff(self):
        return self._kl_coeff
