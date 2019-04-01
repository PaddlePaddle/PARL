#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['np_softmax', 'np_cross_entropy']


def np_softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def np_cross_entropy(probs, labels):
    if labels.shape[-1] == 1:
        # sparse label
        n_classes = probs.shape[-1]
        result_shape = list(labels.shape[:-1]) + [n_classes]
        labels = np.eye(n_classes)[labels.reshape(-1)]
        labels = labels.reshape(result_shape)

    return -np.sum(labels * np.log(probs), axis=-1, keepdims=True)
