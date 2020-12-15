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

import numpy as np


class FeatureProcessor(object):
    def __init__(self, scalar_path):
        scalar = np.load(scalar_path)
        mean = scalar["mean"]
        std = scalar["std"]

        zero_std = np.where(std == 0.0)[0]

        mean = np.array(
            [mean[i] for i in range(len(mean)) if i not in zero_std])
        std = np.array([std[i] for i in range(len(std)) if i not in zero_std])

        self.mean = mean
        self.std = std
        self.zero_std = zero_std

    def process(self, raw_obs):
        obs = raw_obs.to_dict()

        loads = []
        for key in obs['loads']:
            loads.append(obs['loads'][key])
        loads = np.concatenate(loads)

        prods = []
        for key in obs['prods']:
            prods.append(obs['prods'][key])
        prods = np.concatenate(prods)

        lines_or = []
        for key in obs['lines_or']:
            lines_or.append(obs['lines_or'][key])
        lines_or = np.concatenate(lines_or)

        lines_ex = []
        for key in obs['lines_ex']:
            lines_ex.append(obs['lines_ex'][key])
        lines_ex = np.concatenate(lines_ex)

        features = np.concatenate([loads, prods, lines_or, lines_ex])
        features = np.array([
            features[i] for i in range(len(features)) if i not in self.zero_std
        ])
        norm_features = (features - self.mean) / self.std

        rho = obs['rho'] - 1.0

        other_features = rho
        return np.concatenate([norm_features, other_features]).tolist()


class UnitaryFeatureProcessor(object):
    def __init__(self, scalar_path):
        scalar = np.load(scalar_path)
        mean = scalar["mean"]
        std = scalar["std"]

        zero_std = np.where(std == 0.0)[0]

        mean = np.array(
            [mean[i] for i in range(len(mean)) if i not in zero_std])
        std = np.array([std[i] for i in range(len(std)) if i not in zero_std])

        self.mean = mean
        self.std = std
        self.zero_std = zero_std

    def process(self, raw_obs):
        obs = raw_obs.to_dict()

        loads = []
        for key in ['q', 'v']:
            loads.append(obs['loads'][key])
        loads = np.concatenate(loads)

        prods = []
        for key in ['q', 'v']:
            prods.append(obs['prods'][key])
        prods = np.concatenate(prods)

        features = np.concatenate([loads, prods])
        features = np.array([
            features[i] for i in range(len(features)) if i not in self.zero_std
        ])
        norm_features = (features - self.mean) / self.std

        rho = obs['rho']
        time_info = np.array([raw_obs.month - 1, raw_obs.hour_of_day])

        return np.concatenate([norm_features, rho, time_info]).tolist()
