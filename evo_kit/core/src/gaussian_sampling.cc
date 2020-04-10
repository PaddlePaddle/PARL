//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "evo_kit/gaussian_sampling.h"

namespace evo_kit {

bool GaussianSampling::load_config(const EvoKitConfig& config) {
    bool success = true;
    _std = config.gaussian_sampling().std();
    success = set_seed(config.seed());
    return success;
}

bool GaussianSampling::sampling(int* key, float* noise, int64_t size) {
    bool success = true;

    if (noise == nullptr) {
        LOG(ERROR) << "[EvoKit] Input noise array cannot be nullptr.";
        success = false;
        return success;
    }

    int rand_key = rand();
    *key = rand_key;
    std::default_random_engine generator(rand_key);
    std::normal_distribution<float> norm;

    for (int64_t i = 0; i < size; ++i) {
        *(noise + i) = norm(generator) * _std;
    }

    return success;
}

bool GaussianSampling::resampling(int key, float* noise, int64_t size) {
    bool success = true;

    if (noise == nullptr) {
        LOG(ERROR) << "[EvoKit] Input noise array cannot be nullptr.";
        success = false;
    } else {
        std::default_random_engine generator(key);
        std::normal_distribution<float> norm;

        for (int64_t i = 0; i < size; ++i) {
            *(noise + i) = norm(generator) * _std;
        }
    }

    return success;
}

}
