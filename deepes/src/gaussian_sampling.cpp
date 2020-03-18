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

#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gaussian_sampling.h"
#include "utils.h"

namespace DeepES{

void GaussianSampling::load_config(const DeepESConfig& config) {
    _std = config.gaussian_sampling().std();
    set_seed(config.seed());
}

int GaussianSampling::sampling(float* noise, int size) {
    int key = rand();
    std::default_random_engine generator(key);
    std::normal_distribution<float> norm;
    for (int i = 0; i < size; ++i) {
        *(noise + i) = norm(generator) * _std;
    }
    return key;
}

bool GaussianSampling::resampling(int key, float* noise, int size) {
    bool success = true;
    if (noise == nullptr) {
        success = false;
    }
    else {
        std::default_random_engine generator(key);
        std::normal_distribution<float> norm;
        for (int i = 0; i < size; ++i) {
            *(noise + i) = norm(generator) * _std;
        }
    }
    return success;
}

}
