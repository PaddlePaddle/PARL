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
#include "gaussian_table_sampling.h"
#include "utils.h"
#include <glog/logging.h>

namespace DeepES{

GaussianTableSampling::GaussianTableSampling() {}

GaussianTableSampling::~GaussianTableSampling() {
    delete[] _noise_table;
}

void GaussianTableSampling::load_config(const DeepESConfig& config) {
    _std = config.gaussian_sampling().std();
    set_seed(config.seed());
    _buffer_size = config.gaussian_sampling().noise_table_size();
    _noise_table = new float [_buffer_size];
	memset(_noise_table, 0, _buffer_size * sizeof(float));
    _create_noise();
}

int GaussianTableSampling::sampling(float* noise, int64_t size) {
    if (_noise_table == nullptr) {
        LOG(ERROR) << "[ERROR] Please use load_config() first.";
    }
    int key = rand();
    std::default_random_engine generator(key);
    std::uniform_int_distribution<unsigned int> uniform(0, _buffer_size - size);
    int index = uniform(generator);
    for (int64_t i = 0; i < size; ++i) {
        *(noise + i) = *(_noise_table + index + i);
    }
    return index;
}

bool GaussianTableSampling::resampling(int index, float* noise, int64_t size) {
    if (_noise_table == nullptr) {
        LOG(ERROR) << "[ERROR] Please use load_config() first.";
        return false;
    }
    if (noise == nullptr) {
        return false;
    }
    if ((index > _buffer_size - size) || (index < 0)) {
        LOG(ERROR) << "[ERROR] Sampling index out of bound( 0 - " << _buffer_size - size << " ).";
    }
    for (int64_t i = 0; i < size; ++i) {
        *(noise + i) = *(_noise_table + index + i);
    }
    return true;
}

void GaussianTableSampling::_create_noise() {
    int key = rand();
    std::default_random_engine generator(key);
    std::normal_distribution<float> norm;
    for (int64_t i = 0; i < _buffer_size; ++i) {
        *(_noise_table + i) = norm(generator) * _std;
    }
}

}
