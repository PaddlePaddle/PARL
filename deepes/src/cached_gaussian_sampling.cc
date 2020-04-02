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

#include "cached_gaussian_sampling.h"

namespace DeepES{

CachedGaussianSampling::CachedGaussianSampling() {}

CachedGaussianSampling::~CachedGaussianSampling() {
    delete[] _noise_cache;
}

void CachedGaussianSampling::load_config(const DeepESConfig& config) {
    _std = config.gaussian_sampling().std();
    set_seed(config.seed());
    _cache_size = config.gaussian_sampling().cache_size();
    _noise_cache = new float [_cache_size];
	memset(_noise_cache, 0, _cache_size * sizeof(float));
    _create_noise_cache();
}

int CachedGaussianSampling::sampling(float* noise, int64_t size) {
    if (_noise_cache == nullptr) {
        LOG(ERROR) << "[ERROR] Please use load_config() first.";
    }
    int key = rand();
    std::default_random_engine generator(key);
    std::uniform_int_distribution<unsigned int> uniform(0, _cache_size - size);
    int index = uniform(generator);
    for (int64_t i = 0; i < size; ++i) {
        *(noise + i) = *(_noise_cache + index + i);
    }
    return index;
}

bool CachedGaussianSampling::resampling(int index, float* noise, int64_t size) {
    if (_noise_cache == nullptr) {
        LOG(ERROR) << "[ERROR] Please use load_config() first.";
        return false;
    }
    if (noise == nullptr) {
        return false;
    }
    if ((index > _cache_size - size) || (index < 0)) {
        LOG(ERROR) << "[ERROR] Sampling index out of bound( 0 - " << _cache_size - size << " ).";
    }
    for (int64_t i = 0; i < size; ++i) {
        *(noise + i) = *(_noise_cache + index + i);
    }
    return true;
}

void CachedGaussianSampling::_create_noise_cache() {
    std::default_random_engine generator(_seed);
    std::normal_distribution<float> norm;
    for (int64_t i = 0; i < _cache_size; ++i) {
        *(_noise_cache + i) = norm(generator) * _std;
    }
}

}
