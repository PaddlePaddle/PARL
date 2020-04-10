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

#include "evo_kit/cached_gaussian_sampling.h"

namespace evo_kit {

CachedGaussianSampling::CachedGaussianSampling() {}

CachedGaussianSampling::~CachedGaussianSampling() {
    delete[] _noise_cache;
}

bool CachedGaussianSampling::load_config(const EvoKitConfig& config) {
    bool success = true;
    _std = config.gaussian_sampling().std();
    success = set_seed(config.seed());
    CHECK(success) << "[EvoKit] Fail to set seed while load config.";
    _cache_size = config.gaussian_sampling().cache_size();
    _noise_cache = new float [_cache_size];
    memset(_noise_cache, 0, _cache_size * sizeof(float));
    success = _create_noise_cache();
    CHECK(success) << "[EvoKit] Fail to create noise_cache while load config.";
    return success;
}

bool CachedGaussianSampling::sampling(int* key, float* noise, int64_t size) {
    bool success = true;

    if (_noise_cache == nullptr) {
        LOG(ERROR) << "[EvoKit] Please use load_config() first.";
        success = false;
        return success;
    }

    if (noise == nullptr) {
        LOG(ERROR) << "[EvoKit] Input noise array cannot be nullptr.";
        success = false;
        return success;
    }

    if ((size >= _cache_size) || (size < 0)) {
        LOG(ERROR) << "[EvoKit] Input size " << size << " is out of bounds [0, " << _cache_size <<
                   "), cache_size: " << _cache_size;
        success = false;
        return success;
    }

    int rand_key = rand();
    std::default_random_engine generator(rand_key);
    std::uniform_int_distribution<unsigned int> uniform(0, _cache_size - size);
    int index = uniform(generator);
    *key = index;

    for (int64_t i = 0; i < size; ++i) {
        *(noise + i) = *(_noise_cache + index + i);
    }

    return success;
}

bool CachedGaussianSampling::resampling(int key, float* noise, int64_t size) {
    bool success = true;

    if (_noise_cache == nullptr) {
        LOG(ERROR) << "[EvoKit] Please use load_config() first.";
        success = false;
        return success;
    }

    if (noise == nullptr) {
        LOG(ERROR) << "[EvoKit] Input noise array cannot be nullptr.";
        success = false;
        return success;
    }

    if ((size >= _cache_size) || (size < 0)) {
        LOG(ERROR) << "[EvoKit] Input size " << size << " is out of bounds [0, " << _cache_size <<
                   "), cache_size: " << _cache_size;
        success = false;
        return success;
    }

    if ((key > _cache_size - size) || (key < 0)) {
        LOG(ERROR) << "[EvoKit] Resampling key " << key << " is out of bounds [0, "
                    << _cache_size - size <<
                   "], cache_size: " << _cache_size << ", size: " << size;
        success = false;
        return success;
    }

    for (int64_t i = 0; i < size; ++i) {
        *(noise + i) = *(_noise_cache + key + i);
    }

    return success;
}

bool CachedGaussianSampling::_create_noise_cache() {
    std::default_random_engine generator(_seed);
    std::normal_distribution<float> norm;

    for (int64_t i = 0; i < _cache_size; ++i) {
        *(_noise_cache + i) = norm(generator) * _std;
    }

    return true;
}

}
