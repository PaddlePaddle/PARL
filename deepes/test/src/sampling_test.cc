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

#include "gtest/gtest.h"
#include <vector>
#include "evo_kit/sampling_method.h"
#include "evo_kit/gaussian_sampling.h"
#include "evo_kit/cached_gaussian_sampling.h"
#include <memory>

namespace evo_kit {

class SamplingTest : public ::testing::Test {
 protected:
  void init_sampling_method(bool cached) {
    config = std::make_shared<EvoKitConfig>();
    config->set_seed(1024);
    auto sampling_config = config->mutable_gaussian_sampling();
    sampling_config->set_std(1.0);
    sampling_config->set_cached(cached);
    sampling_config->set_cache_size(cache_size);
    if (cached) {
      sampler = std::make_shared<CachedGaussianSampling>();
    } else {
      sampler = std::make_shared<GaussianSampling>();
    }
  }

  std::shared_ptr<SamplingMethod> sampler;
  std::shared_ptr<EvoKitConfig> config;
  float array[3] = {1.0, 2.0, 3.0};
  int cache_size = 100;   // default cache_size 100
  int key = 0;
};


TEST_F(SamplingTest, GaussianSampling_load_config) {
  init_sampling_method(false);
  EXPECT_TRUE(sampler->load_config(*config));
}

TEST_F(SamplingTest, GaussianSampling_sampling) {
  init_sampling_method(false);
  sampler->load_config(*config);

  EXPECT_FALSE(sampler->sampling(&key, nullptr, 0));
  EXPECT_TRUE(sampler->sampling(&key, array, 3));
}

TEST_F(SamplingTest, GaussianSampling_resampling) {
  init_sampling_method(false);
  sampler->load_config(*config);

  EXPECT_FALSE(sampler->resampling(0, nullptr, 0));
  EXPECT_TRUE(sampler->resampling(0, array, 3));
}


TEST_F(SamplingTest, CachedGaussianSampling_load_config) {
  init_sampling_method(true);
  EXPECT_TRUE(sampler->load_config(*config));
}

TEST_F(SamplingTest, CachedGaussianSampling_sampling) {
  init_sampling_method(true);
  EXPECT_FALSE(sampler->sampling(&key, array, 0));

  sampler->load_config(*config);

  EXPECT_FALSE(sampler->sampling(&key, nullptr, 0));
  EXPECT_FALSE(sampler->sampling(&key, array, -1));
  EXPECT_FALSE(sampler->sampling(&key, array, cache_size));

  EXPECT_TRUE(sampler->sampling(&key, array, 0));
  EXPECT_TRUE(sampler->sampling(&key, array, 3));
}

TEST_F(SamplingTest, CachedGaussianSampling_resampling) {
  init_sampling_method(true);
  EXPECT_FALSE(sampler->resampling(0, array, 0));

  sampler->load_config(*config);

  EXPECT_FALSE(sampler->resampling(0, nullptr, 0));
  EXPECT_FALSE(sampler->resampling(0, array, -1));
  EXPECT_FALSE(sampler->resampling(0, array, cache_size));

  EXPECT_TRUE(sampler->resampling(0, array, 0));
  EXPECT_TRUE(sampler->resampling(0, array, 1));
  EXPECT_TRUE(sampler->resampling(0, array, 2));

  EXPECT_FALSE(sampler->resampling(-1, array, 3));
  EXPECT_TRUE(sampler->resampling(0, array, 3));
  EXPECT_TRUE(sampler->resampling(1, array, 3));
  EXPECT_TRUE(sampler->resampling(2, array, 3));
  EXPECT_TRUE(sampler->resampling(cache_size-3, array, 3));
  EXPECT_FALSE(sampler->resampling(cache_size-2, array, 3));
  EXPECT_FALSE(sampler->resampling(cache_size-1, array, 3));
  EXPECT_FALSE(sampler->resampling(cache_size, array, 3));
  EXPECT_FALSE(sampler->resampling(cache_size-3, array, cache_size-1));
}


} // namespace

