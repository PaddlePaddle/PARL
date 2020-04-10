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

#include "evo_kit/sampling_factory.h"

namespace evo_kit {


std::shared_ptr<SamplingMethod> create_sampling_method(const EvoKitConfig& config) {
    std::shared_ptr<SamplingMethod> sampling_method;
    bool cached = config.gaussian_sampling().cached();

    if (cached) {
        sampling_method = std::make_shared<CachedGaussianSampling>();
    } else {
        sampling_method = std::make_shared<GaussianSampling>();
    }

    bool success = sampling_method->load_config(config);

    if (success) {
        return sampling_method;
    } else {
        LOG(ERROR) << "[EvoKit] Fail to create sampling_method";
        return nullptr;
    }

}

}//namespace
