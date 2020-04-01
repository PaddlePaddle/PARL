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

#include "sampling_factory.h"

namespace DeepES{


std::shared_ptr<SamplingMethod> create_sampling_method(const DeepESConfig& config) {
  std::shared_ptr<SamplingMethod> sampling_method;
  std::string sample_type = config.gaussian_sampling().type();
  std::transform(sample_type.begin(), sample_type.end(), sample_type.begin(), ::tolower);
  if (sample_type == "normal") {
    sampling_method = std::make_shared<GaussianSampling>();
  }else if (sample_type == "table") {
    sampling_method = std::make_shared<GaussianTableSampling>();
  }else {
    LOG(ERROR) << "type of GaussianSamplingConfig must be normal or table."; // NotImplementedError
  }
  sampling_method->load_config(config);
  return sampling_method;
}

}//namespace
