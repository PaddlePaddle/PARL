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

#ifndef EVO_KIT_SAMPLING_FACTORY_H
#define EVO_KIT_SAMPLING_FACTORY_H

#include <algorithm>
#include <glog/logging.h>
#include <memory>
#include "evo_kit/cached_gaussian_sampling.h"
#include "evo_kit/evo_kit.pb.h"
#include "evo_kit/gaussian_sampling.h"
#include "evo_kit/sampling_method.h"

namespace evo_kit {
/* @brief: create an sampling_method according to the configuration"
 * @args:
 *    config: configuration for the EvoKit
 *
 */
std::shared_ptr<SamplingMethod> create_sampling_method(const EvoKitConfig& Config);

} // namespace

#endif
