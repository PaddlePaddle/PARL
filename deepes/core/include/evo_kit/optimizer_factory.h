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

#ifndef EVO_KIT_OPTIMIZER_FACTORY_H
#define EVO_KIT_OPTIMIZER_FACTORY_H

#include <algorithm>
#include <glog/logging.h>
#include <memory>
#include "evo_kit/adam_optimizer.h"
#include "evo_kit/evo_kit.pb.h"
#include "evo_kit/optimizer.h"
#include "evo_kit/sgd_optimizer.h"

namespace evo_kit {
/* @brief: create an optimizer according to the configuration"
 * @args:
 *    config: configuration for the optimizer
 *
 */
std::shared_ptr<Optimizer> create_optimizer(const OptimizerConfig& optimizer_config);

} // namespace

#endif
