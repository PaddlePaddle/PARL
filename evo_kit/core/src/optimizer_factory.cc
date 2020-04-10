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

#include "evo_kit/optimizer_factory.h"

namespace evo_kit {

std::shared_ptr<Optimizer> create_optimizer(const OptimizerConfig& optimizer_config) {
    std::shared_ptr<Optimizer> optimizer;
    std::string opt_type = optimizer_config.type();
    std::transform(opt_type.begin(), opt_type.end(), opt_type.begin(), ::tolower);

    if (opt_type == "sgd") {
        optimizer = std::make_shared<SGDOptimizer>(optimizer_config.base_lr(), \
                    optimizer_config.momentum());
    } else if (opt_type == "adam") {
        optimizer = std::make_shared<AdamOptimizer>(optimizer_config.base_lr(), \
                    optimizer_config.beta1(), \
                    optimizer_config.beta2(), \
                    optimizer_config.epsilon());
    } else {
        LOG(ERROR) << "type of OptimizerConfig must be SGD or Adam."; // NotImplementedError
    }

    return optimizer;
}

}//namespace
