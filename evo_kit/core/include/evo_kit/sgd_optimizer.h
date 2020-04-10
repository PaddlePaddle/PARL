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

#ifndef EVO_KIT_SGD_OPTIMIZER_H
#define EVO_KIT_SGD_OPTIMIZER_H

#include <cmath>
#include <unordered_map>
#include "evo_kit/optimizer.h"

namespace evo_kit {

/*@brief SGDOptimizer.
  * Implements stochastic gradient descent (optionally with momentum).
  *
  *@Args:
  *     base_lr: learning rate (default: 1e-3).
  *     momentum: momentum factor (default: 0.9).
  */
class SGDOptimizer: public Optimizer {
public:
    SGDOptimizer(float base_lr, float momentum = 0.9): Optimizer(base_lr), _momentum(momentum) {}
    ~SGDOptimizer();

protected:
    void compute_step(float* gradient, int size, std::string param_name);

private:
    float _momentum;
    std::unordered_map<std::string, float*> _velocity;
};

} // namespace

#endif
