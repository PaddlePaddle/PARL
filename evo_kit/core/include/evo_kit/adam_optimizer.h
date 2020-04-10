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

#ifndef EVO_KIT_ADAM_OPTIMIZER_H
#define EVO_KIT_ADAM_OPTIMIZER_H

#include <cmath>
#include <unordered_map>
#include "evo_kit/optimizer.h"

namespace evo_kit {

/*@brief AdamOptimizer.
  * Implements Adam algorithm.
  *
  *@Args:
  *     base_lr: learning rate (default: 1e-3).
  *     beta1: coefficients used for computing running averages of gradient (default: 0.9).
  *     beta2: coefficients used for computing running averages of gradient's square (default: 0.999).
  *     epsilon: term added to the denominator to improve numerical stability (default: 1e-8).
  */
class AdamOptimizer: public Optimizer {
public:
    AdamOptimizer(float base_lr, float beta1 = 0.9, float beta2 = 0.999,
                  float epsilon = 1e-8): Optimizer(base_lr), \
        _beta1(beta1), _beta2(beta2), _epsilon(epsilon) {}
    ~AdamOptimizer();

protected:
    void compute_step(float* gradient, int size, std::string param_name);

private:
    float _beta1;
    float _beta2;
    float _epsilon;
    std::unordered_map<std::string, float*> _momentum;
    std::unordered_map<std::string, float*> _velocity;
};

}//namespace

#endif
