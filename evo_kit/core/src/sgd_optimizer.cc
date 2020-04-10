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

#include "evo_kit/sgd_optimizer.h"

namespace evo_kit {

SGDOptimizer::~SGDOptimizer() {
    for (auto iter = _velocity.begin(); iter != _velocity.end(); iter++) {
        delete[] iter->second;
    }

    _velocity.clear();
}

void SGDOptimizer::compute_step(float* gradient, int size, std::string param_name = "") {
    if (_velocity.count(param_name) == 0) {
        _velocity[param_name] = new float [size];
        memset(_velocity[param_name], 0, size * sizeof(float));
    }

    for (int i = 0; i < size; ++i) {
        _velocity[param_name][i] = _momentum * _velocity[param_name][i] + (1 - _momentum) * gradient[i];
        gradient[i] = _velocity[param_name][i];
    }
}


}//namespace
