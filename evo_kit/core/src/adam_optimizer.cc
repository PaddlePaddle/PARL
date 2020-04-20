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

#include "evo_kit/adam_optimizer.h"

namespace evo_kit {

AdamOptimizer::~AdamOptimizer() {
    for (auto iter = _momentum.begin(); iter != _momentum.end(); iter++) {
        delete[] iter->second;
    }

    for (auto iter = _velocity.begin(); iter != _velocity.end(); iter++) {
        delete[] iter->second;
    }

    _momentum.clear();
    _velocity.clear();
}

void AdamOptimizer::compute_step(float* gradient, int size, std::string param_name = "") {
    if (_momentum.count(param_name) == 0) {
        _momentum[param_name] = new float [size];
        memset(_momentum[param_name], 0, size * sizeof(float));
    }

    if (_velocity.count(param_name) == 0) {
        _velocity[param_name] = new float [size];
        memset(_velocity[param_name], 0, size * sizeof(float));
    }

    int true_update_times = int(_update_times / _velocity.size());
    float alpha = std::sqrt(1 - std::pow(_beta2, _update_times)) / (1 - std::pow(_beta1,
                  _update_times));

    for (int i = 0; i < size; ++i) {
        _momentum[param_name][i] = _beta1 * _momentum[param_name][i] + (1 - _beta1) * gradient[i];
        _velocity[param_name][i] = _beta2 * _velocity[param_name][i] + (1 - _beta2) * gradient[i] *
                                   gradient[i];
        gradient[i] = alpha * _momentum[param_name][i] / (std::sqrt(_velocity[param_name][i]) + _epsilon);
    }
}

}//namespace
