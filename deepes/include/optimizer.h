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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H
namespace DeepES{

class Optimizer {
public:
  Optimizer() : _base_lr(1e-3), _update_times(0) {}
  Optimizer(float base_lr) : _base_lr(base_lr), _update_times(0) {}
  template<typename T>
  bool update(T weights, float* gradient, int size, std::string param_name="") {
    bool success = true;
    ++_update_times;
    compute_step(gradient, size, param_name);
    for (int i = 0; i < size; ++i) {
      weights[i] -= _base_lr * gradient[i];
    }
    return success;
  } // template function

protected:
  virtual void compute_step(float* graident, int size, std::string param_name="") = 0;
  float _base_lr;
  float _update_times;
};

class SGDOptimizer: public Optimizer {
public:
  SGDOptimizer(float base_lr, float momentum=0.0):Optimizer(base_lr), _momentum(momentum) {}

protected:
  void compute_step(float* gradient, int size, std::string param_name="") {
  }

private:
  float _momentum;

}; //namespace

//class AdamOptimizer: public Optimizer {
//public:
//  AdamOptimizer(float base)
//}

};
#endif
