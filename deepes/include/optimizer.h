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

#include <map>
#include <glog/logging.h>

#ifndef OPTIMIZER_H
#define OPTIMIZER_H
namespace DeepES{

/*@brief Optimizer. Base class for optimizers. 
 * 
 *@Args:
 *     base_lr: learning rate (default: 1e-3).
 *     
 * .. warning: update () is based on the parameter level, 
 *             you need to perform update () on each parameter.
 * 
 * Subclasses are required to implement the following functions:
 * 1. compute_steps
 */
class Optimizer {
public:
  Optimizer() : _base_lr(1e-3), _update_times(0) {}
  Optimizer(float base_lr) : _base_lr(base_lr), _update_times(0) {}
  virtual ~Optimizer() {
    _params_size.clear();
  }

  template<typename T>
  bool update(T weights, float* gradient, int size, std::string param_name="") {
  /*@ Performs a single optimization step (parameter update) at the parameter level.
    *
    *@Args:
    *     weights (array): parameter weights.
    *     gradient (array): gradient for updating weights.
    *     size: size of gradient.
    *     param_name: the name corresponding to the weights.
    */
    if (_params_size.count(param_name) == 0) {
      _params_size[param_name] = size;
    } else if (_params_size[param_name] != size) {
      LOG(WARNING) << "[Warning] Update times: "<< int(_update_times / _params_size.size()) \
       << ". Size of weights[" << param_name << "] is " << _params_size[param_name] << ", not " << size;
      return false;
    }

    ++_update_times;
    compute_step(gradient, size, param_name);
    for (int i = 0; i < size; ++i) {
      weights[i] -= _base_lr * gradient[i];
    }
    return true;
  } // template function

protected:
  virtual void compute_step(float* graident, int size, std::string param_name="") = 0;
  float _base_lr;
  float _update_times;
  std::map<std::string, int> _params_size;
};


/*@brief SGDOptimizer.
  * Implements stochastic gradient descent (optionally with momentum).
  *
  *@Args:
  *     base_lr: learning rate (default: 1e-3).
  *     momentum: momentum factor (default: 0.9).
  */
class SGDOptimizer: public Optimizer {
public:
  SGDOptimizer(float base_lr, float momentum=0.9):Optimizer(base_lr), _momentum(momentum) {}
  ~SGDOptimizer();

protected:
  void compute_step(float* gradient, int size, std::string param_name);

private:
  float _momentum;
  std::map<std::string, float*> _velocity;
};


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
  AdamOptimizer(float base_lr, float beta1=0.9, float beta2=0.999, float epsilon=1e-8):Optimizer(base_lr), \
                                    _beta1(beta1), _beta2(beta2), _epsilon(epsilon) {}
  ~AdamOptimizer();

protected:
  void compute_step(float* gradient, int size, std::string param_name);

private:
  float _beta1;
  float _beta2;
  float _epsilon;
  std::map<std::string, float*> _momentum;
  std::map<std::string, float*> _velocity;
};

}//namespace
#endif
