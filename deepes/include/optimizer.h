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
    }
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
  ~SGDOptimizer() {
    for (std::map<std::string, float*>::iterator iter = _velocity.begin(); iter != _velocity.end(); iter++) {
      delete[] iter->second;
    }
    _velocity.clear();
  }

protected:
  void compute_step(float* gradient, int size, std::string param_name="") {
    if (_velocity.count(param_name) == 0) {
      _velocity[param_name] = new float [size];
      memset(_velocity[param_name], 0, size * sizeof(float));
    }
    for (int i = 0; i < size; ++i) {
      _velocity[param_name][i] = _momentum * _velocity[param_name][i] + (1 - _momentum) * gradient[i];
      gradient[i] = _velocity[param_name][i];
    }
  }

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
  ~AdamOptimizer() {
    for (std::map<std::string, float*>::iterator iter = _momentum.begin(); iter != _momentum.end(); iter++) {
      delete[] iter->second;
    }
    for (std::map<std::string, float*>::iterator iter = _velocity.begin(); iter != _velocity.end(); iter++) {
      delete[] iter->second;
    }
    _momentum.clear();
    _velocity.clear();
  }
protected:
  void compute_step(float* gradient, int size, std::string param_name="") {
    if (_momentum.count(param_name) == 0) {
      _momentum[param_name] = new float [size];
      memset(_momentum[param_name], 0, size * sizeof(float));
    }
    if (_velocity.count(param_name) == 0) {
      _velocity[param_name] = new float [size];
      memset(_velocity[param_name], 0, size * sizeof(float));
    }
    int true_update_times = int(_update_times / _velocity.size());
    float alpha = sqrt(1 - pow(_beta2, _update_times)) / (1 - pow(_beta1, _update_times));
    for (int i = 0; i < size; ++i) {
      _momentum[param_name][i] = _beta1 * _momentum[param_name][i] + (1 - _beta1) * gradient[i];
      _velocity[param_name][i] = _beta2 * _velocity[param_name][i] + (1 - _beta2) * gradient[i] * gradient[i];
      gradient[i] = alpha * _momentum[param_name][i] / (sqrt(_velocity[param_name][i]) + _epsilon);
    }
  }

private:
  float _beta1;
  float _beta2;
  float _epsilon;
  std::map<std::string, float*> _momentum;
  std::map<std::string, float*> _velocity;
};

}//namespace
#endif
