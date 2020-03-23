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

#include <vector>
#include <iostream>
#include "es_agent.h"
#include "paddle_api.h"
#include "optimizer.h"
#include "utils.h"
#include "gaussian_sampling.h"
#include "deepes.pb.h"


namespace DeepES {

typedef paddle::lite_api::PaddlePredictor PaddlePredictor;
typedef paddle::lite_api::Tensor Tensor;
typedef paddle::lite_api::shape_t shape_t;

inline int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

ESAgent::ESAgent() {}

ESAgent::~ESAgent() {
  delete[] _noise;
  if (!_is_sampling_agent)
    delete[] _neg_gradients;
}

ESAgent::ESAgent(
    std::shared_ptr<PaddlePredictor> predictor,
    std::string config_path) {

  _is_sampling_agent = false;
  _predictor = predictor;
  // Original agent can't be used to sample, so keep it same with _predictor for evaluating.
  _sample_predictor = predictor;

  _config = std::make_shared<DeepESConfig>();
  load_proto_conf(config_path, *_config);

  _sampling_method = std::make_shared<GaussianSampling>();
  _sampling_method->load_config(*_config);

  _optimizer = std::make_shared<SGDOptimizer>(_config->optimizer().base_lr());

  _param_names = _predictor->GetParamNames();
  _param_size = _calculate_param_size();

  _noise = new float [_param_size];
  _neg_gradients = new float [_param_size];
}

std::shared_ptr<ESAgent> ESAgent::clone() {
  std::shared_ptr<PaddlePredictor> new_sample_predictor = _predictor->Clone();

  std::shared_ptr<ESAgent> new_agent = std::make_shared<ESAgent>();

  float* new_noise = new float [_param_size];

  new_agent->_predictor = _predictor;
  new_agent->_sample_predictor = new_sample_predictor;

  new_agent->_is_sampling_agent = true;
  new_agent->_sampling_method = _sampling_method;
  new_agent->_param_names = _param_names;
  new_agent->_param_size = _param_size;
  new_agent->_noise = new_noise;

  return new_agent;
}

bool ESAgent::update(
    std::vector<SamplingKey>& noisy_keys,
    std::vector<float>& noisy_rewards) {
  if (_is_sampling_agent) {
    LOG(ERROR) << "[DeepES] Cloned ESAgent cannot call update function, please use original ESAgent.";
    return false;
  }

  compute_centered_ranks(noisy_rewards);
  
  memset(_neg_gradients, 0, _param_size * sizeof(float));
  for (int i = 0; i < noisy_keys.size(); ++i) {
    int key = noisy_keys[i].key(0);
    float reward = noisy_rewards[i];
    bool success = _sampling_method->resampling(key, _noise, _param_size);
    for (int64_t j = 0; j < _param_size; ++j) {
      _neg_gradients[j] += _noise[j] * reward;
    }
  }
  for (int64_t j = 0; j < _param_size; ++j) {
    _neg_gradients[j] /= -1.0 * noisy_keys.size();
  }

  //update
  int64_t counter = 0;

  for (std::string param_name: _param_names) {
    std::unique_ptr<Tensor> tensor = _predictor->GetMutableTensor(param_name);
    float* tensor_data = tensor->mutable_data<float>();
    int64_t tensor_size = ShapeProduction(tensor->shape());
    _optimizer->update(tensor_data, _neg_gradients + counter, tensor_size);
    counter += tensor_size;
  }
  return true;
  
}

bool ESAgent::add_noise(SamplingKey& sampling_key) {
  if (!_is_sampling_agent) {
    LOG(ERROR) << "[DeepES] Original ESAgent cannot call add_noise function, please use cloned ESAgent.";
    return false;
  }

  int key = _sampling_method->sampling(_noise, _param_size);
  sampling_key.add_key(key);
  int64_t counter = 0;

  for (std::string param_name: _param_names) {
    std::unique_ptr<Tensor> sample_tensor = _sample_predictor->GetMutableTensor(param_name);
    std::unique_ptr<const Tensor> tensor = _predictor->GetTensor(param_name);
    int64_t tensor_size = ShapeProduction(tensor->shape());
    for (int64_t j = 0; j < tensor_size; ++j) {
      sample_tensor->mutable_data<float>()[j] = tensor->data<float>()[j] + _noise[counter + j];
    }
    counter += tensor_size;
  }

  return true;
}


std::shared_ptr<PaddlePredictor> ESAgent::get_predictor() {
  return _sample_predictor;
}

int64_t ESAgent::_calculate_param_size() {
  int64_t param_size = 0;
  for (std::string param_name: _param_names) {
    std::unique_ptr<const Tensor> tensor = _predictor->GetTensor(param_name);
    param_size += ShapeProduction(tensor->shape());
  }
  return param_size;
}


}

