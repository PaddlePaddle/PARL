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

#ifndef TORCH_ESAGENT_H
#define TORCH_ESAGENT_H

#include <memory>
#include <string>
#include "evo_kit/optimizer_factory.h"
#include "evo_kit/sampling_factory.h"
#include "evo_kit/utils.h"
#include "evo_kit/evo_kit.pb.h"

namespace evo_kit{

/**
 * @brief DeepES agent for Torch.
 *
 * Our implemtation is flexible to support any model that subclass torch::nn::Module.
 * That is, we can instantiate an agent by: es_agent = ESAgent<Model>(model);
 * After that, users can clone an agent for multi-thread processing, add parametric noise for exploration,
 * and update the parameteres, according to the evaluation resutls of noisy parameters.
 */
template <class T>
class ESAgent{
public:
  ESAgent() {}

  ~ESAgent() {
    delete[] _noise;
    if (!_is_sampling_agent)
      delete[] _neg_gradients;
  }

  ESAgent(std::shared_ptr<T> model, std::string config_path): _model(model) {
    _is_sampling_agent = false;
    _config = std::make_shared<EvoKitConfig>();
    load_proto_conf(config_path, *_config);
    _sampling_method = create_sampling_method(*_config);
    _optimizer = create_optimizer(_config->optimizer());
    // Origin agent can't be used to sample, so keep it same with _model for evaluating.
    _sampling_model = model;
    _param_size = _calculate_param_size();

    _noise = new float [_param_size];
    _neg_gradients = new float [_param_size];
  }

  /** 
   * @breif Clone a sampling agent
   *
   * Only cloned ESAgent can call `add_noise` function.
   * Each cloned ESAgent will have a copy of original parameters.
   * (support sampling in multi-thread way)
   */
  std::shared_ptr<ESAgent> clone() {
    std::shared_ptr<ESAgent> new_agent = std::make_shared<ESAgent>();

    new_agent->_model = _model;
    std::shared_ptr<T> new_model = _model->clone();
    new_agent->_sampling_model = new_model;
  
    new_agent->_is_sampling_agent = true;
    new_agent->_sampling_method = _sampling_method;
    new_agent->_param_size = _param_size;

    float* new_noise = new float [_param_size];
    new_agent->_noise = new_noise;

    return new_agent;
  }

  /**
   * @brief Use the model to predict. 
   *
   * if _is_sampling_agent is true, will use the sampling model with added noise;
   * if _is_sampling_agent is false, will use the original model without added noise.
   */
  torch::Tensor predict(const torch::Tensor& x) {
    return _sampling_model->forward(x);
  }

  /**
   * @brief Update parameters of model based on ES algorithm.
   *
   * Only not cloned ESAgent can call `update` function.
   * Parameters of cloned agents will also be updated.
   */
  bool update(std::vector<SamplingInfo>& noisy_info, std::vector<float>& noisy_rewards) {
    if (_is_sampling_agent) {
      LOG(ERROR) << "[DeepES] Cloned ESAgent cannot call update function, please use original ESAgent.";
      return false;
    }

    compute_centered_ranks(noisy_rewards);

    memset(_neg_gradients, 0, _param_size * sizeof(float));
    for (int i = 0; i < noisy_info.size(); ++i) {
      int key = noisy_info[i].key(0);
      float reward = noisy_rewards[i];
      bool success = _sampling_method->resampling(key, _noise, _param_size);
      CHECK(success) << "[DeepES] resampling error occurs at sample: " << i;
      for (int64_t j = 0; j < _param_size; ++j) {
        _neg_gradients[j] += _noise[j] * reward;
      }
    }
    for (int64_t j = 0; j < _param_size; ++j) {
      _neg_gradients[j] /= -1.0 * noisy_info.size();
    }

    //update
    auto params = _model->named_parameters();
    int64_t counter = 0;
    for (auto& param: params) {
      torch::Tensor tensor = param.value().view({-1});
      auto tensor_a = tensor.accessor<float,1>();
      _optimizer->update(tensor_a, _neg_gradients+counter, tensor.size(0), param.key());
      counter += tensor.size(0);
    }

    return true;
  }

  // copied parameters = original parameters + noise
  bool add_noise(SamplingInfo& sampling_info) {
    bool success = true;
    if (!_is_sampling_agent) {
      LOG(ERROR) << "[DeepES] Original ESAgent cannot call add_noise function, please use cloned ESAgent.";
      success =  false;
      return success;
    }

    auto sampling_params = _sampling_model->named_parameters();
    auto params = _model->named_parameters();
    int key = 0;
    success = _sampling_method->sampling(&key, _noise, _param_size);
    CHECK(success) << "[EvoKit] sampling error occurs while add_noise.";
    sampling_info.add_key(key);
    int64_t counter = 0;
    for (auto& param: sampling_params) {
      torch::Tensor sampling_tensor = param.value().view({-1});
      std::string param_name = param.key();
      torch::Tensor tensor = params.find(param_name)->view({-1});
      auto sampling_tensor_a = sampling_tensor.accessor<float,1>();
      auto tensor_a = tensor.accessor<float,1>();
      for (int64_t j = 0; j < tensor.size(0); ++j) {
        sampling_tensor_a[j] = tensor_a[j] + _noise[counter + j];
      }
      counter += tensor.size(0);
    }
    return success;
  }

  // get param size of model
  int64_t param_size() {
    return _param_size;
  }


private:
  int64_t _calculate_param_size() {
    _param_size = 0;
    auto params = _model->named_parameters();
    for (auto& param: params) {
      torch::Tensor tensor = param.value().view({-1});
      _param_size += tensor.size(0);
    }
    return _param_size;
  }

  std::shared_ptr<T> _model;
  std::shared_ptr<T> _sampling_model;
  bool _is_sampling_agent;
  std::shared_ptr<SamplingMethod> _sampling_method;
  std::shared_ptr<Optimizer> _optimizer;
  std::shared_ptr<EvoKitConfig> _config;
  int64_t _param_size;
  // malloc memory of noise and neg_gradients in advance.
  float* _noise;
  float* _neg_gradients;
};

}

#endif /* TORCH_ESAGENT_H */
