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

#ifndef MODEL_H
#define MODEL_H
#include <memory>
#include <string>
#include "optimizer.h"
#include "utils.h"
#include "gaussian_sampling.h"
#include "deepes.pb.h"

namespace DeepES{

template <class T>
class Predictor{
public:
  Predictor(): _param_size(0){}

  Predictor(std::shared_ptr<T> model, std::string config_path): _model(model) {
    _config = std::make_shared<DeepESConfig>();
    load_proto_conf(config_path, *_config);
    _sampling_method = std::make_shared<GaussianSampling>();
    _sampling_method->load_config(*_config);
    _optimizer = std::make_shared<SGDOptimizer>(_config->optimizer().base_lr());
    _param_size = 0;
    _sampled_model = model;
    param_size();
  }

  std::shared_ptr<Predictor> clone() {
    std::shared_ptr<T> new_model = _model->clone();
    std::shared_ptr<Predictor> new_predictor = std::make_shared<Predictor>();
    new_predictor->set_model(new_model, _model);
    new_predictor->set_sampling_method(_sampling_method);
    new_predictor->set_param_size(_param_size);
    return new_predictor;
  }

  void set_config(std::shared_ptr<DeepESConfig> config) {
    _config = config;
  }

  void set_sampling_method(std::shared_ptr<SamplingMethod> sampling_method) {
    _sampling_method = sampling_method;
  }
  
  void set_model(std::shared_ptr<T> sampled_model, std::shared_ptr<T> model) {
    _sampled_model = sampled_model;
    _model = model;
  }

  std::shared_ptr<SamplingMethod> get_sampling_method() {
    return _sampling_method;
  }

  std::shared_ptr<Optimizer> get_optimizer() {
    return _optimizer;
  }

  void set_optimizer(std::shared_ptr<Optimizer> optimizer) {
    _optimizer = optimizer;
  }

  void set_param_size(int param_size) {
    _param_size = param_size;
  }

  torch::Tensor predict(const torch::Tensor& x) {
    return _sampled_model->forward(x);
  }

  bool update(std::vector<SamplingKey>& noisy_keys, std::vector<float>& noisy_rewards) {
    compute_centered_ranks(noisy_rewards);
    float* noise = new float [_param_size];
    float* neg_gradients = new float [_param_size];
    memset(neg_gradients, 0, _param_size * sizeof(float));
    for (int i = 0; i < noisy_keys.size(); ++i) {
      int key = noisy_keys[i].key(0);
      float reward = noisy_rewards[i];
      bool success = _sampling_method->resampling(key, noise, _param_size);
      for (int j = 0; j < _param_size; ++j) {
        neg_gradients[j] += noise[j] * reward;
      }
    }
    for (int j = 0; j < _param_size; ++j) {
      neg_gradients[j] /= -1.0 * noisy_keys.size();
    }

    //update
    auto params = _model->named_parameters();
    int counter = 0;
    for (auto& param: params) {
      torch::Tensor tensor = param.value().view({-1});
      auto tensor_a = tensor.accessor<float,1>();
      _optimizer->update(tensor_a, neg_gradients+counter, tensor.size(0));
      counter += tensor.size(0);
    }
    delete[] noise;
    delete[] neg_gradients;
  }

  SamplingKey add_noise() {
    SamplingKey sampling_key;
    auto sampled_params = _sampled_model->named_parameters();
    auto params = _model->named_parameters();
    float* noise = new float [_param_size];
    int key = _sampling_method->sampling(noise, _param_size);
    sampling_key.add_key(key);
    int counter = 0;
    for (auto& param: sampled_params) {
      torch::Tensor sampled_tensor = param.value().view({-1});
      std::string param_name = param.key();
      torch::Tensor tensor = params.find(param_name)->view({-1});
      auto sampled_tensor_a = sampled_tensor.accessor<float,1>();
      auto tensor_a = tensor.accessor<float,1>();
      for (int j = 0; j < tensor.size(0); ++j) {
        sampled_tensor_a[j] = tensor_a[j] + noise[counter + j];
      }
      counter += tensor.size(0);
    }
    delete[] noise;
    return sampling_key;
  }

  int param_size() {
    if (_param_size == 0) {
      auto params = _model->named_parameters();
      for (auto& param: params) {
        torch::Tensor tensor = param.value().view({-1});
        _param_size += tensor.size(0);
      }
    }
    return _param_size;
  }

private:
  std::shared_ptr<T> _sampled_model;
  std::shared_ptr<T> _model;
  std::shared_ptr<SamplingMethod> _sampling_method;
  std::shared_ptr<Optimizer> _optimizer;
  std::shared_ptr<DeepESConfig> _config;
  int _param_size;
};

}
#endif
