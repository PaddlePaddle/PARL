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

#ifndef _MODEL_H
#define _MODEL_H

#include <torch/torch.h>

struct Model : public torch::nn::Module{

  Model() = delete;

  Model(const int obs_dim, const int act_dim) {

    _obs_dim = obs_dim;
    _act_dim = act_dim;
    int hid1_size = act_dim * 10;
    fc1 = register_module("fc1", torch::nn::Linear(obs_dim, hid1_size));
    fc2 = register_module("fc2", torch::nn::Linear(hid1_size, act_dim));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.reshape({-1, _obs_dim});
    x = torch::tanh(fc1->forward(x));
    x = torch::softmax(fc2->forward(x), 1);
    return x;
  }

  std::shared_ptr<Model> clone() {
    std::shared_ptr<Model> model = std::make_shared<Model>(_obs_dim, _act_dim);
    std::vector<torch::Tensor> parameters1 = parameters();
    std::vector<torch::Tensor> parameters2 = model->parameters();
    for (int i = 0; i < parameters1.size(); ++i) {
      torch::Tensor src = parameters1[i].view({-1});
      torch::Tensor des = parameters2[i].view({-1});
      auto src_a = src.accessor<float, 1>();
      auto des_a = des.accessor<float, 1>();
      for (int j = 0; j < src.size(0); ++j) {
        des_a[j] = src_a[j];
      }
    }
    return model;
  }

  int _act_dim;
  int _obs_dim;
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

#endif
