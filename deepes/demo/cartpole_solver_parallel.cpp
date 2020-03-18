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

#include <torch/torch.h>
#include <memory>
#include <algorithm>
#include <glog/logging.h>
#include <omp.h>
#include "cartpole.h"
#include "gaussian_sampling.h"
#include "model.h"
#include "torch_predictor.h"

using namespace DeepES;
const int ITER = 100;

float evaluate(CartPole& env, std::shared_ptr<Predictor<Model>> predictor) {
  float total_reward = 0.0;
  env.reset();
  auto obs = env.getState();
  while (true) {
    torch::Tensor action = predictor->predict(obs);
    int act = std::get<1>(action.max(-1)).item<long>(); 
    env.step(act);
    float reward = env.getReward(); 
    auto done = env.isDone();
    total_reward += reward;
    if (done) break;
    obs = env.getState();
  }
  return total_reward;
}

int main(int argc, char* argv[]) {
  //google::InitGoogleLogging(argv[0]);
  std::vector<CartPole> envs;
  for (int i = 0; i < ITER; ++i) {
    envs.push_back(CartPole());
  }

  auto model = std::make_shared<Model>(4, 2);
  std::shared_ptr<Predictor<Model>> predictor = std::make_shared<Predictor<Model>>(model, "../deepes_config.prototxt");
  std::vector<std::shared_ptr<Predictor<Model>>> noisy_predictors;
  for (int i = 0; i < ITER; ++i) {
    noisy_predictors.push_back(predictor->clone());
  }

  std::vector<SamplingKey> noisy_keys;
  std::vector<float> noisy_rewards(ITER, 0.0f);
  noisy_keys.resize(ITER);

  for (int epoch = 0; epoch < 10000; ++epoch) {
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < ITER; ++i) {
      auto noisy_predictor = noisy_predictors[i];
      SamplingKey key = noisy_predictor->add_noise();
      float reward = evaluate(envs[i], noisy_predictor);
      noisy_keys[i] = key;
      noisy_rewards[i] = reward;
    }

    predictor->update(noisy_keys, noisy_rewards);

    int reward = evaluate(envs[0], predictor);
    LOG(INFO) << "Epoch:" << epoch << " Reward: " << reward;
  }
}
