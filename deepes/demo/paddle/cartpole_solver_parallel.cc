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

#include <algorithm>
#include <glog/logging.h>
#include <omp.h>
#include "cartpole.h"
#include "gaussian_sampling.h"
#include "es_agent.h"
#include "paddle_api.h"

using namespace DeepES;
using namespace paddle::lite_api;

const int ITER = 10;

std::shared_ptr<PaddlePredictor> create_paddle_predictor(const std::string& model_dir) {
  // 1. Create CxxConfig
  CxxConfig config;
  config.set_model_dir(model_dir);
  config.set_valid_places({
    Place{TARGET(kX86), PRECISION(kFloat)},
    Place{TARGET(kHost), PRECISION(kFloat)}
  });

  // 2. Create PaddlePredictor by CxxConfig
  std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<CxxConfig>(config);
  return predictor;
}

// Use PaddlePredictor of CartPole model to predict the action.
std::vector<float> forward(std::shared_ptr<PaddlePredictor> predictor, const float* obs) {
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 4});
  input_tensor->CopyFromCpu(obs);
  
  predictor->Run();
  
  std::vector<float> probs(2, 0.0);
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  output_tensor->CopyToCpu(probs.data());
  return probs;
}

int arg_max(const std::vector<float>& vec) {
  return static_cast<int>(std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}


float evaluate(CartPole& env, std::shared_ptr<ESAgent> agent, bool is_eval=false) {
  float total_reward = 0.0;
  env.reset();
  const float* obs = env.getState();

  std::shared_ptr<PaddlePredictor> paddle_predictor;
  if (is_eval) 
    paddle_predictor = agent->get_evaluate_predictor(); // For evaluate
  else 
    paddle_predictor = agent->get_sample_predictor(); // For sampling (ES exploring)

  while (true) {
    std::vector<float> probs = forward(paddle_predictor, obs); 
    int act = arg_max(probs);
    env.step(act);
    float reward = env.getReward(); 
    bool done = env.isDone();
    total_reward += reward;
    if (done) break;
    obs = env.getState();
  }
  return total_reward;
}


int main(int argc, char* argv[]) {
  std::vector<CartPole> envs;
  for (int i = 0; i < ITER; ++i) {
    envs.push_back(CartPole());
  }

  std::shared_ptr<PaddlePredictor> paddle_predictor = create_paddle_predictor("../demo/paddle/cartpole_init_model");
  std::shared_ptr<ESAgent> agent = std::make_shared<ESAgent>(paddle_predictor, "../benchmark/cartpole_config.prototxt");

  std::vector< std::shared_ptr<ESAgent> > sampling_agents{ agent };
  for (int i = 0; i < (ITER - 1); ++i) {
    sampling_agents.push_back(agent->clone());
  }

  std::vector<SamplingKey> noisy_keys;
  std::vector<float> noisy_rewards(ITER, 0.0f);
  noisy_keys.resize(ITER);

  omp_set_num_threads(10);
  for (int epoch = 0; epoch < 10000; ++epoch) {
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < ITER; ++i) {
      std::shared_ptr<ESAgent> sampling_agent = sampling_agents[i];
      SamplingKey key = sampling_agent->add_noise();
      float reward = evaluate(envs[i], sampling_agent);

      noisy_keys[i] = key;
      noisy_rewards[i] = reward;
    }

    // NOTE: all parameters of sampling_agents will be updated
    agent->update(noisy_keys, noisy_rewards);
  
    int reward = evaluate(envs[0], agent, true);
    LOG(INFO) << "Epoch:" << epoch << " Reward: " << reward;
  }
}
