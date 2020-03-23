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

#ifndef DEEPES_PADDLE_ES_AGENT_H_
#define DEEPES_PADDLE_ES_AGENT_H_

#include "paddle_api.h"
#include "optimizer.h"
#include "utils.h"
#include "gaussian_sampling.h"
#include "deepes.pb.h"
#include <vector>


namespace DeepES {

typedef paddle::lite_api::PaddlePredictor PaddlePredictor;

/**
 * @brief DeepES agent for PaddleLite.
 *
 * Users use `clone` fucntion to clone a sampling agent, which can call `add_noise`
 * function to add noise to copied parameters and call `get_predictor` fucntion to 
 * get a paddle predictor with added noise.
 *
 * Then can use `update` function to update parameters based on ES algorithm.
 * Note: parameters of cloned agents will also be updated.
 */
class ESAgent {
 public:
  ESAgent();

  ~ESAgent();

  ESAgent(
      std::shared_ptr<PaddlePredictor> predictor,
      std::string config_path);
  
  /** 
   * @breif Clone a sampling agent
   *
   * Only cloned ESAgent can call `add_noise` function.
   * Each cloned ESAgent will have a copy of original parameters.
   * (support sampling in multi-thread way)
   */
  std::shared_ptr<ESAgent> clone();
  
  /**
   * @brief Update parameters of predictor based on ES algorithm.
   *
   * Only not cloned ESAgent can call `update` function.
   * Parameters of cloned agents will also be updated.
   */
  bool update(
      std::vector<SamplingKey>& noisy_keys,
      std::vector<float>& noisy_rewards);
  
  // copied parameters = original parameters + noise
  bool add_noise(SamplingKey& sampling_key);

  /**
   * @brief Get paddle predict
   *
   * if _is_sampling_agent is true, will return predictor with added noise;
   * if _is_sampling_agent is false, will return predictor without added noise.
   */
  std::shared_ptr<PaddlePredictor> get_predictor();

 private:
  std::shared_ptr<PaddlePredictor> _predictor;
  std::shared_ptr<PaddlePredictor> _sample_predictor;
  bool _is_sampling_agent;
  std::shared_ptr<SamplingMethod> _sampling_method;
  std::shared_ptr<Optimizer> _optimizer;
  std::shared_ptr<DeepESConfig> _config;
  int64_t _param_size;
  std::vector<std::string> _param_names;
  // malloc memory of noise and neg_gradients in advance.
  float* _noise;
  float* _neg_gradients;

  int64_t _calculate_param_size();
};

}

#endif /* DEEPES_PADDLE_ES_AGENT_H_ */
