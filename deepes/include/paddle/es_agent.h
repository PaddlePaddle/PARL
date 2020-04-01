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
#include "optimizer_factory.h"
#include "utils.h"
#include "gaussian_sampling.h"
#include "deepes.pb.h"
#include <vector>

using namespace paddle::lite_api;

namespace DeepES {

int64_t ShapeProduction(const shape_t& shape);

/**
 * @brief DeepES agent with PaddleLite as backend.
 * Users mainly focus on the following functions:
 * 1. clone: clone an agent for multi-thread evaluation
 * 2. add_noise: add noise into parameters.
 * 3. update: update parameters given data collected during evaluation.
 *
 */
class ESAgent {
 public:
  ESAgent() = delete;

  ~ESAgent();


  ESAgent(const std::string& model_dir, const std::string& config_path);
  
  ESAgent(const CxxConfig& cxx_config);
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
      std::vector<SamplingInfo>& noisy_info,
      std::vector<float>& noisy_rewards);
  
  // copied parameters = original parameters + noise
  bool add_noise(SamplingInfo& sampling_info);

  /**
   * @brief Get paddle predict
   *
   * if _is_sampling_agent is true, will return predictor with added noise;
   * if _is_sampling_agent is false, will return predictor without added noise.
   */
  std::shared_ptr<PaddlePredictor> get_predictor();



 protected:
  int64_t _calculate_param_size();

  std::shared_ptr<PaddlePredictor> _predictor;
  std::shared_ptr<PaddlePredictor> _sampling_predictor;
  std::shared_ptr<SamplingMethod> _sampling_method;
  std::shared_ptr<Optimizer> _optimizer;
  std::shared_ptr<DeepESConfig> _config;
  std::shared_ptr<CxxConfig> _cxx_config;
  std::vector<std::string> _param_names;
  // malloc memory of noise and neg_gradients in advance.
  float* _noise;
  float* _neg_gradients;
  int64_t _param_size;
  bool _is_sampling_agent;
};

}

#endif /* DEEPES_PADDLE_ES_AGENT_H_ */
