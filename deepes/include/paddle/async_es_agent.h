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

#ifndef ASYNC_ES_AGENT_H
#define ASYNC_ES_AGENT_H

#include "es_agent.h"
#include <map>
#include <stdlib.h>

namespace DeepES{
/* DeepES agent with PaddleLite as backend. This agent supports asynchronous update.
 * Users mainly focus on the following functions:
 * 1. clone: clone an agent for multi-thread evaluation
 * 2. add_noise: add noise into parameters.
 * 3. update: update parameters given data collected during evaluation.
 */
class AsyncESAgent: public ESAgent {
  public:
  AsyncESAgent() = delete;

  AsyncESAgent(const CxxConfig& cxx_config);

  ~AsyncESAgent();

    /**
     * @args:
     *    predictor: predictor created by users for prediction.
     *    config_path: the path of configuration file.
     * Note that AsyncESAgent will update the configuration file after calling the update function.
     * Please use the up-to-date configuration.
     */
  AsyncESAgent(
      const std::string& model_dir,
      const std::string& config_path);

    /**
     * @brief: Clone an agent for sampling.
     */
    std::shared_ptr<AsyncESAgent> clone();

    /**
     * @brief: update parameters given data collected during evaluation.
     * @args:
     *   noisy_info: sampling information returned by add_noise function.
     *   noisy_reward: evaluation rewards.
     */
    bool update(
        std::vector<SamplingInfo>& noisy_info,
        std::vector<float>& noisy_rewards);

  private:
    std::map<int, std::shared_ptr<PaddlePredictor>> _previous_predictors;
    std::map<int, float*> _param_delta;
    std::string _config_path;

    /**
     * @brief: parse model_iter_id given a string of model directory.
     * @return: an integer indicating the model_iter_id
     */
    int _parse_model_iter_id(const std::string&);

    /**
     * @brief: compute the distance between current parameter and previous models.
     */
    bool _compute_model_diff();

    /**
     * @brief: remove expired models to avoid overuse of disk space.
     * @args: 
     *  max_to_keep: the maximum number of models to keep locally.
     */
    bool _remove_expired_model(int max_to_keep);

    /**
     * @brief: save up-to-date parameters to the disk.
     */
    bool _save();

    /**
     * @brief: load all models in the model warehouse.
     */
    bool _load();

    /**
     * @brief: load a model given the model directory.
     */
    std::shared_ptr<PaddlePredictor> _load_previous_model(std::string model_dir);
};

} //namespace
#endif
