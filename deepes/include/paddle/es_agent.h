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

/* DeepES agent for PaddleLite.
 * Users can use `add_noise` function to add noise to parameters and use `get_sample_predictor`
 * function to get a predictor with added noise to explore.
 * Then can use `update` function to update parameters based on ES algorithm.
 * Users also can `clone` multi agents to sample in multi-thread way.
 */

typedef paddle::lite_api::PaddlePredictor PaddlePredictor;

class ESAgent {
 public:
  ESAgent();

  ~ESAgent();

  ESAgent(
      std::shared_ptr<PaddlePredictor> predictor,
      std::string config_path);
  
  // Return a cloned ESAgent, whose _predictor is same with this->_predictor
  // but _sample_predictor is pointed to a newly created object.
  // This function mainly used to clone a new ESAgent to do sampling in multi-thread way.
  // NOTE: when calling `update` function of current object or cloned one, both of their
  //       parameters will be updated. Because their _predictor is point to same object.
  std::shared_ptr<ESAgent> clone();
  
  // Update parameters of _predictor
  bool update(
      std::vector<SamplingKey>& noisy_keys,
      std::vector<float>& noisy_rewards);
  
  // parameters of _sample_predictor = parameters of _predictor + noise
  SamplingKey add_noise();

  std::shared_ptr<SamplingMethod> get_sampling_method();
  std::shared_ptr<Optimizer> get_optimizer();
  std::shared_ptr<DeepESConfig> get_config();
  int64_t get_param_size();
  std::vector<std::string> get_param_names();
  
  // Return paddle predict _sample_predictor (with addded noise)
  std::shared_ptr<PaddlePredictor> get_sample_predictor();

  // Return paddle predict _predictor (without addded noise)
  std::shared_ptr<PaddlePredictor> get_evaluate_predictor();

  void set_config(std::shared_ptr<DeepESConfig> config);
  void set_sampling_method(std::shared_ptr<SamplingMethod> sampling_method);
  void set_optimizer(std::shared_ptr<Optimizer> optimizer);
  void set_param_size(int64_t param_size);
  void set_param_names(std::vector<std::string> param_names);
  void set_noise(float* noise);
  void set_neg_gradients(float* neg_gradients);
  void set_predictor(
      std::shared_ptr<PaddlePredictor> predictor,
      std::shared_ptr<PaddlePredictor> sample_predictor);

 private:
  std::shared_ptr<PaddlePredictor> _predictor;
  std::shared_ptr<PaddlePredictor> _sample_predictor;
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
