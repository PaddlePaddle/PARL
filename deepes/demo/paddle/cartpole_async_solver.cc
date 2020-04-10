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
#include "evo_kit/async_es_agent.h"
#include "cartpole.h"
#include "paddle_api.h"

using namespace evo_kit;
using namespace paddle::lite_api;

const int ITER = 10;

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


float evaluate(CartPole& env, std::shared_ptr<AsyncESAgent> agent) {
    float total_reward = 0.0;
    env.reset();
    const float* obs = env.getState();

    std::shared_ptr<PaddlePredictor> paddle_predictor;
    paddle_predictor = agent->get_predictor();

    while (true) {
        std::vector<float> probs = forward(paddle_predictor, obs);
        int act = arg_max(probs);
        env.step(act);
        float reward = env.getReward();
        bool done = env.isDone();
        total_reward += reward;

        if (done) {
            break;
        }

        obs = env.getState();
    }

    return total_reward;
}


int main(int argc, char* argv[]) {
    std::vector<CartPole> envs;

    for (int i = 0; i < ITER; ++i) {
        envs.push_back(CartPole());
    }

    std::shared_ptr<AsyncESAgent> agent =
        std::make_shared<AsyncESAgent>("./demo/paddle/cartpole_init_model",
                                       "./demo/cartpole_config.prototxt");

    // Clone agents to sample (explore).
    std::vector< std::shared_ptr<AsyncESAgent> > sampling_agents;

    for (int i = 0; i < ITER; ++i) {
        sampling_agents.push_back(agent->clone());
    }

    std::vector<SamplingInfo> noisy_info;
    std::vector<SamplingInfo> last_noisy_info;
    std::vector<float> noisy_rewards(ITER, 0.0f);
    std::vector<float> last_noisy_rewards;
    noisy_info.resize(ITER);

    omp_set_num_threads(10);

    for (int epoch = 0; epoch < 100; ++epoch) {
        last_noisy_info.clear();
        last_noisy_rewards.clear();

        if (epoch != 0) {
            for (int i = 0; i < ITER; ++i) {
                last_noisy_info.push_back(noisy_info[i]);
                last_noisy_rewards.push_back(noisy_rewards[i]);
            }
        }

        #pragma omp parallel for schedule(dynamic, 1)

        for (int i = 0; i < ITER; ++i) {
            std::shared_ptr<AsyncESAgent> sampling_agent = sampling_agents[i];
            SamplingInfo info;
            bool success = sampling_agent->add_noise(info);
            float reward = evaluate(envs[i], sampling_agent);

            noisy_info[i] = info;
            noisy_rewards[i] = reward;
        }

        for (int i = 0; i < ITER; ++i) {
            last_noisy_info.push_back(noisy_info[i]);
            last_noisy_rewards.push_back(noisy_rewards[i]);
        }

        // NOTE: all parameters of sampling_agents will be updated
        bool success = agent->update(last_noisy_info, last_noisy_rewards);

        int reward = evaluate(envs[0], agent);
        LOG(INFO) << "Epoch:" << epoch << " Reward: " << reward;
    }
}
