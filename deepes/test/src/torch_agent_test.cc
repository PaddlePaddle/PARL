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

#include "gtest/gtest.h"

#include <torch/torch.h>
#include <memory>
#include <algorithm>
#include <glog/logging.h>
#include <omp.h>
#include "gaussian_sampling.h"
#include "torch_demo_model.h"
#include "es_agent.h"

#include <random>
#include <math.h>

namespace DeepES {


// The fixture for testing class Foo.
class TorchDemoTest : public ::testing::Test {
protected:
    float evaluate(float* x_list, float* y_list, int size, std::shared_ptr<ESAgent<Model>> agent) {
        float total_loss = 0.0;
        for (int i = 0; i < size; ++i) {
            torch::Tensor x_input = torch::tensor(x_list[i], torch::dtype(torch::kFloat32));
            torch::Tensor predict_y = agent->predict(x_input);
            auto pred_y = predict_y.accessor<float,2>();
            float loss = pow((pred_y[0][0] - y_list[i]), 2);
            total_loss += loss;
        }
        return -total_loss / float(size);
    }

    float train_loss() {
        return -1.0 * evaluate(x_list, y_list, train_data_size, agent);
    }

    float test_loss() {
        return -1.0 * evaluate(test_x_list, test_y_list, test_data_size, agent);
    }

    float train_test_gap() {
        float train_lo = train_loss();
        float test_lo = test_loss();
        if ( train_lo > test_lo) {
            return train_lo - test_lo;
        }
        else {
            return test_lo - train_lo;
        }
    }

    float train_agent() {
    }


    void SetUp() override {
        x_list = new float [train_data_size];
        memset(x_list, 0, train_data_size * sizeof(float));
        y_list = new float [train_data_size];
        memset(y_list, 0, train_data_size * sizeof(float));
        test_x_list = new float [test_data_size];
        memset(test_x_list, 0, test_data_size * sizeof(float));
        test_y_list = new float [test_data_size];
        memset(test_y_list, 0, test_data_size * sizeof(float));

        std::default_random_engine generator(0); // fix seed
        std::uniform_real_distribution<float> uniform(-3.0, 9.0);
        std::normal_distribution<float> norm;
        for (int i = 0; i < train_data_size; ++i) {
            x_list[i] = uniform(generator); // generate data between [-3, 9]
            y_list[i] = sin(x_list[i]) + norm(generator)*0.05; // noise std 0.05
        }
        for (int i= 0; i < test_data_size; ++i) {
            test_x_list[i] = uniform(generator);
            test_y_list[i] = sin(test_x_list[i]);
        }

        model = std::make_shared<Model>(1, 1);
        agent = std::make_shared<ESAgent<Model>>(model, "../test/torch_sin_config.prototxt");

        // Clone agents to sample (explore).
        std::vector<std::shared_ptr<ESAgent<Model>>> sampling_agents;
        for (int i = 0; i < iter; ++i) {
            sampling_agents.push_back(agent->clone());
        }

        std::vector<SamplingKey> noisy_keys;
        std::vector<float> noisy_rewards(iter, 0.0f);
        noisy_keys.resize(iter);
        LOG(INFO) << "start training...";
        for (int epoch = 0; epoch < 1001; ++epoch) {
#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < iter; ++i) {
                auto sampling_agent = sampling_agents[i];
                SamplingKey key;
                bool success = sampling_agent->add_noise(key);
                float reward = evaluate(x_list, y_list, train_data_size, sampling_agent);
                noisy_keys[i] = key;
                noisy_rewards[i] = reward;
            }
            bool success = agent->update(noisy_keys, noisy_rewards);

            if (epoch % 100 == 0) {
                float reward = evaluate(test_x_list, test_y_list, test_data_size, agent);
                float train_reward = evaluate(x_list, y_list, train_data_size, agent);
                LOG(INFO) << "Epoch:" << epoch << " Loss: " << -reward << ", Train loss" << -train_reward;
            }
        }
    }

    void TearDown() override{
        delete[] x_list;
        delete[] y_list;
        delete[] test_x_list;
        delete[] test_y_list;
    }

    // Class members declared here can be used by all tests in the test suite
    int train_data_size = 300;
    int test_data_size = 100;
    int iter = 10;
    float* x_list;
    float* y_list;
    float* test_x_list;
    float* test_y_list;
    std::shared_ptr<ESAgent<Model>> agent;
    std::shared_ptr<Model> model;
};


TEST_F(TorchDemoTest, TrainingEffectTest) {
	EXPECT_LT(train_loss(), 0.1);
	EXPECT_LT(test_loss(), 0.1);
	EXPECT_LT(train_test_gap(), 0.05);
}


} // namespace
