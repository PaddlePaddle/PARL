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

#include "evo_kit/es_agent.h"
#include <ctime>

namespace evo_kit {

int64_t ShapeProduction(const paddle::lite_api::shape_t& shape) {
    int64_t res = 1;

    for (auto i : shape) {
        res *= i;
    }

    return res;
}

ESAgent::~ESAgent() {
    delete[] _noise;

    if (!_is_sampling_agent) {
        delete[] _neg_gradients;
    }
}

ESAgent::ESAgent(const std::string& model_dir, const std::string& config_path) {
    using namespace paddle::lite_api;
    // 1. Create CxxConfig
    _cxx_config = std::make_shared<CxxConfig>();
    std::string model_path = model_dir + "/model";
    std::string param_path = model_dir + "/param";
    std::string model_buffer = read_file(model_path);
    std::string param_buffer = read_file(param_path);
    _cxx_config->set_model_buffer(model_buffer.c_str(), model_buffer.size(),
                                  param_buffer.c_str(), param_buffer.size());
    _cxx_config->set_valid_places({
        Place{TARGET(kX86), PRECISION(kFloat)},
        Place{TARGET(kHost), PRECISION(kFloat)}
    });

    _predictor = CreatePaddlePredictor<CxxConfig>(*_cxx_config);

    _is_sampling_agent = false;
    // Original agent can't be used to sample, so keep it same with _predictor for evaluating.
    _sampling_predictor = _predictor;

    _config = std::make_shared<EvoKitConfig>();
    load_proto_conf(config_path, *_config);

    _sampling_method = create_sampling_method(*_config);

    _optimizer = create_optimizer(_config->optimizer());

    _param_names = _predictor->GetParamNames();
    _param_size = _calculate_param_size();

    _noise = new float [_param_size];
    _neg_gradients = new float [_param_size];
}

std::shared_ptr<ESAgent> ESAgent::clone() {
    if (_is_sampling_agent) {
        LOG(ERROR) << "[EvoKit] only original ESAgent can call `clone` function.";
        return nullptr;
    }

    std::shared_ptr<ESAgent> new_agent = std::make_shared<ESAgent>();

    float* noise = new float [_param_size];

    new_agent->_sampling_predictor = paddle::lite_api::CreatePaddlePredictor<CxxConfig>(*_cxx_config);
    new_agent->_predictor = _predictor;
    new_agent->_cxx_config = _cxx_config;
    new_agent->_is_sampling_agent = true;
    new_agent->_sampling_method = _sampling_method;
    new_agent->_param_names = _param_names;
    new_agent->_config = _config;
    new_agent->_param_size = _param_size;
    new_agent->_noise = noise;

    return new_agent;
}

bool ESAgent::update(
    std::vector<SamplingInfo>& noisy_info,
    std::vector<float>& noisy_rewards) {
    if (_is_sampling_agent) {
        LOG(ERROR) << "[EvoKit] Cloned ESAgent cannot call update function, please use original ESAgent.";
        return false;
    }

    compute_centered_ranks(noisy_rewards);

    memset(_neg_gradients, 0, _param_size * sizeof(float));

    for (int i = 0; i < noisy_info.size(); ++i) {
        int key = noisy_info[i].key(0);
        float reward = noisy_rewards[i];
        bool success = _sampling_method->resampling(key, _noise, _param_size);
        CHECK(success) << "[EvoKit] resampling error occurs at sample: " << i;

        for (int64_t j = 0; j < _param_size; ++j) {
            _neg_gradients[j] += _noise[j] * reward;
        }
    }

    for (int64_t j = 0; j < _param_size; ++j) {
        _neg_gradients[j] /= -1.0 * noisy_info.size();
    }

    //update
    int64_t counter = 0;

    for (std::string param_name : _param_names) {
        std::unique_ptr<Tensor> tensor = _predictor->GetMutableTensor(param_name);
        float* tensor_data = tensor->mutable_data<float>();
        int64_t tensor_size = ShapeProduction(tensor->shape());
        _optimizer->update(tensor_data, _neg_gradients + counter, tensor_size, param_name);
        counter += tensor_size;
    }

    return true;
}

bool ESAgent::add_noise(SamplingInfo& sampling_info) {
    bool success = true;

    if (!_is_sampling_agent) {
        LOG(ERROR) <<
                   "[EvoKit] Original ESAgent cannot call add_noise function, please use cloned ESAgent.";
        success =  false;
        return success;
    }

    int key = 0;
    success = _sampling_method->sampling(&key, _noise, _param_size);
    CHECK(success) << "[EvoKit] sampling error occurs while add_noise.";
    int model_iter_id = _config->async_es().model_iter_id();
    sampling_info.add_key(key);
    sampling_info.set_model_iter_id(model_iter_id);
    int64_t counter = 0;

    for (std::string param_name : _param_names) {
        std::unique_ptr<Tensor> sample_tensor = _sampling_predictor->GetMutableTensor(param_name);
        std::unique_ptr<const Tensor> tensor = _predictor->GetTensor(param_name);
        int64_t tensor_size = ShapeProduction(tensor->shape());

        for (int64_t j = 0; j < tensor_size; ++j) {
            sample_tensor->mutable_data<float>()[j] = tensor->data<float>()[j] + _noise[counter + j];
        }

        counter += tensor_size;
    }

    return success;
}

std::shared_ptr<PaddlePredictor> ESAgent::get_predictor() {
    return _sampling_predictor;
}

int64_t ESAgent::_calculate_param_size() {
    int64_t param_size = 0;

    for (std::string param_name : _param_names) {
        std::unique_ptr<const Tensor> tensor = _predictor->GetTensor(param_name);
        param_size += ShapeProduction(tensor->shape());
    }

    return param_size;
}

}//namespace
