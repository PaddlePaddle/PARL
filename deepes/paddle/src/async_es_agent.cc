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

#include "evo_kit/async_es_agent.h"

namespace evo_kit {

AsyncESAgent::AsyncESAgent(
    const std::string& model_dir,
    const std::string& config_path): ESAgent(model_dir, config_path) {
    _config_path = config_path;
}
AsyncESAgent::~AsyncESAgent() {
    for (const auto kv : _param_delta) {
        float* delta = kv.second;
        delete[] delta;
    }
}

bool AsyncESAgent::_save() {
    using namespace paddle::lite_api;
    bool success = true;

    if (_is_sampling_agent) {
        LOG(ERROR) <<
            "[EvoKit] Cloned AsyncESAgent cannot call `save`.Please use original AsyncESAgent.";
        success = false;
        return success;
    }

    int model_iter_id = _config->async_es().model_iter_id() + 1;
    //current time
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    std::string model_name = "model_iter_id-" + std::to_string(model_iter_id);
    std::string model_path = _config->async_es().model_warehouse() + "/" + model_name;
    LOG(INFO) << "[save]model_path: " << model_path;
    _predictor->SaveOptimizedModel(model_path, LiteModelType::kProtobuf);
    // save config
    auto async_es = _config->mutable_async_es();
    async_es->set_model_iter_id(model_iter_id);
    success = save_proto_conf(_config_path, *_config);

    if (!success) {
        LOG(ERROR) << "[]unable to save config for AsyncESAgent";
        success = false;
        return success;
    }

    int max_to_keep = _config->async_es().max_to_keep();
    success = _remove_expired_model(max_to_keep);
    return success;
}

bool AsyncESAgent::_remove_expired_model(int max_to_keep) {
    bool success = true;
    std::string model_path = _config->async_es().model_warehouse();
    std::vector<std::string> model_dirs = list_all_model_dirs(model_path);
    int model_iter_id = _config->async_es().model_iter_id() + 1;

    for (const auto& dir : model_dirs) {
        int dir_model_iter_id = _parse_model_iter_id(dir);

        if (model_iter_id - dir_model_iter_id >= max_to_keep) {
            std::string rm_command = std::string("rm -rf ") + dir;
            int ret = system(rm_command.c_str());

            if (ret == 0) {
                LOG(INFO) << "[EvoKit] remove expired Model: " << dir;
            } else {
                LOG(ERROR) << "[EvoKit] fail to remove expired Model: " << dir;
                success = false;
                return success;
            }
        }
    }

    return success;
}

bool AsyncESAgent::_compute_model_diff() {
    bool success = true;

    for (const auto& kv : _previous_predictors) {
        int model_iter_id = kv.first;
        std::shared_ptr<PaddlePredictor> old_predictor = kv.second;
        float* diff = new float[_param_size];
        memset(diff, 0, _param_size * sizeof(float));
        int offset = 0;

        for (const std::string& param_name : _param_names) {
            auto des_tensor = old_predictor->GetTensor(param_name);
            auto src_tensor = _predictor->GetTensor(param_name);
            const float* des_data = des_tensor->data<float>();
            const float* src_data = src_tensor->data<float>();
            int64_t tensor_size = ShapeProduction(src_tensor->shape());

            for (int i = 0; i < tensor_size; ++i) {
                diff[i + offset] = des_data[i] - src_data[i];
            }

            offset += tensor_size;
        }

        _param_delta[model_iter_id] = diff;
    }

    return success;
}

bool AsyncESAgent::_load() {
    bool success = true;
    std::string model_path = _config->async_es().model_warehouse();
    std::vector<std::string> model_dirs = list_all_model_dirs(model_path);

    if (model_dirs.size() == 0) {
        int model_iter_id = _config->async_es().model_iter_id();
        success = model_iter_id == 0 ? true : false;

        if (!success) {
            LOG(WARNING) << "[EvoKit] current_model_iter_id is nonzero, but no model is \
        found at the dir: " << model_path;
        }

        return success;
    }

    for (auto& dir : model_dirs) {
        int model_iter_id = _parse_model_iter_id(dir);

        if (model_iter_id == -1) {
            LOG(WARNING) << "[EvoKit] fail to parse model_iter_id: " << dir;
            success = false;
            return success;
        }

        std::shared_ptr<PaddlePredictor> predictor = _load_previous_model(dir);

        if (predictor == nullptr) {
            success = false;
            LOG(WARNING) << "[EvoKit] fail to load model: " << dir;
            return success;
        }

        _previous_predictors[model_iter_id] = predictor;
    }

    success = _compute_model_diff();
    return success;
}

std::shared_ptr<PaddlePredictor> AsyncESAgent::_load_previous_model(std::string model_dir) {
    using namespace paddle::lite_api;
    // 1. Create CxxConfig
    CxxConfig config;
    config.set_model_file(model_dir + "/model");
    config.set_param_file(model_dir + "/params");
    config.set_valid_places({
        Place{TARGET(kX86), PRECISION(kFloat)},
        Place{TARGET(kHost), PRECISION(kFloat)}
    });

    // 2. Create PaddlePredictor by CxxConfig
    std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<CxxConfig>(config);
    return predictor;
}

std::shared_ptr<AsyncESAgent> AsyncESAgent::clone() {

    std::shared_ptr<AsyncESAgent> new_agent = std::make_shared<AsyncESAgent>();

    float* noise = new float [_param_size];

    new_agent->_predictor = _predictor;
    new_agent->_sampling_predictor = paddle::lite_api::CreatePaddlePredictor<CxxConfig>(*_cxx_config);
    new_agent->_is_sampling_agent = true;
    new_agent->_sampling_method = _sampling_method;
    new_agent->_param_names = _param_names;
    new_agent->_param_size = _param_size;
    new_agent->_config = _config;
    new_agent->_noise = noise;

    return new_agent;
}

bool AsyncESAgent::update(
    std::vector<SamplingInfo>& noisy_info,
    std::vector<float>& noisy_rewards) {

    CHECK(!_is_sampling_agent) << "[EvoKit] Cloned ESAgent cannot call update function. \
    Please use original ESAgent.";

    bool success = _load();
    CHECK(success) << "[EvoKit] fail to load previous models.";

    int current_model_iter_id =  _config->async_es().model_iter_id();

    // validate model_iter_id for each sample before the update
    for (int i = 0; i < noisy_info.size(); ++i) {
        int model_iter_id = noisy_info[i].model_iter_id();

        if (model_iter_id != current_model_iter_id
                && _previous_predictors.count(model_iter_id) == 0) {
            LOG(WARNING) << "[EvoKit] The sample with model_dir_id: " << model_iter_id \
                         << " cannot match any local model";
            success = false;
            return success;
        }
    }

    compute_centered_ranks(noisy_rewards);
    memset(_neg_gradients, 0, _param_size * sizeof(float));

    for (int i = 0; i < noisy_info.size(); ++i) {
        int key = noisy_info[i].key(0);
        float reward = noisy_rewards[i];
        int model_iter_id = noisy_info[i].model_iter_id();
        bool success = _sampling_method->resampling(key, _noise, _param_size);
        CHECK(success) << "[EvoKit] resampling error occurs at sample: " << i;
        float* delta = _param_delta[model_iter_id];

        // compute neg_gradients
        if (model_iter_id == current_model_iter_id) {
            for (int64_t j = 0; j < _param_size; ++j) {
                _neg_gradients[j] += _noise[j] * reward;
            }
        } else {
            for (int64_t j = 0; j < _param_size; ++j) {
                _neg_gradients[j] += (_noise[j] + delta[j]) * reward;
            }
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

    success = _save();
    CHECK(success) << "[EvoKit] fail to save model.";
    return true;
}

int AsyncESAgent::_parse_model_iter_id(const std::string& model_path) {
    int model_iter_id = -1;
    int pow = 1;

    for (int i = model_path.size() - 1; i >= 0; --i) {
        if (model_path[i] >= '0' && model_path[i] <= '9') {
            if (model_iter_id == -1) {
                model_iter_id = 0;
            }
        } else {
            break;
        }

        model_iter_id += pow * (model_path[i] - '0');
        pow *= 10;
    }

    return model_iter_id;
}

}//namespace
