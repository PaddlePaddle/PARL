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

#include "evo_kit/utils.h"
#include <dirent.h>

namespace evo_kit {

bool compute_centered_ranks(std::vector<float>& reward) {
    std::vector<std::pair<float, int>> reward_index;
    float gap = 1.0 / (reward.size() - 1);
    float normlized_rank = -0.5;
    int id = 0;

    for (auto& rew : reward) {
        reward_index.push_back(std::make_pair(rew, id));
        ++id;
    }

    std::sort(reward_index.begin(), reward_index.end());

    for (int i = 0; i < reward.size(); ++i) {
        id = reward_index[i].second;
        reward[id] = normlized_rank;
        normlized_rank += gap;
    }

    return true;
}

std::vector<std::string> list_all_model_dirs(std::string path) {
    std::vector<std::string> model_dirs;
    DIR* dpdf;
    struct dirent* epdf;
    dpdf = opendir(path.data());

    if (dpdf != NULL) {
        while (epdf = readdir(dpdf)) {
            std::string dir(epdf->d_name);

            if (dir.find("model_iter_id") != std::string::npos) {
                model_dirs.push_back(path + "/" + dir);
            }
        }
    }

    closedir(dpdf);
    return model_dirs;
}

std::string read_file(const std::string& filename) {
    std::ifstream ifile(filename.c_str());

    if (!ifile.is_open()) {
        LOG(ERROR) << "Open file: [" << filename << "] failed.";
        return "";
    }

    std::ostringstream buf;
    char ch = '\n';

    while (buf && ifile.get(ch)) {
        buf.put(ch);
    }

    ifile.close();
    return buf.str();
}

}//namespace
