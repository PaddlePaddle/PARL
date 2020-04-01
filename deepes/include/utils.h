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

#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <fstream>
#include <algorithm>
#include <glog/logging.h>
#include "deepes.pb.h"
#include <google/protobuf/text_format.h>
#include <fstream>

namespace DeepES{

/*Return ranks that is normliazed to [-0.5, 0.5] with the rewards as input.
  Args:
    reward: an array of rewards
*/
bool compute_centered_ranks(std::vector<float> &reward);

std::string read_file(const std::string& filename);

/* Load a protobuf-based configuration from the file.
 * Args:
 *  config_file: file path.
 *  proto_config: protobuff message for configuration.
 * return 
 */
template<typename T>
bool load_proto_conf(const std::string& config_file, T& proto_config) {
  bool success = true;
  std::ifstream fin(config_file);
  if (!fin || fin.fail()) {
    LOG(ERROR) << "open prototxt config failed: " << config_file;
    success = false;
  } else {
    fin.seekg(0, std::ios::end);
    size_t file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    char* file_content_buffer = new char[file_size];
    fin.read(file_content_buffer, file_size);

    std::string proto_str(file_content_buffer, file_size);
    if (!google::protobuf::TextFormat::ParseFromString(proto_str, &proto_config)) {
      LOG(ERROR) << "Failed to load config: " << config_file;
      success = false;
    }
    delete[] file_content_buffer;
    fin.close();
  }
  return success;
}

template<typename T>
bool save_proto_conf(const std::string& config_file, T&proto_config) {
  bool success = true;
  std::ofstream ofs(config_file, std::ofstream::out);
  if (!ofs || ofs.fail()) {
    LOG(ERROR) << "open prototxt config failed: " << config_file;
    success = false;
  } else {
    std::string config_str;
    success = google::protobuf::TextFormat::PrintToString(proto_config, &config_str);
    if (!success) {
      return success;
    }
    ofs << config_str;
  }
  return success;
}

std::vector<std::string> list_all_model_dirs(std::string path);

}

#endif
