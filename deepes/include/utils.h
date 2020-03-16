#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <fstream>
#include <algorithm>
#include <glog/logging.h>
#include "deepes.pb.h"
#include <google/protobuf/text_format.h>

namespace DeepES{

void compute_centered_ranks(std::vector<float> &reward) ;

template<typename T>
int load_proto_conf(const std::string& config_file, T& proto_config) {
    std::ifstream fin(config_file);
    CHECK(fin) << "open config file " << config_file; 
    if (fin.fail()) {
        LOG(FATAL) << "open prototxt config failed: " << config_file;
        return -1;
    }

    fin.seekg(0, std::ios::end);
    size_t file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    char* file_content_buffer = new char[file_size];
    fin.read(file_content_buffer, file_size);

    std::string proto_str(file_content_buffer, file_size);
    if (!google::protobuf::TextFormat::ParseFromString(proto_str, &proto_config)) {
        LOG(FATAL) << "Failed to load config: " << config_file;
        return -1;
    }
    delete[] file_content_buffer;
    fin.close();

    return 0;
}



}

#endif
