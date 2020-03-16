#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gaussian_sampling.h"
#include "utils.h"

namespace DeepES{

void GaussianSampling::load_config(const DeepESConfig& config) {
    _std = config.gaussian_sampling().std();
    set_seed(config.seed());
}

int GaussianSampling::sampling(float* noise, int size) {
    int key = rand();
    std::default_random_engine generator(key);
    std::normal_distribution<float> norm;
    for (int i = 0; i < size; ++i) {
        *(noise + i) = norm(generator) * _std;
    }
    return key;
}

bool GaussianSampling::resampling(int key, float* noise, int size) {
    bool success = true;
    if (noise == nullptr) {
        success = false;
    }
    else {
        std::default_random_engine generator(key);
        std::normal_distribution<float> norm;
        for (int i = 0; i < size; ++i) {
            *(noise + i) = norm(generator) * _std;
        }
    }
    return success;
}

}
