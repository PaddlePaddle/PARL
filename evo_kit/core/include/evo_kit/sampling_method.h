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

#ifndef EVO_KIT_SAMPLING_METHOD_H
#define EVO_KIT_SAMPLING_METHOD_H

#include <string>
#include <random>
#include "evo_kit/evo_kit.pb.h"

namespace evo_kit {

/*Base class for sampling algorithms. All algorithms are required to override the following functions:
 *
 * 1. load_config
 * 2. sampling
 * 3. resampling
 *
 * View an demostrative algorithm in gaussian_sampling.h
 * */

class SamplingMethod {

public:

    SamplingMethod(): _seed(0) {}

    virtual ~SamplingMethod() {}

    /*Initialize the sampling algorithm given the config with the protobuf format.
     *EvoKit library uses only one configuration file for all sampling algorithms.
      A defalut configuration file can be found at: . // TODO: where?
      Usally you won't have to modify the configuration items of other algorithms
      if you are not using them.
     */
    virtual bool load_config(const EvoKitConfig& config) = 0;

    /*@brief generate Gaussian noise and the related key.
     *
     *@Args:
     *     key: a unique key associated with the sampled noise.
     *     noise: a pointer pointed to the memory that stores the noise
     *     size: the number of float to be sampled.
     *
     *@return:
     *     success: generate Gaussian successfully or not.
     */
    virtual bool sampling(int* key, float* noise, int64_t size) = 0;

    /*@brief reconstruct the Gaussion noise given the key.
     * This function is often used for updating the neuron network parameters in the offline environment.
     *
     *@Args:
     *     key: a unique key associated with the sampled noise.
     *     noise: a pointer pointed to the memory that stores the noise
     *     size: the number of float to be sampled.
     *
     *@return:
     *     success: reconstruct Gaussian successfully or not.
     */
    virtual bool resampling(int key, float* noise, int64_t size) = 0;

    bool set_seed(int seed) {
        _seed = seed;
        srand(_seed);
        return true;
    }

    int get_seed() {
        return _seed;
    }

protected:
    int _seed;

};

}
#endif
