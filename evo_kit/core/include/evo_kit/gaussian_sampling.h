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
//
#ifndef EVO_KIT_GAUSSIAN_SAMPLING_H
#define EVO_KIT_GAUSSIAN_SAMPLING_H

#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "evo_kit/sampling_method.h"
#include "evo_kit/utils.h"

namespace evo_kit {

class GaussianSampling: public SamplingMethod {

public:
    GaussianSampling() {}

    ~GaussianSampling() {}

    /*Initialize the sampling algorithm given the config with the protobuf format.
     *EvoKit library uses only one configuration file for all sampling algorithms.
      A defalut configuration file can be found at: . // TODO: where?
      Usally you won't have to modify the configuration items of other algorithms
      if you are not using them.
     */
    bool load_config(const EvoKitConfig& config);

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
    bool sampling(int* key, float* noise, int64_t size);

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
    bool resampling(int key, float* noise, int64_t size);

private:
    float _std;
};

}

#endif
