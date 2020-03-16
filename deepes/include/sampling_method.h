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

#ifndef _SAMPLING_METHOD_H
#define _SAMPLING_METHOD_H

#include <string>
#include <random>
#include "deepes.pb.h"

namespace DeepES{

/*Base class for sampling algorithms. All algorithms are required to override the following functions:
 *
 * 1. load_config
 * 2. sampling
 * 3. resampling
 *
 * View an demostrative algorithm in gaussian_sampling.h
 * */

class SamplingMethod{

public:

    SamplingMethod(): _seed(0) {}

    virtual ~SamplingMethod() {}

    /*Initialize the sampling algorithm given the config with the protobuf format.
     *DeepES library uses only one configuration file for all sampling algorithms. A defalut
     configuration file can be found at: . Usally you won't have to modify the configuration items of other algorithms 
     if you are not using them.
     */
    virtual void load_config(const DeepESConfig& config)=0;

    /*@brief add Gaussian noise to the parameter.
     *
     *@Args:
     *     param: a pointer pointed to the memory of the parameter.
     *     size: the number of floats of the parameter.
     *     noisy_param: The pointer pointed to updated parameter.
     *
     *@return:
     *     success: load configuration successfully or not.
     */
    virtual int sampling(float* noise, int size)=0;

    /*@brief reconstruct the Gaussion noise given the key.
     * This function is often used for updating the neuron network parameters in the offline environment.
     *
     *@Args:
     *     key: a unique key associated with the sampled noise.
     *     noise: a pointer pointed to the memory that stores the noise
     *     size: the number of float to be sampled.
     */
    virtual bool resampling(int key, float* noise, int size)=0;
    
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
