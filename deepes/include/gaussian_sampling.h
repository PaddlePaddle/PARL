#ifndef _GAUSSIAN_SAMPLING_H
#define _GAUSSIAN_SAMPLING_H
#include "sampling_method.h"

namespace DeepES{

class GaussianSampling: public SamplingMethod {

public:
    GaussianSampling() {}

    ~GaussianSampling() {}
    /*Initialize the sampling algorithm given the config with the protobuf format.
     *DeepES library uses only one configuration file for all sampling algorithms. A defalut
     configuration file can be found at: . Usally you won't have to modify the configuration items of other algorithms 
     if you are not using them.
     */
    void load_config(const DeepESConfig& config);

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
    int sampling(float* noise, int size);

    /*@brief reconstruct the Gaussion noise given the key.
     * This function is often used for updating the neuron network parameters in the offline environment.
     *
     *@Args:
     *     key: a unique key associated with the sampled noise.
     *     noise: a pointer pointed to the memory that stores the noise
     *     size: the number of float to be sampled.
     */
    bool resampling(int key, float* noise, int size);

private:
    float _std;
};

}

#endif
