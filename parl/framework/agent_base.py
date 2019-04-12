#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.algorithm_base import Algorithm
from parl.framework.model_base import Model
from parl.utils import get_gpu_count

__all__ = ['Agent']


class Agent(object):
    """
    A Agent is responsible for the general data flow
    outside the algorithm.

    A Agent is created in a bottom-up way:
    a. create a Model
    b. create an Algorithm with the model as an input
    c. define a Agent with the algorithm
    """

    def __init__(self, algorithm, gpu_id=None):
        """ build program and run initialization for default_startup_program
        
        Created object:
            self.alg: parl.framework.Algorithm
            self.gpu_id: int
            self.fluid_executor: fluid.Executor
        """
        assert isinstance(algorithm, Algorithm)
        self.alg = algorithm

        self.build_program()

        if gpu_id is None:
            gpu_id = 0 if get_gpu_count() > 0 else -1
        self.gpu_id = gpu_id
        self.place = fluid.CUDAPlace(
            gpu_id) if gpu_id >= 0 else fluid.CPUPlace()
        self.fluid_executor = fluid.Executor(self.place)
        self.fluid_executor.run(fluid.default_startup_program())

    def build_program(self):
        """build your training program and prediction program here, 
        using the functions define_learn and define_predict in algorithm.
        
        Note that it's unnecessary to call this function explictly since 
        it will be called automatically in the initialization function. 
        
        To build the program, you may need to do the following:
        a. create a new program in fluid with program guard
        b. define your data layer
        c. build your training/prediction program, pass the data variable 
           defined in step b to `define_training/define_prediction` of algorithm
        """
        raise NotImplementedError

    def predict(self, obs):
        """This function will predict the action given current observation of the enviroment.

        Note that this function will only do the prediction and it doesn't try any exploration,
        To explore in the action space, you should create your process in `sample` function below.
        In formally, this function is often used in test process.
        """
        raise NotImplementedError

    def sample(self, obs):
        """This function will predict the action given current observation of the enviroment.
        Additionaly, action will be added noise here to explore a new trajectory. In formally,
        this function is often used in training process.
        """
        raise NotImplementedError

    def learn(self, obs, action, reward, next_obs, terminal):
        """pass data to the training program to update model, 
        this function is the training interface for Agent.
        """
        raise NotImplementedError

    def get_params(self):
        """ Get parameters of self.alg

        Returns:
            List of numpy array. 
        """
        return self.alg.get_params()

    def set_params(self, params):
        """ Set parameters of self.alg

        Args:
            params: List of numpy array.
        """
        self.alg.set_params(params, gpu_id=self.gpu_id)
