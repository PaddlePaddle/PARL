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
from parl.framework.base import Model


__all__ = ['ComputationTask']

class ComputationTask(object):
    """
    A ComputationTask is responsible for the general data flow
    outside the algorithm.

    A ComputationTask is created in a bottom-up way:
    a. create a Model
    b. create an Algorithm with the model as an input
    c. define a ComputationTask with the algorithm
    """

    def __init__(self, algorithm):
        assert isinstance(algorithm, Algorithm)
        self.alg = algorithm
        self.build_program()
        self.fluid_executor = fluid.Executor(place)
        self.fluid_executor.run(fluid.default_startup_program())

    def build_program(self):
        """build your training program and prediction program here, 
        using the functions define_learn and define_predict in algorithm.
        
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
        this function is the training interface for ComputationTask.
        """
        raise NotImplementedError
