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

import warnings
warnings.simplefilter('default')

import paddle.fluid as fluid
from parl.core.fluid import layers
from parl.utils.deprecation import deprecated
from parl.core.agent_base import AgentBase
from parl.core.fluid.algorithm import Algorithm
from parl.utils import machine_info

__all__ = ['Agent']


class Agent(AgentBase):
    def __init__(self, algorithm, gpu_id=None):
        """Build program and run initialization for default_startup_program

        Args:
            algorithm (parl.Algorithm): instance of `parl.core.fluid.algorithm.Algorithm`
        """
        if gpu_id is not None:
            warnings.warn(
                "the `gpu_id` argument of `__init__` function in `parl.Agent` is deprecated since version 1.2 and will be removed in version 1.3.",
                DeprecationWarning,
                stacklevel=2)

        assert isinstance(algorithm, Algorithm)
        super(Agent, self).__init__(algorithm)

        # alias for self.algorithm
        # use self.algorithm is suggested
        self.alg = algorithm
        self.gpu_id = 0 if machine_info.is_gpu_available() else -1

        self.build_program()

        self.place = fluid.CUDAPlace(
            0) if machine_info.is_gpu_available() else fluid.CPUPlace()
        self.fluid_executor = fluid.Executor(self.place)
        self.fluid_executor.run(fluid.default_startup_program())

    def build_program(self):
        """Build leran/predict/sample program here with the 
        learn/predict/sample function defined in algorithm.
        
        Note:
            It's unnecessary to call this function explictly since 
            it will be called automatically in the initialization function. 
        
        To build the program, you may need to do the following:
            a. Create a new program of fluid with program guard;
            b. Define data input layers;
            c. Pass the data variable defined in step b to learn/predict/sample of algorithm;
        """
        raise NotImplementedError

    @deprecated(deprecated_in='1.2',
                removed_in='1.3',
                replace_function='get_weights')
    def get_params(self):
        """ Get parameters of self.algorithm

        Returns:
            List of numpy array. 
        """
        return self.algorithm.get_params()

    @deprecated(deprecated_in='1.2',
                removed_in='1.3',
                replace_function='set_weights')
    def set_params(self, params):
        """Set parameters of self.algorithm

        Args:
            params: List of numpy array.
        """
        self.algorithm.set_params(params)

    def learn(self, *args, **kwargs):
        """The training interface for Agent.
        
        This function will usually do the following things:
            1. Accept numpy data as input;
            2. Feed numpy data;
            3. Run learn program defined in `build_program`.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict the action when given the observation of the enviroment.

        In general, this function is used in test process.

        This function will usually do the following things:
           1. Accept numpy data as input;
           2. Feed numpy data;
           3. Run predict program defined in `build_program`.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Sample the action when given the observation of the enviroment.
            
        In general, this function is used in train process.

        This function will usually do the following things:
           1. Accept numpy data as input;
           2. Feed numpy data;
           3. Run predict/sample program defined in `build_program`.
           4. Add sampling operation in numpy level. (unnecessary if sampling operation have done in `Algorithm`).
        """
        raise NotImplementedError
