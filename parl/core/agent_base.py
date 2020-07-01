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


class AgentBase(object):
    """`AgentBase` is the base class of the `parl.Agent` in different frameworks.

    `parl.Agent` is responsible for the general data flow outside the algorithm.
    """

    def __init__(self, algorithm):
        """

        Args:
            algorithm (`AlgorithmBase`): an instance of `AlgorithmBase`
        """
        self.alg = algorithm

    def get_weights(self, model_ids=None):
        """Get weights of the agent.

        If `model_ids` is not None, will only return weights of
        models whose model_id are in `model_ids`.

        Note:
            `ModelBase` in list, tuple and dict will be included. But `ModelBase` in
            nested list, tuple and dict won't be included.

         Args:
            model_ids (List/Set): list/set of model_id, will only return weights of models
                              whiose model_id in the `model_ids`.
        
        Returns:
            (Dict): Dict of weights ({attribute name: numpy array/List/Dict})
        """
        return self.alg.get_weights(model_ids=model_ids)

    def set_weights(self, weights, model_ids=None):
        """Set weights of the agent with given weights.

        If `model_ids` is not None, will only set weights of
        models whose model_id are in `model_ids`.

        Note:
            `ModelBase` in list, tuple and dict will be included. But `ModelBase` in
            nested list, tuple and dict won't be included.

        Args:
            weights (Dict): Dict of weights ({attribute name: numpy array/List/Dict})
            model_ids (List/Set): list/set of model_id, will only set weights of models
                              whiose model_id in the `model_ids`.
        
        """
        self.alg.set_weights(weights, model_ids=model_ids)

    def get_model_ids(self):
        """Get all model ids of the self.alg in the agent.

        Returns:
            List of model_id 
        """
        return self.alg.get_model_ids()

    @property
    def model_ids(self):
        return self.get_model_ids()

    def learn(self, *args, **kwargs):
        """The training interface for Agent.
        
        This function will usually do the following things:
            1. Accept numpy data as input;
            2. Feed numpy data or onvert numpy data to tensor (optional);
            3. Call learn function in `Algorithm`.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict the action when given the observation of the enviroment.

        In general, this function is used in test process.

        This function will usually do the following things:
           1. Accept numpy data as input;
           2. Feed numpy data or onvert numpy data to tensor (optional);
           3. Call predict function in `Algorithm`.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Sample the action when given the observation of the enviroment.
            
        In general, this function is used in train process.

        This function will usually do the following things:
           1. Accept numpy data as input;
           2. Feed numpy data or onvert numpy data to tensor (optional);
           3. Call predict or sample function in `Algorithm`;
           4. Add sampling operation in numpy level. (unnecessary if sampling operation have done in `Algorithm`).
        """
        raise NotImplementedError
