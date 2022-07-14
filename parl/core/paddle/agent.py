#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import paddle
from paddle.static import InputSpec
from parl.core.agent_base import AgentBase
from parl.core.paddle.algorithm import Algorithm

__all__ = ['Agent']


class Agent(AgentBase):
    """
    | `alias`: ``parl.Agent``
    | `alias`: ``parl.core.paddle.agent.Agent``

    | Agent is one of the three basic classes of PARL.

    | It is responsible for interacting with the environment and collecting 
    data for training the policy.
    | To implement a customized ``Agent``, users can:

      .. code-block:: python

        import parl

        class MyAgent(parl.Agent):
            def __init__(self, algorithm, act_dim):
                super(MyAgent, self).__init__(algorithm)
                self.act_dim = act_dim

    Attributes:
        alg (parl.algorithm): algorithm of this agent.

    Public Functions:
        - ``sample``: return a noisy action to perform exploration according to the policy.
        - ``predict``: return an action given current observation.
        - ``learn``: update the parameters of self.alg using the `learn_program` defined in `build_program()`.
        - ``save``: save parameters of the ``agent`` to a given path.
        - ``restore``: restore previous saved parameters from a given path.
        - ``train``: set the agent in training mode.
        - ``eval``: set the agent in evaluation mode.

    Todo:
        - allow users to get parameters of a specified model by specifying the model's name in ``get_weights()``.

    """

    def __init__(self, algorithm):
        """

        Args:
            algorithm (parl.Algorithm): an instance of `parl.Algorithm`. This algorithm is then passed to `self.alg`.
        """

        assert isinstance(algorithm, Algorithm)
        super(Agent, self).__init__(algorithm)
        # agent mode (bool): True is in training mode, False is in evaluation mode.
        self.training = True

    def learn(self, *args, **kwargs):
        """The training interface for ``Agent``.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict an action when given the observation of the environment.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Return an action with noise when given the observation of the environment.

        In general, this function is used in train process as noise is added to the action to preform exploration.

        """
        raise NotImplementedError

    def save(self, save_path, model=None):
        """Save parameters.

        Args:
            save_path(str): where to save the parameters.
            model(parl.Model): model that describes the neural network structure. If None, will use self.alg.model.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model_dir')

        """
        if model is None:
            model = self.alg.model
        paddle.save(model.state_dict(), save_path)

    def save_inference_model(self,
                             save_path,
                             input_shape_list,
                             input_dtype_list,
                             model=None):
        """
        Saves input Layer or function as ``paddle.jit.TranslatedLayer`` format model, which can be used for inference.

        Args:
            save_path(str): where to save the inference_model.
            model(parl.Model): model that describes the policy network structure. If None, will use self.alg.model.
            input_shape_list(list): shape of all inputs of the saved model's forward method.
            input_dtype_list(list): dtype of all inputs of the saved model's forward method.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save_inference_model('./inference_model_dir', [[None, 128]], ['float32'])


        Example with actor-critic:

        .. code-block:: python

            agent = AtariAgent()
            agent.save_inference_model('./inference_ac_model_dir', [[None, 128]], ['float32'], agent.alg.model.actor_model)

        """
        if model is None:
            model = self.alg.model
        assert callable(
            getattr(model, 'forward',
                    None)), "forward should be a function in model class."
        assert model.forward.__func__ is not super(
            model.__class__,
            model).forward.__func__, "model needs to implement forward method."
        assert isinstance(
            input_shape_list, list
        ), 'Type of input_shape_list in save_inference_model() should be list, but received {}'.format(
            type(input_shape_list))
        assert isinstance(
            input_dtype_list, list
        ), 'Type of input_dtype_list in save_inference_model() should be list, but received {}'.format(
            type(input_dtype_list))
        assert len(input_shape_list) == len(input_dtype_list)
        input_spec = []
        for input_shape, input_type in zip(input_shape_list, input_dtype_list):
            input_spec.append(InputSpec(shape=input_shape, dtype=input_type))
        paddle.jit.save(model, save_path, input_spec)

    def restore(self, save_path, model=None):
        """Restore previously saved parameters.
        This method requires a program that describes the network structure.
        The save_path argument is typically a value previously passed to ``save_params()``.

        Args:
            save_path(str): path where parameters were previously saved.
            model(parl.Model): model that describes the neural network structure. If None, will use self.alg.model.

        Raises:
            ValueError: if program is None and self.learn_program does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model_dir')
            agent.restore('./model_dir')

        """
        if model is None:
            model = self.alg.model
        param_dict = paddle.load(save_path)
        model.set_state_dict(param_dict)

    def train(self):
        """Sets the agent in training mode, which is the default setting.
        Model of agent will be affected if it has some modules (e.g. Dropout, BatchNorm) that behave differently in train/evaluation mode.

        Example:

        .. code-block:: python

            agent.train()   # default setting
            assert (agent.training is True)
            agent.eval()
            assert (agent.training is False)

        """
        self.alg.model.train()
        self.training = True

    def eval(self):
        """Sets the agent in evaluation mode.
        """
        self.alg.model.eval()
        self.training = False
