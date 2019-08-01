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
    """
    | `alias`: ``parl.Agent``
    | `alias`: ``parl.core.fluid.agent.Agent``

    | Agent is one of the three basic classes of PARL.

    | It is responsible for interacting with the environment and collecting data for training the policy.
    | To implement a customized ``Agent``, users can:

      .. code-block:: python

        import parl

        class MyAgent(parl.Agent):
            def __init__(self, algorithm, act_dim):
                super(MyAgent, self).__init__(algorithm)
                self.act_dim = act_dim
      This class will initialize the neural network parameters automatically, and provides an executor for users to run the programs (self.fluid_executor).

    Attributes:
        gpu_id (int): deprecated. specify which GPU to be used. -1 if to use the CPU.
        fluid_executor (fluid.Executor): executor for running programs of the agent.
        alg (parl.algorithm): algorithm of this agent.

    Public Functions:
        - ``build_program`` (**abstract function**): build various programs for the agent to interact with outer environment.
        - ``get_weights``: return a Python dictionary containing all the parameters of self.alg.
        - ``set_weights``: copy parameters from ``set_weights()`` to this agent.
        - ``sample``: return a noisy action to perform exploration according to the policy.
        - ``predict``: return an action given current observation.
        - ``learn``: update the parameters of self.alg using the `learn_program` defined in `build_program()`.
        - ``save``: save parameters of the ``agent`` to a given path.
        - ``restore``: restore previous saved parameters from a given path.

    Todo:
        - allow users to get parameters of a specified model by specifying the model's name in ``get_weights()``.

    """

    def __init__(self, algorithm, gpu_id=None):
        """Build programs by calling the method ``self.build_program()`` and run initialization function of ``fluid.default_startup_program()``.

        Args:
            algorithm (parl.Algorithm): an instance of `parl.Algorithm`. This algorithm is then passed to `self.alg`.
            gpu_id (int): deprecated. specify which GPU to be used. -1 if to use the CPU.
        """
        if gpu_id is not None:
            warnings.warn(
                "the `gpu_id` argument of `__init__` function in `parl.Agent` is deprecated since version 1.2 and will be removed in version 1.3.",
                DeprecationWarning,
                stacklevel=2)

        assert isinstance(algorithm, Algorithm)
        super(Agent, self).__init__(algorithm)

        self.alg = algorithm
        self.gpu_id = 0 if machine_info.is_gpu_available() else -1

        self.build_program()

        self.place = fluid.CUDAPlace(
            0) if machine_info.is_gpu_available() else fluid.CPUPlace()
        self.fluid_executor = fluid.Executor(self.place)
        self.fluid_executor.run(fluid.default_startup_program())

    def build_program(self):
        """Build various programs here with the
        learn, predict, sample functions of the algorithm.

        Note:
            | Users **must** implement this function in an ``Agent``.
            | This function will be called automatically in the initialization function.

        To build a program, you must do the following:
            a. Create a fluid program with ``fluid.program_guard()``;
            b. Define data layers for feeding the data;
            c. Build various programs(e.g., learn_program, predict_program) with data layers defined in step b.

        Example:

        .. code-block:: python

	    self.pred_program = fluid.Program()

            with fluid.program_guard(self.pred_program):
                obs = layers.data(
                    name='obs', shape=[self.obs_dim], dtype='float32')
                self.act_prob = self.alg.predict(obs)


        """
        raise NotImplementedError

    @deprecated(
        deprecated_in='1.2', removed_in='1.3', replace_function='get_weights')
    def get_params(self):
        """ Returns a Python dictionary containing the whole parameters of self.alg.

        Returns:
            a Python List containing the parameters of self.alg.
        """
        return self.algorithm.get_params()

    @deprecated(
        deprecated_in='1.2', removed_in='1.3', replace_function='set_weights')
    def set_params(self, params):
        """Copy parameters from ``get_params()`` into this agent.

        Args:
            params(dict): a Python List containing the parameters of self.alg.
        """
        self.algorithm.set_params(params)

    def learn(self, *args, **kwargs):
        """The training interface for ``Agent``.
        This function feeds the training data into the learn_program defined in ``build_program()``.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict an action when given the observation of the environment.

        This function feeds the observation into the prediction program defined in ``build_program()``. It is often used in the evaluation stage.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Return an action with noise when given the observation of the environment.

        In general, this function is used in train process as noise is added to the action to preform exploration.

        """
        raise NotImplementedError

    def save(self, save_path, program=None):
        """Save parameters.

        Args:
            save_path(str): where to save the parameters.
            program(fluid.Program): program that describes the neural network structure. If None, will use self.learn_program.

        Raises:
            ValueError: if program is None and self.learn_program does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model.ckpt')

        """
        if program is None:
            program = self.learn_program
        dirname = '/'.join(save_path.split('/')[:-1])
        filename = save_path.split('/')[-1]
        fluid.io.save_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)

    def restore(self, save_path, program=None):
        """Restore previously saved parameters.
        This method requires a program that describes the network structure.
        The save_path argument is typically a value previously passed to ``save_params()``.

        Args:
            save_path(str): path where parameters were previously saved.
            program(fluid.Program): program that describes the neural network structure. If None, will use self.learn_program.

        Raises:
            ValueError: if program is None and self.learn_program does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model.ckpt')
            agent.restore('./model.ckpt')

        """

        if program is None:
            program = self.learn_program
        dirname = '/'.join(save_path.split('/')[:-1])
        filename = save_path.split('/')[-1]
        fluid.io.load_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=program,
            filename=filename)
