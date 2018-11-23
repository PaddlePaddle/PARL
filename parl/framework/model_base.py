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
"""
Base class to define an Algorithm.
"""

import paddle.fluid as fluid
from abc import ABCMeta, abstractmethod

__all__ = ['Network', 'Model']


class Network(object):
    """
    A Network is an unordered set of LayerFuncs or Networks.
    """
    cached_target_net_id = None
    cached_gpu_id = None
    cached_decay = None
    cached_sync_params_program = None
    cached_fluid_executor = None

    def sync_params_to(self, target_net, gpu_id=0, decay=0.0):
        """
        Args:
            target_net: Network object deepcopy from source network
            gpu_id: gpu id of target_net 
            decay: Float. The decay to use. 
                   target_net_weights = decay * target_net_weights + (1 - decay) * source_net_weights
        """
        if (self.cached_target_net_id == None
                or self.cached_target_net_id != id(target_net)
                or self.cached_gpu_id != gpu_id or self.cached_decay != decay):
            # Can not run cached program, need create a new program
            self.cached_target_net_id = id(target_net)
            self.cached_gpu_id = gpu_id
            self.cached_decay = decay

            assert not target_net is self, "cannot copy between identical networks"
            assert isinstance(target_net, Network)
            assert self.__class__.__name__ == target_net.__class__.__name__, \
                "must be the same class for para syncing!"
            assert (decay >= 0 and decay <= 1)

            # Resolve Circular Imports
            from parl.plutils import get_parameter_pairs, fetch_framework_var

            param_pairs = get_parameter_pairs(self, target_net)

            place = fluid.CPUPlace() if gpu_id < 0 \
                    else fluid.CUDAPlace(gpu_id)
            self.cached_fluid_executor = fluid.Executor(place)
            self.cached_sync_params_program = fluid.Program()

            with fluid.program_guard(self.cached_sync_params_program):
                for (src_var_name, target_var_name, is_bias) in param_pairs:
                    src_var = fetch_framework_var(src_var_name, is_bias)
                    target_var = fetch_framework_var(target_var_name, is_bias)
                    fluid.layers.assign(
                        decay * target_var + (1 - decay) * src_var, target_var)

        self.cached_fluid_executor.run(self.cached_sync_params_program)

    @property
    def parameter_names(self):
        """ param_attr names of all parameters in Network,
            only parameter created by parl.layers included

        Returns:
            list of string, param_attr names of all parameters
        """

        # Resolve Circular Imports
        from parl.plutils import get_parameter_names
        return get_parameter_names(self)


class Model(Network):
    """
    A Model is owned by an Algorithm. 
    It implements the entire network model(forward part) to solve a specific problem.
    In conclusion, Model is responsible for forward and 
    Algorithm is responsible for backward.

    Model can also be used to construct target model, which has the same structure as initial model.
    Here is an example:
        ```python
        class Actor(Model):
            __init__(self, obs_dim, act_dim):
                self.obs_dim = obs_dim
                self.act_dim = act_dim
                self.fc1 = layers.fc(size=128, act='relu')
                self.fc2 = layers.fc(size=64, act='relu')
        actor = Actor(obs_dim=12, act_dim=2)
        target_actor = copy.deepcopy(actor)
        ```

    Note that it's the model structure that is copied from initial actor,
    parameters in initial model havn't been copied to target model.
    To copy parameters, you must explicitly use sync_params_to function after the program is initialized.

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Model, self).__init__()

    def policy(self, *args):
        """
        Implement your policy here. 
        The function was later used by algorithm 
        Return: action_dists: a dict of action distribution objects
                states
        Optional: a model might not always have to implement policy()
        """
        raise NotImplementedError()

    def value(self, *args):
        """
        Return: values: a dict of estimated values for the current observations and states
                        For example, "q_value" and "v_value"
        Optional: a model might not always have to implement value()
        """
        raise NotImplementedError()
