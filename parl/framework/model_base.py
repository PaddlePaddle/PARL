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

import hashlib
import paddle.fluid as fluid
from abc import ABCMeta, abstractmethod

__all__ = ['Network', 'Model']


class Network(object):
    """
    A Network is an unordered set of LayerFuncs or Networks.
    """

    def sync_params_to(self,
                       target_net,
                       gpu_id=0,
                       decay=0.0,
                       share_vars_parallel_executor=None):
        """
        Args:
            target_net: Network object deepcopy from source network
            gpu_id: gpu id of target_net 
            decay: Float. The decay to use. 
                   target_net_weights = decay * target_net_weights + (1 - decay) * source_net_weights
            share_vars_parallel_executor: if not None, will use fluid.ParallelExecutor 
                                          to run program instead of fluid.Executor
        """
        args_hash_id = hashlib.md5('{}_{}_{}'.format(
            id(target_net), gpu_id, decay).encode('utf-8')).hexdigest()
        has_cached = False
        try:
            if self._cached_id == args_hash_id:
                has_cached = True
        except AttributeError:
            has_cached = False

        if not has_cached:
            # Can not run _cached program, need create a new program
            self._cached_id = args_hash_id

            assert not target_net is self, "cannot copy between identical networks"
            assert isinstance(target_net, Network)
            assert self.__class__.__name__ == target_net.__class__.__name__, \
                "must be the same class for para syncing!"
            assert (decay >= 0 and decay <= 1)

            # Resolve Circular Imports
            from parl.plutils import get_parameter_pairs, fetch_framework_var

            param_pairs = get_parameter_pairs(self, target_net)

            self._cached_sync_params_program = fluid.Program()

            with fluid.program_guard(self._cached_sync_params_program):
                for (src_var_name, target_var_name) in param_pairs:
                    src_var = fetch_framework_var(src_var_name)
                    target_var = fetch_framework_var(target_var_name)
                    fluid.layers.assign(
                        decay * target_var + (1 - decay) * src_var, target_var)

            if share_vars_parallel_executor is None:
                # use fluid.Executor
                place = fluid.CPUPlace() if gpu_id < 0 \
                        else fluid.CUDAPlace(gpu_id)
                self._cached_fluid_executor = fluid.Executor(place)
            else:
                # use fluid.ParallelExecutor
                use_cuda = True if gpu_id >= 0 else False

                # specify strategy to make ParallelExecutor run faster
                exec_strategy = fluid.ExecutionStrategy()
                exec_strategy.use_experimental_executor = True
                exec_strategy.num_threads = 4
                build_strategy = fluid.BuildStrategy()
                build_strategy.remove_unnecessary_lock = True

                with fluid.scope_guard(fluid.global_scope().new_scope()):
                    self._cached_fluid_executor = fluid.ParallelExecutor(
                        use_cuda=use_cuda,
                        main_program=self._cached_sync_params_program,
                        share_vars_from=share_vars_parallel_executor,
                        exec_strategy=exec_strategy,
                        build_strategy=build_strategy,
                    )
        if share_vars_parallel_executor is None:
            self._cached_fluid_executor.run(self._cached_sync_params_program)
        else:
            self._cached_fluid_executor.run(fetch_list=[])

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

    Model can also use deepcopy way to construct target model, which has the same structure as initial model. 
    Note that only the model definition is copied here. To copy the parameters from the current model 
    to the target model, you must explicitly use sync_params_to function after the program is initialized.

    Here is an example:
        ```python
        import parl.layers as layers
        import parl.Model as Model

        class MLPModel(Model):
            def __init__(self):
                self.fc = layers.fc(size=64)

            def policy(self, obs):
                out = self.fc(obs)
                return out
                
        model = MLPModel() 
        target_model = deepcopy(model) # automatically create new unique parameters names for target_model.fc

        # build program
        x = layers.data(name='x', shape=[100], dtype="float32")
        y1 = model.policy(x) 
        y2 = target_model.policy(x)  

        ...
        # Need initialize program before calling sync_params_to
        fluid_executor.run(fluid.default_startup_program()) 
        ...

        # synchronize parameters
        model.sync_params_to(target_model, gpu_id=gpu_id)
        ```
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
