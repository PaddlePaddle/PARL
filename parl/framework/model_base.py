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
from abc import ABCMeta
from parl.layers.layer_wrappers import LayerFunc
from parl.plutils import *

__all__ = ['Network', 'Model']


class Network(object):
    """
    A Network is a collection of LayerFuncs or Networks.
    """

    def sync_params_to(self,
                       target_net,
                       gpu_id,
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
                "must be the same class for params syncing!"
            assert (decay >= 0 and decay <= 1)

            param_pairs = self._get_parameter_pairs(self, target_net)

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
            only parameter created by parl.layers included.
            The order of parameter names will be consistent between
            different instances of same parl.Network.

        Returns:
            list of string, param_attr names of all parameters
        """

        try:
            return self._parameter_names
        except AttributeError:
            self._parameter_names = self._get_parameter_names(self)
            return self._parameter_names

    def get_params(self):
        """ Get numpy arrays of parameters in this Network
        
        Returns:
            List of numpy array.
        """
        params = []
        for param_name in self.parameter_names:
            param = fetch_value(param_name)
            params.append(param)

        return params

    def set_params(self, params, gpu_id):
        """ Set parameters in this Network with params
        
        Args:
            params: List of numpy array.
            gpu_id: gpu id where this Network in. (if gpu_id < 0, means in cpu.)
        """
        assert len(params) == len(self.parameter_names), \
                'size of input params should be same as parameters number of current Network'
        for (param_name, param) in list(zip(self.parameter_names, params)):
            set_value(param_name, param, gpu_id)

    def _get_parameter_names(self, obj):
        """ Recursively get parameter names in obj,

        Args:
            obj (parl.Network/parl.LayerFunc/list/tuple/dict): input object

        Returns:
            parameter_names (list of string): all parameter names in obj
        """

        parameter_names = []
        for attr in sorted(obj.__dict__.keys()):
            val = getattr(obj, attr)
            if isinstance(val, Network):
                parameter_names.extend(self._get_parameter_names(val))
            elif isinstance(val, LayerFunc):
                for attr in val.attr_holder.sorted():
                    if attr:
                        parameter_names.append(attr.name)
            elif isinstance(val, tuple) or isinstance(val, list):
                for x in val:
                    parameter_names.extend(self._get_parameter_names(x))
            elif isinstance(val, dict):
                for x in list(val.values()):
                    parameter_names.extend(self._get_parameter_names(x))
            else:
                # for any other type, won't be handled. E.g. set
                pass
        return parameter_names

    def _get_parameter_pairs(self, src, target):
        """ Recursively gets parameters in source network and 
        corresponding parameters in target network.

        Args:
            src (parl.Network/parl.LayerFunc/list/tuple/dict): source object
            target (parl.Network/parl.LayerFunc/list/tuple/dict): target object

        Returns:
            param_pairs (list of tuple): all parameter names in source network
                                         and corresponding parameter names in 
                                         target network.
        """

        param_pairs = []
        if isinstance(src, Network):
            for attr in src.__dict__:
                if not attr in target.__dict__:
                    continue
                src_var = getattr(src, attr)
                target_var = getattr(target, attr)
                param_pairs.extend(
                    self._get_parameter_pairs(src_var, target_var))
        elif isinstance(src, LayerFunc):
            src_attrs = src.attr_holder.sorted()
            target_attrs = target.attr_holder.sorted()
            assert len(src_attrs) == len(target_attrs), \
                    "number of ParamAttr between source layer and target layer should be same."
            for (src_attr, target_attr) in zip(src_attrs, target_attrs):
                if src_attr:
                    assert target_attr, "ParamAttr between source layer and target layer is inconsistent."
                    param_pairs.append((src_attr.name, target_attr.name))
        elif isinstance(src, tuple) or isinstance(src, list):
            for src_var, target_var in zip(src, target):
                param_pairs.extend(
                    self._get_parameter_pairs(src_var, target_var))
        elif isinstance(src, dict):
            for k in src.keys():
                assert k in target
                param_pairs.extend(
                    self._get_parameter_pairs(src[k], target[k]))
        else:
            # for any other type, won't be handled. E.g. set
            pass
        return param_pairs


class Model(Network):
    """
    A Model is owned by an Algorithm. 
    It implements the entire network model(forward part) to solve a specific problem.
    In general, Model is responsible for forward and 
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
