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

import hashlib
import paddle.fluid as fluid
from parl.core.fluid.layers.layer_wrappers import LayerFunc
from parl.core.fluid.plutils import *
from parl.core.model_base import ModelBase
from parl.utils import machine_info

__all__ = ['Model']


class Model(ModelBase):
    """
    | `alias`: ``parl.Model``
    | `alias`: ``parl.core.fluid.agent.Model``

    | ``Model`` is a base class of PARL for the neural network. A ``Model`` is usually a policy or Q-value function, which predicts an action or an estimate according to the environmental observation.

    | To track all the layers , users are required to implement neural networks with the layers from ``parl.layers`` (e.g., parl.layers.fc). These layers has the same APIs as fluid.layers.

    | ``Model`` supports duplicating a ``Model`` instance in a pythonic way:

    | ``copied_model = copy.deepcopy(model)``

    Example:

    .. code-block:: python

        import parl

        class Policy(parl.Model):
            def __init__(self):
                self.fc = parl.layers.fc(size=12, act='softmax')

            def policy(self, obs):
                out = self.fc(obs)
                return out
               
        policy = Policy() 
        copied_policy = copy.deepcopy(model)

    Attributes:
        model_id(str): each model instance has its unique model_id.

    Public Functions:
        - ``sync_weights_to``: synchronize parameters of the current model to another model.
        - ``get_weights``: return a list containing all the parameters of the current model.
        - ``set_weights``: copy parameters from ``set_weights()`` to the model.
        - ``forward``: define the computations of a neural network. **Should** be overridden by all subclasses.
        - ``parameters``: return a list containting names of parameters of the model. 
        - ``set_model_id``: set ``model_id`` of current model explicitly.
        - ``get_model_id``: return the ``model_id`` of current model.

    """

    def sync_weights_to(self,
                        target_model,
                        decay=0.0,
                        share_vars_parallel_executor=None):
        """Synchronize parameters of current model to another model.
        
        To speed up the synchronizing process, it will create a program implicitly to finish the process. It
        also stores a program as the cache to avoid creating program repeatedly.

        target_model_weights = decay * target_model_weights + (1 - decay) * current_model_weights

        Args:
            target_model (`parl.Model`): an instance of ``Model`` that has the same neural network architecture as the current model.
            decay (float):  the rate of decline in copying parameters. 0 if no parameters decay when synchronizing the parameters.
            share_vars_parallel_executor (fluid.ParallelExecutor): Optional. If not None, will use ``fluid.ParallelExecutor``
                                                                   to run program instead of ``fluid.Executor``.

        Example:

        .. code-block:: python

            import copy
            # create a model that has the same neural network structures.
            target_model = copy.deepcopy(model)

            # after initilizing the parameters ...
            model.sync_weights_to(target_mdodel)

        Note:
            Before calling ``sync_weights_to``, parameters of the model must have been initialized.
        """

        args_hash_id = hashlib.md5('{}_{}'.format(
            id(target_model), decay).encode('utf-8')).hexdigest()
        has_cached = False
        try:
            if self._cached_id == args_hash_id:
                has_cached = True
        except AttributeError:
            has_cached = False

        if not has_cached:
            # Can not run _cached program, need create a new program
            self._cached_id = args_hash_id

            assert not target_model is self, "cannot copy between identical model"
            assert isinstance(target_model, Model)
            assert self.__class__.__name__ == target_model.__class__.__name__, \
                "must be the same class for params syncing!"
            assert (decay >= 0 and decay <= 1)

            param_pairs = self._get_parameter_pairs(self, target_model)

            self._cached_sync_weights_program = fluid.Program()

            with fluid.program_guard(self._cached_sync_weights_program):
                for (src_var_name, target_var_name) in param_pairs:
                    src_var = fetch_framework_var(src_var_name)
                    target_var = fetch_framework_var(target_var_name)
                    fluid.layers.assign(
                        decay * target_var + (1 - decay) * src_var, target_var)

            if share_vars_parallel_executor is None:
                # use fluid.Executor
                place = fluid.CUDAPlace(0) if machine_info.is_gpu_available(
                ) else fluid.CPUPlace()
                self._cached_fluid_executor = fluid.Executor(place)
            else:
                # use fluid.ParallelExecutor

                # specify strategy to make ParallelExecutor run faster
                exec_strategy = fluid.ExecutionStrategy()
                exec_strategy.use_experimental_executor = True
                exec_strategy.num_threads = 4
                build_strategy = fluid.BuildStrategy()
                build_strategy.remove_unnecessary_lock = True

                with fluid.scope_guard(fluid.global_scope().new_scope()):
                    self._cached_fluid_executor = fluid.ParallelExecutor(
                        use_cuda=machine_info.is_gpu_available(),
                        main_program=self._cached_sync_weights_program,
                        share_vars_from=share_vars_parallel_executor,
                        exec_strategy=exec_strategy,
                        build_strategy=build_strategy,
                    )
        if share_vars_parallel_executor is None:
            self._cached_fluid_executor.run(self._cached_sync_weights_program)
        else:
            self._cached_fluid_executor.run(fetch_list=[])

    def parameters(self):
        """Get names of all parameters in this ``Model``.

        Only parameters created by ``parl.layers`` are included.
        The order of parameter names is consistent among
        different instances of the same `Model`.

        Returns:
            param_names(list): list of string containing parameter names of all parameters

        Example:

        .. code-block:: python

            model = Model()
            model.parameters()

            # output: 
            ['fc0.w0', 'fc0.bias0']
            
        """
        try:
            return self._parameter_names
        except AttributeError:
            self._parameter_names = self._get_parameter_names(self)
            return self._parameter_names

    def get_weights(self):
        """Returns a Python list containing parameters of current model.

        Returns: a Python list containing the parameters of current model.
        """
        weights = []
        for param_name in self.parameters():
            weight = fetch_value(param_name)
            weights.append(weight)

        return weights

    def set_weights(self, weights):
        """Copy parameters from ``set_weights()`` to the model.
        
        Args:
            weights (list): a Python list containing the parameters.
        """
        assert len(weights) == len(self.parameters()), \
                'size of input weights should be same as weights number of current model'
        try:
            is_gpu_available = self._is_gpu_available
        except AttributeError:
            self._is_gpu_available = machine_info.is_gpu_available()
            is_gpu_available = self._is_gpu_available

        for (param_name, weight) in list(zip(self.parameters(), weights)):
            set_value(param_name, weight, is_gpu_available)

    def _get_parameter_names(self, obj):
        """ Recursively get parameter names in an object.

        Args:
            obj (Object): any object

        Returns:
            parameter_names (list): all parameter names in this object.
        """

        parameter_names = []
        if isinstance(obj, Model):
            for attr in sorted(obj.__dict__.keys()):
                val = getattr(obj, attr)
                parameter_names.extend(self._get_parameter_names(val))
        elif isinstance(obj, LayerFunc):
            for attr in obj.attr_holder.sorted():
                if attr:
                    parameter_names.append(attr.name)
        elif isinstance(obj, tuple) or isinstance(obj, list):
            for x in obj:
                parameter_names.extend(self._get_parameter_names(x))
        elif isinstance(obj, dict):
            for x in list(obj.values()):
                parameter_names.extend(self._get_parameter_names(x))
        else:
            # for any other type, won't be handled. E.g. set
            pass
        return parameter_names

    def _get_parameter_pairs(self, src, target):
        """ Recursively gets parameters in source model and 
        corresponding parameters in target model.

        Args:
            src (`Model`/`LayerFunc`/list/tuple/dict): source object
            target (`Model`/`LayerFunc`/list/tuple/dict): target object

        Returns:
            param_pairs (list of tuple): all parameter names in source model
                                         and corresponding parameter names in 
                                         target model.
        """

        param_pairs = []
        if isinstance(src, Model):
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
