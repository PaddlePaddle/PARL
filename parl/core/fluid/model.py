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
from parl.utils.deprecation import deprecated
from parl.utils import machine_info

__all__ = ['Model']


class Model(ModelBase):
    """A `Model`, a collection of `parl.layers`, is owned by an `Algorithm`. 

    It implements the entire network (forward part) to solve a specific problem.

    `Model` can also use deepcopy way to construct target model, which has the same structure as initial model. 
    Note that only the model definition is copied here. To copy the parameters from the current model 
    to the target model, you must explicitly use `sync_weights_to` function after the program is initialized.

    NOTE:
      You need initialize start up program before calling `sync_weights_to` API.

    Here is an example:
    .. code-block:: python
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
       # Need initialize program before calling sync_weights_to
       fluid_executor.run(fluid.default_startup_program()) 
       ...

       # synchronize parameters
       model.sync_weights_to(target_model)

    """

    @deprecated(
        deprecated_in='1.2',
        removed_in='1.3',
        replace_function='sync_weights_to')
    def sync_params_to(self,
                       target_net,
                       decay=0.0,
                       share_vars_parallel_executor=None):
        """Synchronize parameters in the model to another model (target_net).

        target_net_weights = decay * target_net_weights + (1 - decay) * source_net_weights

        Args:
            target_net (`Model`): `Model` object deepcopy from source `Model`.
            decay (float): Float. The decay to use. 
            share_vars_parallel_executor (fluid.ParallelExecutor): if not None, will use fluid.ParallelExecutor 
                                                                   to run program instead of fluid.Executor
        """
        self.sync_weights_to(
            other_model=target_net,
            decay=decay,
            share_vars_parallel_executor=share_vars_parallel_executor)

    def sync_weights_to(self,
                        other_model,
                        decay=0.0,
                        share_vars_parallel_executor=None):
        """Synchronize weights in the model to another model.
        
        To speed up the synchronizing process, will create a program implictly to finish the process. And will
        also cache the program to avoid creating program repeatedly.

        other_model_weights = decay * other_model_weights + (1 - decay) * current_model_weights

        Args:
            other_model (`parl.Model`): object instanced from the same `parl.Model` class with current model.
            decay (float): Float. The decay to use. 
            share_vars_parallel_executor (fluid.ParallelExecutor): if not None, will use fluid.ParallelExecutor 
                                                                   to run program instead of fluid.Executor
        """

        args_hash_id = hashlib.md5('{}_{}'.format(
            id(other_model), decay).encode('utf-8')).hexdigest()
        has_cached = False
        try:
            if self._cached_id == args_hash_id:
                has_cached = True
        except AttributeError:
            has_cached = False

        if not has_cached:
            # Can not run _cached program, need create a new program
            self._cached_id = args_hash_id

            assert not other_model is self, "cannot copy between identical model"
            assert isinstance(other_model, Model)
            assert self.__class__.__name__ == other_model.__class__.__name__, \
                "must be the same class for params syncing!"
            assert (decay >= 0 and decay <= 1)

            param_pairs = self._get_parameter_pairs(self, other_model)

            self._cached_sync_weights_program = fluid.Program()

            with fluid.program_guard(self._cached_sync_weights_program):
                for (src_var_name, target_var_name) in param_pairs:
                    src_var = fetch_framework_var(src_var_name)
                    target_var = fetch_framework_var(target_var_name)
                    fluid.layers.assign(
                        decay * target_var + (1 - decay) * src_var, target_var)

            if share_vars_parallel_executor is None:
                # use fluid.Executor
                place = fluid.CUDAPlace(0) if machine_info.is_gpu_available() else fluid.CPUPlace()
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

    @property
    @deprecated(
        deprecated_in='1.2', removed_in='1.3', replace_function='parameters')
    def parameter_names(self):
        """Get param_attr names of all parameters in the Model.

        Only parameter created by parl.layers included.
        The order of parameter names will be consistent between
        different instances of same `Model`.

        Returns:
            list of string, param_attr names of all parameters
        """
        return self.parameters()

    def parameters(self):
        """Get param_attr names of all parameters in the Model.

        Only parameter created by parl.layers included.
        The order of parameter names will be consistent between
        different instances of same `Model`.

        Returns:
            list of string, param_attr names of all parameters
        """
        try:
            return self._parameter_names
        except AttributeError:
            self._parameter_names = self._get_parameter_names(self)
            return self._parameter_names

    @deprecated(
        deprecated_in='1.2', removed_in='1.3', replace_function='get_weights')
    def get_params(self):
        """Get numpy arrays of parameters in the model.
        
        Returns:
            List of numpy array.
        """
        return self.get_weights()

    @deprecated(
        deprecated_in='1.2', removed_in='1.3', replace_function='set_weights')
    def set_params(self, params):
        """Set parameters in the model with params.
        
        Args:
            params (List): List of numpy array.
        """
        self.set_weights(weights=params)

    def get_weights(self):
        """Get numpy arrays of weights in the model.

        Returns:
            List of numpy array.
        """
        weights = []
        for param_name in self.parameters():
            weight = fetch_value(param_name)
            weights.append(weight)

        return weights

    def set_weights(self, weights):
        """Set weights in the model with given `weights`.
        
        Args:
            weights (List): List of numpy array.
        """
        assert len(weights) == len(self.parameters()), \
                'size of input weights should be same as weights number of current model'
        for (param_name, weight) in list(zip(self.parameters(), weights)):
            set_value(param_name, weight)

    def _get_parameter_names(self, obj):
        """ Recursively get parameter names in obj,

        Args:
            obj (`Model`/`LayerFunc`/list/tuple/dict): input object

        Returns:
            parameter_names (List): all parameter names in obj
        """

        parameter_names = []
        for attr in sorted(obj.__dict__.keys()):
            val = getattr(obj, attr)
            if isinstance(val, Model):
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
