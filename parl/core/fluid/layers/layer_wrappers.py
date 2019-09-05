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
Wrappers for fluid.layers. It helps to easily share parameters between layers.

NOTE:
    We only encapsulated some of layers with parameters in the fluid, which are frequently used in RL scene.
    If you need use some layers with parameters that are not encapsulated, please submit an issue
    or PR.

Here is an example:
    ```python
    from parl import layers

    class MLPModel(Model):
        def __init__(self):
            self.fc = layers.fc(size=64) # automatically create parameters names "fc_0.w" and "fc_0.b"

        def policy1(self, obs):
            out = self.fc(obs) # Really create parameters with parameters names "fc_0.w" and "fc_0.b"
        
        def policy2(self, obs):
            out = self.fc(obs) # Reusing parameters
    ```
"""

import inspect
import paddle.fluid.layers as layers
import paddle.fluid.unique_name as unique_name
import paddle.fluid as fluid
import six
from copy import deepcopy
from paddle.fluid.executor import _fetch_var
from paddle.fluid.framework import Variable
from paddle.fluid.layers import *
from paddle.fluid.param_attr import ParamAttr
from parl.core.fluid.layers.attr_holder import AttrHolder


def update_attr_name(name, default_name, attr, is_bias):
    """
    Update the name in an attribute
    1. If the user provides a name, then generate the candidate name using the
       provided name;
    2. else generate the candidate name using the default name (which should be
       the name of the layer wrapper).
    3. After obtaining the candidate name, if the attr is False, then we return False;
    4. if the attr is None or attr.name is None, then we set the attr's name as the candidate name;
    5. else we ignore the candidate name and do nothing.
    """

    def check_or_replace_name(name, attr):
        ## if this para is not used
        if attr == False:
            return False

        if attr is None:
            return ParamAttr(name=name)

        if attr.name is None:
            attr.name = name
        return attr

    attr = ParamAttr._to_attr(attr)
    name = (default_name if name is None else name)
    suffix = "b" if is_bias else "w"
    new_name = unique_name.generate(name + "." + suffix)
    return check_or_replace_name(new_name, attr)


class LayerFunc(object):
    def __init__(self, attr_holder):
        self.attr_holder = attr_holder

    def __deepcopy__(self, memo):
        cls = self.__class__
        ## __new__ won't init the class, we need to do that ourselves
        copied = cls.__new__(cls)
        ## record in the memo that self has been copied to avoid recursive copying
        memo[id(self)] = copied

        ## first copy all content
        for k, v in six.iteritems(self.__dict__):
            setattr(copied, k, deepcopy(v, memo))

        ## then we need to create new para names for param_attr in self.attr_holder
        def create_new_para_name(attr):
            if attr:
                assert attr.name, "attr should have a name already!"
                name_key = 'PARL_target_' + attr.name
                attr.name = unique_name.generate(name_key)

        for attr in copied.attr_holder.tolist():
            create_new_para_name(attr)

        ## We require the user to sync the parameter values later, because
        ## this deepcopy is supposed to be called only before the startup
        ## program. This function will cause the computation graph change, so
        ## it cannot be called during the execution.
        return copied

    @property
    def param_name(self):
        if self.attr_holder.param_attr:
            return self.attr_holder.param_attr.name
        else:
            return None

    @property
    def bias_name(self):
        if self.attr_holder.bias_attr:
            return self.attr_holder.bias_attr.name
        else:
            return None

    @property
    def all_params_names(self):
        params_names = []
        for attr in self.attr_holder.tolist():
            if attr:
                params_names.append(attr.name)
        return params_names


def fc(size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       name=None):
    """
    Return a function that creates a paddle.fluid.layers.fc.
    """
    default_name = "fc"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class FC_(LayerFunc):
        def __init__(self):
            super(FC_, self).__init__(
                AttrHolder(param_attr=param_attr, bias_attr=bias_attr))

        def __call__(self, input):
            return layers.fc(
                input=input,
                size=size,
                num_flatten_dims=num_flatten_dims,
                param_attr=self.attr_holder.param_attr,
                bias_attr=self.attr_holder.bias_attr,
                act=act)

    return FC_()


def embedding(size,
              is_sparse=False,
              is_distributed=False,
              padding_idx=None,
              param_attr=None,
              dtype="float32",
              name=None):
    """
    Return a function that creates a paddle.fluid.layers.embedding.
    """
    param_attr = update_attr_name(name, "embedding", param_attr, False)

    class Embedding_(LayerFunc):
        def __init__(self):
            super(Embedding_, self).__init__(AttrHolder(param_attr=param_attr))

        def __call__(self, input):
            return layers.embedding(
                input=input,
                size=size,
                is_sparse=is_sparse,
                is_distributed=is_distributed,
                padding_idx=padding_idx,
                param_attr=self.attr_holder.param_attr,
                dtype=dtype)

    return Embedding_()


def dynamic_lstm(size,
                 param_attr=None,
                 bias_attr=None,
                 use_peepholes=True,
                 is_reverse=False,
                 gate_activation="sigmoid",
                 cell_activation="tanh",
                 candidate_activation="tanh",
                 dtype="float32",
                 name=None):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_lstm.
    """
    default_name = "dynamic_lstm"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class DynamicLstm_(LayerFunc):
        def __init__(self):
            super(DynamicLstm_, self).__init__(
                AttrHolder(param_attr=param_attr, bias_attr=bias_attr))

        def __call__(self, input, h_0=None, c_0=None):
            return layers.dynamic_lstm(
                input=input,
                h_0=h_0,
                c_0=c_0,
                size=size,
                param_attr=self.attr_holder.param_attr,
                bias_attr=self.attr_holder.bias_attr,
                use_peepholes=use_peepholes,
                is_reverse=is_reverse,
                gate_activation=gate_activation,
                cell_activation=cell_activation,
                candidate_activation=candidate_activation,
                dtype=dtype)

    return DynamicLstm_()


def dynamic_lstmp(size,
                  proj_size,
                  param_attr=None,
                  bias_attr=None,
                  use_peepholes=True,
                  is_reverse=False,
                  gate_activation='sigmoid',
                  cell_activation='tanh',
                  candidate_activation='tanh',
                  proj_activation='tanh',
                  dtype='float32',
                  name=None,
                  cell_clip=None,
                  proj_clip=None):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_lstmp.
    """
    default_name = "dynamic_lstmp"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class DynamicLstmp_(LayerFunc):
        def __init__(self):
            super(DynamicLstmp_, self).__init__(
                AttrHolder(param_attr=param_attr, bias_attr=bias_attr))

        def __call__(self, input):
            return layers.dynamic_lstmp(
                input=input,
                size=size,
                proj_size=proj_size,
                param_attr=self.attr_holder.param_attr,
                bias_attr=self.attr_holder.bias_attr,
                use_peepholes=use_peepholes,
                is_reverse=is_reverse,
                gate_activation=gate_activation,
                cell_activation=cell_activation,
                candidate_activation=candidate_activation,
                proj_activation=proj_activation,
                dtype=dtype,
                cell_clip=cell_clip,
                proj_clip=proj_clip)

    return DynamicLstmp_()


def dynamic_gru(size,
                param_attr=None,
                bias_attr=None,
                is_reverse=False,
                gate_activation='sigmoid',
                candidate_activation='tanh',
                origin_mode=False,
                name=None):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_gru.
    """
    default_name = "dynamic_gru"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class DynamicGru_(LayerFunc):
        def __init__(self):
            super(DynamicGru_, self).__init__(
                AttrHolder(param_attr=param_attr, bias_attr=bias_attr))

        def __call__(self, input, h_0=None):
            return layers.dynamic_gru(
                input=input,
                size=size,
                param_attr=self.attr_holder.param_attr,
                bias_attr=self.attr_holder.bias_attr,
                is_reverse=is_reverse,
                gate_activation=gate_activation,
                candidate_activation=candidate_activation,
                h_0=h_0,
                origin_mode=origin_mode)

    return DynamicGru_()


def gru_unit(**kwargs):
    """
    We cannot pass param_attr or bias_attr to paddle.fluid.layers.gru_unit yet.
    """
    raise NotImplementedError()


def linear_chain_crf(**kwargs):
    raise NotImplementedError()


def crf_decoding(**kwargs):
    raise NotImplementedError()


def sequence_conv(num_filters,
                  filter_size=3,
                  filter_stride=1,
                  padding=None,
                  bias_attr=None,
                  param_attr=None,
                  act=None,
                  name=None):
    """
    Return a function that creates a paddle.fluid.layers.sequence_conv.
    """
    default_name = "sequence_conv"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class SequenceConv_(LayerFunc):
        def __init__(self):
            super(SequenceConv_, self).__init__(
                AttrHolder(param_attr=param_attr, bias_attr=bias_attr))

        def __call__(self, input):
            return layers.sequence_conv(
                input=input,
                num_filters=num_filters,
                filter_size=filter_size,
                filter_stride=filter_stride,
                padding=padding,
                bias_attr=self.attr_holder.bias_attr,
                param_attr=self.attr_holder.param_attr,
                act=act)

    return SequenceConv_()


def conv2d(num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           act=None,
           name=None):
    """
    Return a function that creates a paddle.fluid.layers.conv2d.
    """
    default_name = "conv2d"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class Conv2D_(LayerFunc):
        def __init__(self):
            super(Conv2D_, self).__init__(
                AttrHolder(param_attr=param_attr, bias_attr=bias_attr))

        def __call__(self, input):
            return layers.conv2d(
                input=input,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                param_attr=self.attr_holder.param_attr,
                bias_attr=self.attr_holder.bias_attr,
                use_cudnn=use_cudnn,
                act=act)

    return Conv2D_()


def conv2d_transpose(num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=None,
                     param_attr=None,
                     bias_attr=None,
                     use_cudnn=True,
                     act=None,
                     name=None):
    """
    Return a function that creates a paddle.fluid.layers.conv2d_transpose.
    """
    default_name = "conv2d_transpose"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class Conv2DTranspose_(LayerFunc):
        def __init__(self):
            super(Conv2DTranspose_, self).__init__(
                AttrHolder(param_attr=param_attr, bias_attr=bias_attr))

        def __call__(self, input):
            return layers.conv2d_transpose(
                input=input,
                num_filters=num_filters,
                output_size=output_size,
                filter_size=filter_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=groups,
                param_attr=self.attr_holder.param_attr,
                bias_attr=self.attr_holder.bias_attr,
                use_cudnn=use_cudnn,
                act=act)

    return Conv2DTranspose_()


def lstm_unit(forget_bias=0.0, param_attr=None, bias_attr=None, name=None):
    """
    Return a function that creates a paddle.fluid.layers.lstm_unit.
    """
    default_name = "lstm_unit"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class LstmUnit_(LayerFunc):
        def __init__(self):
            super(LstmUnit_, self).__init__(
                AttrHolder(param_attr=param_attr, bias_attr=bias_attr))

        def __call__(self, x_t, hidden_t_prev, cell_t_prev):
            return layers.lstm_unit(
                x_t=x_t,
                hidden_t_prev=hidden_t_prev,
                cell_t_prev=cell_t_prev,
                forget_bias=forget_bias,
                param_attr=self.attr_holder.param_attr,
                bias_attr=self.attr_holder.bias_attr)

    return LstmUnit_()


def nce(**kwargs):
    raise NotImplementedError()


def row_conv(future_context_size, param_attr=None, act=None, name=None):
    """
    Return a function that creates a paddle.fluid.layers.row_conv.
    """
    param_attr = update_attr_name(name, "row_conv", param_attr, False)

    class RowConv_(LayerFunc):
        def __init__(self):
            super(RowConv_, self).__init__(AttrHolder(param_attr=param_attr))

        def __call__(self, input):
            return layers.row_conv(
                input=input,
                future_context_size=future_context_size,
                param_attr=self.attr_holder.param_attr,
                act=act)

    return RowConv_()


def layer_norm(**kwargs):
    raise NotImplementedError()


def batch_norm(act=None,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW',
               in_place=False,
               name=None,
               moving_mean_name=None,
               moving_variance_name=None,
               do_model_average_for_mean_and_var=False,
               fuse_with_relu=False,
               use_global_stats=False):
    """
    Return a function that creates a paddle.fluid.layers.batch_norm.

    """
    default_name = "batch_norm"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)
    moving_mean_attr = update_attr_name(name, default_name + "_moving_mean",
                                        None, False)
    moving_variance_attr = update_attr_name(
        name, default_name + "_moving_variance", None, False)

    class BatchNorm_(LayerFunc):
        def __init__(self):
            super(BatchNorm_, self).__init__(
                AttrHolder(
                    param_attr=param_attr,
                    bias_attr=bias_attr,
                    moving_mean_attr=moving_mean_attr,
                    moving_variance_attr=moving_variance_attr))

        def __call__(self, input, is_test=False):
            return layers.batch_norm(
                input=input,
                act=act,
                is_test=is_test,
                momentum=momentum,
                epsilon=epsilon,
                param_attr=self.attr_holder.param_attr,
                bias_attr=self.attr_holder.bias_attr,
                data_layout=data_layout,
                in_place=in_place,
                name=name,
                moving_mean_name=self.attr_holder.moving_mean_attr.name,
                moving_variance_name=self.attr_holder.moving_variance_attr.
                name,
                do_model_average_for_mean_and_var=
                do_model_average_for_mean_and_var,
                fuse_with_relu=fuse_with_relu,
                use_global_stats=use_global_stats)

    return BatchNorm_()


def create_parameter(shape,
                     dtype,
                     name=None,
                     attr=None,
                     is_bias=False,
                     default_initializer=None):
    """
    Return a function that creates a paddle.fluid.layers.create_parameter.

    """
    param_attr = update_attr_name(name, "create_parameter", attr, False)

    class CreateParameter_(LayerFunc):
        def __init__(self):
            super(CreateParameter_, self).__init__(
                AttrHolder(param_attr=param_attr))

        def __call__(self):
            return layers.create_parameter(
                shape=shape,
                dtype=dtype,
                attr=self.attr_holder.param_attr,
                is_bias=is_bias,
                default_initializer=default_initializer)

    return CreateParameter_()
