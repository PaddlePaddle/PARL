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
Wrappers for fluid.layers so that the layers can share parameters conveniently.
"""

from paddle.fluid.layers import *
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid.layers as layers
import paddle.fluid.unique_name as unique_name
import warnings
import inspect


class LayerFunc(object):
    def __init__(self, param_attr=False, bias_attr=False):
        self.param_attr = param_attr
        self.bias_attr = bias_attr

    @property
    def param_name(self):
        if self.param_attr:
            return self.param_attr.name
        else:
            return None

    @property
    def bias_name(self):
        if self.bias_attr:
            return self.bias_attr.name
        else:
            return None


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

    name = (default_name if name is None else name)
    suffix = "b" if is_bias else "w"
    new_name = unique_name.generate(name + "." + suffix)
    return check_or_replace_name(new_name, attr)


def fc(size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       use_mkldnn=False,
       act=None,
       is_test=False,
       name=None):
    """
    Return a function that creates a paddle.fluid.layers.fc.
    """
    default_name = "fc"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class FC_(LayerFunc):
        def __init__(self):
            super(FC_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.fc(input=input,
                             size=size,
                             num_flatten_dims=num_flatten_dims,
                             param_attr=param_attr,
                             bias_attr=bias_attr,
                             use_mkldnn=use_mkldnn,
                             act=act,
                             is_test=is_test)

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
            super(Embedding_, self).__init__(param_attr)

        def __call__(self, input):
            return layers.embedding(
                input=input,
                size=size,
                is_sparse=is_sparse,
                is_distributed=is_distributed,
                padding_idx=padding_idx,
                param_attr=param_attr,
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
            super(DynamicLstm_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.dynamic_lstm(
                input=input,
                size=size,
                param_attr=param_attr,
                bias_attr=bias_attr,
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
                  name=None):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_lstmp.
    """
    default_name = "dynamic_lstmp"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class DynamicLstmp_(LayerFunc):
        def __init__(self):
            super(DynamicLstmp_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.dynamic_lstmp(
                input=input,
                size=size,
                proj_size=proj_size,
                param_attr=param_attr,
                bias_attr=bias_attr,
                use_peepholes=use_peepholes,
                is_reverse=is_reverse,
                gate_activation=gate_activation,
                cell_activation=cell_activation,
                candidate_activation=candidate_activation,
                proj_activation=proj_activation,
                dtype=dtype)

    return DynamicLstmp_()


def dynamic_gru(size,
                param_attr=None,
                bias_attr=None,
                is_reverse=False,
                gate_activation='sigmoid',
                candidate_activation='tanh',
                h_0=None,
                name=None):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_gru.
    """
    default_name = "dynamic_gru"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class DynamicGru_(LayerFunc):
        def __init__(self):
            super(DynamicGru_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.dynamic_gru(
                input=input,
                size=size,
                param_attr=param_attr,
                bias_attr=bias_attr,
                is_reverse=is_reverse,
                gate_activation=gate_activation,
                candidate_activation=candidate_activation,
                h_0=h_0)

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
            super(SequenceConv_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.sequence_conv(
                input=input,
                num_filters=num_filters,
                filter_size=filter_size,
                filter_stride=filter_stride,
                padding=padding,
                bias_attr=bias_attr,
                param_attr=param_attr,
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
           use_mkldnn=False,
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
            super(Conv2D_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.conv2d(
                input=input,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                param_attr=param_attr,
                bias_attr=bias_attr,
                use_cudnn=use_cudnn,
                use_mkldnn=use_mkldnn,
                act=act)

    return Conv2D_()


def conv2d_transpose(num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=0,
                     stride=1,
                     dilation=1,
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
            super(Conv2DTranspose_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.conv2d_transpose(
                input=input,
                num_filters=num_filters,
                output_size=output_size,
                filter_size=filter_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                param_attr=param_attr,
                bias_attr=bias_attr,
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
            super(LstmUnit_, self).__init__(param_attr, bias_attr)

        def __call__(self, x_t, hidden_t_prev, cell_t_prev):
            return layers.lstm_unit(
                x_t=x_t,
                hidden_t_prev=hidden_t_prev,
                cell_t_prev=cell_t_prev,
                forget_bias=forget_bias,
                param_attr=param_attr,
                bias_attr=bias_attr)

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
            super(RowConv_, self).__init__(param_attr)

        def __call__(self, input):
            return layers.row_conv(
                input=input,
                future_context_size=future_context_size,
                param_attr=param_attr,
                act=act)

    return RowConv_()


def layer_norm(**kwargs):
    raise NotImplementedError()
