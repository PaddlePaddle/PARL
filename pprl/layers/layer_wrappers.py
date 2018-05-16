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
import warnings
import inspect

all_wrapped_layers = [
    "create_parameters", "fc", "embedding", "dynamic_lstm", "dynamic_lstmp",
    "dynamic_gru", "sequence_conv", "conv2d", "conv2d_transpose", "lstm_unit",
    "row_conv"
]


class LayerCounter:
    custom = {}
    create_parameter = 0
    fc = 0
    embedding = 0
    dynamic_lstm = 0
    dynamic_lstmp = 0
    dynamic_gru = 0
    sequence_conv = 0
    conv2d = 0
    conv2d_transpose = 0
    lstm_unit = 0
    row_conv = 0


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

    def parameters(self):
        return (self.param_attr, self.bias_attr)


def get_set_paras(set_paras):
    param_name, bias_name = None, None
    if set_paras is not None:
        assert (type(set_paras) is tuple) and len(set_paras) == 2
        param_attr, bias_attr = set_paras
        if param_attr:
            param_name = param_attr.name
        if bias_attr:
            bias_name = bias_attr.name
    return param_name, bias_name


def check_or_replace_name(name, new_name, attr):
    name = (new_name if name is None else name)
    ## if this para is not used
    if attr == False:
        return False

    if attr is None:
        return ParamAttr(name=name)

    assert attr.name is None, \
        "Do not set parameter name for pprl.layers; leave it as None"
    attr.name = name
    return attr


def update_attr_name(name, param_name, field_name, attr, suffix,
                     counter_increase):
    if name is None:
        name = field_name
    else:
        name = "_" + name
        field_name = "custom"

    if field_name == "custom":
        custom_counter = getattr(LayerCounter, field_name)
        if not name in custom_counter:
            custom_counter[name] = 0
        idx = custom_counter[name]
        if counter_increase:
            custom_counter[name] += 1
        setattr(LayerCounter, field_name, custom_counter)
    else:
        idx = getattr(LayerCounter, field_name)
        if counter_increase:
            setattr(LayerCounter, field_name, idx + 1)

    new_name = "%s_%d%s" % (name, idx, suffix)
    return check_or_replace_name(param_name, new_name, attr)


def create_parameter(shape,
                     dtype,
                     attr=None,
                     is_bias=False,
                     default_initializer=None,
                     name=None,
                     set_paras=None):
    """
    Return a function that creates paddle.fluid.layers.create_parameter.
    """

    param_name, _ = get_set_paras(set_paras)
    self_name = inspect.stack[0][3]
    attr = update_attr_name(name, param_name, self_name, attr, ".w", True)

    class CreateParameter_(LayerFunc):
        def __init__(self):
            super(CreateParameter_, self).__init__(attr)

        def __call__(self):
            return layers.create_parameter(
                shape=shape,
                dtype=dtype,
                attr=attr,
                is_bias=is_bias,
                default_initializer=default_initializer)

    return CreateParameter_()


def fc(size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       use_mkldnn=False,
       act=None,
       is_test=False,
       name=None,
       set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.fc.
    """
    param_name, bias_name = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", False)
    bias_attr = update_attr_name(name, bias_name, self_name, bias_attr,
                                 ".wbias", True)

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
              name=None,
              set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.embedding.
    """
    param_name, _ = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", True)

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
                 name=None,
                 set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_lstm.
    """
    param_name, bias_name = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", False)
    bias_attr = update_attr_name(name, bias_name, self_name, bias_attr,
                                 ".wbias", True)

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
                  name=None,
                  set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_lstmp.
    """
    param_name, bias_name = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", False)
    bias_attr = update_attr_name(name, bias_name, self_name, bias_attr,
                                 ".wbias", True)

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
                name=None,
                set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_gru.
    """
    param_name, bias_name = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", False)
    bias_attr = update_attr_name(name, bias_name, self_name, bias_attr,
                                 ".wbias", True)

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
                  name=None,
                  set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.sequence_conv.
    """
    param_name, bias_name = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", False)
    bias_attr = update_attr_name(name, bias_name, self_name, bias_attr,
                                 ".wbias", True)

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
           name=None,
           set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.conv2d.
    """
    param_name, bias_name = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", False)
    bias_attr = update_attr_name(name, bias_name, self_name, bias_attr,
                                 ".wbias", True)

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
                     name=None,
                     set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.conv2d_transpose.
    """
    param_name, bias_name = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", False)
    bias_attr = update_attr_name(name, bias_name, self_name, bias_attr,
                                 ".wbias", True)

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


def lstm_unit(forget_bias=0.0,
              param_attr=None,
              bias_attr=None,
              name=None,
              set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.lstm_unit.
    """
    param_name, bias_name = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", False)
    bias_attr = update_attr_name(name, bias_name, self_name, bias_attr,
                                 ".wbias", True)

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


def row_conv(future_context_size,
             param_attr=None,
             act=None,
             name=None,
             set_paras=None):
    """
    Return a function that creates a paddle.fluid.layers.row_conv.
    """
    param_name, _ = get_set_paras(set_paras)
    self_name = inspect.stack()[0][3]
    param_attr = update_attr_name(name, param_name, self_name, param_attr,
                                  ".w", True)

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
