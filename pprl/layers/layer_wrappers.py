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

all_wrapped_layers = [
    "create_parameters", "fc", "embedding", "dynamic_lstm", "dynamic_lstmp",
    "dynamic_gru", "sequence_conv", "conv2d", "conv2d_transpose", "lstm_unit",
    "row_conv"
]


class LayerCounter:
    custom = 0
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
        self.param_name = (None if not param_attr else param_attr.name)
        self.bias_name = (None if not bias_attr else bias_attr.name)

    def parameters(self):
        return (self.param_name, self.bias_name)

    @staticmethod
    def check_type(layer_func):
        """
        Check whether the input is a LayerFunc
        """
        bases = layer_func.__class__.__bases__
        return len(bases) == 1 and bases[0].__name__ == "LayerFunc"


def get_set_paras(set_paras):
    param_name, bias_name = None, None
    if set_paras is not None:
        assert (type(set_paras) is tuple) and len(set_paras) == 2
        param_name, bias_name = set_paras
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

    if name is None:
        attr = check_or_replace_name(param_name, "para_%d.w" %
                                     LayerCounter.create_parameter, attr)
        LayerCounter.create_parameter += 1
    else:
        attr = check_or_replace_name(param_name, "%s_%d_.w" %
                                     (name, LayerCounter.custom), attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "fc_%d.w" %
                                           LayerCounter.fc, param_attr)
        bias_attr = check_or_replace_name(bias_name, "fc_%d.wbias" %
                                          LayerCounter.fc, bias_attr)
        LayerCounter.fc += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" % (
            name, LayerCounter.custom), param_attr)
        bias_attr = check_or_replace_name(bias_name, "%s_%d_.wbias" % (
            name, LayerCounter.custom), bias_attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "embedding_%d.w" %
                                           LayerCounter.embedding, param_attr)
        LayerCounter.embedding += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" % (
            name, LayerCounter.custom), param_attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "dynamic_lstm_%d.w" %
                                           LayerCounter.dynamic_lstm,
                                           param_attr)
        bias_attr = check_or_replace_name(bias_name, "dynamic_lstm_%d.wbias" %
                                          LayerCounter.dynamic_lstm, bias_attr)
        LayerCounter.dynamic_lstm += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" % (
            name, LayerCounter.custom), param_attr)
        bias_attr = check_or_replace_name(bias_name, "%s_%d_.wbias" % (
            name, LayerCounter.custom), bias_attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "dynamic_lstmp_%d.w" %
                                           LayerCounter.dynamic_lstmp,
                                           param_attr)
        bias_attr = check_or_replace_name(bias_name, "dynamic_lstmp_%d.wbias" %
                                          LayerCounter.dynamic_lstmp,
                                          bias_attr)
        LayerCounter.dynamic_lstmp += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" % (
            name, LayerCounter.custom), param_attr)
        bias_attr = check_or_replace_name(bias_name, "%s_%d_.wbias" % (
            name, LayerCounter.custom), bias_attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "dynamic_gru_%d.w" %
                                           LayerCounter.dynamic_gru,
                                           param_attr)
        bias_attr = check_or_replace_name(bias_name, "dynamic_gru_%d.wbias" %
                                          LayerCounter.dynamic_gru, bias_attr)
        LayerCounter.dynamic_gru += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" % (
            name, LayerCounter.custom), param_attr)
        bias_attr = check_or_replace_name(bias_name, "%s_%d_.wbias" % (
            name, LayerCounter.custom), bias_attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "sequence_conv_%d.w" %
                                           LayerCounter.sequence_conv,
                                           param_attr)
        bias_attr = check_or_replace_name(bias_name, "sequence_conv_%d.wbias" %
                                          LayerCounter.sequence_conv,
                                          bias_attr)
        LayerCounter.sequence_conv += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" % (
            name, LayerCounter.custom), param_attr)
        bias_attr = check_or_replace_name(bias_name, "%s_%d_.wbias" % (
            name, LayerCounter.custom), bias_attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "conv2d_%d.w" %
                                           LayerCounter.conv2d, param_attr)
        bias_attr = check_or_replace_name(bias_name, "conv2d_%d.wbias" %
                                          LayerCounter.conv2d, bias_attr)
        LayerCounter.conv2d += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" % (
            name, LayerCounter.custom), param_attr)
        bias_attr = check_or_replace_name(bias_name, "%s_%d_.wbias" % (
            name, LayerCounter.custom), bias_attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "conv2d_trans_$d.w" %
                                           LayerCounter.conv2d_transpose,
                                           param_attr)
        bias_attr = check_or_replace_name(bias_name, "conv2d_trans_%d.wbias" %
                                          LayerCounter.conv2d_transpose,
                                          bias_attr)
        LayerCounter.conv2d_transpose += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" % (
            name, LayerCounter.custom), param_attr)
        bias_attr = check_or_replace_name(bias_name, "%s_%d_.wbias" % (
            name, LayerCounter.custom), bias_attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "lstm_unit_%d.w" %
                                           LayerCounter.lstm_unit, param_attr)
        bias_attr = check_or_replace_name(bias_name, "lstm_unit_%d.wbias" %
                                          LayerCounter.lstm_unit, bias_attr)
        LayerCounter.lstm_unit += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" % (
            name, LayerCounter.custom), param_attr)
        bias_attr = check_or_replace_name(bias_name, "%s_%d_.wbias" % (
            name, LayerCounter.custom), bias_attr)
        LayerCounter.custom += 1

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

    if name is None:
        param_attr = check_or_replace_name(param_name, "row_conv_%d.w" %
                                           LayerCounter.row_conv, param_attr)
        LayerCounter.row_conv += 1
    else:
        param_attr = check_or_replace_name(param_name, "%s_%d_.w" %
                                           LayerCounter.custom, param_attr)
        LayerCounter.custom += 1

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
