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

from paddle.fluid.executor import fetch_var
import paddle.fluid as fluid
from paddle.fluid.layers import *
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid.layers as layers
import paddle.fluid.unique_name as unique_name
from copy import deepcopy
import warnings
import inspect


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


class LayerFunc(object):
    def __init__(self, param_attr=False, bias_attr=False):
        self.param_attr = param_attr
        self.bias_attr = bias_attr

    def sync_paras_to(self, target_layer, gpu_id):
        """
        Copy the paras from self to a target layer
        """
        ## isinstance can handle subclass
        assert isinstance(target_layer, LayerFunc)
        src_attrs = [self.param_attr, self.bias_attr]
        target_attrs = [target_layer.param_attr, target_layer.bias_attr]

        place = fluid.CPUPlace() if gpu_id < 0 \
                else fluid.CUDAPlace(gpu_id)

        for i, attrs in enumerate(zip(src_attrs, target_attrs)):
            src_attr, target_attr = attrs
            assert (src_attr and target_attr) \
                or (not src_attr and not target_attr)
            if not src_attr:
                continue
            src_var = fetch_var(src_attr.name)
            target_var = fetch_var(target_attr.name, return_numpy=False)
            target_var.set(src_var, place)

    def __deepcopy__(self, memo):
        cls = self.__class__
        ## __new__ won't init the class, we need to do that ourselves
        copied = cls.__new__(cls)
        ## record in the memo that self has been copied to avoid recursive copying
        memo[id(self)] = copied

        ## first copy all content
        for k, v in self.__dict__.items():
            setattr(copied, k, deepcopy(v, memo))

        ## then we need to create new para names for self.param_attr and self.bias_attr
        def create_new_para_name(attr):
            if attr:
                assert attr.name, "attr should have a name already!"
                ## remove the last number id but keep the name key
                name_key = "_".join(attr.name.split("_")[:-1])
                attr.name = unique_name.generate(name_key)

        create_new_para_name(copied.param_attr)
        create_new_para_name(copied.bias_attr)
        ## We require the user to sync the parameter values later, because
        ## this deepcopy is supposed to be called only before the startup
        ## program. This function will cause the computation graph change, so
        ## it cannot be called during the execution.
        return copied

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


class Network(object):
    """
    A Network is an unordered set of LayerFunc, Layers, or Networks.
    """

    def sync_paras_to(self, target_net, gpu_id):
        assert not target_net is self, "cannot copy between identical networks"

        for attr in self.__dict__:
            if not attr in target_net.__dict__:
                continue
            val = getattr(self, attr)
            target_val = getattr(target_net, attr)

            assert type(val) == type(target_val)
            ## only these two types of members will be copied
            ## the others will be ignorped
            if isinstance(val, Network) or isinstance(val, LayerFunc):
                val.sync_paras_to(target_val, gpu_id)
            elif isinstance(val, tuple) or isinstance(val, list) or isinstance(
                    val, set):
                for v, tv in zip(val, target_val):
                    v.sync_paras_to(tv, gpu_id)
            elif isinstance(val, dict):
                for k in val.keys():
                    assert k in target_val
                    val[k].sync_paras_to(target_val[k], gpu_id)
            else:
                # for any other type, we do not copy
                pass


def check_caller_name():
    stack = inspect.stack()
    the_class = stack[2][0].f_locals["self"].__class__
    the_method = stack[2][0].f_code.co_name
    assert issubclass(the_class, Network) \
        and the_method == "__init__", \
        "parl.layers can only be called in Network.__init__()!"


def fc(size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       use_mkldnn=False,
       act=None,
       name=None):
    """
    Return a function that creates a paddle.fluid.layers.fc.
    """
    default_name = "fc"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)
    check_caller_name()

    class FC_(LayerFunc):
        def __init__(self):
            super(FC_, self).__init__(param_attr, bias_attr)

        def __call__(self, input, is_test=False):
            return layers.fc(input=input,
                             size=size,
                             num_flatten_dims=num_flatten_dims,
                             param_attr=self.param_attr,
                             bias_attr=self.bias_attr,
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
    check_caller_name()

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
                param_attr=self.param_attr,
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
    check_caller_name()

    class DynamicLstm_(LayerFunc):
        def __init__(self):
            super(DynamicLstm_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.dynamic_lstm(
                input=input,
                size=size,
                param_attr=self.param_attr,
                bias_attr=self.bias_attr,
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
    check_caller_name()

    class DynamicLstmp_(LayerFunc):
        def __init__(self):
            super(DynamicLstmp_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.dynamic_lstmp(
                input=input,
                size=size,
                proj_size=proj_size,
                param_attr=self.param_attr,
                bias_attr=self.bias_attr,
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
    check_caller_name()

    class DynamicGru_(LayerFunc):
        def __init__(self):
            super(DynamicGru_, self).__init__(param_attr, bias_attr)

        def __call__(self, input):
            return layers.dynamic_gru(
                input=input,
                size=size,
                param_attr=self.param_attr,
                bias_attr=self.bias_attr,
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
    check_caller_name()

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
                bias_attr=self.bias_attr,
                param_attr=self.param_attr,
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
    check_caller_name()

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
                param_attr=self.param_attr,
                bias_attr=self.bias_attr,
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
    check_caller_name()

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
                param_attr=self.param_attr,
                bias_attr=self.bias_attr,
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
    check_caller_name()

    class LstmUnit_(LayerFunc):
        def __init__(self):
            super(LstmUnit_, self).__init__(param_attr, bias_attr)

        def __call__(self, x_t, hidden_t_prev, cell_t_prev):
            return layers.lstm_unit(
                x_t=x_t,
                hidden_t_prev=hidden_t_prev,
                cell_t_prev=cell_t_prev,
                forget_bias=forget_bias,
                param_attr=self.param_attr,
                bias_attr=self.bias_attr)

    return LstmUnit_()


def nce(**kwargs):
    raise NotImplementedError()


def row_conv(future_context_size, param_attr=None, act=None, name=None):
    """
    Return a function that creates a paddle.fluid.layers.row_conv.
    """
    param_attr = update_attr_name(name, "row_conv", param_attr, False)
    check_caller_name()

    class RowConv_(LayerFunc):
        def __init__(self):
            super(RowConv_, self).__init__(param_attr)

        def __call__(self, input):
            return layers.row_conv(
                input=input,
                future_context_size=future_context_size,
                param_attr=self.param_attr,
                act=act)

    return RowConv_()


def layer_norm(**kwargs):
    raise NotImplementedError()
