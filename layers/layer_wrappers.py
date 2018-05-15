"""
Wrappers for fluid.layers so that the layers can share parameters conveniently.
"""

from paddle.fluid.layers import *
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid.layers as layers
import warnings


class LayerCounter:
    create_parameter = 0
    fc = 0
    embedding = 0
    dynamic_lstm = 0
    dynamic_lstmp = 0
    dynamic_gru = 0
    sequence_conv = 0
    conv2d = 0
    conv2d_transpose = 0
    row_conv = 0


class LayerFunc(object):
    def __init__(self, param_name=None, bias_name=None):
        self.param_name = param_name
        self.bias_name = bias_name

    @staticmethod
    def check_type(layer):
        bases = layer.__class__.__bases__
        return len(bases) == 1 and bases[0].__name__ == "LayerFunc"


def check_param_name(attr):
    """
    This function is used to check if the user has set a name for parameter.
    """
    if attr is not None:
        if attr:  ## attr could be False, in which case the param is not used
            assert attr.name is None, \
                "Do not set parameter name for pprl.layers; leave it as None"
        return attr
    else:
        return ParamAttr(name="")


def paras_binding(src_layer, target_layer):
    """
    This function manually let the parameters of target_layer point to those of src_layer,
    *within the same program scope*.
    This is useful if the parameters are shared between two different layers.

    For example,

        ## assume the input is 10000 dim
        embed = pprl.layers.embedding(size=100)
        ## assume the input is 100 dim
        softmax = pprl.layers.fc(size=10000)
        pprl.layers.paras_binding(embed, softmax)

    If the two layers have different numbers of parameters, then an error will occur
    when executing the target_layer. The user is responsible for the result of the binding.

    If the src_layer has the param or bias missing in the target_layer, then the target_layer
    still uses its own copy of parameters;
    if the target_layer has the param or bias missing in the src_layer, then it does not matter.
    """
    assert LayerFunc.check_type(src_layer)
    assert LayerFunc.check_type(target_layer)

    if src_layer.param_name is not None:
        target_layer.param_name = src_layer.param_name
    elif target_layer.param_name is not None:
        warnings.warn(target_layer.__class__ + "'s params are not bound to " + src_layer.__class__)

    if src_layer.bias_name is not None:
        target_layer.bias_name = src_layer.bias_name
    elif target_layer.param_name is not None:
        warnings.warn(target_layer.__class__ + "'s biases are not bound to " + src_layer.__class__)


def create_parameter(shape, dtype):
    """
    Return a function that creates paddle.fluid.layers.create_parameter
    with shape and dtype.
    """
    param_name = "para_%d.w" % LayerCounter.create_parameter
    LayerCounter.create_parameter += 1

    class CreateParameter_(LayerFunc):
        def __init__(self):
            super(CreateParameter_, self).__init__(param_name)

        def __call__(self,
                     name=None,
                     attr=None,
                     is_bias=False,
                     default_initializer=None):
            attr = check_param_name(attr)
            if attr:
                attr.name = self.param_name

            return layers.create_parameter(
                shape=shape,
                dtype=dtype,
                name=name,
                attr=attr, # this cannot be None
                is_bias=is_bias,
                default_initializer=default_initializer)

    return CreateParameter_()


def fc(size):
    """
    Return a function that creates a paddle.fluid.layers.fc layer with the
    specified output size.
    """
    param_name = "fc_%d.w" % LayerCounter.fc
    bias_name = "fc_%d.wbias" % LayerCounter.fc
    LayerCounter.fc += 1

    class FC_(LayerFunc):
        def __init__(self):
            super(FC_, self).__init__(param_name, bias_name)

        def __call__(self,
                     input,
                     num_flatten_dims=1,
                     param_attr=None,
                     bias_attr=None,
                     use_mkldnn=False,
                     act=None,
                     is_test=False,
                     name=None):

            param_attr = check_param_name(param_attr)
            bias_attr = check_param_name(bias_attr)
            if param_attr:
                param_attr.name = self.param_name
            if bias_attr:
                bias_attr.name = self.bias_name

            return layers.fc(
                input=input,
                size=size,
                num_flatten_dims=num_flatten_dims,
                param_attr=param_attr,
                bias_attr=bias_attr,
                use_mkldnn=use_mkldnn,
                act=act,
                is_test=is_test,
                name=name)

    return FC_()


def embedding(sizes):
    """
    Return a function that creates a paddle.fluid.layers.embedding layer
    with the specified sizes.
    """
    param_name = "embedding_%d.w" % LayerCounter.embedding
    LayerCounter.embedding += 1

    class Embedding_(LayerFunc):
        def __init__(self):
            super(Embedding_, self).__init__(param_name)

        def __call__(self,
                     input,
                     is_sparse=False,
                     is_distributed=False,
                     padding_idx=None,
                     param_attr=None,
                     dtype="float32"):

            param_attr = check_param_name(param_attr)
            if param_attr:
                param_attr.name = self.param_name

            return layers.embedding(
                input=input,
                size=sizes,
                is_sparse=is_sparse,
                is_distributed=is_distributed,
                padding_idx=padding_idx,
                param_attr=param_attr,
                dtype=dtype)

    return Embedding_()


def dynamic_lstm(size):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_lstm
    with a hidden state size of (size/4)
    """
    param_name = "dynamic_lstm_%d.w" % LayerCounter.dynamic_lstm
    bias_name = "dynamic_lstm_%d.wbias" % LayerCounter.dynamic_lstm
    LayerCounter.dynamic_lstm += 1

    class DynamicLstm_(LayerFunc):
        def __init__(self):
            super(DynamicLstm_, self).__init__(param_name, bias_name)

        def __call__(self,
                     input,
                     param_attr=None,
                     bias_attr=None,
                     use_peepholes=True,
                     is_reverse=False,
                     gate_activation="sigmoid",
                     cell_activation="tanh",
                     candidate_activation="tanh",
                     dtype="float32",
                     name=None):
            param_attr = check_param_name(param_attr)
            bias_attr = check_param_name(bias_attr)
            if param_attr:
                param_attr.name = self.param_name
            if bias_attr:
                bias_attr.name = self.bias_name

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
                dtype=dtype,
                name=name)

    return DynamicLstm_()


def dynamic_lstmp(size, proj_size):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_lstmp layer,
    with a hidden state size of (size/4) and a projection size of proj_size
    """
    param_name = "dynamic_lstmp_%d.w" % LayerCounter.dynamic_lstmp
    bias_name = "dynamic_lstmp_%d.wbias" % LayerCounter.dynamic_lstmp
    LayerCounter.dynamic_lstmp += 1

    class DynamicLstmp_(LayerFunc):
        def __init__(self):
            super(DynamicLstmp_, self).__init__(param_name, bias_name)

        def __call__(self,
                     input,
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

            param_attr = check_param_name(param_attr)
            bias_attr = check_param_name(bias_attr)
            if param_attr:
                param_attr.name = self.param_name
            if bias_attr:
                bias_attr.name = self.bias_name

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
                dtype=dtype,
                name=name)

    return DynamicLstmp_()


def dynamic_gru(size):
    """
    Return a function that creates a paddle.fluid.layers.dynamic_gru layer
    with the specified hidden state size.
    """
    param_name = "dynamic_gru_%d.w" % LayerCounter.dynamic_gru
    bias_name = "dynamic_gru_%d.wbias" % LayerCounter.dynamic_gru
    LayerCounter.dynamic_gru += 1

    class DynamicGru_(LayerFunc):
        def __init__(self):
            super(DynamicGru_, self).__init__(param_name, bias_name)

        def __call__(self,
                     input,
                     param_attr=None,
                     bias_attr=None,
                     is_reverse=False,
                     gate_activation='sigmoid',
                     candidate_activation='tanh',
                     h_0=None):

            param_attr = check_param_name(param_attr)
            bias_attr = check_param_name(bias_attr)
            if param_attr:
                param_attr.name = self.param_name
            if bias_attr:
                bias_attr.name = self.bias_name

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
    raise NotImplementedError()


def linear_chain_crf(**kwargs):
    raise NotImplementedError()


def crf_decoding(**kwargs):
    raise NotImplementedError()


def sequence_conv(num_filters, filter_size=3):
    """
    Return a function that creates a paddle.fluid.layers.sequence_conv layer,
    with num_filters filters and a filter size of filter_size.
    """
    param_name = "sequence_conv_%d.w" % LayerCounter.sequence_conv
    bias_name = "sequence_conv_%d.wbias" % LayerCounter.sequence_conv
    LayerCounter.sequence_conv += 1

    class SequenceConv_(LayerFunc):
        def __init__(self):
            super(SequenceConv_, self).__init__(param_name, bias_name)

        def __call__(self,
                     input,
                     filter_stride=1,
                     padding=None,
                     bias_attr=None,
                     param_attr=None,
                     act=None):

            param_attr = check_param_name(param_attr)
            bias_attr = check_param_name(bias_attr)
            if param_attr:
                param_attr.name = self.param_name
            if bias_attr:
                bias_attr.name = self.bias_name

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


def conv2d(num_filters, filter_size, groups=None):
    """
    Return a function that creates a paddle.fluid.layers.conv2d layer,
    with num_filters filters, a filter size of filter_size, and groups groups
    """
    param_name = "conv2d_%d.w" % LayerCounter.conv2d
    bias_name = "conv2d_%d.wbias" % LayerCounter.conv2d
    LayerCounter.conv2d += 1

    class Conv2D_(LayerFunc):
        def __init__(self):
            super(Conv2D_, self).__init__(param_name, bias_name)

        def __call__(self,
                     input,
                     stride=1,
                     padding=0,
                     dilation=1,
                     param_attr=None,
                     bias_attr=None,
                     use_cudnn=True,
                     use_mkldnn=False,
                     act=None,
                     name=None):

            param_attr = check_param_name(param_attr)
            bias_attr = check_param_name(bias_attr)
            if param_attr:
                param_attr.name = self.param_name
            if bias_attr:
                bias_attr.name = self.bias_name

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
                act=act,
                name=name)

    return Conv2D_()


def conv2d_transpose(num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=0,
                     stride=1,
                     dilation=1):
    """
    Return a function that creates a paddle.fluid.layers.conv2d_transpose layer.
    """
    param_name = "conv2d_trans_$d.w" % LayerCounter.conv2d_transpose
    bias_name = "conv2d_trans_%d.wbias" % LayerCounter.conv2d_transpose
    LayerCounter.conv2d_transpose += 1

    class Conv2DTranspose_(LayerFunc):
        def __init__(self):
            super(Conv2DTranspose_, self).__init__(param_name, bias_name)

        def __call__(self,
                     input,
                     param_attr=None,
                     bias_attr=None,
                     use_cudnn=True,
                     act=None,
                     name=None):
            param_attr = check_param_name(param_attr)
            bias_attr = check_param_name(bias_attr)
            if param_attr:
                param_attr.name = self.param_name
            if bias_attr:
                bias_attr.name = self.bias_name

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
                act=act,
                name=name)

    return Conv2DTranspose_()


def lstm_unit(**kwargs):
    raise NotImplementedError()


def nce(**kwargs):
    raise NotImplementedError()


def row_conv(future_context_size):
    """
    Return a function that creates a paddle.fluid.layers.row_conv layer
    """
    param_name = "row_conv_%d.w" % LayerCounter.row_conv
    LayerCounter.row_conv += 1

    class RowConv_(LayerFunc):
        def __init__(self):
            super(RowConv_, self).__init__(param_name)

        def __call__(self,
                     input,
                     param_attr=None,
                     act=None):
            param_attr = check_param_name(param_attr)
            if param_attr:
                param_attr.name = self.param_name

            return layers.row_conv(
                input=input,
                future_context_size=future_context_size,
                param_attr=param_attr,
                act=act)

    return RowConv_()


def layer_norm(**kwargs):
    raise NotImplementedError()
