"""
This file wraps Fluid layers that have parameters to support parameter sharing.
For other layers that don't have parameters, we simply copy them to this namespace.
"""
from paddle.fluid.layers import *
from layer_wrappers import *
