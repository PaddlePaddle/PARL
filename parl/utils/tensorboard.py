#-*- coding: utf-8 -*-
#File: tensorboard.py
from tensorboardX import SummaryWriter
from parl.utils import logger

__all__ = []

_writer = SummaryWriter(logdir=logger.get_dir())
logger.info("logdir:{}".format(logger.get_dir()))
_WRITTER_METHOD = ['add_scalar', 'add_histogram', 'close', 'flush']
# export writter functions
for func in _WRITTER_METHOD:
    locals()[func] = getattr(_writer, func)
    __all__.append(func)
