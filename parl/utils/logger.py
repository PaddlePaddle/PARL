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

import errno
import logging
import os
import os.path
import sys
from termcolor import colored
import shutil
from datetime import datetime

__all__ = ['set_dir', 'get_dir', 'set_level', 'auto_set_dir']

# globals: logger file and directory:
LOG_DIR = None
_FILE_HANDLER = None


def _makedirs(dirname):
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')


class _Formatter(logging.Formatter):
    def format(self, record):
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            date = colored(
                '[%(asctime)s %(threadName)s @%(filename)s:%(lineno)d]',
                'yellow')
            fmt = date + ' ' + colored(
                'WRN', 'yellow', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            date = colored(
                '[%(asctime)s %(threadName)s @%(filename)s:%(lineno)d]', 'red')
            fmt = date + ' ' + colored(
                'WRN', 'yellow', attrs=['blink']) + ' ' + msg
            fmt = date + ' ' + colored(
                'ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        elif record.levelno == logging.DEBUG:
            date = colored(
                '[%(asctime)s %(threadName)s @%(filename)s:%(lineno)d]',
                'blue')
            fmt = date + ' ' + colored(
                'DEBUG', 'blue', attrs=['blink']) + ' ' + msg
        else:
            date = colored(
                '[%(asctime)s %(threadName)s @%(filename)s:%(lineno)d]',
                'green')
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_Formatter, self).format(record)


def _getlogger():
    logger = logging.getLogger('PARL')
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    if 'DEBUG' in os.environ:
        handler = logging.FileHandler('parl_debug.log')
        handler.setFormatter(_Formatter(datefmt='%m-%d %H:%M:%S'))
        logger.addHandler(handler)
        return logger

    if 'XPARL' not in os.environ:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_Formatter(datefmt='%m-%d %H:%M:%S'))
        logger.addHandler(handler)
    return logger


_logger = _getlogger()
_LOGGING_METHOD = [
    'info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug',
    'setLevel'
]

# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)
    __all__.append(func)

# export Level information
_LOGGING_LEVEL = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
for level in _LOGGING_LEVEL:
    locals()[level] = getattr(logging, level)
    __all__.append(level)


def _set_file(path):
    global _FILE_HANDLER, _logger
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    hdl = logging.FileHandler(filename=path, encoding='utf-8', mode='w')
    hdl.setFormatter(_Formatter(datefmt='%m-%d %H:%M:%S'))

    _FILE_HANDLER = hdl
    _logger.addHandler(hdl)


def set_level(level):
    global _logger, LOG_DIR
    # To set level, need create new handler
    if LOG_DIR is not None:
        set_dir(get_dir())
    _logger.setLevel(level)


def set_dir(dirname):
    global LOG_DIR, _FILE_HANDLER, _logger
    if _FILE_HANDLER:
        # unload and close the old file handler, so that we may safely delete the logger directory
        _logger.removeHandler(_FILE_HANDLER)
        _FILE_HANDLER.close()
        del _FILE_HANDLER

    shutil.rmtree(dirname, ignore_errors=True)
    _makedirs(dirname)
    LOG_DIR = dirname
    _set_file(os.path.join(dirname, 'log.log'))


def auto_set_dir(action=None):
    """Set the global logging directory automatically. The default path is "./train_log/{scriptname}". "scriptname" is the name of the main python file currently running"

    Note: This function references `https://github.com/tensorpack/tensorpack/blob/master/tensorpack/utils/logger.py#L93`

    Args:
        dir_name(str): log directory
        action(str): an action of ["k","d","q"] to be performed
            when the directory exists. Will ask user by default.
                "d": delete the directory. Note that the deletion may fail when
                the directory is used by tensorboard.
                "k": keep the directory. This is useful when you resume from a
                previous training and want the directory to look as if the
                training was not interrupted.
                Note that this option does not load old models or any other
                old states for you. It simply does nothing.

    Returns:
        dirname(str): log directory used in the global logging directory.
    """
    mod = sys.modules['__main__']
    if hasattr(mod, '__file__'):
        basename = os.path.basename(mod.__file__)
    else:
        basename = ''
    dirname = os.path.join('train_log', basename[:basename.rfind('.')])
    dirname = os.path.normpath(dirname)

    global LOG_DIR, _FILE_HANDLER
    if _FILE_HANDLER:
        # unload and close the old file handler, so that we may safely delete the logger directory
        _logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER

    def dir_nonempty(dirname):
        # If directory exists and nonempty (ignore hidden files), prompt for action
        return os.path.isdir(dirname) and len(
            [x for x in os.listdir(dirname) if x[0] != '.'])

    if dir_nonempty(dirname):
        if not action:
            _logger.warning("""\
Log directory {} exists! Use 'd' to delete it. """.format(dirname))
            _logger.warning("""\
If you're resuming from a previous run, you can choose to keep it.
Press any other key to exit. """)
        while not action:
            action = input("Select Action: k (keep) / d (delete) / q (quit):"
                           ).lower().strip()
        act = action
        if act == 'd':
            shutil.rmtree(dirname, ignore_errors=True)
            if dir_nonempty(dirname):
                shutil.rmtree(dirname, ignore_errors=False)
        elif act == 'n':
            dirname = dirname + _get_time_str()
            info("Use a new log directory {}".format(dirname))  # noqa: F821
        elif act == 'k':
            pass
        else:
            raise OSError("Directory {} exits!".format(dirname))
    LOG_DIR = dirname
    _makedirs(dirname)
    _set_file(os.path.join(dirname, 'log.log'))
    return dirname


def get_dir():
    return LOG_DIR


# Will save log to log_dir/main_file_name/log.log by default

mod = sys.modules['__main__']
if hasattr(mod, '__file__') and 'XPARL' not in os.environ:
    _logger.info("Argv: " + ' '.join(sys.argv))
