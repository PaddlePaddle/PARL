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

__all__ = ['set_dir', 'get_dir', 'set_level']

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
    global _FILE_HANDLER
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
    # To set level, need create new handler
    set_dir(get_dir())
    _logger.setLevel(level)


def set_dir(dirname):
    global LOG_DIR, _FILE_HANDLER
    if _FILE_HANDLER:
        # unload and close the old file handler, so that we may safely delete the logger directory
        _logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER

    if not os.path.isdir(dirname):
        _makedirs(dirname)
    LOG_DIR = dirname
    _set_file(os.path.join(dirname, 'log.log'))


def get_dir():
    return LOG_DIR


# Will save log to log_dir/main_file_name/log.log by default
mod = sys.modules['__main__']
if hasattr(mod, '__file__'):
    basename = os.path.basename(mod.__file__)
    auto_dirname = os.path.join('log_dir', basename[:basename.rfind('.')])
    shutil.rmtree(auto_dirname, ignore_errors=True)
    set_dir(auto_dirname)
    _logger.info("Argv: " + ' '.join(sys.argv))
