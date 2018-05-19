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

class LastElementError(Exception):
    """
    Raised when the last element or an element with non-zero game status is
    sampled.

    Attributes:
        message(string): error message
    """

    def __init__(self, idx, status):
        self.message = 'The element at {}' .format(idx)
        if status:
            self.message += ' has game status: {}'.format(status)
        else:
            self.message += ' is the last element.'


def check_error(error_type, *args):
    """
    Check if there is a specific error and raise the corresponding Exception
    if so.

    Args:
        error_type(string): the type of error to check
        args: variable-length argument list used to check the error's existence
    """
    if error_type == 'TypeError':
        if args[0].__name__ != args[1].__name__:
            raise TypeError('{} expected, but {} given.'
                            .format(args[0].__name__, args[1].__name__))
    elif error_type == 'LastElementError':
        if args[0]:
            raise LastElementError(args[1], args[2])
    elif error_type == 'ValueError':
        if args[0] == '==' and args[1] != args[2] or \
           args[0] == '>' and args[1] <= args[2] or \
           args[0] == '>=' and args[1] < args[2] or \
           args[0] == '<' and args[1] >= args[2] or \
           args[0] == '<=' and args[1] > args[2] or \
           args[0] == '!=' and args[1] == args[2]:
            raise ValueError('{} {} {} not holds'.format(args[1], args[0], args[2]))
