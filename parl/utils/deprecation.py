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
Reference:
    https://github.com/briancurtin/deprecation
"""

__all__ = ['deprecated']

import functools
import textwrap
import warnings

warnings.simplefilter('default')


class CustomDeprecationWarning(DeprecationWarning):
    def __init__(self,
                 function,
                 deprecated_in,
                 removed_in,
                 replace_function=None):
        """ 
        Args:
            function (String): The function being deprecated.
            deprecated_in (String): The version that ``function`` is deprecated in
            removed_in (String): The version that ``function`` gets removed in
            replace_function (String): The replacement function of deprecated function.
        """

        self.function = function
        self.deprecated_in = deprecated_in
        self.removed_in = removed_in
        self.replace_function = replace_function
        super(CustomDeprecationWarning, self).__init__(
            function, deprecated_in, removed_in, replace_function)

    def __str__(self):
        warn_string = '[PARL] API `{}` is deprecated since version {} and will be removed in version {}'.format(
            self.function, self.deprecated_in, self.removed_in)
        if self.replace_function is not None:
            warn_string += ", please use `{}` instead.".format(
                self.replace_function)
        else:
            warn_string += "."
        return warn_string


def deprecated(deprecated_in, removed_in, replace_function=None):
    """Decorator of decarated function.

    Args:
        function (String): The function being deprecated.
        deprecated_in (String): The version that ``function`` is deprecated in
        removed_in (String): The version that ``function`` gets removed in
        replace_function (String): The replacement function of deprecated function.
    """

    def _function_wrapper(function):
        existing_docstring = function.__doc__ or ""

        deprecated_doc = '.. deprecated:: {}\n    This will be removed in {}'.format(
            deprecated_in, removed_in)
        if replace_function is not None:
            deprecated_doc += ", please use `{}` instead.".format(
                replace_function)
        else:
            deprecated_doc += "."

        # split docstring at first occurrence of newline
        string_list = existing_docstring.split("\n", 1)

        if len(string_list) > 1:
            # in-place dedent docstring content
            string_list[1] = textwrap.dedent(string_list[1])

            # we need another newline
            string_list.insert(1, "\n")

        # insert deprecation note and dual newline
        string_list.insert(1, deprecated_doc)
        string_list.insert(1, "\n\n")

        function.__doc__ = "".join(string_list)

        @functools.wraps(function)
        def _inner(*args, **kwargs):
            the_warning = CustomDeprecationWarning(
                function.__name__, deprecated_in, removed_in, replace_function)
            warnings.warn(
                the_warning, category=CustomDeprecationWarning, stacklevel=2)

            return function(*args, **kwargs)

        return _inner

    return _function_wrapper
