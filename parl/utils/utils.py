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

__all__ = ['has_func']


def has_func(obj, fun):
    """check if a class has specified function: https://stackoverflow.com/a/5268474

    Args:
        obj: the class to check
        fun: specified function to check
    Returns:
        A bool to indicate if obj has funtion "fun"
    """
    check_fun = getattr(obj, fun, None)
    return callable(check_fun)
