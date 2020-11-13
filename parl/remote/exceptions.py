#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


class ResourceError(Exception):
    """
    No available cpu resources error.
    """

    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class RemoteError(Exception):
    """
    Super class of exceptions in remote module.
    """

    def __init__(self, func_name, error_info):
        self.error_info = "[PARL remote error when calling " +\
            "function `{}`]:\n{}".format(func_name, error_info)

    def __str__(self):
        return self.error_info


class RemoteSerializeError(RemoteError):
    """
    Serialize error from remote
    """

    def __init__(self, func_name, error_info):
        super(RemoteSerializeError, self).__init__(func_name, error_info)

    def __str__(self):
        return self.error_info


class RemoteDeserializeError(RemoteError):
    """
    Deserialize error from remote
    """

    def __init__(self, func_name, error_info):
        super(RemoteDeserializeError, self).__init__(func_name, error_info)

    def __str__(self):
        return self.error_info


class RemoteAttributeError(RemoteError):
    """
    Attribute error from remote
    """

    def __init__(self, func_name, error_info):
        super(RemoteAttributeError, self).__init__(func_name, error_info)

    def __str__(self):
        return self.error_info


class FutureGetRepeatedlyError(Exception):
    """
    Calling the get function of `FutureObject` repeatedly.
    """

    def __init__(self):
        self.error_info = "The `get` function of `FutureObject` has been called, you cannot call the `get` function repeatedly."

    def __str__(self):
        return self.error_info


class AsyncFunctionError(Exception):
    """
    Error raised when calling async function.
    """

    def __init__(self, func_name):
        self.error_info = "There is an error raised when calling the async function `{}`.\n".format(func_name) + \
                "You can see the detailed error message above, which is printed by another thread."

    def __str__(self):
        return self.error_info
