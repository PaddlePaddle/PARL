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


class UtilsError(Exception):
    """
    Super class of exceptions in utils module.
    """

    def __init__(self, error_info):
        self.error_info = '[PARL Utils Error]: {}'.format(error_info)


class SerializeError(UtilsError):
    """
    Serialize error raised by pyarrow.
    """

    def __init__(self, error_info):
        error_info = (
            'Serialize error, you may have provided an object that cannot be '
            + 'serialized by pyarrow. Detailed error:\n{}'.format(error_info))
        super(SerializeError, self).__init__(error_info)

    def __str__(self):
        return self.error_info


class DeserializeError(UtilsError):
    """
    Deserialize error raised by pyarrow.
    """

    def __init__(self, error_info):
        error_info = (
            'Deserialize error, you may have provided an object that cannot be '
            +
            'deserialized by pyarrow. Detailed error:\n{}'.format(error_info))
        super(DeserializeError, self).__init__(error_info)

    def __str__(self):
        return self.error_info
