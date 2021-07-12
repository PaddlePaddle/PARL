#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from parameterized import parameterized, parameterized_class
from unittest import mock

# How to add new configuration: provide tuples with two arguments, where the first one represents the function to be mocked, and the second one is the mocked return.
env_config_for_test = parameterized_class(('to_mock_function', 'return_value'), \
        [('parl.utils.machine_info.is_gpu_available', False),
         ('parl.utils.machine_info.is_gpu_available', True)])

env_config = parameterized_class(('to_mock_function', 'return_value'), \
        [('parl.remote.remote_class_serialization.is_implemented_in_notebook', False), # mock that remote class is not implemented in the notebook
         ('parl.remote.remote_class_serialization.is_implemented_in_notebook', True)]) # mock that remote class is implemented in the notebook


class MockingEnv(unittest.TestCase):
    """ The class is the base class for tests under the remote module. It can provide different mocked environments for tests, and each test should inherit this class.
        Note that users **SHOULD NOT** override the setUp function, and they should implement _setUp for initialization instead.

    Usage: Users should inherit this class to implement the unit tests for remote modules. The class must be decorated with the env_config declared above. (e.g., @env_config_for_test at line 19)

    Example: Please refer to parl/remote/tests/mocking_env_test.py.
    """

    def patch(self, target, **kwargs):
        p = mock.patch(target, **kwargs)
        p.start()
        self.addCleanup(p.stop)

    def setUp(self):
        return_value = mock.Mock(return_value=self.return_value)
        self.patch(self.to_mock_function, new=return_value)
        if hasattr(self, '_setUp'):
            self._setUp()
