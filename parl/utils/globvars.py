#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from parl.utils import logger
import argparse

__all__ = ['global_config', 'GlobalConfig']

import types
CONFIG = types.SimpleNamespace

class GlobalConfig(CONFIG):
    """
    A namespace to store global variables.
    Example:
    .. code-block:: none
        from parl.utils import global_config as config
        config.batch_size = 18
        config.learning_rate = 3e-4
        config.load_argument(parser.parse_args())
    """
    def load_argument(self, args):
        """
        Add the content of :class:`argparse.Namespace` to this ns.
        Args:
            args (argparse.Namespace): arguments
        """
        assert isinstance(args, argparse.Namespace), type(args)
        for k in vars(args):
            if hasattr(self, k):
                logger.warn("Attribute {} in global_config will be overwritten!")
            setattr(self, k, getattr(args, k))

global_config = GlobalConfig()
