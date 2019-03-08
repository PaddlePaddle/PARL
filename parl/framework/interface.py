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

__all__ = ['NetworkInterface']


class NetworkInterface(object):
    """ Interface of Network.
    """

    def sync_params_to(self, target_net, **kwargs):
        """ Synchronize parameters of this Network to target Network
        
        Args:
            target_net: target Network
        """
        raise NotImplementedError

    def get_params(self, **kwargs):
        """ Get numpy array of parameters in this Network
        
        Returns:
            List of numpy array.
        """
        raise NotImplementedError

    def set_params(self, params, **kwargs):
        """ Set parameters in this Network with params
        
        Args:
            params: List of numpy array.
        """
        raise NotImplementedError
