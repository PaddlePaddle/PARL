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

import threading

__all__ = ['global_model_helper']


class ModelHelper(object):
    """Model id helper.

    This helper is used to help `parl.Model` generate a new model id
    or register a given model id in a thread-safe way.
    """

    def __init__(self):
        self._registered_ids = set([])
        self.index = 0
        self.lock = threading.Lock()

    def generate_model_id(self):
        """Generate a unique model_id in a thread-safe way.

        Returns:
            String of model id.
        """
        self.lock.acquire()
        model_id = 'parl_model_{}'.format(self.index)
        while model_id in self._registered_ids:
            self.index += 1
            model_id = 'parl_model_{}'.format(self.index)
        self._registered_ids.add(model_id)
        self.index += 1
        self.lock.release()
        return model_id

    def register_model_id(self, model_id):
        """Register given model id in a thread-safe way.
        
        Raises:
            AssertionError: if the model id is already used.
        """
        model_id_used = False
        self.lock.acquire()
        if model_id in self._registered_ids:
            model_id_used = True
        else:
            self._registered_ids.add(model_id)
        self.lock.release()

        assert not model_id_used, "model id `{}` has been used before, please try another model_id.".format(
            model_id)


global_model_helper = ModelHelper()
