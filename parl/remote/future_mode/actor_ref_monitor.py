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

import sys

from parl.utils import _IS_PY2


class ActorRefMonitor(object):
    def __init__(self, actor):
        """Monitor of actor reference count, which is used in the future mode.
        It checks if we should release the CPU resource of the actor or not.

        Args:
            actor(instance of ProxyWrapperNoWait)
        """
        self.actor_ref = actor

        # @TODO(zenghsh3): hardcoded
        self.actor_deleted_refcount = 5
        if _IS_PY2:
            self.actor_deleted_refcount = 6

    def is_deleted(self):
        """Check wheter the actor is deleted or out of scope
        if `self.actor_ref` is None, will always return False.

        """
        cur_refcount = sys.getrefcount(self.actor_ref)
        return cur_refcount == self.actor_deleted_refcount
