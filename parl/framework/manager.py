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

from multiprocessing import Queue
from threading import Thread
from parl.common import Communicator


class Manager(object):
    def __init__(self, wrapper_creators, helper_creators):
        # TODO: a centralized logger
        self.agents = []
        self.wrappers = {}
        for name, creator in wrapper_creators.iteritems():
            self.wrappers[name] = creator(name)
        self.helper_creators = helper_creators

    def add_agent(self, agent):
        agent.id = len(self.agents)
        self.agents.append(agent)
        for name, wrapper in self.wrappers.iteritems():
            comm = wrapper.create_communicator(agent.id)
            agent.add_helper(self.helper_creators[name](name, comm))

    def remove_agent(self):
        self.agents[-1].exit_flag.value = 1
        self.agents[-1].join()
        self.agents.pop()

    def run(self):
        # TODO: logger thread
        for wrapper in self.wrappers.values():
            wrapper.run()
        for agent in self.agents:
            agent.start()

        while self.agents:
            self.agents[-1].join()
            self.agents.pop()
        for wrapper in self.wrappers.values():
            wrapper.stop()
