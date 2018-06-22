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

from parl.framework.computation_task import ComputationTask
from multiprocessing import Queue
from threading import Thread


class Manager(object):
    def __init__(self, ct_settings):
        # TODO: a centralized logger
        self.agents = []
        self.cts = {}
        self.wrappers = {}
        for name, setting in ct_settings.iteritems():
            self.cts[name] = ComputationTask(name, **setting)
            self.wrappers[name] = self.cts[name].wrapper

    def add_agent(self, agent):
        agent.id = len(self.agents)
        self.agents.append(agent)
        for name, wrapper in self.wrappers.iteritems():
            agent.add_helper(wrapper.create_helper(agent.id))

    def remove_agent(self):
        self.agents[-1].exit_flag.value = 1
        self.agents[-1].join()
        self.agents.pop()

    def start(self):
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
