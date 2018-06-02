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

import parl.layers as layers
from parl.framework.algorithm import Model
import parl.framework.model_helpers as mh
from parl.layers import common_functions as comf


class SimpleModelDet(Model):
    def __init__(self, dims, mlp_layer_confs):
        super(SimpleModelDet, self).__init__()
        self.dims = dims
        self.mlp = [layers.fc(**c) for c in mlp_layer_confs]

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("continuous_action", dict(shape=[self.dims]))]

    def policy(self, inputs, states):
        hidden = comf.feedforward(inputs.values()[0], self.mlp)
        return dict(continuous_action=mh.Deterministic(hidden)), states


class SimpleModelAC(Model):
    def __init__(self, dims, num_actions, mlp_layer_confs):
        super(SimpleModelAC, self).__init__()
        self.dims = dims
        self.mlp = [layers.fc(**c) for c in mlp_layer_confs]
        assert mlp_layer_confs[-1]["act"] == "softmax"
        self.value_layer = layers.fc(size=1)
        self.exploration = 0.01

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def _perceive(self, inputs, states):
        return comf.feedforward(inputs.values()[0], self.mlp[:-1])

    def policy(self, inputs, states):
        dist = mh.discrete_dist(
            self._perceive(inputs, states), self.mlp[-1:], self.exploration)
        return dict(action=dist), states

    def value(self, inputs, states):
        return dict(v_value=self.value_layer(self._perceive(inputs, states)))


class SimpleModelQ(Model):
    def __init__(self, dims, num_actions, mlp_layer_confs):
        super(SimpleModelQ, self).__init__()
        self.dims = dims
        self.num_actions = num_actions
        self.mlp = [layers.fc(**c) for c in mlp_layer_confs]
        assert "act" not in mlp_layer_confs[-1], "should be linear act"
        self.exploration = 0.2

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def policy(self, inputs, states):
        values = self.value(inputs, states)
        q_value = values["q_value"]
        return dict(action=mh.q_discrete_dist(q_value,
                                              self.exploration)), states

    def value(self, inputs, states):
        return dict(q_value=comf.feedforward(inputs.values()[0], self.mlp))
