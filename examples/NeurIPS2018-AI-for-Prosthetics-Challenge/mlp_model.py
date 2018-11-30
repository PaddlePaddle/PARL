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

import numpy as np
import parl.layers as layers
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from parl.framework.model_base import Model


class ActorModel(Model):
    def __init__(self,
                 obs_dim,
                 vel_obs_dim,
                 act_dim,
                 stage_name=None,
                 model_id=0,
                 shared=False):
        super(ActorModel, self).__init__()
        hid0_size = 800
        hid1_size = 400
        hid2_size = 200
        vel_hid0_size = 200
        vel_hid1_size = 400

        self.obs_dim = obs_dim
        self.vel_obs_dim = vel_obs_dim

        # buttom layers
        if shared:
            scope_name = 'policy_shared'
        else:
            scope_name = 'policy_identity_{}'.format(model_id)
        if stage_name is not None:
            scope_name = '{}_{}'.format(stage_name, scope_name)

        self.fc0 = layers.fc(
            size=hid0_size,
            act='tanh',
            param_attr=ParamAttr(name='{}/h0/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/h0/b'.format(scope_name)))
        self.fc1 = layers.fc(
            size=hid1_size,
            act='tanh',
            param_attr=ParamAttr(name='{}/h1/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/h1/b'.format(scope_name)))
        self.vel_fc0 = layers.fc(
            size=vel_hid0_size,
            act='tanh',
            param_attr=ParamAttr(name='{}/vel_h0/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/vel_h0/b'.format(scope_name)))
        self.vel_fc1 = layers.fc(
            size=vel_hid1_size,
            act='tanh',
            param_attr=ParamAttr(name='{}/vel_h1/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/vel_h1/b'.format(scope_name)))

        # top layers
        scope_name = 'policy_identity_{}'.format(model_id)
        if stage_name is not None:
            scope_name = '{}_{}'.format(stage_name, scope_name)

        self.fc2 = layers.fc(
            size=hid2_size,
            act='tanh',
            param_attr=ParamAttr(name='{}/h2/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/h2/b'.format(scope_name)))
        self.fc3 = layers.fc(
            size=act_dim,
            act='tanh',
            param_attr=ParamAttr(name='{}/means/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/means/b'.format(scope_name)))

    def predict(self, obs):
        real_obs = layers.slice(
            obs, axes=[1], starts=[0], ends=[self.obs_dim - self.vel_obs_dim])
        vel_obs = layers.slice(
            obs, axes=[1], starts=[-self.vel_obs_dim], ends=[self.obs_dim])
        hid0 = self.fc0(real_obs)
        hid1 = self.fc1(hid0)
        vel_hid0 = self.vel_fc0(vel_obs)
        vel_hid1 = self.vel_fc1(vel_hid0)
        concat = layers.concat([hid1, vel_hid1], axis=1)
        hid2 = self.fc2(concat)
        means = self.fc3(hid2)
        return means


class CriticModel(Model):
    def __init__(self,
                 obs_dim,
                 vel_obs_dim,
                 act_dim,
                 stage_name=None,
                 model_id=0,
                 shared=False):
        super(CriticModel, self).__init__()
        hid0_size = 800
        hid1_size = 400
        vel_hid0_size = 200
        vel_hid1_size = 400

        self.obs_dim = obs_dim
        self.vel_obs_dim = vel_obs_dim

        # buttom layers
        if shared:
            scope_name = 'critic_shared'
        else:
            scope_name = 'critic_identity_{}'.format(model_id)
        if stage_name is not None:
            scope_name = '{}_{}'.format(stage_name, scope_name)

        self.fc0 = layers.fc(
            size=hid0_size,
            act='selu',
            param_attr=ParamAttr(name='{}/w1/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/w1/b'.format(scope_name)))
        self.fc1 = layers.fc(
            size=hid1_size,
            act='selu',
            param_attr=ParamAttr(name='{}/h1/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/h1/b'.format(scope_name)))
        self.vel_fc0 = layers.fc(
            size=vel_hid0_size,
            act='selu',
            param_attr=ParamAttr(name='{}/vel_h0/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/vel_h0/b'.format(scope_name)))
        self.vel_fc1 = layers.fc(
            size=vel_hid1_size,
            act='selu',
            param_attr=ParamAttr(name='{}/vel_h1/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/vel_h1/b'.format(scope_name)))
        self.act_fc0 = layers.fc(
            size=hid1_size,
            act='selu',
            param_attr=ParamAttr(name='{}/a1/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/a1/b'.format(scope_name)))

        # top layers
        scope_name = 'critic_identity_{}'.format(model_id)
        if stage_name is not None:
            scope_name = '{}_{}'.format(stage_name, scope_name)

        self.fc2 = layers.fc(
            size=hid1_size,
            act='selu',
            param_attr=ParamAttr(name='{}/h3/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/h3/b'.format(scope_name)))
        self.fc3 = layers.fc(
            size=1,
            act='selu',
            param_attr=ParamAttr(name='{}/value/W'.format(scope_name)),
            bias_attr=ParamAttr(name='{}/value/b'.format(scope_name)))

    def predict(self, obs, action):
        real_obs = layers.slice(
            obs, axes=[1], starts=[0], ends=[self.obs_dim - self.vel_obs_dim])
        vel_obs = layers.slice(
            obs, axes=[1], starts=[-self.vel_obs_dim], ends=[self.obs_dim])
        hid0 = self.fc0(real_obs)
        hid1 = self.fc1(hid0)
        vel_hid0 = self.vel_fc0(vel_obs)
        vel_hid1 = self.vel_fc1(vel_hid0)
        a1 = self.act_fc0(action)
        concat = layers.concat([hid1, a1, vel_hid1], axis=1)
        hid2 = self.fc2(concat)
        V = self.fc3(hid2)
        V = layers.squeeze(V, axes=[1])
        return V
