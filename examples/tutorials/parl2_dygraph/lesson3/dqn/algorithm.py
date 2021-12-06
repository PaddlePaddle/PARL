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

#-*- coding: utf-8 -*-

import copy
import parl
import paddle


class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            gamma (float): reward的衰减因子
            lr (float): learning_rate，学习率.
        """
        # checks
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.parameters())  # 使用Adam优化器

    def predict(self, obs):
        """ 使用self.model的网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 获取Q预测值
        pred_values = self.model(obs)
        action_dim = pred_values.shape[-1]
        action = paddle.squeeze(action, axis=-1)
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = paddle.nn.functional.one_hot(
            action, num_classes=action_dim)
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        pred_value = pred_values * action_onehot
        #  ==> pred_value = [[3.9]]
        pred_value = paddle.sum(pred_value, axis=1, keepdim=True)

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        with paddle.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)
            target = reward + (1 - terminal) * self.gamma * max_v
        loss = self.mse_loss(pred_value, target)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model)
