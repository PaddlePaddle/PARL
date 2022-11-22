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

import torch
import numpy as np
import parl
import time

__all__ = ['DecisionTransformer']


class DecisionTransformer(parl.Algorithm):
    def __init__(self, model, learning_rate, warmup_steps, weight_decay):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))

    def predict(self, states, actions, rewards, returns_to_go, timesteps,
                **kwargs):
        return self.model.get_action(states, actions, rewards, returns_to_go,
                                     timesteps)

    def learn(self, states, actions, rewards, dones, rtg, timesteps,
              attention_mask):
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(
            -1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(
            -1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_target)**2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()
        self.scheduler.step()

        return loss
