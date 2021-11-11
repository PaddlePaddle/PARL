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

import os
import parl
import torch


class Agent(parl.Agent):
    """Agent.
    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
    """

    def __init__(self, algorithm):
        super().__init__(algorithm)

    def learn(self, obs, act, value, returns, log_prob, adv):
        """Updating network
        Args:
            obs (np.array): representation of current observation
            act (np.array): current action
            value (np.array): state value
            returns (np.array): discounted return
            log_prob (np.array): the log probabilities of action
            adv (np.array): advantage value
        """

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.int64)
        value = torch.tensor(value, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        log_prob = torch.tensor(log_prob, dtype=torch.float32)
        adv = torch.tensor(adv, dtype=torch.float32)

        value_loss, action_loss, entropy = self.alg.learn(
            obs, act, value, returns, log_prob, adv)

        return value_loss, action_loss, entropy

    def predict(self, state):
        """Predict action
        Args:
            state (np.array): representation of current state 

        Return:
            action (np.array): action to be executed
        """

        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():

            action = self.alg.predict(state_tensor).cpu().numpy()

        return action

    def sample(self, state):
        """Sampling action
        Args:
            state (np.array): representation of current state 
        Return:
            action (np.array): action to be executed
        """

        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            value, action, action_log_prob = self.alg.sample(state_tensor)

        value = value.detach().cpu().numpy().flatten()
        action = action.detach().cpu().numpy()
        action_log_prob = action_log_prob.detach().cpu().numpy()

        return value, action, action_log_prob

    def value(self, state):
        """Predict the critic value
        Args:
            state (np.array): representation of current state 
        Return:
            value (np.array): state value
        """

        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():

            value = self.alg.value(state_tensor)

        return value.detach().cpu().numpy()

    def save(self, model_path):
        """Save Model
        Args:
            model_path (str): the path to save model
        """
        sep = os.sep
        dirname = sep.join(model_path.split(sep)[:-1])
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)
        model_dict = {}
        model_dict["critic"] = self.alg.critic.state_dict()
        model_dict["actor"] = self.alg.actor.state_dict()
        model_dict["critic_optimizer"] = self.alg.critic_optimizer.state_dict()
        model_dict["actor_optimizer"] = self.alg.actor_optimizer.state_dict()

        torch.save(model_dict, model_path)

    def restore(self, model_path):
        """Restore model
        Args:
            model_path (str): the path to restore model
        """
        model_dict = torch.load(model_path)
        self.alg.critic.load_state_dict(model_dict["critic"])
        self.alg.actor.load_state_dict(model_dict["actor"])
        self.alg.critic_optimizer.load_state_dict(
            model_dict["critic_optimizer"])
        self.alg.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])
