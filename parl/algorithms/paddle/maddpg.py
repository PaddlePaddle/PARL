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

import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from parl.utils.utils import check_model_method
from copy import deepcopy

__all__ = ['MADDPG']

from parl.core.paddle.policy_distribution import SoftCategoricalDistribution
from parl.core.paddle.policy_distribution import SoftMultiCategoricalDistribution


def SoftPDistribution(logits, act_space):
    """ Select SoftCategoricalDistribution or SoftMultiCategoricalDistribution according to act_space.

    Args:
        logits (paddle tensor): the output of policy model
        act_space: action space, must be gym.spaces.Discrete or multiagent.multi_discrete.MultiDiscrete

    Returns:
        instance of SoftCategoricalDistribution or SoftMultiCategoricalDistribution
    """
    # is instance of gym.spaces.Discrete
    if (hasattr(act_space, 'n')):
        return SoftCategoricalDistribution(logits)
    # is instance of multiagent.multi_discrete.MultiDiscrete
    elif (hasattr(act_space, 'num_discrete_space')):
        return SoftMultiCategoricalDistribution(logits, act_space.low,
                                                act_space.high)
    else:
        raise AssertionError("act_space must be instance of \
            gym.spaces.Discrete or multiagent.multi_discrete.MultiDiscrete")


class MADDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """  MADDPG algorithm

        Args:
            model (parl.Model): forward network of actor and critic.
                                The function get_actor_params() of model should be implemented.
            agent_index (int): index of agent, in multiagent env
            act_space (list): action_space, gym space
            gamma (float): discounted factor for reward computation.
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            critic_lr (float): learning rate of the critic model
            actor_lr (float): learning rate of the actor model
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(agent_index, int)
        assert isinstance(act_space, list)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(model)
        self.sync_target(0)

        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=self.actor_lr,
            parameters=self.model.get_actor_params(),
            grad_clip=nn.ClipGradByNorm(clip_norm=0.5))
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=self.critic_lr,
            parameters=self.model.get_critic_params(),
            grad_clip=nn.ClipGradByNorm(clip_norm=0.5))

    def predict(self, obs, use_target_model=False):
        """ use the policy model to predict actions
        
        Args:
            obs (paddle tensor): observation, shape([B] + shape of obs_n[agent_index])
            use_target_model (bool): use target_model or not
    
        Returns:
            act (paddle tensor): action, shape([B] + shape of act_n[agent_index])
        """
        if use_target_model:
            policy = self.target_model.policy(obs)
        else:
            policy = self.model.policy(obs)
        action = SoftPDistribution(
            logits=policy,
            act_space=self.act_space[self.agent_index]).sample()
        return action

    def Q(self, obs_n, act_n, use_target_model=False):
        """ use the value model to predict Q values
        
        Args: 
            obs_n (list of paddle tensor): all agents' observation, len(agent's num) + shape([B] + shape of obs_n)
            act_n (list of paddle tensor): all agents' action, len(agent's num) + shape([B] + shape of act_n)
            use_target_model (bool): use target_model or not

        Returns:
            Q (paddle tensor): Q value of this agent, shape([B])
        """
        if use_target_model:
            return self.target_model.value(obs_n, act_n)
        else:
            return self.model.value(obs_n, act_n)

    def learn(self, obs_n, act_n, target_q):
        """ update actor and critic model with MADDPG algorithm
        """
        actor_cost = self._actor_learn(obs_n, act_n)
        critic_cost = self._critic_learn(obs_n, act_n, target_q)
        self.sync_target()
        return critic_cost

    def _actor_learn(self, obs_n, act_n):
        i = self.agent_index

        this_policy = self.model.policy(obs_n[i])
        sample_this_action = SoftPDistribution(
            logits=this_policy,
            act_space=self.act_space[self.agent_index]).sample()

        # action_input_n = deepcopy(act_n)
        action_input_n = act_n + []
        action_input_n[i] = sample_this_action
        eval_q = self.Q(obs_n, action_input_n)
        act_cost = paddle.mean(-1.0 * eval_q)

        act_reg = paddle.mean(paddle.square(this_policy))

        cost = act_cost + act_reg * 1e-3

        self.actor_optimizer.clear_grad()
        cost.backward()
        self.actor_optimizer.step()
        return cost

    def _critic_learn(self, obs_n, act_n, target_q):
        pred_q = self.Q(obs_n, act_n)
        cost = paddle.mean(F.square_error_cost(pred_q, target_q))

        self.critic_optimizer.clear_grad()
        cost.backward()
        self.critic_optimizer.step()
        return cost

    def sync_target(self, decay=None):
        """ update the target network with the training network

        Args:
            decay(float): the decaying factor while updating the target network with the training network. 
                        0 represents the **assignment**. None represents updating the target network slowly that depends on the hyperparameter `tau`.
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
