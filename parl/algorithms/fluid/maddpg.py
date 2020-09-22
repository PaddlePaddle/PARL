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

import warnings
warnings.simplefilter('default')

from parl.core.fluid import layers
from copy import deepcopy
from paddle import fluid
from parl.core.fluid.algorithm import Algorithm

__all__ = ['MADDPG']

from parl.core.fluid.policy_distribution import SoftCategoricalDistribution
from parl.core.fluid.policy_distribution import SoftMultiCategoricalDistribution


def SoftPDistribution(logits, act_space):
    """Args:
            logits: the output of policy model
            act_space: action space, must be gym.spaces.Discrete or multiagent.multi_discrete.MultiDiscrete

        Return:
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


class MADDPG(Algorithm):
    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=None,
                 tau=None,
                 lr=None,
                 actor_lr=None,
                 critic_lr=None):
        """  MADDPG algorithm
        
        Args:
            model (parl.Model): forward network of actor and critic.
                                The function get_actor_params() of model should be implemented.
            agent_index: index of agent, in multiagent env
            act_space: action_space, gym space
            gamma (float): discounted factor for reward computation.
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            lr (float): learning rate, lr will be assigned to both critic_lr and actor_lr
            critic_lr (float): learning rate of the critic model
            actor_lr (float): learning rate of the actor model
        """

        assert isinstance(agent_index, int)
        assert isinstance(act_space, list)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        # compatible upgrade of lr
        if lr is None:
            assert isinstance(actor_lr, float)
            assert isinstance(critic_lr, float)
        else:
            assert isinstance(lr, float)
            assert actor_lr is None, 'no need to set `actor_lr` if `lr` is not None'
            assert critic_lr is None, 'no need to set `critic_lr` if `lr` is not None'
            critic_lr = lr
            actor_lr = lr
            warnings.warn(
                "the `lr` argument of `__init__` function in `parl.Algorithms.MADDPG` is deprecated \
                    since version 1.4 and will be removed in version 2.0. \
                    Recommend to use `actor_lr` and `critic_lr`. ",
                DeprecationWarning,
                stacklevel=2)
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(model)

    def predict(self, obs):
        """ input:  
                obs: observation, shape([B] + shape of obs_n[agent_index])
            output: 
                act: action, shape([B] + shape of act_n[agent_index])
        """
        this_policy = self.model.policy(obs)
        this_action = SoftPDistribution(
            logits=this_policy,
            act_space=self.act_space[self.agent_index]).sample()
        return this_action

    def predict_next(self, obs):
        """ input:  observation, shape([B] + shape of obs_n[agent_index])
            output: action, shape([B] + shape of act_n[agent_index])
        """
        next_policy = self.target_model.policy(obs)
        next_action = SoftPDistribution(
            logits=next_policy,
            act_space=self.act_space[self.agent_index]).sample()
        return next_action

    def Q(self, obs_n, act_n):
        """ input:  
                obs_n: all agents' observation, shape([B] + shape of obs_n)
            output: 
                act_n: all agents' action, shape([B] + shape of act_n)
        """
        return self.model.value(obs_n, act_n)

    def Q_next(self, obs_n, act_n):
        """ input:  
                obs_n: all agents' observation, shape([B] + shape of obs_n)
            output: 
                act_n: all agents' action, shape([B] + shape of act_n)
        """
        return self.target_model.value(obs_n, act_n)

    def learn(self, obs_n, act_n, target_q):
        """ update actor and critic model with MADDPG algorithm
        """
        actor_cost = self._actor_learn(obs_n, act_n)
        critic_cost = self._critic_learn(obs_n, act_n, target_q)
        return critic_cost

    def _actor_learn(self, obs_n, act_n):
        i = self.agent_index
        this_policy = self.model.policy(obs_n[i])
        sample_this_action = SoftPDistribution(
            logits=this_policy,
            act_space=self.act_space[self.agent_index]).sample()

        action_input_n = act_n + []
        action_input_n[i] = sample_this_action
        eval_q = self.Q(obs_n, action_input_n)
        act_cost = layers.reduce_mean(-1.0 * eval_q)

        act_reg = layers.reduce_mean(layers.square(this_policy))

        cost = act_cost + act_reg * 1e-3

        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByNorm(clip_norm=0.5),
            param_list=self.model.get_actor_params())

        optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
        return cost

    def _critic_learn(self, obs_n, act_n, target_q):
        pred_q = self.Q(obs_n, act_n)
        cost = layers.reduce_mean(layers.square_error_cost(pred_q, target_q))

        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByNorm(clip_norm=0.5),
            param_list=self.model.get_critic_params())

        optimizer = fluid.optimizer.AdamOptimizer(self.critic_lr)
        optimizer.minimize(cost, parameter_list=self.model.get_critic_params())
        return cost

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
