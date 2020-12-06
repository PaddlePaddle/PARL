#!/usr/bin/env python
# coding=utf8
# File: sac.py
import parl
import torch
import torch.nn.functional as F
import copy
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ['SAC']

epsilon = 1e-6

class SAC(parl.Algorithm):
  def __init__(self, actor, critic, max_action, alpha=0.2, gamma=None, tau=None, actor_lr=None, critic_lr=None):
    """ SAC algorithm
    Args:
        actor (parl.Model): forward network of actor.
        critic (patl.Model): forward network of the critic.
        max_action (float): the largest value that an action can be, env.action_space.high[0]
        alpha (float): Temperature parameter determines the relative importance of the entropy against the reward
        gamma (float): discounted factor for reward computation.
        tau (float): decay coefficient when updating the weights of self.target_model with self.model
        actor_lr (float): learning rate of the actor model
        critic_lr (float): learning rate of the critic model
    """
    assert isinstance(gamma, float)
    assert isinstance(tau, float)
    assert isinstance(actor_lr, float)
    assert isinstance(critic_lr, float)
    assert isinstance(alpha, float)
    self.max_action = max_action
    self.gamma = gamma
    self.tau = tau
    self.alpha = alpha
    self.actor = actor.to(device)
    self.critic = critic.to(device)
    self.target_critic = copy.deepcopy(self.critic)

    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

  def predict(self, obs):
    """ use actor model of self.policy to predict the action
    """
    mean, _ = self.actor(obs)
    mean = torch.tanh(mean) * self.max_action
    return mean

  def sample(self, obs):
    mean, log_std = self.actor(obs)
    std = torch.exp(log_std)
    normal = Normal(mean, std)
    x_t = normal.rsample()
    y_t = torch.tanh(x_t)
    action = y_t * self.max_action
    log_prob = normal.log_prob(x_t)
    log_prob -= torch.log(self.max_action * (1 - y_t * y_t) + epsilon)
    log_prob = log_prob.sum(dim=1, keepdim=True)
    return action, log_prob
  
  def learn(self, obs, act, reward, next_obs, terminal):
    actor_cost = self.actor_learn(obs)
    critic_cost = self.critic_learn(obs, act, reward, next_obs, terminal)
    self.sync_target()
    return critic_cost, actor_cost

  def actor_learn(self, obs):
    action, log_pi = self.sample(obs)
    q1_pred, q2_pred = self.critic(obs, action)
    q_pred = torch.min(q1_pred, q2_pred)
    #print("pi", log_pi.shape, "q_pred", q_pred.shape)
    policy_loss = (self.alpha * log_pi - q_pred).mean()
    self.actor_optimizer.zero_grad()
    policy_loss.backward()
    self.actor_optimizer.step()
    return policy_loss

  def critic_learn(self, obs, act, reward, next_obs, terminal):
    next_action, next_log_pi = self.sample(next_obs)
    q1_next_target, q2_next_target = self.target_critic(next_obs, next_action)
    min_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_pi
    target_Q = (reward + (1.0 - terminal) * self.gamma * min_next_target).detach()

    current_Q1, current_Q2 = self.critic(obs, act)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    critic_loss = critic_loss.mean()
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return critic_loss

  def sync_target(self, decay=None):
    if decay is None:
      decay = 1.0 - self.tau
    for param, target_param in zip(self.critic.parameters(),
                                   self.target_critic.parameters()):
        target_param.data.copy_((1 - decay) * param.data +
                                decay * target_param.data)
