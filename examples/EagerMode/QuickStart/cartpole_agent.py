import numpy as np
import paddle.fluid as fluid
from parl.utils import machine_info


class CartpoleAgent():
    def __init__(
            self,
            alg,
            obs_dim,
            act_dim,
    ):
        self.alg = alg
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.alg.predict(obs).numpy()
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.random.choice(self.act_dim, p=act_prob)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.alg.predict(obs).numpy()
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        cost = self.alg.learn(obs, act, reward)
        return cost
