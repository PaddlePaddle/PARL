import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers


class PolicyGradient():
    def __init__(self, model, lr):
        self.model = model
        self.optimizer = fluid.optimizer.Adam(learning_rate=lr)

    def predict(self, obs):
        obs = fluid.dygraph.to_variable(obs)
        obs = layers.cast(obs, dtype='float32')
        return self.model(obs)

    def learn(self, obs, action, reward):
        obs = fluid.dygraph.to_variable(obs)
        obs = layers.cast(obs, dtype='float32')
        act_prob = self.model(obs)
        action = fluid.dygraph.to_variable(action)
        reward = fluid.dygraph.to_variable(reward)

        log_prob = layers.cross_entropy(act_prob, action)
        # cost = np.squeeze(log_prob) * reward
        cost = log_prob * reward
        cost = layers.cast(cost, dtype='float32')
        cost = layers.reduce_mean(cost)
        cost.backward()
        self.optimizer.minimize(cost)
        self.model.clear_gradients()
        return cost
