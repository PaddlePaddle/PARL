import torch
import parl

# torch use full CPU by default, which will decrease the performance. Use one thread for one actor here.
torch.set_num_threads(1)


class Agent(parl.Agent):
    def __init__(self, algorithm, config):
        super(Agent, self).__init__(algorithm)
        self.obs_shape = config['obs_shape']

    def sample(self, obs):
        sample_actions, values = self.alg.sample(obs)
        return sample_actions, values

    def predict(self, obs):
        predict_actions = self.alg.predict(obs)
        return predict_actions

    def value(self, obs):
        values = self.alg.value(obs)
        return values

    def learn(self, obs, actions, advantages, target_values):
        total_loss, pi_loss, vf_losss, entropy, lr, entropy_coeff = self.alg.learn(
            obs, actions, advantages, target_values)

        return total_loss, pi_loss, vf_losss, entropy, lr, entropy_coeff

    # def set_weights(self, weights):
    #     self.alg.set_weights(weights)

    # def get_weights(self):
    #     return self.alg.get_weights()
