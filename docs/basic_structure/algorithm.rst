Algorithm (*Backward Part*)
=============================

Methods
---------
1. define_predict(self, obs)

    Use method policy( ) from Agent to predict the probabilities of actions.

2. define_learn(self, obs, action, reward, next_obs, terminal)

    Define loss function and optimizer here to update the policy model.

An Example
-----------



.. code-block:: python
    :linenos:

    # From https://github.com/PaddlePaddle/PARL/blob/develop/parl/algorithms/policy_gradient.py

    class PolicyGradient(Algorithm):
    def __init__(self, model, hyperparas):
        Algorithm.__init__(self, model, hyperparas)
        self.model = model
        self.lr = hyperparas['lr']

    def define_predict(self, obs):
        """ use policy model self.model to predict the action probability
        """
        return self.model.policy(obs)

    def define_learn(self, obs, action, reward):
        """ update policy model self.model with policy gradient algorithm
        """
        act_prob = self.model.policy(obs)
        log_prob = layers.cross_entropy(act_prob, action)
        cost = log_prob * reward
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost
