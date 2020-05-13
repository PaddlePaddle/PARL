Create Customized Algorithms
===============================

Goal of this tutorial:

- Learn how to implement your own algorithms.


Overview
-----------

To build a new algorithm, you need to inherit class ``parl.Algorithm``
and implement three basic functions: ``sample``, ``predict`` and ``learn``.


Methods
-----------

- ``__init__``

  As algorithms update weights of the models, this method needs to define some models inherited from ``parl.Model``, like ``self.model`` in this example.
  You can also set some hyperparameters in this method, like learning_rate, reward_decay and action_dimension,
  which might be used in the following steps.

- ``predict``

  This function defines how to choose actions. For instance, you can use a policy model to predict actions. 

- ``sample``

  Based on ``predict`` method, ``sample`` generates actions with noises. Use this method to do exploration if needed.

- ``learn``

  Define loss function in ``learn`` method, which will be used to update weights of ``self.model``.


Example: DQN
--------------

This example shows how to implement DQN algorithm based on class ``parl.Algorithm`` according to the steps mentioned above.

Within class ``DQN(Algorithm)``, we define the following methods:


- \_\_init\_\_(self, model, act_dim=None, gamma=None, lr=None)

  We define ``self.model`` and ``self.target_model`` of DQN in this method, which are instances of class ``parl.Model``. 
  And we also set hyperparameters act_dim, gamma and lr here. We will use these parameters in ``learn`` method.

  .. code-block:: python

    def __init__(self,
                 model,
                 act_dim=None,
                 gamma=None,
                 lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): model defining forward network of Q function
            act_dim (int): dimension of the action space
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

- predict(self, obs)

  We use the forward network defined in ``self.model`` here, which uses observations to predict action values directly.

  .. code-block:: python

    def predict(self, obs):
            """ use value model self.model to predict the action value
            """
            return self.model.value(obs)

- learn(self, obs, action, reward, next_obs, terminal)

  ``learn`` method calculates the cost of value function according to the predict value and the target value.
  ``Agent`` will use the cost to update weights in ``self.model``.

  .. code-block:: python

    def learn(self, obs, action, reward, next_obs, terminal):
        """ update value model self.model with DQN algorithm
        """

        pred_value = self.model.value(obs)
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True
        target = reward + (
            1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * best_v

        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(self.lr, epsilon=1e-3)
        optimizer.minimize(cost)
        return cost

- sync_target(self)

  Use this method to synchronize the weights in ``self.target_model`` with those in ``self.model``. 
  This is the step used in DQN algorithm.

  .. code-block:: python

    def sync_target(self, gpu_id=None):
        """ sync weights of self.model to self.target_model
        """
        self.model.sync_weights_to(self.target_model)
