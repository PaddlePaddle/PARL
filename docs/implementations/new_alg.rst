Create Customized Algorithms
============================

Goal of this tutorial:

- Learn how to implement your own algorithms.


Overview
---------

To build a new algorithm, you need to inherit class ``parl.Algorithm``
and implement three basic functions: ``predict`` and ``learn``.


Methods
--------

- ``__init__``

  As algorithms update weights of the models, this method needs to define some models inherited from ``parl.Model``, like ``self.model`` in this example.
  You can also set some hyperparameters in this method, like ``learning_rate``, ``reward_decay`` and ``action_dimension``,
  which might be used in the following steps.

- ``predict``

  This function defines how to choose actions. For instance, you can use a policy model to predict actions.

- ``learn``

  Define loss function in ``learn`` method, which will be used to update weights of ``self.model``.


Example: DQN
--------------

This example shows how to implement DQN algorithm based on class ``parl.Algorithm`` according to the steps mentioned above.

Within class ``DQN(Algorithm)``, we define the following methods:


- \_\_init\_\_(self, model, gamma=None, lr=None)

  We define ``self.model`` and ``self.target_model`` of DQN in this method, which are instances of class ``parl.Model``. 
  And we also set hyperparameters gamma and lr here. We will use these parameters in ``learn`` method.

  .. code-block:: python

    def __init__(self, model, gamma=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): forward neural network representing the Q function.
            gamma (float): discounted factor for `accumulative` reward computation
            lr (float): learning rate.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.parameters())

- predict(self, obs)

  We use the forward network defined in ``self.model`` here, which uses observations to predict action values directly.

  .. code-block:: python

    def predict(self, obs):
        """ use self.model (Q function) to predict the action values
        """
        return self.model.value(obs)

- learn(self, obs, action, reward, next_obs, terminal)

  ``learn`` method calculates the cost of value function according to the predict value and the target value.
  ``Agent`` will use the cost to update weights in ``self.model``.

  .. code-block:: python

    def learn(self, obs, action, reward, next_obs, terminal):
        """ update the Q function (self.model) with DQN algorithm
        """
        # Q
        pred_values = self.model.value(obs)
        action_dim = pred_values.shape[-1]
        action = paddle.squeeze(action, axis=-1)
        action_onehot = paddle.nn.functional.one_hot(
            action, num_classes=action_dim)
        pred_value = paddle.multiply(pred_values, action_onehot)
        pred_value = paddle.sum(pred_value, axis=1, keepdim=True)

        # target Q
        with paddle.no_grad():
            max_v = self.target_model.value(next_obs).max(1, keepdim=True)
            target = reward + (1 - terminal) * self.gamma * max_v

        loss = self.mse_loss(pred_value, target)

        # optimize
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        return loss

- sync_target(self)

  Use this method to synchronize the weights in ``self.target_model`` with those in ``self.model``. 
  This is the step used in DQN algorithm.

  .. code-block:: python

    def sync_target(self):

        self.model.sync_weights_to(self.target_model)
