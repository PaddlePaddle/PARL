Getting Started
===========

Goal of this tutorial:

  - Understand PARL's abstraction at a high level
  - Train an agent to solve the Cartpole problem with Policy Gradient algorithm

This tutorial assumes that you have a basic familiarity of policy gradient.

Model
-----
First, let's build a ``Model`` that predicts an action given the observation. As an objective-oriented programming framework, we build models on the top of ``parl.Model`` and implement the ``forward`` function.

Here, we construct a neural network with two fully connected layers.

.. code-block:: python

		import parl
		from parl import layers
		
		class CartpoleModel(parl.Model):
		    def __init__(self, act_dim):
		        act_dim = act_dim
		        hid1_size = act_dim * 10
		
		        self.fc1 = layers.fc(size=hid1_size, act='tanh')
		        self.fc2 = layers.fc(size=act_dim, act='softmax')
		
		    def forward(self, obs):
		        out = self.fc1(obs)
		        out = self.fc2(out)
		        return out

Algorithm
----------
``Algorithm`` will update the parameters of the model passed to it. In general, we define the loss function in ``Algorithm``.
In this tutorial, we solve the benchmark `Cartpole` using the `Policy Graident` algorithm, which has been implemented in our repository.
Thus, we can simply use this algorithm by importting it from ``parl.algorithms``.

We have also published various algorithms in PARL, please visit this `page <https://parl.readthedocs.io/en/latest/implementations.html>`_ for more detail. 
For those who want to implement a new algorithm, please follow this `tutorial <https://parl.readthedocs.io/en/latest/new_alg.html>`_.

.. code-block:: python

  model = CartpoleModel(act_dim=2)
  algorithm = parl.algorithms.PolicyGradient(model, lr=1e-3)

Note that each ``algorithm`` should have two functions implemented:

- ``learn``

  updates the model's parameters given transition data 
- ``predict``

  predicts an action given current environmental state. 

Agent
----------
Now we pass the algorithm to an agent, which is used to interact with the environment to generate training data. Users should build their agents on the top of ``parl.Agent`` and  implement four functions:

- ``build_program``

  define programs of fluid. In general, two programs are built here, one for prediction and the other for training.
- ``learn``

  preprocess transition data and feed it into the training program.
- ``predict``

  feed current environmental state into the prediction program and return an exectuive action.
- ``sample``

  this function is usually used for exploration, fed with current state.

.. code-block:: python

		class CartpoleAgent(parl.Agent):
		    def __init__(self, algorithm, obs_dim, act_dim):
		        self.obs_dim = obs_dim
		        self.act_dim = act_dim
		        super(CartpoleAgent, self).__init__(algorithm)
		
		    def build_program(self):
		        self.pred_program = fluid.Program()
		        self.train_program = fluid.Program()
		
		        with fluid.program_guard(self.pred_program):
		            obs = layers.data(
		                name='obs', shape=[self.obs_dim], dtype='float32')
		            self.act_prob = self.alg.predict(obs)
		
		        with fluid.program_guard(self.train_program):
		            obs = layers.data(
		                name='obs', shape=[self.obs_dim], dtype='float32')
		            act = layers.data(name='act', shape=[1], dtype='int64')
		            reward = layers.data(name='reward', shape=[], dtype='float32')
		            self.cost = self.alg.learn(obs, act, reward)
		
		    def sample(self, obs):
		        obs = np.expand_dims(obs, axis=0)
		        act_prob = self.fluid_executor.run(
		            self.pred_program,
		            feed={'obs': obs.astype('float32')},
		            fetch_list=[self.act_prob])[0]
		        act_prob = np.squeeze(act_prob, axis=0)
		        act = np.random.choice(range(self.act_dim), p=act_prob)
		        return act
		
		    def predict(self, obs):
		        obs = np.expand_dims(obs, axis=0)
		        act_prob = self.fluid_executor.run(
		            self.pred_program,
		            feed={'obs': obs.astype('float32')},
		            fetch_list=[self.act_prob])[0]
		        act_prob = np.squeeze(act_prob, axis=0)
		        act = np.argmax(act_prob)
		        return act
		
		    def learn(self, obs, act, reward):
		        act = np.expand_dims(act, axis=-1)
		        feed = {
		            'obs': obs.astype('float32'),
		            'act': act.astype('int64'),
		            'reward': reward.astype('float32')
		        }
		        cost = self.fluid_executor.run(
		            self.train_program, feed=feed, fetch_list=[self.cost])[0]
		        return cost

Start Training
-----------
First, let's build an ``agent``. As the code shown below, we usually build a model, an algorithm and finally agent.

.. code-block:: python

    model = CartpoleModel(act_dim=2)
    alg = parl.algorithms.PolicyGradient(model, lr=1e-3)
    agent = CartpoleAgent(alg, obs_dim=OBS_DIM, act_dim=2)

Then we use this agent to interact with the environment, and run around 1000 episodes for training, after which this agent can solve the problem.

.. code-block:: python

		def run_episode(env, agent, train_or_test='train'):
		    obs_list, action_list, reward_list = [], [], []
		    obs = env.reset()
		    while True:
		        obs_list.append(obs)
		        if train_or_test == 'train':
		            action = agent.sample(obs)
		        else:
		            action = agent.predict(obs)
		        action_list.append(action)
		
		        obs, reward, done, info = env.step(action)
		        reward_list.append(reward)
		
		        if done:
		            break
		    return obs_list, action_list, reward_list

  		env = gym.make("CartPole-v0")
  		for i in range(1000):
  		      obs_list, action_list, reward_list = run_episode(env, agent)
  		      if i % 10 == 0:
  		          logger.info("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))

  		      batch_obs = np.array(obs_list)
  		      batch_action = np.array(action_list)
  		      batch_reward = calc_discount_norm_reward(reward_list, GAMMA)

  		      agent.learn(batch_obs, batch_action, batch_reward)
  		      if (i + 1) % 100 == 0:
  		          _, _, reward_list = run_episode(env, agent, train_or_test='test')
  		          total_reward = np.sum(reward_list)
  		          logger.info('Test reward: {}'.format(total_reward))

Summary
-----------

.. image:: ../../examples/QuickStart/performance.gif
  :width: 300px
.. image:: ../images/quickstart.png
  :width: 300px
In this tutorial, we have shown how to build an agent step-by-step to solve the `Cartpole` problem.

The whole training code could be found `here <https://github.com/PaddlePaddle/PARL/tree/develop/examples/QuickStart>`_. Have a try quickly by running several commands:

.. code-block:: shell

	# Install dependencies
	pip install paddlepaddle  
	
	pip install gym
	git clone https://github.com/PaddlePaddle/PARL.git
	cd PARL
	pip install .
	
	# Train model
	cd examples/QuickStart/
	python train.py  
	
