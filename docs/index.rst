.. PARL_docs documentation master file, created by
   sphinx-quickstart on Mon Apr 22 11:12:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*PARL is a flexible, distributed and object-oriented programming reinforcement learning framework.*

Features
----------------
+------------------------------------------+---------------------------------------+
| **Object Oriented Programming**          | **Distributed Training**              |
+------------------------------------------+---------------------------------------+
|.. code-block:: python                    |.. code-block:: python                 |
|                                          |                                       |
|                                          |  # Absolute multi-thread programming  |
|   class MLPModel(parl.Model):            |  # witout the GIL limitation          |
|     def __init__(self, act_dim):         |                                       |
|       self.fc1 = layers.fc(size=10)      |  @parl.remote_class                   |
|       self.fc2 = layers.fc(size=act_dim) |  class HelloWorld(object):            |
|                                          |      def sum(self, a, b):             |
|     def forward(self, obs):              |          return a + b                 |
|       out = self.fc1(obs)                |                                       |
|       out = self.fc2(out)                |  parl.connect('localhost:8003')       |
|       return out                         |  obj = HelloWorld()                   |
|                                          |  ans = obj.sum(a, b)                  |
|   model = MLPModel()                     |                                       |
|   target_model = copy.deepcopy(model)    |                                       |
+------------------------------------------+---------------------------------------+

Abstractions
----------------
.. image:: ../.github/abstractions.png
  :align: center
  :width: 400px

| PARL aims to build an **agent** for training algorithms to perform complex tasks.
| The main abstractions introduced by PARL that are used to build an agent recursively are the following:

* **Model** is abstracted to construct the forward network which defines a policy network or critic network given state as input.

* **Algorithm** describes the mechanism to update parameters in the *model* and often contains at least one model.

* **Agent**, a data bridge between the *environment* and the *algorithm*, is responsible for data I/O with the outside environment and describes data preprocessing before feeding data into the training process.

.. toctree::
    :maxdepth: 1
    :caption: Installation

    installation.rst

.. toctree::
    :maxdepth: 1
    :caption: Features

    features.rst

.. toctree::
    :maxdepth: 1
    :caption: Tutorial

    tutorial/getting_started.rst
    tutorial/new_alg.rst
    tutorial/save_param.rst
    tutorial/tensorboard.rst

.. toctree::
    :maxdepth: 2
    :caption: Parallel Training

    parallel_training/overview.rst
    parallel_training/setup.rst
    parallel_training/recommended_practice.rst

.. toctree::
   :maxdepth: 1
   :caption: High-quality Implementations

   implementations.rst

.. toctree::
   :maxdepth: 1
   :caption: APIs

   model.rst
   algorithm.rst
   agent.rst

.. toctree::
   :maxdepth: 2
   :caption: EvoKit

   EvoKit/overview.rst
   EvoKit/minimal_example.rst
   EvoKit/online_example.rst
