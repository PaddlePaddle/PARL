.. PARL_docs documentation master file, created by
   sphinx-quickstart on Mon Apr 22 11:12:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PARL
=====================================
*PARL is a flexible, distributed and eager mode oriented reinforcement learning framework.*

Features
----------------
+----------------------------------------------+-----------------------------------------------+
| **Eager Mode**                               | **Distributed Training**                      |
+----------------------------------------------+-----------------------------------------------+
|.. code-block:: python                        |.. code-block:: python                         |
|                                              |                                               |
|  # Target Network in DQN                     |  # Real multi-thread programming              |
|                                              |  # witout the GIL limitation                  |
|                                              |                                               |
|    target_network = copy.deepcopy(Q_network) |  @parl.remote_class                           |
|    ...                                       |  class HelloWorld(object):                    |
|    #reset parameters periodically            |      def sum(self, a, b):                     |
|    target_network.load(Q_network)            |          return a + b                         |
|                                              |                                               |
|                                              |  parl.init()                                  |
|                                              |  obj = HelloWorld()                           |
|                                              |  # NOT consume local computation resources    |
|                                              |  ans = obj.sum(a, b)                          |
|                                              |                                               |
+----------------------------------------------+-----------------------------------------------+


| PARL is distributed on PyPI and can be installed with pip:

.. centered:: ``pip install parl``

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
   :caption: Basic_structure

   ./basic_structure/overview
   ./basic_structure/model
   ./basic_structure/algorithm
   ./basic_structure/agent

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   tutorial.rst

.. toctree::
   :maxdepth: 1
   :caption: High-quality Implementations

   implementations.rst

.. toctree::
   :maxdepth: 3
   :caption: API

   algo_docs/index
   framework_docs/index
   layers_docs/index
   plutils_docs/index
   remote_docs/index
   utils_docs/index

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
