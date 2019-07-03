Overview
==========
Three Components
------------------
PARL is made up of three components: **Model, Algorithm, Agent**. They are constructed layer-by-layer to build the main body.

Model
---------
A Model is owned by an Algorithm. Model is responsible for the entire network model (**forward part**) for the specific problems.

Algorithm
----------
Algorithm defines the way to update the parameters in the Model (**backward part**). We already implemented some common
used algorithms__, like DQN/DDPG/PPO/A3C, you can directly import and use them.

.. __: https://github.com/PaddlePaddle/PARL/tree/develop/parl/algorithms

Agent
--------
Agent interates with the environment and **generate data flow** outside the Algorithm. 
