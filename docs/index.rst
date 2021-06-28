PARL
=============
.. PARL_docs documentation master file, created by
   sphinx-quickstart on Mon Apr 22 11:12:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*PARL is a flexible, distributed and object-oriented programming reinforcement learning framework.*

.. image:: images/PARL-logo-1.png
   :align: center
   :width: 600px

\

.. toctree::
    :maxdepth: 1
    :caption: Overview

    overview/features.rst
    overview/abstractions.rst
    overview/parallelization.rst

.. toctree::
    :maxdepth: 1
    :caption: Installation

    installation.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   tutorial/getting_started.rst
   tutorial/maa.rst
   implementations/new_alg.rst
   tutorial/save_param.rst
   tutorial/tensorboard.rst
   tutorial/output_as_csv.rst

.. toctree::
   :maxdepth: 2
   :caption: High-quality Implementations

   implementations/pg.rst
   implementations/dqn.rst
   implementations/ddpg.rst
   implementations/ddqn.rst
   implementations/oac.rst
   implementations/a2c.rst
   implementations/td3.rst
   implementations/qmix.rst
   implementations/sac.rst
   implementations/ppo.rst
   implementations/maddpg.rst

.. toctree::
    :maxdepth: 2
    :caption: Parallel Training

    parallel_training/overview.rst
    parallel_training/setup.rst
    parallel_training/recommended_practice.rst
    parallel_training/debug.rst
    parallel_training/file_distribution.rst
    parallel_training/serialization.rst

.. toctree::
   :maxdepth: 1
   :caption: APIs

   apis/model.rst
   apis/algorithm.rst
   apis/agent.rst

.. toctree::
   :maxdepth: 1
   :caption: EvoKit

   EvoKit/overview.rst
   EvoKit/minimal_example.rst
   EvoKit/online_example.rst
