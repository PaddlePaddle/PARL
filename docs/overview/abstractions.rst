Abstractions
----------------
.. image:: ../../.github/abstractions.png
  :align: center
  :width: 400px

| PARL aims to build an **agent** for training algorithms to perform complex tasks.
| The main abstractions introduced by PARL that are used to build an agent recursively are the following:

* ``Model`` is abstracted to construct the forward network which defines a policy network or critic network given state as input.

* ``Algorithm`` describes the mechanism to update parameters in the *model* and often contains at least one model.

* ``Agent``, a data bridge between the *environment* and the *algorithm*, is responsible for data I/O with the outside environment and describes data preprocessing before feeding data into the training process.

Note: For more information about base classes, please visit our :doc:`tutorial <../tutorial/getting_started>` and :doc:`API document <../apis/model>`.