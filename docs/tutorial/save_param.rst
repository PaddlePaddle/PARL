Save and Restore Parameters
=============================

Goal of this tutorial:

- Learn how to save and restore parameters.

**Scene 1:**

Sometimes we need to save the parameters into a file and reuse them later on. PARL provides operators
to save parameters to a file and restore parameters from a file easily. You only need several lines to implement this.

Here is a demonstration of usage:

.. code-block:: python

    agent = AtariAgent()
    # save the parameters of agent to ./model_dir
    agent.save('./model_dir')             
    # restore the parameters from ./model_dir to agent  
    agent.restore('./model_dir')    

**Scene 2:**

Sometimes during training procedure, we want to sync the latest model parameters to Agents (Actors) on other servers. To deal with this, we need to first move the parameters to memory then
set the parameters of Agents (Actors) on other servers.

.. code-block:: python

    #--------------Agent---------------
    weights = agent.get_weights()
    #--------------Remote Actor--------------
    actor.set_weights(weights)