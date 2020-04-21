Save and Restore Parameters
=============================

Goal of this tutorial:

- Learn how to save and restore parameters.

Example
---------------

Sometimes we need to save the parameters into a file and reuse them later on. PARL provides operators 
to save parameters to a file and restore parameters from a file easily. You only need several lines to implement this.

Here is a demonstration of usage:

.. code-block:: python

    agent = AtariAgent()
    # save the parameters of agent to ./model.ckpt
    agent.save('./model.ckpt')             
    # restore the parameters from ./model.ckpt to agent  
    agent.restore('./model.ckpt')    

    # restore the parameters from ./model.ckpt to another_agent
    another_agent = AtariAgent()
    another_agent.restore('./model.ckpt')    
