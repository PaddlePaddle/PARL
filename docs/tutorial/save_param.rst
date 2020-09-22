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
    # save the parameters of agent to ./model_dir
    agent.save('./model_dir')             
    # restore the parameters from ./model_dir to agent  
    agent.restore('./model_dir')    

    # restore the parameters from ./model_dir to another_agent
    another_agent = AtariAgent()
    another_agent.restore('./model_dir')    
