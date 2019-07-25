Model (*Forward Part*)
=======================
A Model is owned by an Algorithm. Model is responsible for the entire network model (**forward part**) for the specific problems.


Methods
----------
1. policy(self, obs)

    Define the structure of networks here. Algorithm will call this method to predict probabilities of actions. 
    It is optional. 

2. value(self, obs)

    Return: values: a dict of estimated values for the current observations and states. 
    For example, "q_value" and "v_value".

3. sync_params_to(self, target_net, gpu_id, decay=0.0, share_vars_parallel_executor=None)

    This method deepcopied the parameters from the current network to the target network, which two have the same structure.  

An example
------------
.. code-block:: python
    :linenos:

    class MLPModel(Model):
        def __init__(self):
            self.fc = layers.fc(size=64)

        def policy(self, obs):
            out = self.fc(obs)
            return out
            
    model = MLPModel() 
    target_model = deepcopy(model) # automatically create new unique parameters names for target_model.fc

    # build program
    x = layers.data(name='x', shape=[100], dtype="float32")
    y1 = model.policy(x) 
    y2 = target_model.policy(x)  
