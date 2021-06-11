Parallelization
----------------


| PARL provides a compact API for distributed training, allowing users to transfer the code into a parallelized version by simply adding a decorator. For more information about our APIs for parallel training, please visit our :doc:`tutorial <../parallel_training/overview>`.

| Here is a ``Hello World`` example to demonstrate how easy it is to leverage outer computation resources:

.. code-block:: python

    #============Agent.py=================
    @parl.remote_class
    class Agent(object):

        def say_hello(self):
            print("Hello World!")

        def sum(self, a, b):
            return a+b
    parl.connect('localhost:8037')
    agent = Agent()
    agent.say_hello()
    ans = agent.sum(1,5) # it runs remotely, without consuming any local computation resources

| Two steps to use outer computation resources:

1. use the ``parl.remote_class`` to decorate a class at first, after which it is transferred to be a new class that can run in other CPUs or machines.
2. call ``parl.connect`` to initialize parallel communication before creating an object. Calling any function of the objects **does not** consume local computation resources since they are executed elsewhere.


.. image:: ../../.github/decorator.png
  :align: center
  :width: 450px

| As shown in the above figure, real actors (orange circle) are running at the cpu cluster, while the learner (blue circle) is running at the local gpu with several remote actors (yellow circle with dotted edge).

| For users, they can write code in a simple way, just like writing multi-thread code, but with actors consuming remote resources. We have also provided examples of parallelized algorithms like IMPALA, :doc:`A2C <../implementations/a2c>`. For more details in usage please refer to these examples.
