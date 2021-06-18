Xparl Usage
=============
Setup Command
###################
This tutorial demonstrates how to set up a cluster.

To start a PARL cluster, we can execute the following two ``xparl`` commands:

.. code-block:: bash

  xparl start --port 6006

This command starts a master node to manage computation resources and adds the local CPUs to the cluster.
We use the port `6006` for demonstration, and it can be any available port.

Adding More Resources
#######################

.. note::
    If you have only one machine, you can ignore this part.

If you would like to add more CPUs(computation resources) to the cluster, run the following command on other machines.

.. code-block:: bash

  xparl connect --address localhost:6006

It starts a worker node that provides CPUs of the machine for the master. A worker will use all the CPUs by default. If you wish to specify the number of CPUs to be used, run the command with ``--cpu_num <cpu_num>`` (e.g.------cpu_num 10). 

Note that the command ``xparl connect`` can be run at any time, at any machine to add more CPUs to the cluster.

Example
###################
Here we give an example demonstrating how to use ``@parl.remote_class`` for parallel computation.

.. code-block:: python

  import parl

  @parl.remote_class
  class Actor(object):
      def hello_world(self):
          print("Hello world.")

      def add(self, a, b):
          return a + b

  # Connect to the master node.
  parl.connect("localhost:6006")

  actor = Actor()
  actor.hello_world()# no log in the current terminal, as the computation is placed in the cluster.
  actor.add(1, 2)  # return 3

Shutdown the Cluster
#######################
run ``xparl stop`` at the machine that runs as a master node to stop the cluster processes. Worker nodes at different machines will exit automatically after the master node is stopped.

Further Reading
#######################
| Now we know how to set up a cluster and use this cluster by simply adding ``@parl.remote_class``. 
| In `next_tutorial`_, we will show how this decorator help us implement the **real** multi-thread computation in Python, breaking the limitation of Python Global Interpreter Lock(GIL).

.. _`next_tutorial`: https://parl.readthedocs.io/en/latest/parallel_training/recommended_practice.html
