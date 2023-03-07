GPU Cluster
=============

Author: wuzewu@baidu.com 

This tutorial demonstrates how to set up a GPU cluster.

First we run the following command to launch a GPU cluster with the port 8002.

.. code-block:: bash

  xparl start --port 8002 --gpu_cluster


Then we add GPU resource of the computation server to the cluster. 
Users should specify the GPUs added to the cluster with the argument ``--gpu``.

The following command is an example that adds the first 4 GPUs into the cluster.

.. code-block:: bash

  xparl connect --address ${CLUSTER_IP}:${CLUSTER_PORT} --gpu 0,1,2,3


Once the GPU cluster based on xparl has been established, we can leverage the ``parl.remote_class`` decorator 
to execute parallel computations. The number of GPUs to be utilized can be specified by the ``n_gpu`` argument.

Here is an entry level example to test the GPU cluster we have set up.

.. code-block:: python

	import parl
	import os
	
	# connect to the Cluster. replace the ip and port with the actual IP address.
	parl.connect("localhost:8002")
	
	# Use a decorator to decorate a local class, which will be sent to a remote instance.
	# n_gpu=2 means that this Actor will be allocated two GPU cards.
	@parl.remote_class(n_gpu=2)
	class Actor:
	    def get_device(self):
	        return os.environ['CUDA_VISIBLE_DEVICES']
	        
	    def step(self, a, b):
	        return a + b
	
	actor = Actor()
	# execute remotely and return the value of the CUDA_VISIBLE_DEVICES environment variable.
	print(actor.get_device())
