summary
===============

Visualize the results with tensorboard. 

add_scalar
-------------

Common used arguments:

* summary.add_scalar(tag, scalar_value, global_step=None)
    * tag *(string)* – Data identifier
    * scalar_value *(float or string/blobname)* – Value to save
    * global_step *(int)* – Global step value to record

Example:

.. code-block:: python

    from parl.utils import summary

    x = range(100)
    for i in x:
        summary.add_scalar('y=2x', i * 2, i)

Expected result:

    .. image:: add_scalar.jpg
        :scale: 50 %
            
add_histogram
----------------

Common used arguments:

* summary.add_scalar(tag, scalar_value, global_step=None)
    * tag *(string)* – Data identifier
    * values *(torch.Tensor, numpy.array, or string/blobname)* – Values to build histogram
    * global_step *(int)* – Global step value to record

Example:

.. code-block:: python

    from parl.utils import summary
    import numpy as np

    for i in range(10):
        x = np.random.random(1000)
        summary.add_histogram('distribution centers', x + i, i)

Expected result:

    .. image:: add_histogram.jpg
        :scale: 50 %
