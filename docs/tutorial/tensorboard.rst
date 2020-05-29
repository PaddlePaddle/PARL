logger
===============

Visualize the results with tensorboard. 

add_scalar
-------------

Common used arguments:

* logger.add_scalar(tag, scalar_value, global_step=None)
    * tag *(string)* – Data identifier
    * scalar_value *(float or string/blobname)* – Value to save
    * global_step *(int)* – Global step value to record

Example:

.. code-block:: python

    from parl.utils import logger

    x = range(100)
    for i in x:
        logger.add_scalar('y=2x', i * 2, i)

Expected result:

    .. image:: add_scalar.jpg
        :scale: 50 %
            
add_histogram
----------------

Common used arguments:

* logger.add_scalar(tag, scalar_value, global_step=None)
    * tag *(string)* – Data identifier
    * values *(torch.Tensor, numpy.array, or string/blobname)* – Values to build histogram
    * global_step *(int)* – Global step value to record

Example:

.. code-block:: python

    from parl.utils import logger
    import numpy as np

    for i in range(10):
        x = np.random.random(1000)
        logger.add_histogram('distribution centers', x + i, i)

Expected result:

    .. image:: add_histogram.jpg
        :scale: 50 %
