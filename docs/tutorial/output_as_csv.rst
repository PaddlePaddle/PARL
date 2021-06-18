CSV Logger
==========

PARL provides a tool to output the indicators during the training process to a CSV table. The tool can be imported using:

.. code-block:: python

    from parl.utils import CSVLogger

How to Use
-------------

1. Input path for saving the CSV file and initialize ``CSVLogger``:

.. code-block:: python

    csv_logger = CSVLogger("result.csv")

2. Output a dictionary that contains the indicators:

`Parameters`:

* result(dict) - indicators that need to be outputted as CSV file

`Method`:

.. code-block:: python

    csv_logger.log_dict({"loss": 1, "reward": 2})

Example
-------------

.. code-block:: python

    from parl.utils import CSVLogger

    csv_logger = CSVLogger("result.csv")
    csv_logger.log_dict({"loss": 1, "reward": 2})
    csv_logger.log_dict({"loss": 3, "reward": 4})

The CSV file will contain:

.. code-block::

    loss,reward
    1,2
    3,4
