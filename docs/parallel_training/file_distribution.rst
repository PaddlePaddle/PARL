File Distribution
==================

File distribution is an important function of distributed parallel computing. It is responsible for distributing user's code
and configuration files to different machines, so that all machines perform parallel computing using same code. By default, all ``.py`` files that are located in the same directory
as the XPARL distribution main file (such as ``main.py`` ) will be distributed. But sometimes users need to distribute some specific files, such as model files, configuration files, and Python code in subdirectories (submodules for import).
In order to meet this demand, ``parl.connect`` provides an interface where users can directly specify the files or codes that need to be distributed.

Example:
################

The file directory structure is as follows, we want to distribute the ``.py`` files in the policy folder. We can pass the files that we want to distribute to the ``distributed_files`` parameter when ``connect``, this parameter also supports regular expressions.

.. code-block::

    .
    ├── main.py
    └── policy
        ├── agent.py
        ├── config.ini
        └── __init__.py

.. code-block:: python

    parl.connect("localhost:8004", distributed_files=['./policy/*.py', './policy/*.ini'])
