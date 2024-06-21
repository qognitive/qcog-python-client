Installation Quickstart
=======================

We recommend that you set up a python 3.10+ virtual environment to install the package.

Further documentation on virtual environments can be found `here <https://docs.python.org/3/library/venv.html>`_.

Package Installation
--------------------

Once your virtual environment is set up you can install the package by running the following command:

.. code-block:: console

   (venv) $ pip install qcog-python-client

API Token
---------

You'll need an API token from Qognitive to use the platform. This can be obtained by contacting us at `support@qognitive.io`.

Once you have the token you have two ways to set the token in your system:

1. Set the token as the environment variable `QCOG_API_TOKEN`
2. Pass the token as an argument to the `QcogClient` when calling `QcogClient.create()`

Next Steps
----------

Let's go onto uploading a dataset that we can train
