Introduction to Datasets
========================

A dataset is what we call a collection of data that is used for training a model.

A model is defined by both the data that it was trained on (the dataset) as well as the parameterization of the model (specific values that define the behavior of the model) as well as some hyperparameters for how to train (such as batch sizes and number of passes that the model will do over the dataset).

We'll need to upload some data in order to train a model on it.

Binary XY = Z
-------------
As a simple example, we create a small synthetic dataset which we call `XY=Z`.  Super simple, it is a dataset that has two input features `X` and `Y` and one output feature `Z`.  The output feature `Z` is just `X` times `Y`.

For simplicity, we confine ourselves to the values of `1` and `-1` so we can be even smaller.  Our complete dataset then looks like this:

.. code-block:: python

    import numpy as np
    import pandas as pd

    xs = np.array([1, 1, -1 , -1])
    ys = np.array([1, -1, -1, 1])
    zs = xs * ys
    dataset = pd.DataFrame(
        data=(np.vstack([xs, ys, zs])).T,
        index=range(4),
        columns=["X", "Y", "Z"]
    )
    print(dataset)
    # X  Y  Z
    # 0  1  1  1
    # 1  1 -1 -1
    # 2 -1 -1  1
    # 3 -1  1 -1

We want to now send this to Qognitive to store so we can train on this later

Uploading to the API
--------------------

First we'll instantiate our client object.

.. note::
    We should have more defaults and fix the DNS

.. code-block:: python

    from qcog_python_client import QcogClient
    qcml = QcogClient.create(
        token=API_TOKEN,
        hostname="api.qognitive.io",
        port=443,
        verify=False,
        secure=True,
    )

Now we'll upload our data to the API.

.. code-block:: python

    dataset = qcml.data(dataset)

That's in, now we just need to get our parameters and we can train a model on this dataset.

Using Async
-----------

We can do the same operations but using our Async client for those whose infrastructure is built for async.

.. code-block:: python

    from qcog_python_client import AsyncQcogClient
    qcml = await AsyncQcogClient.create(
        token=API_TOKEN,
        hostname="api.qognitive.io",
        port=443,
        verify=False,
        secure=True,
    )
    await qcml.data(dataset)

Asyncronous programming is very powerful when your applications become ``io`` bound, either by the network (APIs, Databases, etc) or through file operations. This allows you to keep the CPU busy while waiting for the IO to complete. For more detail about using asyncronous programming in python `the python asyncio docs are a good place to start <https://docs.python.org/3/library/asyncio.html>`_.
