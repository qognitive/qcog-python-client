Model Selection
===============

Here we will choose one of our QCML models and parameterize it for training.  There are several QCML models that you can choose from and we are continuing to develop new models as well as improve existing ones.  You can see which models are currently available on the sidebar.

We're going to use the Pauli model for this example.

Setting the Model Parameters
----------------------------

.. code-block:: console

    >>> qcml = qcml.pauli(
        operators=["X", "Y", "Z"],
        qbits=2,
        pauli_weight=2
    )



Setting the Training Parameters
-------------------------------