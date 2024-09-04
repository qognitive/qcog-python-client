Stand Alone Models
==================

Here we will choose one of our QCML models and parameterize it for training.  There are several QCML models that you can choose from and we are continuing to develop new models as well as improve existing ones.  You can see which models are currently available on the sidebar.

We're going to use the Pauli model for this example.

Setting the Model Parameters
----------------------------

We configure our model by a call on our ``qcml`` object, which returns the object with the model configuration. This is the same procedure for both sync and async clients.

.. code-block:: python

    qcml = qcml.pauli(
        operators=["X", "Y", "Z"],
        qbits=2,
        pauli_weight=2
    )

You can read more about the pauli model, the parameterization, and the available options on the :doc:`/models/stand_alone/pauli` page.

.. note::
    The operators list does have to match the dataset that you are going to be running on. This might seem redundant but it happens in lots of other machine learning models where you have to match the dimensionality of your input to the dimensions of the model - or at least do some preprocessing to get the data into the proper scale.
