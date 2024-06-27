Ensemble QCML
=============

The Ensemble QCML model uses an ensemble of state vectors in order to build a quantum representation of the problem. For the Ensemble QCML model the most important parameters are the ``dims`` and the ``axes``. You can read about the function of these parameters below, they get passed when ``qcml.ensemble(...)`` is called.

.. autoclass:: qcog_python_client.EnsembleSchema

Special Training Parameters
---------------------------

The Ensemble QCML model has additional parameters to be passed to training.

These parameters are additional to the :doc:`/parameters/state` and the additional parameters are valid for all methods of state solution.

These parameters are as folows:

.. hlist::
    :columns: 1

    * ``learning_rate_axes``: float
        This is the learning rate that the axes use to optimize them. In some sense you can think of the axes as a set of basis vectors that will continually adjust themselves to better represent the data. Recommended values are around ``1e-3`` during training. This should be set to 0 or omitted during inference or the model will be updated based on partial inference data. This is important to include during training in order to get the best results.
    * ``normalize_axes``: bool, default True
        Set this to False in order to not enforce the unitarity of the axes vectors.
