Ensemble QCML
=============

The Ensemble QCML model uses an ensemble of state vectors in order to build a quantum representation of the problem. For the Ensemble QCML model the most important parameters are the ``dims`` and the ``axes``. You can read about the function of these parameters below, they get passed when ``qcml.ensemble(...)`` is called.

.. autoclass:: qcog_python_client.EnsembleSchema
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

The Ensemble QCML model has additional parameters to be passed to training.  See :doc:`/parameters/optimization` for more information.