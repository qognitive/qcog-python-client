Client
======

This class provides a synchronous interface to qognitive's API. It wraps the async client resolving the async calls to synchronous ones.
It's important that this client is used in a synchronous context, otherwise it will block the event loop.

.. autoclass:: qcog_python_client.QcogClient
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
