State Parameters
================

Our QCML model will build a representation of the data as quantum states. The method for solving for these states is a parameter that can be chosen and usually results in a tradeoff between cost and accuracy.

The recommendation from the Qognitive team is that the ``state_method = 'LOBPCG_FAST'`` should be used for the majority of cases.

The state parameterization has to be passed for both training (as the model will build quantum states of the data as it trains, refining the state representation as it goes) and for inference (the model will build a state from the provided data and generate inference from that state).

This documentation is broken up by each ``state_method`` parameter, each having their own parameters.

LOBPCG_FAST
-----------

The recommended method for most cases. This method uses the Locally Optimal Block Preconditioned Conjugate Gradient method to find the optimal state. We call this ``FAST`` because it exploits internal symmetries of the problem to speed up the computation versus a generic implementation of LOBPCG.

Parameters
#. iterations: int, default=100. The number of iterations to run the LOBPCG algorithm for.
#. tol: float, default=1e-6. The tolerance for the LOBPCG algorithm.

.. glossary::

    iterations: int, default=100.
        The number of iterations to run the LOBPCG algorithm for.
    tol: float, default=1e-6.
        The tolerance for the LOBPCG algorithm.

.. hlist::
    :columns: 1

    * ``iterations``: int, default=100.
        The number of iterations to run the LOBPCG algorithm for.
    * ``tol``: float, default=1e-6.
        The tolerance for the LOBPCG algorithm.

.. hlist::
    :columns: 2

    * ``iterations``: int, default=100.
    * The number of iterations to run the LOBPCG algorithm for.
    * ``tol``: float, default=1e-6.
    * The tolerance for the LOBPCG algorithm.


POWER_ITER
-----------

Here we use the power iteration method to find the optimal state. This method also exploits internal symmetries of the problem to speed up the computation versus more generic implementations of such as subspace solvers.

EIGS
----

This uses the scipy ``EIGS`` solver.

EIGH
----

This uses the scipy ``EIGH`` solver.

NP_EIGH
-------

This uses the numpy ``EIGH`` solver.

LOBPCG
------

This uses the scipy ``LOBPCG`` solver.


GRAD
----

Using a gradient descent procedure with fixed learning rate we find our way to the optimal state.

