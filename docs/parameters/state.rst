State Parameters
================

Our QCML model will build a representation of the data as quantum states. The method for solving for these states is a parameter that can be chosen and usually results in a tradeoff between cost and accuracy.

The recommendation from the Qognitive team is that the ``state_method = 'LOBPCG_FAST'`` should be used for the majority of cases.

The state parameterization has to be passed for both training (as the model will build quantum states of the data as it trains, refining the state representation as it goes) and for inference (the model will build a state from the provided data and generate inference from that state).

This documentation is broken up by each ``state_method`` parameter, each having their own parameters.

LOBPCG_FAST
-----------

The recommended method for most cases. This method uses the Locally Optimal Block Preconditioned Conjugate Gradient method to find the optimal state. We call this ``FAST`` because it exploits internal symmetries of the problem to speed up the computation versus a generic implementation of LOBPCG.

.. hlist::
    :columns: 1

    * ``iterations``: int
        The maximum number of iterations to run the LOBPCG algorithm for. You should think of this as a maximum number of internal iterations that are made to attempt the states to converge to within the tolerance. Both parameters work together, so if your tolerance is very large then you will need few iterations, so even if you set iteration count to 100 if it only takes 5 to converge to your tolerance only 5 iterations will be done. If your tolerance is very low but the iteration count is low then the algorithm will stop after the iteration count is reached, regardless of the tolerance. A good recommended range is 5-20, noting that the more iterations you do the more accurate the state will be, but the more computationally expensive it will be.
    * ``tol``: float, default=0.2.
        The tolerance for the LOBPCG algorithm. This is a relative tolerance and not an absolute one, so the units are arbitrary. Generally 0.2 which is the default is a very loose tolerance. If you want to have the output from this method close to one of the more exact solvers then try ``1e-4 -> 1e-8`` for the tolerance. As per the above discussion you should also increase your iterations if you are decreasing your tolerance. The tolerance and iterations are more important for inference as the model will only pass over that data once.


POWER_ITER
-----------

Here we use the power iteration method to find the optimal state. This method also exploits internal symmetries of the problem to speed up the computation versus more generic implementations of such as subspace solvers.

.. hlist::
    :columns: 1

    * ``iterations``: int
        The maximum number of iterations to run the POWER_ITER algorithm for. You should think of this as a maximum number of internal iterations that are made to attempt the states to converge to within the tolerance. Both parameters work together, so if your tolerance is very large then you will need few iterations, so even if you set iteration count to 100 if it only takes 5 to converge to your tolerance only 5 iterations will be done. If your tolerance is very low but the iteration count is low then the algorithm will stop after the iteration count is reached, regardless of the tolerance. A good recommended range is 5-20, noting that the more iterations you do the more accurate the state will be, but the more computationally expensive it will be.
    * ``tol``: float, default=0.2.
        The tolerance for the POWER_ITER algorithm. This is a relative tolerance and not an absolute one, so the units are arbitrary. Generally 0.2 which is the default is a very loose tolerance. If you want to have the output from this method close to one of the more exact solvers then try ``1e-4 -> 1e-8`` for the tolerance. As per the above discussion you should also increase your iterations if you are decreasing your tolerance. The tolerance and iterations are more important for inference as the model will only pass over that data once.
    * ``max_eig_iter``: int, default=5
        The number of iterations to execute to find the largest eigenvalue, which is used as a parameter in a spectral shift to find the smallest eigenvalue. 5 is generally a good default.

EIGS
----

This uses the scipy ``EIGS`` solver.

.. hlist::
    :columns: 1

    * ``tol``: float, default=0.2.
        The tolerance for the EIGS algorithm. This is a relative tolerance and not an absolute one, so the units are arbitrary. Generally 0.2 which is the default is a very loose tolerance. If you want to have the output from this method close to one of the more exact solvers then try ``1e-4 -> 1e-8`` for the tolerance. You can see more details about this method `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_. Only ``tol`` is supported as a parameter for this method.

EIGH
----

This uses the scipy ``EIGH`` solver. ``EIGH`` takes no parameters. You can see more details about this method `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html#scipy.linalg.eigh>`_.

NP_EIGH
-------

This uses the numpy ``EIGH`` solver. ``EIGH`` takes no parameters. You can see more details about this method `here <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html>`_.

LOBPCG
------

This uses the scipy ``LOBPCG`` solver. You can see more details about this method `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html>`_.

.. hlist::
    :columns: 1

    * ``iterations``: int
        You should think of this as a maximum number of internal iterations that are made to attempt the states to converge to within the tolerance. Both parameters work together, so if your tolerance is very large then you will need few iterations, so even if you set iteration count to 100 if it only takes 5 to converge to your tolerance only 5 iterations will be done. If your tolerance is very low but the iteration count is low then the algorithm will stop after the iteration count is reached, regardless of the tolerance. A good recommended range is 5-20, noting that the more iterations you do the more accurate the state will be, but the more computationally expensive it will be.
    * ``tol``: float, default=0.2.
        This is a relative tolerance and not an absolute one, so the units are arbitrary. Generally 0.2 which is the default is a very loose tolerance. If you want to have the output from this method close to one of the more exact solvers then try ``1e-4 -> 1e-8`` for the tolerance. As per the above discussion you should also increase your iterations if you are decreasing your tolerance. The tolerance and iterations are more important for inference as the model will only pass over that data once.


GRAD
----

Using a gradient descent procedure with fixed learning rate we find our way to the optimal state.

.. hlist::
    :columns: 1

    * ``iterations``: int
        This is how many gradient descent steps will be made before considering the state to have converged. Having more iterations and a lower learning rate corresponds to a better path through the energy landscape. So if you were to take 10 steps at ``1e-3`` learning rate that is more accurate, as we recompute our gradient 10 times, than a single step of ``1e-2`` learning rate. The recommended range is 3-10.
    * ``learning_rate``: float
        The learning rate for the gradient descent algorithm. This is fixed and does not decay during optimization. Recommended values are around ``1e-3``.
