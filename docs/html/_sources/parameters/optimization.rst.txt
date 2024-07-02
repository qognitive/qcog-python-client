Optimization Parameters
=======================

The optimization parameters are used to control how our model finds the best internal representation of the problem space. These parameters are used only for training the model and do not affect the model's predictions (since the model is fixed after training).

There are several different methods of optimization that we currently support, each one with its own set of parameters.

ANALYTIC
--------

The analytic optimizer is a closed form solution to the internal weight matrix in our model.

.. note::
    Not all QCML models have an analytic solution.

.. warning::
    The caveat is that the entire dataset needs to be in a single batch and this places memory constraints on the dataset that is used. Small datasets can benefit from the analytic optimizer, but large datasets which cannot fit in memory should use another optimizer.

The analytic optimizer takes no parameters.

GRAD
----

The grad optimizer uses gradient descent with a fixed learning rate to tune the weights in the internal state of our model. GRAD is appropriate for any batch size or QCML model.

.. hlist::
    :columns: 1

    * ``iterations``: int
        This is how many gradient descent steps will be made before considering the state to have converged. Having more iterations and a lower learning rate corresponds to a better path through the energy landscape. So if you were to take 10 steps at ``1e-3`` learning rate that is more accurate, as we recompute our gradient 10 times, than a single step of ``1e-2`` learning rate. The recommended range is 3-10.
    * ``learning_rate``: float
        The learning rate for the gradient descent algorithm. This is fixed and does not decay during optimization. Recommended values are around ``1e-3``.


ADAM
----

ADAM is a stochastic optimizer that tunes the learning rate by various momentum parameters, details of which can be found `in the original paper here <https://arxiv.org/abs/1412.6980>`_. ADAM is appropriate for any batch size or QCML model.

.. hlist::
    :columns: 1

    * ``iterations``: int
        This is how many steps will be made before considering the state to have converged. Having more iterations and a lower learning rate corresponds to a better path through the energy landscape. So if you were to take 10 steps at ``1e-3`` learning rate that is more accurate, as we recompute our gradient 10 times, than a single step of ``1e-2`` learning rate. The recommended range is 3-10.
    * ``step_size``: float, default 1e-3
        The learning rate for the ADAM algorithm. This is the starting point which it will decay from. fixed and does not decay during optimization. Recommended values are around ``1e-3``.
    * ``epsilon``: float, default 1e-8
        A small parameter that is used for numerical stability in the ADAM algorithm. Recommended values are around ``1e-8``.
    * ``first_moment_decay``: float, default 0.9
        The first moment decay, eseentially scaling a term that is linear in the gradient. Recommended values are around ``0.9``.
    * ``second_moment_decay``: float, default 0.999
        The second moment decay, essentially scaling a term that is quadratic in the gradient. Recommended values are around ``0.999``.
