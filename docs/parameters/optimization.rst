Optimization Parameters
=======================

The optimization parameters are used to control how our model finds the best internal representation of the problem space. These parameters are used only for training the model and do not affect the model's predictions (since the model is fixed after training).

There are several different methods of optimization that we currently support, each one with its own set of parameters.

ANALYTIC
--------

The analytic optimizer is a closed form solution to the internal weight matrix in our model. Not all QCML models have an analytic solution, but when they do, the analytic optimizer is the fastest and most accurate way to train the model. The caveat is that the entire dataset needs to be in a single batch and this places memory constraints on the dataset that is used. Small datasets can benefit from the analytic optimizer, but large datasets which cannot fit in memory should use another optimizer.

GRAD
----

The grad optimizer uses gradient descent with a fixed learning rate to tune the weights in the internal state of our model. GRAD is appropriate for any batch size or QCML model.

ADAM
----

ADAM is a stochastic optimizer that tunes the learning rate by various momentum parameters, details of which can be found `in the original paper here <https://arxiv.org/abs/1412.6980>`_. ADAM is appropriate for any batch size or QCML model.