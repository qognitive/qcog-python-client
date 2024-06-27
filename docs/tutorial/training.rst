Running a training job
======================

So far, the settings that parameterize our model to pertain to both the inference as well as the training phases. Now we introduce the `training_parameters` which __only__ affect the training of our model. We will upload these training specific parameters to qogntive's servers so a training run can be launched.

We'll set the training parameters and trigger our training run.

.. code-block:: python

    qcml = qcml.train(
        batch_size=4,
        num_passes=10,
        weight_optimization={
            "learning_rate": 1e-3,
            "iterations": 10,
            "optimization_method": "GRAD"
        },
        get_states_extra={
            "state_method": "LOBPCG_FAST",
            "iterations": 10
        }
    )

While the model is training, let's look at what these parameters here are - since they are important for your model.

The first two parameters are the batch size and the number of passes. The batch size is what is also called the minibatch size, it is the size of the data that is used for training the model. In the :doc:`/tutorial/datasets` section we have 4 datapoints in the training dataset that we are using. By setting our batch size to 4 we are using the entire dataset as a single batch, if we set it to 2 then each pass would consist of 2 batches of 2 datapoints. The second number is the number of passes over the entire data set, so in this case we will run our optimization over the entire data set 10 times.

The next set of parameters is ``weight_optimization``. This determines how the internal weights that generate the quantum representation are optimized with each pass over the data. In this example we are using gradient descent. There are more details on the optimization methods and their parameterization in the :doc:`/parameters/optimization` section. The parameters from that section are placed as a sub dictionary on the ``weight_optimization`` key in the training parameters.

The last set of parameters are ``get_states_extra``. This determines how the internal quantum state is calculated, and this is used for both the inference and for training. Since multiple passes are made over the data in training the tolerances can be lower than for inference, since only a single inference pass is made in that step. There are more details in the :doc:`/parameters/state` section. The parameters from that section are placed as a sub dictionary on the ``get_states_extra`` key in the training parameters.

Introspecting the training progress
-----------------------------------

Running ``train`` will trigger a training run on our servers. The model and the dataset have references saved in the ``qcml`` object so that is why they are not passed in here.

The `train` method is not a blocking call, so the training job can be in progress or complete - you will need to check. The ``qcml`` object can be used to check the status of your training run.


.. code-block:: python

    qcml.status()  # returns 'completed' or 'processing'


You can build your own poller, or you can use our builtin

.. code-block:: python

    qcml.wait_for_training()

This is a blocking call that will wait for the training job to complete.


Using the Async Client
-----------------------

Using the async client is essentially calling await on the same methods

.. code-block:: python

    qcml = await qcml.train(**training_parameters)
    await qcml.wait_for_training()

Here the power of async calls really shines, our application will come back when training is complete but will not be blocked on the training job.

Saving your model ID
--------------------

When the model has completed training we can moveon to inference. A training run is complete when the ``wait_for_training()`` call on either client has returned succesfully. Once it has then the ``qcml`` object becomes populated with a trained model ID. Here you will want to save the ID, such as in a database, so you can access the model easily later.

You can access the model ID with either client in the following way:

.. code-block:: python

    model_id = qcml.trained_model["guid"]

You can load this model into your client by instantiating a ``qcml`` client and passing the model ID to it as such:

.. code-block:: python

    qcml = qcml.preloaded_model(model_id)

This will override any trained model in the ``qcml`` local client. If you are using many models at once you will need many ``qcml`` instances.
