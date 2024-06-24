Inference
=========

With our model trained we can now use it for inference.

First we need to build our input data. We'll put together 10 samples of X and Y but not provide Z.

.. code:: python

    n_forecastdata = 10
    xis_forecast = np.random.randn(n_forecastdata, 2)
    xs_forecast = np.ones(n_forecastdata)
    xs_forecast[xis_forecast[:, 0] < 0] = -1
    ys_forecast = np.ones(n_forecastdata)
    ys_forecast[xis_forecast[:, 1] < 0] = -1
    forecast_data = pandas.DataFrame(
        data=(np.vstack([xs_forecast, ys_forecast])).T,
        index=range(n_forecastdata),
        columns=["X", "Y"]
    )

Now we will construct some inference parameters that will determine how the model is run on the inference data. Here we have to specify ``state parameters``, which you can read more about these state parameters in the :doc:`/parameters/state` section.

.. code:: python

    parameters = {
        "state_method": "LOBPCG_FAST",
        "iterations": 5
    }

Finally we execute an inference call against our trained model, providing the forecast data and the parameters.

.. code:: python

    predicted_df = model.inference(forecast_data, parameters)

You can print the dataframe returned and see how close it gets to what we expect.

Loading a pre-trained model
----------------------------

If you have trained a model in a different script, session, or process you can load it back into the qcml object and then use it for inference, that way you do not have to deal with the dataset or the parameters again.

First make sure you get and save the ID of your model after you train it.

.. code:: python

    model_id = qcml.trained_model["guid"]

Then you can load the model back in using the ID. You will need to instantiate a ``qcml`` object but you do not need to provide any other parameters other than the trained model ID to proceed with inference.

.. code:: python

    qcml = qcml.preloaded_model(model_id)


Using the async client
----------------------

The async client has the same interface except we have to await our inference call.

.. code:: python

    result_df = await model.inference(forecast_data, parameters)


Next Steps
----------

That's it!  You have all the pieces to get started with the QCML library. You can find some more complex examples in our examples section on the left. There are also a lot of options and parameters for you to explore to find the best fit to your problem space.

Try some more complicated examples and see how the model performs, or dive right into your own data.

* :doc:`/examples/wisconsin`
* :doc:`/examples/coil20`
* :doc:`/examples/timeseries`
