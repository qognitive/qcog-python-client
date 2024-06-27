Time Series Regression
======================

Our other examples have been classification examples, but let us switch to a different regime. One of the most powerful features of QCML is that the same model architectures can function well across many different problem regimes. Here we'll apply it to time series forecasting.

The data set that we are using for this example contains the responses of a gas multisensor device deployed in an Italian city. Hourly responses averages are recorded along with gas concentrations references from a certified analyzer. See the `paper <https://www.sciencedirect.com/science/article/abs/pii/S0925400507007691?via%3Dihub>`_ and `dataset <https://archive.ics.uci.edu/ml/datasets/Air+Quality>`_ for more information.

In this example we will use QCML models to predict all observed features for a given horizon using a specified lookback window.

Uploading the Dataset
----------------------

First we need to get our hands on the data and upload it to the qognitive servers. We use the `UC Irvine Machine Learning Repository <https://archive.ics.uci.edu/>`_ to download and access the data. After that, we need to convert our datetime features, scale the data, and add lagged features. Each measurement is taken hourly so we will use a lookback window of 24 hours and a horizon of 24 hours. Since the original dataset has 15 features, we will have 15*(24+24) = 720 features in total during training and each of these will correspond to an observable operator.

We'll be using some extra packages here such as ``scikit-learn`` and ``ucimlrepo``.  You can install these with the following command:

.. code-block:: bash

    (venv)$ pip install ucimlrepo scikit-learn

Let's download the data and format it into a dataframe suitable for training and inference.

.. code-block:: python

    # std
    import os
    import pickle
    import json

    # external
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    from ucimlrepo import fetch_ucirepo

    # internal
    from qcog.src.classical.pauli_symbolic import ClassicalPauliSymbolic as PauliHSM
    from qcog.utils.model_default_init import default_config_map


    def add_lagged_features(
        train_test_idx: np.ndarray,
        data_scaled: np.ndarray,
        features: list[str],
        lookback_window: int,
        horizon: int,
        column_names: list[str],
    ) -> pd.DataFrame:
        """Add lagged features to data.

        This function creates and returns a new array (data_raw), where it has
        introduced F*(L+H-1) new columns, where F is the number of features, L
        is the size of the lookback window, and H is the size of the horizon. For
        each feature, it adds values from the original data_scaled array with a
        lag of 0 to L+H-1.

        Every feature uses the same lookback window and horizon.

        Parameters
        ----------
        train_test_idx : np.ndarray
            The indices in data_scaled that will be used to create the new array for either
            the train or test data.These must be in the range
            [lookback_window + horizon, data_scaled.shape[0]].
        data_scaled : np.ndarray
            The raw data array to add the lagged features to.
        features : list[str]
            The list of feature names.
        lookback_window : int
            The size of the lookback window.
        horizon : int
            The size of the horizon.
        column_names : list[str]
            The names of the columns after adding lagged features.

        Returns
        -------
        pd.DataFrame
            the new dataframe containing the union of old features and new lagged features.
        """

        data_raw = np.zeros(
            (train_test_idx.shape[0], len(features) * (lookback_window + horizon))
        )

        for ti in range(train_test_idx.shape[0]):
            t = train_test_idx[ti]
            for i, f in enumerate(features):
                col_start = i * (lookback_window + horizon)
                for j in range(lookback_window + horizon):
                    data_raw[ti, col_start + j] = data_scaled[t - j, i]

        return pd.DataFrame(data_raw, columns=column_names)


    def load_air_quality(
        n_train: int, n_test: int, lookback_window: int, horizon: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the air quality dataset from UCI ML repository.

        See https://archive.ics.uci.edu/dataset/360/air+quality and the original paper
        https://www.semanticscholar.org/paper/a90a54a39ff934772df57771a0012981f355949d.

        Testing and training data are chosen from disjoint sets of data points.

        Parameters
        ----------
        n_train : int
            The number of data points to use for training.
        n_test : int
            The number of data points to use for testing.
        lookback_window : int
            Size of the lookback window.
        horizon : int
            Size of the horizon to predict

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Training data, test data (missing data we want to predict), and target data
            (labels for test data).
        """

        data_path = os.path.join("/tmp", "air_quality.pkl")

        # Cache dataset
        if os.path.exists(data_path):
            print("Using cached data")
            with open(data_path, "rb") as f:
                air_quality = pickle.load(f)
        else:
            air_quality = fetch_ucirepo(id=360)
            with open(data_path, "wb") as f:
                pickle.dump(air_quality, f)

        # data (as pandas dataframes)
        X = air_quality.data.features
        X["datetime"] = pd.to_datetime(X["Date"] + " " + X["Time"])
        X["month"] = X["datetime"].dt.month
        X["day_of_week"] = X["datetime"].dt.dayofweek
        X["hour"] = X["datetime"].dt.hour
        X.drop(columns=["Date", "Time", "datetime"], inplace=True)

        # Features
        features = X.columns.tolist()
        forecast_features = [f"{f}_{t}" for f in features for t in range(horizon)]
        column_names = [
            f"{f}_{t}" for f in features for t in range(lookback_window + horizon)
        ]

        # Train, validation, and test boundaries in data
        # Train = [0, 60%], Validation = (60%, 80%], Test = (80%, 100%]
        n_data = X.shape[0]
        boundaries = [0, int(n_data * 0.6), int(n_data * 0.8), n_data]

        # Input checking
        if boundaries[1] - lookback_window - horizon - n_train < 0:
            raise ValueError(
                "Not enough training data points for lookback window and horizon"
            )
        if boundaries[3] - boundaries[2] - lookback_window - horizon - n_test < 0:
            raise ValueError("Not enough test data points for lookback window and horizon")

        # Scaling data (scaled just by the training data)
        scaler = StandardScaler().fit(X[boundaries[0] : boundaries[1]])
        df_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

        # Select indices
        train_idx = np.random.choice(
            np.arange(lookback_window + horizon, boundaries[1]), n_train, replace=False
        )
        test_idx = np.random.choice(
            np.arange(lookback_window + horizon + boundaries[2], boundaries[3]),
            n_test,
            replace=False,
        )

        # Dataframes with lagged features
        df_train = add_lagged_features(
            train_test_idx=train_idx,
            data_scaled=df_scaled.values,
            features=df_scaled.columns,
            lookback_window=lookback_window,
            horizon=horizon,
            column_names=column_names,
        )

        df_test = add_lagged_features(
            train_test_idx=test_idx,
            data_scaled=df_scaled.values,
            features=df_scaled.columns,
            lookback_window=lookback_window,
            horizon=horizon,
            column_names=column_names,
        )

        df_target = pd.DataFrame(
            df_test[forecast_features].values,
            index=df_test.index,
            columns=forecast_features,
        ).reindex(sorted(forecast_features), axis=1)

        # Drop forecast features from test data
        df_test.drop(columns=forecast_features, inplace=True)

        return df_train, df_test, df_target

Let's instantiate a client object and set the dataset to our timeseries dataframe.  We're only going to upload the ``df_train`` dataframe as the test data is only used for evaluation.

.. code-block:: python

    from qcog_python_client import QcogClient

    # Set the random seed for consistent selection of training and test data
    np.random.seed(42)
    df_train, df_test, df_test_labels = load_air_quality(1000, 200, 72, 24)

    # Send the training data to the server
    qcml = QcogClient.create(
        token=API_TOKEN,
        hostname="api.qognitive.io",
        port=443,
        verify=False,
        secure=True,
    )
    qcml.data(df_train)


Parameterizing our Model
------------------------

Let's pick a Pauli model to run.

.. code-block:: python

    qcml = qcml.pauli(
        operators=df_train.columns.tolist(),
        qbits=4,
        pauli_weight=2
    )

Here we remember our operators have to match the dataset that we are going to run.

Training the Model
------------------

Now set some training specific parameters and execute the training.

.. code-block:: python

    qcml = qcml.train(
        batch_size=32,
        num_passes=10,
        weight_optimization={
            "optimization_method": "GRAD",
            "learning_rate": 1e-5,
            "iterations": 3,

        },
        get_states_extra={
            "state_method": "LOBPCG_FAST",
            "iterations": 15
        }
    )
    qcml.wait_for_training()
    print(qcml.trained_model["guid"])

Here we use the gradient descent optimizer with a learning rate of 1e-5 and 3 iterations. We also use the LOBPCG_FAST state method with 15 iterations. We are not using the analytic solver because we are not passing the entire dataset at the same time.

.. note::

    The training process may take a while to complete, here we call ``wait_for_training`` which will block until training is complete.

.. note::

    We print out the trained model ``guid`` so we can use it in a different interpreter session if needed.

Executing Inference
-------------------

If you are running in the same session you can skip the next step, but if you are running in a different session you can load the model using the ``guid`` we printed out.

.. code-block:: python

    qcml = qcml.preloaded_model(MODEL_GUID)

With our trained model loaded into the client, we can now run inference on the dataset.

.. code-block:: python

    result_df = qcml.inference(
        data=df_test,
        parameters={
            "state_method": "LOBPCG_FAST",
            "iterations": 25,
            "tolerance": 1e-4
        }
    )
    mse = mean_squared_error(df_test_labels, results_df)
    mape = mean_absolute_percentage_error(df_test_labels, results_df)
    print(f"MSE:  {mse:.4f}")
    print(f"MAPE: {mape:.4f}")

Results
-------

Some example results for various qubit counts and Pauli weights are shown below. The mean squared error (MSE) and mean absolute percentage error (MAPE) are calculated for each case.

.. list-table:: Sample Results
    :header-rows: 1

    * - Qubits
      - Pauli Weight
      - MSE
      - MAPE
    * - 2
      - 1
      - 1.098
      - 7.770
    * - 4
      - 2
      - 0.983
      - 4.912
    * - 6
      - 2
      - 0.903
      - 6.17
