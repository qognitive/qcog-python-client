Wisconsin Breast Cancer
=======================

The Wisconsin Breast Cancer dataset is a standard training dataset that is used to classify if a breast cancer tumor is benign or malignant.  The dataset contains 569 samples with 30 features each.  The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  They describe characteristics of the cell nuclei present in the image. You can read some more about the dataset `here <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>`_.

Uploading the Dataset
----------------------

Let's pull the dataset from scikit-learn and upload it to the Qcog platform.  We'll split the dataset into training and testing sets, and scale the data using a standard scaler.

First let's make sure we install some extra dependencies

.. code-block:: bash

    (venv)$ pip install scikit-learn torch


.. code-block:: python

    import numpy as np

    import pandas as pd

    from sklearn import datasets as sk_datasets
    from sklearn.preprocessing import StandardScaler

    import torch

    test_fraction = 0.2

    data = sk_datasets.load_breast_cancer()
    n_data = data.data.shape[0]
    train_size = int(n_data * (1 - test_fraction))
    test_size = n_data - train_size

    # Randomly sample data
    train_idx = np.random.choice(n_data, train_size, replace=False)
    test_idx = np.random.choice(
        np.setdiff1d(np.arange(n_data), train_idx, assume_unique=True),
        test_size,
        replace=False,
    )

    targets = torch.nn.functional.one_hot(
        torch.tensor(data.target), num_classes=2
    ).numpy()

    train_data = data.data[train_idx]
    train_target = targets[train_idx]
    test_data = data.data[test_idx]
    test_target = targets[test_idx]

    # Scale data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Convert to DataFrame
    df_train = pd.DataFrame(
        np.concatenate([train_data, train_target], axis=1),
        columns=data.feature_names.tolist() + data.target_names.tolist(),
    )

    df_test = pd.DataFrame(test_data, columns=data.feature_names)
    df_target = pd.DataFrame(test_target, columns=data.target_names.tolist())

Let's instantiate a client object and set the dataset to the train dataframe we built.  We're only going to upload the ``df_train`` dataframe as the test data is only used for evaluation.

.. code-block:: python

    from qcog_python_client import QcogClient
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

Let's pick an Ensemble model to run.

.. code-block:: python

    qcml = qcml.ensemble(
        operators=df_train.columns.tolist(),
        dim=64,
        num_axes=64
    )

Here we remember our operators have to match the dataset that we are going to run.

Training the Model
------------------

Now set some training specific parameters and execute the training.

.. code-block:: python

    qcml = qcml.train(
        batch_size=64,
        num_passes=10,
        weight_optimization={
            "learning_rate": 1e-3,
            "iterations": 5,
            "optimization_method": "GRAD"
        },
        get_states_extra={
            "state_method": "LOBPCG_FAST",
            "iterations": 10,
            "learning_rate_axes": 1e-3
        }
    )
    qcml.wait_for_training()
    print(qcml.trained_model["guid"])

.. note::

    The training process may take a while to complete, here we call ``wait_for_training`` which will block until training is complete.  It should take about 4 minutes to train the model from a cold start.

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
            "iterations": 20,
            "tolerance": 1e-6
        }
    )
    num_correct = (
        result_df.idxmax(axis=1) == df_target.idxmax(axis=1)
    ).sum()
    print(f"Correct: {num_correct * 100 / len(df_test):.2f}% out of {len(df_test)}")

Results
-------

Some example results for various dimensionalities and axes numbers are shown below.

.. list-table:: Sample Results
    :header-rows: 1

    * - Dimensionality
      - Num of Axes
      - Accuracy
    * - 64
      - 64
      - 87.72 %
    * - 64
      - 256
      - 88.60 %
    * - 256
      - 512
      - 88.60 %
