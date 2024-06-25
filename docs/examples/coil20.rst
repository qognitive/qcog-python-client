COIL20
======

COIL20 (Columbia Object Image LIbrary - 20) is a dataset involving 20 images that are black and white and each have 72 pictures of each object taken at different angles. The dataset is available at `Columbia university's website <http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php>`_.

We will build a QCML model that will train on COIL20 and then classify the images.

Uploading the Dataset
----------------------

First we need to get our hands on the data and upload it to the qognitive servers. We'll download COIL20 as a zip file, unzip it and construct a dataframe.  We'll define each pixel in our image to be an operator along with an operator for every category (so our categorization will be between 20 operators where we will take 1 to be the image is of that category and 0 to be that it is not of that category).

We'll be using some extra packages here such as ``PIL``, ``requests``, ``scikit-learn`` and ``pytorch``.  You can install these with the following command:

.. code-block:: bash

    (venv)$ pip install pillow requests torch scikit-learn

Let's download the data and format it into a dataframe suitable for training and inference.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import torch
    import os
    import requests
    import re
    import zipfile
    from PIL import Image
    import tempfile

    from sklearn.model_selection import train_test_split

    file_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(file_path, 'data')
    data_file = os.path.join(data_dir, 'data.npy')
    labels_file = os.path.join(data_dir, 'labels.npy')

    # checks whether data has been downloaded already

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_file = os.path.join(temp_dir, 'coil20.zip')

        results = requests.get(
            'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip'
        )
        with open(zip_file, "wb") as code:
            code.write(results.content)

        # unzip image files
        images_zip = zipfile.ZipFile(zip_file)
        mylist = images_zip.namelist()
        filelist = list(filter(re.compile(r".*\.png$").match, mylist))
        filelist = [os.path.join(temp_dir, f) for f in filelist]

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        labels = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False).values.astype('int') - 1

        images = []
        for file in filelist:
            im = Image.open(file).convert('L')
            images.append(np.array(im).flatten())

        data = np.array(images) / 255  # We scale our grayscale values from 0->255 to 0->1

    # one-hot encoding of labels

    targets = torch.nn.functional.one_hot(
        torch.tensor(labels), num_classes=20
    ).numpy()

    # split data into train/test

    train_data, test_data, train_target, test_target = train_test_split(data,
                                                                        targets,
                                                                        test_size=test_fraction,
                                                                        random_state=random_state,
                                                                        stratify=labels)

    # Convert to DataFrame
    pixel_operators = [f"pixel_{x}" for x in range(128*128)]
    label_operators = [f"label_{i+1}" for i in range(20)]

    df_train = pd.DataFrame(
        np.concatenate([train_data, train_target], axis=1),
        columns=pixel_operators + label_operators,
    )
    df_test = pd.DataFrame(test_data, columns=pixel_operators)
    df_target = pd.DataFrame(test_target, columns=label_operators)

Let's instantiate a client object and set the dataset to COIL20.  We're only going to upload the ``df_train`` dataframe as the test data is only used for evaluation.

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

Let's pick a Pauli model to run.

.. code-block:: python

    qcml = qcml.pauli(
        operators=df_train.columns.tolist(),
        qbits=5,
        pauli_weight=2
    )

Here we remember our operators have to match the dataset that we are going to run.

Training the Model
------------------

Now set some training specific parameters and execute the training.

.. code-block:: python

    qcml = qcml.train(
        batch_size=len(df_train),
        num_passes=10,
        weight_optimization={
            "optimization_method": "ANALYTIC"
        },
        get_states_extra={
            "state_method": "LOBPCG_FAST",
            "iterations": 10
        }
    )
    qcml.wait_for_training()
    print(qcml.trained_model["guid"])

Here we are using our analytic solver which is avaliable for the Pauli model. As per the documentation for the analytic optimization method we set our batch size to the number of samples in our dataset so we process all data in a single batch.
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
        get_states_extra={
            "state_method": "LOBPCG_FAST",
            "iterations": 20,
            "tolerance": 1e-6
        }
    )
    num_correct = (
        result_df.idxmax(axis=1) == df_target.idxmax(axis=1)
    ).sum()
    print(f"Correct: {num_correct * 100 / len(df.test):.2f}% out of {len(df.test)}")

Results
-------

.. note::

    TODO we should put some example results in here!
