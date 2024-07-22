"""In order to run the tests, you need to have a valid API_TOKEN.

**Run train test**
```bash
export API_TOKEN=<your_api_token>
pytest tests/test_example_2.py -s -vv -k test_train_two

Once the test is executed you should get the id of the trained model.

**Run inference test**
```bash
export API_TOKEN=<your_api_token>
export TRAINED_MODEL=<trained_model_id>
pytest tests/test_example_2.py -s -vv -k test_inference_two
```
"""

import os
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from qcog_python_client.qcog import QcogClient
from qcog_python_client.schema import (
    AnalyticOptimizationParameters,
    LOBPCGFastStateParameters,
)

API_TOKEN = os.getenv("API_TOKEN")
QCOG_VERSION = "0.0.89"

if API_TOKEN is None:
    raise ValueError("API_TOKEN is not set")

file_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_path, "data")
data_file = os.path.join(data_dir, "data.npy")
labels_file = os.path.join(data_dir, "labels.npy")
test_fraction = 0.2

# checks whether data has been downloaded already
with tempfile.TemporaryDirectory() as temp_dir:
    zip_file = os.path.join(temp_dir, "coil20.zip")

    results = requests.get(
        "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
    )
    with open(zip_file, "wb") as code:
        code.write(results.content)

    # unzip image files
    images_zip = zipfile.ZipFile(zip_file)
    mylist = images_zip.namelist()
    filelist = list(filter(re.compile(r".*\.png$").match, mylist))
    filelist = [os.path.join(temp_dir, f) for f in filelist]

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    labels = (
        pd.Series(filelist)
        .str.extract("obj([0-9]+)", expand=False)
        .values.astype("int")
        - 1
    )

    images = []
    for file in filelist:
        im = Image.open(file).convert("L")
        images.append(np.array(im).flatten())

    data = np.array(images) / 255  # We scale our grayscale values from 0->255 to 0->1

# one-hot encoding of labels

targets = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=20).numpy()

# split data into train/test

train_data, test_data, train_target, test_target = train_test_split(
    data, targets, test_size=test_fraction, stratify=labels
)

# Convert to DataFrame
pixel_operators = [f"pixel_{x}" for x in range(128 * 128)]
label_operators = [f"label_{i+1}" for i in range(20)]

df_train = pd.DataFrame(
    np.concatenate([train_data, train_target], axis=1),
    columns=pixel_operators + label_operators,
)

df_test = pd.DataFrame(test_data, columns=pixel_operators)
df_target = pd.DataFrame(test_target, columns=label_operators)


API_TOKEN = os.getenv("API_TOKEN")
TRAINED_MODEL = os.getenv("TRAINED_MODEL")


def test_train_two():
    """Run training test."""
    qcml = QcogClient.create(
        token=API_TOKEN, hostname="localhost", port=8000, version=QCOG_VERSION
    )

    # IF a trained model guid is provided,
    # load the model and check the status

    if TRAINED_MODEL is not None:
        qcml = qcml.preloaded_model(TRAINED_MODEL)
        print(qcml.trained_model["guid"])

        status = qcml.status()

        if status != "completed":
            raise ValueError(f"Model is not trained yet. Status: {status}")

        else:
            print(f"Model {TRAINED_MODEL} is already trained. Skipping training.")
            return

    qcml.data(df_train)

    # Model selection
    qcml = qcml.pauli(operators=df_train.columns.to_list(), qbits=5, pauli_weight=2)

    qcml.train(
        batch_size=len(df_train),
        num_passes=10,
        weight_optimization=AnalyticOptimizationParameters(),
        get_states_extra=LOBPCGFastStateParameters(iterations=20),
    )

    qcml.wait_for_training(poll_time=10)
    print(qcml.trained_model["guid"])


def test_inference_two():
    """Run inference test."""
    if API_TOKEN is None:
        raise ValueError("API_TOKEN is not set")

    if TRAINED_MODEL is None:
        raise ValueError("TRAINED_MODEL is not set")

    qcml = QcogClient.create(
        token=API_TOKEN, hostname="localhost", port=8000, version=QCOG_VERSION
    )

    print("Loading model...")
    qcml = qcml.preloaded_model(TRAINED_MODEL)

    print("Running predictions...")
    result_df = qcml.inference(
        data=df_test,
        parameters={
            "state_parameters": LOBPCGFastStateParameters(iterations=20, tol=1e-6)
        },
    )

    num_correct = (result_df.idxmax(axis=1) == df_target.idxmax(axis=1)).sum()

    print(f"Correct: {num_correct * 100 / len(df_test):.2f}% out of {len(df_test)}")
